"""
    get_batch_shadows(data::Array{Int}, ξ::Vector{Index{Int64}}, u::Vector{Vector{ITensor}}, n::Int64)

Constructs n batch shadows from measured data
"""
function get_batch_shadows(data::Array{Int}, ξ::Vector{Index{Int64}}, u::Vector{Vector{ITensor}}, n::Int64;G::Union{Vector{Float64},Nothing}=nothing)
    shadow = Vector{ITensor}()
    NA = size(data, 3)
    nu = size(data, 1)
    for m in 1:n
        push!(shadow, ITensor(vcat(ξ, ξ')))
    end
    nu_n = nu ÷ n
    for t in 1:n
        for r in 1+(t-1)*nu_n:t*nu_n
            P = get_Born(data[r, :,:], ξ)
            shadow_temp = get_shadow(P, ξ, u[r];G=G)
            shadow[t] += shadow_temp / nu_n
        end
    end
    return shadow
end



"""
    get_moments(shadow::Vector{ITensor}, ξ::Vector{Index{Int64}})

Obtain trace moments from  a vector of (batch) shadows using U-statistics
"""
function get_moments(shadow::Vector{ITensor}, ξ::Vector{Index{Int64}}, kth_moments::Vector{Int})

    p = Vector{Float64}() # Initialize the vector with zeros
    n_shadows = length(shadow)

    # Check if any entry is smaller than 1
    @assert all(kth_moments .>= 1) "Only integer valued moments Tr[rho^k] with k >=1 can be computed."

    # Check if any entry is larger than n_shadows
    @assert all(kth_moments .<= n_shadows) "The number of (batch) shadows needs to be greater or equal to the largest moment k."

    for k in kth_moments # Iterate over elements of moments
        est = 0
        for r in permutations(1:n_shadows, k)
            X = shadow[r[1]]
            for m in 2:k
                X = multiply(X, shadow[r[m]])
            end
            est += real(trace(X, ξ)) # Accumulate the estimate
        end
        push!(p, est / length(permutations(1:n_shadows, k))) # Update the corresponding element in p
    end

    return p
end

function get_moment(shadow::Vector{ITensor}, ξ::Vector{Index{Int64}}, kth_moment::Int)
    return get_moments(shadow, ξ, [kth_moment])[1]
end


"""
    get_shadow(P::ITensor, ξ::Vector{Index{Int64}}, u::Vector{ITensor};G::Union{Vector{Float64},Nothing}=nothing)

    Form shadow from Born probability represented as an ITensor
"""
function get_shadow(P::ITensor, ξ::Vector{Index{Int64}}, u::Vector{ITensor};G::Union{Vector{Float64},Nothing}=nothing)
    NA = length(u)
    i1 = Index(2, "i1")
    i2 = Index(2, "i2")
    rho = 2^NA * deepcopy(P)
    for i in 1:NA
        s = ξ[i]
        Hamming_matrix = zeros(Float64, (2, 2))
        if isnothing(G)
            Hamming_matrix[1, 1] = 1
            Hamming_matrix[2, 2] = 1
            Hamming_matrix[2, 1] = -0.5
            Hamming_matrix[1, 2] = -0.5
        else
            α  = 3 / (2 * G[i] - 1)
            β  = (G[i] - 2) / (2 * G[i] - 1)
            Hamming_matrix[1, 1] = (α+β)/2
            Hamming_matrix[2, 2] = (α+β)/2
            Hamming_matrix[2, 1] = β/2
            Hamming_matrix[1, 2] = β/2
        end
        Hamming_tensor = itensor(Hamming_matrix, i1, i2)
        h = Hamming_tensor * δ(i1, s) * δ(i2, s'')
        rho *= h
        rho *= δ(s, s', s'')

        ut = u[i] * δ(s'', s)
        ut *= δ(s, s')
        rho = mapprime(ut * rho, 2, 0)
        ut = dag(u[i]) * δ(s'', s)
        rho = mapprime(ut * rho, 2, 1)
    end
    return rho
end

"""
    get_shadow_factorized(data::Array{Int}, s::Vector{Index{Int64}}, u::Vector{};G_vec::Union{Nothing,Vector{Float64}}=nothing)

    build shadow as a tensor-product (memory-efficient)
"""
function get_shadow_factorized(data::Array{Int}, ξ::Vector{Index{Int64}}, u::Vector{ITensor};G_vec::Union{Nothing,Vector{Float64}}=nothing)
    N = length(u)
    ρ = Vector{ITensor}()
    for i in 1:N
        if G_vec ===nothing
            α = 3
            β = -1
        else
            α = 3 / (2 * G_vec[i] - 1)
            β = (G_vec[i] - 2) / (2 * G_vec[i] - 1)
        end
        #u*_{s',s}|s'><s'|=u^dag_{s,s'}|s'><s'|
        ψ = dag(u[i]) * onehot(ξ[i]' => data[i])
        push!(ρ, α * ψ' * dag(ψ) + β * δ(ξ[i], ξ[i]'))
    end
    return ρ
end


"""
    get_expect_shadow(O, shadow, ξ::Vector{Index{Int64}})

Contract shadow with operator O to estimate the expectation value ``\\mathrm{tr}(O\\rho)``
"""
function get_expect_shadow(O::MPO, shadow::ITensor, ξ::Vector{Index{Int64}})
    N = size(ξ, 1)
    X = 1 * shadow'
    for i in 1:N
        s = ξ[i]
        X *= O[i] * δ(s, s'')
    end
    return real(X[])
end

function get_expect_shadow(O::MPO, shadow::Vector{ITensor}, ξ::Vector{Index{Int64}})
    N = size(ξ, 1)
    X = 1
    for i in 1:N
        s = ξ[i]
        X *= shadow[i]'
        X *= O[i] * δ(s, s'')
    end
    return real(X[])
end

function get_expect_shadow(ψ::MPS, shadow::Vector{ITensor}, ξ::Vector{Index{Int64}})
    N = size(ξ, 1)
    X = 1
    for i in 1:N
        s = ξ[i]
        X *= shadow[i]
        X *= dag(ψ[i]') * ψ[i]
    end
    return real(X[])
end

function get_expect_shadow(O::MPO, shadow::Vector{MPO}, ξ::Vector{Index{Int64}})
    NM = size(shadow, 1)
    N = size(ξ, 1)
    O_e = 0
    for s in 1:NM
        O_e += inner(O, shadow[s]) / NM
    end
    return O_e
end

function get_expect_shadow(ψ::MPS, shadow::Vector{MPO}, ξ::Vector{Index{Int64}})
    NM = size(shadow, 1)
    N = size(ξ, 1)
    O_e = 0
    for s in 1:NM
        O_e += inner(ψ', shadow[s], ψ) / NM
    end
    return O_e
end

"""
    square(shadow::ITensor)
"""
function square(shadow::ITensor)
    Y = multiply(shadow, shadow)
    return Y
end

"""
    multiply(shadow::ITensor, shadow2::ITensor)
"""
function multiply(shadow::ITensor, shadow2::ITensor)
    return mapprime(shadow * prime(shadow2), 2, 1)
end

"""
    power(shadow::ITensor, n::Int64)
"""
function power(shadow::ITensor, n::Int64)
    Y = deepcopy(shadow)
    for m in 1:n-1
        Y = multiply(Y, shadow)
    end
    return Y
end

"""
    trace(shadow::ITensor, ξ::Vector{Index{Int64}})
"""
function trace(shadow::ITensor, ξ::Vector{Index{Int64}})
    NA = size(ξ, 1)
    Y = copy(shadow)
    for i in 1:NA
        Y *= δ(ξ[i],ξ[i]')
    end
    return Y[]
end
