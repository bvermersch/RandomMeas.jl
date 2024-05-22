"""
    acquire_shadows_batch_fromdata(data::Array{Int8}, ξ::Vector{Index{Int64}}, u::Vector{Vector{ITensor}}, n::Int64)

Constructs n batch shadows from measured data
"""
function acquire_shadows_batch_fromdata(data::Array{Int8}, ξ::Vector{Index{Int64}}, u::Vector{Vector{ITensor}}, n::Int64)
    shadow = Vector{ITensor}()
    NA = size(data, 3)
    nu = size(data, 1)
    for m in 1:n
        push!(shadow, ITensor(vcat(ξ, ξ')))
    end
    nu_n = nu ÷ n
    for t in 1:n
        for r in 1+(t-1)*nu_n:t*nu_n
            P = get_Born_data_binary(data[r, :,:], ξ)
            shadow_temp = get_shadow(P, ξ, u[r])
            shadow[t] += shadow_temp / nu_n
        end
    end
    return shadow
end



"""
    get_moments_shadows_batch(shadow::Vector{ITensor}, ξ::Vector{Index{Int64}}, n::Int64, nu::Int64)

Obtain trace moments from batch shadows using U-statistics
"""
function get_moments_shadows_batch(shadow::Vector{ITensor}, ξ::Vector{Index{Int64}}, n::Int64, nu::Int64)
    p = Vector{Float64}()

    for m in 2:n
        r_a = collect(permutations(1:n, m)) #m_uplet of n batches
        alpha = length(r_a)
        est = 0
        push!(p, 0)
        for r in r_a
            X = multiply(shadow[r[1]], shadow[r[2]])
            for m1 in 3:m
                X = multiply(X, shadow[r[m1]])
            end
            p[m-1] += real(trace(X, ξ))
        end
        p[m-1] /= alpha

    end
    return p
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
    get_shadow_factorized!(rho::Vector{ITensor}, M::Array{Int8}, s::Vector{Index{Int64}}, u::Vector{ITensor};G_vec::Union{Nothing,Vector{Float64}}=nothing)

    build shadow as a tensor-product (memory-efficient)
"""
function get_shadow_factorized!(rho::Vector{ITensor}, M::Array{Int8}, s::Vector{Index{Int64}}, u::Vector{ITensor};G_vec::Union{Nothing,Vector{Float64}}=nothing)
#function get_shadow_factorized!(rho::Vector{ITensor}, M::Array{Int8}, s::Vector{Index{Int64}}, u::Vector{ITensor};G_vec::Vector{Float64}=undef)
    N = length(u)
    for i in 1:N
        if G_vec ==nothing
            α = 3
            β = -1
        else
            α = 3 / (2 * G_vec[i] - 1)
            β = (G_vec[i] - 2) / (2 * G_vec[i] - 1)
        end
        #u*_{s',s}|s'><s'|=u^dag_{s,s'}|s'><s'|
        ψ = dag(u[i]) * onehot(s[i]' => M[i])
        rho[i] = α * ψ' * dag(ψ) + β * δ(s[i], s[i]')
    end
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
