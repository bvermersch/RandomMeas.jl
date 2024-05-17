function acquire_shadows_batch_fromdata(data::Array{Int8}, sitesA::Vector{Index{Int64}}, u::Vector{Vector{ITensor}}, n::Int64)
    shadow = Vector{ITensor}()
    NA = size(data, 3)
    nu = size(data, 1)
    for m in 1:n
        push!(shadow, ITensor(vcat(sitesA, prime(sitesA))))
    end
    nu_n = nu ÷ n
    for t in 1:n
        for r in 1+(t-1)*nu_n:t*nu_n
            P = get_Born_data_binary(data[r, :,:], sitesA)
            shadow_temp = get_shadow(P, sitesA, u[r])
            shadow[t] += shadow_temp / nu_n
        end
    end
    return shadow
end

function get_moments_shadows_batch(shadow::Vector{ITensor}, sitesA::Vector{Index{Int64}}, n::Int64, nu::Int64)
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
            p[m-1] += real(trace(X, sitesA))
        end
        p[m-1] /= alpha

    end
    return p
end

function get_shadow(P::ITensor, sites::Vector{Index{Int64}}, u::Vector{ITensor};G::Union{Vector{Float64},Nothing}=nothing)
    NA = length(u)
    i1 = Index(2, "i1")
    i2 = Index(2, "i2")
    rho = 2^NA * deepcopy(P)
    for i in 1:NA
        s = sites[i]
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
        h = Hamming_tensor * delta(i1, s) * delta(i2, s'')
        rho *= h
        rho *= delta(s, s', s'')

        ut = u[i] * delta(s'', s)
        ut *= delta(s, s')
        rho = mapprime(ut * rho, 2, 0)
        ut = dag(u[i]) * delta(s'', s)
        rho = mapprime(ut * rho, 2, 1)
    end
    return rho
end

function get_shadow_factorized!(rho::Vector{ITensor}, M::Array{Int8}, s::Vector{Index{Int64}}, u::Vector{ITensor}, G_vec::Vector{Float64}, imax::Int64)
    N = length(u)
    for i in 1:N
        if i <= imax #std shadows
            alpha = 3 / (2 * G_vec[i] - 1)
            beta = (G_vec[i] - 2) / (2 * G_vec[i] - 1)
        else #diagonal shadows
            alpha = 1 / (2 * G_vec[i] - 1)
            beta = (G_vec[i] - 1) / (2 * G_vec[i] - 1)
        end
        #u*_{s',s}|s'><s'|=u^dag_{s,s'}|s'><s'|
        psi = dag(u[i]) * onehot(s[i]' => M[i])
        rho[i] = alpha * psi' * dag(psi) + beta * delta(s[i], s[i]')
    end
end


function get_expect_shadow(O::MPO, shadow::ITensor, sites::Vector{Index{Int64}})
    N = size(sites, 1)
    X = 1 * prime(shadow)
    for i in 1:N
        s = sites[i]
        X *= O[i] * delta(s, prime(s, 2))
    end
    return real(scalar(X))
end

function get_expect_shadow(O::MPO, shadow::Vector{ITensor}, sites::Vector{Index{Int64}})
    N = size(sites, 1)
    X = 1
    for i in 1:N
        s = sites[i]
        X *= shadow[i]'
        X *= O[i] * delta(s, prime(s, 2))
    end
    return real(scalar(X))
end

function get_expect_shadow(O::MPO, shadow::Vector{MPO}, sites::Vector{Index{Int64}})
    NM = size(shadow, 1)
    N = size(sites, 1)
    O_e = 0
    for s in 1:NM
        O_e += inner(O, shadow[s]) / NM
    end
    return O_e
end

function get_expect_shadow(psi::MPS, shadow::Vector{MPO}, sites::Vector{Index{Int64}})
    NM = size(shadow, 1)
    N = size(sites, 1)
    O_e = 0
    for s in 1:NM
        O_e += inner(psi', shadow[s], psi) / NM
    end
    return O_e
end

function get_expect_shadow(O::MPO, shadow::MPO, sites::Vector{Index{Int64}})
    O_e = inner(O, shadow)
    return O_e
end


function square(shadow::ITensor)
    Y = multiply(shadow, shadow)
    return Y
end

function multiply(shadow::ITensor, shadow2::ITensor)
    return mapprime(shadow * prime(shadow2), 2, 1)
end


function power(shadow::ITensor, n::Int64)
    Y = deepcopy(shadow)
    for m in 1:n-1
        Y = multiply(Y, shadow)
    end
    return Y
end

function trace(shadow::ITensor, sites::Vector{Index{Int64}})
    NA = size(sites, 1)
    Y = copy(shadow)
    for i in 1:NA
        Y *= delta(sites[i], prime(sites[i]))
    end
    return scalar(Y)
end

function partial_trace(shadow::ITensor, sites::Vector{Index{Int64}})
    NA = size(sites, 1)
    Y = copy(shadow)
    for i in 1:NA
        Y *= delta(sites[i], prime(sites[i]))
    end
    return Y
end
