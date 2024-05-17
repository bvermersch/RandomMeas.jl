using Random
using ProgressBars

function get_rotations(s::Vector{Index{Int64}}, cat::Int=1)
    u = Vector{ITensor}()
    N = length(s)
    for i in 1:N
        push!(u, get_rotation(s[i], cat))
    end
    return u
end

function get_rotation(s::Index{Int64}, cat::Int=1)
    #1: Haar
    #2: Pauli
    #3: Clifford (TBD)
    #4: Pauli X twirling
    r_matrix = zeros(ComplexF64, (2, 2))
    if cat == 1
        return op("RandomUnitary", s)
    elseif cat == 2
        b = rand(1:3)
        #println("basis B", b)
        if b == 1
            r_matrix[1, 1] = 1
            r_matrix[2, 2] = 1
        elseif b == 2
            r_matrix[1, 1] = 1 / sqrt(2)
            r_matrix[2, 1] = 1 / sqrt(2)
            r_matrix[1, 2] = 1 / sqrt(2)
            r_matrix[2, 2] = -1 / sqrt(2)
        else
            r_matrix[1, 1] = 1 / sqrt(2)
            r_matrix[2, 2] = 1 / sqrt(2)
            r_matrix[1, 2] = -1im / sqrt(2)
            r_matrix[2, 1] = -1im / sqrt(2)
        end
        r_tensor = itensor(r_matrix, s', s)
        return r_tensor
    elseif cat == 4
        b = rand(1:2)
        #println("basis B", b)
        if b == 1
            r_matrix[1, 1] = 1
            r_matrix[2, 2] = 1
        else
            r_matrix[1, 2] = 1
            r_matrix[2, 1] = 1
        end
        r_tensor = itensor(r_matrix, s', s)
        return r_tensor
    elseif cat == 5
        H = rand()*op("Sz",s)+op("Sx",s)+op("Sy",s)
        u = exp(-1im*H)
        return u
    elseif cat == 6
        b = rand(1:2)
        r_matrix[1, 1] = 1
        r_matrix[2, 2] = 1
        r_tensor = itensor(r_matrix, s', s)
        return r_tensor
    end
end


"""
    rotate_b

Rotate an MPO/MPS with single qubit unitaries u 
"""
function rotate_b(psi::MPS, u::Vector{ITensor})
    sites = siteinds(psi)
    psiu = deepcopy(psi)
    NA = length(sites)
    for i in 1:NA
        psiu[i] = noprime(u[i] * psi[i])
    end
    set_ortho_lims!(psiu, 1:1)
    return psiu
end

function rotate_b(rhoA::MPO, u::Vector{ITensor})
    sites = siteinds(rhoA)
    rhou = deepcopy(rhoA)
    NA = length(u)
    for i in 1:NA
        s = noprime(sites[i][1])
        ut = u[i]'
        rhou[i] = mapprime(ut * rhoA[i], 2, 1)
        udag = dag(u[i]) * delta(s', s'')
        rhou[i] = mapprime(udag * rhou[i], 2, 0)
    end
    orthogonalize!(rhou, 1)
    return rhou
end

"""
    get_RandomMeas!

Sample randomized measurements from a MPS/MPO representation ρ 
"""
function get_RandomMeas!(data_s::Array{Int8}, ρ::Union{MPO,MPS}, u::Vector{ITensor}, NM::Int64)
    ρu = rotate_b(ρ, u)
    NA = length(u)
    get_Samples_Flat!(data_s, ρu, NA, NM)
end

"""
    get_Samples_Flat!

Sample randomized measurements from a MPS/MPO representation ρ 
"""
function get_Samples_Flat!(data_s::Array{Int8}, state::Union{MPO,MPS}, NA::Int64, NM::Int64)
    #This is borrowed from PastaQ
    Prob = get_Born(state)
    prob = real(array(Prob))
    prob = reshape(prob, 2^NA)
    for m in 1:NM
        data = StatsBase.sample(0:(1<<NA-1), StatsBase.Weights(prob), 1)
        data_s[m, :] = 1 .+ digits(data[1], base=2, pad=NA)
    end
end

"""
    get_RandomMeas_MPO!

Sample randomized measurements from a MPS/MPO representation ρ. The sampling is based from the MPO directly, i.e is memory-efficient 
"""
function get_RandomMeas_MPO!(data::Array{Int8}, ρ::MPO, u::Vector{ITensor}, NM::Int64)
    ξ = [x[1] for x in siteinds(ρ;plev=0)]
    ρu = rotate_b(ρ, u)
    NA = length(u)
    ρu[1] /= trace(ρu, ξ)
    if NA > 1
        for m in 1:NM
            data[m, :] = ITensors.sample(ρu)
        end
    else
        s = ξ[1]
        prob = [real(ρu[1][s=>1, s'=>1][]), real(ρu[1][s=>2, s'=>2][])]
        data[:, 1] = StatsBase.sample(1:2, StatsBase.Weights(prob), NM)
    end
    return data
end


function get_RandomMeas_MPS!(data::Array{Int8}, psi::MPS, u::Vector{ITensor}, NM::Int64)
    psiu = rotate_b(psi, u)
    for m in 1:NM
        data[m, :] = ITensors.sample(psiu)#[1:NA]
    end
end


function get_Born_MPS(rho::MPO, sites::Vector{Index{Int64}})
    NA = size(sites, 1)
    P = MPS(sites)
    for i in 1:NA
        Ct = delta(sites[i], prime(sites[i]), prime(sites[i], 2))
        P[i] = rho[i] * Ct
        P[i] *= delta(sites[i], prime(sites[i], 2))
    end
    return P
end

function get_Born_MPS(psi::MPS)
    sites = siteinds(psi)
    N = size(sites, 1)
    P = MPS(sites)
    for i in 1:N
        Ct = delta(sites[i], prime(sites[i]), prime(sites[i], 2))
        P[i] = psi[i] * prime(conj(psi[i])) * Ct
        P[i] *= delta(sites[i], prime(sites[i], 2))
    end
    return P
end


function get_Born(rhoA::MPO)
    sites = siteinds(rhoA)
    NA = size(sites, 1)
    s = noprime(sites[1][1])
    a = Index(2, "a")
    P = rhoA[1] * delta(s, prime(s), a)
    P *= delta(a, s)
    for i in 2:NA
        s = noprime(sites[i][1])
        C = rhoA[i] * delta(s, prime(s), a)
        C *= delta(a, s)
        P *= C
    end
    return noprime(P)
end


function get_Born(rhoA::ITensor, sites::Vector{Index{Int64}})
    NA = size(sites, 1)
    s = sites[1]
    a = Index(2, "a")
    P = deepcopy(rhoA)
    for i in 1:NA
        s = sites[i]
        P = P * delta(s, prime(s), a)
        P = P * delta(a, s)
    end
    return P
end

function get_Born(psi::MPS)
    sites = siteinds(psi)
    N = size(sites, 1)

    C = delta(sites[1], prime(sites[1]), prime(sites[1], 2))
    R = C * psi[1] * prime(conj(psi[1]))
    R *= delta(sites[1], prime(sites[1], 2))
    P = R
    for i in 2:N
        Ct = delta(sites[i], prime(sites[i]), prime(sites[i], 2))
        Rt = psi[i] * prime(conj(psi[i])) * Ct
        Rt *= delta(sites[i], prime(sites[i], 2))
        P *= Rt
    end
    return noprime(P)
end

