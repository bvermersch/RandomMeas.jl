"""
    get_rotations(ξ::Vector{Index{Int64}}, ensemble::String="Haar")

Generate a list of N  single qubit unitaries with indices (ξ'[i],ξ[i]) (i=1,...,N) sampled randomly from the ensemble:
    "Haar" (default): Haar random single qubit unitary
    "Pauli": Random Pauli rotation sampled uniformly from {RX, RY, RZ}
    "Identity": (fixed) Identity matrix
"""
function get_rotations(ξ::Vector{Index{Int64}}, ensemble::String="Haar")
    u = Vector{ITensor}()
    N = length(ξ)
    for i in 1:N
        push!(u, get_rotation(ξ[i], ensemble))
    end
    return u
end

"""
    get_rotation(ξ::Index{Int64}, ensemble::String = "Haar")

Generate a single qubit unitary with indices (ξ',ξ)
sampled from the ensemble:
    "Haar" (default): Haar random single qubit unitary
    "Pauli": Random Pauli rotation sampled uniformly from {RX, RY, RZ}
    "Identity": (fixed) Identity matrix
"""
function get_rotation(ξ::Index{Int64}, ensemble::String = "Haar")
    r_matrix = zeros(ComplexF64, (2, 2))
    if ensemble == "Haar"
        return op("RandomUnitary", ξ)
    elseif ensemble == "Pauli"
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
        r_tensor = itensor(r_matrix, ξ', ξ)
        return r_tensor
    elseif ensemble == "Identity"
        b = rand(1:2)
        r_matrix[1, 1] = 1
        r_matrix[2, 2] = 1
        r_tensor = itensor(r_matrix, ξ', ξ)
        return r_tensor
    end
end

function get_RandomMeas(ρ::Union{MPO,MPS}, u::Vector{ITensor}, NM::Int64, mode::String)

    @assert mode in ["dense", "MPS", "MPO"] "Invalid mode"

    if mode == "dense"
        return get_RandomMeas_dense(ρ, u, NM)
    elseif mode == "MPS"
        return get_RandomMeas_MPS(ρ, u, NM)
    elseif mode == "MPO"
        return get_RandomMeas_MPO(ρ, u, NM)
    end

end


"""
    get_RandomMeas_dense(ρ::Union{MPO,MPS}, u::Vector{ITensor}, NM::Int64)

Sample randomized measurements from a MPS/MPO representation ρ
"""
function get_RandomMeas_dense(ρ::Union{MPO,MPS}, u::Vector{ITensor}, NM::Int64)
    if typeof(ρ)==MPS
        ρu = apply(u,ρ)
    else
        ρu = apply(u,ρ;apply_dag=true)
    end
    return get_samples_flat(ρu,NM)
end

"""
    get_samples_flat(state::Union{MPO,MPS},NM::Int64)

Sample randomized measurements from a MPS/MPO representation ρ
"""
function get_samples_flat(state::Union{MPO,MPS},NM::Int64)
    N = length(state)
    data_s = zeros(Int,NM,N)
    #Note: This is borrowed from PastaQ
    Prob = get_Born(state)
    prob = real(array(Prob))
    prob = reshape(prob, 2^N)
    for m in 1:NM
        data = StatsBase.sample(0:(1<<N-1), StatsBase.Weights(prob), 1)
        data_s[m, :] = 1 .+ digits(data[1], base=2, pad=N)
    end
    return data_s
end

"""
    get_RandomMeas_MPO

Sample randomized measurements from an MPO representation ρ. The sampling is based from the MPO directly, i.e., is memory-efficient
"""
function get_RandomMeas_MPO(ρ::MPO, u::Vector{ITensor}, NM::Int64)
    ξ = firstsiteinds(ρ;plev=0)
    ρu = apply(u,ρ;apply_dag=true)
    N= length(u)
    data = zeros(Int,NM,N)
    ρu[1] /= trace(ρu, ξ)
    if N > 1
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


"""
    get_RandomMeas_MPS(ψ::MPS, u::Vector{ITensor},NM::Int64)

Sample randomized measurements from an MPS representation ψ. The sampling is based from the MPS directly, i.e is memory-efficient
"""
function get_RandomMeas_MPS(ψ::MPS, u::Vector{ITensor},NM::Int64)
    N = length(ψ)
    data = zeros(Int,NM,N)
    ψu = apply(reverse(u),ψ) #using reverse allows us to maintain orthocenter(ψ)=1 ;)
    for m in 1:NM
        data[m, :] = ITensors.sample(ψu)#[1:NA]
    end
    return data
end


"""
    get_Born_MPS(ρ::MPO)

Construct Born Probability vector P(s)=<s|ρ|s> as an MPS from an MPO representation ρ
"""
function get_Born_MPS(ρ::MPO)
    ξ = firstsiteinds(ρ;plev=0)
    N = size(ξ, 1)
    P = MPS(ξ)
    for i in 1:N
        Ct = δ(ξ[i], ξ[i]', ξ[i]'')
        P[i] = ρ[i] * Ct
        P[i] *= δ(ξ[i], ξ[i]'')
    end
    return P
end

"""
    get_Born_MPS(ψ::MPS)

Construct Born Probability vector P(s)=|ψ(s)|^2 as an MPS from an MPS representation ψ
"""
function get_Born_MPS(ψ::MPS)
    ξ = siteinds(ψ)
    N = size(ξ, 1)
    P = MPS(ξ)
    for i in 1:N
        Ct = δ(ξ[i], ξ[i]', ξ[i]'')
        P[i] = ψ[i] * conj(ψ[i]') * Ct
        P[i] *= δ(ξ[i], ξ[i]'')
    end
    return P
end


 """
     get_Born(ρ::MPO)

 Construct Born Probability vector P(s) from an MPO representation ρ
"""
function get_Born(ρ::MPO)
    ξ = firstsiteinds(ρ;plev=0)
    N = size(ξ, 1)
    P = ρ[1] * δ(ξ[1],ξ[1]',ξ[1]'')
    P *= δ(ξ[1]'', ξ[1])
    for i in 2:N
        C = ρ[i] * delta(ξ[i], ξ[i]', ξ[i]'')
        C *= delta(ξ[i]'', ξ[i])
        P *= C
    end
    return P
end


"""
    get_Born(ψ::MPS)

Construct Born Probability vector P(s)=|ψ(s)|^2 from an MPS representation ψ
"""
function get_Born(ψ::MPS)
    ξ = siteinds(ψ )
    N = size(ξ, 1)
    C = δ(ξ[1], ξ[1]',ξ[1]'')
    R = C * ψ[1] * conj(ψ[1]')
    R *= δ(ξ[1], ξ[1]'')
    P = R
    for i in 2:N
        Ct = δ(ξ[i], ξ[i]', ξ[i]'')
        Rt = ψ[i] * conj(ψ[i]') * Ct
        Rt *= δ(ξ[i], ξ[i]'')
        P *= Rt
    end
    return P
end

"""
    get_XEB(ψ::MPS,ρ::MPO,NM::Int64)

Return the linear cross-entropy for NM samples of the mixed state, with respect to a
theory state ψ
"""
function get_XEB(ψ::MPS,ρ::MPO,NM::Int64)
    ξ = siteinds(ψ )
    data = get_samples_flat(ρ,NM)
    P0 = get_Born_MPS(ψ)
    XEB = 0.
    N = length(ψ)
    for m in 1:NM
        V = ITensor(1.)
        for j=1:N
              V *= (P0[j]*state(ξ[j],data[m,j]))
        end
        XEB += 2^N/NM*real(V[])-1/NM
    end
    return XEB
end

"""
    get_selfXEB(ψ::MPS)

Returns the self-XEB 2^N sum_s |ψ(s)|^4-1
"""
function get_selfXEB(ψ::MPS)
    P0 = get_Born_MPS(ψ)
    N = length(ψ)
    return 2^N*real(inner(P0,P0))-1
end
