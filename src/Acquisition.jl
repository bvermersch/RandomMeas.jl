"""
    get_rotations!

Generate a list of N single qubit unitaries to be applied on each qubit
"""
function get_rotations(ξ::Vector{Index{Int64}}, cat::Int=1)
    u = Vector{ITensor}()
    N = length(ξ)
    for i in 1:N
        push!(u, get_rotation(ξ[i], cat))
    end
    return u
end

"""
    get_rotation(ξ::Index{Int64}, cat::Int)

Generate a single qubit unitary with indices (ξ',ξ)
Categories specified by  cat:
    #1: Haar
    #2: Pauli
    #6: Identity matrix
"""
function get_rotation(ξ::Index{Int64}, cat::Int)
    r_matrix = zeros(ComplexF64, (2, 2))
    if cat == 1
        return op("RandomUnitary", ξ)
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
        r_tensor = itensor(r_matrix, ξ', ξ)
        return r_tensor
    elseif cat == 6
        b = rand(1:2)
        r_matrix[1, 1] = 1
        r_matrix[2, 2] = 1
        r_tensor = itensor(r_matrix, ξ', ξ)
        return r_tensor
    end
end

"""
    get_RandomMeas!(data_s::Array{Int8}, ρ::Union{MPO,MPS}, u::Vector{ITensor})

Sample randomized measurements from a MPS/MPO representation ρ 
"""
function get_RandomMeas!(data_s::Array{Int8}, ρ::Union{MPO,MPS}, u::Vector{ITensor})
    #ρu = rotate_b(ρ, u)
    if typeof(ρ)==MPS
        ρu = apply(u,ρ)
    else
        ρu = apply(u,ρ;apply_dag=true)
    end
    get_Samples_Flat!(data_s, ρu)
end

"""
    get_Samples_Flat!(data_s::Array{Int8}, state::Union{MPO,MPS})

Sample randomized measurements from a MPS/MPO representation ρ 
"""
function get_Samples_Flat!(data_s::Array{Int8}, state::Union{MPO,MPS})
    NM,N = size(data_s)
    #This is borrowed from PastaQ
    Prob = get_Born(state)
    prob = real(array(Prob))
    prob = reshape(prob, 2^N)
    for m in 1:NM
        data = StatsBase.sample(0:(1<<N-1), StatsBase.Weights(prob), 1)
        data_s[m, :] = 1 .+ digits(data[1], base=2, pad=N)
    end
end

"""
    get_RandomMeas_MPO!

Sample randomized measurements from an MPO representation ρ. The sampling is based from the MPO directly, i.e is memory-efficient 
"""
function get_RandomMeas_MPO!(data::Array{Int8}, ρ::MPO, u::Vector{ITensor}, NM::Int64)
    ξ = firstsiteinds(ρ;plev=0)
    #ρu = rotate_b(ρ, u)
    ρu = apply(u,ρ;apply_dag=true)
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


"""
    get_RandomMeas_MPS!(data::Array{Int8}, ψ::MPS, u::Vector{ITensor})

Sample randomized measurements from an MPS representation ψ. The sampling is based from the MPS directly, i.e is memory-efficient 
"""
function get_RandomMeas_MPS!(data::Array{Int8}, ψ::MPS, u::Vector{ITensor})
    NM = size(data,1)
    #ppsiu = rotate_b(psi, u)
    ψu = apply(reverse(u),ψ) #using reverse allows us to maintain orthocenter(ψ)=1 ;)
    for m in 1:NM
        data[m, :] = ITensors.sample(ψu)#[1:NA]
    end

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

Construct Born Probability vector P(s)=<s|ρ|s> from an MPO representation ρ 
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

