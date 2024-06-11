"""
    Apply_depo_channel(ρ::MPO,p::Vector{Float64})

Apply depolarization channel on all qubits with probabilities specficied by the vector p
"""
function Apply_depo_channel(ρ::MPO,p::Vector{Float64})
    N = length(ρ)
    ξ = firstsiteinds(ρ;plev=0)
    ρ1 = copy(ρ)
    for i in 1:N
        s = ξ[i]
        X = ρ1[i]*δ(s,s')
        ρ1[i]=(1-p[i])*ρ1[i]+p[i]/2*X*δ(s,s')
    end
    return ρ1
end

"""
    RandomCircuit(ξ::Vector{Index{Int64}},depth::Int64)

Create a random circuit of given depth. Returns the list of gates as a vector of ITensors
"""
function RandomCircuit(ξ::Vector{Index{Int64}},depth::Int64)
    N = length(ξ)
    circuit = ITensor[]
    for d in 1:depth
        if d%2==1
            random_layer = [op("RandomUnitary", ξ[j], ξ[j+1]) for j in 1:2:N-1]
        else
            random_layer = [op("RandomUnitary", ξ[j], ξ[j+1]) for j in 2:2:N-1]
        end
        append!(circuit, random_layer)
    end
    return circuit
end