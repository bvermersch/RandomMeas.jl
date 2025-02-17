"""
    apply_depo_channel(ρ::MPO,p::Vector{Float64})

Apply depolarization channel on all qubits with probabilities specficied by the vector p
"""
function apply_depo_channel(ρ::MPO,p::Vector{Float64})
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
    apply_depo_channel(ψ::MPS,p::Vector{Float64})

Apply depolarization channel on all qubits with probabilities specficied by the vector p
"""
function apply_depo_channel(ψ::MPS,p::Vector{Float64})
    return apply_depo_channel(outer(ψ',ψ),p)
end

"""
    random_circuit(ξ::Vector{Index{Int64}},depth::Int64)

Create a random circuit of given depth. Returns the list of gates as a vector of ITensors
"""
function random_circuit(ξ::Vector{Index{Int64}},depth::Int64)
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

# """
#     random_Pauli_layer(ξ::Vector{Index{Int64}},p::Vector{Float64})

# Create a layer of stochastic Pauli operations with probability (1-3*p/4,p/4,p/4,p/4)
# (corresponding to local depolarization with probability p)
# """
# function random_Pauli_layer(ξ::Vector{Index{Int64}},p::Vector{Float64})
#     N = length(ξ)
#     circuit = ITensor[]
#     for i in 1:N
#         if rand()>1-3*p[i]/4
#             a = rand()
#             if a<1/3
#                 push!(circuit,op("X",ξ[i]))
#             elseif a<2/3
#                 push!(circuit,op("Y",ξ[i]))
#             else
#                 push!(circuit,op("Z",ξ[i]))
#             end
#         end
#     end
#     return circuit
# end

# """
#     RandomMagneticFieldLayer(ξ::Vector{Index{Int64}},p::Vector{Float64})

# Create a layer with Random Rz gates of average angle π p.
# """
# function random_magnetic_field_layer(ξ::Vector{Index{Int64}},p::Vector{Float64})
#     N = length(ξ)
#     circuit = ITensor[]
#     for i in 1:N
#         push!(circuit,op("Rz",ξ[i];θ=2*π*p[i]*rand()))
#     end
#     return circuit
# end