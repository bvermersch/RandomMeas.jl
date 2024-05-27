using ITensors, ITensorMPS
using RandomMeas

N  = 2
ψ = random_mps(siteinds("Qubit", 2*N); linkdims=2^N);
ρ,ξ = reduce_dm(ψ,1,N)

nu=100 #number of random unitaries
NM=100 #number of projective measurements
data = zeros(Int8,(nu,NM,N))
for r in 1:nu
    #generate Haar-random single qubit rotations
    u = get_rotations(ξ,1)
    #acquire RM measurements
    data[r,:,:] = get_RandomMeas(ρ,u,NM)
end

purity_e = get_purity_hamming(data,ξ)
println("estimated purity ", purity_e)
println("exact purity ", get_purity(ρ))

