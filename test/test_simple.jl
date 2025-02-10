using ITensors, ITensorMPS
using RandomMeas

N  = 2
ψ = random_mps(siteinds("Qubit", 2*N); linkdims=2^N);
ρ,ξ = reduce_dm(ψ,1,N)

Nu=100 #number of random unitaries
NM=1000 #number of projective measurements
data = zeros(Int,(Nu,NM,N))
u = Vector{Vector{ITensor}}()
for r in 1:Nu
    #generate Haar-random single qubit rotations
    push!(u,get_rotations(ξ,"Haar"))
    #acquire RM measurements
    data[r,:,:] = get_RandomMeas(ρ,u[r],NM,"dense")
end

O = MPO(ξ)
O[1] = op("Z",ξ[1])
O[2] = op("Id",ξ[2])
O_e = 0.
for r in 1:Nu
    P = get_Born(data[r,:,:],ξ)
    shadow = get_shadow(P,ξ,u[r])
    global O_e += get_expect_shadow(O,shadow,ξ)/Nu
end
println("estimated expectation value ", O_e)
println("exact expectation value ", inner(O,ρ))


purity_e = get_purity_hamming(data,ξ)
println("----")
println("estimated purity ", purity_e)
println("exact purity ", get_purity(ρ))
