using ITensors, ITensorMPS
using RandomMeas

N = 3
χ = 2^(N÷2)
nu = 100
NM = 100
ξ = siteinds("Qubit", N)
ψ = random_mps(ξ; linkdims=χ);
data = zeros(Int8,(nu,NM,N))
u = Vector{Vector{ITensor}}()


##Sampling
for r in 1:nu
    push!(u,get_rotations(ξ,1)) #Haar rotations in A
    data[r,:,:] = get_RandomMeas_MPS(ψ,u[r],NM)
end

##Factorized shadows
Fidelity_s = 0.
for r in 1:nu
    for m in 1:NM
        shadow = get_shadow_factorized(data[r,m,:],ξ,u[r])
        global Fidelity_s += get_expect_shadow(ψ,shadow,ξ)/nu/NM
    end
end
println("estimated fidelity ", Fidelity_s)

##Purity
purity_s = get_purity_hamming(data,ξ)
println("estimated purity ", purity_s)

