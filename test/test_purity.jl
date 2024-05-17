using ITensors,ITensorMPS
using RandomMeas

N  = 3
χ = 2^(N÷2)
nu=1000
NM=100
ξ = siteinds("Qubit", N)
ψ = random_mps(ξ; linkdims=χ);
data = zeros(Int8,(nu,NM,N))
datat = zeros(Int8,(NM,N))
u = Vector{Vector{ITensor}}()
for r in 1:nu
    push!(u,get_rotations(ξ,1)) #Haar rotations in A
    get_RandomMeas_MPS!(datat,ψ,u[r])
    data[r,:,:] = datat[:,:]
end
purity = get_purity_hamming(data,ξ)

