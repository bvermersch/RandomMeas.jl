using Random
using ITensors
using NPZ
push!(LOAD_PATH,pwd()*"/src/")
using RandomMeas
using StatsBase
using PastaQ

	
N = 6 ## number of qubits
nu = 10 ## number of random unitaries
NM = 1000  ## number of measurements per random unitary
d = "5"  ## depth of the random circuit generate the state 
p = 0.001  ## per gate error probability in the random circuit
pure = false  



## get the random circuit state 
psi,rho_noisy,sites = get_State(d,N, pure, p)

chimax = 3  ## maximum bond-dimension of sigma to the considered
chisigma = range(1,chimax,step =1) 

##Store  Fidelity esitmations for multiple states
Fid_uni = zeros(Float64, length(chisigma))
Fid_CRM = zeros(Float64, length(chisigma))
Fid_sigma = 1 
Fid_theory  = zeros(Float64, length(chisigma))


## perform randomized Pauli measurements on quantum device and store the shadows

b = rand(1:3, N, nu) ## generate  measurement settings (b=1,2,3 means Z,X,Y measurement settings respectively)
#Actual measurement
shadow = Vector{Vector{MPO}}()
for r in 1:nu   #length(chisigma)
	data = get_RandomMeas_MPO(rho_noisy,b[:,r],NM,sites)
	push!(shadow,get_shadow_factorized(data,sites,b[:,r]))
end

#Off-line postprocessing of fidelity for multiple states: 
for ichi in 1:length(chisigma)

	## Form approximation of the state sigma
	psi_sigma = deepcopy(psi)
	truncate!(psi_sigma, maxdim = chisigma[ichi])
	psi_sigma[1] = psi_sigma[1]/norm(psi_sigma[1])
	sigma = state_to_dm(psi_sigma)

	Fid_theory[ichi] = real(inner(psi_sigma',rho_noisy,psi_sigma))
	temp_sigma = 0

	## compute the DFE wrt approximations sigma
	for r in 1:nu   
		sigma_shadow = get_shadow(sigma, sites,b[:,r])
		Fid_uni[ichi] += real(get_expect_shadow(sigma,shadow[r],sites))/nu
		temp_sigma += real(get_expect_shadow(sigma,sigma_shadow,sites))/nu
	end
	
	Fid_CRM[ichi] = Fid_uni[ichi] - temp_sigma + Fid_sigma

println(" -- True value of Fidelity -- ",Fid_theory[ichi]) 
println(" -- Shadow estimation with chi ",chisigma[ichi], " -- ",Fid_uni[ichi], " Error %: ",floor(sum(abs.(Fid_uni[ichi]-Fid_theory[ichi])./Fid_uni[ichi]*100)))
println(" -- CRM Shadow estimation with chi ",chisigma[ichi], " -- ",Fid_CRM[ichi], " Error %: ",floor(sum(abs.(Fid_CRM[ichi]-Fid_theory[ichi])./Fid_CRM[ichi]*100)))

end
