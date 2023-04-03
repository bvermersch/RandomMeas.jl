using Random
using ITensors
using NPZ
push!(LOAD_PATH,pwd()*"/src/")
using RandomMeas
using StatsBase
let 
	N  = 8
	NA = 4
	nu = 100
	NM = 200
	chi = 5
	chisigma = 2
	n = 3

	ITensors.set_warn_order(20)
	# We form the critical Ising model
	ops = OpSum()
	for j in 1:(N - 1)
		ops += -1, "Z", j, "Z", j + 1
	end
	for j in 1:N
		ops += "X", j
	end
	sites = siteinds("S=1/2", N)
	H = MPO(ops, sites)
	psi0 = randomMPS(sites; linkdims=chi)
	# define parameters for DMRG sweeps
	nsweeps = 10
	maxdim = [10, 20, chi]
	cutoff = [1E-10]
	# We obtain the ground state via DMRG
	energy, psi = dmrg(H, psi0; nsweeps, maxdim, cutoff)
	println(" -- State Built with Max Bond Dim -- ", maxlinkdim(psi)) 
	sites = siteinds(psi)
	rho = state_to_dm(psi)
	println(" -- D.M. state Built -- ") 

	part = collect(1:NA)
	# We obtain the reduced density matrix
	rhoA,sitesA = reduce_dm(rho,part)
	println(" -- Reduced D.M. state Built -- ") 

	spec = get_spectrum(psi,NA)
	p = get_moment(spec,n)
	S = get_entropy(spec)
	S3 = 137/60-4*p[1]+ 7/4*p[2]
	println(" -- von Neumman Entropy -- ", S) 
	println(" -- S3 Entropy -- ", S3) 
	
	#We form the approximation sigma
	psi_sigma = deepcopy(psi)
	truncate!(psi_sigma,maxdim=chisigma)
	psi_sigma[1] = psi_sigma[1]/norm(psi_sigma[1])
	println(" -- Bond dim psi -- ",maxlinkdim(psi))
	println(" -- Bond dim psi_sigma -- ",maxlinkdim(psi_sigma))
	spec_sigma = get_spectrum(psi_sigma,NA)
	p= get_moment(spec_sigma,n)
	S3_sigma = 137/60-4*p[1]+ 7/4*p[2]
	println(" -- S3 sigma Entropy -- ", S3_sigma) 
	sigma = state_to_dm(psi_sigma)
	sigmaA,sitesA = reduce_dm(sigma,part)
	#In this specific example, we flatten the MPO and form an array 2^Nx2^N to minimize the runtime in postprocessing batch shadows
	sigmaA = flatten(sigmaA)

	#Generation of nu measurement settings (b=1,2,3 means Z,X,Y measurements respectively)
	b = rand(1:3,nu,NA)
	shadow = acquire_shadows_batch(rhoA,sitesA,b,n,NM)
	ps = get_moments_shadows_batch(shadow,sitesA,n,nu)
	S3s = 137/60-4*ps[1]+ 7/4*ps[2]
	println(" -- Shadow -- ",S3s, " Error %: ",floor(sum(abs.(S3s-S3)./S3*100)))

	shadow = acquire_shadows_batch(rhoA,sitesA,b,n,NM,sigmaA)
	pCRM = get_moments_shadows_batch(shadow,sitesA,n,nu)
	S3CRM = 137/60-4*pCRM[1]+ 7/4*pCRM[2]
	println(" -- CRM Shadow -- ",S3CRM, " Error %: ",floor(sum(abs.(S3CRM-S3)./S3*100)))
end
