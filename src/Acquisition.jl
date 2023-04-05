function get_rotations()
	c = Index(2,"c")
	d = Index(2,"d")
	e = Index(3,"e")
	r_matrix = zeros(ComplexF64,(2,2,3))
	r_matrix[1,1,1] = 1
	r_matrix[2,2,1] = 1

	r_matrix[1,1,2] = 1/sqrt(2)
	r_matrix[2,2,2] = 1/sqrt(2)
	r_matrix[1,2,2] = -1/sqrt(2)
	r_matrix[2,1,2] = 1/sqrt(2)

	r_matrix[1,1,3] = 1/sqrt(2)
	r_matrix[2,2,3] = 1/sqrt(2)
	r_matrix[1,2,3] = -1im/sqrt(2)
	r_matrix[2,1,3] = -1im/sqrt(2)
	r_tensor = itensor(r_matrix,c,d,e)
	return r_tensor,c,d,e
end

function rotate_b(psi::MPS,b::Array{Int})
	sites  = siteinds(psi)
	r_tensor,c,d,e = get_rotations()
	psiu = deepcopy(psi)
	N = size(b,1)
	for i in 1:N
		u = r_tensor*onehot(e=>b[i])
		u *= delta(d,sites[i])*delta(c,prime(sites[i]))
		psiu[i] = noprime(u*psi[i])
	end
	orthogonalize!(psiu,1) #THIS STEP IS A BIT STUPID TO BE FIXED
	return psiu
end

function rotate_b(rhoA::MPO,b::Array{Int})
	sites  = siteinds(rhoA)
	r_tensor,c,d,e = get_rotations()
	rhou = deepcopy(rhoA)
	NA = size(b,1)
	for i in 1:NA
		s = noprime(sites[i][1])
		u = r_tensor*onehot(e=>b[i])
		u *= delta(d,s)*delta(c,prime(s,2))
		rhou[i] =  mapprime(u*rhoA[i],2,0)
		udag = dag(r_tensor)*onehot(e=>b[i])
		udag *= delta(d,prime(s))*delta(c,prime(s,2))
		rhou[i] =  mapprime(udag*rhou[i],2,1)
	end
	orthogonalize!(rhou,1) #THIS STEP IS A BIT STUPID TO BE FIXED
	return rhou
end

function rotate_b(rhoA::ITensor,sites::Vector{Index{Int64}},b::Array{Int})
	r_tensor,c,d,e = get_rotations()
	rhou = deepcopy(rhoA)
	NA = size(b,1)
	for i in 1:NA
		s = sites[i]
		u = r_tensor*onehot(e=>b[i])
		u *= delta(d,s)*delta(c,prime(s,2))
		rhou *=  u
		rhou = mapprime(rhou,2,0)
		udag = dag(r_tensor)*onehot(e=>b[i])
		udag *= delta(d,prime(s))*delta(c,prime(s,2))
		rhou *= udag
		rhou =  mapprime(rhou,2,1)
	end
	return rhou
end

function get_RandomMeas(state::Union{MPO,MPS},b::Vector{Int64},NM::Int64,f::Float64=1.)
	stateu = rotate_b(state,b)
	NA = size(b,1)
	data = get_Samples_Flat(stateu,b,NM,f)
	return data
end

function get_RandomMeas_MPO(rhoA::MPO,b::Vector{Int64},NM::Int64, sites::Vector{Index{Int64}})
	## A BIT SLOW FOR SMALL N
	rhou = rotate_b(rhoA,b)
	NA = size(b,1)
	rhou[1] /= trace(rhou,sites)
	data = zeros(Int64,(NM,NA))
	for m in 1:NM
		data[m,:] = ITensors.sample(rhou)
	end
	return data
end

function get_Samples_Flat(state::Union{MPO,MPS},b::Array{Int},NM::Int64,f::Float64)
	#This is borrowed from PastaQ
	NA = size(b,1)
	data = zeros(Int64,(NM,NA))
	Prob = get_Born(state,f)
	prob  = real(array(Prob))
	prob = reshape(prob,2^NA)
	#println("Acquisition")
	#@time begin
	data = StatsBase.sample(0:(1 << NA - 1), StatsBase.Weights(prob), NM)
	data = data .+ 1
	#end
	return data
end

function get_Born_MPS(rho::MPO,sites::Vector{Index{Int64}})
	NA = size(sites,1)
	P = MPS(sites)
	for i in 1:NA
		Ct = delta(sites[i],prime(sites[i]),prime(sites[i],2))
		P[i] = rho[i]*Ct
		P[i] *= delta(sites[i],prime(sites[i],2))
	end
	return P
end

function get_Born_MPS(psi::MPS)
	sites = siteinds(psi)
	N = size(sites,1)
	P = MPS(sites)
	for i in 1:NA
		Ct = delta(sites[i],prime(sites[i]),prime(sites[i],2))
		P[i] = psi[i]*prime(conj(psi[i]))*Ct
		P[i] *= delta(sites[i],prime(sites[i],2))
	end
	#if NA<N
	#	right = commoninds(psi[NA],psi[NA+1])
	#	P[NA]*= delta(right,prime(right))
	#end
	return P
end

function get_Born(psi::MPS,f::Float64)
	sites = siteinds(psi)
	N = size(sites,1)

	C = delta(sites[1],prime(sites[1]),prime(sites[1],2))
	R = C*psi[1]*prime(conj(psi[1]))
	R *= delta(sites[1],prime(sites[1],2))
	P = R
	for i in 2:N
		Ct = delta(sites[i],prime(sites[i]),prime(sites[i],2))
		Rt = psi[i]*prime(conj(psi[i]))*Ct
		Rt *= delta(sites[i],prime(sites[i],2))
		P  *= Rt
	end
#	if NA<N
#		right = commoninds(psi[NA],psi[NA+1])
#		P*= delta(right,prime(right))
#	end
	if f<1-10^(-9)
		a = Index(2,"a")
		b = Index(2,"b")
		C = f*delta(a,b)
		C += (1-f)*onehot(a=>1,b=>2)
		C += (1-f)*onehot(a=>2,b=>1)
		for i in 1:N
			P *= C*delta(a,sites[i])
			P *= delta(b,sites[i])
		end
	end
	return P
end

function get_Born(rhoA::MPO,f::Float64)
	sites = siteinds(rhoA)
	NA = size(sites,1)
	s = noprime(sites[1][1])
	a = Index(2,"a")
	P = rhoA[1]*delta(s,prime(s),a)
	P *= delta(a,s)
	for i in 2:NA
		s = noprime(sites[i][1])
		C = rhoA[i]*delta(s,prime(s),a)
		C *= delta(a,s)
		P  *= C
	end
	if f<1-10^(-9)
		a = Index(2,"a")
		b = Index(2,"b")
		C = f*delta(a,b)
		C += (1-f)*onehot(a=>1,b=>2)
		C += (1-f)*onehot(a=>2,b=>1)
		for i in 1:NA
			s = noprime(sites[i][1])
			P *= C*delta(a,s)
			P *= delta(b,s)
		end
	end
	return P
end


#function get_Born(rhoA::MPO,sites::Vector{Index{Int64}})
#	NA = size(sites,1)
#	s = sites[1]
#	a = Index(2,"a")
#	P = rhoA[1]*delta(s,prime(s),a)
#	P *= delta(a,s)
#	for i in 2:NA
#		s = sites[i]
#		C = rhoA[i]*delta(s,prime(s),a)
#		C *= delta(a,s)
#		P  *= C
#	end
#	return P
#end
function get_Born(rhoA::ITensor,sites::Vector{Index{Int64}})
	NA = size(sites,1)
	s = sites[1]
	a = Index(2,"a")
	P = deepcopy(rhoA)
	for i in 1:NA
		s = sites[i]
		P = P*delta(s,prime(s),a)
		P = P*delta(a,s)
	end
	return P
end

function get_State(state::String,N::Int64,pure::Bool,p=0.)
	if state=="GHZ"
		sites = siteinds("S=1/2", N);
		psi = MPS(sites,linkdims=2)
		l = commonind(psi[1],psi[2])
		psi[1][sites[1]=>1,l=>1] = 1/sqrt(2) 
		psi[1][sites[1]=>2,l=>2] = 1/sqrt(2) 
		for i in 2:N-1
			l_l = commonind(psi[i-1],psi[i])
			l_r = commonind(psi[i],psi[i+1])
			psi[i][sites[i]=>1,l_l=>1,l_r=>1] = 1
			psi[i][sites[i]=>2,l_l=>2,l_r=>2] = 1
		end
		l = commonind(psi[N],psi[N-1])
		psi[N][sites[N]=>1,l=>1] = 1
		psi[N][sites[N]=>2,l=>2] = 1
		orthogonalize!(psi,1)
	elseif state == "product"
		sites = siteinds("S=1/2", N);
		psi = MPS(sites)
		for i in 1:N
			psi[i]= onehot(sites[i]=>1)
		end
		orthogonalize!(psi,1)
	elseif state == "randomproduct"
		sites = siteinds("S=1/2", N);
		psi = randomMPS(ComplexF64, sites; linkdims=1)
	else
		depth = parse(Int64,state)
		circuit = randomcircuit(N, depth=depth; twoqubitgates = "CX", onequbitgates = "Rn")
		hilbert = qubits(N)
		psi = runcircuit(hilbert, circuit)
		sites = siteinds(psi)
		if pure==false
			#rho = runcircuit(circuit; noise=("amplitude_damping", (γ=gamma,)))
			noisemodel = (1 => ("depolarizing", (p = p,)),2 => ("depolarizing", (p = p,)))
                        rho = runcircuit(circuit; noise=noisemodel)

			sites_MPO = siteinds(rho)
			for i in 1:N
				s = noprime(sites_MPO[i][1])
				rho[i]*= delta(sites[i],s)
				rho[i]*= delta(prime(sites[i]),prime(s))
			end
			orthogonalize!(rho,1)
		end
	end
	if pure
		rho = state_to_dm(psi)
	end
	return psi,rho,sites
end



