#This function gets the first two trace moments p2,p3
#from the U-stat on nu shadows
function get_moments_shadows(shadow::Vector{ITensor},sitesA::Vector{Index{Int64}},n::Int64,nu::Int64)
	p = Vector{Float64}()
	ps = real(trace(square(shadow[1])-shadow[2],sitesA))
	push!(p,ps/(nu*(nu-1)))

	if n==3
		p3s = trace(power(shadow[1],3),sitesA)
		p3s -= 3*trace(multiply(shadow[2],shadow[1]),sitesA)
		p3s += 2*trace(shadow[3],sitesA)
		push!(p,real(p3s)/(nu*(nu-1)*(nu-2)))
	end
	return p
end

#This function acquires nu shadows from Pauli randomized measurements
function acquire_shadows(rhoA,sitesA::Vector{Index{Int64}},b::Matrix{Int64},n::Int64,NM::Int64,sigmaA::Union{ITensor,Nothing}=nothing)
	shadow = Vector{ITensor}()
	NA = size(b,2)
	nu = size(b,1)
	for m in 1:n
		push!(shadow,ITensor(vcat(sitesA,prime(sitesA))))
	end
	for r in 1:nu
		data = get_RandomMeas(rhoA,b[r,:],NM)
		P = get_Born_data(data,sitesA)
		shadow_temp = get_shadow(P,sitesA,b[r,:])
		if sigmaA != nothing
			#this realizes CRM shadows by simulating measurements on the reference state sigma
			sigmaA_b = rotate_b(sigmaA,sitesA,b[r,:])
			P_sigmaA_b = get_Born(sigmaA_b,sitesA)
			shadow_sigmaA = get_shadow(P_sigmaA_b,sitesA,b[r,:])
			shadow_temp += -shadow_sigmaA+sigmaA
		end
		# it is useful to also export the moments for shadows when estimating trace moments
		for m in 1:n
			shadow[m] += power(shadow_temp,m)
		end
	end
	return shadow
end

#get trace moments from batch shadows using U-stat
function get_moments_shadows_batch(shadow::Vector{ITensor},sitesA::Vector{Index{Int64}},n::Int64,nu::Int64)
	p = Vector{Float64}()

	for m in 2:n
		r_a = collect(permutations(1:n,m)) #m_uplet of n batches
		alpha = length(r_a)
		est = 0
		push!(p,0)
		for r in r_a
			X = multiply(shadow[r[1]],shadow[r[2]])
			for m1 in 3:m
				X = multiply(X,shadow[r[m1]])
			end
			p[m-1] += real(trace(X,sitesA))
		end
		p[m-1] /= alpha

	end
	return p
end

#acquire CRM batch shadows (n is the number of batches, nu the total number of shadows)
function acquire_shadows_batch(rhoA,sitesA::Vector{Index{Int64}},b::Matrix{Int64},n::Int64,NM::Int64,sigmaA::Union{ITensor,Vector{ITensor},Nothing}=nothing)
	shadow = Vector{ITensor}()
	NA = size(b,2)
	nu = size(b,1)
	for m in 1:n
		push!(shadow,ITensor(vcat(sitesA,prime(sitesA))))
	end
	nu_n = nu ÷ n
	for t in 1:n
		for r in 1+(t-1)*nu_n:t*nu_n
			data = get_RandomMeas(rhoA,b[r,:],NM)
			P = get_Born_data(data,sitesA)
			shadow_temp = get_shadow(P,sitesA,b[r,:])
			if sigmaA != nothing
				if typeof(sigmaA)==ITensor
					sigmaA_b = 1*sigmaA
				else
					sigmaA_b = 1*sigmaA[t]
				end
				#First we shift by sigma
				shadow_temp += sigmaA_b
				sigmaA_b = rotate_b(sigmaA_b,sitesA,b[r,:])
				P_sigmaA_b = get_Born(sigmaA_b,sitesA)
				shadow_sigmaA = get_shadow(P_sigmaA_b,sitesA,b[r,:])
				#Then we strict the simulated measurement shadow sigma(r)
				shadow_temp += -shadow_sigmaA
			end
			shadow[t] += shadow_temp/nu_n
		end
	end
	return shadow
end

#form shadows associated with a MPS/MPO and a basis transformation b, return shadow as MPO
function get_shadow(state::Union{MPS,MPO},sites::Vector{Index{Int64}},bases::Array{Int})
	stateu = rotate_b(state,bases)
	N = size(bases,1)
	P = get_Born_MPS(stateu,sites)
	u,c,d,e = get_rotations()
	h_tensor,a,b = get_h_tensor()

	shadow = MPO(sites)
	for i in 1:N
		si = sites[i]
		shadow[i] = P[i]* 2*h_tensor*delta(a,si)
		shadow[i] *= delta(b,si)
		ut = u*onehot(e=>bases[i])
		shadow[i] *= prime(ut)
		shadow[i] *= dag(ut)
		shadow[i] *= delta(si,prime(c),c)
		shadow[i] *= delta(d,si)
		shadow[i] *= delta(prime(d),prime(si))
	end
	return shadow
end

#form shadows from measurement results
function get_shadow(data::Array{Int64},sites::Vector{Index{Int64}},b::Array{Int})
	NA = size(b,1)
	NM = size(data,1)
	#(Borrowed from PastaQ)
	M = hcat(digits.(data.-1, base=2, pad=NA)...)'
	M = M .+1
	u,c,d,e = get_rotations()
	rho = 0*ITensor(vcat(sites,prime(sites)))
	for s in 1:NM
		rho_t  = 1
		for i in 1:NA
			si = sites[i]
			psi = dag(u)*onehot(e=>b[i])*onehot(c=>M[s,i])
			psi *= delta(d,si)
			rho_t *= (3*psi*prime(dag(psi))-delta(si,prime(si)))
		end
		rho += rho_t
	end
	return rho/NM
end

#from shadows expressed as tensor product from measurement data M
function get_shadow_factorized(M::Array{Int64},sites::Vector{Index{Int64}},b::Array{Int})
	NA = size(b,1)
	NM = size(M,1)
	if length(size(M))<2
		M  = hcat(digits.(M.-1, base=2, pad=NA)...)'
		M = M .+1
	end
	u,c,d,e = get_rotations()
	rho = Vector{MPO}()
	for s in 1:NM
		rho_t = MPO(sites)
		for i in 1:NA
			si = sites[i]
			psi = dag(u)*onehot(e=>b[i])*onehot(c=>M[s,i])
			psi *= delta(d,si)
			rho_t[i] = (3*psi*prime(dag(psi))-delta(si,prime(si)))
		end
		push!(rho,rho_t)
	end
	return rho
end


#Expectation of O for a shadow expressed as MPO
function get_expect_shadow(O::MPO,shadow::Vector{MPO},sites::Vector{Index{Int64}})
	NM = size(shadow,1)
	N = size(sites,1)
	O_e = 0
	for s in 1:NM
		O_e += inner(O,shadow[s])/NM
	end
	return O_e
end

#Expectation of |psi><psi| for a shadow expressed as MPO
function get_expect_shadow(psi::MPS,shadow::Vector{MPO},sites::Vector{Index{Int64}})
	NM = size(shadow,1)
	N = size(sites,1)
	O_e = 0
	for s in 1:NM
		O_e += inner(psi',shadow[s],psi)/NM
	end
	return O_e
end

#Expectation of O for a shadow expressed as dense Itensor
function get_expect_shadow(O::MPO,shadow::MPO,sites::Vector{Index{Int64}})
	O_e = inner(O,shadow)
	return O_e
end


#form shadow from Born probabilities P expressed as dense ITensor, returns shadows as dense ITensor
function get_shadow(P::ITensor,sites::Vector{Index{Int64}},b::Array{Int})
	NA = size(b,1)
	Hamming_matrix = zeros(Float64,(2,2))	
	Hamming_matrix[1,1] = 1
	Hamming_matrix[2,2] = 1
	Hamming_matrix[2,1] = -0.5
	Hamming_matrix[1,2] = -0.5

	r_tensor,c,d,e = get_rotations()
	i1 = Index(2,"i1")
	i2 = Index(2,"i2")
	Hamming_tensor = itensor(Hamming_matrix,i1,i2)
	rho = 2^NA*copy(P)
	for i in 1:NA
		s = sites[i]
		h = Hamming_tensor*delta(i1,s)*delta(i2,prime(s,2))
		rho = rho*h
		rho *= delta(s,prime(s),prime(s,2))

		u = dag(r_tensor)*onehot(e=>b[i])
		u *= delta(c,s)*delta(d,prime(s,2))
		rho =  mapprime(u*rho,2,0)

		udag = r_tensor*onehot(e=>b[i])
		udag *= delta(c,prime(s))*delta(d,prime(s,2))
		rho =  mapprime(udag*rho,2,1)
	end
	return rho
end

function square(shadow::ITensor)
	Y = multiply(shadow,shadow)
	return Y
end

function multiply(shadow::ITensor,shadow2::ITensor)
	Y = (shadow*prime(shadow2))
	Y = mapprime(Y,2,1)
	return Y
end

function power(shadow::ITensor,n::Int64)
	Y = deepcopy(shadow)
	for m in 1:n-1
		Y = multiply(Y,shadow)
	end
	return Y
end

function trace(shadow::ITensor,sites::Vector{Index{Int64}})
	NA = size(sites,1)
	Y = copy(shadow)
	for i in 1:NA
		Y *= delta(sites[i],prime(sites[i]))
	end
	return scalar(Y)
end
