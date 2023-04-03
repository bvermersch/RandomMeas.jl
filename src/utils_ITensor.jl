function flatten(O::Union{MPS,MPO})
	N = length(O)
	O_f = O[1]
	for i in 2:N
		O_f *= O[i]
	end
	return O_f
end

function trace(rho::MPO,s::Vector{Index{Int64}})
	NA = size(s,1)
	X = rho[1]*delta(s[1],prime(s[1]))
	for i in 2:NA
		X *= rho[i]*delta(s[i],prime(s[i]))
	end
	return scalar(X)
end

function reduce_dm(rho::MPO,part::Vector{Int64})
	N = length(rho)
	NA = size(part,1)
	so = siteinds(rho)
	s = siteinds(2,N)
	for i in 1:N
		s[i] = noprime(so[i][1])
	end
	sA = s[part]
	rhoA = MPO(sA)
	L = 1
	for i in 1:part[1]-1
		L *= rho[i]*delta(s[i],prime(s[i]))
	end
	for j in 1:NA
		if j<NA
			imax = part[j+1]-1
		else
			imax = N
		end
		R = 1
		for i in part[j]+1:imax
			R *= rho[i]*delta(s[i],prime(s[i]))
		end
		rhoA[j] = L*rho[part[j]]*R
		L = 1
	end
	orthogonalize!(rhoA,1)
	return rhoA,sA
end

function state_to_dm(state::MPS)
	s = siteinds(state)
	rho = MPO(s)
	N = length(state)
	for i in 1:N
		si = noprime(s[i])
		rho[i] = state[i]*dag(prime(state[i],2))
		rho[i] *= delta(si,prime(si))
		rho[i] *= delta(si,prime(si,2))
	end
	orthogonalize!(rho,1) #NOT NEEDED IN PRINCIPLE
	#orthogonalize!(rho,1,cutoff=1e-8)
	return rho
end

function multiply(O1::MPO,O2::MPO)
	sites = siteinds(O1)
	NA = length(sites)
	sitesnp = siteinds("S=1/2", NA)
	for i in 1:NA
		sitesnp[i] = noprime(sites[i][1])
	end
	O = MPO(sitesnp)
	for i in 1:NA
		O[i] = O1[i]*prime(O2[i])
		s = noprime(sitesnp[i])	
		O[i] *= delta(prime(s,2),prime(s))
	end
	orthogonalize!(O,1)
	return O
end

function power(O1::MPO,n::Int)
	O = copy(O1)
	for m in 1:n-1
		O = multiply(O,O1)
	end
	orthogonalize!(O,1)
	return O
end



function reduce_dm_onesite(state::Union{MPS,MPO},i::Int64)
	N = length(state)
	if typeof(state)==MPS
		rho  = MPO([s])
		s = siteinds(state)[i]
		orthogonalize!(state,i)
		rho[i] = state[i]*prime(dag(state[i]),s)
	else	##TO BE DONE REDUCED STATES OF MIXED STATES NA<=N
	end
	return rho
end


function get_purity(state::MPS,NA::Int64)
	N = length(state)
	if NA<N
		spec = get_spectrum(state,NA)
		p = get_moment(spec,2)
	else
		p = 1
	end
	return p
end


function get_purity(state::MPO)
	orthogonalize!(state,1)
	return real(scalar(state[1]*dag(state[1])))
end


function get_spectrum(state::MPS,NA::Int64)
	orthogonalize!(state,NA)
	if NA>1
		U,spec,V = svd(state[NA], (linkind(state, NA-1), siteind(state,NA)))
	else
		U,spec,V = svd(state[NA], siteind(state,NA))
	end
	return spec
end

function get_moment(spec::ITensor,n::Int)
	p = Vector{Float64}()
	for m in 2:n
		pm = 0
		for l=1:dim(spec, 1)
			pm += spec[l,l]^(2*m)
		end
		push!(p,pm)
	end
	return p
end

function get_entropy(spec::ITensor)
	S = 0
	for l=1:dim(spec, 1)
		x = spec[l,l]^2
		S -= x*log(x)
	end
	return S
end

function ITensortoMPO(X::ITensor,sites::Vector{Index{Int64}})
	rho = MPO(sites)
	NA = size(sites,1)
	for i in 1:NA-1
		s = sites[i]
		if i>1
			U,spec,V = svd(X, (li,s,prime(s)))
		else
			U,spec,V = svd(X, (s,prime(s)))
		end
		rho[i] = U
		global li = commonindex(U,spec)
		X = spec*V
	end
	rho[NA] = X
	orthogonalize!(rho,1)
	return rho
end

function get_KL(P::MPS,Q::MPS)
	N = length(P)
	Pf = 1
	Qf = 1
	for i in 1:N
		Pf *= P[i]
		Qf *= Q[i]
	end
	Pa = abs.(array(Pf))
	Qa = abs.(array(Qf))
	Pa /= sum(Pa)
	Qa /= sum(Qa)
	return sum(Pa .* log.(Pa ./Qa))
end
