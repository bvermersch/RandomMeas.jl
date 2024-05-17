function mergeindices(A::ITensor,B::ITensor,s1::Index{Int64},s2::Index{Int64})
	s3 = Index(dim(s1))
	X = A*B*delta(s3,s1,s2)
	return X*delta(s3,s1)
end

function flatten(O::Union{MPS,MPO,Vector{ITensor}})
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

function reduce_dm(rho::MPO,i::Int64,j::Int64)
	N = length(rho)
	so = siteinds(rho)
	s = siteinds(2,N)
	for i in 1:N
		s[i] = noprime(so[i][1])
	end
	sA = s[i:j]
	rhoA = MPO(sA)

	for k in i:j
		rhoA[k-i+1] = rho[i]
	end

	L = 1
	for k in 1:i-1
		L *= rho[k]*delta(s[k],s[k]')
	end
	rhoA[i] *= L
	
	R = 1
	for k in j+1:N
		R *= rho[k]*delta(s[k],s[k]')
	end
	rhoA[j] *= R
	orthogonalize!(rhoA,1)
	return rhoA,sA
end

function reduce_dm(psi::MPS,i::Int64,j::Int64)
	N = length(psi)
	s = siteinds(psi)
	sA = s[i:j]
	rhoA = MPO(sA)
	for l in i:j
		rhoA[l-i+1] = psi'[l]*dag(psi[l])
	end
	if i>1
		l = commonindex(psi[i-1],psi[i])
		rhoA[1] *= delta(l,l')
	end
	if j<N
		l = commonindex(psi[j],psi[j+1])
		rhoA[j-i+1] *= delta(l,l')
	end
	return rhoA,sA
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

function power(O1::Union{MPS,MPO},n::Int)
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
		p = get_moment(spec,2)[1]
	else
		p = 1
	end
	return p
end

function get_purity(ψ::MPS,i::Int64,j::Int64)
    N = length(ψ)
    ξ = siteinds(ψ)
    ξA = ξ[i:j]
    X = 1
    if i>1
        l = commonindex(ψ[i-1],ψ[i])
        X *= δ(l,l''')*δ(l'',l')
    end
    for l in i:j
        Y = ψ[l]*dag(ψ'[l])*δ(ξ[l],ξ[l]')
        X *= Y*Y''
    end
    if j<N
        l = commonindex(ψ[j],ψ[j+1])
        X *= δ(l,l''')*δ(l'',l')
    end
    return real(X[])
end

function get_purity(state::MPO)
	orthogonalize!(state,1)
	return real(scalar(state[1]*dag(state[1])))
end

function get_purity(state::MPO,part::Vector{Int64},s::Vector{Index{Int64}})
	A = 1
	N = length(state)
	for l in 1:N
		if l in part
			B = state[l]*state[l]'
			B *= delta(s[l],s[l]'')
			A *= B
		else
			B = state[l]*delta(s[l],s[l]')
			B *= B'
			A *= B
		end
	end
	return real(scalar(A))
end


function get_spectrum(state::MPS,NA::Int64)
        statel = copy(state)
	orthogonalize!(statel,NA)
	if NA>1
		U,spec,V = svd(statel[NA], (linkind(statel, NA-1), siteind(statel,NA)))
	else
		U,spec,V = svd(statel[NA], siteind(statel,NA))
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
		S -= x*log2(x)
	end
	return S
end
