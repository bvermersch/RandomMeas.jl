function flatten(O::Union{MPS,MPO,Vector{ITensor}})
	N = length(O)
	O_f = O[1]
	for i in 2:N
		O_f *= O[i]
	end
	return O_f
end

function trace(ρ::MPO,s::Vector{Index{Int64}})
	NA = size(s,1)
  X = ρ[1]*δ(s[1],s[1]')
	for i in 2:NA
		X *= ρ[i]*δ(s[i],s[i]')
	end
  return X[]
end

function reduce_dm(ρ::MPO,part::Vector{Int64})
	N = length(ρ)
	NA = size(part,1)
  s = firstsiteinds(ρ;plev=0)
	sA = s[part]
	ρA = MPO(sA)
	L = 1
	for i in 1:part[1]-1
		L *= ρ[i]*δ(s[i],s[i]')
	end
	for j in 1:NA
		if j<NA
			imax = part[j+1]-1
		else
			imax = N
		end
		R = 1
		for i in part[j]+1:imax
        R *= ρ[i]*δ(s[i],s[i]')
		end
		ρA[j] = L*ρ[part[j]]*R
		L = 1
	end
	orthogonalize!(ρA,1)
	return ρA,sA
end

function reduce_dm(ρ::MPO,i::Int64,j::Int64)
	N = length(rho)
  s = firstsiteinds(ρ;plev=0)
	sA = s[i:j]
	ρA = MPO(sA)

	for k in i:j
		ρA[k-i+1] = ρ[i]
	end

	L = 1
	for k in 1:i-1
		L *=ρ[k]*δ(s[k],s[k]')
	end
	rhoA[i] *= L
	
	R = 1
	for k in j+1:N
		R *= ρ[k]*δ(s[k],s[k]')
	end
	ρA[j] *= R
	orthogonalize!(ρA,1)
	return ρA,sA
end

function reduce_dm(ψ::MPS,i::Int64,j::Int64)
	N = length(ψ)
	s = siteinds(ψ)
	sA = s[i:j]
	ρA = MPO(sA)
	for l in i:j
		ρA[l-i+1] = ψ'[l]*dag(ψ[l])
	end
	if i>1
		l = commonindex(ψ[i-1],ψ[i])
		ρA[1] *= δ(l,l')
	end
	if j<N
		l = commonindex(ψ[j],ψ[j+1])
		ρA[j-i+1] *= δ(l,l')
	end
	return ρA,sA
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
