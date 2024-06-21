"""
    flatten(O::Union{MPS,MPO,Vector{ITensor}})

convert a MPS/MPO to ITensor by a brutal multiplication A[1] x A[2] x ...
"""
function flatten(O::Union{MPS,MPO,Vector{ITensor}})
	N = length(O)
	O_f = O[1]
	for i in 2:N
		O_f *= O[i]
	end
	return O_f
end

"""
    trace(ρ::MPO,s::Vector{Index{Int64}})
"""
function trace(ρ::MPO,s::Vector{Index{Int64}})
	NA = size(s,1)
  X = ρ[1]*δ(s[1],s[1]')
	for i in 2:NA
		X *= ρ[i]*δ(s[i],s[i]')
	end
  return X[]
end

"""
    reduce_dm(ρ::MPO,part::Vector{Int64})

compute the reduce density matrix over sites mentionned in part
"""
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
		if j==1
			ρA[1] = L*ρ[part[1]]*R
		else
			ρA[j] =  ρ[part[j]]*R
		end
	end
	#orthogonalize!(ρA,1)
	return ρA,sA
end

# """
#     reduce_dm(ρ::MPO,i::Int64,j::Int64)

# compute the reduced density matrix for sites i:j

# """
# function reduce_dm(ρ::MPO,i::Int64,j::Int64)
# 	N = length(ρ)
#   	s = firstsiteinds(ρ;plev=0)
# 	sA = s[i:j]
# 	ρA = MPO(sA)

# 	for k in i:j
# 		ρA[k-i+1] = ρ[i]
# 	end

# 	L = 1
# 	for k in 1:i-1
# 		L *=ρ[k]*δ(s[k],s[k]')
# 	end
# 	ρA[i] *= L
	
# 	R = 1
# 	for k in j+1:N
# 		R *= ρ[k]*δ(s[k],s[k]')
# 	end
# 	ρA[j] *= R
# 	orthogonalize!(ρA,1)
# 	return ρA,sA
# end


"""
    reduce_dm(ψ::MPS,i::Int64,j::Int64)

compute the reduced density matrix for sites i:j

"""
function reduce_dm(ψ::MPS,i::Int64,j::Int64)
	N = length(ψ)
	s = siteinds(ψ)
	sA = s[i:j]
	ρA = MPO(sA)
	for l in i:j
		ρA[l-i+1] = ψ'[l]*dag(ψ[l])
	end
	if i>1
		l = commonind(ψ[i-1],ψ[i])
		ρA[1] *= δ(l,l')
	end
	if j<N
		l = commonind(ψ[j],ψ[j+1])
		ρA[j-i+1] *= δ(l,l')
	end
	return ρA,sA
end

"""
    get_purity(ψ::MPS,NA::Int64)

compute the purity of an MPS over the first NA sites
"""
function get_purity(ψ::MPS,NA::Int64)
	N = length(state)
	if NA<N
		spec = get_spectrum(ψ,NA)
		p = get_moment(spec,2)[1]
	else
		p = 1
	end
	return p
end

"""
    get_purity(ψ::MPS,i::Int64,j::Int64)

compute the purity over sites i:j
"""
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

"""
    get_purity(ρ::MPO)

compute the purity of an MPO
"""
function get_purity(ρ::MPO)
	orthogonalize!(ρ,1)
	return real(scalar(ρ[1]*dag(ρ[1])))
end

"""
    get_purity(ρ::MPO,part::Vector{Int64},ξ::Vector{Index{Int64}})

Compute the purity over the sites of part
"""
function get_purity(ρ::MPO,part::Vector{Int64},ξ::Vector{Index{Int64}})
ξ = firstsiteinds(ρ;plev=0)
	A = 1
	N = length(ρ)
	for l in 1:N
		if l in part
			B = ρ[l]*ρ[l]'
			B *= δ(ξ[l],ξ[l]'')
			A *= B
		else
			B = ρ[l]*δ(ξ[l],ξ[l]')
			B *= B'
			A *= B
		end
	end
	return real(scalar(A))
end

"""
    get_spectrum(ψ::MPS,NA::Int64)
    Compute entanglement spectrum of the first NA sites density matrix
"""
function get_spectrum(ψ::MPS,NA::Int64)
        statel = copy(ψ)
	orthogonalize!(statel,NA)
	if NA>1
		U,spec,V = svd(statel[NA], (linkind(statel, NA-1), siteind(statel,NA)))
	else
		U,spec,V = svd(statel[NA], siteind(statel,NA))
	end
	return spec
end

"""
    get_moment(spec::ITensor,n::Int)

compute the moments ``\\mathrm{tr}(\\rho)'' from the entanglement spectrum
"""
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

"""
    get_entropy(spec::ITensor)

compute von Neumann entropy from entanglement spectrum
"""
function get_entropy(spec::ITensor)
	S = 0
	for l=1:dim(spec, 1)
		x = spec[l,l]^2
		S -= x*log2(x)
	end
	return S
end
