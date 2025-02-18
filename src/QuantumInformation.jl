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
    get_siteinds(ψ::Union{MPS,MPO})

TBW
"""
function get_siteinds(ψ::Union{MPS,MPO})
    if typeof(ψ) == MPS
        ξ = siteinds(ψ)
    else
        ξ = firstsiteinds(ψ;plev=0)
        return ξ
    end
end

"""
    get_trace(ρ::MPO,s::Vector{Index{Int64}})
"""
function get_trace(ρ::MPO)
    s = firstsiteinds(ρ;plev=0)
	NA = size(s,1)
        X = ρ[1]*δ(s[1],s[1]')
	for i in 2:NA
		X *= ρ[i]*δ(s[i],s[i]')
	end
  return X[]
end

"""
    reduce_to_subsystem(ρ::MPO,subsystem::Vector{Int64})

compute the reduce density matrix over sites mentionned in part
"""

function reduce_to_subsystem(ρ::MPO,subsystem::Vector{Int64})
	N = length(ρ)
	NA = size(subsystem,1)
    s = firstsiteinds(ρ;plev=0)
	sA = s[subsystem]
	ρA = MPO(sA)
	L = 1
	for i in 1:subsystem[1]-1
		L *= ρ[i]*δ(s[i],s[i]')
	end

	for j in 1:NA
		if j<NA
			imax = subsystem[j+1]-1
		else
			imax = N
		end
		R = 1
		for i in subsystem[j]+1:imax
       		 R *= ρ[i]*δ(s[i],s[i]')
		end
		if j==1
			ρA[1] = L*ρ[subsystem[1]]*R
		else
			ρA[j] =  ρ[subsystem[j]]*R
		end
	end
	return ρA
end

function reduce_to_subsystem(ψ::MPS,subsystem::Vector{Int64})
	return reduce_to_subsystem(outer(ψ',ψ),subsystem)
end

"""
    get_entanglement_spectrum(ψ::MPS,NA::Int64)
    Compute entanglement spectrum of the first NA sites density matrix
"""
function get_entanglement_spectrum(ψ::MPS,NA::Int64)
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
    get_trace_moment(spec::ITensor,k::Int)

compute the moments ``\\mathrm{tr}(\\rho)'' from the entanglement spectrum
"""
function get_trace_moment(spec::ITensor, k::Int)
        # Check if any entry is smaller than 1 or larger than n_shadows
        @assert k .>= 1 "Only integer valued moments Tr[rho^k] with k >=1 can be computed."

        p= Vector{Float64}()

        pk = 0
        for l in 1:dim(spec, 1)
            pk += spec[l, l]^(2 * k)
        end
        return pk
end


"""
    partial_transpose(ρ::MPO,subsystem::Vector{Int})

TBW
"""
function partial_transpose(ρ::MPO,subsystem::Vector{Int})
    ξ = firstsiteinds(ρ;plev=0)
    ρT = MPO(ξ)
    for i in 1:length(ξ)
        if i in subsystem
            ρT[i]  = swapind(ρ[i],ξ[i],ξ[i]')
        else
            ρT[i] = 1. * ρ[i] 
        end
    end
    return ρT
end


"""
    get_trace_moment(ψ::Union{MPS,MPO},k::Int,subsystem::Vector{Int}=collect(1:length(ψ)),partial_transpose::Bool=false)

TBW
"""
function get_trace_moment(ψ::Union{MPS,MPO},k::Int,subsystem::Vector{Int}=collect(1:length(ψ)))
    if diff(subsystem) == ones(Int,length(subsystem)-1) && subsystem[1]==1 && typeof(ψ)==MPS
        spec = get_entanglement_spectrum(ψ,subsystem[end])
        return get_trace_moment(spec,k)
    else
        ρ = reduce_to_subsystem(ψ,subsystem)
        if k==2 #In the purity case k=2, tr(rho^2) = inner product of the mpo <ρ|ρ>
            return norm(ρ)^2
        else
            ρk = 1. * ρ
            for _ in 2:k
                ρk = apply(ρk,ρ;cutoff=1e-12)
            end
            return get_trace(ρk)
        end
    end
end

"""
    get_trace_moments(ψ::Union{MPS,MPO},k_vector::Vector{Int},subsystem::Vector{Int}=collect(1:length(ψ)))

TBW
"""
function get_trace_moments(ψ::Union{MPS,MPO},k_vector::Vector{Int},subsystem::Vector{Int}=collect(1:length(ψ)))
    return [get_trace_moment(ψ,k,subsystem) for k in k_vector]
end

"""
    get_Born_MPS(ρ::MPO)

Construct Born Probability vector P(s)=<s|ρ|s> as an MPS from an MPO representation ρ
"""
function get_Born_MPS(ρ::MPO)
    ξ = firstsiteinds(ρ;plev=0)
    N = size(ξ, 1)
    P = MPS(ξ)
    for i in 1:N
        Ct = δ(ξ[i], ξ[i]', ξ[i]'')
        P[i] = ρ[i] * Ct
        P[i] *= δ(ξ[i], ξ[i]'')
    end
    return P
end

"""
    get_Born_MPS(ψ::MPS)

Construct Born Probability vector P(s)=|ψ(s)|^2 as an MPS from an MPS representation ψ
"""
function get_Born_MPS(ψ::MPS)
    ξ = siteinds(ψ)
    N = size(ξ, 1)
    P = MPS(ξ)
    for i in 1:N
        Ct = δ(ξ[i], ξ[i]', ξ[i]'')
        P[i] = ψ[i] * conj(ψ[i]') * Ct
        P[i] *= δ(ξ[i], ξ[i]'')
    end
    return P
end


"""
    get_selfXEB(ψ::MPS)

Returns the self-XEB 2^N sum_s |ψ(s)|^4-1
"""
function get_selfXEB(ψ::MPS)
    P0 = get_Born_MPS(ψ)
    N = length(ψ)
    return 2^N*real(inner(P0,P0))-1
end

"""
    get_average_MPS(ψ_list::Vector{MPS},χ::Int64,nsweeps::Int64)

TBW
"""
function get_average_mps(ψ_list::Vector{MPS},χ::Int64,nsweeps::Int64)
    NU = length(ψ_list)
    N = length(ψ_list[1])

    #ψ = truncate(ψ_list[1];maxdim=χ)
    #orthogonalize!(ψ,1)


    ψ = random_mps(siteinds(ψ_list[1]); linkdims=χ);

    L = Array{ITensor}(undef,NU,N)
    R = Array{ITensor}(undef,NU,N)
    Ma = Array{ITensor}(undef,NU,N)
    for r in 1:NU
        Ma[r,:] = ψ_list[r].data
    end
    #init the right environments
    for r in 1:NU
        X = 1.
        for j in N:-1:2
            X *= Ma[r,j]*dag(ψ[j])
            R[r,j] = X
        end
    end
    #first overlap
    @showprogress dt=1  for sw in 1:nsweeps
        dist2 = real(inner(ψ,ψ))
        for ψ_r in ψ_list
            dist2 -= real(inner(ψ_r,ψ))/NU
            dist2 -= real(inner(ψ,ψ_r))/NU
        end
        println("Cost function ",dist2)
        #println("overlap ",real(sum([inner(m,ψ) for m in M]))/Nu/norm(ψ)^2)
        #left part of the sweep
        for i in 1:N
            if i==1
                ψ[1] = sum(Ma[:,1].*R[:,2])/NU
            elseif i<N
                ψ[i] = sum(R[:,i+1].*Ma[:,i].*L[:,i-1])/NU
            else
                ψ[N] = sum(L[:,i-1].*Ma[:,i])/NU
            end
            if i<N
                #bindex = commonind(ψ[i],ψ[i+1])
                orthogonalize!(ψ,i+1)
                #bindex2 = commonind(ψ[i],ψ[i+1])
                #replaceind!(ψ[i],bindex2,bindex)
                #replaceind!(ψ[i+1],bindex2,bindex)
            end
            #updating the left environments
            if i==1
                L[:,1] = [Ma[r,1]*dag(ψ[1]) for r in 1:NU]
            elseif i<=N
                L[:,i] = [L[r,i-1]*Ma[r,i]*dag(ψ[i]) for r in 1:NU]
            end
            #println("right norm ",norm(flatten(ψ)-Mf))
        end
        #right part of the sweep
        @showprogress dt=1 for i in N:-1:1
            if i==1
                ψ[1] = sum(Ma[:,1].*R[:,2])/NU
            elseif i<N
                ψ[i] = sum(R[:,i+1].*Ma[:,i].*L[:,i-1])/NU
            else
                ψ[N] = sum(L[:,i-1].*Ma[:,i])/NU
            end
            if i>1
                #bindex = commonind(ψ[i],ψ[i-1])
                orthogonalize!(ψ,i-1)
                #bindex2 = commonind(ψ[i],ψ[i-1])
                #replaceind!(ψ[i],bindex2,bindex)
                #replaceind!(ψ[i-1],bindex2,bindex)
            end
            #updating the right environments
            if i==N
                R[:,N] = [Ma[r,N]*dag(ψ[N]) for r in 1:NU]
            elseif i>1
                R[:,i] = [R[r,i+1]*Ma[r,i]*dag(ψ[i]) for r in 1:NU]
            end
            #println("left norm ",norm(flatten(ψ)-Mf))
        end
    end
    dist2 = real(inner(ψ,ψ))
    @showprogress dt=1 for ψ_r in ψ_list
           dist2 -= real(inner(ψ,ψ_r))/NU
           dist2 -= real(inner(ψ_r,ψ))/NU
    end
    println("Cost function ",dist2)
    return ψ
end
