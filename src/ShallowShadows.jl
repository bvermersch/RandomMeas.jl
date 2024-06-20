function PostRotator(s::Vector{Index{Int64}},ξ::Vector{Index{Int64}},u::Vector{ITensor})
        N = length(ξ)
        ρe = MPO(ξ)
        for i in 1:N
            ρe[i] = δ(ξ[i],ξ[i]',s[i])
        end
        ## Realizes the post-selected state U^\dag|ket{s}\bra{s}U (for any s)
        ud = reverse([swapprime(dag(ut),0,1) for ut in u])
        return apply(ud,ρe,apply_dag=true)
end

"""
    EvaluateMeasurementChannel(ψ::MPS,u::Vector{Vector{ITensor}})

Evaluate a measurement channel Pu(s)ud|s><s|u using tensor network simulations
"""
function EvaluateMeasurementChannel(ψ::MPS,u::Vector{Vector{ITensor}})
    ξ = siteinds(ψ)
    N = length(ψ)
    s = siteinds("Qubit", N;addtags="input")
    Nu = length(u)
    M = Vector{MPO}()
    @showprogress dt=1 for r in 1:Nu
        Mt = MPO(ξ)
        ψu = apply(u[r],ψ)
        Pu = get_Born_MPS(ψu)
        Pu = truncate(Pu;cutoff=1e-16)
        # println("Pu ", maxlinkdim(Pu) )
        PostState = PostRotator(s,ξ,u[r])
        PostState = truncate(PostState;cutoff=1e-16)
        # println("PS ", maxlinkdim(PostState) )
        for i in 1:N
            Mt[i] = (Pu[i]*δ(ξ[i],s[i]))*PostState[i]
        end
        # println("before ",maxlinkdim(Mt))
        # println(inds(Pu[N÷2];tags="Link"))
        # println(inds(PostState[N÷2];tags="Link"))
        # println(inds(Mt[N÷2];tags="Link"))
        Mt = truncate(Mt;cutoff=1e-16)
        # println("after ",maxlinkdim(Mt))
        push!(M,Mt)
    end
    return M
end

"""
    FitChannelMPO(M::Vector{MPO},χ::Int64,nsweeps::Int64)

FInd the best MPO that represents a measurement channel evaluated by EvaluateMeasurementChannel
"""
function FitChannelMPO(M::Vector{MPO},χ::Int64,nsweeps::Int64)
    Nu = length(M)
    ξ = firstsiteinds(M[1];plev=0)
    N = length(ξ)
    σ = truncate(M[1];maxdim=χ)
    orthogonalize!(σ,1)

    L = Array{ITensor}(undef,Nu,N)
    R = Array{ITensor}(undef,Nu,N)
    Ma = Array{ITensor}(undef,Nu,N)
    for r in 1:Nu
        Ma[r,:] = M[r].data
    end
    R[1,1] = ITensor(ξ[1])
    #init the right environments
    for r in 1:Nu
        X = 1.
        for j in N:-1:2
            X *= Ma[r,j]*dag(σ[j])
            R[r,j] = X
        end
    end
    #first overlap
    for sw in 1:nsweeps
        dist2 = real(inner(σ,σ))
        @showprogress dt=1 for m in M
            dist2 -= real(inner(m,σ))/Nu
            dist2 -= real(inner(σ,m))/Nu
        end
        println("Lower bound to distance ",dist2)
        #println("overlap ",real(sum([inner(m,σ) for m in M]))/Nu/norm(σ)^2)
        #left part of the sweep
        @showprogress dt=1 for i in 1:N
            if i==1
                σ[1] = sum(Ma[:,1].*R[:,2])/Nu
            elseif i<N
                σ[i] = sum(R[:,i+1].*Ma[:,i].*L[:,i-1])/Nu
            else
                σ[N] = sum(L[:,i-1].*Ma[:,i])/Nu
            end
            if i<N
                bindex = commonind(σ[i],σ[i+1])
                orthogonalize!(σ,i+1)
                bindex2 = commonind(σ[i],σ[i+1])
                replaceind!(σ[i],bindex2,bindex)
                replaceind!(σ[i+1],bindex2,bindex)
            end
            #updating the left environments
            if i==1
                L[:,1] = [Ma[r,1]*dag(σ[1]) for r in 1:Nu]
            elseif i<=N
                L[:,i] = [L[r,i-1]*Ma[r,i]*dag(σ[i]) for r in 1:Nu]
            end
            #println("right norm ",norm(flatten(σ)-Mf))
        end
        #right part of the sweep
        @showprogress dt=1 for i in N:-1:1
            if i==1
                σ[1] = sum(Ma[:,1].*R[:,2])/Nu
            elseif i<N
                σ[i] = sum(R[:,i+1].*Ma[:,i].*L[:,i-1])/Nu
            else
                σ[N] = sum(L[:,i-1].*Ma[:,i])/Nu
            end
            if i>1
                bindex = commonind(σ[i],σ[i-1])
                orthogonalize!(σ,i-1)
                bindex2 = commonind(σ[i],σ[i-1])
                replaceind!(σ[i],bindex2,bindex)
                replaceind!(σ[i-1],bindex2,bindex)
            end
            #updating the right environments
            if i==N
                R[:,N] = [Ma[r,N]*dag(σ[N]) for r in 1:Nu]
            elseif i>1
                R[:,i] = [R[r,i+1]*Ma[r,i]*dag(σ[i]) for r in 1:Nu]
            end
            #println("left norm ",norm(flatten(σ)-Mf))
        end
    end
    dist2 = real(inner(σ,σ))
    @showprogress dt=1 for m in M
           dist2 -= real(inner(σ,m))/Nu
           dist2 -= real(inner(m,σ))/Nu
    end
    println("Lower bound to distance ",dist2)
    return σ
end

function Dissipators(ξ::Vector{Index{Int64}},s::Vector{Index{Int64}},v::Vector{Index{Int64}})
    N = length(ξ)
    D = Vector{ITensor}(undef,N)
    for i in 1:N
        D[i] = onehot(v[i]=>1)*δ(ξ[i],s[i])*δ(ξ'[i],s'[i])
        D[i] += onehot(v[i]=>2)*δ(ξ[i],ξ'[i])*δ(s'[i],s[i])/2
    end
    return D
end


function Cost_InversionChannel(c::Vector{ITensor},ρ0::MPO,Dσ0::Vector{ITensor})
    N = length(c)
    X1 = 1
    for i in 1:N
        Y = Dσ0[i]*c[i]
        X1 *= Y*prime(dag(Y),tags="Link")
    end
    X2a = 1
    for i in 1:N
        Y = Dσ0[i]*c[i]
        X2a *= Y*prime(dag(ρ0[i]),tags="Link")
    end
    X2b = 1
    for i in 1:N
        Y = Dσ0[i]*c[i]
        X2b *= dag(Y)*prime(ρ0[i],tags="Link")
    end
    X3 = 1
    for i in 1:N
        Y = ρ0[i]
        X3*= Y*prime(dag(ρ0[i]),tags="Link")
    end
    return real(X1[]-X2a[]-X2b[]+X3[])
end

"""
    InversionChannel(ρ0::MPO,σ0::MPO,χ ::Int64,nsweeps::Int64)

From a measured postselected state σ0, and an initial state, find the inverse channel that maps
back σ0 to ρ0. χ is the bond dimension of the MPS that parametrizes the quantum channel.

"""
function InversionChannel(ρ0::MPO,σ0::MPO,χ ::Int64,nsweeps::Int64)
    N = length(ρ0)
    ξ = firstsiteinds(ρ0;plev=0)
    s = siteinds("Qubit", N;addtags="input")
    v = siteinds("Qubit", N;addtags="virtual")

    σ0t = MPO(s)
    for i in 1:N
        σ0t[i] = σ0[i]*δ(ξ[i],s[i])*δ(ξ[i]',s[i]')
    end
    D = Dissipators(ξ,s,v)
    Dσ0 = [D[i]*σ0t[i] for i in 1:N]
    dt = randomMPS(Float64,v,;linkdims=χ)
    loss(d) = Cost_InversionChannel(d,ρ0,Dσ0)
    optimizer = LBFGS(; maxiter=200, verbosity=0, gradtol = 1e-5)
    for s in 1:nsweeps
        for i in 1:N
            orthogonalize!(dt,i)
            lossi(di) = loss([dt[1:i-1];di;dt[i+1:N]])
            loss_and_grad(x) = lossi(x),lossi'(x)
            di, fs, gs, niter, normgradhistory = optimize(loss_and_grad, dt[i], optimizer)
            dt[i] = di
        end
        println("sweep ",s, " loss ", loss(dt.data))
    end
    #loss_and_grad(x) = loss(x),loss'(x)
    return dt
end


"""
    get_ShallowShadow(data::Array{Int8},u::Vector{ITensor},d::MPS,ξ::Vector{Index{Int64}})

Build shadows from measured data under shallow unitaries d and learnt channel parametrized by the MPS
d
"""
function get_ShallowShadow(data::Array{Int},u::Vector{ITensor},d::MPS,ξ::Vector{Index{Int64}})
    NM,N = size(data)
    x = [firstind(d[i],tags="Site") for i in 1:N]
    s = siteinds("Qubit", N;addtags="input")
    PostState = PostRotator(s,ξ,u)
    D = Dissipators(s,ξ,x)
    Dd = [D[i]*d[i] for i in 1:N]
    shadow = Vector{MPO}()
    for m in 1:NM
        push!(shadow,MPO(ξ))
        for i in 1:N
            shadow[m][i] = PostState[i]*onehot(s[i]=>data[m,i])
            shadow[m][i] *= Dd[i]
            replaceind!(shadow[m][i],s[i],ξ[i])
            replaceind!(shadow[m][i],s[i]',ξ[i]')
        end
    end
    return shadow
end
