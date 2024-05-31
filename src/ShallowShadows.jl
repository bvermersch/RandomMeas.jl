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

function EvaluateMeasurementChannel(ψ::MPS,u::Vector{Vector{ITensor}},χ::Int64)
    ξ = siteinds(ψ)
    N = length(ψ)
    s = siteinds("Qubit", N;addtags="input")
    Nu = length(u)
    M = Vector{MPO}()
    @showprogress dt=1 for r in 1:Nu
        Mt = MPO(ξ)
        ψu = apply(u[r],ψ)
        Pu = get_Born_MPS(ψu)
        truncate!(Pu;cutoff=1e-6)
        PostState = PostRotator(s,ξ,u[r])
        for i in 1:N
            Mt[i] = (Pu[i]*δ(ξ[i],s[i]))*PostState[i]
        end
        push!(M,truncate(Mt;cutoff=1e-16))
    end
    return M
end

function FitChannelMPO(M::Vector{MPO},χ::Int64,nsweeps::Int64)
    Nu = length(M)
    ξ = firstsiteinds(M[1];plev=0)
    N = length(ξ)
    #ψ = random_mps(ξ,1)
    #σ = outer(ψ',ψ)
    #for t in 2:χ
    #    ψ = random_mps(ξ,1)
    #    σ += outer(ψ',ψ)
    #end
    σ = truncate(M[1];maxdim=χ)
    orthogonalize!(σ,1)

    L = Array{ITensor}(undef,Nu,N)
    R = Array{ITensor}(undef,Nu,N)
    Ma = Array{ITensor}(undef,Nu,N)
    for r in 1:Nu
        Ma[r,:] = M[r].data
    end
    R[1,1] = ITensor(ξ[1])
    #Mf = sum([flatten(m) for m in M])/Nu
    #println("first norm ",norm(flatten(σ)-Mf))
    #init the right environments
    for r in 1:Nu
        X = 1.
        for j in N:-1:2
            X *= Ma[r,j]*dag(σ[j])
            R[r,j] = X
        end
    end
    println("env done")
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

function get_c_local(v::Vector{Index{Int64}},χ::Int64,p::Float64)
    c0 = Vector{ITensor}()
    N = length(v)
    for i in 1:N
        push!(c0,p*onehot(v[i]=>1)+(1-p)*onehot(v[i]=>2))
    end
    if χ>1
        c0 = MPS(c0)+0*randomMPS(Float64,v;linkdims=χ-1)
        c0 = c0.data
    end
    return c0
end

#function Cost_c(c::Vector{ITensor},Dρ2::Vector{ITensor},DρM::Vector{Vector{ITensor}})
#    N = length(c)
#    X1 = 1
#    for i in 1:N
#        X1 *= Dρ2[i]*c[i]*c[i]'
#    end
#    X2 = 0
#    Nu = length(DρM)
#    for r in 1:Nu
#        Xt = 1
#        for i in 1:N
#            Xt *= DρM[r][i]*c[i]
#        end
#        X2 += Xt[]/Nu
#        end
#    return real(X1[]-2*X2)
#end
function Cost_c(c::Vector{ITensor},Dρ::Vector{ITensor},M::MPO)
    N = length(c)
    X1 = 1
    for i in 1:N
        Y = Dρ[i]*c[i]
        X1 *= Y*prime(Y,tags="Link")
    end
    X2 = 1
    for i in 1:N
        Y = Dρ[i]*c[i]
        X2 *= Y*prime(M[i],tags="Link")
    end
    X3 = 1
    for i in 1:N
        Y = M[i]
        X3*= Y*prime(M[i],tags="Link")
    end
    return real(X1[]-2*X2[]+X3[])
end

function Find_c(ψ0::MPS,M::MPO,χ ::Int64)
#function Find_c(ψ0::MPS,M::Vector{Vector{ITensor}},χ ::Int64)
    N = length(ψ0)
    Nu = length(M)
    ξ = siteinds(ψ0)
    s = siteinds("Qubit", N;addtags="input")
    v = siteinds("Qubit", N;addtags="virtual")
    ψ0t = replace_siteinds(ψ0,s)
    ρ0 = outer(ψ0t',ψ0t)
    D = Dissipators(ξ,s,v)
    Dρ0 = [D[i]*ρ0[i] for i in 1:N]
    #Dρ02 = [Dρ0[i]*(δ(v[i],v[i]')*prime(Dρ0[i],tags="Link")) for i in 1:N]
    #Dρ0M = [Dρ0[i]*prime(M[i],tags="Link") for i in 1:N]
    #Dρ0M = [ [Dρ0[i]*prime(M[r][i],tags="Link") for i in 1:N] for r in 1:Nu]
    #loss(c) = Cost(c,ρ0,D,M)
    #c0 = guess_c(v,χ,1/3)
    ci = randomMPS(Float64,v,;linkdims=χ).data
    loss(c) = Cost_c(c,Dρ0,M)
    #using Zygote
    optimizer = LBFGS(; maxiter=100, verbosity=1, gradtol = 1e-4)
    loss_and_grad(x) = loss(x),loss'(x)
    c, fs, gs, niter, normgradhistory = optimize(loss_and_grad, ci, optimizer)
    return c
end

function Cost_d(d::Vector{ITensor},c::Vector{ITensor},D::Vector{ITensor},D2::Vector{ITensor},s::Vector{Index{Int64}},η::Vector{Index{Int64}})
    N = length(c)
    X1 = 1
    for i in 1:N
        Y = D[i]*c[i]*D2[i]*d[i]
        X1 *= Y*prime(Y,tags="Link")
    end
    X2 = 1
    for i in 1:N
        Y = D[i]*c[i]*D2[i]*d[i]
        X2 *= Y*δ(s[i],η[i])*δ(s[i]',η[i]')
    end
    #||1||_2^2 = 4^N 
    X3 = 4^N
    return real(X1[]-2*X2[]+X3)
end

function Find_d(c::Vector{ITensor},χ::Int64)
    N = length(c)
    v = [firstind(c[i],tags="Site") for i in 1:N]
    ξ = siteinds("Qubit", N;addtags="output")
    η = siteinds("Qubit", N;addtags="output")
    s = siteinds("Qubit", N;addtags="input")
    x = siteinds("Qubit", N;addtags="virtual")
    D = Dissipators(ξ,s,v)
    D2 = Dissipators(η,ξ,x)
    loss(d) = Cost_d(d,c,D,D2,s,η)
    loss_and_grad(x) = loss(x),loss'(x)
    #optimizer = ConjugateGradient(; maxiter=500, verbosity=1, gradtol = 1e-4)
    optimizer = LBFGS(; maxiter=5000, verbosity=1, gradtol = 1e-4)
    di = randomMPS(Float64,x,;linkdims=χ ).data
    println("random guess",loss(di))
    di = MPS(v,["Up" for n in 1:N])
    di = add(2*di,-MPS(c);maxdim=χ)
    replace_siteinds!(di,x)
    di = di.data
    println("taylor guess",loss(di))
    #di = get_c_local(x,χ,3.)
    #d, fs, gs, niter, normgradhistory = optimize(loss_and_grad, di, optimizer);
    #di = add(MPS(d),randomMPS(Float64,x,;linkdims=χ-1)).data
    d, fs, gs, niter, normgradhistory = optimize(loss_and_grad, di, optimizer);
    return d
end

function get_ShallowShadow(data::Array{Int8},u::Vector{ITensor},d::Vector{ITensor},ξ::Vector{Index{Int64}})
    N = length(d)
    #ξ = [firstind(u[i];plev=0) for i in 1:N]
    x = [firstind(d[i],tags="Site") for i in 1:N]
    s = siteinds("Qubit", N;addtags="input")
    PostState = PostRotator(s,ξ,u)
    D = Dissipators(s,ξ,x)
    shadow = MPO(ξ)
    for i in 1:N
        shadow[i] = PostState[i]*onehot(s[i]=>data[i])
        shadow[i] *= d[i]*D[i]
        replaceind!(shadow[i],s[i],ξ[i])
        replaceind!(shadow[i],s[i]',ξ[i]')
    end
    return shadow
end

