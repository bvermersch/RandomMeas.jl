"""
    get_depolarization_vectors(u:: Vector{Vector{ITensor}},ξ::Vector{Index{Int64}})

TBW
"""
function get_depolarization_vectors(u:: Vector{Vector{ITensor}},ξ::Vector{Index{Int64}})
    N = length(ξ)
    v = siteinds("Qubit", N; addtags="virtual")
    s = siteinds("Qubit", N;addtags="input")

    Nu = length(u)
    c = Vector{MPS}()
    ψ0 = MPS(ξ,["Dn" for n in 1:N]  ) 

    @showprogress dt=1 for r in 1:Nu
        ψu = apply(u[r],ψ0)
        Pu = get_Born_MPS(ψu)

        O = MPO(ξ)
        for i in 1:N
            s0 = state(ξ[i],"Dn")
            s1 = state(ξ[i],"Up")
            O[i] = s0*s0'*onehot(v''[i]=>1)-s1*s1'*onehot(v''[i]=>1)
            O[i] += 2*s1*s1'*onehot(v''[i]=>2)
        end
        Ou = apply(u[r],O;apply_dag=true)
        POu = get_Born_MPS(Ou)

        for i in 1:N
            Pu[i] *= POu[i]*δ(v[i],v[i]'')
        end
        orthogonalize!(Pu,1)
        push!(c,Pu)
    end
    return c
end

"""
    fit_depolarization_vector(M::Union{Vector{MPO},Vector{MPS}},χ::Int64,nsweeps::Int64)

TBW
"""
function fit_depolarization_vector(M::Union{Vector{MPO},Vector{MPS}},χ::Int64,nsweeps::Int64)
    Nu = length(M)
    #ξ = firstsiteinds(M[1];plev=0)
    N = length(M[1])
    σ = truncate(M[1];maxdim=χ)
    orthogonalize!(σ,1)

    L = Array{ITensor}(undef,Nu,N)
    R = Array{ITensor}(undef,Nu,N)
    Ma = Array{ITensor}(undef,Nu,N)
    for r in 1:Nu
        Ma[r,:] = M[r].data
    end
    #init the right environments
    for r in 1:Nu
        X = 1.
        for j in N:-1:2
            X *= Ma[r,j]*dag(σ[j])
            R[r,j] = X
        end
    end
    #first overlap
    @showprogress dt=1  for sw in 1:nsweeps
        dist2 = real(inner(σ,σ))
        for m in M
            dist2 -= real(inner(m,σ))/Nu
            dist2 -= real(inner(σ,m))/Nu
        end
        println("Cost function ",dist2)
        #println("overlap ",real(sum([inner(m,σ) for m in M]))/Nu/norm(σ)^2)
        #left part of the sweep
        for i in 1:N
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
    println("Cost function ",dist2)
    return σ
end

#useful function for get_inverse_depolarization_vector
function inner_vec(A::Vector{ITensor},B::Vector{ITensor})
    N = length(A)
    X = 1
    for i in 1:N
        X *= A[i]*prime(dag(B[i]);tags="Link")
    end
    return real(X[])
end

#useful function for get_inverse_depolarization_vector
function or_product(A::Vector{ITensor},B::Vector{ITensor},or_op::Vector{ITensor})
    N = length(A)
    return  map((i) -> A[i]'*B[i]''*or_op[i], range(1,N)) 
end


"""
    get_inverse_depolarization_vector(c::MPS,χ::Int,sweeps::Int)

TBW
"""
function get_inverse_depolarization_vector(c::MPS,χ::Int,nsweeps::Int)
    v = siteinds(c)
    N = length(c)
    c_ = c.data
    d = randomMPS(Float64,v;linkdims=χ)
    optimizer = LBFGS(; maxiter=100, verbosity=0, gradtol = 1e-6)

    e = ITensor[]
    or_op = ITensor[]
    for i in 1:N
        push!(e,onehot(v[i]=>1))
        X = onehot(v[i]=>1)*onehot(v[i]'=>1)*onehot(v[i]''=>1)
        X += onehot(v[i]=>2)*onehot(v[i]'=>1)*onehot(v[i]''=>2)
        X += onehot(v[i]=>2)*onehot(v[i]'=>2)*onehot(v[i]''=>1)
        X += onehot(v[i]=>2)*onehot(v[i]'=>2)*onehot(v[i]''=>2)
        push!(or_op,X)
    end
    

    loss(x) = inner_vec(or_product(c_,x,or_op),or_product(c_,x,or_op))-inner_vec(or_product(c_,x,or_op),e)-inner_vec(e,or_product(c_,x,or_op))+inner_vec(e,e)

    for s in 1:nsweeps
        for j in 1:N
                if s÷2==0 #right moving sweep
                    i = j
                else
                    i = N+1-j #left moving
                end
                orthogonalize!(d,j)
                lossi(xi) = loss([d[1:i-1];xi;d[i+1:N]])
                loss_and_grad(xi) = lossi(xi),lossi'(xi)
                #@show i,s,lossi'(d[i])
                di, fs, gs, niter, normgradhistory = optimize(loss_and_grad, d[i], optimizer)
                d[i] = di
        end
        println("sweep ",s, " cost function ", loss(d.data)) 
    end
    return d
end

"""
    apply_inverse_channel(O::MPO,d::MPS)

TBW
"""
function apply_inverse_channel(O::MPO,d::MPS)
    ξ = firstsiteinds(O;plev=0)
    v = siteinds(d)
    N = length(ξ)
    s = siteinds("Qubit", N;addtags="input")
    Of = copy(O)
    for i in 1:N
        #initial state
        Oi = O[i]*δ(ξ[i],s[i])*δ(ξ'[i],s'[i])

        #dissipator
        D = onehot(v[i]=>1)*δ(ξ[i],s[i])*δ(ξ'[i],s'[i])
        D += onehot(v[i]=>2)*δ(ξ[i],ξ'[i])*δ(s'[i],s[i])/2
        D *= d[i]

        #final state
        Of[i] = D*Oi
    end
    return orthogonalize(Of,1)
end