"""
    get_depolarization_vectors(group::MeasurementGroup{ShallowUnitaryMeasurementSetting})

TBW
"""
function get_depolarization_vectors(group::MeasurementGroup{ShallowUnitaryMeasurementSetting})
    N = group.N
    ξ = group.measurements[1].measurement_setting.site_indices

    v = siteinds("Qubit", N; addtags="virtual")
    #s = siteinds("Qubit", N;addtags="input")

    NU = group.NU
    depolarization_vectors = Vector{MPS}()
    ψ0 = MPS(ξ,["Dn" for n in 1:N]  ) 

    @showprogress dt=1 for r in 1:NU
        local_unitary = group.measurements[r].measurement_setting.local_unitary
        ψu = apply(local_unitary,ψ0)
        Pu = get_Born_MPS(ψu)

        O = MPO(ξ)
        for i in 1:N
            s0 = state(ξ[i],"Dn")
            s1 = state(ξ[i],"Up")
            O[i] = s0*s0'*onehot(v''[i]=>1)-s1*s1'*onehot(v''[i]=>1)
            O[i] += 2*s1*s1'*onehot(v''[i]=>2)
        end
        Ou = apply(local_unitary,O;apply_dag=true)
        POu = get_Born_MPS(Ou)

        for i in 1:N
            Pu[i] *= POu[i]*δ(v[i],v[i]'')
        end
        orthogonalize!(Pu,1)
        push!(depolarization_vectors,Pu)
    end
    return depolarization_vectors
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
        println("sweep ",s, " Cost function ", loss(d.data)) 
    end
    return d
end

"""
    apply_inverse_channel(O::MPO,d::MPS)

something
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