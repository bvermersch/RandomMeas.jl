"""
    get_shallow_depolarization_mps(group::MeasurementGroup{ShallowUnitaryMeasurementSetting})

TBW
"""
function get_shallow_depolarization_mps(group::MeasurementGroup{ShallowUnitaryMeasurementSetting})
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

"""
    get_depolarization_map(depolarization_mps::MPS,s::Vector{Index{Int64}},ξ::Vector{Index{Int64}})

returns a shallow map \\mathcal{M} parametrization by a depolarization_mps c(\\nu)
 where the state is depolarized over partition \\A_{
u} with probability c(\\nu)=1
"""
function get_depolarization_map(depolarization_mps_data::Vector{ITensor},v::Vector{Index{Int64}},s::Vector{Index{Int64}},ξ::Vector{Index{Int64}})
    N = length(depolarization_mps_data)
    depolarization_op(vi,si,ξi) = onehot(vi=>1)*δ(ξi,si)*δ(ξi',si')+onehot(vi=>2)*δ(ξi,ξi')*δ(si',si)/2
    depolarization_map = [depolarization_op(v[i],s[i],ξ[i])*depolarization_mps_data[i] for i in 1:N]
    return depolarization_map
end

"""
    get_depolarization_map(depolarization_mps::MPS,s::Vector{Index{Int64}},ξ::Vector{Index{Int64}})

returns a shallow map \\mathcal{M} parametrization by a depolarization_mps c(\\nu)
 where the state is depolarized over partition \\A_{
u} with probability c(\\nu)=1
"""
function get_depolarization_map(depolarization_mps::MPS,s::Vector{Index{Int64}},ξ::Vector{Index{Int64}})
    return get_depolarization_map(depolarization_mps.data,siteinds(depolarization_mps),s,ξ)
end

#Computes the square norm 2 between the data of the two MPS/MPO (similar to norm(A-B) in ITensor)
function norm2_vec(A::Vector{ITensor},B::Vector{ITensor})
    term = inner_vec(A,A)
    term += inner_vec(B,B)
    term -= inner_vec(B,A)
    term -= inner_vec(A,B)
    return real(term)
end

#Computes the inner product between the data of the two MPS/MPO (similar to inner of ITensor)
function inner_vec(A::Vector{ITensor},B::Vector{ITensor})
    N = length(A)
    X = 1
    for i in 1:N
        X *= A[i]*prime(dag(B[i]);tags="Link")
    end
    return X[]
end

function loss_inverse_depolarization_map(inverse_depolarization_mps_data::Vector{ITensor},depolarization_map::Vector{ITensor},v::Vector{Index{Int64}},s::Vector{Index{Int64}},ξ::Vector{Index{Int64}})    
    N = length(inverse_depolarization_mps_data)
    η = siteinds("Qubit", N)  # Output Site indices
    inverse_depolarization_map = get_depolarization_map(inverse_depolarization_mps_data,v,ξ,η)
    combined_map = [depolarization_map[i]*inverse_depolarization_map[i] for i in 1:N]
    identity_map = [δ(s[i],η[i])*δ(s'[i],η'[i]) for i in 1:N]
    return norm2_vec(combined_map,identity_map)
end

# Constructor for ShallowSShadow from raw measurement results and unitaries
"""
    ShallowShadow(measurement_results::Vector{Int}, local_unitary::Vector{ITensor};
                     G::Vector{Float64} = fill(1.0, length(local_unitary)))

Construct a `ShallowSShadow` object from raw measurement results and unitary transformations.

# Arguments
- `measurement_results::Vector{Int}`: Vector of binary measurement results for each qubit/site.
- `local_unitary::Vector{ITensor}`: Vector of local unitary transformations applied during the measurement.

# Returns
A `ShallowShadow` object.
"""
function ShallowShadow(measurement_results::Vector{Int}, local_unitary::Vector{ITensor}, inverse_shallow_map::Vector{ITensor},s::Vector{Index{Int64}},ξ::Vector{Index{Int64}})
    # Number of qubits/sites

    N = length(local_unitary)

    local_unitary_dag = reverse([swapinds(dag(local_unitary[i]),ξ[i],ξ[i]') for i in 1:N])

    # Construct the factorized shadow for each qubit/site
    shadow_data = Vector{ITensor}(undef, N)

    states = [measurement_results[i]==2 ? "Dn" : "Up" for i in 1:N]
    ψ0  = MPS(ComplexF64,ξ,states);
    ψ = apply(local_unitary_dag,ψ0)
    replace_siteinds!(ψ,s)
    ρ = outer(ψ',ψ)
    shadow_data = MPO([inverse_shallow_map[i]*ρ[i] for i in 1:N])
    return ShallowShadow(shadow_data, N, ξ)
end

"""
    get_shallow_shadows(measurement_data::MeasurementData{ShallowUnitaryMeasurementSetting}, inverse_shallow_map::Vector{ITensor},s::Vector{Index{Int64}},ξ::Vector{Index{Int64}})

Construct a `ShallowSShadow` object from MeasurementData

# Arguments


# Returns
A `ShallowShadow` object.
"""
function get_shallow_shadows(measurement_data::MeasurementData{ShallowUnitaryMeasurementSetting}, inverse_shallow_map::Vector{ITensor},s::Vector{Index{Int64}},ξ::Vector{Index{Int64}})
    # Number of qubits/sites

    N = measurement_data.N
    setting = measurement_data.measurement_setting

    # Extract site indices from local unitaries
    @assert ξ == setting.site_indices
    local_unitary = setting.local_unitary
    measurement_results = measurement_data.measurement_results
    NM = measurement_data.NM

    return [ShallowShadow(measurement_results[m,:], local_unitary, inverse_shallow_map,s,ξ) for m in 1:NM]
end