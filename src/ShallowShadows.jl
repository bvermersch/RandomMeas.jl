# Copyright (c) 2024 Benoît Vermersch and Andreas Elben
# SPDX-License-Identifier: Apache-2.0
# http://www.apache.org/licenses/LICENSE-2.0

"""
    get_shallow_depolarization_mps(group::MeasurementGroup{ShallowUnitaryMeasurementSetting})

TBW
"""
function get_shallow_depolarization_mps(settings::Vector{ShallowUnitaryMeasurementSetting})
    NU = length(settings)
    ξ = settings[1].site_indices
    N = settings[1].N

    v = siteinds("Qubit", N; addtags="virtual")

    depolarization_vectors = Vector{MPS}()
    ψ0 = MPS(ξ,["Dn" for n in 1:N]  )

    @showprogress dt=1 for r in 1:NU
        basis_transformation = group.measurements[r].measurement_setting.basis_transformation
        ψu = apply(basis_transformation,ψ0)
        Pu = get_Born_MPS(ψu)

        O = MPO(ξ)
        for i in 1:N
            s0 = state(ξ[i],"Dn")
            s1 = state(ξ[i],"Up")
            O[i] = s0*s0'*onehot(v''[i]=>1)-s1*s1'*onehot(v''[i]=>1)
            O[i] += 2*s1*s1'*onehot(v''[i]=>2)
        end
        Ou = apply(basis_transformation,O;apply_dag=true)
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

"""
    apply_map(map::MPO,state::MPO,s::Vector{Index{Int64}},ξ::Vector{Index{Int64}})

Apply a map map((s,s')→(ξ,ξ')) on a state of indices (ξ,ξ')
"""
function apply_map(map::Vector{ITensor},state::MPO,s::Vector{Index{Int64}},ξ::Vector{Index{Int64}})
    N = length(map)
    state_s = [state[i]*δ(s[i],ξ[i])*δ(s'[i],ξ'[i]) for i in 1:N] #relabel indices as input indices
    return MPO([map[i]*state_s[i] for i in 1:N])

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
    return norm2_vec(combined_map,identity_map)/4^N #we normalize with the norm of the identity_map
end

# Constructor for ShallowSShadow from raw measurement results and unitaries
"""
    ShallowShadow(measurement_results::Vector{Int}, basis_transformation::Vector{ITensor};
                     G::Vector{Float64} = fill(1.0, length(basis_transformation)))

Construct a `ShallowSShadow` object from raw measurement results and unitary transformations.

# Arguments
- `measurement_results::Vector{Int}`: Vector of binary measurement results for each qubit/site.
- `basis_transformation::Vector{ITensor}`: Vector of local unitary transformations applied during the measurement.

# Returns
A `ShallowShadow` object.
"""
function ShallowShadow(measurement_results::Vector{Int}, basis_transformation::Vector{ITensor}, inverse_shallow_map::Vector{ITensor},s::Vector{Index{Int64}},ξ::Vector{Index{Int64}})
    N = length(measurement_results) # Number of qubits/sites
    K = length(basis_transformation)

    basis_transformation_dag = reverse([swapprime(dag(basis_transformation[k]),0,1) for k in 1:K])

    # Construct the factorized shadow for each qubit/site
    shadow_data = Vector{ITensor}(undef, N)

    states = [measurement_results[i]==2 ? "Dn" : "Up" for i in 1:N]
    ψ0  = MPS(ComplexF64,ξ,states);

    ψ = apply(basis_transformation_dag,ψ0)
    ρ = outer(ψ',ψ)
    shadow_data = apply_map(inverse_shallow_map,ρ,s,ξ)
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
    basis_transformation = setting.basis_transformation
    measurement_results = measurement_data.measurement_results
    NM = measurement_data.NM

    return [ShallowShadow(measurement_results[m,:], basis_transformation, inverse_shallow_map,s,ξ) for m in 1:NM]
end

"""
    get_expect_shadow(O::MPO, shadow::ShallowShadow)

Compute the expectation value of an MPO operator `O` using a shallow shadow.

# Arguments:
- `O::MPO`: The MPO operator for which the expectation value is computed.
- `shadow::ShallowShadow`: A factorized shadow object.

# Returns:
The expectation value as a `ComplexF64` (or `Float64` if purely real).
"""
function get_expect_shadow(O::MPO, shadow::ShallowShadow)
    N = shadow.N
    ξ = shadow.site_indices
    X = 1
    for i in 1:N
        s = ξ[i]
        X *= shadow.shadow_data[i]'
        X *= O[i] * δ(s, s'')
    end
    return X[]  # Return the full complex value
end

"""
    get_expect_shadow(O::MPO, measurement_data::MeasurementData{ShallowUnitaryMeasurementSetting}, inverse_shallow_map::Vector{ITensor},s::Vector{Index{Int64}},ξ::Vector{Index{Int64}})

Compute the expectation value of an MPO operator `O` from a shallow MeasurementData and and inverse shallow_map

# Returns:
The expectation value as a `ComplexF64` (or `Float64` if purely real).
"""
function get_expect_shadow(O::MPO, measurement_data::MeasurementData{ShallowUnitaryMeasurementSetting}, inverse_shallow_map::Vector{ITensor},s::Vector{Index{Int64}},ξ::Vector{Index{Int64}})
    N = measurement_data.N
    @assert N<17 "Expensive routine memorywise, reduce N or use different get_expect_shadow method"
    inverse_observable = apply_map(inverse_shallow_map,O,s,ξ)
    basis_transformation = measurement_data.measurement_setting.basis_transformation
    inverse_observable_u = apply(basis_transformation,inverse_observable;apply_dag=true)
    inverse_observable_u_diagonal = flatten(get_Born_MPS(inverse_observable_u))

    probability = MeasurementProbability(measurement_data)
    return (inverse_observable_u_diagonal*probability.measurement_probability)[]
end
