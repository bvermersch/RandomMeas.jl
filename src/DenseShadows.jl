# Copyright (c) 2024 Benoît Vermersch and Andreas Elben
# SPDX-License-Identifier: Apache-2.0
# http://www.apache.org/licenses/LICENSE-2.0

# Constructor with a precomputed probability tensor `P`
"""
    DenseShadow(measurement_probability::MeasurementProbability; G::Vector{Float64} = fill(1.0, length(u)))

Construct a `DenseShadow` object from a precomputed probability tensor.

# Arguments
- `P::ITensor`: Probability tensor representing measurement results.
- `u::Vector{ITensor}`: Vector of local unitary transformations.
- `G::Vector{Float64}` (optional): Vector of G values to account for measurement errors (default: 1.0 for all sites).

# Returns
A `DenseShadow` object.
"""
function DenseShadow(Probability::MeasurementProbability{LocalUnitaryMeasurementSetting}; G::Vector{Float64} = fill(1.0, Probability.N))
    N = Probability.N  # Number of qubits/sites
    setting  = Probability.measurement_setting
    ξ = setting.site_indices
    u = setting.local_unitary
    P = Probability.measurement_probability
    rho = 2^N * deepcopy(P)  # Scale the probability tensor

    # Apply transformations for each qubit
    for i in 1:N
        s = ξ[i]  # Site index
        Hamming_tensor = get_h_tensor(s, s'', G[i])  # Construct Hamming tensor
        rho *= Hamming_tensor  # Apply Hamming tensor
        rho *= δ(s, s', s'')  # Add delta constraints
        ut = u[i] * δ(s'', s) * δ(s, s')  # Unitary transformation
        rho = mapprime(ut * rho, 2, 0)  # Prime index management
        ut = dag(u[i]) * δ(s'', s)  # Adjoint unitary transformation
        rho = mapprime(ut * rho, 2, 1)
    end

    return DenseShadow(rho, N, ξ)
end

# Constructor with MeasurementData object
"""
    DenseShadow(measurement_data::MeasurementData{LocalUnitaryMeasurementSetting}; G::Vector{Float64} = fill(1.0, size(measurement_data.N, 2)))

Construct a `DenseShadow` object from a MeasurementDataObject

# Arguments
- `measurement_data::MeasurementData{LocalUnitaryMeasurementSetting}:
- `G::Vector{Float64}` (optional): Vector of G values to account for measurement errors (default: 1.0 for all sites).

# Returns
A `DenseShadow` object.
"""
function DenseShadow(measurement_data::MeasurementData{LocalUnitaryMeasurementSetting}; G::Vector{Float64} = fill(1.0, measurement_data.N))
    Probability = MeasurementProbability(measurement_data)
    return DenseShadow(Probability, G=G)  # Construct the shadow
end


# Batch Dense Shadows
"""
    get_dense_shadows(measurement_group::MeasurementGroup{LocalUnitaryMeasurementSetting};
                      G::Vector{Float64} = fill(1.0, N),
                      number_of_ru_batches::Int = NU)

Compute dense shadows for the provided measurement data in batches.

# Arguments
- `measurement_group::MeasurementGroup{LocalUnitaryMeasurementSetting}`: Measurement data object.
- `G::Vector{Float64}` (optional): Vector of G values for robustness (default: 1.0 for all sites).
- `number_of_ru_batches::Int` (optional): Number of random unitary batches (default: `NU`).

# Returns
A Vector of `DenseShadow` objects.
"""
function get_dense_shadows(
    measurement_group::MeasurementGroup{LocalUnitaryMeasurementSetting};
    G::Vector{Float64} = fill(1.0, measurement_group.N),
    number_of_ru_batches::Int = measurement_group.NU
)
    # Extract dimensions
    NU, N = measurement_group.NU, measurement_group.N
    #u = measurement_data.measurement_settings.local_unitaries
    data = measurement_group.measurements
    ξ = data[1].measurement_setting.site_indices

    # Ensure G length matches the number of qubits
    @assert length(G) == N "Length of G must match the number of qubits/sites."

    # Create batches for RUs and projective measurements
    batch_size = div(NU, number_of_ru_batches)
    ru_batches = [((b - 1) * batch_size + 1):(b == number_of_ru_batches ? NU : b * batch_size) for b in 1:number_of_ru_batches]

    # Initialize array to store dense shadows
    shadows = Vector{DenseShadow}(undef, number_of_ru_batches)

    # Compute shadows for each batch
    for (batch_id, ru_batch) in enumerate(ru_batches)
            batch_shadow = ITensor(vcat(ξ, prime(ξ)))  # Initialize batch shadow tensor
            for r in ru_batch
                shadow_temp = DenseShadow(data[r]; G = G).shadow_data  # Compute shadow
                batch_shadow += shadow_temp
            end
            shadows[batch_id] = DenseShadow(batch_shadow / length(ru_batch) , N, ξ)
    end

    return shadows
end


"""
    get_expect_shadow(O::MPO, shadow::DenseShadow)

Compute the expectation value of an MPO operator `O` using a dense shadow.

# Arguments:
- `O::MPO`: The MPO operator for which the expectation value is computed.
- `shadow::DenseShadow`: A dense shadow object.

# Returns:
The expectation value as a `ComplexF64` (or `Float64` if purely real).
"""
function get_expect_shadow(O::MPO, shadow::DenseShadow)
    N = shadow.N
    ξ = shadow.site_indices
    X = 1 * shadow.shadow_data'
    for i in 1:N
        s = ξ[i]
        X *= O[i] * δ(s, s'')
    end
    return X[]  # Return the full complex value
end



"""
    multiply(shadow1::DenseShadow, shadow2::DenseShadow)

Compute the trace product of two dense shadows.

# Arguments
- `shadow1::DenseShadow`: The first dense shadow.
- `shadow2::DenseShadow`: The second dense shadow.

# Returns
A new `DenseShadow` object that represents the trace product of the two input shadows.

# Notes
- The shadows must have the same site indices (`ξ`) and number of qubits (`N`).
"""
function multiply(shadow1::DenseShadow, shadow2::DenseShadow)::DenseShadow
    @assert shadow1.N == shadow2.N "Number of qubits/sites mismatch between shadows."
    @assert shadow1.site_indices == shadow2.site_indices "Site indices mismatch between shadows."

    # Perform the trace product of the shadows
    product_shadow = mapprime(shadow1.shadow_data * prime(shadow2.shadow_data), 2, 1)

    # Return a new DenseShadow object with the resulting shadow, while retaining the original indices and G values
    return DenseShadow(product_shadow, shadow1.N, shadow1.site_indices)
end


"""
    trace(shadow::DenseShadow)

Compute the trace of a `DenseShadow` object.

# Arguments
- `shadow::DenseShadow`: The `DenseShadow` object whose trace is to be computed.

# Returns
The trace of the shadow as a `Float64` or `ComplexF64`.

# Notes
- The function contracts the `ξ` and `ξ'` indices of the shadow's ITensor.
"""
function trace(shadow::DenseShadow)

    # Copy the shadow ITensor to avoid modifying the original
    shadow_tensor = copy(shadow.shadow_data)

    # Contract all indices site_indices[i] with their primes site_indices'[i]
    for i in 1:shadow.N
        shadow_tensor *= δ(shadow.site_indices[i], prime(shadow.site_indices[i]))
    end

    # Extract the resulting scalar value
    return scalar(shadow_tensor)
end


"""
    partial_trace(shadow::DenseShadow, subsystem::Vector{Int})

Compute the partial trace of a `DenseShadow` object over the complement of the specified subsystem.

# Arguments
- `shadow::DenseShadow`: The dense shadow to compute the partial trace for.
- `subsystem::Vector{Int}`: A vector of site indices (1-based) specifying the subsystem to retain.

# Returns
A new `DenseShadow` object reduced to the specified subsystem.
"""
function partial_trace(shadow::DenseShadow, subsystem::Vector{Int})::DenseShadow
    # Validate the subsystem
    @assert all(x -> x >= 1 && x <= shadow.N, subsystem) "Subsystem indices must be between 1 and N."
    @assert length(unique(subsystem)) == length(subsystem) "Subsystem indices must be unique."

    # Determine indices to trace out
    trace_out_indices = setdiff(1:shadow.N, subsystem)
    trace_out_ξ = shadow.site_indices[trace_out_indices]

    # Compute the partial trace
    reduced_shadow_data = copy(shadow.shadow_data)
    for idx in trace_out_ξ
        reduced_shadow_data *= δ(idx, prime(idx))  # Contract indices to perform the trace
    end

    # Extract the reduced site indices
    reduced_ξ = shadow.site_indices[subsystem]

    # Construct and return the reduced DenseShadow
    return DenseShadow(reduced_shadow_data, length(subsystem), reduced_ξ)
end


"""
    partial_transpose(shadow::DenseShadow, subsystem::Vector{Int})::DenseShadow

Compute the partial transpose of a DenseShadow over the specified subsystem by swapping, for each site,
the unprimed index with its primed partner. This is done using the `swapind` function, which returns a view of
the underlying ITensor.

# Arguments
- `shadow::DenseShadow`: The dense classical shadow.
- `subsystem::Vector{Int}`: A vector of 1-based site indices on which to perform the partial transpose.

# Returns
A new DenseShadow with the specified sites partially transposed.
"""
function partial_transpose(shadow::DenseShadow, subsystem::Vector{Int})::DenseShadow
    @assert all(i -> i ≥ 1 && i ≤ shadow.N, subsystem) "Subsystem indices must be between 1 and N."
    @assert length(unique(subsystem)) == length(subsystem) "Subsystem indices must be unique."

    # Work on a view of the internal ITensor.
    A = shadow.shadow_data
    for i in subsystem
        a = shadow.site_indices[i]      # unprimed index for site i
        b = prime(a)         # its primed partner
        A = swapind(A, a, b)  # swap the indices; swapind returns a view
    end
    return DenseShadow(A, shadow.N, shadow.site_indices)
end
