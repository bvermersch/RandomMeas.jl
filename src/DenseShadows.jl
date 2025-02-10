# Dense Classical Shadow: Represents a 2^N x 2^N ITensor
"""
    DenseShadow

A struct representing a dense classical shadow, stored as a single ITensor.

# Fields
- `shadow_data::ITensor`: The dense shadow as an ITensor with legs `ξ` and `ξ'`.
- `N::Int`: Number of qubits/sites.
- `ξ::Vector{Index{Int64}}`: Vector of site indices.

# Constructor
`DenseShadow(shadow_data::ITensor, N::Int, ξ::Vector{Index{Int64}})`
"""
struct DenseShadow <: AbstractShadow
    shadow_data::ITensor
    N::Int
    ξ::Vector{Index{Int64}}

    """
    Create a `DenseShadow` object with validation.

    # Arguments
    - `shadow_data::ITensor`: The dense shadow tensor.
    - `N::Int`: Number of qubits/sites.
    - `ξ::Vector{Index{Int64}}`: Vector of site indices.

    # Throws
    - `AssertionError` if dimensions of `ξ` do not match `N`.
    """
    function DenseShadow(shadow_data::ITensor, N::Int, ξ::Vector{Index{Int64}})
        @assert length(ξ) == N "Length of site indices ξ must match N."
        new(shadow_data, N, ξ)
    end
end

# Constructor with a precomputed probability tensor `P`
"""
    DenseShadow(P::ITensor, u::Vector{ITensor}; G::Vector{Float64} = fill(1.0, length(u)))

Construct a `DenseShadow` object from a precomputed probability tensor.

# Arguments
- `P::ITensor`: Probability tensor representing measurement results.
- `u::Vector{ITensor}`: Vector of local unitary transformations.
- `G::Vector{Float64}` (optional): Vector of G values to account for measurement errors (default: 1.0 for all sites).

# Returns
A `DenseShadow` object.
"""
function DenseShadow(P::ITensor, u::Vector{ITensor}; G::Vector{Float64} = fill(1.0, length(u)))
    N = length(u)  # Number of qubits/sites
    ξ = [noprime(first(inds(ui))) for ui in u]  # Extract site indices from unitaries
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

# Constructor with integer array `measurement_results`
"""
    DenseShadow(measurement_results::Array{Int}, u::Vector{ITensor}; G::Vector{Float64} = fill(1.0, length(u)))

Construct a `DenseShadow` object from raw measurement results.

# Arguments
- `measurement_results::Array{Int}`: Array of measurement results (binary).
- `u::Vector{ITensor}`: Vector of local unitary transformations.
- `G::Vector{Float64}` (optional): Vector of G values to account for measurement errors (default: 1.0 for all sites).

# Returns
A `DenseShadow` object.
"""
function DenseShadow(measurement_results::Array{Int}, u::Vector{ITensor}; G::Vector{Float64} = fill(1.0, size(measurement_results, 2)))
    ξ = [noprime(first(inds(ui))) for ui in u]  # Extract site indices from unitaries
    P = get_Born(measurement_results, ξ)  # Compute Born probabilities
    return DenseShadow(P, u, G=G)  # Construct the shadow
end

# Batch Dense Shadows
"""
    get_dense_shadows(measurement_data::MeasurementData{LocalUnitaryMeasurementSettings};
                      G::Vector{Float64} = fill(1.0, N),
                      number_of_ru_batches::Int = NU,
                      number_of_projective_measurement_batches::Int = 1)

Compute dense shadows for the provided measurement data in batches.

# Arguments
- `measurement_data::MeasurementData{LocalUnitaryMeasurementSettings}`: Measurement data object.
- `G::Vector{Float64}` (optional): Vector of G values for robustness (default: 1.0 for all sites).
- `number_of_ru_batches::Int` (optional): Number of random unitary batches (default: `NU`).
- `number_of_projective_measurement_batches::Int` (optional): Number of projective measurement batches (default: 1).

# Returns
A 2D array of `DenseShadow` objects.
"""
function get_dense_shadows(
    measurement_data::MeasurementData{LocalUnitaryMeasurementSettings};
    G::Vector{Float64} = fill(1.0, measurement_data.N),
    number_of_ru_batches::Int = measurement_data.measurement_settings.NU,
    number_of_projective_measurement_batches::Int = 1
)
    # Extract dimensions
    NU, NM, N = measurement_data.NU, measurement_data.NM, measurement_data.N
    ξ = measurement_data.measurement_settings.site_indices
    u = measurement_data.measurement_settings.local_unitaries
    data = measurement_data.measurement_results

    # Ensure G length matches the number of qubits
    @assert length(G) == N "Length of G must match the number of qubits/sites."

    # Create batches for RUs and projective measurements
    batch_size = div(NU, number_of_ru_batches)
    ru_batches = [((b - 1) * batch_size + 1):(b == number_of_ru_batches ? NU : b * batch_size) for b in 1:number_of_ru_batches]
    batch_size = div(NM, number_of_projective_measurement_batches)
    measurement_batches = [((b - 1) * batch_size + 1):(b == number_of_projective_measurement_batches ? NM : b * batch_size) for b in 1:number_of_projective_measurement_batches]

    # Initialize array to store dense shadows
    shadows = Array{DenseShadow}(undef, number_of_ru_batches, number_of_projective_measurement_batches)

    # Compute shadows for each batch
    for (batch_idx, ru_batch) in enumerate(ru_batches)
        for (batch_idy, m_batch) in enumerate(measurement_batches)
            batch_shadow = ITensor(vcat(ξ, prime(ξ)))  # Initialize batch shadow tensor
            for r in ru_batch
                shadow_temp = DenseShadow(data[r, m_batch, :], u[r, :]; G = G).shadow_data  # Compute shadow
                batch_shadow += shadow_temp
            end
            shadows[batch_idx, batch_idy] = DenseShadow(batch_shadow / length(ru_batch) , N, ξ)
        end
    end

    return shadows
end



# Batch Dense Shadows
"""
    get_dense_shadows(measurement_probabilities::MeasurementProbabilities{LocalUnitaryMeasurementSettings};
                      G::Vector{Float64} = fill(1.0, N),
                      number_of_ru_batches::Int = NU)

Compute dense shadows for the provided the probabilities of measurement outcomes in batches.

# Arguments
- `measurement_probabilities::MeasurementProbabilities{LocalUnitaryMeasurementSettings}`: Measurement probabilities object.
- `G::Vector{Float64}` (optional): Vector of G values for robustness (default: 1.0 for all sites).
- `number_of_ru_batches::Int` (optional): Number of random unitary batches (default: `NU`).

# Returns
A 2D array of `DenseShadow` objects.
"""
function get_dense_shadows(
    measurement_probabilities::MeasurementProbabilities{LocalUnitaryMeasurementSettings};
    G::Vector{Float64} = fill(1.0, measurement_probabilities.N),
    number_of_ru_batches::Int = measurement_probabilities.measurement_settings.NU,
)
    # Extract dimensions
    NU, N = measurement_probabilities.NU, measurement_probabilities.N
    ξ = measurement_probabilities.measurement_settings.site_indices
    u = measurement_probabilities.measurement_settings.local_unitaries
    probabilities = measurement_probabilities.measurement_probabilities

    # Ensure G length matches the number of qubits
    @assert length(G) == N "Length of G must match the number of qubits/sites."

    # Create batches for RUs and projective measurements
    batch_size = div(NU, number_of_ru_batches)
    ru_batches = [((b - 1) * batch_size + 1):(b == number_of_ru_batches ? NU : b * batch_size) for b in 1:number_of_ru_batches]

    # Initialize array to store dense shadows
    shadows = Array{DenseShadow}(undef, number_of_ru_batches, 1)

    # Compute shadows for each batch
    for (batch_idx, ru_batch) in enumerate(ru_batches)
            batch_shadow = ITensor(vcat(ξ, prime(ξ)))  # Initialize batch shadow tensor
            for r in ru_batch
                shadow_temp = DenseShadow(probabilities[r], u[r, :]; G = G).shadow_data  # Compute shadow
                batch_shadow += shadow_temp
            end
            shadows[batch_idx, 1] = DenseShadow(batch_shadow / length(ru_batch) , N, ξ)
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
    ξ = shadow.ξ
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
    @assert shadow1.ξ == shadow2.ξ "Site indices mismatch between shadows."

    # Perform the trace product of the shadows
    product_shadow = mapprime(shadow1.shadow_data * prime(shadow2.shadow_data), 2, 1)

    # Return a new DenseShadow object with the resulting shadow, while retaining the original indices and G values
    return DenseShadow(product_shadow, shadow1.N, shadow1.ξ)
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

    # Contract all indices ξ[i] with their primes ξ'[i]
    for i in 1:shadow.N
        shadow_tensor *= δ(shadow.ξ[i], prime(shadow.ξ[i]))
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
    trace_out_ξ = shadow.ξ[trace_out_indices]

    # Compute the partial trace
    reduced_shadow_data = copy(shadow.shadow_data)
    for idx in trace_out_ξ
        reduced_shadow_data *= δ(idx, prime(idx))  # Contract indices to perform the trace
    end

    # Extract the reduced site indices
    reduced_ξ = shadow.ξ[subsystem]

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
        a = shadow.ξ[i]      # unprimed index for site i
        b = prime(a)         # its primed partner
        A = swapind(A, a, b)  # swap the indices; swapind returns a view
    end
    return DenseShadow(A, shadow.N, shadow.ξ)
end

############


"""
    get_purity_dense_shadows(measurement_data::MeasurementData{LocalUnitaryMeasurementSettings})

Compute the purity (second trace moment) using dense shadows.
# Arguments:
- `measurement_data::MeasurementData{LocalUnitaryMeasurementSettings}`: The measurement data object containing results, settings, and site indices.

# Returns:
The purity (second trace moment) as a `Float64`.

# Notes:
This function is specifically optimized fork = 2, providing significant speed-up for large datasets:
- **Linear Scaling:** The method scales linearly in the number of unitaries ( NU ), as opposed to the quadratic scaling of general trace moment estimators.
- **No Batching:** All measurement results are processed without dividing into batches, ensuring smallest statistical errors.

"""
function get_purity_dense_shadows(measurement_data::MeasurementData{LocalUnitaryMeasurementSettings}, subsystem::Vector{Int} = collect(1:data.N))

    #TODO  "This function is not yet optimized for k=2. Instead, it use standard dense batch shadows to compute the purity."

    measurement_data = reduce_to_subsystem(measurement_data, subsystem)

    dense_shadows = get_dense_shadows(measurement_data,number_of_ru_batches=2)

    return get_trace_moment(dense_shadows,2)

    # NU, NM, N = measurement_data.NU, measurement_data.NM, measurement_data.N
    # ξ = measurement_data.measurement_settings.site_indices
    # u = measurement_data.measurement_settings.local_unitaries
    # data = measurement_data.measurement_results

    # # Initialize accumulators for mean shadow and mean squared shadow
    # shadow_mean = ITensor(vcat(ξ, prime(ξ)))  # Mean shadow
    # shadow2_mean = ITensor(vcat(ξ, prime(ξ)))  # Mean squared shadow

    # # Accumulate shadows and squared shadows
    # for r in 1:NU
    #     # Construct the DenseShadow for each random unitary
    #     shadow_temp = DenseShadow(data[r, :, :], u[r, :]).dense_shadow

    #     # Update accumulators
    #     shadow_mean += shadow_temp
    #     shadow2_mean += power(shadow_temp, 2)  # Compute squared shadow
    # end

    # # Compute the purity using the trace
    # purity = real(trace(power(shadow_mean, 2), ξ) - trace(shadow2_mean, ξ)) / (NU * (NU - 1))
    # return purity
end
