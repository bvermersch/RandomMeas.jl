# Abstract Shadow Type
"""
    AbstractShadow

An abstract type representing a general classical shadow.
Subtypes should implement specific shadow methodologies, such as dense or factorized shadows.
"""
abstract type AbstractShadow end

# Dense Classical Shadow: Represents a 2^N x 2^N ITensor
"""
    DenseShadow

A struct representing a dense classical shadow, stored as a single ITensor.

# Fields
- `shadow_data::ITensor`: The dense shadow as an ITensor with legs `ξ` and `ξ'`.
- `N::Int`: Number of qubits/sites.
- `ξ::Vector{Index{Int64}}`: Vector of site indices.
- `G::Vector{Float64}`: Vector of `G` values to account for measurement errors and define robust classical shadows.

# Constructor
`DenseShadow(shadow_data::ITensor, N::Int, ξ::Vector{Index{Int64}}, G::Vector{Float64})`
"""
struct DenseShadow <: AbstractShadow
    shadow_data::ITensor
    N::Int
    ξ::Vector{Index{Int64}}
    G::Vector{Float64}

    """
    Create a `DenseShadow` object with validation.

    # Arguments
    - `shadow_data::ITensor`: The dense shadow tensor.
    - `N::Int`: Number of qubits/sites.
    - `ξ::Vector{Index{Int64}}`: Vector of site indices.
    - `G::Vector{Float64}`: Vector of G values.

    # Throws
    - `AssertionError` if dimensions of `ξ` or `G` do not match `N`.
    """
    function DenseShadow(shadow_data::ITensor, N::Int, ξ::Vector{Index{Int64}}, G::Vector{Float64})
        @assert length(ξ) == N "Length of site indices ξ must match N."
        @assert length(G) == N "Length of G must match N."
        new(shadow_data, N, ξ, G)
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

    return DenseShadow(rho, N, ξ, G)
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
    return DenseShadow(P, u; G = G)  # Construct the shadow
end

# Batch Dense Shadows
"""
    get_dense_shadows(measurement_data::MeasurementData{LocalUnitaryMeasurementSettings};
                      G_vec::Vector{Float64} = fill(1.0, N),
                      number_of_ru_batches::Int = NU,
                      number_of_projective_measurement_batches::Int = 1)

Compute dense shadows for the provided measurement data in batches.

# Arguments
- `measurement_data::MeasurementData{LocalUnitaryMeasurementSettings}`: Measurement data object.
- `G_vec::Vector{Float64}` (optional): Vector of G values for robustness (default: 1.0 for all sites).
- `number_of_ru_batches::Int` (optional): Number of random unitary batches (default: `NU`).
- `number_of_projective_measurement_batches::Int` (optional): Number of projective measurement batches (default: 1).

# Returns
A 2D array of `DenseShadow` objects.
"""
function get_dense_shadows(
    measurement_data::MeasurementData{LocalUnitaryMeasurementSettings};
    G_vec::Vector{Float64} = fill(1.0, measurement_data.N),
    number_of_ru_batches::Int = measurement_data.measurement_settings.NU,
    number_of_projective_measurement_batches::Int = 1
)
    # Extract dimensions
    NU, NM, N = measurement_data.NU, measurement_data.NM, measurement_data.N
    ξ = measurement_data.measurement_settings.site_indices
    u = measurement_data.measurement_settings.local_unitaries
    data = measurement_data.measurement_results

    # Ensure G_vec length matches the number of qubits
    @assert length(G_vec) == N "Length of G_vec must match the number of qubits/sites."

    # Create batches for RUs and projective measurements
    batch_size = div(NU, number_of_ru_batches)
    ru_batches = [((b - 1) * batch_size + 1):(b == number_of_ru_batches ? NU : b * batch_size) for b in 1:number_of_ru_batches]
    batch_size = div(NM, number_of_projective_measurement_batches)
    measurement_batches = [((b - 1) * batch_size + 1):(b == number_of_projective_measurement_batches ? NM : b * batch_size) for b in 1:number_of_projective_measurement_batches]
    @show ru_batches
    @show measurement_batches

    # Initialize array to store dense shadows
    shadows = Array{DenseShadow}(undef, number_of_ru_batches, number_of_projective_measurement_batches)

    # Compute shadows for each batch
    for (batch_idx, ru_batch) in enumerate(ru_batches)
        for (batch_idy, m_batch) in enumerate(measurement_batches)
            batch_shadow = ITensor(vcat(ξ, prime(ξ)))  # Initialize batch shadow tensor
            for r in ru_batch
                shadow_temp = DenseShadow(data[r, m_batch, :], u[r, :]; G = G_vec).shadow_data  # Compute shadow
                batch_shadow += shadow_temp / length(ru_batch)
            end
            shadows[batch_idx, batch_idy] = DenseShadow(batch_shadow, N, ξ, G_vec)
        end
    end

    return shadows
end


# Factorized Classical Shadow Constructor
"""
    FactorizedShadow

A struct representing a factorized classical shadow for a quantum system.

# Fields
- `shadow_data::Vector{ITensor}`: Array of `N` ITensors, each 2x2, representing the factorized shadow for each qubit/site.
- `N::Int`: Number of qubits/sites.
- `ξ::Vector{Index{Int64}}`: Vector of site indices corresponding to the qubits/sites.
- `G::Vector{Float64}`: Vector of `G` values to account for measurement errors and define robust classical shadows.

# Constructor
`FactorizedShadow(shadow_data::Vector{ITensor}, N::Int, ξ::Vector{Index{Int64}}, G::Vector{Float64})`
"""
struct FactorizedShadow <: AbstractShadow
    shadow_data::Vector{ITensor}  # Array of N ITensors, each 2x2
    N::Int                             # Number of qubits/sites
    ξ::Vector{Index{Int64}}            # Vector of site indices
    G::Vector{Float64}                 # Vector of G values

    """
    Create a `FactorizedShadow` object with validation.

    # Arguments
    - `shadow_data::Vector{ITensor}`: Array of ITensors representing the factorized shadow for each qubit/site.
    - `N::Int`: Number of qubits/sites.
    - `ξ::Vector{Index{Int64}}`: Vector of site indices corresponding to the qubits/sites.
    - `G::Vector{Float64}`: Vector of `G` values for measurement error correction.

    # Throws
    - `AssertionError` if the dimensions of `shadow_data`, `ξ`, or `G` do not match `N`.
    """
    function FactorizedShadow(shadow_data::Vector{ITensor}, N::Int, ξ::Vector{Index{Int64}}, G::Vector{Float64})
        @assert length(shadow_data) == N "Length of shadow_data must match N."
        @assert length(ξ) == N "Length of site indices ξ must match N."
        @assert length(G) == N "Length of G must match N."
        new(shadow_data, N, ξ, G)
    end
end

# Constructor for FactorizedShadow from raw measurement results and unitaries
"""
    FactorizedShadow(measurement_results::Vector{Int}, local_unitaries::Vector{ITensor};
                     G_vec::Vector{Float64} = fill(1.0, length(local_unitaries)))

Construct a `FactorizedShadow` object from raw measurement results and unitary transformations.

# Arguments
- `measurement_results::Vector{Int}`: Vector of binary measurement results for each qubit/site.
- `local_unitaries::Vector{ITensor}`: Vector of local unitary transformations applied during the measurement.
- `G_vec::Vector{Float64}` (optional): Vector of `G` values for measurement error correction (default: 1.0 for all sites).

# Returns
A `FactorizedShadow` object.
"""
function FactorizedShadow(measurement_results::Vector{Int}, local_unitaries::Vector{ITensor}; G_vec::Vector{Float64} = fill(1.0, length(local_unitaries)))
    # Number of qubits/sites
    N = length(local_unitaries)

    # Validate dimensions
    @assert length(G_vec) == N "Length of G_vec ($length(G_vec)) must match the number of qubits/sites (N = $N)."
    @assert length(measurement_results) == N "Length of measurement_results ($length(measurement_results)) must match the number of qubits/sites (N = $N)."

    # Extract site indices from local unitaries
    ξ = [noprime(first(inds(u))) for u in local_unitaries]

    # Construct the factorized shadow for each qubit/site
    shadow_data = Vector{ITensor}(undef, N)
    for i in 1:N
        # Coefficients for error correction
        α = 3.0 / (2.0 * G_vec[i] - 1.0)
        β = (G_vec[i] - 2.0) / (2.0 * G_vec[i] - 1.0)

        # Construct the shadow ITensor
        ψ = dag(local_unitaries[i]) * onehot(ξ[i]' => measurement_results[i])  # State vector after measurement
        shadow = α * ψ' * dag(ψ) + β * δ(ξ[i], ξ[i]')  # Weighted sum of rank-1 projector and identity
        shadow_data[i] = shadow

    end

    return FactorizedShadow(shadow_data, N, ξ, G_vec)
end

# Factorized Shadows
"""
    get_factorized_shadows(measurement_data::MeasurementData{LocalUnitaryMeasurementSettings};
                           G_vec::Vector{Float64} = fill(1.0, measurement_data.N))

Compute factorized shadows for all measurement results in the provided `MeasurementData`.

# Arguments
- `measurement_data::MeasurementData{LocalUnitaryMeasurementSettings}`: Measurement data object containing measurement results and settings.
- `G_vec::Vector{Float64}` (optional): Vector of `G` values for measurement error correction (default: 1.0 for all sites).

# Returns
A 2D array of `FactorizedShadow` objects with dimensions `(NU, NM)`.
"""
function get_factorized_shadows(measurement_data::MeasurementData{LocalUnitaryMeasurementSettings}; G_vec::Vector{Float64} = fill(1.0, measurement_data.N))
    # Extract dimensions from measurement data
    NU, NM, _ = size(measurement_data.measurement_results)
    shadows = Array{FactorizedShadow}(undef, NU, NM)

    for r in 1:NU
        for m in 1:NM
            # Extract local unitary transformations and measurement results for this RU/shot
            local_unitaries = measurement_data.measurement_settings.local_unitaries[r, :]
            data = measurement_data.measurement_results[r, m, :]

            # Construct a FactorizedShadow for this RU/shot
            shadows[r, m] = FactorizedShadow(data, local_unitaries; G_vec = G_vec)
        end
    end

    return shadows
end



# Shadow analysis

"""
    get_moments(shadows::AbstractArray{<:AbstractShadow, 2}, kth_moments::Vector{Int})

Generalized function to compute moments from a 2D array of `AbstractShadow` objects.

# Arguments
- `shadows::AbstractArray{<:AbstractShadow, 2}`: A 2D array of `AbstractShadow` objects.
- `kth_moments::Vector{Int}`: Vector of integers specifying which trace moments to compute.

# Returns
A vector of computed moments corresponding to `kth_moments`.

# Notes
- For now, this function is implemented only for cases where the second dimension equals 1 (e.g., `shadows[:, 1]`).
- The function falls back to the existing implementation for 1D arrays of `DenseShadow` objects.
"""
function get_moments(shadows::AbstractArray{<:AbstractShadow, 2}, kth_moments::Vector{Int})
    # Check that the second dimension equals 1
    @assert size(shadows, 2) == 1 "Currently, only arrays with second dimension = 1 are supported."

    # Flatten the array to a 1D vector and use the existing `get_moments` implementation
    return get_moments([shadows[:, 1]...], kth_moments)
end

"""
    get_moments(shadows::Vector{<:AbstractShadow}, kth_moments::Vector{Int})

Compute moments from a 1D vector of `AbstractShadow` objects.

# Arguments
- `shadows::Vector{<:AbstractShadow}`: A 1D vector of `AbstractShadow` objects.
- `kth_moments::Vector{Int}`: Vector of integers specifying which trace moments to compute.

# Returns
A vector of computed moments corresponding to `kth_moments`.

# Notes
- Internally, this function converts the 1D vector into a 2D array with a trivial second dimension.
"""
function get_moments(shadows::Vector{<:AbstractShadow}, kth_moments::Vector{Int})
    # Convert the vector into a 2D array with a single column
    reshaped_shadows = reshape(shadows, :, 1)

    # Call the more general 2D array implementation
    return get_moments(reshaped_shadows, kth_moments)
end

"""
    get_moment(shadows::AbstractArray{<:AbstractShadow, 2}, kth_moment::Int)

Compute the kth trace moment from a 2D array of shadows.

# Arguments
- `shadows::AbstractArray{<:AbstractShadow, 2}`: A 2D array of shadows, where the first dimension corresponds to the random unitaries (RU).
- `kth_moment::Int`: The kth moment to compute.

# Returns
- The computed kth trace moment.

# Notes
If the second dimension is `1`, the shadows array is flattened, and the computation falls back to `get_moments`.
"""
function get_moment(shadows::AbstractArray{<:AbstractShadow, 2}, kth_moment::Int)
    # Special case for second dimension equals 1
    if size(shadows, 2) == 1
        return get_moments([shadows[:, 1]...], [kth_moment])[1]
    else
        error("get_moment currently only supports 2D arrays of shadows with a second dimension of 1.")
    end
end

"""
    get_moment(shadows::Vector{<:AbstractShadow}, kth_moment::Int)

Compute the kth trace moment from a vector of shadows.

# Arguments
- `shadows::Vector{<:AbstractShadow}`: A vector of shadows.
- `kth_moment::Int`: The kth moment to compute.

# Returns
- The computed kth trace moment.

# Notes
This method adds a trivial second dimension to the shadows vector and falls back to the `get_moment` method for 2D arrays.
"""
function get_moment(shadows::Vector{<:AbstractShadow}, kth_moment::Int)
    # Wrap the vector in a trivial second dimension and call the 2D method
    return get_moment(reshape(shadows, :, 1), kth_moment)
end

"""
    get_moments(shadows::Vector{DenseShadow}, kth_moments::Vector{Int})

Obtain trace moments from a vector of dense shadows using U-statistics.

# Arguments:
- `shadows::Vector{DenseShadow}`: A vector of DenseShadow objects.
- `kth_moments::Vector{Int}`: Vector of moments `k` to compute (e.g., `k=1,2,...`).

# Returns:
A vector of computed trace moments.
"""
function get_moments(shadows::Vector{DenseShadow}, kth_moments::Vector{Int})
    p = Float64[]  # Initialize the vector for trace moments
    n_shadows = length(shadows)

    # Validate kth_moments
    @assert all(kth_moments .>= 1) "Only integer valued moments Tr[rho^k] with k >= 1 can be computed."
    @assert all(kth_moments .<= n_shadows) "The number of shadows must be >= the largest moment k."

    for k in kth_moments
        est = 0.0
        for r in permutations(1:n_shadows, k)
            X = shadows[r[1]].shadow_data
            for m in 2:k
                X = mapprime(X * prime(shadows[r[m]].shadow_data), 2, 1)
            end
            est += real(trace(X, shadows[1].ξ))  # Use helper to compute the trace
        end
        push!(p, est / length(permutations(1:n_shadows, k)))
    end

    return p
end


"""
    get_moment(shadows::Vector{DenseShadow}, kth_moment::Int)

Compute a single trace moment from a vector of dense shadows.

# Arguments:
- `shadows::Vector{DenseShadow}`: A vector of DenseShadow objects.
- `kth_moment::Int`: The moment `k` to compute (e.g., `k=1,2,...`).

# Returns:
The computed trace moment for `kth_moment`.
"""
function get_moment(shadows::Vector{DenseShadow}, kth_moment::Int)
    return get_moments(shadows, [kth_moment])[1]
end

"""
    get_purity_shadows(measurement_data::MeasurementData{LocalUnitaryMeasurementSettings})

Compute the purity (second trace moment) using shadows.
# Arguments:
- `measurement_data::MeasurementData{LocalUnitaryMeasurementSettings}`: The measurement data object containing results, settings, and site indices.

# Returns:
The purity (second trace moment) as a `Float64`.

# Notes:
This function is specifically optimized fork = 2, providing significant speed-up for large datasets:
- **Linear Scaling:** The method scales linearly in the number of unitaries ( NU ), as opposed to the quadratic scaling of general trace moment estimators.
- **No Batching:** All measurement results are processed without dividing into batches, ensuring smallest statistical errors.

"""
function get_purity_shadows(measurement_data::MeasurementData{LocalUnitaryMeasurementSettings})
    NU, NM, N = measurement_data.NU, measurement_data.NM, measurement_data.N
    ξ = measurement_data.measurement_settings.site_indices
    u = measurement_data.measurement_settings.local_unitaries
    data = measurement_data.measurement_results

    # Initialize accumulators for mean shadow and mean squared shadow
    shadow_mean = ITensor(vcat(ξ, prime(ξ)))  # Mean shadow
    shadow2_mean = ITensor(vcat(ξ, prime(ξ)))  # Mean squared shadow

    # Accumulate shadows and squared shadows
    for r in 1:NU
        # Construct the DenseShadow for each random unitary
        shadow_temp = DenseShadow(data[r, :, :], u[r, :]).dense_shadow

        # Update accumulators
        shadow_mean += shadow_temp
        shadow2_mean += power(shadow_temp, 2)  # Compute squared shadow
    end

    # Compute the purity using the trace
    purity = real(trace(power(shadow_mean, 2), ξ) - trace(shadow2_mean, ξ)) / (NU * (NU - 1))
    return purity
end


"""
    get_expect_shadow(O::MPO, shadows::AbstractArray{<:AbstractShadow}; compute_sem::Bool = false)

Compute the average expectation value of an MPO operator `O` using an array of shadows.
Optionally computes the standard error of the mean (SEM) of the expectation values.

# Arguments:
- `O::MPO`: The MPO operator for which the expectation value is computed.
- `shadows::AbstractArray{<:AbstractShadow}`: An array of shadow objects, either dense or factorized, of any dimension.
- `compute_sem::Bool` (optional): Whether to compute and return the standard error of the mean (SEM) (default: `false`).

# Returns:
- If `compute_sem` is `false`: The average expectation value as a `ComplexF64` (or `Float64` if purely real).
- If `compute_sem` is `true`: A tuple `(mean, sem)` where `mean` is the average expectation value and `sem` is the standard error of the mean.
"""
function get_expect_shadow(
    O::MPO,
    shadows::AbstractArray{<:AbstractShadow};
    compute_sem::Bool = false
)
    # Ensure the array of shadows is not empty
    @assert !isempty(shadows) "Array of shadows is empty."

    # Compute all expectation values
    expect_values = [get_expect_shadow(O, shadow) for shadow in shadows]

    # Compute mean
    mean_value = mean(expect_values)

    if compute_sem
        # Compute standard error of the mean (SEM)
        sem_value = std(expect_values) / sqrt(length(expect_values))
        return mean_value, sem_value
    else
        return mean_value
    end
end


"""
    get_expect_shadow(O::MPO, shadow::AbstractShadow)

Compute the expectation value of an MPO operator `O` using a generic shadow.

# Arguments:
- `O::MPO`: The MPO operator for which the expectation value is computed.
- `shadow::AbstractShadow`: A shadow object, either dense or factorized.

# Returns:
The expectation value as a `ComplexF64` (or `Float64` if purely real).
"""
function get_expect_shadow(O::MPO, shadow::AbstractShadow)
    if shadow isa DenseShadow
        return get_expect_shadow(O, shadow::DenseShadow)
    elseif shadow isa FactorizedShadow
        return get_expect_shadow(O, shadow::FactorizedShadow)
    else
        throw(ArgumentError("Unsupported shadow type: $(typeof(shadow))"))
    end
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
    get_expect_shadow(O::MPO, shadow::FactorizedShadow)

Compute the expectation value of an MPO operator `O` using a factorized shadow.

# Arguments:
- `O::MPO`: The MPO operator for which the expectation value is computed.
- `shadow::FactorizedShadow`: A factorized shadow object.

# Returns:
The expectation value as a `ComplexF64` (or `Float64` if purely real).
"""
function get_expect_shadow(O::MPO, shadow::FactorizedShadow)
    N = shadow.N
    ξ = shadow.ξ
    X = 1
    for i in 1:N
        s = ξ[i]
        X *= shadow.shadow_data[i]'
        X *= O[i] * δ(s, s'')
    end
    return X[]  # Return the full complex value
end


#### Helper functions


"""
    square(shadow::ITensor)
"""
function square(shadow::ITensor)
    Y = multiply(shadow, shadow)
    return Y
end

"""
    multiply(shadow::ITensor, shadow2::ITensor)
"""
function multiply(shadow::ITensor, shadow2::ITensor)
    return mapprime(shadow * prime(shadow2), 2, 1)
end

"""
    power(shadow::ITensor, n::Int64)
"""
function power(shadow::ITensor, n::Int64)
    Y = deepcopy(shadow)
    for m in 1:n-1
        Y = multiply(Y, shadow)
    end
    return Y
end

"""
    trace(shadow::ITensor, ξ::Vector{Index{Int64}})
"""
function trace(shadow::ITensor, ξ::Vector{Index{Int64}})
    NA = size(ξ, 1)
    Y = copy(shadow)
    for i in 1:NA
        Y *= δ(ξ[i],ξ[i]')
    end
    return Y[]
end
