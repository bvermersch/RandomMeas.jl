# Factorized Classical Shadow Constructor
"""
    FactorizedShadow

A struct representing a factorized classical shadow for a quantum system.

# Fields
- `shadow_data::Vector{ITensor}`: Array of `N` ITensors, each 2x2, representing the factorized shadow for each qubit/site.
- `N::Int`: Number of qubits/sites.
- `ξ::Vector{Index{Int64}}`: Vector of site indices corresponding to the qubits/sites.

# Constructor
`FactorizedShadow(shadow_data::Vector{ITensor}, N::Int, ξ::Vector{Index{Int64}})`
"""
struct FactorizedShadow <: AbstractShadow
    shadow_data::Vector{ITensor}  # Array of N ITensors, each 2x2
    N::Int                             # Number of qubits/sites
    ξ::Vector{Index{Int64}}            # Vector of site indices

    """
    Create a `FactorizedShadow` object with validation.

    # Arguments
    - `shadow_data::Vector{ITensor}`: Array of ITensors representing the factorized shadow for each qubit/site.
    - `N::Int`: Number of qubits/sites.
    - `ξ::Vector{Index{Int64}}`: Vector of site indices corresponding to the qubits/sites.

    # Throws
    - `AssertionError` if the dimensions of `shadow_data`, `ξ` do not match `N`.
    """
    function FactorizedShadow(shadow_data::Vector{ITensor}, N::Int, ξ::Vector{Index{Int64}})
        @assert length(shadow_data) == N "Length of shadow_data must match N."
        @assert length(ξ) == N "Length of site indices ξ must match N."
        new(shadow_data, N, ξ)
    end
end

# Constructor for FactorizedShadow from raw measurement results and unitaries
"""
    FactorizedShadow(measurement_results::Vector{Int}, local_unitaries::Vector{ITensor};
                     G::Vector{Float64} = fill(1.0, length(local_unitaries)))

Construct a `FactorizedShadow` object from raw measurement results and unitary transformations.

# Arguments
- `measurement_results::Vector{Int}`: Vector of binary measurement results for each qubit/site.
- `local_unitaries::Vector{ITensor}`: Vector of local unitary transformations applied during the measurement.
- G::Vector{Float64}` (optional): Vector of `G` values for measurement error correction (default: 1.0 for all sites).

# Returns
A `FactorizedShadow` object.
"""
function FactorizedShadow(measurement_results::Vector{Int}, local_unitaries::Vector{ITensor}; G::Vector{Float64} = fill(1.0, length(local_unitaries)))
    # Number of qubits/sites
    N = length(local_unitaries)

    # Validate dimensions
    @assert length(G) == N "Length of G ($length(G)) must match the number of qubits/sites (N = $N)."
    @assert length(measurement_results) == N "Length of measurement_results ($length(measurement_results)) must match the number of qubits/sites (N = $N)."

    # Extract site indices from local unitaries
    ξ = [noprime(first(inds(u))) for u in local_unitaries]

    # Construct the factorized shadow for each qubit/site
    shadow_data = Vector{ITensor}(undef, N)
    for i in 1:N
        # Coefficients for error correction
        α = 3.0 / (2.0 * G[i] - 1.0)
        β = (G[i] - 2.0) / (2.0 * G[i] - 1.0)

        # Construct the shadow ITensor
        ψ = dag(local_unitaries[i]) * onehot(ξ[i]' => measurement_results[i])  # State vector after measurement
        shadow = α * ψ' * dag(ψ) + β * δ(ξ[i], ξ[i]')  # Weighted sum of rank-1 projector and identity
        shadow_data[i] = shadow

    end

    return FactorizedShadow(shadow_data, N, ξ)
end

# Factorized Shadows
"""
    get_factorized_shadows(measurement_data::MeasurementData{LocalUnitaryMeasurementSettings};
                           G::Vector{Float64} = fill(1.0, measurement_data.N))

Compute factorized shadows for all measurement results in the provided `MeasurementData`.

# Arguments
- `measurement_data::MeasurementData{LocalUnitaryMeasurementSettings}`: Measurement data object containing measurement results and settings.
- `G::Vector{Float64}` (optional): Vector of `G` values for measurement error correction (default: 1.0 for all sites).

# Returns
A 2D array of `FactorizedShadow` objects with dimensions `(NU, NM)`.
"""
function get_factorized_shadows(measurement_data::MeasurementData{LocalUnitaryMeasurementSettings}; G::Vector{Float64} = fill(1.0, measurement_data.N))
    # Extract dimensions from measurement data
    NU, NM, _ = size(measurement_data.measurement_results)
    shadows = Array{FactorizedShadow}(undef, NU, NM)

    for r in 1:NU
        for m in 1:NM
            # Extract local unitary transformations and measurement results for this RU/shot
            local_unitaries = measurement_data.measurement_settings.local_unitaries[r, :]
            data = measurement_data.measurement_results[r, m, :]

            # Construct a FactorizedShadow for this RU/shot
            shadows[r, m] = FactorizedShadow(data, local_unitaries; G = G)
        end
    end

    return shadows
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


"""
    multiply(shadow1::FactorizedShadow, shadow2::FactorizedShadow)

Multiply two `FactorizedShadow` objects element-wise.

# Arguments
- `shadow1::FactorizedShadow`: The first `FactorizedShadow` object.
- `shadow2::FactorizedShadow`: The second `FactorizedShadow` object.

# Returns
A new `FactorizedShadow` object representing the element-wise product of the two inputs.

# Notes
- Both `shadow1` and `shadow2` must have the same number of qubits/sites.
"""
function multiply(shadow1::FactorizedShadow, shadow2::FactorizedShadow)
    @assert shadow1.N == shadow2.N "Number of qubits/sites must match."
    @assert shadow1.ξ == shadow2.ξ "Site indices must match."

    # Perform element-wise multiplication of the shadows with mapprime
    combined_shadows = Vector{ITensor}(undef, shadow1.N)
    for i in 1:shadow1.N
        combined_shadows[i] = mapprime(shadow1.shadow_data[i] * prime(shadow2.shadow_data[i]), 2, 1)
    end

    # Return the new FactorizedShadow
    return FactorizedShadow(combined_shadows, shadow1.N, shadow1.ξ)
end


"""
    trace(shadow::FactorizedShadow)

Compute the trace of a `FactorizedShadow` object.

# Arguments
- `shadow::FactorizedShadow`: The `FactorizedShadow` object whose trace is to be computed.

# Returns
The trace of the shadow as a `Float64` or `ComplexF64`.

# Notes
- The function computes the product of the traces of individual tensors in the factorized shadow.
"""
function trace(shadow::FactorizedShadow)

    # Initialize the total trace
    total_trace = 1.0

    # Compute the product of traces of individual tensors
    for i in 1:shadow.N
        tensor_trace = scalar(shadow.shadow_data[i] * δ(shadow.ξ[i], prime(shadow.ξ[i])))
        total_trace *= tensor_trace
    end

    return total_trace
end
