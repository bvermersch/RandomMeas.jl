# Constructor for FactorizedShadow from raw measurement results and unitaries
"""
    FactorizedShadow(measurement_results::Vector{Int}, local_unitary::Vector{ITensor};
                     G::Vector{Float64} = fill(1.0, length(local_unitary)))

Construct a `FactorizedShadow` object from raw measurement results and unitary transformations.

# Arguments
- `measurement_results::Vector{Int}`: Vector of binary measurement results for each qubit/site.
- `local_unitary::Vector{ITensor}`: Vector of local unitary transformations applied during the measurement.
- G::Vector{Float64}` (optional): Vector of `G` values for measurement error correction (default: 1.0 for all sites).

# Returns
A `FactorizedShadow` object.
"""
function FactorizedShadow(measurement_results::Vector{Int}, local_unitary::Vector{ITensor}; G::Vector{Float64} = fill(1.0, length(local_unitary)))
    # Number of qubits/sites
    N = length(local_unitary)

    # Validate dimensions
    @assert length(G) == N "Length of G ($length(G)) must match the number of qubits/sites (N = $N)."
    @assert length(measurement_results) == N "Length of measurement_results ($length(measurement_results)) must match the number of qubits/sites (N = $N)."

    # Extract site indices from local unitaries
    ξ = [noprime(first(inds(u))) for u in local_unitary]

    # Construct the factorized shadow for each qubit/site
    shadow_data = Vector{ITensor}(undef, N)
    for i in 1:N
        # Coefficients for error correction
        α = 3.0 / (2.0 * G[i] - 1.0)
        β = (G[i] - 2.0) / (2.0 * G[i] - 1.0)

        # Construct the shadow ITensor
        ψ = dag(local_unitary[i]) * onehot(ξ[i]' => measurement_results[i])  # State vector after measurement
        shadow = α * ψ' * dag(ψ) + β * δ(ξ[i], ξ[i]')  # Weighted sum of rank-1 projector and identity
        shadow_data[i] = shadow

    end

    return FactorizedShadow(shadow_data, N, ξ)
end

# Factorized Shadows
"""
    get_factorized_shadows(measurement_data::MeasurementData{LocalUnitaryMeasurementSetting};
                           G::Vector{Float64} = fill(1.0, measurement_data.N))

Compute factorized shadows for all measurement results in the provided `MeasurementData`.

# Arguments
- `measurement_data::MeasurementData{LocalUnitaryMeasurementSetting}`: Measurement data object containing measurement results and settings.
- `G::Vector{Float64}` (optional): Vector of `G` values for measurement error correction (default: 1.0 for all sites).

# Returns
A Vector of NM `FactorizedShadow` objects with dimensions.
"""
function get_factorized_shadows(measurement_data::MeasurementData{LocalUnitaryMeasurementSetting}; G::Vector{Float64} = fill(1.0, measurement_data.N))
    # Extract dimensions from measurement data
    NM = measurement_data.NM
    shadows = Vector{FactorizedShadow}(undef, NM)
    local_unitary = measurement_data.measurement_setting.local_unitary

        for m in 1:NM
            # Extract local unitary transformations and measurement results for this shot
            data = measurement_data.measurement_results[m, :]

            # Construct a FactorizedShadow for this shot
            shadows[m] = FactorizedShadow(data, local_unitary; G = G)
        end
    return shadows
end

"""
    get_factorized_shadows(measurement_group::MeasurementGroup{LocalUnitaryMeasurementSetting};
                           G::Vector{Float64} = fill(1.0, measurement_group.N))

Compute factorized shadows for all measurement results in the provided `MeasurementGroup`.

# Arguments
- `measurement_group::MeasurementGroup{LocalUnitaryMeasurementSetting}`: Measurement data object containing measurement results and settings.
- `G::Vector{Float64}` (optional): Vector of `G` values for measurement error correction (default: 1.0 for all sites).

# Returns
A Array of NU*NM `FactorizedShadow` objects with dimensions.
"""
function get_factorized_shadows(measurement_group::MeasurementGroup{LocalUnitaryMeasurementSetting}; G::Vector{Float64} = fill(1.0, measurement_group.N))
    # Extract dimensions from measurement data
    NM = measurement_group.NM
    NU = measurement_group.NU
    shadows = Array{FactorizedShadow}(undef, NU, NM)
    for r in 1:NU
        shadows[r,:] = get_factorized_shadows(measurement_group.measurements[r];G=G)
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
    @assert shadow1.site_indices == shadow2.site_indices "Site indices must match."

    # Perform element-wise multiplication of the shadows with mapprime
    combined_shadows = Vector{ITensor}(undef, shadow1.N)
    for i in 1:shadow1.N
        combined_shadows[i] = mapprime(shadow1.shadow_data[i] * prime(shadow2.shadow_data[i]), 2, 1)
    end

    # Return the new FactorizedShadow
    return FactorizedShadow(combined_shadows, shadow1.N, shadow1.site_indices)
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
        tensor_trace = scalar(shadow.shadow_data[i] * δ(shadow.site_indices[i], prime(shadow.site_indices[i])))
        total_trace *= tensor_trace
    end

    return total_trace
end

"""
    partial_trace(shadow::FactorizedShadow, subsystem::Vector{Int}; assume_unit_trace::Bool = false)

Compute the partial trace of a `FactorizedShadow` object over the complement of the specified subsystem.

# Arguments
- `shadow::FactorizedShadow`: The factorized shadow to compute the partial trace for.
- `subsystem::Vector{Int}`: A vector of site indices (1-based) specifying the subsystem to retain.
- `assume_unit_trace::Bool` (optional): If `true`, assumes all traced-out tensors have unit trace and skips explicit computation (default: `false`).

# Returns
A new `FactorizedShadow` object reduced to the specified subsystem.

# Notes
- If `assume_unit_trace` is `true`, avoids explicit trace computation for efficiency.
- If `assume_unit_trace` is `false`, computes the traces of all tensors outside the subsystem and multiplies their product into the remaining tensors.
- Issues a warning if the trace product deviates significantly from 1 when `assume_unit_trace` is `false`.
"""
function partial_trace(shadow::FactorizedShadow, subsystem::Vector{Int}; assume_unit_trace::Bool = false)::FactorizedShadow
    # Validate the subsystem
    @assert all(x -> x >= 1 && x <= shadow.N, subsystem) "Subsystem indices must be between 1 and N."
    @assert length(unique(subsystem)) == length(subsystem) "Subsystem indices must be unique."

    # Extract tensors and site indices for the subsystem
    reduced_shadow_data = shadow.shadow_data[subsystem]
    reduced_ξ = shadow.site_indices[subsystem]

    if !assume_unit_trace

        # Initialize trace product (default to 1 if assume_unit_trace is true)
        trace_product = 1.0

        # Iterate over sites not in the subsystem and compute the trace
        for i in setdiff(1:shadow.N, subsystem)
            tensor = shadow.shadow_data[i]
            trace_value = scalar(tensor * δ(shadow.site_indices[i], prime(shadow.site_indices[i])))
            trace_product *= trace_value
        end

        # Issue a warning if the total trace product deviates significantly from 1
        if !isapprox(trace_product, 1.0; atol=1e-10)
            @warn "The trace product of the traced-out tensors is not 1 (actual value: $trace_product)."
        end

        # Multiply the trace product into the subsystem tensors
        for i in eachindex(reduced_shadow_data)
            reduced_shadow_data[i] *= trace_product
        end

    end

    # Construct and return the reduced FactorizedShadow
    return FactorizedShadow(reduced_shadow_data, length(subsystem), reduced_ξ)
end


"""
    partial_transpose(shadow::FactorizedShadow, subsystem::Vector{Int})::FactorizedShadow

Compute the partial transpose of a FactorizedShadow over the specified subsystem by swapping, for each site,
the unprimed and primed indices using the `swapind` function. This function returns views of the underlying ITensors,
avoiding unnecessary data duplication.

# Arguments
- `shadow::FactorizedShadow`: The factorized classical shadow.
- `subsystem::Vector{Int}`: A vector of 1-based site indices on which to perform the partial transpose.

# Returns
A new FactorizedShadow with the specified sites partially transposed.
"""
function partial_transpose(shadow::FactorizedShadow, subsystem::Vector{Int})::FactorizedShadow
    @assert all(i -> i ≥ 1 && i ≤ shadow.N, subsystem) "Subsystem indices must be between 1 and N."
    @assert length(unique(subsystem)) == length(subsystem) "Subsystem indices must be unique."

    # Create a new vector for the ITensor views.
    new_shadow_data = copy(shadow.shadow_data)
    for i in subsystem
        a = shadow.site_indices[i]      # unprimed index for site i
        b = prime(a)         # its primed partner
        new_shadow_data[i] = swapind(new_shadow_data[i], a, b)
    end
    return FactorizedShadow(new_shadow_data, shadow.N, shadow.site_indices)
end

"""
    convert_to_dense_shadow(factorized_shadow::FactorizedShadow)

Convert a `FactorizedShadow` object into a `DenseShadow` object.

# Arguments
- `factorized_shadow::FactorizedShadow`: The `FactorizedShadow` object to convert.

# Returns
A `DenseShadow` object with the combined ITensor.
"""
function convert_to_dense_shadow(factorized_shadow::FactorizedShadow)::DenseShadow
    N = factorized_shadow.N
    ξ = factorized_shadow.site_indices

    # Start with the first shadow tensor
    dense_tensor = factorized_shadow.shadow_data[1]

    # Multiply all shadow tensors to combine them into a dense ITensor
    for i in 2:N
        dense_tensor *= factorized_shadow.shadow_data[i]
    end

    # Return a DenseShadow object
    return DenseShadow(dense_tensor, N, ξ)
end
