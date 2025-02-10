"""
    get_purity_direct(data::MeasurementData, subsystem::Vector{Int} = collect(1:data.N))

Compute the purity of a quantum state from measurement data by averaging the overlap of measurement results.

# Arguments
- `data::MeasurementData`: Measurement data containing the results and settings of randomized measurements.
- `subsystem::Vector{Int}` (optional): A vector of site indices specifying the subsystem to retain. Defaults to the full system.

# Returns
- The computed purity for the specified subsystem.
"""
function get_purity_direct(data::MeasurementData, subsystem::Vector{Int} = collect(1:data.N))
    return get_overlap_direct(data, data, subsystem; apply_bias_correction=true)
end

"""
    get_overlap_direct(
        data_1::MeasurementData,
        data_2::MeasurementData,
        subsystem::Vector{Int} = collect(1:data_1.N);
        apply_bias_correction::Bool = false
    )

Compute the overlap of two quantum states from measurement data by averaging the overlap of measurement results.

# Arguments
- `data_1::MeasurementData`: Measurement data for the first state.
- `data_2::MeasurementData`: Measurement data for the second state.
- `subsystem::Vector{Int}` (optional): A vector of site indices specifying the subsystem to retain. Defaults to the full system.
- `apply_bias_correction::Bool` (optional): Whether to apply bias correction for the overlap. Defaults to `false`.

# Returns
- The computed overlap (or purity if `data_1 == data_2` and bias correction is applied).
"""
function get_overlap_direct(
    data_1::MeasurementData,
    data_2::MeasurementData,
    subsystem::Vector{Int} = collect(1:data_1.N);
    apply_bias_correction::Bool = false
)
    # Ensure subsystem is a valid selection
    @assert all(x -> x >= 1 && x <= data_1.N, subsystem)
    @assert length(unique(subsystem)) == length(subsystem)  # Ensure no duplicates
    @assert data_1.NU == data_2.NU "Number of unitaries (NU) must match between data_1 and data_2."
    @assert data_1.N == data_2.N "Number of qubits (N) must match between data_1 and data_2."

    # Compute overlap averaged over all random unitaries (measurement settings)
    overlap = sum(get_overlap_direct_single_meas_setting(
        data_1.measurement_results[r, :, subsystem],
        data_2.measurement_results[r, :, subsystem]
    ) for r in 1:data_1.NU) / data_1.NU

    if apply_bias_correction
        @assert data_1.NM == data_2.NM "Number of measurements (NM) must match for bias correction."
        NM = data_1.NM
        N = length(subsystem)
        overlap = overlap * NM^2 / (NM * (NM - 1)) - 2.0^N / (NM - 1)
    end

    return overlap
end

"""
    get_overlap_direct_single_meas_setting(
        data_1::Array{Int},
        data_2::Array{Int}
    )

Compute the overlap between two quantum states for a single measurement setting.

# Arguments
- `data_1::Array{Int}`: Measurement results for the first state, with dimensions `(NM, N)`.
- `data_2::Array{Int}`: Measurement results for the second state, with dimensions `(NM, N)`.

# Returns
- The computed overlap for the single measurement setting.
"""
function get_overlap_direct_single_meas_setting(data_1::Array{Int}, data_2::Array{Int})
    # Extract the number of qubits N is the number of qubits
    N = size(data_1,2)
    @assert N==size(data_2,2) "Number of qubits must match between data_1 and data_2."

    # Generate site indices for qubits
    ξ = siteinds("Qubit", N)

    # TODO: The construction of the Born probability tensors is not strictly necessary. One could di
    # directly compute the weighted overlap using the Hamming tensor and the measurement data.
    # This would be slower but memory efficient. Should we provide this option?
    # It is similar to "dense" (batch) shadows vs "factorized shadows" in the shadows module.

    # Compute the Born probability tensors from measurement data
    prob_1 = get_Born(data_1, ξ)
    prob_2 = data_1 === data_2 || data_1 == data_2 ? prob_1 : get_Born(data_2, ξ)

    # Compute the weighted overlap using the Hamming tensor
    overlap = get_weighted_overlap(prob_1, prob_2)

    return overlap
end


"""
    get_Born(data::Array{Int}, ξ::Vector{Index{Int64}}) -> ITensor

Compute the Born probabilities from binary measurement results.

This function processes a binary dataset of measurement outcomes and computes
the corresponding Born probability tensor, normalized by the total number of measurements.

# Arguments:
- `data::Array{Int}`: A 2D array of measurement results of size `(NM, N)`,
  where `NM` is the number of measurements, and `N` is the number of sites (qubits).
  Each entry should be `1` or `2`, representing the binary outcomes for qubits.
- `ξ::Vector{Index{Int64}}`: A vector of site indices corresponding to the qubits.

# Returns:
- An `ITensor` object representing the Born probability tensor normalized
  over all measurement outcomes.

# Example:
```julia
NM, N = 1000, 4
data = rand(1:2, NM, N)  # Generate random binary outcomes
ξ = siteinds("Qubit", N)  # Generate site indices
prob_tensor = get_Born(data, ξ)
println(prob_tensor)
```
"""
function get_Born(data::Array{Int}, ξ::Vector{Index{Int64}})
    # Get dimensions: NM is the number of measurements, N is the number of sites
    NM, N = size(data)
    # Count occurrences of each unique binary state in the dataset
    probf = StatsBase.countmap(eachrow(data))  # Dictionary: {state => count}

    # Initialize a dense tensor to store probabilities
    prob = zeros(Int64, (2 * ones(Int, N))...)

    # Populate the tensor with counts from the dictionary
    for (state, val) in probf
        prob[state...] = val
    end

    # Normalize the tensor by the total number of measurements
    probT = ITensor(prob, ξ) / NM
    return probT
end


"""
    get_weighted_overlap(prob1::ITensor, prob2::ITensor, ξ::Vector{Index{Int64}}, N::Int64) -> Float64

Compute the weighted overlap  `\\2^N sum_s (-2)^{-D[s,s']}P(s)P(s')]` by sequentially applying the Hamming tensor to each qubit index and contracting with the second probability tensor.

# Arguments

- `prob1::ITensor`: The first Born probability tensor representing quantum state `rho1`.
- `prob2::ITensor`: The second Born probability tensor representing quantum state `rho2`.

# Returns

- `weighted_overlap::Float64`: The computed trace `Tr(rho1 rho2)` scaled appropriately..

# Example

```julia
using ITensors

# Assume prob1 and prob2 are predefined ITensors, ξ is the site indices vector, and N is the number of qubits
weighted_overlap = get_weighted_overlap(prob1, prob2, ξ, N)
println("Weighted Overlap: ", weighted_overlap)
```
"""
function get_weighted_overlap(prob1::ITensor, prob2::ITensor)

    # Extract the number of qubits from the site indices
    @assert inds(prob2) == inds(prob1)  # Ensure site indices match
    ξ = inds(prob1)
    N = length(ξ)

    # Initialize trace_temp with prob1
    weighted_overlap = prob1

    # Sequentially apply the Hamming tensor to each qubit index
    for i in 1:N
        weighted_overlap *= get_h_tensor(ξ[i], prime(ξ[i]))
    end

    # Contract the resulting tensor with the Hermitian conjugate of prob2 (prob2')
    weighted_overlap *= prob2'

    # Extract the scalar value from the final tensor and scale by 2^N
    return real(scalar(weighted_overlap)) * 2.0^N
end


"""
    get_h_tensor(s::Index, s_prime::Index) -> ITensor

Construct the Hamming tensor for given indices.

# Arguments

- `s::Index`: Unprimed site index.
- `s_prime::Index`: Primed site index.

# Returns

- `Hamming_tensor::ITensor`: The Hamming tensor connecting `s` and `s_prime`.

# Method

- Initializes an `ITensor` with indices `s` and `s_prime`.
- Assigns values to represent the Hamming distance operation:
  - Diagonal elements are set to `1.0`.
  - Off-diagonal elements are set to `-0.5`.

"""
function get_h_tensor(s::Index, s_prime::Index, G::Float64 = 1.0)
    Hamming_tensor = ITensor(Float64, s, s_prime)

    # Compute α and β for the Hamming matrix
    α = 3.0 / (2.0 * G - 1.0)
    β = (G - 2.0) / (2.0 * G - 1.0)

    # Populate the Hamming tensor
    Hamming_tensor[s => 1, s_prime => 1] = (α + β) / 2.0
    Hamming_tensor[s => 2, s_prime => 2] = (α + β) / 2.0
    Hamming_tensor[s => 1, s_prime => 2] = β / 2.0
    Hamming_tensor[s => 2, s_prime => 1] = β / 2.0

    return Hamming_tensor
end


"""
    get_XEB(ψ::MPS, measurement_data::MeasurementData{LocalUnitaryMeasurementSettings})

Return the linear cross-entropy for the measurement results in `measurement_data`, with respect to a theory state `ψ`.

# Arguments:
- `ψ::MPS`: The theoretical state to compare against.
- `measurement_data::MeasurementData{LocalUnitaryMeasurementSettings}`: The measurement data object containing results and settings.

# Returns:
The linear cross-entropy as a `Float64`.
"""
function get_XEB(ψ::MPS, measurement_data::MeasurementData{LocalUnitaryMeasurementSettings})
    # Extract site indices and measurement results
    ξ = siteinds(ψ)
    data = measurement_data.measurement_results
    NU, NM, N = measurement_data.NU, measurement_data.NM, measurement_data.N  # Extract number of measurement settings (NU), measurements per settings (NM) and qubits/sites (N)
    @assert NU == 1 "Only one computational basis measurements are supported for XEB."

    P0 = get_Born_MPS(ψ)  # Compute theoretical Born probabilities

    # Initialize XEB value
    XEB = 0.0

    # Loop through the measurement results
    for m in 1:NM
        V = ITensor(1.0)
        for j in 1:N
            V *= (P0[j] * state(ξ[j], data[1, m, j]))
        end
        XEB += 2^N / NM * real(V[]) - 1 / NM
    end

    return XEB
end
