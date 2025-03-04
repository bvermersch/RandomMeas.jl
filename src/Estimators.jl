# Copyright (c) 2024 Benoît Vermersch and Andreas Elben 
# SPDX-License-Identifier: Apache-2.0
# http://www.apache.org/licenses/LICENSE-2.0

"""
    get_fidelity(
        group_1::MeasurementGroup,
        group_2::MeasurementGroup,
        subsystem::Vector{Int} = collect(1:group_1.N)
    )

Compute the fidelity of two quantum states Tr(ρ1 ρ2)/SROOT(Tr(ρ1^2),Tr(ρ2^2)) from measurement data by averaging the overlap of measurement results.

# Arguments
- `group_1::MeasurementGroup`: Measurement data for the first state.
- `group_2::MeasurementGroup`: Measurement data for the second state.
- `subsystem::Vector{Int}` (optional): A vector of site indices specifying the subsystem to retain. Defaults to the full system.

# Returns
- The computed fidelity.
"""
function get_fidelity(
    group_1::MeasurementGroup,
    group_2::MeasurementGroup,
    subsystem::Vector{Int} = collect(1:group_1.N);
)
    overlap = get_overlap(group_1,group_2,subsystem)
    purity1 = get_purity(group_1,subsystem)
    purity2 = get_purity(group_2,subsystem)
    return overlap/sqrt(purity1*purity2)
end


"""
    get_purity(group::Measurementgroup, subsystem::Vector{Int} = collect(1:group.N))

Compute the purity of a quantum state from measurement data by averaging the overlap of measurement results.

# Arguments
- `group::MeasurementGroup`: Measurement data containing the results and settings of randomized measurements.
- `subsystem::Vector{Int}` (optional): A vector of site indices specifying the subsystem to retain. Defaults to the full system.

# Returns
- The computed purity for the specified subsystem.
"""
function get_purity(group::MeasurementGroup, subsystem::Vector{Int} = collect(1:group.N))
    return get_overlap(group, group, subsystem; apply_bias_correction=true)
end

"""
    get_overlap(
        group_1::MeasurementGroup,
        group_2::MeasurementGroup,
        subsystem::Vector{Int} = collect(1:group_1.N);
        apply_bias_correction::Bool = false
    )

Compute the overlap of two quantum states from measurement data by averaging the overlap of measurement results.

# Arguments
- `group_1::MeasurementGroup`: Measurement data for the first state.
- `group_2::MeasurementGroup`: Measurement data for the second state.
- `subsystem::Vector{Int}` (optional): A vector of site indices specifying the subsystem to retain. Defaults to the full system.
- `apply_bias_correction::Bool` (optional): Whether to apply bias correction for the overlap. Defaults to `false`.

# Returns
- The computed overlap (or purity if `group_1 == group_2` and bias correction is applied).
"""
function get_overlap(
    group_1::MeasurementGroup,
    group_2::MeasurementGroup,
    subsystem::Vector{Int} = collect(1:group_1.N);
    apply_bias_correction::Bool = false
)
    # Ensure subsystem is a valid selection
    @assert all(x -> x >= 1 && x <= group_1.N, subsystem)
    @assert length(unique(subsystem)) == length(subsystem)  # Ensure no duplicates
    @assert group_1.NU == group_2.NU "Number of unitaries (NU) must match between group_1 and group_2."
    @assert group_1.N == group_2.N "Number of qubits (N) must match between group_1 and group_2."


    # Compute overlap averaged over all random unitaries (measurement settings)
    group_1_subsystem = reduce_to_subsystem(group_1,subsystem)
    group_2_subsystem = reduce_to_subsystem(group_2,subsystem)

    overlap = sum(get_overlap(
        group_1_subsystem.measurements[r],
        group_2_subsystem.measurements[r];apply_bias_correction=apply_bias_correction
    ) for r in 1:group_1.NU) / group_1.NU



    return overlap
end

"""
    get_overlap(
        data_1::MeasurementData,
        data_2::MeasurementData
    )

Compute the overlap between two quantum states for a single measurement setting.

# Arguments
- `data_1::MeasurementData`: Measurement Data for the first state, with dimensions `(NM, N)`.
- `data_2::MeasurementData`: Measurement Data for the second state, with dimensions `(NM, N)`.

# Returns
- The computed overlap for the single measurement setting.
"""
function get_overlap(data_1::MeasurementData,data_2::MeasurementData;
    apply_bias_correction::Bool = false)
    # Extract the number of qubits N is the number of qubits
    N = data_1.N
    @assert N==data_2.N "Number of qubits must match between group_1 and group_2."

    # TODO: The construction of the Born probability tensors is not strictly necessary. One could di
    # directly compute the weighted overlap using the Hamming tensor and the measurement data.
    # This would be slower but memory efficient. Should we provide this option?
    # It is similar to "dense" (batch) shadows vs "factorized shadows" in the shadows module.

    prob_1 = MeasurementProbability(data_1)
    prob_2 = MeasurementProbability(data_2)

    # Compute the weighted overlap using the Hamming tensor
    overlap = get_overlap(prob_1, prob_2)

    if apply_bias_correction
        @assert data_1.NM == data_2.NM "Number of measurements (NM) must match for bias correction."
        NM = data_1.NM
        overlap = overlap * NM^2 / (NM * (NM - 1)) - 2.0^N / (NM - 1)
    end

    return overlap
end

"""
    get_overlap(prob1::MeasurementProbability, prob2::MeasurementProbability) -> Float64

Compute the weighted overlap  `\\2^N sum_s (-2)^{-D[s,s']}P(s)P(s')]` by sequentially applying the Hamming tensor to each qubit index and contracting with the second probability tensor.

# Arguments

- `prob1::MeasurementProbability`: The first Born probability tensor representing quantum state `rho1`.
- `prob2::MeasurementProbability`: The second Born probability tensor representing quantum state `rho2`.

# Returns

- `weighted_overlap::Float64`: The computed trace `Tr(rho1 rho2)` scaled appropriately..

# Example

```julia
using ITensors

# Assume prob1 and prob2 are predefined MeasurementProbabilities
overlap = get_overlap(prob1, prob2)
println("Overlap: ", overlap)
```
"""
function get_overlap(prob1::MeasurementProbability, prob2::MeasurementProbability)

    # Extract the number of qubits from the site indices
    ξ1 = prob1.site_indices
    ξ2 = prob2.site_indices

    N = prob1.N
    @assert N==prob2.N "Number of qubits must match"

    # Initialize trace_temp with prob1
    overlap = prob1.measurement_probability
    # Sequentially apply the Hamming tensor to each qubit index
    for i in 1:N
        overlap *= get_h_tensor(ξ1[i], prime(ξ2[i]))
    end

    # Contract the resulting tensor with the Hermitian conjugate of prob2 (prob2')
    overlap = real(scalar(overlap*prob2.measurement_probability'))*2.0^N

    # Extract the scalar value from the final tensor and scale by 2^N
    return overlap
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
    get_XEB(ψ::MPS, measurement_data::MeasurementData)

Return the linear cross-entropy for the measurement results in `measurement_data`, with respect to a theory state `ψ`.

# Arguments:
- `ψ::MPS`: The theoretical state to compare against.
- `measurement_data::MeasurementData`: The measurement data object containing results and settings.

# Returns:
The linear cross-entropy as a `Float64`.
"""
function get_XEB(ψ::MPS, measurement_data::MeasurementData)
    # Extract site indices and measurement results
    ξ = siteinds(ψ)
    data = measurement_data.measurement_results
    NM, N = measurement_data.NM, measurement_data.N  # Extract number of measurement settings (NU), measurements per settings (NM) and qubits/sites (N)

    P0 = get_Born_MPS(ψ)  # Compute theoretical Born probabilities
    # Initialize XEB value
    XEB = 0.0

    # Loop through the measurement results
    for m in 1:NM
        V = ITensor(1.0)
        for j in 1:N
            V *= (P0[j] * state(ξ[j], data[m, j]))
        end
        XEB += 2^N / NM * real(V[]) - 1 / NM
    end

    return XEB
end
