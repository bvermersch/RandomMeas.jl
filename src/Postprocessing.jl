"""
    get_purity_direct(data::Array{Int}, subsystem::Vector{Int64}=1:N)

Extract the purity from the direct Hamming distance formula:
    purity = ``\\sum_s (-2)^{-D[s,s']}P(s)P(s')`` [Brydges et al, Science 2019]

# Arguments
- `data::Array{Int}`: The measurement data of shape (NU, NM, N) where NU is the number of realizations, NM is the number of shots, and N is the number of qubits.
- `subsystem::Vector{Int64}` (optional): A vector of integers specifying the subsystem of qubits to compute the purity for. Default is `1:N`, meaning all qubits.

# Returns
- The computed purity for the specified subsystem of qubits.

# Example
```julia
purity = get_purity_hamming(data, [1, 2])  # Compute purity for qubits 1 and 2
```
"""
function get_purity_direct(data::MeasurementData, subsystem::Vector{Int} = collect(1:data.N))

    # Ensure subsystem is a valid selection
    @assert all(x -> x >= 1 && x <= data.N, subsystem)
    @assert length(unique(subsystem)) == length(subsystem)  # Ensure no duplicates

    return sum(get_purity_direct_single_meas_setting(data.measurement_results[r, :, subsystem]) for r in 1:data.NU) / data.NU

end


"""
    get_purity_direct_single_meas_setting(data::Array{Int}) -> Float64

Compute the purity of a quantum state directly from measurement data using sequential tensor contractions.

This function calculates the purity of a quantum state represented by the given measurement data. It efficiently applies the Hamming tensor to each leg of the Born probability tensor sequentially, avoiding the construction of the full Hamming operator. This approach optimizes both time and memory usage.

# Arguments

- `data::Array{Int}`: A 2D array of measurement results with dimensions `(NM, N)`, where `NM` is the number of measurements, and `N` is the number of qubits. Each entry should be `1` or `2`, representing the binary outcomes for qubits.

# Returns

- `purity::Float64`: The computed purity of the quantum state.

# Example

```julia
using ITensors

# Generate random measurement data
NM = 1000  # Number of measurements
N = 4      # Number of qubits
data = rand(1:2, NM, N)

# Compute purity
purity = get_purity_direct_single_meas_setting(data)
println("Purity: ", purity)
```
"""
function get_purity_direct_single_meas_setting(data::Array{Int})
    # Extract dimensions: NM is the number of measurements, N is the number of qubits
    NM, N = size(data)

    # Generate site indices for qubits
    ξ = siteinds("Qubit", N)

    # Compute the Born probability tensor from measurement data
    prob = get_Born(data, ξ)

    # Compute the weighted overlap using the Hamming tensor
    purity = get_weighted_overlap(prob, prob)

    # Correct for the bias
    purity = purity * NM^2 / (NM * (NM - 1)) - 2.0^N / (NM - 1)

    # Return the computed purity
    return purity
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
