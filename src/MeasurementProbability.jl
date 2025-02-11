"""
    MeasurementProbability{T}

A container for measurement Probability and setting used in quantum experiments.

# Fields
- `N::Int`: Number of sites (qubits).
- `measurement_Probability::Array{MPS, 1}`: An array of Born Probability for each measurement setting.
- `measurement_settings::T`: Measurement settings of type `T` or `nothing` if not provided.

# Type Parameter
- `T`: The type of `measurement_settings`. This can be any subtype of `AbstractMeasurementSettings` or `Nothing` if no settings are provided.

# Usage
The `MeasurementProbability` struct can be constructed using either a `MeasurementData` object or directly from a quantum state (MPS/MPO) and measurement settings.
"""
struct MeasurementProbability{T}
    N::Int                              # Number of sites (qubits)
    measurement_probability::Array{ITensor, 1} # Measurement Probability (size: NU)
    measurement_setting::T             # Measurement settings (or nothing if not provided)
end

"""
    MeasurementProbability(data::MeasurementData{T})

Construct a `MeasurementProbability` object from a `MeasurementData` object.

# Arguments
- `data::MeasurementData{T}`: A `MeasurementData` object containing binary measurement results and settings.

# Returns
A `MeasurementProbability` object with the following:
- `measurement_probability`: A list of Born probability tensors (one for each measurement setting).
- `measurement_setting`: The same measurement setting as in the input `MeasurementData`.

# Details
- Computes Born Probability using `get_Born` for each measurement setting.
- The Probability are stored as an array of ITensor objects.

# Example
```julia
# Generate random measurement data
measurement_results = rand(1:2, 100, 5)
setting = LocalUnitaryMeasurementSetting(5, ensemble="Haar")
data = MeasurementData(measurement_results, measurement_settings=settings)

# Construct MeasurementProbability from MeasurementData
Probability = MeasurementProbability(data)
println(Probability.measurement_Probability[1])  # Print Probability for the first setting
"""

function MeasurementProbability(data::MeasurementData{T}) where {T <: AbstractMeasurementSettings}
    N = data.N
    ξ = data.measurement_setting.site_indices
    #measurement_Probability = Array{ITensor}(undef, NU)

    #measurement_Probability[r] = get_Born(data.measurement_results[r, :, :], ξ)
    probf = StatsBase.countmap(eachrow(data.measurement_results))  # Dictionary: {state => count}

    # Initialize a dense tensor to store Probability
    prob = zeros(Int64, (2 * ones(Int, N))...)

    # Populate the tensor with counts from the dictionary
    for (state, val) in probf
        prob[state...] = val
    end

    # Normalize the tensor by the total number of measurements
    measurement_Probability = ITensor(prob, ξ) / NM # Compute Born Probability  #TODO: Check whether this is okay and gives a Probability MPS

    return MeasurementProbability(N, measurement_Probability, data.measurement_setting)

end

"""
MeasurementProbability(ψ::Union{MPS, MPO}, settings::LocalUnitaryMeasurementSettings)

Construct a MeasurementProbability object from a quantum state (MPS/MPO) and measurement settings.

Arguments
	•	ψ::Union{MPS, MPO}: The quantum state (either pure state MPS or mixed state MPO) from which Probability are computed.
	•	settings::LocalUnitaryMeasurementSettings: Measurement settings describing the unitary operations and measurement configurations.

Returns

A MeasurementProbability object with the following:
	•	measurement_Probability: A list of Born probability tensors, one for each measurement setting.
	•	measurement_settings: The input measurement settings.

Details
	•	Applies the local unitaries from settings to the quantum state ψ.
	•	Computes Born Probability using get_Born for each measurement setting.

Example

# Generate a random MPS and measurement settings
ψ = random_mps(siteinds("Qubit", 5))
settings = LocalUnitaryMeasurementSettings(5, 10, ensemble="Haar")

# Construct MeasurementProbability from state and settings
Probability = MeasurementProbability(ψ, settings)
println(Probability.measurement_Probability[1])  # Print Probability for the first setting


Notes
	•	If ψ is an MPS, it assumes the state is pure.
	•	If ψ is an MPO, it assumes the state is mixed and applies unitaries with conjugation.
"""

function MeasurementProbability(ψ::Union{MPS, MPO}, setting::LocalUnitaryMeasurementSetting)
    N = settings.N
    local_unitaries = setting.local_unitaries
    #measurement_Probability = Array{ITensor}(undef, NU)
    if typeof(ψ) == MPS
        P = get_Born(apply(local_unitaries, ψ))  # Apply unitaries to MPS and compute Probability
    else
        P = get_Born(apply(local_unitaries, ψ; apply_dag=true))  # MPO version
    end
    measurement_Probability = P
    return MeasurementProbability(N, measurement_Probability, setting)
end
