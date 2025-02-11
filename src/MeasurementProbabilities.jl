"""
    MeasurementProbabilities{T}

A container for measurement probabilities and settings used in quantum experiments.

# Fields
- `N::Int`: Number of sites (qubits).
- `NU::Int`: Number of measurement settings.
- `measurement_probabilities::Array{MPS, 1}`: An array of Born probabilities for each measurement setting.
- `measurement_settings::T`: Measurement settings of type `T` or `nothing` if not provided.

# Type Parameter
- `T`: The type of `measurement_settings`. This can be any subtype of `AbstractMeasurementSettings` or `Nothing` if no settings are provided.

# Usage
The `MeasurementProbabilities` struct can be constructed using either a `MeasurementData` object or directly from a quantum state (MPS/MPO) and measurement settings.
"""
struct MeasurementProbabilities{T}
    N::Int                              # Number of sites (qubits)
    NU::Int                             # Number of measurement settings
    measurement_probabilities::Array{ITensor, 1} # Measurement probabilities (size: NU)
    measurement_settings::T             # Measurement settings (or nothing if not provided)
end

"""
    MeasurementProbabilities(data::MeasurementData{T})

Construct a `MeasurementProbabilities` object from a `MeasurementData` object.

# Arguments
- `data::MeasurementData{T}`: A `MeasurementData` object containing binary measurement results and settings.

# Returns
A `MeasurementProbabilities` object with the following:
- `measurement_probabilities`: A list of Born probability tensors (one for each measurement setting).
- `measurement_settings`: The same measurement settings as in the input `MeasurementData`.

# Details
- Computes Born probabilities using `get_Born` for each measurement setting.
- The probabilities are stored as an array of ITensor objects.

# Example
```julia
# Generate random measurement data
measurement_results = rand(1:2, 10, 100, 5)
settings = LocalUnitaryMeasurementSettings(5, 10, ensemble="Haar")
data = MeasurementData(measurement_results, measurement_settings=settings)

# Construct MeasurementProbabilities from MeasurementData
probabilities = MeasurementProbabilities(data)
println(probabilities.measurement_probabilities[1])  # Print probabilities for the first setting
"""

function MeasurementProbabilities(data::MeasurementData{T}) where {T <: AbstractMeasurementSettings}
    N = data.N
    NU = data.NU
    ξ = data.measurement_settings.site_indices
    measurement_probabilities = Array{ITensor}(undef, NU)

    for r in 1:NU
        #measurement_probabilities[r] = get_Born(data.measurement_results[r, :, :], ξ)
        probf = StatsBase.countmap(eachrow(data.measurement_results[r,:,:]))  # Dictionary: {state => count}

        # Initialize a dense tensor to store probabilities
        prob = zeros(Int64, (2 * ones(Int, N))...)
    
        # Populate the tensor with counts from the dictionary
        for (state, val) in probf
            prob[state...] = val
        end
    
        # Normalize the tensor by the total number of measurements
        measurement_probabilities[r]= ITensor(prob, ξ) / NM # Compute Born probabilities  #TODO: Check whether this is okay and gives a Probability MPS
    end

    return MeasurementProbabilities(N, NU, measurement_probabilities, data.measurement_settings)

end

"""
MeasurementProbabilities(ψ::Union{MPS, MPO}, settings::LocalUnitaryMeasurementSettings)

Construct a MeasurementProbabilities object from a quantum state (MPS/MPO) and measurement settings.

Arguments
	•	ψ::Union{MPS, MPO}: The quantum state (either pure state MPS or mixed state MPO) from which probabilities are computed.
	•	settings::LocalUnitaryMeasurementSettings: Measurement settings describing the unitary operations and measurement configurations.

Returns

A MeasurementProbabilities object with the following:
	•	measurement_probabilities: A list of Born probability tensors, one for each measurement setting.
	•	measurement_settings: The input measurement settings.

Details
	•	Applies the local unitaries from settings to the quantum state ψ.
	•	Computes Born probabilities using get_Born for each measurement setting.

Example

# Generate a random MPS and measurement settings
ψ = random_mps(siteinds("Qubit", 5))
settings = LocalUnitaryMeasurementSettings(5, 10, ensemble="Haar")

# Construct MeasurementProbabilities from state and settings
probabilities = MeasurementProbabilities(ψ, settings)
println(probabilities.measurement_probabilities[1])  # Print probabilities for the first setting


Notes
	•	If ψ is an MPS, it assumes the state is pure.
	•	If ψ is an MPO, it assumes the state is mixed and applies unitaries with conjugation.
"""

function MeasurementProbabilities(ψ::Union{MPS, MPO}, settings::LocalUnitaryMeasurementSettings)
    N = settings.N
    NU = settings.NU
    local_unitaries = settings.local_unitaries
    measurement_probabilities = Array{ITensor}(undef, NU)
    for r in 1:NU
        if typeof(ψ) == MPS
            P = get_Born(apply(local_unitaries[r, :], ψ))  # Apply unitaries to MPS and compute probabilities
        else
            P = get_Born(apply(local_unitaries[r, :], ψ; apply_dag=true))  # MPO version
        end
        measurement_probabilities[r] = P
    end
    return MeasurementProbabilities(N, NU, measurement_probabilities, settings)
end
