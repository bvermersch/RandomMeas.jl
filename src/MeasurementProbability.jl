"""
    MeasurementProbability(data::MeasurementData{T})

Construct a `MeasurementProbability` object from a `MeasurementData` object.

# Arguments
- `data::MeasurementData{T}`: A `MeasurementData` object containing binary measurement results and settings.

# Returns
A `MeasurementProbability` object with the following:
- `measurement_probability`:  Born probability tensor.
- `measurement_setting`: The same measurement setting as in the input `MeasurementData`.

# Details
- Computes Born Probability using `get_Born` for each measurement setting.
- The Probability are stored as an array of ITensor objects.

# Example
```julia
# Generate random measurement data
measurement_results = rand(1:2, 100, 5)
setting = LocalUnitaryMeasurementSetting(5, ensemble="Haar")
data = MeasurementData(measurement_results, measurement_setting=settings)

# Construct MeasurementProbability from MeasurementData
Probability = MeasurementProbability(data)
println(Probability.measurement_probability[1])  # Print Probability for the first setting
"""

function MeasurementProbability(data::MeasurementData{T}) where  T <: Union{Nothing, AbstractMeasurementSetting}
    N = data.N
    NM = data.NM
    if isnothing(data.measurement_setting)
        ξ  = siteinds("Qubit", N)
    else
        ξ = data.measurement_setting.site_indices
    end

    probf = StatsBase.countmap(eachrow(data.measurement_results))  # Dictionary: {state => count}

    # Initialize a dense tensor to store Probability
    prob = zeros(Int64, (2 * ones(Int, N))...)

    # Populate the tensor with counts from the dictionary
    for (state, val) in probf
        prob[state...] = val
    end

    # Normalize the tensor by the total number of measurements
    measurement_Probability = ITensor(prob, ξ) / NM # Compute Born Probability  #TODO: Check whether this is okay and gives a Probability MPS

    return MeasurementProbability(N, measurement_Probability, data.measurement_setting,ξ)

end

"""
MeasurementProbability(ψ::Union{MPS, MPO}, setting::Union{Nothing,LocalUnitaryMeasurementSetting}=nothing)

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

function MeasurementProbability(ψ::Union{MPS, MPO}, setting::Union{ComputationalBasisMeasurementSetting,LocalUnitaryMeasurementSetting})
    N = length(ψ)
    if typeof(ψ) == MPS
        ξ = siteinds(ψ)
    else
        ξ = firstsiteinds(ψ;plev=0)
    end

    if typeof(ψ) == MPS
        ψu = apply(setting.local_unitary, ψ)
        C = δ(ξ[1], ξ[1]',ξ[1]'')
        R = C * ψu[1] * conj(ψu[1]')
        R *= δ(ξ[1], ξ[1]'')
        P = R
        for i in 2:N
            Ct = δ(ξ[i], ξ[i]', ξ[i]'')
            Rt = ψu[i] * conj(ψu[i]') * Ct
            Rt *= δ(ξ[i], ξ[i]'')
            P *= Rt
        end
    else
        ρu = apply(setting.local_unitary,ψ; apply_dag=true)
        P = ρu[1] * δ(ξ[1],ξ[1]',ξ[1]'')
        P *= δ(ξ[1]'', ξ[1])
        for i in 2:N
            C = ρu[i] * delta(ξ[i], ξ[i]', ξ[i]'')
            C *= delta(ξ[i]'', ξ[i])
            P *= C
        end
    end
    return MeasurementProbability(N, P, setting,ξ)
end
