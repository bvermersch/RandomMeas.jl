"""
    MeasurementProbability(data::MeasurementData{T})

Construct a `MeasurementProbability` object from a `MeasurementData` object.

# Arguments
- `data::MeasurementData{T}`: A `MeasurementData` object containing binary measurement results and a measurement setting.

# Returns
A `MeasurementProbability` object with:
- `measurement_probability`: A Born probability tensor computed from the measurement results.
- `measurement_setting`: The same measurement setting as in the input `MeasurementData`.

# Details
- The Born probability is computed by mapping each unique measurement outcome (as obtained via `eachrow(data.measurement_results)`)
  to its count and then populating a dense tensor.
- The tensor is normalized by the total number of measurements (`NM`).
- If no measurement setting is provided (`data.measurement_setting` is `nothing`), a default set of site indices is generated.

# Example
```julia
# Generate random measurement data (NM = 100, N = 5)
measurement_results = rand(1:2, 100, 5)
setting = LocalUnitaryMeasurementSetting(5, ensemble="Haar")
data = MeasurementData(measurement_results, measurement_setting=setting)

# Construct MeasurementProbability from MeasurementData
prob_obj = MeasurementProbability(data)
println(prob_obj.measurement_probability[1])  # Print probability tensor for the first measurement
"""

function MeasurementProbability(data::MeasurementData{T}) where  T <: Union{Nothing, AbstractMeasurementSetting}
    N = data.N
    NM = data.NM
    if isnothing(data.measurement_setting)
        ξ  = siteinds("Qubit", N)
    else
        ξ = data.measurement_setting.site_indices
    end

    # Compute the count of each measurement outcome (each row in measurement_results).
    probf = StatsBase.countmap(eachrow(data.measurement_results))  # Dictionary: {state => count}

    # Create a dense N-dimensional tensor of size 2 in each dimension.to store Probability
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
    MeasurementProbability(ψ::Union{MPS, MPO}, setting::Union{ShallowUnitaryMeasurementSetting, ComputationalBasisMeasurementSetting, LocalUnitaryMeasurementSetting})

Construct a `MeasurementProbability` object from a quantum state (either an MPS or MPO) and measurement settings.

# Arguments
- `ψ::Union{MPS, MPO}`: The quantum state (pure if MPS, mixed if MPO) from which the Born probability is computed.
- `setting`: The measurement settings describing the local unitary operations (or shallow, or computational basis).

# Returns
A `MeasurementProbability` object with:
- `measurement_probability`: A Born probability tensor computed by applying the measurement settings to `ψ`.
- `measurement_setting`: The provided measurement setting.

# Details
- For an MPS, the local unitaries from `setting` are applied to `ψ`, and the Born probability is computed using a series of contractions.
- For an MPO, a similar procedure is followed with conjugation (apply_dag=true).
- The site indices are extracted from `ψ` using `get_siteinds(ψ)` and are asserted to match those in `setting`.

# Example
```julia
ψ = random_mps(siteinds("Qubit", 5))
settings = LocalUnitaryMeasurementSetting(5, ensemble="Haar")
prob_obj = MeasurementProbability(ψ, settings)
println(prob_obj.measurement_probability[1])
"""

function MeasurementProbability(ψ::Union{MPS, MPO}, setting::Union{ShallowUnitaryMeasurementSetting,ComputationalBasisMeasurementSetting,LocalUnitaryMeasurementSetting})
    N = length(ψ)
    ξ = get_siteinds(ψ)
    @assert ξ==setting.site_indices "ψ and setting must have the same site indices"

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
