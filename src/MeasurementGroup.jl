# using ITensors
# using NPZ
# include("MeasurementSetting.jl") #Todo: Remove this line when include in the pacakge
include("MeasurementData.jl")
"""
    struct MeasurementGroup{T}

A container for a group of NU measurement data used in quantum experiments.

# Fields
- `N::Int`: Number of sites (qubits).
- `NU::Int`: Number of MeasurementData objects
- `measurements::Vector{MeasurementData{T}}

# Type Parameter
- `T`: The type of `MeasurementGroup`. This can be any subtype of `AbstractMeasurementSetting` or `Nothing` if no settings are provided.

# Usage
The `MeasurementData` struct is typically constructed using the provided constructor functions.
"""
struct MeasurementGroup{T}
    N::Int                              # Number of sites (qubits)
    NU::Int                             # Number of measurementData
    NM::Int                             # Number of projectivemeasurements per RU
    measurements::Vector{MeasurementData{T}} # NU MeasurementsData objects
end

"""
    MeasurementGroup(
    measurements::Vector{MeasurementData{T}}
) where {T <: AbstractMeasurementSetting}

Creates a `MeasurementGroup` object by inferring the dimensions of the measurement results and validating the provided setting.

# Arguments
- measurements::Vector{MeasurementData{T}} a vector of MeasurementData objects of type T

# Returns
A `MeasurementGroup` object with inferred dimensions a


# Examples
```julia
#
setting1 = LocalUnitaryMeasurementSetting(4, ensemble="Haar")
results1 = rand(1:2, 10, 4)
data1 = MeasurementData(results; measurement_setting=setting)
setting2 = LocalUnitaryMeasurementSetting(4, ensemble="Haar")
results2 = rand(1:2, 10, 4)
data2 = MeasurementData(results; measurement_setting=setting)
measurements = [data1,data2]
group = MeasurementGroup(measurements)
```
"""
function MeasurementGroup(
    measurements::Vector{MeasurementData{T}}
) where {T <: AbstractMeasurementSetting}
    # Infer dimensions from measurements
    NU = length(measurements)
    N = measurements[1].N
    NM = measurements[1].NM

    # Delegate to the struct constructor
    return MeasurementGroup(N, NU, NM, measurements)
end
