
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
) where T <: Union{Nothing, AbstractMeasurementSetting}
    # Infer dimensions from measurements
    NU = length(measurements)
    N = measurements[1].N
    NM = measurements[1].NM

    # Delegate to the struct constructor
    return MeasurementGroup(N, NU, NM, measurements)
end

"""
    MeasurementGroup(
    ψ::Union{MPO, MPS},
    NU::Int
    NM::Int,
    mode::String = "MPS/MPO",
)::MeasurementGroup{LocalUnitaryMeasurementSetting}

Implements a MeasurementGroup from ψ based on generating NU LocalMeasurementSetting objects
"""
function MeasurementGroup(
    ψ::Union{MPO, MPS},
    NU::Int,
    NM::Int;
    mode::String = "MPS/MPO",
)::MeasurementGroup{LocalUnitaryMeasurementSetting}

    measurements = Vector{MeasurementData{LocalUnitaryMeasurementSetting}}(undef,NU)
    ξ = siteinds(ψ)
    N = length(ξ)
    for r in 1:NU
        measurement_setting = LocalUnitaryMeasurementSetting(N; site_indices=ξ,ensemble="Haar")
        measurements[r] = MeasurementData(ψ,NM,measurement_setting;mode="dense")
    end
    return MeasurementGroup(measurements)
end

"""
    reduce_to_subsystem(
    group::MeasurementGroup{T},
    subsystem::Vector{Int}
)::MeasurementGroup{T} where T <: LocalMeasurementSetting
    
Reduce a `MeasurementGroup object (with `LocalUnitaryMeasurementSetting`) to a specified subsystem.

# Arguments
- `group::MeasurementGroup{LocalUnitaryMeasurementSetting}`: The original measurement data object.
- `subsystem::Vector{Int}`: A vector of site indices (1-based) specifying the subsystem to retain.

# Returns
A new `MeasurementGroup` object corresponding to the specified subsystem.
"""
function reduce_to_subsystem(
    group::MeasurementGroup{T},
    subsystem::Vector{Int}
)::MeasurementGroup{T} where T <: Union{Nothing, LocalMeasurementSetting}
    @assert all(x -> x >= 1 && x <= group.N, subsystem) "Subsystem indices must be between 1 and N."
    @assert length(unique(subsystem)) == length(subsystem) "Subsystem indices must be unique."

    NU = group.NU
    reduced_measurements = Vector{MeasurementData{T}}(undef, NU)
    for r in 1:NU
        reduced_measurements[r] = reduce_to_subsystem(group.measurements[r], subsystem)
    end
    return MeasurementGroup(reduced_measurements)
end
