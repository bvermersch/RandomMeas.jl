
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
    reduce_to_subsystem(
    group::MeasurementGroup{LocalUnitaryMeasurementSetting},
    subsystem::Vector{Int}
)::MeasurementGroup{LocalUnitaryMeasurementSetting}

Reduce a `MeasurementGroup object (with `LocalUnitaryMeasurementSetting`) to a specified subsystem.

# Arguments
- `group::MeasurementGroup{LocalUnitaryMeasurementSetting}`: The original measurement data object.
- `subsystem::Vector{Int}`: A vector of site indices (1-based) specifying the subsystem to retain.

# Returns
A new `MeasurementGroup` object corresponding to the specified subsystem.
"""
function reduce_to_subsystem(
    group::MeasurementGroup{LocalMeasurementSetting},
    subsystem::Vector{Int}
)::MeasurementGroup{LocalMeasurementSetting}
    # Validate the subsystem
    @assert all(x -> x >= 1 && x <= group.N, subsystem) "Subsystem indices must be between 1 and N."
    @assert length(unique(subsystem)) == length(subsystem) "Subsystem indices must be unique."

    # Reduce the measurement setting
    NU = group.NU
    reduced_measurements = Vector{MeasurementData{LocalUnitaryMeasurementSetting}}(undef,NU)
    for r in 1:NU
        reduced_measurements[r] = reduce_to_subsystem(group.measurements[r],subsystem)
    end
    # Create and return the new MeasurementData object
    return MeasurementGroup(reduced_measurements)
end
