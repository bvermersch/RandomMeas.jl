"""
    MeasurementGroup(measurements::Vector{MeasurementData{T}}) where {T <: Union{Nothing, AbstractMeasurementSetting}}

Construct a `MeasurementGroup` object by inferring dimensions from a vector of `MeasurementData` objects.

# Arguments
- `measurements::Vector{MeasurementData{T}}`: A vector of `MeasurementData` objects.

# Returns
A `MeasurementGroup` object with:
- `N`: Inferred from the first element (assumed consistent across all elements).
- `NU`: Number of measurement data objects.
- `NM`: Inferred from the first element.
- `measurements`: The provided vector.

# Example
```julia
setting1 = LocalUnitaryMeasurementSetting(4, ensemble="Haar")
results1 = rand(1:2, 10, 4)
data1 = MeasurementData(results1; measurement_setting=setting1)
setting2 = LocalUnitaryMeasurementSetting(4, ensemble="Haar")
results2 = rand(1:2, 10, 4)
data2 = MeasurementData(results2; measurement_setting=setting2)
group = MeasurementGroup([data1, data2])
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
MeasurementGroup(ψ::Union{MPO, MPS}, NU::Int, NM::Int; mode::String = “MPS/MPO”, progress_bar::Bool=false)
::MeasurementGroup{LocalUnitaryMeasurementSetting}

Construct a MeasurementGroup from a quantum state ψ by generating NU local measurement settings and simulating NM
projective measurements per setting.

# Arguments
	-  ψ::Union{MPO, MPS}: The quantum state.
	-	NU::Int: Number of measurement data objects to generate.
	-	NM::Int: Number of measurements per setting.
	-	mode::String: Simulation mode; defaults to “MPS/MPO”.
	-	progress_bar::Bool: Whether to show a progress bar.

# Returns
A MeasurementGroup{LocalUnitaryMeasurementSetting} object.

"""
function MeasurementGroup(
    ψ::Union{MPO, MPS},
    NU::Int,
    NM::Int;
    mode::String = "MPS/MPO",
    progress_bar::Bool=false
)::MeasurementGroup{LocalUnitaryMeasurementSetting}
    ξ = get_siteinds(ψ)
    measurements = Vector{MeasurementData{LocalUnitaryMeasurementSetting}}(undef,NU)
    ξ = get_siteinds(ψ)
    N = length(ξ)
    if progress_bar==true
        @showprogress dt=1 for r in 1:NU
            measurement_setting = LocalUnitaryMeasurementSetting(N; site_indices=ξ,ensemble="Haar")
            measurements[r] = MeasurementData(ψ,NM,measurement_setting;mode=mode)
        end
    else
        for r in 1:NU
            measurement_setting = LocalUnitaryMeasurementSetting(N; site_indices=ξ,ensemble="Haar")
            measurements[r] = MeasurementData(ψ,NM,measurement_setting;mode=mode)
        end
    end
    return MeasurementGroup(measurements)
end

"""
MeasurementGroup(ψ::Union{MPO, MPS}, NU::Int, NM::Int, depth::Int; mode::String = “MPS/MPO”, progress_bar::Bool=false)
::MeasurementGroup{ShallowUnitaryMeasurementSetting}

Construct a MeasurementGroup from a quantum state ψ by generating NU shallow measurement settings and simulating
NM measurements per unitary.

# Arguments
	-  ψ::Union{MPO, MPS}: The quantum state.
	-	NU::Int: Number of measurement data objects to generate.
	-	NM::Int: Number of measurements per setting.
    -   depth::Int: Circuit depth for shallow settings.
	-	mode::String: Simulation mode; defaults to “MPS/MPO”.
	-	progress_bar::Bool: Whether to show a progress bar.
# Returns

A MeasurementGroup{ShallowUnitaryMeasurementSetting} object.

"""
function MeasurementGroup(
    ψ::Union{MPO, MPS},
    NU::Int,
    NM::Int,
    depth::Int;
    mode::String = "MPS/MPO",
    progress_bar::Bool=false
)::MeasurementGroup{ShallowUnitaryMeasurementSetting}
    ξ = get_siteinds(ψ)
    measurements = Vector{MeasurementData{ShallowUnitaryMeasurementSetting}}(undef,NU)
    ξ = get_siteinds(ψ)
    N = length(ξ)
    if progress_bar==true
        @showprogress dt=1 for r in 1:NU
            measurement_setting = ShallowUnitaryMeasurementSetting(N,depth; site_indices=ξ)
            measurements[r] = MeasurementData(ψ,NM,measurement_setting;mode=mode)
        end
    else
        for r in 1:NU
            measurement_setting = ShallowUnitaryMeasurementSetting(N,depth; site_indices=ξ)
            measurements[r] = MeasurementData(ψ,NM,measurement_setting;mode=mode)
        end
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
