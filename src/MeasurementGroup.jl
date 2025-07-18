# Copyright (c) 2024 Benoît Vermersch and Andreas Elben
# SPDX-License-Identifier: Apache-2.0
# http://www.apache.org/licenses/LICENSE-2.0

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
) where T<:Union{AbstractMeasurementSetting, Nothing}
    # Infer dimensions from measurements
    NU = length(measurements)
    N = measurements[1].N
    NM = measurements[1].NM

    # Delegate to the struct constructor
    return MeasurementGroup(N, NU, NM, measurements)
end

"""
    MeasurementGroup(ψ::Union{MPO, MPS}, measurement_settings::Vector{AbstractMeasurementSetting}, NM::Int; mode::String = “MPS/MPO”, progress_bar::Bool=false)
    ::MeasurementGroup{T} where T <: AbstractMeasurementSetting

Construct a MeasurementGroup from a quantum state `ψ` by generating `NU` local measurement settings and simulating `NM`
projective measurements per setting.

# Arguments
- `ψ::Union{MPO, MPS}`: The quantum state.
- `measurement_settings::Vector{AbstractMeasurementSetting}`: A vector with measurement settings
- `NM::Int`: Number of measurements per setting.
- `mode::String`: Simulation mode; defaults to “MPS/MPO”.
- `progress_bar::Bool`: Whether to show a progress bar.

# Returns
A MeasurementGroup{T} object.

"""
function MeasurementGroup(
    ψ::Union{MPO, MPS},
    measurement_settings::Vector{T},
    NM::Int;
    mode::String = "MPS/MPO",
    progress_bar::Bool=false
)::MeasurementGroup{T} where T <: AbstractMeasurementSetting
    NU = length(measurement_settings)
    measurements = Vector{MeasurementData{T}}(undef,NU)
    if progress_bar==true
        @showprogress dt=1 for r in 1:NU
            measurements[r] = MeasurementData(ψ,NM,measurement_settings[r];mode=mode)
        end
    else
        for r in 1:NU
            measurements[r] = MeasurementData(ψ,NM,measurement_settings[r];mode=mode)
        end
    end
    return MeasurementGroup(measurements)
end

"""
    MeasurementGroup(ψ::Union{MPO, MPS}, NU::Int, NM::Int; mode::String = “MPS/MPO”, progress_bar::Bool=false)
::MeasurementGroup{LocalUnitaryMeasurementSetting}

Construct a MeasurementGroup from a quantum state `ψ` by generating `NU` local measurement settings and simulating `NM`
projective measurements per setting.

# Arguments
- `ψ::Union{MPO, MPS}`: The quantum state.
- `NU::Int`: Number of measurement data objects to generate.
- `NM::Int`: Number of measurements per setting.
- `mode::String`: Simulation mode; defaults to “MPS/MPO”.
- `progress_bar::Bool`: Whether to show a progress bar.

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

Construct a MeasurementGroup from a quantum state `ψ` by generating `NU` shallow measurement settings and simulating
`NM` measurements per unitary.

# Arguments
- `ψ::Union{MPO, MPS}`: The quantum state.
- `NU::Int`: Number of measurement data objects to generate.
- `NM::Int`: Number of measurements per setting.
- `depth::Int`: Circuit depth for shallow settings.
- `mode::String`: Simulation mode; defaults to “MPS/MPO”.
- `progress_bar`::Bool: Whether to show a progress bar.
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


"""
    export_MeasurementGroup(group::MeasurementGroup{T}, filepath::String)

Export a MeasurementGroup object to an NPZ file.

# Arguments
- `group::MeasurementGroup{T}`: A MeasurementGroup object where each MeasurementData may have its own measurement setting of type T (with T <: Union{Nothing, LocalUnitaryMeasurementSetting, ComputationalBasisMeasurementSetting}).
- `filepath::String`: The file path where the NPZ file will be written.

# Details
- The measurement results from each MeasurementData object (each of shape (NM, N)) are stacked into a 3D array of shape (NU, NM, N),
  where NU is the number of MeasurementData objects.
- The measurement settings are exported as a Complex array of size NU x N x 2 x 2 if present.
"""
function export_MeasurementGroup(group::MeasurementGroup{T}, filepath::String) where T<:Union{Nothing, LocalUnitaryMeasurementSetting, ComputationalBasisMeasurementSetting}
    N  = group.N
    NU = group.NU
    NM = group.NM

    # Stack measurement results into a 3D array: (NU, NM, N)
    results_array = zeros(Int, NU, NM, N)
    for i in 1:NU
        results_array[i, :, :] = group.measurements[i].measurement_results
    end

    export_dict = Dict{String,Any}()
    export_dict["N"] = N
    export_dict["NU"] = NU
    export_dict["NM"] = NM
    export_dict["measurement_results"] = results_array

   # Only export measurement settings if they are not nothing.
    if T !== Nothing
        # Preallocate a 4D array of dimensions (NU, N, 2, 2) to hold the local unitaries.
        settings_array = Array{ComplexF64}(undef, NU, N, 2, 2)
        for i in 1:NU
            ms = group.measurements[i].measurement_setting
            for j in 1:N
                settings_array[i, j, :, :] = Array(ms.local_unitary[j], ms.site_indices[j]', ms.site_indices[j])
            end
        end
        export_dict["measurement_settings"] = settings_array
    end


    npzwrite(filepath, export_dict)
end


"""
    import_MeasurementGroup(filepath::String; predefined_settings=nothing, site_indices=nothing) -> MeasurementGroup

Import a MeasurementGroup object from an NPZ file.

# Arguments
- `filepath::String`: The path to the NPZ file containing the exported MeasurementGroup data.
- `predefined_settings` (optional): A vector of predefined measurement settings (one per MeasurementData object). If provided, its length must equal the exported NU.
- `site_indices` (optional): A vector of N site indices to use when reconstructing the measurement setting. If not provided, default site indices are generated using `siteinds("Qubit", N)`.

# Returns
A MeasurementGroup object with:
- Measurement results reconstructed from a 3D array of shape (NU, NM, N).
- A measurement setting for each MeasurementData object reconstructed from a 4D array of shape (NU, N, 2, 2) if present, or taken from `predefined_settings` if provided.
"""
function import_MeasurementGroup(filepath::String; predefined_settings=nothing, site_indices=nothing)
    data = npzread(filepath)

    measurement_results = data["measurement_results"]  # Expected shape: (NU, NM, N)
    NU, _, N = size(measurement_results)

    # Check if 0 is contained and print a message if true
    if 0 in measurement_results
        @warn "Julia works with indices starting at 1. Binary data should therefore use 1 and 2, not 0 and 1."
    end

    # If a vector of predefined settings is provided, check its length and ensure consistency.
    local T
    if predefined_settings !== nothing
        @assert length(predefined_settings) == NU "Expected predefined_settings vector to have length NU = $NU, got $(length(predefined_settings))."
        T = typeof(predefined_settings[1])
        for s in predefined_settings
            @assert typeof(s) == T "Predefined settings must all have the same type; found $(typeof(s)) vs $(T)."
        end
    elseif haskey(data, "measurement_settings")
        # Assume settings from file are of LocalUnitaryMeasurementSetting type.
        T = LocalUnitaryMeasurementSetting
    else
        T = Nothing
    end

    # Reconstruct MeasurementData objects.
    measurements = Vector{MeasurementData{T}}(undef, NU)
    for i in 1:NU
        # Extract measurement results for this MeasurementData (shape: (NM, N))
        m_results = measurement_results[i, :, :]
        if predefined_settings !== nothing
            # Use the corresponding predefined setting.
            ms = predefined_settings[i]
        elseif haskey(data, "measurement_settings") 
            if site_indices === nothing
                site_indices = siteinds("Qubit", N)
            end
            # Reconstruct from exported settings: assume settings_array is a 4D array (NU, N, 2, 2)
            local_unitaries = [ITensor(data["measurement_settings"][i, j, :, :], site_indices[j]', site_indices[j]) for j in 1:N]
            ms = LocalUnitaryMeasurementSetting(N, local_unitaries, site_indices)
        else
            ms = nothing
        end
        measurements[i] = MeasurementData(m_results; measurement_setting=ms)
    end

    #println("We are constructing a MeasurementGroup object with measurement settings of type $T.")

    return MeasurementGroup(measurements)
end
