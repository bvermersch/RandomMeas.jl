# using ITensors
# using NPZ
# include("MeasurementSettings.jl") #Todo: Remove this line when include in the pacakge

"""
    struct MeasurementData{T}

A container for measurement data and settings used in quantum experiments.

# Fields
- `N::Int`: Number of sites (qubits).
- `NU::Int`: Number of measurement settings.
- `NM::Int`: Number of measurements per setting.
- `measurement_results::Array{Int, 3}`: A 3D array of binary measurement results with dimensions `(NU, NM, N)`.
- `measurement_settings::T`: Measurement settings of type `T` or `nothing` if not provided.

# Type Parameter
- `T`: The type of `measurement_settings`. This can be any subtype of `AbstractMeasurementSettings` or `Nothing` if no settings are provided.

# Usage
The `MeasurementData` struct is typically constructed using the provided constructor functions.
"""
struct MeasurementData{T}
    N::Int                              # Number of sites (qubits)
    NU::Int                             # Number of measurement settings
    NM::Int                             # Number of measurements per setting
    measurement_results::Array{Int, 3} # Binary measurement results (size: NU x NM x N)
    measurement_settings::T   # Measurement settings (or nothing if not provided)
end

"""
    MeasurementData(measurement_results::Array{Int, 3}; measurement_settings::Union{T, Nothing} = nothing)

Creates a `MeasurementData` object by inferring the dimensions of the measurement results and validating the provided settings.

# Arguments
- `measurement_results::Array{Int, 3}`: A 3D array of binary measurement results with shape `(NU, NM, N)`.
- `measurement_settings::Union{T <: AbstractMeasurementSettings, Nothing}` (optional): Measurement settings or `nothing` if not provided.

# Returns
A `MeasurementData` object with inferred dimensions and validated settings.

# Throws
- `AssertionError`: If the dimensions of `measurement_results` and `measurement_settings` are inconsistent.

# Examples
```julia
# With measurement settings
settings = LocalUnitaryMeasurementSettings(4, 3, ensemble="Haar")
results = rand(Int, 3, 10, 4)
data_with_settings = MeasurementData(results; measurement_settings=settings)

# Without measurement settings
data_without_settings = MeasurementData(rand(Int, 3, 10, 4))
```
"""
function MeasurementData(
    measurement_results::Array{Int, 3};
    measurement_settings::Union{T, Nothing} = nothing
) where {T <: AbstractMeasurementSettings}
    # Infer dimensions from measurement_results
    NU, NM, N = size(measurement_results)

    # Validate dimensions of measurement_settings, if provided
    if measurement_settings !== nothing
        @assert measurement_settings.N == N "Measurement settings must have the same N as the results."
        @assert measurement_settings.NU == NU "Measurement settings must have the same NU as the results."
    end

    # Delegate to the struct constructor
    return MeasurementData(N, NU, NM, measurement_results, measurement_settings)
end


### **Import Functions**
"""
    import_measurement_data(filepath::String; predefined_settings=nothing, site_indices=nothing)

Imports measurement results and optional measurement settings from an archive file.

# Arguments
- `filepath::String`: Path to the `.npz` file containing the measurement results and optionally local unitaries.
  The file should contain at least a field `measurement_results` (3D binary array of shape `(NU, NM, N)`),
  and optionally a field `local_unitaries` (local unitaries as a NUxNx2x2 array).
- `predefined_settings` (optional): A predefined `MeasurementSettings` object. If provided, this will be used instead of the file's settings.
- `site_indices` (optional): A vector of site indices to be used when constructing `LocalUnitaryMeasurementSettings` from the field `local_unitaries` (only relevant if predefined_settings are not provided).
  If not provided, the default site indices will be generated internally.

# Returns
A `MeasurementData` object containing the imported results and settings.

# Examples
```julia
# Import with predefined settings
settings = LocalUnitaryMeasurementSettings(local_unitaries; site_indices=siteinds("Qubit", 5))
data_with_settings = import_measurement_data("data.npz"; predefined_settings=settings)

# Import with site indices provided
data_with_indices = import_measurement_data("data.npz"; site_indices=siteinds("Qubit", 5))

# Import without any additional options
data = import_measurement_data("data.npz")
```
"""
function import_measurement_data(filepath::String; predefined_settings=nothing, site_indices=nothing, add_value=0)::MeasurementData
    # Load data from the archive
    data = npzread(filepath)

    # Extract measurement results
    measurement_results = data["measurement_results"]  # Shape: NU x NM x N

    # Check if 0 is contained and print a message if true
    if 0 in measurement_results
        println("Warning: Julia works with indices starting at 1. Binary data should therefore use 1 and 2, not 0 and 1. Please check the data and consider changing add_value parameter.")
    end

    # Optionally add a value to all elements
    if add_value != 0
        measurement_results .+= add_value
        println("Warning: Added $add_value to all elements of the measurement results.")
    end

    # Determine measurement settings
    if predefined_settings !== nothing
        # Use predefined settings if provided
        measurement_settings = predefined_settings
    elseif haskey(data, "local_unitaries")
        #TODO: write a new function to import the local unitaries as arrays
        # Load settings from the file if available
        NU,_,N = size(measurement_results)
        local_unitaries = Array{ITensor, 2}(undef, NU, N)
        for r in 1:NU
            for n in 1:N
                local_unitaries[r, n] = ITensor(data["local_unitaries"][r, n, :, :], site_indices[n]', site_indices[n])
            end
        end
        measurement_settings = LocalUnitaryMeasurementSettings(N,NU,local_unitaries,site_indices)
    else
        # Default to nothing if no settings are provided or found
        measurement_settings = nothing
    end

    # Create and return MeasurementData
    return MeasurementData(measurement_results; measurement_settings=measurement_settings)
end

"""
    export_measurement_data(data::MeasurementData, filepath::String)

Exports measurement data to a `.npz` file.

# Arguments
- `data::MeasurementData`: The measurement data object containing measurement results and optional measurement settings.
- `filepath::String`: The file path where the data will be exported.

# Details
- The `measurement_results` are exported directly as they are.
- If `measurement_settings` are provided, the associated `local_unitaries` are extracted, reshaped, and included in the export.

# Notes
- The exported `.npz` file will contain:
  - `"measurement_results"`: A 3D array of shape `(NU, NM, N)`, where:
    - `NU`: Number of unitary settings.
    - `NM`: Number of measurements per setting.
    - `N`: Number of qubits/sites.
  - `"local_unitaries"` (optional): A 4D array of shape `(NU, N, 2, 2)` representing the unitary transformations for each site.

# Example
```julia
# Example measurement results
measurement_results = rand(1:2, 10, 20, 5)  # NU x NM x N

# Example measurement settings
measurement_settings = LocalUnitaryMeasurementSettings(randn(10, 4, 4))  # Create settings

# Create MeasurementData object
data = MeasurementData(measurement_results; measurement_settings=measurement_settings)

# Export to a file
export_measurement_data(data, "exported_data.npz")
```
"""
function export_measurement_data(data::MeasurementData, filepath::String)

    export_dict = Dict{String, Any}()

    # Export measurement results
    export_dict["measurement_results"] = data.measurement_results

    # If measurement settings are present, process and add them to the export dictionary
    if data.measurement_settings !== nothing
        NU, N = data.measurement_settings.NU, data.measurement_settings.N
        local_unitaries = Array{ComplexF64}(undef, NU, N, 2, 2)
        for r in 1:NU
            for n in 1:N
                local_unitaries[r, n, :, :] = Array(
                    data.measurement_settings.local_unitaries[r, n],
                    data.measurement_settings.site_indices[n]',
                    data.measurement_settings.site_indices[n]
                )
            end
        end
        export_dict["local_unitaries"] = local_unitaries
    end

    # Write the data to the specified file path
    npzwrite(filepath, export_dict)
    #println("Measurement data successfully exported to $filepath.")
end
