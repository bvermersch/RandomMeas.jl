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

#### **1. `import_measurement_data`**
"""
    import_measurement_data(results_path::String)

Imports measurement results and optional unitary settings from an `.npz` archive.

# Arguments
- `results_path::String`: Path to the `.npz` file containing measurement results and optional unitaries.

# Returns
A `MeasurementData` object containing the imported results and settings.

# Examples
```julia
# Import with unitaries included in the same file
data_with_unitaries = import_measurement_data("data.npz")
```
"""
function import_measurement_data(results_path::String)::MeasurementData
    # Load the .npz file
    data = npzread(results_path)
    # Extract and validate measurement_results
    @assert haskey(data, "measurement_results") "The NPZ file must contain a 'measurement_results' field."
    measurement_results = data["measurement_results"]  # Shape: NU x NM x N

    # Extract local_unitaries if available
    measurement_settings = haskey(data, "local_unitaries") ? LocalUnitaryMeasurementSettings(data["local_unitaries"]) : nothing

    # Create and return MeasurementData
    return MeasurementData(measurement_results; measurement_settings=measurement_settings)
end

#### **2. `import_measurement_results`**
"""
    import_measurement_results(results_path::String; measurement_settings=nothing)

Imports only the measurement results from an `.npz` file and optionally associates predefined measurement settings.

# Arguments
- `results_path::String`: Path to the `.npz` file containing the measurement results and optionally unitaries.
- `measurement_settings` (optional): A predefined measurement settings object. If not provided, unitaries from the archive are used, if available.

# Returns
A `MeasurementData` object containing the imported results and the provided or extracted settings.

# Examples
```julia
# Import with predefined settings
settings = LocalUnitaryMeasurementSettings(...)
data_with_settings = import_measurement_results("data.npz"; measurement_settings=settings)

# Import and use unitaries from the file
data_with_unitaries = import_measurement_results("data.npz")
```
"""
function import_measurement_results(
    results_path::String;
    measurement_settings=nothing
    )::MeasurementData
    # Load the .npz file
    data = npzread(results_path)

    # Extract and validate measurement_results
    @assert haskey(data, "measurement_results") "The NPZ file must contain a 'measurement_results' field."
    measurement_results = data["measurement_results"]  # Shape: NU x NM x N

    # Use provided settings or extract from file
    if measurement_settings === nothing && haskey(data, "local_unitaries")
        measurement_settings = LocalUnitaryMeasurementSettings(data["local_unitaries"])
    end

    # Create and return MeasurementData
    return MeasurementData(measurement_results; measurement_settings=measurement_settings)
end
