
"""
    MeasurementData(measurement_results::Array{Int, 2}; measurement_setting::Union{T, Nothing} = nothing)

Creates a `MeasurementData` object by inferring the dimensions of the measurement results and validating the provided setting.

# Arguments
- `measurement_results::Array{Int, 2}`: A 2D array of binary measurement results with shape `(NM, N)`.
- `measurement_setting::Union{T <: AbstractMeasurementSetting, Nothing}` (optional): Measurement setting or `nothing` if not provided.

# Returns
A `MeasurementData` object with inferred dimensions and validated setting.

# Throws
- `AssertionError`: If the dimensions of `measurement_results` and `measurement_setting` are inconsistent.

# Examples
```julia
# With measurement setting
setting = LocalUnitaryMeasurementSetting(4, ensemble="Haar")
results = rand(1:2, 10, 4)
data_with_setting = MeasurementData(results; measurement_setting=setting)

# Without measurement setting
data_without_setting = MeasurementData(rand(1:2, 10, 4))
```
"""
function MeasurementData(
    measurement_results::Array{Int, 2};
    measurement_setting::Union{T, Nothing} = nothing
) where {T <: AbstractMeasurementSetting}
    # Infer dimensions from measurement_results
    NM, N = size(measurement_results)

    # Validate dimensions of measurement_setting, if provided
    if measurement_setting !== nothing
        @assert measurement_setting.N == N "Measurement setting must have the same N as the results."
    end

    # Delegate to the struct constructor
    return MeasurementData(N, NM, measurement_results, measurement_setting)
end

"""
    MeasurementData(measurement_probability::MeasurementProbability{T},NM::Int) where {T <: AbstractMeasurementSetting}

Returns a Measurement Data Object by sampling NM projective measurements from the array measurement_probability

# Arguments
- `measurement_probability::MeasurementProbability`.
- `NM::Int' number of projective measurements

# Returns
A `MeasurementData` object with inferred dimensions and validated setting.
"""
function MeasurementData(probability::MeasurementProbability{T},NM::Int) where T #{T <: AbstractMeasurementSetting}
    N = probability.N
    Prob = probability.measurement_probability
    prob = real(array(Prob))
    prob = reshape(prob, 2^N)
    measurement_results = zeros(Int,NM,N)
    measurement_setting = probability.measurement_setting
    for m in 1:NM
        data = StatsBase.sample(0:(1<<N-1), StatsBase.Weights(prob), 1)
        measurement_results[m, :] = 1 .+ digits(data[1], base=2, pad=N)
    end
    return MeasurementData(N,NM,measurement_results,measurement_setting)
end

"""
    MeasurementData(
    ψ::Union{MPO, MPS},
    NM::Int;
    mode::String = "MPS/MPO",
    measurement_settings::Union{LocalUnitaryMeasurementSettings, Nothing} = nothing,
)::MeasurementData{LocalUnitaryMeasurementSettings}
Returns a Measurement Data Object by sampling NM projective measurements from a quantum state

# Arguments
- `state::Union{MPO, MPS}`: The quantum state to be measured, represented as a matrix product operator (MPO) or matrix product state (MPS).
- `NM::Int64`: The number of measurement shots to simulate for each unitary setting.
- `mode::String` (optional): Specifies the simulation method.
  - `"dense"`: Simulates measurements using the dense representation of the state.
  - `"MPS/MPO"` (default): Simulates measurements using tensor network (TN) methods for memory efficiency.
  - Any other value will result in an error.
- `measurement_settings::Union{LocalUnitaryMeasurementSettings, Nothing}` (optional): Specifies the local unitary settings for the measurements.
  - If `nothing`, defaults to computational basis measurements.
# Returns
A `MeasurementData` object
"""
function MeasurementData(
    ψ::Union{MPO, MPS},
    NM::Int;
    mode::String = "MPS/MPO",
    measurement_setting::Union{LocalUnitaryMeasurementSetting, Nothing} = nothing,
)#::MeasurementData{LocalUnitaryMeasurementSetting}
    if mode=="dense"
        measurement_probability = MeasurementProbability(ψ,measurement_setting)
        return MeasurementData(measurement_probability,NM)
    else
        N = measurement_setting.N
        data = zeros(Int,NM,N)
        ξ = measurement_setting.site_indices
        u = measurement_setting.local_unitary
        if isa(ψ,MPS)
            ψu = apply(reverse(u),ψ) #using reverse allows us to maintain orthocenter(ψ)=1 ;)
            for m in 1:NM
                data[m, :] = ITensorMPS.sample(ψu)
            end
        else
            ρu = apply(u,ψ;apply_dag=true)
            ρu[1] /= trace(ρu, ξ)
            if N > 1
                for m in 1:NM
                    data[m, :] = ITensorMPS.sample(ρu)
                end
            else
                s = ξ[1]
                prob = [real(ρu[1][s=>1, s'=>1][]), real(ρu[1][s=>2, s'=>2][])]
                data[:, 1] = StatsBase.sample(1:2, StatsBase.Weights(prob), NM)
            end
        end
        return MeasurementData(N,NM,data,measurement_setting)
    end
end


# ### **Import Functions**
# """
#     import_measurement_data(filepath::String; predefined_settings=nothing, site_indices=nothing)

# Imports measurement results and optional measurement settings from an archive file.

# # Arguments
# - `filepath::String`: Path to the `.npz` file containing the measurement results and optionally local unitaries.
#   The file should contain at least a field `measurement_results` (3D binary array of shape `(NU, NM, N)`),
#   and optionally a field `local_unitaries` (local unitaries as a NUxNx2x2 array).
# - `predefined_settings` (optional): A predefined `MeasurementSettings` object. If provided, this will be used instead of the file's settings.
# - `site_indices` (optional): A vector of site indices to be used when constructing `LocalUnitaryMeasurementSettings` from the field `local_unitaries` (only relevant if predefined_settings are not provided).
#   If not provided, the default site indices will be generated internally.

# # Returns
# A `MeasurementData` object containing the imported results and settings.

# # Examples
# ```julia
# # Import with predefined settings
# settings = LocalUnitaryMeasurementSettings(local_unitaries; site_indices=siteinds("Qubit", 5))
# data_with_settings = import_measurement_data("data.npz"; predefined_settings=settings)

# # Import with site indices provided
# data_with_indices = import_measurement_data("data.npz"; site_indices=siteinds("Qubit", 5))

# # Import without any additional options
# data = import_measurement_data("data.npz")
# ```
# """
# function import_measurement_data(filepath::String; predefined_settings=nothing, site_indices=nothing, add_value=0)::MeasurementData
#     # Load data from the archive
#     data = npzread(filepath)

#     # Extract measurement results
#     measurement_results = data["measurement_results"]  # Shape: NU x NM x N

#     # Optionally add a value to all elements
#     if add_value != 0
#         measurement_results .+= add_value
#         @warn "The add_value parameter is $add_value and added to all measurement results. The measurement results contain now only $(Set(measurement_results)) ."
#     end

#     # Check if 0 is contained and print a message if true
#     if 0 in measurement_results
#         @warn "Julia works with indices starting at 1. Binary data should therefore use 1 and 2, not 0 and 1. To fix this, use the add_value parameter."
#     end

#     # Determine measurement settings
#     if predefined_settings !== nothing
#         # Use predefined settings if provided
#         measurement_settings = predefined_settings
#     elseif haskey(data, "local_unitaries")
#         #TODO: write a new function to import the local unitaries as arrays
#         # Load settings from the file if available
#         NU,_,N = size(measurement_results)
#         local_unitaries = Array{ITensor, 2}(undef, NU, N)
#         for r in 1:NU
#             for n in 1:N
#                 local_unitaries[r, n] = ITensor(data["local_unitaries"][r, n, :, :], site_indices[n]', site_indices[n])
#             end
#         end
#         measurement_settings = LocalUnitaryMeasurementSettings(N,NU,local_unitaries,site_indices)
#     else
#         # Default to nothing if no settings are provided or found
#         measurement_settings = nothing
#         @warn "Measurement settings not found in the file and not provided. Using default settings."
#     end

#     # Create and return MeasurementData
#     return MeasurementData(measurement_results; measurement_settings=measurement_settings)
# end

# """
#     export_measurement_data(data::MeasurementData, filepath::String)

# Exports measurement data to a `.npz` file.

# # Arguments
# - `data::MeasurementData`: The measurement data object containing measurement results and optional measurement settings.
# - `filepath::String`: The file path where the data will be exported.

# # Details
# - The `measurement_results` are exported directly as they are.
# - If `measurement_settings` are provided, the associated `local_unitaries` are extracted, reshaped, and included in the export.

# # Notes
# - The exported `.npz` file will contain:
#   - `"measurement_results"`: A 3D array of shape `(NU, NM, N)`, where:
#     - `NU`: Number of unitary settings.
#     - `NM`: Number of measurements per setting.
#     - `N`: Number of qubits/sites.
#   - `"local_unitaries"` (optional): A 4D array of shape `(NU, N, 2, 2)` representing the unitary transformations for each site.

# # Example
# ```julia
# # Example measurement results
# measurement_results = rand(1:2, 10, 20, 5)  # NU x NM x N

# # Example measurement settings
# measurement_settings = LocalUnitaryMeasurementSettings(randn(10, 4, 4))  # Create settings

# # Create MeasurementData object
# data = MeasurementData(measurement_results; measurement_settings=measurement_settings)

# # Export to a file
# export_measurement_data(data, "exported_data.npz")
# ```
# """
# function export_measurement_data(data::MeasurementData, filepath::String)

#     export_dict = Dict{String, Any}()

#     # Export measurement results
#     export_dict["measurement_results"] = data.measurement_results

#     # If measurement settings are present, process and add them to the export dictionary
#     if data.measurement_settings !== nothing
#         NU, N = data.measurement_settings.NU, data.measurement_settings.N
#         local_unitaries = Array{ComplexF64}(undef, NU, N, 2, 2)
#         for r in 1:NU
#             for n in 1:N
#                 local_unitaries[r, n, :, :] = Array(
#                     data.measurement_settings.local_unitaries[r, n],
#                     data.measurement_settings.site_indices[n]',
#                     data.measurement_settings.site_indices[n]
#                 )
#             end
#         end
#         export_dict["local_unitaries"] = local_unitaries
#     end

#     # Write the data to the specified file path
#     npzwrite(filepath, export_dict)
#     #println("Measurement data successfully exported to $filepath.")
# end


"""
    reduce_to_subsystem(data::MeasurementData{LocalUnitaryMeasurementSetting}, subsystem::Vector{Int})

Reduce a `MeasurementData` object (with `LocalUnitaryMeasurementSetting`) to a specified subsystem.

# Arguments
- `data::MeasurementData{LocalUnitaryMeasurementSetting}`: The original measurement data object.
- `subsystem::Vector{Int}`: A vector of site indices (1-based) specifying the subsystem to retain.

# Returns
A new `MeasurementData` object corresponding to the specified subsystem.
"""
function reduce_to_subsystem(
    data::MeasurementData{LocalUnitaryMeasurementSetting},
    subsystem::Vector{Int}
)::MeasurementData{LocalUnitaryMeasurementSetting}
    # Validate the subsystem
    @assert all(x -> x >= 1 && x <= data.N, subsystem) "Subsystem indices must be between 1 and N."
    @assert length(unique(subsystem)) == length(subsystem) "Subsystem indices must be unique."

    # Reduce the measurement setting
    reduced_setting = reduce_to_subsystem(data.measurement_setting, subsystem)

    # Reduce the measurement results: NM x N →  NM x |subsystem|
    reduced_results = data.measurement_results[:, subsystem]

    # Create and return the new MeasurementData object
    return MeasurementData(reduced_results; measurement_setting=reduced_setting)
end
