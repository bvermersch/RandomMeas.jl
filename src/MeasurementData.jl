# Copyright (c) 2024 Benoît Vermersch and Andreas Elben
# SPDX-License-Identifier: Apache-2.0
# http://www.apache.org/licenses/LICENSE-2.0

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
) where T <: Union{Nothing, AbstractMeasurementSetting}
    # Infer dimensions from measurement_results
    NM, N = size(measurement_results)
    # Delegate to the struct constructor
    return MeasurementData(N, NM, measurement_results, measurement_setting)
end

"""
    MeasurementData(measurement_probability::MeasurementProbability{T}, NM::Int) where T <: Union{Nothing, AbstractMeasurementSetting}

Returns a `MeasurementData` object by sampling `NM` projective measurements based on the provided measurement probability.

# Arguments
- `measurement_probability::MeasurementProbability`: A container with the measurement probability (an ITensor) and associated settings.
- `NM::Int`: The number of projective measurements to sample.

# Returns
A `MeasurementData` object with dimensions inferred from the measurement probability.
"""
function MeasurementData(probability::MeasurementProbability{T},NM::Int) where T <: Union{Nothing, AbstractMeasurementSetting}
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
    MeasurementData(ψ::Union{MPO, MPS}, NM::Int; mode::String = "MPS/MPO", measurement_setting::Union{LocalUnitaryMeasurementSetting, ComputationalBasisMeasurementSetting, ShallowUnitaryMeasurementSetting} = nothing)

Returns a `MeasurementData` object by sampling `NM` projective measurements from the quantum state `ψ`.

# Arguments
- `ψ::Union{MPO, MPS}`: The quantum state represented as a Matrix Product Operator (MPO) or Matrix Product State (MPS).
- `NM::Int`: The number of measurement shots to simulate for each setting.
- `mode::String` (optional): Specifies the simulation method. Options:
   - `"dense"`: Uses the dense representation.
   - `"MPS/MPO"` (default): Uses tensor network methods for memory efficiency.
- `measurement_setting` (optional): A measurement setting object (if not provided, defaults to computational basis measurements).

# Returns
A `MeasurementData` object with the corresponding measurement results and setting.
"""
function MeasurementData(
    ψ::Union{MPO, MPS},
    NM::Int,
    measurement_setting::Union{LocalUnitaryMeasurementSetting, ComputationalBasisMeasurementSetting, ShallowUnitaryMeasurementSetting};
    mode::String = "MPS/MPO",
)

    N = measurement_setting.N
    @assert length(ψ) == N "The number of sites of the MPS/MPO ψ and the MeasurementSettings Object do not match."

    ψ_indices = vcat(siteinds(ψ, plev=0)...)
    site_indices = measurement_setting.site_indices
    if site_indices != ψ_indices
        # Build a map old→new (and old′→new′)
        repl = Dict{Index,Index}()
        for (old, new) in zip(site_indices, ψ_indices)
            repl[ old        ] = new
            repl[ prime(old) ] = prime(new)
        end

        # Rename every gate tensor in one shot
        u_old = measurement_setting.local_unitary
        u_new = [ replaceinds(ui, pairs(repl)...) for ui in u_old ]

        # Rebuild the same concrete setting with updated indices
        T = typeof(measurement_setting)
        measurement_setting = T(
          measurement_setting;
          local_unitary = u_new,
          site_indices  = ψ_indices,
        )
    end

    u=measurement_setting.local_unitary
    if mode=="dense"
        measurement_probability = MeasurementProbability(ψ,measurement_setting)
        return MeasurementData(measurement_probability,NM)
    elseif mode == "MPS/MPO"
        data = zeros(Int,NM,N)
        ξ = measurement_setting.site_indices

        if isa(ψ,MPS)
            ψu = apply(reverse(u),ψ) #using reverse allows us to maintain orthocenter(ψ)=1 ;)
            orthogonalize!(ψu,1)
            for m in 1:NM
                data[m, :] = ITensorMPS.sample(ψu)
            end
        else
            ρu = apply(u,ψ;apply_dag=true)
            ρu[1] /= get_trace(ρu)
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
    else
        throw(ArgumentError("Invalid simulation mode: \"$mode\". Expected either \"dense\" or \"MPS/MPO\"."))
    end
end


"""
    reduce_to_subsystem(data::MeasurementData{T}, subsystem::Vector{Int}) where T <: Union{Nothing, LocalMeasurementSetting}

Reduce a `MeasurementData` object to a specified subsystem, preserving the measurement setting type if available.

# Arguments
- `data::MeasurementData{T}`: The original measurement data object, where `T` is either `nothing` or a subtype of `LocalMeasurementSetting`.
- `subsystem::Vector{Int}`: A vector of site indices (1-based) specifying the subsystem to retain. Each index must be between 1 and `data.N`.

# Returns
A new `MeasurementData{T}` object with:
- The measurement results reduced from dimensions `(NM, N)` to `(NM, |subsystem|)`.
- The measurement setting reduced accordingly (if one is provided), or remaining as `nothing`.

# Example
```julia
# Suppose `data` is a MeasurementData object with N = 4.
# To retain only sites 1 and 3:
reduced_data = reduce_to_subsystem(data, [1, 3])
```
"""
function reduce_to_subsystem(data::MeasurementData{T}, subsystem::Vector{Int})::MeasurementData{T} where T <: Union{Nothing, LocalMeasurementSetting}
    # Validate that each index in the subsystem is in the valid range.
    @assert all(x -> x >= 1 && x <= data.N, subsystem) "Subsystem indices must be between 1 and N."
    @assert length(unique(subsystem)) == length(subsystem) "Subsystem indices must be unique."

    # Reduce the measurement setting if it is provided.
    reduced_setting = data.measurement_setting === nothing ? nothing :
        reduce_to_subsystem(data.measurement_setting, subsystem)

    # Reduce the measurement results: From dimensions (NM x N) to (NM x |subsystem|)
    reduced_results = data.measurement_results[:, subsystem]

    # Create and return the new MeasurementData object with the same type parameter.
    return MeasurementData(reduced_results; measurement_setting=reduced_setting)
end


### **Import Functions**
"""
    import_measurement_data(filepath::String; predefined_setting=nothing, site_indices=nothing)

Imports measurement results and optional measurement settings from an archive file.

# Arguments
- `filepath::String`: Path to the `.npz` file containing the measurement results and optionally local unitaries.
  The file should contain at least a field `measurement_results` (2D binary array of shape `(NM, N)`),
  and optionally a field `local_unitaries` (local unitaries as a Nx2x2 array).
- `predefined_setting` (optional): A predefined `MeasurementSetting` object. If provided, this will be used instead of the file's setting.
- `site_indices` (optional): A vector of site indices to be used when constructing `LocalUnitaryMeasurementSetting` from the field `local_unitaries` (only relevant if predefined_setting is not provided).
  If not provided, the default site indices will be generated internally.

# Returns
A `MeasurementData` object containing the imported results and settings.

# Examples
```julia
# Import with predefined settings
setting = LocalUnitaryMeasurementSetting(local_unitaries; site_indices=siteinds("Qubit", 5))
data_with_setting = import_measurement_data("data.npz"; predefined_setting=setting)

# Import with site indices provided
data_with_indices = import_measurement_data("data.npz"; site_indices=siteinds("Qubit", 5))

# Import without any additional options
data = import_measurement_data("data.npz")
```
"""
function import_MeasurementData(filepath::String; predefined_setting=nothing, site_indices=nothing)::MeasurementData
    # Load data from the archive
    data = npzread(filepath)

    # Extract measurement results
    measurement_results = data["measurement_results"]  # Shape: NM x N

    # Check if 0 is contained and print a message if true
    if 0 in measurement_results
        @warn "Julia works with indices starting at 1. Binary data should therefore use 1 and 2, not 0 and 1."
    end

    # Determine measurement settings
    if predefined_setting !== nothing
        # Use predefined settings if provided
        measurement_setting = predefined_setting
    elseif haskey(data, "local_unitaries")
        # Load settings from the file if available
        _,N = size(measurement_results)
        if site_indices === nothing
            site_indices = siteinds("Qubit", N)
        end
        local_unitaries = Array{ITensor}(undef, N)
        for n in 1:N
            local_unitaries[n] = ITensor(data["local_unitaries"][n, :, :], site_indices[n]', site_indices[n])
        end
        measurement_setting = LocalUnitaryMeasurementSetting(N,local_unitaries,site_indices)
    else
        # Default to nothing if no settings are provided or found
        measurement_setting = nothing
        @warn "Measurement setting not found in the file and not provided; defaulting to nothing."
    end

    # Create and return MeasurementData
    return MeasurementData(measurement_results; measurement_setting=measurement_setting)
end

"""
    export_measurement_data(data::MeasurementData, filepath::String)

Exports measurement data to a `.npz` file.

# Arguments
- `data::MeasurementData`: The measurement data object containing measurement results and optionally a LocalUnitaryMeasurementSetting setting.
- `filepath::String`: The file path where the data will be exported.

# Details
- The `measurement_results` are exported directly as they are.
- If `measurement_setting` is provided, the associated `local_unitaries` are extracted, reshaped, and included in the export.

# Notes
- The exported `.npz` file will contain:
  - `"measurement_results"`: A 2D array of shape `(NM, N)`, where:
    - `NM`: Number of measurements per setting.
    - `N`: Number of qubits/sites.
  - `"local_unitaries"` (optional): A 4D array of shape `(N, 2, 2)` representing the unitary transformations for each site.

# Example
```julia

# Create MeasurementData object
data = MeasurementData(measurement_results; measurement_setting=measurement_setting)

# Export to a file
export_measurement_data(data, "exported_data.npz")
```
"""
function export_MeasurementData(data::MeasurementData{T}, filepath::String) where {T<:Union{Nothing, LocalUnitaryMeasurementSetting,ComputationalBasisMeasurementSetting}}

    export_dict = Dict{String, Any}()

    # Export measurement results
    export_dict["measurement_results"] = data.measurement_results

    # If measurement settings are present, process and add them to the export dictionary
    if data.measurement_setting !== nothing
        N = data.measurement_setting.N
        local_unitaries = Array{ComplexF64}(undef, N, 2, 2)
        for n in 1:N
            local_unitaries[n, :, :] = Array(data.measurement_setting.local_unitary[n],data.measurement_setting.site_indices[n]',data.measurement_setting.site_indices[n])
        end
        export_dict["local_unitaries"] = local_unitaries
    end

    # Write the data to the specified file path
    npzwrite(filepath, export_dict)
    #println("Measurement data successfully exported to $filepath.")
end
