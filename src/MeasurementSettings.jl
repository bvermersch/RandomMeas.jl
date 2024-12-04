# Abstract Base Type for Measurement Settings
"""
Abstract type representing general measurement settings.
Specific implementations (e.g., LocalUnitaryMeasurementSettings) should subtype this.
"""
abstract type AbstractMeasurementSettings end

# Local Unitary Measurement Settings
"""
LocalUnitaryMeasurementSettings

A struct representing local unitary measurement settings for quantum systems.

# Fields:
- `N::Int`: Number of sites (qubits).
- `NU::Int`: Number of measurement settings.
- `local_unitaries::Array{ITensor, 2}`: NU x N array of local unitary ITensors.
- `site_indices::Vector{Index{Int64}}`: Vector of site indices of length N.
"""
struct LocalUnitaryMeasurementSettings <: AbstractMeasurementSettings
    N::Int                              # Number of sites
    NU::Int                             # Number of measurement settings
    local_unitaries::Array{ITensor, 2}  # NU x N array of unitaries
    site_indices::Vector{Index{Int64}}  # Vector of site indices (length N)

    """
    Create a LocalUnitaryMeasurementSettings object with validation.

    # Arguments:
    - `N::Int`: Number of sites.
    - `NU::Int`: Number of measurement settings.
    - `local_unitaries::Array{ITensor, 2}`: Array of local unitaries with dimensions NU x N.
    - `site_indices::Vector{Index{Int64}}`: Vector of site indices.

    # Throws:
    - `AssertionError` if dimensions of `local_unitaries` or `site_indices` do not match `N` or `NU`.
    """
    function LocalUnitaryMeasurementSettings(
        N::Int, NU::Int, local_unitaries::Array{ITensor, 2}, site_indices::Vector{Index{Int64}}
    )
        @assert size(local_unitaries, 1) == NU "Mismatch in number of measurement settings (NU)."
        @assert size(local_unitaries, 2) == N "Mismatch in number of sites (N)."
        @assert length(site_indices) == N "Length of site_indices must match N."
        return new(N, NU, local_unitaries, site_indices)
    end
end

"""
Create a `LocalUnitaryMeasurementSettings` object from an NU x N x 2 x 2 array.

# Arguments:
- `local_unitaries_array::Array{ComplexF64, 4}`: NU x N x 2 x 2 array of unitaries.
- `site_indices::Union{Vector{Index{Int64}}, Nothing}`: Optional vector of site indices. If not provided, it will be generated.

# Returns:
- A `LocalUnitaryMeasurementSettings` object.
"""
function LocalUnitaryMeasurementSettings(local_unitaries_array::Array{ComplexF64, 4}; site_indices::Union{Vector{Index{Int64}}, Nothing} = nothing)
    # Extract dimensions
    NU, N, rows, cols = size(local_unitaries_array)
    @assert rows == 2 && cols == 2 "Unitary matrices must have size 2x2."

    # Generate or validate site indices
    site_indices = site_indices === nothing ? siteinds("Qubit", N) : site_indices
    @assert length(site_indices) == N "Length of site_indices must match N."

    # Convert the array into ITensors
    local_unitaries = Array{ITensor, 2}(undef, NU, N)
    for r in 1:NU
        for n in 1:N
            local_unitaries[r, n] = ITensor(local_unitaries_array[r, n, :, :], site_indices[n]', site_indices[n])
        end
    end

    # Call the main constructor
    return LocalUnitaryMeasurementSettings(N, NU, local_unitaries, site_indices)
end

"""
Create a `LocalUnitaryMeasurementSettings` object by random sampling local unitaries

# Arguments:
- `N::Int`: Number of sites.
- `NU::Int`: Number of measurement settings.
- `site_indices::Union{Vector{Index}, Nothing}`: Optional vector of site indices. If not provided, it will be generated.
- `ensemble::String`: Type of random unitaries to generate ("Haar", "Pauli", or "Identity").

# Returns:
- A LocalUnitaryMeasurementSettings object.
"""
function LocalUnitaryMeasurementSettings(
    N::Int, NU::Int;
    site_indices::Union{Vector{Index{Int64}}, Nothing} = nothing,
    ensemble::String = "Haar"
)
    # Generate site indices if not provided
    site_indices = site_indices === nothing ? siteinds("Qubit", N) : site_indices
    @assert length(site_indices) == N "Length of site_indices must match N."

    # Generate local unitaries
    local_unitaries = [get_rotation(site_indices[j], ensemble) for _ in 1:NU, j in 1:N]

    # Call the main constructor
    return LocalUnitaryMeasurementSettings(N, NU, local_unitaries, site_indices)
end

# Helper Function to Generate Single Qubit Unitaries
"""
    get_rotation(ξ::Index{Int64}, ensemble::String = "Haar")

Generate a single qubit unitary with indices (ξ', ξ) sampled from the specified ensemble.

# Arguments:
- `ξ::Index{Int64}`: Site index.
- `ensemble::String`: Type of unitary ensemble ("Haar", "Pauli", "CompBasis").

# Returns:
- An ITensor representing the unitary.
"""
function get_rotation(site_index::Index{Int64}, ensemble::String = "Haar")
    r_matrix = zeros(ComplexF64, (2, 2))
    if ensemble == "Haar"
        return op("RandomUnitary", site_index)
    elseif ensemble == "Pauli"
        b = rand(1:3)
        if b == 1
            r_matrix[1, 1] = 1
            r_matrix[2, 2] = 1
        elseif b == 2
            r_matrix[1, 1] = 1 / sqrt(2)
            r_matrix[2, 1] = 1 / sqrt(2)
            r_matrix[1, 2] = 1 / sqrt(2)
            r_matrix[2, 2] = -1 / sqrt(2)
        else
            r_matrix[1, 1] = 1 / sqrt(2)
            r_matrix[2, 2] = 1 / sqrt(2)
            r_matrix[1, 2] = -1im / sqrt(2)
            r_matrix[2, 1] = -1im / sqrt(2)
        end
        return ITensor(r_matrix, site_index', site_index)
    elseif ensemble == "CompBasis"
        r_matrix[1, 1] = 1
        r_matrix[2, 2] = 1
        return ITensor(r_matrix, site_index', site_index)
    end
end

# Export Method
"""
Export the unitaries in a LocalUnitaryMeasurementSettings object to an .npz file with a single field: local_unitaries.

# Arguments:
- `ms::LocalUnitaryMeasurementSettings`: The measurement settings to export.
- `filepath::String`: Path to the output .npz file.
"""
function export_unitaries(ms::LocalUnitaryMeasurementSettings, filepath::String)
    # Prepare the local_unitaries array for export
    local_unitaries = Array{ComplexF64}(undef, ms.NU, ms.N, 2, 2)
    for r in 1:ms.NU
        for n in 1:ms.N
            local_unitaries[r, n, :, :] = Array(ms.local_unitaries[r, n], ms.site_indices[n]', ms.site_indices[n])
        end
    end

    # Write to the .npz file
    npzwrite(filepath, Dict("local_unitaries" => local_unitaries))
    println("Exported to NPZ file: $filepath")
end

# Import Method
"""
Import unitaries from an .npz file and create a LocalUnitaryMeasurementSettings object.

# Arguments:
- `filepath::String`: Path to the input .npz file.
- `site_indices::Union{Vector{Index{Int64}}, Nothing}`: Optional site indices. If not provided, they will be generated.

# Returns:
- A LocalUnitaryMeasurementSettings object.
"""
function import_unitaries(filepath::String; site_indices::Union{Vector{Index{Int64}}, Nothing} = nothing)::LocalUnitaryMeasurementSettings
    # Read the .npz file
    data = npzread(filepath)

    # Extract the local_unitaries field
    @assert haskey(data, "local_unitaries") "Missing 'local_unitaries' field in the NPZ file."

    return LocalUnitaryMeasurementSettings(data["local_unitaries"]; site_indices=site_indices)
end


"""
    reduce_to_subsystem(settings::LocalUnitaryMeasurementSettings, subsystem::Vector{Int})

Reduce a `LocalUnitaryMeasurementSettings` object to a specified subsystem.

# Arguments
- `settings::LocalUnitaryMeasurementSettings`: The original measurement settings object.
- `subsystem::Vector{Int}`: A vector of site indices (1-based) specifying the subsystem to retain.

# Returns
A new `LocalUnitaryMeasurementSettings` object corresponding to the specified subsystem.
"""
function reduce_to_subsystem(
    settings::LocalUnitaryMeasurementSettings,
    subsystem::Vector{Int}
)::LocalUnitaryMeasurementSettings
    # Validate the subsystem
    @assert all(x -> x >= 1 && x <= settings.N, subsystem) "Subsystem indices must be between 1 and N."
    @assert length(unique(subsystem)) == length(subsystem) "Subsystem indices must be unique."

    # Extract the reduced local unitaries and site indices
    reduced_unitaries = settings.local_unitaries[:, subsystem]
    reduced_indices = settings.site_indices[subsystem]

    # Create the new LocalUnitaryMeasurementSettings object
    return LocalUnitaryMeasurementSettings(
        length(subsystem),  # New N is the size of the subsystem
        settings.NU,        # NU remains the same
        reduced_unitaries,
        reduced_indices
    )
end
