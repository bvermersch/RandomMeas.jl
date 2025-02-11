# Abstract Base Type for Measurement Setting
"""
Abstract type representing general a measurement setting.
Specific implementations (e.g., LocalUnitaryMeasurementSetting) should subtype this.
"""
abstract type AbstractMeasurementSetting end

# Local Unitary Measurement Setting
"""
LocalUnitaryMeasurementSetting

A struct representing local unitary measurement settings for quantum systems.

# Fields:
- `N::Int`: Number of sites (qubits).
- `local_unitary::Vector{ITensor}`: A local unitary represented by N 2x2 ITensors.
- `site_indices::Vector{Index{Int64}}`: Vector of site indices of length N.
"""
struct LocalUnitaryMeasurementSetting <: AbstractMeasurementSetting
    N::Int                              # Number of sites
    local_unitary::Vector{ITensor}  # local unitary represented by a vector of N 2x2 unitary
    site_indices::Vector{Index{Int64}}  # Vector of site indices (length N)

    """
    Create a LocalUnitaryMeasurementSettings object with validation.

    # Arguments:
    - `N::Int`: Number of sites.
    - `local_unitary::Vector{ITensor}`: Vector of local unitary of length N.
    - `site_indices::Vector{Index{Int64}}`: Vector of site indices of length N.

    # Throws:
    - `AssertionError` if dimensions of `local_unitary` or `site_indices` do not match `N`.
    """
    function LocalUnitaryMeasurementSetting(
        N::Int, local_unitary::Vector{ITensor}, site_indices::Vector{Index{Int64}}
    )
        @assert length(local_unitary) == N "Mismatch in number of sites (N)."
        @assert length(site_indices) == N "Length of site_indices must match N."
        return new(N, local_unitary, site_indices)
    end
end

"""
Create a `LocalUnitaryMeasurementSetting` object from an N x 2 x 2 array.

# Arguments:
- `local_unitary_array::Array{ComplexF64, 3}`:  N x 2 x 2 array of unitary.
- `site_indices::Union{Vector{Index{Int64}}, Nothing}`: Optional vector of site indices. If not provided, it will be generated.

# Returns:
- A `LocalUnitaryMeasurementSetting` object.
"""
function LocalUnitaryMeasurementSetting(local_unitary_array::Array{ComplexF64, 3}; site_indices::Union{Vector{Index{Int64}}, Nothing} = nothing)
    # Extract dimensions
    N, rows, cols = size(local_unitary_array)
    @assert rows == 2 && cols == 2 "Unitary matrices must have size 2x2."

    # Generate or validate site indices
    site_indices = site_indices === nothing ? siteinds("Qubit", N) : site_indices
    @assert length(site_indices) == N "Length of site_indices must match N."

    # Convert the array into ITensors
    local_unitary = Vector{ITensor}(undef, N)
    for n in 1:N
            local_unitary[n] = ITensor(local_unitary_array[n, :, :], site_indices[n]', site_indices[n])
    end

    # Call the main constructor
    return LocalUnitaryMeasurementSetting(N, local_unitary, site_indices)
end

"""
Create a `LocalUnitaryMeasurementSetting` object by random sampling local unitary

# Arguments:
- `N::Int`: Number of sites.
- `site_indices::Union{Vector{Index}, Nothing}`: Optional vector of site indices. If not provided, it will be generated.
- `ensemble::String`: Type of random unitary to generate ("Haar", "Pauli", or "Identity").

# Returns:
- A LocalUnitaryMeasurementSetting object.
"""
function LocalUnitaryMeasurementSetting(
    N::Int;
    site_indices::Union{Vector{Index{Int64}}, Nothing} = nothing,
    ensemble::String = "Haar"
)
    # Generate site indices if not provided
    site_indices = site_indices === nothing ? siteinds("Qubit", N) : site_indices
    @assert length(site_indices) == N "Length of site_indices must match N."

    # Generate local unitary
    local_unitary = [get_rotation(site_indices[j], ensemble) for j in 1:N]
    #print(local_unitary)

    # Call the main constructor
    return LocalUnitaryMeasurementSetting(N, local_unitary, site_indices)
end

# Helper Function to Generate Single Qubit unitary
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
    elseif ensemble == "Identity"
        r_matrix[1, 1] = 1
        r_matrix[2, 2] = 1
        return ITensor(r_matrix, site_index', site_index)
    else
        throw(ArgumentError("Invalid ensemble: $ensemble"))
    end
end

# # Export Method
# """
# Export the unitary in a LocalUnitaryMeasurementSettings object to an .npz file with a single field: local_unitary.

# # Arguments:
# - `ms::LocalUnitaryMeasurementSettings`: The measurement settings to export.
# - `filepath::String`: Path to the output .npz file.
# """
# function export_unitary(ms::LocalUnitaryMeasurementSettings, filepath::String)
#     # Prepare the local_unitary array for export
#     local_unitary = Array{ComplexF64}(undef, ms.NU, ms.N, 2, 2)
#     for r in 1:ms.NU
#         for n in 1:ms.N
#             local_unitary[r, n, :, :] = Array(ms.local_unitary[r, n], ms.site_indices[n]', ms.site_indices[n])
#         end
#     end

#     # Write to the .npz file
#     npzwrite(filepath, Dict("local_unitary" => local_unitary))
#     println("Exported to NPZ file: $filepath")
# end

# # Import Method
# """
# Import unitary from an .npz file and create a LocalUnitaryMeasurementSettings object.

# # Arguments:
# - `filepath::String`: Path to the input .npz file.
# - `site_indices::Union{Vector{Index{Int64}}, Nothing}`: Optional site indices. If not provided, they will be generated.

# # Returns:
# - A LocalUnitaryMeasurementSettings object.
# """
# function import_unitary(filepath::String; site_indices::Union{Vector{Index{Int64}}, Nothing} = nothing)::LocalUnitaryMeasurementSettings
#     # Read the .npz file
#     data = npzread(filepath)

#     # Extract the local_unitary field
#     @assert haskey(data, "local_unitary") "Missing 'local_unitary' field in the NPZ file."

#     return LocalUnitaryMeasurementSettings(data["local_unitary"]; site_indices=site_indices)
# end


"""
    reduce_to_subsystem(settings::LocalUnitaryMeasurementSetting, subsystem::Vector{Int})

Reduce a `LocalUnitaryMeasurementSetting` object to a specified subsystem.

# Arguments
- `settings::LocalUnitaryMeasurementSetting`: The original measurement settings object.
- `subsystem::Vector{Int}`: A vector of site indices (1-based) specifying the subsystem to retain.

# Returns
A new `LocalUnitaryMeasurementSetting` object corresponding to the specified subsystem.
"""
function reduce_to_subsystem(
    settings::LocalUnitaryMeasurementSetting,
    subsystem::Vector{Int}
)::LocalUnitaryMeasurementSetting
    # Validate the subsystem
    @assert all(x -> x >= 1 && x <= settings.N, subsystem) "Subsystem indices must be between 1 and N."
    @assert length(unique(subsystem)) == length(subsystem) "Subsystem indices must be unique."

    # Extract the reduced local unitary and site indices
    reduced_unitary = settings.local_unitary[subsystem]
    reduced_indices = settings.site_indices[subsystem]

    # Create the new LocalUnitaryMeasurementSetting object
    return LocalUnitaryMeasurementSetting(
        length(subsystem),  # New N is the size of the subsystem
        reduced_unitary,
        reduced_indices
    )
end
