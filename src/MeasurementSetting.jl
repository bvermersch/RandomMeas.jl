# Copyright (c) 2024 Benoît Vermersch and Andreas Elben
# SPDX-License-Identifier: Apache-2.0
# http://www.apache.org/licenses/LICENSE-2.0

"""
    ComputationalBasisMeasurementSetting(N; site_indices=nothing)

Create a `ComputationalBasisMeasurementSetting` for `N` sites. This setting corresponds to measurement in the computational basis.

# Arguments
- `N::Int`: Number of sites (qubits).
- `site_indices::Union{Vector{Index{Int64}}, Nothing}` (optional): Site indices. If `nothing`, they are automatically generated.

# Returns
A `ComputationalBasisMeasurementSetting` object.

# Example
```julia
setting = ComputationalBasisMeasurementSetting(4)
```
"""
function ComputationalBasisMeasurementSetting(N::Int; site_indices::Union{Vector{Index{Int64}}, Nothing} = nothing)
    site_indices = site_indices === nothing ? siteinds("Qubit", N) : site_indices
    return ComputationalBasisMeasurementSetting(N, site_indices)
end


"""
    LocalUnitaryMeasurementSetting(basis_transformation_array; site_indices=nothing)

Create a `LocalUnitaryMeasurementSetting` object from an `N × 2 × 2` array of unitary matrices.

# Arguments
- `basis_transformation_array::Array{ComplexF64, 3}`: An `N × 2 × 2` array of unitary matrices.
- `site_indices::Union{Vector{Index{Int64}}, Nothing}` (optional): Site indices. If `nothing`, they are automatically generated.

# Returns
A `LocalUnitaryMeasurementSetting` object.

# Example
```julia
unitary_array = rand(ComplexF64, 4, 2, 2)
setting = LocalUnitaryMeasurementSetting(unitary_array)
```
"""
function LocalUnitaryMeasurementSetting(basis_transformation_array::Array{ComplexF64, 3}; site_indices::Union{Vector{Index{Int64}}, Nothing} = nothing)
    # Extract dimensions
    N, rows, cols = size(basis_transformation_array)
    @assert rows == 2 && cols == 2 "Unitary matrices must have size 2x2."

    # Generate or validate site indices
    site_indices = site_indices === nothing ? siteinds("Qubit", N) : site_indices

    # Convert the array into ITensors
    basis_transformation = [ITensor(basis_transformation_array[n, :, :], site_indices[n]', site_indices[n]) for n in 1:N]

    # Call the main constructor
    return LocalUnitaryMeasurementSetting(N, basis_transformation, site_indices)
end

"""
    LocalUnitaryMeasurementSetting(N; site_indices=nothing, ensemble=Haar)

Create a `LocalUnitaryMeasurementSetting` object by randomly sampling local unitary operators.

# Arguments
- `N::Int`: Number of sites (qubits).
- `site_indices::Union{Vector{Index{Int64}}, Nothing}` (optional): Site indices. If `nothing`, they are automatically generated.
- `ensemble::UnitaryEnsemble`: Type of random unitary (`Haar`, `Pauli`, `Identity`).

# Returns
A `LocalUnitaryMeasurementSetting` object.

# Example
```julia
setting = LocalUnitaryMeasurementSetting(4, ensemble=Haar)
```
"""
function LocalUnitaryMeasurementSetting(
    N::Int;
    site_indices::Union{Vector{Index{Int64}}, Nothing} = nothing,
    ensemble::UnitaryEnsemble = Haar
)
    # Generate site indices if not provided
    site_indices = site_indices === nothing ? siteinds("Qubit", N) : site_indices

    # Generate local unitary
    basis_transformation = [get_rotation(site_indices[j], ensemble) for j in 1:N]

    # Call the main constructor
    return LocalUnitaryMeasurementSetting(N, basis_transformation, site_indices)
end



"""
    ShallowUnitaryMeasurementSetting(N, depth; site_indices=nothing)

Create a `ShallowUnitaryMeasurementSetting` object by generating a random quantum circuit.

# Arguments
- `N::Int`: Number of sites (qubits).
- `depth::Int`: Depth of the random circuit.
- `site_indices::Union{Vector{Index{Int64}}, Nothing}` (optional): Site indices. If `nothing`, they are automatically generated.

# Returns
A `ShallowUnitaryMeasurementSetting` object.

# Example
```julia
setting = ShallowUnitaryMeasurementSetting(4, 3)
```
"""
function ShallowUnitaryMeasurementSetting(
    N::Int,depth::Int;
    site_indices::Union{Vector{Index{Int64}}, Nothing} = nothing,
)
    # Generate site indices if not provided
    site_indices = site_indices === nothing ? siteinds("Qubit", N) : site_indices

    # Generate random_circuit
    basis_transformation = random_circuit(site_indices,depth)
    K = length(basis_transformation)

    # Call the main constructor
    return ShallowUnitaryMeasurementSetting(N, K, basis_transformation, site_indices)
end

"""
    get_rotation(site_index, ensemble=Haar)

Generate a single-qubit unitary sampled from a specified ensemble.

# Arguments
- `site_index::Index{Int64}`: Site index.
- `ensemble::UnitaryEnsemble`: Type of unitary ensemble (`Haar`, `Pauli`, `Identity`).

# Returns
An `ITensor` representing the unitary transformation.

# Example
```julia
U = get_rotation(site_index, Pauli)
```
"""
function get_rotation(site_index::Index{Int64}, ensemble::UnitaryEnsemble = Haar)
    r_matrix = zeros(ComplexF64, (2, 2))
    if ensemble == Haar
        return op("RandomUnitary", site_index)
    elseif ensemble == Pauli
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
    elseif ensemble == Identity
        r_matrix[1, 1] = 1
        r_matrix[2, 2] = 1
        return ITensor(r_matrix, site_index', site_index)
    else
        throw(ArgumentError("Invalid ensemble: $ensemble"))
    end
end




"""
    reduce_to_subsystem(settings, subsystem)

Reduce a `LocalMeasurementSetting` object to a specified subsystem.

# Arguments
- `settings::LocalMeasurementSetting`: The original measurement settings object.
- `subsystem::Vector{Int}`: A vector of site indices (1-based) specifying the subsystem to retain.

# Returns
A new `LocalMeasurementSetting` object corresponding to the specified subsystem.

# Example
```julia
reduced_setting = reduce_to_subsystem(full_setting, [1, 3])
```
"""
function reduce_to_subsystem(
    settings::LocalMeasurementSetting,
    subsystem::Vector{Int}
)::LocalMeasurementSetting
    # Validate the subsystem
    @assert all(1 .≤ subsystem .≤ settings.N) "Subsystem indices must be between 1 and N."
    @assert length(unique(subsystem)) == length(subsystem) "Subsystem indices must be unique."

    # Extract the reduced local unitary and site indices
    reduced_unitary = settings.basis_transformation[subsystem]
    reduced_indices = settings.site_indices[subsystem]
    reduced_N = length(subsystem)

    # Create the new LocalUnitaryMeasurementSetting object
    return typeof(settings)(reduced_N, reduced_unitary, reduced_indices)
end


"""
    export_LocalUnitaryMeasurementSetting(ms, filepath)

Export the unitary in a LocalUnitaryMeasurementSetting object to an .npz file with a single field: basis_transformation.

# Arguments
- `ms::LocalUnitaryMeasurementSetting`: The measurement settings to export.
- `filepath::String`: Path to the output .npz file.
"""
function export_LocalUnitaryMeasurementSetting(ms::LocalUnitaryMeasurementSetting, filepath::String)
    # Prepare the basis_transformation array for export
    basis_transformation = Array{ComplexF64}(undef, ms.N, 2, 2)
    for n in 1:ms.N
            basis_transformation[n, :, :] = Array(ms.basis_transformation[n], ms.site_indices[n]', ms.site_indices[n])
    end

    # Write to the .npz file
    npzwrite(filepath, Dict("basis_transformation" => basis_transformation))
    println("Exported to NPZ file: $filepath")
end

"""
    import_LocalUnitaryMeasurementSetting(filepath; site_indices=nothing)

Import unitary from an .npz file and create a LocalUnitaryMeasurementSetting object.

# Arguments
- `filepath::String`: Path to the input .npz file.
- `site_indices::Union{Vector{Index{Int64}}, Nothing}` (optional): Site indices. If not provided, they will be generated.

# Returns
A `LocalUnitaryMeasurementSetting` object.
"""
function import_LocalUnitaryMeasurementSetting(filepath::String; site_indices::Union{Vector{Index{Int64}}, Nothing} = nothing)::LocalUnitaryMeasurementSetting
    # Read the .npz file
    data = npzread(filepath)

    # Extract the basis_transformation field
    @assert haskey(data, "basis_transformation") "Missing 'basis_transformation' field in the NPZ file."

    return LocalUnitaryMeasurementSetting(data["basis_transformation"]; site_indices=site_indices)
end
