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
    struct MeasurementData{T}

A container for measurement data and setting used in quantum experiments.

# Fields
- `N::Int`: Number of sites (qubits).
- `NM::Int`: Number of measurements per setting.
- `measurement_results::Array{Int, 2}`: A 3D array of binary measurement results with dimensions `(NM, N)`.
- `measurement_setting::T`: Measurement setting of type `T` or `nothing` if not provided.

# Type Parameter
- `T`: The type of `measurement_setting`. This can be any subtype of `AbstractMeasurementSetting` or `Nothing` if no settings are provided.

# Usage
The `MeasurementData` struct is typically constructed using the provided constructor functions.
"""
struct MeasurementData{T}
    N::Int                              # Number of sites (qubits)
    NM::Int                             # Number of measurements per setting
    measurement_results::Array{Int, 2} # Binary measurement results (size: NM x N)
    measurement_setting::T   # Measurement setting (or nothing if not provided)
end

"""
    MeasurementProbability{T}

A container for measurement Probability and setting used in quantum experiments.

# Fields
- `N::Int`: Number of sites (qubits).
- `measurement_probability::ITensor representing of Born Probability.
- `measurement_setting::T`: Measurement setting of type `T` or `nothing` if not provided.

# Type Parameter
- `T`: The type of `measurement_setting`. This can be any subtype of `AbstractMeasurementSetting` or `Nothing` if no settings are provided.

# Usage
The `MeasurementProbability` struct can be constructed using either a `MeasurementData` object or directly from a quantum state (MPS/MPO) and measurement settings.
"""
struct MeasurementProbability{T}
    N::Int                              # Number of sites (qubits)
    measurement_probability::ITensor # Measurement Probability 
    measurement_setting::T             # Measurement settings (or nothing if not provided)
end


"""
    struct MeasurementGroup{T}

A container for a group of NU measurement data used in quantum experiments.

# Fields
- `N::Int`: Number of sites (qubits).
- `NU::Int`: Number of MeasurementData objects
- `measurements::Vector{MeasurementData{T}}

# Type Parameter
- `T`: The type of `MeasurementGroup`. This can be any subtype of `AbstractMeasurementSetting` or `Nothing` if no settings are provided.

# Usage
The `MeasurementData` struct is typically constructed using the provided constructor functions.
"""
struct MeasurementGroup{T}
    N::Int                              # Number of sites (qubits)
    NU::Int                             # Number of measurementData
    NM::Int                             # Number of projectivemeasurements per RU
    measurements::Vector{MeasurementData{T}} # NU MeasurementsData objects
end

