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
        for i in 1:N
            inds_i = inds(local_unitary[i])
            @assert length(inds_i) == 2 "Local unitary for site $i must have exactly two indices."
            @assert site_indices[i] in inds_i "Local unitary for site $i must contain the unprimed site index."
            @assert prime(site_indices[i]) in inds_i "Local unitary for site $i must contain the primed site index."
        end
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

    # Inner constructor with checks:
    function MeasurementData{T}(
        N::Int, NM::Int, measurement_results::Array{Int,2}, measurement_setting::T
    ) where T
        @assert size(measurement_results, 1) == NM "measurement_results must have NM rows."
        @assert size(measurement_results, 2) == N "measurement_results must have N columns."
        if measurement_setting !== nothing
            @assert measurement_setting isa AbstractMeasurementSetting "measurement_setting must be a subtype of AbstractMeasurementSetting."
            @assert measurement_setting.N == N "measurement_setting.N ($(measurement_setting.N)) must match N ($N)."
        end
        new(N, NM, measurement_results, measurement_setting)
    end
end
#Simplified constructor for type inference
MeasurementData(N::Int, NM::Int, measurement_results::Array{Int,2}, measurement_setting::T) where T = MeasurementData{T}(N, NM, measurement_results, measurement_setting)

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

    function MeasurementProbability{T}(N::Int, measurement_probability::ITensor, measurement_setting::T) where T
        @assert length(inds(measurement_probability)) == N "The ITensor must have exactly N indices (got $(length(inds(measurement_probability))) vs N = $N)."
        if measurement_setting !== nothing
            @assert length(measurement_setting.site_indices) == N "The length of measurement_setting.site_indices must equal N (got $(length(measurement_setting.site_indices)) vs N = $N)."
            @assert inds(measurement_probability) == measurement_setting.site_indices "The indices in the measurement_probability ITensor must match measurement_setting.site_indices."
        end
        new{T}(N, measurement_probability, measurement_setting)
    end
end
#Simplified constructor for type inference
MeasurementProbability(N::Int, measurement_probability::ITensor, measurement_setting::T) where T = MeasurementProbability{T}(N, measurement_probability, measurement_setting)


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
        # Inner constructor with dimension and consistency checks.
        function MeasurementGroup{T}(N::Int, NU::Int, NM::Int, measurements::Vector{MeasurementData{T}}) where T
            @assert length(measurements) == NU "Expected $NU MeasurementData objects, but got $(length(measurements))."
            for (i, m) in enumerate(measurements)
                @assert m.N == N "MeasurementData object at index $i has inconsistent number of sites: expected $N, got $(m.N)."
                @assert m.NM == NM "MeasurementData object at index $i has inconsistent number of measurements: expected $NM, got $(m.NM)."
            end
            new{T}(N, NU, NM, measurements)
        end
end
#Simplified constructor for type inference
MeasurementGroup(N::Int, NU::Int, NM::Int, measurements::Vector{MeasurementData{T}}) where T = MeasurementGroup{T}(N, NU, NM, measurements)


# Abstract Shadow Type
"""
    AbstractShadow

An abstract type representing a general classical shadow.
Subtypes should implement specific shadow methodologies, such as dense or factorized shadows.
"""
abstract type AbstractShadow end


# Factorized Classical Shadow Constructor
"""
    FactorizedShadow

A struct representing a factorized classical shadow for a quantum system.

# Fields
- `shadow_data::Vector{ITensor}`: Array of `N` ITensors, each 2x2, representing the factorized shadow for each qubit/site.
- `N::Int`: Number of qubits/sites.
- `ξ::Vector{Index{Int64}}`: Vector of site indices corresponding to the qubits/sites.

# Constructor
`FactorizedShadow(shadow_data::Vector{ITensor}, N::Int, ξ::Vector{Index{Int64}})`
"""
struct FactorizedShadow <: AbstractShadow
    shadow_data::Vector{ITensor}  # Array of N ITensors, each 2x2
    N::Int                             # Number of qubits/sites
    ξ::Vector{Index{Int64}}            # Vector of site indices

    """
    Create a `FactorizedShadow` object with validation.

    # Arguments
    - `shadow_data::Vector{ITensor}`: Array of ITensors representing the factorized shadow for each qubit/site.
    - `N::Int`: Number of qubits/sites.
    - `ξ::Vector{Index{Int64}}`: Vector of site indices corresponding to the qubits/sites.

    # Throws
    - `AssertionError` if the dimensions of `shadow_data`, `ξ` do not match `N`.
    """
    function FactorizedShadow(shadow_data::Vector{ITensor}, N::Int, ξ::Vector{Index{Int64}})
        @assert length(shadow_data) == N "Length of shadow_data must match N."
        @assert length(ξ) == N "Length of site indices ξ must match N."
        new(shadow_data, N, ξ)
    end
end


# Dense Classical Shadow: Represents a 2^N x 2^N ITensor
"""
    DenseShadow

A struct representing a dense classical shadow, stored as a single ITensor.

# Fields
- `shadow_data::ITensor`: The dense shadow as an ITensor with legs `ξ` and `ξ'`.
- `N::Int`: Number of qubits/sites.
- `ξ::Vector{Index{Int64}}`: Vector of site indices.

# Constructor
`DenseShadow(shadow_data::ITensor, N::Int, ξ::Vector{Index{Int64}})`
"""
struct DenseShadow <: AbstractShadow
    shadow_data::ITensor
    N::Int
    ξ::Vector{Index{Int64}}

    """
    Create a `DenseShadow` object with validation.

    # Arguments
    - `shadow_data::ITensor`: The dense shadow tensor.
    - `N::Int`: Number of qubits/sites.
    - `ξ::Vector{Index{Int64}}`: Vector of site indices.

    # Throws
    - `AssertionError` if dimensions of `ξ` do not match `N`.
    """
    function DenseShadow(shadow_data::ITensor, N::Int, ξ::Vector{Index{Int64}})
        @assert length(ξ) == N "Length of site indices ξ must match N."
        new(shadow_data, N, ξ)
    end
end