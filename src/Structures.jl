# Abstract Base Type for Measurement Setting
"""
Abstract type representing general a measurement setting.
Specific implementations (e.g., LocalUnitaryMeasurementSetting) should subtype this.
"""
abstract type AbstractMeasurementSetting end

abstract type LocalMeasurementSetting <: AbstractMeasurementSetting end

# Local Unitary Measurement Setting
"""
LocalUnitaryMeasurementSetting

A struct representing local unitary measurement settings for quantum systems.

# Fields:
- `N::Int`: Number of sites (qubits).
- `local_unitary::Vector{ITensor}`: A local unitary represented by N 2x2 ITensors.
- `site_indices::Vector{Index{Int64}}`: Vector of site indices of length N.
"""
struct LocalUnitaryMeasurementSetting <: LocalMeasurementSetting
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

# Concrete type for computational basis measurement settings:
# It contains only N, site_indices, and a local_unitary vector composed of identity ITensors.
struct ComputationalBasisMeasurementSetting <: LocalMeasurementSetting
    N::Int                              # Number of sites
    local_unitary::Vector{ITensor}      # Vector of N identity ITensors
    site_indices::Vector{Index{Int64}}  # Vector of site indices (length N)

    function ComputationalBasisMeasurementSetting(N::Int, site_indices::Vector{Index{Int64}})
        @assert length(site_indices) == N "Length of site_indices must match N."
        # Create a vector of identity ITensors, one for each site.
        local_unitary = Vector{ITensor}(undef, N)
        for i in 1:N
            local_unitary[i] = delta(site_indices[i], prime(site_indices[i]))
        end
        new(N, local_unitary, site_indices)
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
struct MeasurementData{T<:Union{Nothing, AbstractMeasurementSetting}}
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
MeasurementData(N::Int, NM::Int, measurement_results::Array{Int,2}, measurement_setting::T) where T<:Union{Nothing, AbstractMeasurementSetting} = MeasurementData{T}(N, NM, measurement_results, measurement_setting)

"""
    MeasurementProbability{T}

A container for measurement Probability and setting used in quantum experiments.

# Fields
- `N::Int`: Number of sites (qubits).
- `measurement_probability::ITensor representing of Born Probability.
- `measurement_setting::T`: Measurement setting of type `T` or `nothing` if not provided.
- `site_indices::Vector{Index{Int64}}`  # Vector of site indices (length N)

# Type Parameter
- `T`: The type of `measurement_setting`. This can be any subtype of `AbstractMeasurementSetting` or `Nothing` if no settings are provided.

# Usage
The `MeasurementProbability` struct can be constructed using either a `MeasurementData` object or directly from a quantum state (MPS/MPO) and measurement settings.
"""
struct MeasurementProbability{T<:Union{AbstractMeasurementSetting, Nothing}}
    N::Int                              # Number of sites (qubits)
    measurement_probability::ITensor # Measurement Probability
    measurement_setting::T            # Measurement settings (or nothing if not provided)
    site_indices::Vector{Index{Int64}}

    function MeasurementProbability{T}(N::Int, measurement_probability::ITensor, measurement_setting::T,site_indices::Vector{Index{Int64}}) where T
        @assert length(inds(measurement_probability)) == N "The ITensor must have exactly N indices (got $(length(inds(measurement_probability))) vs N = $N)."
        @assert length(site_indices) == N "The length of measurement_setting.site_indices must equal N (got $(length(site_indices)) vs N = $N)."
        if measurement_setting !== nothing
            @assert site_indices == measurement_setting.site_indices "The indices in the measurement_probability ITensor must match measurement_setting.site_indices."
        end
        new{T}(N, measurement_probability, measurement_setting,site_indices)
    end
end
#Simplified constructor for type inference
MeasurementProbability(N::Int, measurement_probability::ITensor, measurement_setting::T,site_indices::Vector{Index{Int64}}) where T<:Union{Nothing, AbstractMeasurementSetting} = MeasurementProbability{T}(N, measurement_probability, measurement_setting,site_indices)


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
struct MeasurementGroup{T<:Union{AbstractMeasurementSetting, Nothing}}
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
MeasurementGroup(N::Int, NU::Int, NM::Int, measurements::Vector{MeasurementData{T}}) where T<:Union{Nothing, AbstractMeasurementSetting} = MeasurementGroup{T}(N, NU, NM, measurements)


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
- `site_indices::Vector{Index{Int64}}`: Vector of site indices corresponding to the qubits/sites.

# Constructor
`FactorizedShadow(shadow_data::Vector{ITensor}, N::Int, ξ::Vector{Index{Int64}})`
"""
struct FactorizedShadow <: AbstractShadow
    shadow_data::Vector{ITensor}  # Array of N ITensors, each 2x2
    N::Int                             # Number of qubits/sites
    site_indices::Vector{Index{Int64}}            # Vector of site indices

    """
    Create a `FactorizedShadow` object with validation.

    # Arguments
    - `shadow_data::Vector{ITensor}`: Array of ITensors representing the factorized shadow for each qubit/site.
    - `N::Int`: Number of qubits/sites.
    - `site_indices::Vector{Index{Int64}}`: Vector of site indices corresponding to the qubits/sites.

    # Throws
    - `AssertionError` if the dimensions of `shadow_data`, `ξ` do not match `N`.
    """
    function FactorizedShadow(shadow_data::Vector{ITensor}, N::Int, site_indices::Vector{Index{Int64}})
        @assert length(shadow_data) == N "Length of shadow_data must match N."
        @assert length(site_indices) == N "Length of site indices ξ must match N."
        for i in 1:N
            # Get the indices of the ITensor for site i.
            inds_i = inds(shadow_data[i])
            @assert length(inds_i) == 2 "ITensor at index $i must have exactly two indices."
            # Assert that the indices exactly match: first the unprimed index, then its primed version.
            @assert site_indices[i] in inds_i "ITensor at index $i: first index must be equal to the corresponding site index."
            @assert prime(site_indices[i]) in inds_i "ITensor at index $i: second index must be equal to the primed site index."
        end
        new(shadow_data, N, site_indices)
    end
end


# Dense Classical Shadow: Represents a 2^N x 2^N ITensor
"""
    DenseShadow

A struct representing a dense classical shadow, stored as a single ITensor.

# Fields
- `shadow_data::ITensor`: The dense shadow as an ITensor with legs `site_indices` and `site_indices'`.
- `N::Int`: Number of qubits/sites.
- `site_indices::Vector{Index{Int64}}`: Vector of site indices.

# Constructor
`DenseShadow(shadow_data::ITensor, N::Int, site_indices::Vector{Index{Int64}})`
"""
struct DenseShadow <: AbstractShadow
    shadow_data::ITensor
    N::Int
    site_indices::Vector{Index{Int64}}
    """
    Create a `DenseShadow` object with validation.

    # Arguments
    - `shadow_data::ITensor`: The dense shadow tensor.
    - `N::Int`: Number of qubits/sites.
    - `site_indices::Vector{Index{Int64}}`: Vector of site indices.

    # Throws
    - `AssertionError` if dimensions of `site_indices` do not match `N`.
    """
    function DenseShadow(shadow_data::ITensor, N::Int, site_indices::Vector{Index{Int64}})
        @assert length(site_indices) == N "Length of site indices site_indices must match N."
        unprimed = [i for i in inds(shadow_data) if plev(i)==0]
        primed = [i for i in inds(shadow_data) if plev(i)==1]
        @assert length(unprimed)==N "Expected N unprimed indices"
        @assert length(primed)==N "Expected N primed indices"
        @assert Set(unprimed)==Set(site_indices) "The unprimed indices do not match the site_indices"
        @assert Set(primed)==Set(map(prime, site_indices)) "The primed indices do not match the primed version of site_indices"
        new(shadow_data, N, site_indices)
    end
end
