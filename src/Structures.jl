# Abstract Base Type for Measurement Settings
"""
    AbstractMeasurementSetting

An abstract type representing a general measurement setting.
Concrete implementations (e.g. `LocalUnitaryMeasurementSetting`,
`ComputationalBasisMeasurementSetting`) should subtype this.
"""
abstract type AbstractMeasurementSetting end

"""
    LocalMeasurementSetting

An abstract type for measurement settings that correspond to local (i.e. single qubit/site) measurements.
"""
abstract type LocalMeasurementSetting <: AbstractMeasurementSetting end


# ---------------------------------------------------------------------------
# Local Unitary Measurement Setting
# ---------------------------------------------------------------------------
"""
    LocalUnitaryMeasurementSetting(N, local_unitary, site_indices)

A measurement setting where each qubit is specified by a single-qubit rotation.
Rotates from the computational basis into the measurement basis.

# Fields
- `N::Int`: Number of sites (qubits).
- `local_unitary::Vector{ITensor}`: A vector of `N` ITensors representing the local unitary basis rotations.
- `site_indices::Vector{Index{Int64}}`: A vector of site indices of length `N`.

# Constraints
- `N == length(local_unitary) == length(site_indices)`.
- Each ITensor in `local_unitary` has exactly **two indices**:
  - One unprimed (`site_indices[i]`)
  - One primed (`prime(site_indices[i])`).
"""
struct LocalUnitaryMeasurementSetting <: LocalMeasurementSetting
    N::Int
    local_unitary::Vector{ITensor}
    site_indices::Vector{Index{Int64}}

    function LocalUnitaryMeasurementSetting(
        N::Int, local_unitary::Vector{ITensor}, site_indices::Vector{Index{Int64}}
    )
        @assert length(local_unitary) == N "Expected $N ITensors in local_unitary, got $(length(local_unitary))."
        @assert length(site_indices) == N "Expected $N site_indices, got $(length(site_indices))."

        for (i, U) in enumerate(local_unitary)
            inds_i = inds(U)
            @assert length(inds_i) == 2 "ITensor at site $i must have exactly two indices, got $(length(inds_i))."
            @assert site_indices[i] in inds_i "ITensor at site $i must contain the unprimed site index."
            @assert prime(site_indices[i]) in inds_i "ITensor at site $i must contain the primed site index."
        end

        return new(N, local_unitary, site_indices)
    end
end


# ---------------------------------------------------------------------------
# Computational Basis Measurement Setting
# ---------------------------------------------------------------------------
"""
    ComputationalBasisMeasurementSetting

A struct representing computational basis measurement settings for quantum systems.
This setting uses the computational basis, so that each local unitary is by construction simply the identity operator.

# Fields
- `N::Int`: Number of sites (qubits).
- `local_unitary::Vector{ITensor}`: A vector of N identity ITensors.
- `site_indices::Vector{Index{Int64}}`: A vector of site indices (length N).

# Constraints
- `N == length(site_indices)`.
"""
struct ComputationalBasisMeasurementSetting <: LocalMeasurementSetting
    N::Int                              # Number of sites
    local_unitary::Vector{ITensor}      # Vector of N identity ITensors
    site_indices::Vector{Index{Int64}}  # Vector of site indices (length N)

    function ComputationalBasisMeasurementSetting(N::Int, site_indices::Vector{Index{Int64}})
        @assert length(site_indices) == N "Expected $N site_indices, got $(length(site_indices))."
        # Create a vector of identity ITensors for each site.
        local_unitary = [delta(site_indices[i], prime(site_indices[i])) for i in 1:N]

        new(N, local_unitary, site_indices)
    end
end

# ---------------------------------------------------------------------------
# Shallow Unitary Measurement Setting
# ---------------------------------------------------------------------------
"""
    ShallowUnitaryMeasurementSetting

A struct representing measurement settings which is, for each qubit, specified through a single qubit rotation, rotating from the computational basis into the measurement basis.

# Fields
- `N::Int`: Number of sites (qubits).
- 'K::Int`: Number of gates that creates the shallow_unitary
- `localunitary::Vector{ITensor}`: A vector of Ngates representing the shallow unitary
- `site_indices::Vector{Index{Int64}}`: A vector of site indices of length N.

# Constructor
Creates a `ShallowUnitaryMeasurementSetting` object after validating that:
- The length of `local_unitary` equals `K`
- The length of `site_indices` equals `N`.
"""
struct ShallowUnitaryMeasurementSetting <: AbstractMeasurementSetting
    N::Int                              # Number of sites
    K::Int                               # Number of gates
    local_unitary::Vector{ITensor}      # Vector of K 2x2 ITensors (one and two qubit gates)
    site_indices::Vector{Index{Int64}}  # Vector of site indices (length N)

    function ShallowUnitaryMeasurementSetting(
        N::Int, K::Int,local_unitary::Vector{ITensor}, site_indices::Vector{Index{Int64}}
    )
        @assert length(local_unitary) == K "Expected $K ITensors in local_unitary, got $(length(shallow_unitary))."
        @assert length(site_indices) == N "Expected $N site_indices, got $(length(site_indices))."
        return new(N, K, local_unitary, site_indices)
    end
end



# ---------------------------------------------------------------------------
# Measurement Data
# ---------------------------------------------------------------------------
"""
    MeasurementData{T}

A container for measurement data and settings obtained in actual or simulated quantum experiments.

# Fields
- `N::Int`: Number of sites (qubits).
- `NM::Int`: Number of measurements per setting.
- `measurement_results::Array{Int, 2}`: A 2D array of binary measurement results with dimensions `(NM, N)`.
- `measurement_setting::T`: A measurement setting of type `T` (subtype of `AbstractMeasurementSetting`) or `nothing`.

# Type Parameter
- `T`: The type of the measurement setting, constrained to `Union{Nothing, AbstractMeasurementSetting}`.

# Usage
Typically constructed via the provided constructors.
"""
struct MeasurementData{T<:Union{Nothing, AbstractMeasurementSetting}}
    N::Int                              # Number of sites (qubits)
    NM::Int                             # Number of measurements per setting
    measurement_results::Array{Int, 2}  # Measurement results (dimensions: NM x N)
    measurement_setting::T              # Measurement setting (or nothing)

    # Inner constructor with validations.
    function MeasurementData{T}(
        N::Int, NM::Int, measurement_results::Array{Int,2}, measurement_setting::T
    ) where T
        @assert size(measurement_results, 1) == NM "measurement_results must have NM ($NM) rows; got $(size(measurement_results, 1))."
        @assert size(measurement_results, 2) == N "measurement_results must have N ($N) columns; got $(size(measurement_results, 2))."
        if measurement_setting !== nothing
            @assert measurement_setting isa AbstractMeasurementSetting "measurement_setting must be a subtype of AbstractMeasurementSetting."
            @assert measurement_setting.N == N "Measurement setting has N = $(measurement_setting.N), expected $N."
        end
        new(N, NM, measurement_results, measurement_setting)
    end
end

# Simplified outer constructor for type inference.
MeasurementData(N::Int, NM::Int, measurement_results::Array{Int,2}, measurement_setting::T) where T<:Union{Nothing, AbstractMeasurementSetting} =
    MeasurementData{T}(N, NM, measurement_results, measurement_setting)


"""
    MeasurementProbability{T}

A container for measurement probabilities and settings obtained either f.

# Fields
- `N::Int`: Number of sites (qubits).
- `measurement_probability::ITensor`: An ITensor representing Born probability.
- `measurement_setting::T`: A measurement setting of type `T` or `nothing`.
- `site_indices::Vector{Index{Int64}}`: A vector of site indices (length N).

# Type Parameter
- `T`: The type of measurement setting, constrained to `Union{Nothing, AbstractMeasurementSetting}`.

# Usage
Constructed either from measurement data or directly from quantum states.
"""
struct MeasurementProbability{T<:Union{AbstractMeasurementSetting, Nothing}}
    N::Int                              # Number of sites (qubits)
    measurement_probability::ITensor    # ITensor representing measurement probability
    measurement_setting::T              # Measurement setting (or nothing)
    site_indices::Vector{Index{Int64}}  # Vector of site indices (length N)

    function MeasurementProbability{T}(N::Int, measurement_probability::ITensor, measurement_setting::T, site_indices::Vector{Index{Int64}}) where T
        @assert length(inds(measurement_probability)) == N "Expected ITensor to have N ($N) indices, got $(length(inds(measurement_probability)))."
        @assert length(site_indices) == N "Expected site_indices to have length N ($N), got $(length(site_indices))."
        if measurement_setting !== nothing
            @assert site_indices == measurement_setting.site_indices "site_indices must match measurement_setting.site_indices."
        end
        new(N, measurement_probability, measurement_setting, site_indices)
    end
end

# Simplified outer constructor for type inference.
MeasurementProbability(N::Int, measurement_probability::ITensor, measurement_setting::T, site_indices::Vector{Index{Int64}}) where T<:Union{Nothing, AbstractMeasurementSetting} =
    MeasurementProbability{T}(N, measurement_probability, measurement_setting, site_indices)


# ---------------------------------------------------------------------------
# Measurement Group
# ---------------------------------------------------------------------------
"""
    MeasurementGroup{T}

A container for a group of measurement data objects used in actual or simulated quantum experiments.

# Fields
- `N::Int`: Number of sites (qubits).
- `NU::Int`: Number of measurement data objects.
- `NM::Int`: Number of measurements per setting.
- `measurements::Vector{MeasurementData{T}}`: A vector of measurement data objects.

# Type Parameter
- `T`: The type of measurement setting for each measurement data object,
  constrained to `Union{Nothing, AbstractMeasurementSetting}`.

# Usage
Typically constructed via the provided constructors.
"""
struct MeasurementGroup{T<:Union{AbstractMeasurementSetting, Nothing}}
    N::Int                              # Number of sites (qubits)
    NU::Int                             # Number of measurement data objects
    NM::Int                             # Number of measurements per setting
    measurements::Vector{MeasurementData{T}}  # Vector of MeasurementData objects

    # Inner constructor with validations.
    function MeasurementGroup{T}(N::Int, NU::Int, NM::Int, measurements::Vector{MeasurementData{T}}) where T
        @assert length(measurements) == NU "Expected $NU MeasurementData objects, got $(length(measurements))."
        for (i, m) in enumerate(measurements)
            @assert m.N == N "MeasurementData at index $i has N = $(m.N), expected $N."
            @assert m.NM == NM "MeasurementData at index $i has NM = $(m.NM), expected $NM."
        end
        new(N, NU, NM, measurements)
    end
end

# Simplified outer constructor for type inference.
MeasurementGroup(N::Int, NU::Int, NM::Int, measurements::Vector{MeasurementData{T}}) where T<:Union{Nothing, AbstractMeasurementSetting} =
    MeasurementGroup{T}(N, NU, NM, measurements)


# ---------------------------------------------------------------------------
# Abstract Shadow Type
# ---------------------------------------------------------------------------
"""
    AbstractShadow

An abstract type representing a general classical shadow.
Concrete subtypes should implement specific shadow methodologies,
such as factorized or dense shadows.
"""
abstract type AbstractShadow end


# ---------------------------------------------------------------------------
# Factorized Classical Shadow
# ---------------------------------------------------------------------------
"""
    FactorizedShadow

A struct representing a factorized classical shadow which can be represented as a tensor product of single qubit shadows.

# Fields
- `shadow_data::Vector{ITensor}`: A vector of ITensors (each 2×2) representing the shadow for each qubit/site.
- `N::Int`: Number of qubits/sites.
- `site_indices::Vector{Index{Int64}}`: A vector of site indices (length N).

# Constructor
`FactorizedShadow(shadow_data::Vector{ITensor}, N::Int, site_indices::Vector{Index{Int64}})`
validates that:
- The length of `shadow_data` and `site_indices` equals `N`.
- Each ITensor in `shadow_data` has exactly two indices,
  which include the corresponding unprimed and primed site index.
"""
struct FactorizedShadow <: AbstractShadow
    shadow_data::Vector{ITensor}       # Vector of ITensors, one per site.
    N::Int                             # Number of sites.
    site_indices::Vector{Index{Int64}} # Vector of site indices.

    function FactorizedShadow(shadow_data::Vector{ITensor}, N::Int, site_indices::Vector{Index{Int64}})
        @assert length(shadow_data) == N "Expected shadow_data length to be N ($N), got $(length(shadow_data))."
        @assert length(site_indices) == N "Expected site_indices length to be N ($N), got $(length(site_indices))."
        for i in 1:N
            inds_i = inds(shadow_data[i])
            @assert length(inds_i) == 2 "ITensor at index $i must have exactly two indices, got $(length(inds_i))."
            @assert site_indices[i] in inds_i "ITensor at index $i must contain the unprimed site index."
            @assert prime(site_indices[i]) in inds_i "ITensor at index $i must contain the primed site index."
        end
        new(shadow_data, N, site_indices)
    end
end


# ---------------------------------------------------------------------------
# Dense Classical Shadow
# ---------------------------------------------------------------------------
"""
    DenseShadow

A struct representing a dense classical shadow (a 2^N x 2^N matrix), stored as a single ITensor with 2N indices.

# Fields
- `shadow_data::ITensor`: An ITensor with 2N indices representing the dense shadow.
- `N::Int`: Number of sites (qubits).
- `site_indices::Vector{Index{Int64}}`: A vector of site indices (length N).

# Constructor
`DenseShadow(shadow_data::ITensor, N::Int, site_indices::Vector{Index{Int64}})`
validates that:
- `site_indices` has length N.
- `shadow_data` has exactly 2N indices.
- The set of unprimed indices in `shadow_data` matches `site_indices`.
- The set of primed indices in `shadow_data` matches `map(prime, site_indices)`.
"""
struct DenseShadow <: AbstractShadow
    shadow_data::ITensor
    N::Int
    site_indices::Vector{Index{Int64}}

    function DenseShadow(shadow_data::ITensor, N::Int, site_indices::Vector{Index{Int64}})
        @assert length(site_indices) == N "Expected site_indices length to be N ($N), got $(length(site_indices))."
        inds_all = inds(shadow_data)
        @assert length(inds_all) == 2*N "Expected ITensor to have 2N ($(2*N)) indices, got $(length(inds_all))."
        # Separate unprimed and primed indices.
        unprimed = [i for i in inds_all if plev(i)==0]
        primed   = [i for i in inds_all if plev(i)==1]
        @assert length(unprimed) == N "Expected N unprimed indices, got $(length(unprimed))."
        @assert length(primed) == N "Expected N primed indices, got $(length(primed))."
        @assert Set(unprimed) == Set(site_indices) "Unprimed indices do not match site_indices."
        @assert Set(primed) == Set(map(prime, site_indices)) "Primed indices do not match map(prime, site_indices)."
        new(shadow_data, N, site_indices)
    end
end


# ---------------------------------------------------------------------------
# Shallow Classical Shadow
# ---------------------------------------------------------------------------
"""
    ShallowShadow

A struct representing a shallow classical shadow, stored as a MPO ITensor object.

# Fields
- `shadow_data::MPOr`: An MPO representing the shallow shadow.
- `N::Int`: Number of sites (qubits).
- `site_indices::Vector{Index{Int64}}`: A vector of site indices (length N).

# Constructor
`ShallowShadow(shadow_data::MPO, N::Int, site_indices::Vector{Index{Int64}})`
validates that:
- `site_indices` has length N.
- `shadow_data` has exactly N Tensors, and the site indices match with site_indices
"""
struct ShallowShadow <: AbstractShadow
    shadow_data::MPO
    N::Int
    site_indices::Vector{Index{Int64}}

    function ShallowShadow(shadow_data::MPO, N::Int, site_indices::Vector{Index{Int64}})
        @assert length(site_indices) == N "Expected site_indices length to be N ($N), got $(length(site_indices))."
        #inds_all = inds(shadow_data)
        @assert site_indices == get_siteinds(shadow_data) "sites indices should match"
        new(shadow_data, N, site_indices)
    end
end
