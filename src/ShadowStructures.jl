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
