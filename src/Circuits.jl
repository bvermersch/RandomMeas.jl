"""
    apply_depo_channel(ρ::MPO, p::Vector{Float64})

Apply a local depolarization channel to an MPO by modifying each site tensor according to the depolarization probability.

For each site, the channel acts as:

  ρ[i] → (1 - p[i]) * ρ[i] + (p[i] / 2) * (ρ[i] * δ(s, s') * δ(s, s'))

where δ(s, s') is the delta tensor that contracts the site index with its primed counterpart.

# Arguments
- `ρ::MPO`: The input Matrix Product Operator representing a density matrix.
- `p::Vector{Float64}`: A vector of depolarization probabilities, one per site.

# Returns
An MPO with the depolarization channel applied on each site.
"""
function apply_depo_channel(ρ::MPO, p::Vector{Float64})
    N = length(ρ)
    ξ = firstsiteinds(ρ; plev=0)
    ρ1 = copy(ρ)
    for i in 1:N
        s = ξ[i]
        X = ρ1[i] * δ(s, s')
        ρ1[i] = (1 - p[i]) * ρ1[i] + p[i] / 2 * X * δ(s, s')
    end
    return ρ1
end

"""
    apply_depo_channel(ψ::MPS, p::Vector{Float64})

Apply the local depolarization channel to an MPS by converting it to an MPO density matrix
(using the outer product) and then applying the depolarization channel.

# Arguments
- `ψ::MPS`: The input Matrix Product State representing a pure state.
- `p::Vector{Float64}`: A vector of depolarization probabilities, one per site.

# Returns
An MPO representing the depolarized density matrix.
"""
function apply_depo_channel(ψ::MPS, p::Vector{Float64})
    return apply_depo_channel(outer(ψ', ψ), p)
end

"""
    random_circuit(ξ::Vector{Index{Int64}}, depth::Int64)

Create a random circuit of the given depth. The function returns a vector of ITensors, each representing a gate in the circuit.

- If `depth == 0`, a single-qubit random unitary is applied to each site.
- For `depth > 0`, the circuit is built layer-by-layer:
  - On odd layers, random two-qubit gates are applied on sites 1-2, 3-4, etc.
  - On even layers, random two-qubit gates are applied on sites 2-3, 4-5, etc.

# Arguments
- `ξ::Vector{Index{Int64}}`: A vector of site indices for the qubits.
- `depth::Int64`: The depth of the circuit (non-negative integer).

# Returns
A vector of ITensors representing the random circuit gates.

# Example
```julia
circuit = random_circuit(siteinds("Qubit", 10), 3)
```
"""
function random_circuit(ξ::Vector{Index{Int64}}, depth::Int64)
    N = length(ξ)
    @assert depth >= 0 "Circuit depth must be non-negative."
    circuit = ITensor[]

    if depth == 0
        # Apply a single-qubit random unitary on each site.
        append!(circuit, [op("RandomUnitary", ξ[j]) for j in 1:N])
    else
        for layer in 1:depth
            if isodd(layer)
                # Odd layers: apply gates on sites 1-2, 3-4, etc.
                random_layer = [op("RandomUnitary", ξ[j], ξ[j+1]) for j in 1:2:(N-1)]
            else
                # Even layers: apply gates on sites 2-3, 4-5, etc.
                random_layer = [op("RandomUnitary", ξ[j], ξ[j+1]) for j in 2:2:(N-1)]
            end
            append!(circuit, random_layer)
        end
    end
    return circuit
end


"""
    random_Pauli_layer(ξ::Vector{Index{Int64}}, p::Vector{Float64})

Construct a layer of random single-qubit Pauli operations to simulate local depolarization. Upon avereraging, this corresponds to the local depolarization channel with strength p.

For each qubit (with index i), a random Pauli operation is applied with the following probabilities:
- With probability 1 - 3p_i/4: No operation is applied (the qubit remains unchanged).
- With probability p_i/4} each: Apply the X, Y, or Z gate.

Here, p_i is the depolarization probability for qubit i.

# Arguments
- `ξ::Vector{Index{Int64}}`: A vector of ITensor indices representing the qubit sites.
- `p::Vector{Float64}`: A vector of depolarization probabilities (one per qubit).

# Returns
A vector of ITensors representing the applied Pauli gates. If no gate is applied on a site (with probability 1 - 3p_i/4, that site is omitted from the returned circuit.

# Example
```julia
circuit = random_Pauli_layer(siteinds("Qubit", 5), 0.05 * ones(5))
```
"""
function random_Pauli_layer(ξ::Vector{Index{Int64}}, p::Vector{Float64})
    N = length(ξ)
    circuit = ITensor[]
    for i in 1:N
        if rand() > 1 - 3 * p[i] / 4
            a = rand()
            if a < 1/3
                push!(circuit, op("X", ξ[i]))
            elseif a < 2/3
                push!(circuit, op("Y", ξ[i]))
            else
                push!(circuit, op("Z", ξ[i]))
            end
        end
    end
    return circuit
end

"""
    random_magnetic_field_layer(ξ::Vector{Index{Int64}}, p::Vector{Float64})

Construct a layer of random Rz gates representing a random magnetic field along the z-axis.

For each qubit i, a random rotation Rz is applied with a rotation angle drawn uniformly from [0, 2 pi p_i).
This gives an average rotation angle of pi p_i on each site.

# Arguments
- `ξ::Vector{Index{Int64}}`: A vector of ITensor indices corresponding to the qubit sites.
- `p::Vector{Float64}`: A vector of parameters (one per qubit) that set the scale of the rotation angles.

# Returns
A vector of ITensors representing the random Rz gates applied to each qubit.

# Example
```julia
circuit = random_magnetic_field_layer(siteinds("Qubit", 5), 0.1 * ones(5))
```
"""
function random_magnetic_field_layer(ξ::Vector{Index{Int64}}, p::Vector{Float64})
    N = length(ξ)
    circuit = ITensor[]
    for i in 1:N
        push!(circuit, op("Rz", ξ[i]; θ=2π * p[i] * rand()))
    end
    return circuit
end
