# Copyright (c) 2024 Benoît Vermersch and Andreas Elben
# SPDX-License-Identifier: Apache-2.0
# http://www.apache.org/licenses/LICENSE-2.0

"""
    flatten(O::Union{MPS, MPO, Vector{ITensor}})

Flatten a Matrix Product State (MPS), Matrix Product Operator (MPO), or a vector of ITensors into a single ITensor by sequentially multiplying the constituent tensors.

# Arguments
- `O`: An MPS, MPO, or vector of ITensors to be flattened.

# Returns
An ITensor representing the product of the individual tensors in `O`.

# Example
```julia
A = random_mps(siteinds("Qubit", 5))
flatA = flatten(A)
```
"""
function flatten(O::Union{MPS,MPO,Vector{ITensor}})
    return reduce(*, O)
end

"""
    get_siteinds(ψ::Union{MPS, MPO})

Retrieve the site indices for a quantum state represented as an MPS or MPO.

# Arguments
- `ψ`: The quantum state, which can be either a Matrix Product State (MPS) or a Matrix Product Operator (MPO).

# Returns
A vector of site indices corresponding to the quantum state `ψ`.

# Example
```julia
ξ = get_siteinds(ψ)
```
"""
function get_siteinds(ψ::Union{MPS, MPO})
    if isa(ψ, MPS)
        return siteinds(ψ)
    else
        return firstsiteinds(ψ; plev=0)
    end
end

"""
    get_trace(ρ::MPO)

Compute the trace of a Matrix Product Operator (MPO) ρ by contracting each tensor with a delta function that equates its unprimed and primed indices.

# Arguments
- `ρ::MPO`: A matrix product operator representing a quantum state or operator.

# Returns
A scalar representing the trace of ρ.

# Example
```julia
t = get_trace(ρ)
```
"""
function get_trace(ρ::MPO)
    s = firstsiteinds(ρ; plev=0)
    NA = size(s, 1)
    X = ρ[1] * δ(s[1], s[1]')
    for i in 2:NA
        X *= ρ[i] * δ(s[i], s[i]')
    end
    return X[]
end

# """
#     reduce_to_subsystem(ρ::MPO,subsystem::Vector{Int64})

# compute the reduce density matrix over sites mentionned in part
# """

# function reduce_to_subsystem(ρ::MPO,subsystem::Vector{Int64})
# 	N = length(ρ)
# 	NA = size(subsystem,1)
#     s = firstsiteinds(ρ;plev=0)
# 	sA = s[subsystem]
# 	ρA = MPO(sA)
# 	L = 1
# 	for i in 1:subsystem[1]-1
# 		L *= ρ[i]*δ(s[i],s[i]')
# 	end

# 	for j in 1:NA
# 		if j<NA
# 			imax = subsystem[j+1]-1
# 		else
# 			imax = N
# 		end
# 		R = 1
# 		for i in subsystem[j]+1:imax
#        		 R *= ρ[i]*δ(s[i],s[i]')
# 		end
# 		if j==1
# 			ρA[1] = L*ρ[subsystem[1]]*R
# 		else
# 			ρA[j] =  ρ[subsystem[j]]*R
# 		end
# 	end
# 	return ρA
# end

"""
    reduce_to_subsystem(ρ::MPO, subsystem::Vector{Int64})

Compute the reduced density matrix (as an MPO) for a specified subsystem.

# Arguments
- `ρ::MPO`: A Matrix Product Operator representing the full density matrix.
- `subsystem::Vector{Int64}`: A vector of site indices (1-based) specifying the subsystem to retain.

# Returns
An MPO representing the reduced density matrix over the sites specified in `subsystem`.

# Example
```julia
ρ_reduced = reduce_to_subsystem(ρ, [2, 3])
```
"""
function reduce_to_subsystem(ρ::MPO, subsystem::Vector{Int64}, renormalize = false)
    n_sites = length(ρ)                        # Total number of sites in ρ
    n_subsys = length(subsystem)               # Number of sites in the subsystem
    s_full = firstsiteinds(ρ; plev=0)            # Full vector of site indices for ρ
    # Construct the site indices for the reduced MPO:
    s_subsys = s_full[subsystem]
    ρ_reduced = MPO(s_subsys)                    # Create an MPO with the reduced site indices

    # Compute the left environment (L) for sites before the first subsystem site.
    L = 1
    for i in 1:(subsystem[1] - 1)
        L *= ρ[i] * δ(s_full[i], s_full[i]')
    end

    # Loop over each subsystem site.
    for j in 1:n_subsys
        # Determine the right boundary for the current subsystem block.
        if j < n_subsys
            imax = subsystem[j + 1] - 1
        else
            imax = n_sites
        end

        # Compute the right environment (R) for sites following the current subsystem site.
        R = 1
        for i in (subsystem[j] + 1):imax
            R *= ρ[i] * δ(s_full[i], s_full[i]')
        end

        # Replace the MPO tensor at the subsystem site with the contracted result.
        if j == 1
            ρ_reduced[1] = L * ρ[subsystem[1]] * R
        else
            ρ_reduced[j] = ρ[subsystem[j]] * R
        end
    end

    return ρ_reduced
end

"""
    reduce_to_subsystem(ψ::MPS, subsystem::Vector{Int64})

Compute the reduced density matrix for a pure state represented by the MPS `ψ` over the specified subsystem.

This function first constructs the density matrix by taking the outer product of `ψ` with itself, and then
applies the MPO version of `reduce_to_subsystem` to obtain the reduced density matrix for the sites specified
in `subsystem`.

# Arguments
- `ψ::MPS`: A Matrix Product State representing a pure quantum state.
- `subsystem::Vector{Int64}`: A vector of site indices (1-based) specifying the subsystem to retain.

# Returns
An MPO representing the reduced density matrix over the specified subsystem.

# Example
```julia
ρ_sub = reduce_to_subsystem(ψ, [2, 3])
```
"""
function reduce_to_subsystem(ψ::MPS, subsystem::Vector{Int64})
    return reduce_to_subsystem(outer(ψ', ψ), subsystem)
end

"""
    partial_transpose(ρ::MPO, subsystem::Vector{Int})

Compute the partial transpose of an MPO over the sites specified by `subsystem`.

For each site index in the MPO:
- If the index is in the `subsystem`, the tensor is transposed by swapping its unprimed and primed indices using `swapind`.
- Otherwise, the tensor is left unchanged (multiplied by 1.0 for type consistency).

# Arguments
- `ρ::MPO`: A Matrix Product Operator representing a density matrix.
- `subsystem::Vector{Int}`: A vector of site indices (1-based) over which to apply the transpose.

# Returns
An MPO in which the tensors corresponding to the sites in `subsystem` have been partially transposed.

# Example
```julia
ρT = partial_transpose(ρ, [2, 3])
```
"""
function partial_transpose(ρ::MPO, subsystem::Vector{Int})
    ξ = firstsiteinds(ρ; plev=0)
    ρT = MPO(ξ)
    for i in 1:length(ξ)
        if i in subsystem
            ρT[i] = swapind(ρ[i], ξ[i], ξ[i]')
        else
            ρT[i] = 1.0 * ρ[i]
        end
    end
    return ρT
end


"""
    get_entanglement_spectrum(ψ::MPS, NA::Int64)

Compute the entanglement spectrum for the bipartition defined by the first `NA` sites of the MPS `ψ`.

This function first creates a copy of `ψ` and orthogonalizes it up to site `NA`. Then, it performs an SVD on the tensor at site `NA`:
- If `NA > 1`, the SVD is taken with respect to the indices corresponding to the link between sites `NA-1` and `NA` and the physical index at site `NA`.
- If `NA == 1`, only the physical index at site `NA` is used.

The returned object `spec` contains the singular values (which are related to the Schmidt coefficients) for the bipartition.

# Arguments
- `ψ::MPS`: A matrix product state representing a pure quantum state.
- `NA::Int64`: The number of sites from the left that define the subsystem for which the entanglement spectrum is computed.

# Returns
- `spec`: An ITensor containing the singular values from the SVD at site `NA`.

# Example
```julia
spectrum = get_entanglement_spectrum(ψ, 3)
```
"""
function get_entanglement_spectrum(ψ::MPS, NA::Int64)
    statel = copy(ψ)
    orthogonalize!(statel, NA)
    if NA > 1
        U, spec, V = svd(statel[NA], (linkind(statel, NA-1), siteind(statel, NA)))
    else
        U, spec, V = svd(statel[NA], siteind(statel, NA))
    end
    return spec
end


"""
    get_trace_moment(spec::ITensor, k::Int)

Compute the kth moment of the entanglement spectrum represented by the ITensor `spec`.

The function assumes that `spec` is a square ITensor whose diagonal elements correspond
to the singular values (Schmidt coefficients) of a reduced density matrix. The kth moment is
computed as:

    pk = ∑ₗ (spec[l, l]^(2*k))

which effectively computes the sum over the kth powers of the squared singular values.

# Arguments
- `spec::ITensor`: A square ITensor representing the entanglement spectrum.
- `k::Int`: The moment order to compute (must be an integer ≥ 1).

# Returns
A scalar (Float64) corresponding to the kth moment.

# Example
```julia
moment = get_trace_moment(spec, 2)
```
"""
function get_trace_moment(spec::ITensor, k::Int)
    @assert k >= 1 "Only integer valued moments with k ≥ 1 can be computed."
    return sum(spec[l, l]^(2*k) for l in 1:dim(spec, 1))
end




"""
    get_trace_moment(ψ::Union{MPS, MPO}, k::Int, subsystem::Vector{Int}=collect(1:length(ψ)))

Compute the kth trace moment of the reduced density matrix for a given subsystem of a quantum state.

For a pure state (MPS) and when the subsystem is contiguous starting from site 1, the function
computes the entanglement spectrum of the bipartition defined by the last site in `subsystem`
and returns the kth moment of the squared Schmidt coefficients. Otherwise, the function reduces
the state to the specified subsystem.

For k = 2 (purity), it returns the squared norm (which is equivalent to tr(ρ²)). For k > 2,
it computes ρ^k via repeated application of the `apply` function (with a cutoff) and returns the trace of the resulting tensor.

# Arguments
- `ψ::Union{MPS, MPO}`: The quantum state, represented as an MPS (for pure states) or MPO (for mixed states).
- `k::Int`: The moment order to compute (must be an integer ≥ 1).
- `subsystem::Vector{Int}` (optional): A vector of site indices (1-based) specifying the subsystem to retain. Defaults to all sites.

# Returns
A scalar (Float64) representing the kth trace moment of the reduced density matrix.

# Example
```julia
moment = get_trace_moment(ψ, 3, [1, 2, 3]; partial_transpose=false)
```
"""
function get_trace_moment(ψ::Union{MPS, MPO}, k::Int, subsystem::Vector{Int}=collect(1:length(ψ)))
    @assert k >= 1 "k must be at least 1."
    # For a contiguous subsystem starting at 1 and for pure states (MPS), use the entanglement spectrum.
    if diff(subsystem) == ones(Int, length(subsystem)-1) && subsystem[1] == 1 && isa(ψ, MPS)
        spec = get_entanglement_spectrum(ψ, subsystem[end])
        return get_trace_moment(spec, k)
    else
        ρ = reduce_to_subsystem(ψ, subsystem)
        if k == 2
            return norm(ρ)^2
        else
            ρk = 1.0 * ρ
            for _ in 2:k
                ρk = apply(ρk, ρ; cutoff=1e-12)
            end
            return get_trace(ρk)
        end
    end
end


"""
    get_trace_moments(ψ::Union{MPS, MPO}, k_vector::Vector{Int}, subsystem::Vector{Int}=collect(1:length(ψ)))

Compute a vector of trace moments for a quantum state `ψ` over a specified subsystem.

For each moment order `k` in `k_vector`, the function computes the kth trace moment of the reduced density matrix obtained by applying `reduce_to_subsystem(ψ, subsystem)`.

# Arguments
- `ψ::Union{MPS, MPO}`: The quantum state, represented as an MPS (for pure states) or an MPO (for mixed states).
- `k_vector::Vector{Int}`: A vector of integer moment orders (each ≥ 1) for which the trace moments are computed.
- `subsystem::Vector{Int}` (optional): A vector of site indices (1-based) specifying the subsystem to consider. Defaults to all sites.

# Returns
A vector of scalars, each being the kth trace moment corresponding to the entries of `k_vector`.

# Example
```julia
moments = get_trace_moments(ψ, [1, 2, 3], [1, 2, 3])
```
"""
function get_trace_moments(ψ::Union{MPS, MPO}, k_vector::Vector{Int}, subsystem::Vector{Int}=collect(1:length(ψ)))
    return [get_trace_moment(ψ, k, subsystem) for k in k_vector]
end

"""
    get_Born_MPS(ρ::MPO)

Construct the Born probability vector as an MPS from an MPO representation of a density matrix ρ.

This function computes the Born probability vector P(s) = ⟨s|ρ|s⟩, where |s⟩ is a basis state. It does so by contracting each tensor of the MPO ρ with appropriate delta tensors that enforce equality between the unprimed and primed indices. The result is returned as an MPS that represents the Born probabilities over the computational basis.

# Arguments
- `ρ::MPO`: A Matrix Product Operator representing the density matrix.

# Returns
An MPS representing the Born probability vector.

# Example
```julia
P = get_Born_MPS(ρ)
```
"""
function get_Born_MPS(ρ::MPO)
    ξ = firstsiteinds(ρ; plev=0)
    N = size(ξ, 1)
    P = MPS(ξ)
    for i in 1:N
        Ct = δ(ξ[i], ξ[i]', ξ[i]'')
        P[i] = ρ[i] * Ct
        P[i] *= δ(ξ[i], ξ[i]'')
    end
    return P
end

"""
    get_Born_MPS(ψ::MPS)

Construct the Born probability vector P(s) = |ψ(s)|² as an MPS from an MPS representation ψ.

This function computes the probability for each computational basis state by contracting each tensor of the MPS ψ with its complex conjugate, using appropriate delta tensors to enforce index equality. The resulting MPS represents the Born probability distribution of the state.

# Arguments
- `ψ::MPS`: A matrix product state representing a pure quantum state.

# Returns
An MPS representing the Born probability vector, where each tensor P[i] corresponds to the probability contribution at site i.

# Example
```julia
P = get_Born_MPS(ψ)
```
"""
function get_Born_MPS(ψ::MPS)
    ξ = siteinds(ψ)
    N = length(ξ)
    P = MPS(ξ)
    for i in 1:N
        Ct = δ(ξ[i], ξ[i]', ξ[i]'')
        P[i] = ψ[i] * conj(ψ[i]') * Ct
        P[i] *= δ(ξ[i], ξ[i]'')
    end
    return P
end


"""
    get_selfXEB(ψ::MPS)

Compute the self-XEB (cross-entropy benchmarking) metric for a pure state represented as an MPS.

The self-XEB is defined as:

    selfXEB = 2^N * ∑ₛ |ψ(s)|⁴ - 1

where the sum is over all computational basis states s and N is the number of sites (qubits). This function first computes the Born probability MPS from ψ, then calculates the inner product of the probability MPS with itself, scales the result by 2^N, and finally subtracts 1.

# Arguments
- `ψ::MPS`: A Matrix Product State representing a pure quantum state.

# Returns
A scalar (Float64) representing the self-XEB value.

# Example
```julia
x = get_selfXEB(ψ)
```
"""
function get_selfXEB(ψ::MPS)
    P0 = get_Born_MPS(ψ)
    N = length(ψ)
    return 2^N * real(inner(P0, P0)) - 1
end

"""
    get_average_mps(ψ_list::Vector{MPS}, χ::Int64, nsweeps::Int64)

Approximate the average state σ from a collection of MPS using a DMRG-like algorithm.

The algorithm finds an MPS ψ (with maximum bond dimension χ) that approximates the average state
σ = Average(ψ_list). To monitor convergence, it tracks a cost function defined as:

    cost_function = ⟨ψ|ψ⟩ - ⟨ψ|σ⟩ - ⟨σ|ψ⟩,

which is equivalent to (||σ - ψ||² - ⟨σ|σ⟩).

# Arguments
- `ψ_list::Vector{MPS}`: A vector of MPS objects representing individual quantum states.
- `χ::Int64`: The desired maximum bond dimension for the averaged MPS.
- `nsweeps::Int64`: The number of sweeps (iterations) to perform in the DMRG-like algorithm.

# Returns
An MPS representing the approximate average state with bond dimension χ.

# Example
```julia
avg_state = get_average_mps(ψ_list, 20, 10)
```
"""
function get_average_mps(ψ_list::Vector{MPS},χ::Int64,nsweeps::Int64)
    NU = length(ψ_list)
    N = length(ψ_list[1])

    #ψ = truncate(ψ_list[1];maxdim=χ)
    #orthogonalize!(ψ,1)
    ψ = random_mps(siteinds(ψ_list[1]); linkdims=χ);

    L = Array{ITensor}(undef,NU,N)
    R = Array{ITensor}(undef,NU,N)
    Ma = Array{ITensor}(undef,NU,N)
    for r in 1:NU
        Ma[r,:] = ψ_list[r].data
    end
    #init the right environments
    for r in 1:NU
        X = 1.
        for j in N:-1:2
            X *= Ma[r,j]*dag(ψ[j])
            R[r,j] = X
        end
    end
    #first overlap
    @showprogress dt=1  for sw in 1:nsweeps
        cost_function = real(inner(ψ,ψ))
        for ψ_r in ψ_list
            cost_function -= real(inner(ψ_r,ψ))/NU
            cost_function -= real(inner(ψ,ψ_r))/NU
        end
        println("Cost function ",cost_function)
        #left part of the sweep
        for i in 1:N
            if i==1
                ψ[1] = sum(Ma[:,1].*R[:,2])/NU
            elseif i<N
                ψ[i] = sum(R[:,i+1].*Ma[:,i].*L[:,i-1])/NU
            else
                ψ[N] = sum(L[:,i-1].*Ma[:,i])/NU
            end
            if i<N
                orthogonalize!(ψ,i+1)
            end
            #updating the left environments
            if i==1
                L[:,1] = [Ma[r,1]*dag(ψ[1]) for r in 1:NU]
            elseif i<=N
                L[:,i] = [L[r,i-1]*Ma[r,i]*dag(ψ[i]) for r in 1:NU]
            end
           end
        #right part of the sweep
        @showprogress dt=1 for i in N:-1:1
            if i==1
                ψ[1] = sum(Ma[:,1].*R[:,2])/NU
            elseif i<N
                ψ[i] = sum(R[:,i+1].*Ma[:,i].*L[:,i-1])/NU
            else
                ψ[N] = sum(L[:,i-1].*Ma[:,i])/NU
            end
            if i>1
                orthogonalize!(ψ,i-1)
                end
            #updating the right environments
            if i==N
                R[:,N] = [Ma[r,N]*dag(ψ[N]) for r in 1:NU]
            elseif i>1
                R[:,i] = [R[r,i+1]*Ma[r,i]*dag(ψ[i]) for r in 1:NU]
            end
        end
    end
    cost_function = real(inner(ψ,ψ))
    @showprogress dt=1 for ψ_r in ψ_list
        cost_function -= real(inner(ψ,ψ_r))/NU
        cost_function -= real(inner(ψ_r,ψ))/NU
    end
    println("Cost function ",cost_function)
    return ψ
end
