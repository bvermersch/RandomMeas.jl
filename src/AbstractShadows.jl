# Copyright (c) 2024 Benoît Vermersch and Andreas Elben
# SPDX-License-Identifier: Apache-2.0
# http://www.apache.org/licenses/LICENSE-2.0

# ---------------------------------------------------------------------------
# Abstract Shadow Methods
# ---------------------------------------------------------------------------
"""
    get_expect_shadow(O::MPO, shadows::AbstractArray{<:AbstractShadow}; compute_sem::Bool = false)

Compute the average expectation value of an MPO operator `O` using an array of shadow objects.

This function estimates the expectation value ⟨O⟩ = Tr[O·ρ] of a matrix product operator (MPO) `O`
with respect to the quantum state ρ represented by classical shadows. Classical shadows provide
an efficient way to estimate expectation values of observables from randomized measurements,
enabling scalable quantum state characterization.

# Arguments
- `O::MPO`: The matrix product operator whose expectation value is to be computed. MPOs are
  efficient representations of many-body observables in quantum systems.
- `shadows::AbstractArray{<:AbstractShadow}`: An array of shadow objects (of any shape) over which
  the expectation values are computed. Each shadow represents a classical snapshot of the quantum state.
- `compute_sem::Bool` (optional): If `true`, also compute the standard error of the mean (SEM)
  for statistical error analysis. Default is `false`.

# Returns
- If `compute_sem` is `false`, returns the average expectation value.
- If `compute_sem` is `true`, returns a tuple `(mean, sem)`, where `mean` is the average expectation value and `sem` is the standard error.

# Example
```julia
mean_val = get_expect_shadow(O, shadows)
mean_val, sem_val = get_expect_shadow(O, shadows; compute_sem=true)
```
"""
function get_expect_shadow(
    O::MPO,
    shadows::AbstractArray{<:AbstractShadow};
    compute_sem::Bool = false
)
    # Ensure the array of shadows is not empty
    @assert !isempty(shadows) "Array of shadows is empty."

    # Compute all expectation values
    expect_values = [get_expect_shadow(O, shadow) for shadow in shadows]

    # Compute mean
    mean_value = mean(expect_values)

    if compute_sem
        # Compute standard error of the mean (SEM)
        sem_value = std(expect_values) / sqrt(length(expect_values))
        return mean_value, sem_value
    else
        return mean_value
    end
end


"""
    get_expect_shadow(O::MPO, shadow::AbstractShadow)

Compute the expectation value of an MPO operator `O` using a single shadow object.

# Arguments:
- `O::MPO`: The MPO operator for which the expectation value is computed.
- `shadow::AbstractShadow`: A shadow object, either dense, factorized, or shallow.

# Returns
The expectation value as a scalar.

# Example
```julia
val = get_expect_shadow(O, shadow)
```
"""
function get_expect_shadow(O::MPO, shadow::AbstractShadow)
    if shadow isa DenseShadow
        return get_expect_shadow(O, shadow::DenseShadow)
    elseif shadow isa FactorizedShadow
        return get_expect_shadow(O, shadow::FactorizedShadow)
    elseif shadow isa ShallowShadow
        return get_expect_shadow(O, shadow::ShallowShadow)
    else
        throw(ArgumentError("Unsupported shadow type: $(typeof(shadow))"))
    end
end


"""
    get_trace_moment(shadows::Array{<:AbstractShadow, 2}, kth_moment::Int; O::Union{Nothing, MPO}=nothing, compute_sem::Bool=false, compute_renyi::Bool=false)

Compute a single trace moment from an array of `AbstractShadow` objects.

This function estimates the k-th trace moment T_k = Tr[ρ^k] (or Tr[O·ρ^k] if an observable O is provided)
from classical shadow data. Trace moments are fundamental quantities in quantum state characterization
and are used to compute various entanglement measures and state properties.

# Arguments
- `shadows::Array{<:AbstractShadow, 2}`: An array of shadow objects with dimensions `(n_ru, n_m)`,
  where `n_ru` is the number of random unitaries and `n_m` is the number of measurements.
- `kth_moment::Int`: The moment `k` to compute (e.g., `k = 1, 2, ...`).
- `O::Union{Nothing, MPO}` (optional): If provided, computes `Tr[O * ρ^k]`; otherwise, computes `p_k=Tr[ρ^k]` (default: `nothing`).
- `compute_sem::Bool` (optional): If `true`, also computes the standard error of the mean (SEM) and bias correction
  using jackknife resampling (default: `false`).
- `compute_renyi::Bool` (optional): If `true`, returns the Rényi-k entropy S_k = (1/(1-k)) * log₂(p_k)
  instead of the raw trace moment (default: `false`).

# Returns
- If `compute_sem` is `false`: The computed trace moment (or Rényi entropy) as a scalar.
- If `compute_sem` is `true`: A tuple `(estimate, bias, sem)` where:
  - `estimate`: The point estimate of the trace moment
  - `bias`: The bias correction from jackknife resampling
  - `sem`: The standard error of the mean

# Example
```julia
moment1 = get_trace_moment(shadows, 1)
moment2 = get_trace_moment(shadows, 2; O=my_operator)
estimate, bias, sem = get_trace_moment(shadows, 2; compute_sem=true)
renyi_entropy = get_trace_moment(shadows, 2; compute_renyi=true)
```
"""
function get_trace_moment(shadows::Array{<:AbstractShadow, 2}, kth_moment::Int; O::Union{Nothing, MPO}=nothing,compute_sem::Bool = false,compute_renyi::Bool = false)

    if compute_sem
        s, bias, cov = get_trace_moments(shadows, [kth_moment]; O=O, compute_cov = compute_sem, compute_renyi = compute_renyi)
        return s[1], bias[1], sqrt(cov[1,1])
    else
        s = get_trace_moments(shadows, [kth_moment]; O=O, compute_cov = compute_sem, compute_renyi = compute_renyi)
        return s[1]
    end

end

"""
    get_trace_moments(
        shadows::Array{<:AbstractShadow,2},
        k_vec::Vector{Int};
        O::Union{Nothing,MPO}=nothing,
        compute_cov::Bool = false,
        compute_renyi::Bool = false,
    )

Estimate several trace moments from classical shadow data.

This function computes multiple trace moments of the form:

\\[
p_k = \\operatorname{tr}\\bigl[\\,\\rho^{\\,k}\\bigr]
\\]

(or, if `O` is supplied, the generalized moments
\\( \\operatorname{tr}[\\,O\\,\\rho^{\\,k}] \\))
from an array of classical–shadow objects.

Trace moments are fundamental quantities in quantum state characterization that capture
higher-order correlations and entanglement properties. They are used to compute various
entanglement measures, state fidelities, and other quantum information quantities.

If `compute_renyi=true` for each \\(k\\) is converted on the fly to
the (binary-log) Rényi-\\(k\\) entropy \\(S_k\\) is estimated:

\\[
S_k \\,=\\, \\frac{1}{1-k}\\,\\log_{2} p_k .
\\]

Optionally `compute_cov=true` returns, in addition to the vector of point
estimates, the full jack-knife covariance matrix
\\( \\operatorname{Cov}(T_{k_a},T_{k_b}) \\) (or of the Rényi entropies,
depending on `compute_renyi`).

# Arguments
- `shadows` : 2-D array of size `(n_ru, n_m)` holding the classical shadows.
  - `n_ru`: number of random unitaries
  - `n_m`: number of measurements per unitary
- `k_vec`   : vector of positive integers specifying which moments to compute.
- `O`       : optional MPO observable; if given, moments of \\( O\\,\\rho^{k} \\) are computed.
- `compute_cov` : whether to return the jack-knife covariance matrix for error analysis.
- `compute_renyi`    : return Rényi entropies instead of raw trace moments.

# Returns
- If `compute_cov=false`: `θ̂::Vector{Float64}` - point estimates for each `k_vec[i]`
- If `compute_cov=true`: A tuple `(θ̂, bias, Σ̂)` where:
  - `θ̂::Vector{Float64}`: point estimates for each `k_vec[i]`
  - `bias::Vector{Float64}`: bias corrections from jackknife resampling
  - `Σ̂::Matrix{Float64}`: jack-knife covariance matrix

# Example
```julia
# Compute first three trace moments
moments = get_trace_moments(shadows, [1, 2, 3])

# Compute with covariance matrix for error analysis
moments, bias, cov = get_trace_moments(shadows, [1, 2, 3]; compute_cov=true)

# Compute Rényi entropies
renyi_entropies = get_trace_moments(shadows, [1, 2, 3]; compute_renyi=true)
```
"""
function get_trace_moments(
    shadows::Array{<:AbstractShadow,2},
    k_vec::Vector{Int};
    O::Union{Nothing,MPO} = nothing,
    compute_cov::Bool     = false,
    compute_renyi::Bool        = false,
)
    n_ru, n_m = size(shadows)
    k_vec_sorted = sort(unique(k_vec)) # work on distinct, ascending k
    nK = length(k_vec_sorted)

    # containers
    θ_est   = zeros(Float64, nK)
    jackmat = compute_cov ? zeros(Float64, n_ru, nK) : nothing

    # --- helper: single-k estimator with optional jackknife ----------------
    function single_k(k::Int)
        # pre-enumerate permutations and m–cartesian product
        perms   = collect(permutations(1:n_ru, k))
        cprod   = collect(CartesianIndices(ntuple(_ -> 1:n_m, k)))
        n_perm  = length(perms)


        # average over measurements for each permutation
        perm_avg = zeros(Float64, n_perm)
        for (pidx, r) in enumerate(perms)
            svals = Float64[]
            for m in cprod
                push!(svals,
                      real(get_trace_product(
                          (shadows[r[i], m[i]] for i in 1:k)...; O)))
            end
            perm_avg[pidx] = mean(svals)
        end

        # define the averaging functional
        avgfun(x) = compute_renyi ?
            (1/(1-k))*log2(mean(x)) :
            mean(x)

        θ  = avgfun(perm_avg)

        if !compute_cov
            return θ, nothing
        end

        # jackknife groups: permutations not containing unitary i
        groups = Vector{Vector{Int}}(undef, n_ru)
        for i in 1:n_ru
            groups[i] = [idx for (idx,r) in enumerate(perms) if i ∉ r]
        end

        jackvals = similar(jackmat, n_ru)
        for i in 1:n_ru
            jackvals[i] = avgfun(perm_avg[groups[i]])
        end
        return θ, jackvals
    end
    # -----------------------------------------------------------------------

    # loop over desired moments
    for (idx, k) in enumerate(k_vec_sorted)
        θ_est[idx], jv = single_k(k)
        if compute_cov
            jackmat[:,idx] = jv
        end
    end

    # build covariance if requested
    if compute_cov
        Σ = zeros(Float64, nK, nK)
        for a in 1:nK, b in a:nK           # symmetric
            cov = (n_ru-1)^2/n_ru *
                  dot(jackmat[:,a] .- mean(jackmat[:,a]),
                      jackmat[:,b] .- mean(jackmat[:,b])) / (n_ru-1)
            Σ[a,b] = Σ[b,a] = cov
        end

        θ_jack = n_ru * θ_est .- (n_ru - 1) * vec(mean(jackmat; dims = 1))

        return θ_est, θ_est - θ_jack , Σ
    else
        return θ_est
    end
end


"""
    get_trace_moment(shadows::Vector{<:AbstractShadow}, kth_moment::Int; O::Union{Nothing, MPO}=nothing, compute_sem::Bool=false, compute_renyi::Bool=false)

Wrapper function. Compute a single trace moment for a vector of shadow objects by reshaping the vector into a 2D array.

This is a convenience function that reshapes a vector of shadow objects into a 2D array and then
calls the main `get_trace_moment` function. It's useful when you have a flat collection of shadows
but need to compute trace moments using the full statistical machinery.

# Arguments
- `shadows::Vector{<:AbstractShadow}`: A vector of shadow objects.
- `kth_moment::Int`: The moment order `k` to compute (e.g., `k = 1, 2, ...`).
- `O::Union{Nothing, MPO}` (optional): An MPO observable. If provided, computes `Tr[O * ρ^k]`;
  otherwise, computes `Tr[ρ^k]` (default: `nothing`).
- `compute_sem::Bool` (optional): If `true`, also computes the standard error of the mean (SEM) and bias correction
  using jackknife resampling (default: `false`).
- `compute_renyi::Bool` (optional): If `true`, returns the Rényi-k entropy S_k = (1/(1-k)) * log₂(T_k)
  instead of the raw trace moment (default: `false`).

# Returns
- If `compute_sem` is `false`: The computed trace moment (or Rényi entropy) as a scalar.
- If `compute_sem` is `true`: A tuple `(estimate, bias, sem)` where:
  - `estimate`: The point estimate of the trace moment
  - `bias`: The bias correction from jackknife resampling
  - `sem`: The standard error of the mean

# Example
```julia
moment = get_trace_moment(shadows_vector, 2; O=my_operator)
estimate, bias, sem = get_trace_moment(shadows_vector, 2; compute_sem=true)
renyi_entropy = get_trace_moment(shadows_vector, 2; compute_renyi=true)
```
"""
function get_trace_moment(shadows::Vector{<:AbstractShadow}, kth_moment::Int; O::Union{Nothing, MPO}=nothing, compute_sem::Bool = false,compute_renyi::Bool = false)
     return get_trace_moment(reshape(shadows, :, 1), kth_moment; O=O, compute_sem = compute_sem, compute_renyi = compute_renyi)
end

"""
    get_trace_moments(shadows::Vector{<:AbstractShadow}, kth_moments::Vector{Int}; O::Union{Nothing, MPO}=nothing, compute_cov::Bool=false, compute_renyi::Bool=false)

Wrapper function. Compute multiple trace moments from a vector of shadow objects by reshaping the vector into a 2D array.

This is a convenience function that reshapes a vector of shadow objects into a 2D array and then
calls the main `get_trace_moments` function. It's useful when you have a flat collection of shadows
but need to compute multiple trace moments using the full statistical machinery.

# Arguments
- `shadows::Vector{<:AbstractShadow}`: A vector of shadow objects.
- `kth_moments::Vector{Int}`: A vector of moment orders to compute (e.g., `[1, 2, 3]`).
- `O::Union{Nothing, MPO}` (optional): An MPO observable. If provided, computes `Tr[O * ρ^k]`;
  otherwise, computes `Tr[ρ^k]` (default: `nothing`).
- `compute_cov::Bool` (optional): Whether to return the jack-knife covariance matrix for error analysis (default: `false`).
- `compute_renyi::Bool` (optional): Return Rényi entropies instead of raw trace moments (default: `false`).

# Returns
- If `compute_cov=false`: `θ̂::Vector{Float64}` - point estimates for each moment in `kth_moments`
- If `compute_cov=true`: A tuple `(θ̂, bias, Σ̂)` where:
  - `θ̂::Vector{Float64}`: point estimates for each moment in `kth_moments`
  - `bias::Vector{Float64}`: bias corrections from jackknife resampling
  - `Σ̂::Matrix{Float64}`: jack-knife covariance matrix

# Example
```julia
moments = get_trace_moments(shadows_vector, [1, 2, 3])
moments, bias, cov = get_trace_moments(shadows_vector, [1, 2, 3]; compute_cov=true)
renyi_entropies = get_trace_moments(shadows_vector, [1, 2, 3]; compute_renyi=true)
```
"""
function get_trace_moments(shadows::Vector{<:AbstractShadow}, kth_moments::Vector{Int}; O::Union{Nothing, MPO}=nothing, compute_cov::Bool = false ,compute_renyi::Bool = false)
    return get_trace_moments(reshape(shadows, :, 1), kth_moments; O=O, compute_cov = compute_cov, compute_renyi = compute_renyi)
end


"""
    get_trace_product(shadows::AbstractShadow...; O::Union{Nothing, MPO}=nothing)

Compute the product of multiple shadow objects and return its trace or expectation value.

This function computes trace of products of classical shadows.

If `O` is `nothing`, returns the trace of the product:
    trace(shadow₁ * shadow₂ * ... * shadowₙ).
If `O` is provided, returns the expectation value computed by:
    get_expect_shadow(O, shadow₁ * shadow₂ * ... * shadowₙ).

# Arguments
- `shadows...`: A variable number of shadow objects. The product is computed in the order provided.
- `O::Union{Nothing, MPO}` (optional): An MPO observable. If provided, computes the expectation value
  of O with respect to the product of shadows.

# Returns
The trace of the product if `O` is `nothing`, or the expectation value if `O` is provided.

# Example
```julia
result = get_trace_product(shadow1, shadow2, shadow3)
result_with_O = get_trace_product(shadow1, shadow2, shadow3; O=my_operator)
```
"""
function get_trace_product(shadows::AbstractShadow...; O::Union{Nothing, MPO}=nothing)
    result = shadows[1]
    for shadow in shadows[2:end]
        result = multiply(result, shadow)
    end
    if O === nothing
        return trace(result)
    else
        return get_expect_shadow(O, result)
    end
end


"""
    multiply(shadow1::AbstractShadow, shadow2::AbstractShadow)

Multiply two shadow objects of the same concrete type.

# Arguments:
- `shadow1::AbstractShadow`: The first shadow object.
- `shadow2::AbstractShadow`: The second shadow object.

# Returns
A new shadow object representing the product.

# Throws
An `ArgumentError` if the types of `shadow1` and `shadow2` do not match.
# Example
```julia
prod_shadow = multiply(shadow1, shadow2)
```
"""
function multiply(shadow1::AbstractShadow, shadow2::AbstractShadow)
    if shadow1 isa DenseShadow && shadow2 isa DenseShadow
        return multiply(shadow1::DenseShadow, shadow2::DenseShadow)
    elseif shadow1 isa FactorizedShadow && shadow2 isa FactorizedShadow
        return multiply(shadow1::FactorizedShadow, shadow2::FactorizedShadow)
    else
        throw(ArgumentError("Cannot multiply shadows of types $(typeof(shadow1)) and $(typeof(shadow2))."))
    end
end

"""
    trace(shadow::AbstractShadow)

Compute the trace of a shadow object.

# Arguments:
- `shadow::AbstractShadow`: A shadow object.

# Returns
The trace of the shadow object as a scalar.
# Example
```julia
t = trace(shadow)
```
"""
function trace(shadow::AbstractShadow)
    if shadow isa DenseShadow
        return trace(shadow::DenseShadow)
    elseif shadow isa FactorizedShadow
        return trace(shadow::FactorizedShadow)
    else
        throw(ArgumentError("Unsupported shadow type: $(typeof(shadow))"))
    end
end

"""
    trace(shadows::AbstractArray{<:AbstractShadow})

Compute the trace for each shadow in a collection of shadow objects.

# Arguments:
- `shadows::AbstractArray{<:AbstractShadow}`: A collection (vector, matrix, etc.) of shadow objects.

# Returns
An array of scalar trace values corresponding to each shadow, with the same dimensions as the input.
"""
function trace(shadows::AbstractArray{<:AbstractShadow})
    return [trace(shadow) for shadow in shadows]
end



"""
    partial_trace(shadow::AbstractShadow, subsystem::Vector{Int}; assume_unit_trace::Bool=false)

Compute the partial trace of a shadow object over the complement of the specified subsystem.

The partial trace operation is fundamental in quantum information theory for studying entanglement
and reduced density matrices. It allows you to focus on a specific subsystem of a larger quantum
system by "tracing out" the degrees of freedom of the complementary subsystem.

# Arguments
- `shadow::AbstractShadow`: The shadow object representing the full quantum state.
- `subsystem::Vector{Int}`: A vector of site indices (1-based) specifying the subsystem to retain.
  The complement of this subsystem will be traced out.
- `assume_unit_trace::Bool` (optional): If `true`, assumes the shadow has unit trace (default: `false`).
    This can speed up the calculation for factorized shadows (as the trace of "traced out" qubits is not computed).

# Returns
A new shadow object reduced to the specified subsystem, representing the reduced density matrix
of the subsystem.

# Example
```julia
reduced_shadow = partial_trace(shadow, [1, 3])
```
"""
function partial_trace(shadow::AbstractShadow, subsystem::Vector{Int};assume_unit_trace::Bool=false)
    if shadow isa DenseShadow
        return partial_trace(shadow::DenseShadow, subsystem)
    elseif shadow isa FactorizedShadow
        return partial_trace(shadow::FactorizedShadow, subsystem;assume_unit_trace=assume_unit_trace)
    else
        throw(ArgumentError("Unsupported shadow type: $(typeof(shadow))"))
    end
end


"""
    partial_trace(shadows::AbstractArray{<:AbstractShadow}, subsystem::Vector{Int})

Compute the partial trace for each shadow in a collection of shadows.

# Arguments
- `shadows::AbstractArray{<:AbstractShadow}`: A collection of shadow objects (vector or 2D array).
- `subsystem::Vector{Int}`: A vector of site indices (1-based) specifying the subsystem to retain.

# Returns
An array of shadows reduced to the specified subsystem, with the same dimensions as the input array.
"""
function partial_trace(shadows::AbstractArray{<:AbstractShadow}, subsystem::Vector{Int};assume_unit_trace::Bool=false)
    # Allocate a new array with the same dimensions as the input
    reduced_shadows = similar(shadows)

    # Iterate over all elements in the array
    for idx in eachindex(shadows)
        reduced_shadows[idx] = partial_trace(shadows[idx], subsystem ; assume_unit_trace=assume_unit_trace)
    end

    return reduced_shadows
end

"""
    partial_transpose(shadow::AbstractShadow, subsystem::Vector{Int})

Compute the partial transpose of a shadow object over the specified subsystem(s).

The partial transpose is a crucial operation in quantum information theory, particularly for
entanglement detection. The Peres-Horodecki criterion states that if a bipartite quantum state
is separable, then its partial transpose must be positive semidefinite. This operation is
essential for computing entanglement measures like negativity and for studying quantum correlations.

This operation is analogous to QuTiP's partial transpose method.

# Arguments
- `shadow::AbstractShadow`: The shadow object for which the partial transpose is computed.
- `subsystem::Vector{Int}`: A vector of site indices (1-based) specifying the subsystem(s)
  over which to perform the transpose.

# Returns
A new shadow object that is the partial transpose of the input.

# Example
```julia
transposed_shadow = partial_transpose(shadow, [2, 4])
```
"""
function partial_transpose(shadow::AbstractShadow, subsystem::Vector{Int})
    if shadow isa DenseShadow
        return partial_transpose(shadow::DenseShadow, subsystem)
    elseif shadow isa FactorizedShadow
        return partial_transpose(shadow::FactorizedShadow, subsystem)
    else
        throw(ArgumentError("Unsupported shadow type: $(typeof(shadow))"))
    end
end

"""
    partial_transpose(shadows::AbstractArray{<:AbstractShadow}, subsystem::Vector{Int})

Compute the partial transpose for each shadow in a collection.

# Arguments
- `shadows::AbstractArray{<:AbstractShadow}`: A collection (vector or 2D array) of shadow objects.
- `subsystem::Vector{Int}`: A vector of site indices (1-based) specifying the subsystem(s)
  over which the transpose is to be performed.

# Returns
An array of shadow objects with the partial transpose applied, preserving the input dimensions.
"""
function partial_transpose(shadows::AbstractArray{<:AbstractShadow}, subsystem::Vector{Int})
    transposed_shadows = similar(shadows)
    for idx in eachindex(shadows)
        transposed_shadows[idx] = partial_transpose(shadows[idx], subsystem)
    end
    return transposed_shadows
end
