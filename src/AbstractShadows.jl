# Copyright (c) 2024 Benoît Vermersch and Andreas Elben
# SPDX-License-Identifier: Apache-2.0
# http://www.apache.org/licenses/LICENSE-2.0

# ---------------------------------------------------------------------------
# Abstract Shadow Methods
# ---------------------------------------------------------------------------
"""
    get_expect_shadow(O::MPO, shadows::AbstractArray{<:AbstractShadow}; compute_sem::Bool = false)

Compute the average expectation value of an MPO operator `O` using an array of shadow objects.

# Arguments
- `O::MPO`: The MPO operator whose expectation value is to be computed.
- `shadows::AbstractArray{<:AbstractShadow}`: An array of shadow objects (of any shape) over which the expectation values are computed.
- `compute_sem::Bool` (optional): If `true`, also compute the standard error of the mean (SEM). Default is `false`.

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
    get_trace_moment(shadows::Array{<:AbstractShadow, 2}, kth_moment::Int; O::Union{Nothing, MPO}=nothing)

Compute a single trace moment from an array of `AbstractShadow` objects.

# Arguments
- `shadows::Array{<:AbstractShadow, 2}`: An array of shadow objects with dimensions `(n_ru, n_m)`,
  where `n_ru` is the number of random unitaries and `n_m` is the number of measurements.
- `kth_moment::Int`: The moment `k` to compute (e.g., `k = 1, 2, ...`).
- `O::Union{Nothing, MPO}` (optional): If provided, computes `Tr[O * ρ^k]`; otherwise, computes `Tr[ρ^k]` (default: `nothing`).

# Returns
The computed trace moment as a scalar.

# Example
```julia
moment1 = get_trace_moment(shadows, 1)
moment2 = get_trace_moment(shadows, 2; O=my_operator)
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

    # n_ru, n_m = size(shadows)
    # n_shadows = n_ru * n_m

    # # Validate kth_moment
    # @assert kth_moment >= 1 "Only integer valued moments Tr[ρ^k] with k >= 1 can be computed."
    # @assert kth_moment <= n_shadows "The number of shadows $n_shadows  must be >= the moment $kth_moment. "

    # # Precompute total evaluations (for a warning only)
    # num_permutations = prod(n_ru - i for i in 0:(kth_moment - 1))
    # num_cartesian_products = n_m^kth_moment
    # total_evaluations = num_permutations * num_cartesian_products

    # if total_evaluations > 100000
    #     @warn "Total number of trace product evaluations to estimate moment $kth_moment equals $total_evaluations."
    # end

    # # Precompute all permutations (rows) and the Cartesian product (columns) indices.
    # all_perms = collect(permutations(1:n_ru, kth_moment))
    # cart_prod = collect(CartesianIndices(ntuple(_ -> 1:n_m, kth_moment)))

    # n_perm = length(all_perms)
    # perm_avgs = zeros(Float64, n_perm)


    # # Precompute contributions for each permutation (averaged over the Cartesian product).
    # for idx in 1:n_perm
    #     r = all_perms[idx]
    #     contributions = Float64[]
    #     for m in cart_prod
    #         val = real(get_trace_product((shadows[r[i], m[i]] for i in 1:kth_moment)...; O=O))
    #         push!(contributions, val)
    #     end
    #     perm_avgs[idx] = mean(contributions)
    # end

    # # Full U-statistic estimator: average over all permutation averages.
    # averaging(x) = compute_renyi ? 1/(1 - kth_moment)*log2(mean(x)) : mean(x)

    # θ_est = averaging(perm_avgs)

    # if !compute_sem
    #     return θ_est
    # else
    #     # Precompute jackknife groups: for each sample i, collect indices of permutations that do NOT include i.
    #     jackknife_groups = Vector{Vector{Int}}(undef, n_ru)
    #     for i in 1:n_ru
    #         jackknife_groups[i] = [j for j in 1:n_perm if !(i in all_perms[j])]
    #     end

    #     # Compute jackknife estimates using the precomputed groups.
    #     jack_vals = zeros(Float64, n_ru)
    #     for i in 1:n_ru
    #         # Directly use the precomputed indices for sample i.
    #         filtered_vals = perm_avgs[jackknife_groups[i]]
    #         jack_vals[i] = averaging(filtered_vals)
    #     end

    #     # Jacknife unbiased estimate
    #     θ_jack = n_ru * θ_est - (n_ru - 1) * mean(jack_vals)

    #     # Jackknife variance: (n_ru - 1)^2/n_ru * var(jack_vals)
    #     var_est = (n_ru - 1)^2 / n_ru * var(jack_vals)
    #     se = sqrt(var_est)
    #     return θ_est , θ_est - θ_jack, se
    # end

    # # Loop over all combinations: permutations over rows and Cartesian product over columns
    # est = ComplexF64[]
    # for r in permutations(1:n_ru, kth_moment)
    #     for m in CartesianIndices(ntuple(_ -> 1:n_m, kth_moment))
    #         trace_prod = get_trace_product((shadows[r[i], m[i]] for i in 1:kth_moment)...; O=O)
    #         push!(est, real(trace_prod))
    #     end
    # end

    # if compute_sem
    #     # Compute standard error of the mean (SEM)
    #     sem_value = std(est) / sqrt(length(est))
    #     return mean(est), sem_value
    # else
    #     return mean(est)
    # end

end

"""
    get_trace_moments(
        shadows::Array{<:AbstractShadow,2},
        k_vec::Vector{Int};
        O::Union{Nothing,MPO}=nothing,
        compute_cov::Bool = false,
        as_renyi::Bool = false,
    )

Estimate several trace moments

\\[
T_k = \\operatorname{tr}\\bigl[\\,\\rho^{\\,k}\\bigr]
\\]

(or, if `O` is supplied, the generalized moments
\\( \\operatorname{tr}[\\,O\\,\\rho^{\\,k}] \\))
from an array of classical–shadow objects.
`k_vec` is the vector of integer moments to evaluate.
If `as_renyi=true` the estimator for each \\(k\\) is converted on the fly to
the (binary-log) Rényi-\\(k\\) entropy

\\[
S_k \\,=\\, \\frac{1}{1-k}\\,\\log_{2} T_k .
\\]

Optionally `compute_cov=true` returns, in addition to the vector of point
estimates, the full jack-knife covariance matrix
\\( \\operatorname{Cov}(T_{k_a},T_{k_b}) \\) (or of the Rényi entropies,
depending on `as_renyi`).

# Arguments
- `shadows` : 2-D array of size `(n_ru, n_m)` holding the classical shadows.
- `k_vec`   : vector of positive integers.
- `O`       : optional MPO; if given, moments of \\( O\\,\\rho^{k} \\) are computed.
- `compute_cov` : whether to return the jack-knife covariance matrix.
- `as_renyi`    : return Rényi entropies instead of raw trace moments.

# Returns
- `θ̂::Vector{Float64}`                          : point estimates for each `k_vec[i]`
- `Σ̂::Matrix{Float64}` (if `compute_cov=true`) : jack-knife covariance matrix.
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
    get_trace_moment(shadows::Vector{<:AbstractShadow}, kth_moment::Int; O::Union{Nothing, MPO}=nothing)

Wrapper function. Compute a single trace moment for a vector of shadow objects by reshaping the vector into a 2D array.

# Arguments
- `shadows::Vector{<:AbstractShadow}`: A vector of shadow objects.
- `kth_moment::Int`: The moment order `k` to compute.
- `O::Union{Nothing, MPO}` (optional): An MPO observable.

# Returns
The computed trace moment as a scalar.

# Example
```julia
moment = get_trace_moment(shadows_vector, 2; O=my_operator)
```
"""
function get_trace_moment(shadows::Vector{<:AbstractShadow}, kth_moment::Int; O::Union{Nothing, MPO}=nothing, compute_sem::Bool = false,compute_renyi::Bool = false)
     return get_trace_moment(reshape(shadows, :, 1), kth_moment; O=O, compute_sem = compute_sem, compute_renyi = compute_renyi)
end

"""
    get_trace_moments(shadows::Array{<:AbstractShadow, 2}, kth_moments::Vector{Int}; O::Union{Nothing, MPO}=nothing)

Wrapper function. Compute multiple trace moments from an array of shadow objects.

# Arguments
- `shadows::Array{<:AbstractShadow, 2}`: An array of shadow objects with dimensions `(n_ru, n_m)`.
- `kth_moments::Vector{Int}`: A vector of moment orders.
- `O::Union{Nothing, MPO}` (optional): An MPO observable; if provided, computes `Tr[O * ρ^k]` for each moment (default: `nothing`).

# Returns
A vector of trace moments corresponding to each moment in `kth_moments`.

# Example
```julia
moments = get_trace_moments(shadows_array, [1, 2, 3])
```
"""
# function get_trace_moments(shadows::Array{<:AbstractShadow, 2}, kth_moments::Vector{Int}; O::Union{Nothing, MPO}=nothing , compute_sem::Bool = false,compute_renyi::Bool = false )
#     return [get_trace_moment(shadows, k; O=O, compute_sem = compute_sem, compute_renyi = compute_renyi) for k in kth_moments]
# end

"""
    get_trace_moments(shadows::Vector{<:AbstractShadow}, kth_moments::Vector{Int}; O::Union{Nothing, MPO}=nothing)

Wrapper function. Compute multiple trace moments from a vector of shadow objects by reshaping the vector into a 2D array.

# Arguments
- `shadows::Vector{<:AbstractShadow}`: A vector of shadow objects.
- `kth_moments::Vector{Int}`: A vector of moment orders.
- `O::Union{Nothing, MPO}` (optional): An MPO observable.

# Returns
A vector of trace moments.

# Example
```julia
moments = get_trace_moments(shadows_vector, [1, 2, 3])
```
"""
function get_trace_moments(shadows::Vector{<:AbstractShadow}, kth_moments::Vector{Int}; O::Union{Nothing, MPO}=nothing, compute_cov::Bool = false ,compute_renyi::Bool = false)
    return get_trace_moments(reshape(shadows, :, 1), kth_moments; O=O, compute_cov = compute_cov, compute_renyi = compute_renyi)
end


"""
    get_trace_product(shadows::AbstractShadow...; O::Union{Nothing, MPO}=nothing)

Compute the product of multiple shadow objects and return its trace or expectation value.

If `O` is `nothing`, returns the trace of the product:
    trace(shadow₁ * shadow₂ * ... * shadowₙ).
If `O` is provided, returns the expectation value computed by:
    get_expect_shadow(O, shadow₁ * shadow₂ * ... * shadowₙ).

# Arguments
- `shadows...`: A variable number of shadow objects.
- `O::Union{Nothing, MPO}` (optional): An MPO observable.

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

# Arguments
- `shadow::AbstractShadow`: The shadow object.
- `subsystem::Vector{Int}`: A vector of site indices (1-based) specifying the subsystem to retain.
- `assume_unit_trace::Bool` (optional): If `true`, assumes the shadow has unit trace (default: `false`).
    This can speed up the calculation for factorized shadows (as the trace of "traced out" qubits is not computed)/

# Returns
A new shadow object reduced to the specified subsystem.
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
