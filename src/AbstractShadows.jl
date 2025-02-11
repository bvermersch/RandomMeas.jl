# Abstract Shadow Type
"""
    AbstractShadow

An abstract type representing a general classical shadow.
Subtypes should implement specific shadow methodologies, such as dense or factorized shadows.
"""
abstract type AbstractShadow end


"""
    get_expect_shadow(O::MPO, shadows::AbstractArray{<:AbstractShadow}; compute_sem::Bool = false)

Compute the average expectation value of an MPO operator `O` using an array of shadows.
Optionally computes the standard error of the mean (SEM) of the expectation values.

# Arguments:
- `O::MPO`: The MPO operator for which the expectation value is computed.
- `shadows::AbstractArray{<:AbstractShadow}`: An array of shadow objects, either dense or factorized, of any dimension.
- `compute_sem::Bool` (optional): Whether to compute and return the standard error of the mean (SEM) (default: `false`).

# Returns:
- If `compute_sem` is `false`: The average expectation value as a `ComplexF64` (or `Float64` if purely real).
- If `compute_sem` is `true`: A tuple `(mean, sem)` where `mean` is the average expectation value and `sem` is the standard error of the mean.
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

Compute the expectation value of an MPO operator `O` using a generic shadow.

# Arguments:
- `O::MPO`: The MPO operator for which the expectation value is computed.
- `shadow::AbstractShadow`: A shadow object, either dense or factorized.

# Returns:
The expectation value as a `ComplexF64` (or `Float64` if purely real).
"""
function get_expect_shadow(O::MPO, shadow::AbstractShadow)
    if shadow isa DenseShadow
        return get_expect_shadow(O, shadow::DenseShadow)
    elseif shadow isa FactorizedShadow
        return get_expect_shadow(O, shadow::FactorizedShadow)
    else
        throw(ArgumentError("Unsupported shadow type: $(typeof(shadow))"))
    end
end

"""
    get_trace_moment(shadows::Array{<:AbstractShadow, 2}, kth_moment::Int; O::Union{Nothing, MPO}=nothing)

Compute a single trace moment from an array of `AbstractShadow` objects.

# Arguments
- `shadows::Array{<:AbstractShadow, 2}`: An array of shadows with dimensions `(n_ru, n_m)`,
  where `n_ru` is the number of random unitaries and `n_m` is the number of measurements.
- `kth_moment::Int`: The moment `k` to compute (e.g., `k = 1, 2, ...`).
- `O::Union{Nothing, MPO}` (optional): If provided, computes `Tr[O * rho^k]`; otherwise, computes `Tr[rho^k]`
  (default: `nothing`).

# Returns
The computed trace moment (averaged over all permutations and Cartesian products) as a scalar.
"""
function get_trace_moment(shadows::Array{<:AbstractShadow, 2}, kth_moment::Int; O::Union{Nothing, MPO}=nothing)
    n_ru, n_m = size(shadows)
    n_shadows = n_ru * n_m

    # Validate kth_moment
    @assert kth_moment >= 1 "Only integer valued moments Tr[ρ^k] with k >= 1 can be computed."
    @assert kth_moment <= n_shadows "The number of shadows must be >= the largest moment k."

    # Precompute total evaluations (for a warning only)
    num_permutations = prod(n_ru - i for i in 0:(kth_moment - 1))
    num_cartesian_products = n_m^kth_moment
    total_evaluations = num_permutations * num_cartesian_products

    if total_evaluations > 10000
        @warn "Total number of trace product evaluations to estimate moment $kth_moment equals $total_evaluations."
    end

    # Loop over all combinations: permutations over rows and Cartesian product over columns
    est = ComplexF64[]
    for r in permutations(1:n_ru, kth_moment)
        for m in CartesianIndices(ntuple(_ -> 1:n_m, kth_moment))
            trace_prod = get_trace_product((shadows[r[i], m[i]] for i in 1:kth_moment)...; O=O)
            push!(est, real(trace_prod))
        end
    end

    return mean(est)
end

"""
    get_trace_moment(shadows::Vector{<:AbstractShadow}, kth_moments::Vector{Int}; O::Union{Nothing, MPO}=nothing)

Wrapper for computing trace moments from a vector of shadows. Reshapes the vector into a 2D array
and then calls the main `get_trace_moment` function.
"""
function get_trace_moments(shadows::Vector{<:AbstractShadow}, kth_moment::Int; O::Union{Nothing, MPO}=nothing)
    return get_trace_moments(reshape(shadows, :, 1), kth_moment; O=O)
end

"""
    get_trace_moments(shadows::Array{<:AbstractShadow, 2}, kth_moments::Vector{Int}; O::Union{Nothing, MPO}=nothing)

Compute multiple trace moments by calling `get_trace_moment` for each requested moment.

# Arguments
- `shadows::Array{<:AbstractShadow, 2}`: An array of shadows with dimensions `(n_ru, n_m)`.
- `kth_moments::Vector{Int}`: A vector of integers specifying which moments to compute.
- `O::Union{Nothing, MPO}` (optional): An MPO observable. If provided, computes `Tr[O * rho^k]`
  for each moment (default: `nothing`).

# Returns
A vector of trace moments corresponding to each moment in `kth_moments`.
"""
function get_trace_moments(shadows::Array{<:AbstractShadow, 2}, kth_moments::Vector{Int}; O::Union{Nothing, MPO}=nothing)
    return [get_trace_moment(shadows, k; O=O) for k in kth_moments]
end

"""
    get_trace_moments(shadows::Vector{<:AbstractShadow}, kth_moments::Vector{Int}; O::Union{Nothing, MPO}=nothing)

Wrapper for computing trace moments from a vector of shadows. Reshapes the vector into a 2D array
and then calls the main `get_trace_moments` function.
"""
function get_trace_moments(shadows::Vector{<:AbstractShadow}, kth_moments::Vector{Int}; O::Union{Nothing, MPO}=nothing)
    return get_trace_moments(reshape(shadows, :, 1), kth_moments; O=O)
end


"""
    get_trace_product(shadows...; O::Union{Nothing, MPO}=nothing)

Compute the product of multiple shadow objects. By default (when `O` is `nothing`),
this function returns the trace of the product:

    trace(shadow₁ * shadow₂ * ... * shadowₙ).

If an MPO observable `O` is provided, instead of computing the trace, the function
returns the expectation value computed by

    get_expect_shadow(O, shadow₁ * shadow₂ * ... * shadowₙ).

# Arguments
- `shadows...`: A variable number of `AbstractShadow` objects to multiply.
- `O::Union{Nothing, MPO}` (optional): An MPO observable. If provided, the expectation
  value is computed using `get_expect_shadow(O, result)` (default: `nothing`).

# Returns
The trace of the product if `O` is `nothing`, or the expectation value if `O` is provided.
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

Multiply two shadow objects of the same type.

# Arguments:
- `shadow1::AbstractShadow`: The first shadow object.
- `shadow2::AbstractShadow`: The second shadow object.

# Returns:
A new shadow object representing the product.

# Notes:
- Throws an error if the types of `shadow1` and `shadow2` do not match.
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
- `shadow::AbstractShadow`: The shadow object.

# Returns:
The trace of the shadow object as a scalar value.
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

# Returns:
An array of trace values corresponding to each shadow, with the same dimensions as the input array.
"""
function trace(shadows::AbstractArray{<:AbstractShadow})
    return [trace(shadow) for shadow in shadows]
end


"""
    partial_trace(shadow::AbstractShadow, subsystem::Vector{Int})

Compute the partial trace of a shadow object over the complement of the specified subsystem.

# Arguments
- `shadow::AbstractShadow`: The shadow object to compute the partial trace for.
- `subsystem::Vector{Int}`: A vector of site indices (1-based) specifying the subsystem to retain.

# Returns
A new shadow object reduced to the specified subsystem.
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
