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
    get_trace_moment(shadows::Array{AbstractShadow, 2}, kth_moment::Int)

Compute a single trace moment from an array of `AbstractShadow` objects.

# Arguments
- `shadows::Array{AbstractShadow, 2}`: An array of shadows with dimensions `(n_ru, n_m)`,
  where `n_ru` is the number of random unitaries and `n_m` is the number of measurements.
- `kth_moment::Int`: The moment `k` to compute (e.g., `k=1,2,...`).

# Returns
The computed trace moment for `kth_moment`.
"""
function get_trace_moment(shadows::Array{<:AbstractShadow, 2}, kth_moment::Int)
    return get_trace_moments(shadows, [kth_moment])[1]
end

"""
    get_trace_moments(shadows::Vector{<:AbstractShadow}, kth_moments::Vector{Int})

Wrapper for computing trace moments from a vector of shadows.

# Arguments
- `shadows::Vector{<:AbstractShadow}`: A vector of shadows.
- `kth_moments::Vector{Int}`: A vector of integers specifying the moments to compute.

# Returns
A vector of trace moments corresponding to `kth_moments`.
"""
function get_trace_moments(shadows::Vector{<:AbstractShadow}, kth_moments::Vector{Int})
    # Reshape the vector to a 2D array with a trivial second dimension
    return get_trace_moments(reshape(shadows, :, 1), kth_moments)
end

"""
    get_trace_moment(shadows::Vector{<:AbstractShadow}, kth_moment::Int)

Wrapper for computing a single trace moment from a vector of shadows.

# Arguments
- `shadows::Vector{<:AbstractShadow}`: A vector of shadows.
- `kth_moment::Int`: The moment to compute (e.g., `k=1,2,...`).

# Returns
The computed trace moment for `kth_moment`.
"""
function get_trace_moment(shadows::Vector{<:AbstractShadow}, kth_moment::Int)
    return get_trace_moments(shadows, [kth_moment])[1]
end


"""
    get_trace_moments(shadows::Array{AbstractShadow, 2}, kth_moments::Vector{Int})

Compute trace moments from an array of `AbstractShadow` objects.

# Arguments
- `shadows::Array{AbstractShadow, 2}`: An array of shadows with dimensions `(n_ru, n_m)`,
  where `n_ru` is the number of random unitaries and `n_m` is the number of measurements.
- `kth_moments::Vector{Int}`: A vector of integers specifying the moments to compute.

# Returns
A vector of trace moments corresponding to `kth_moments`.

# Notes
- Uses all combinations of rows (random unitaries) and Cartesian products of columns (measurements) to compute the moments.
"""
function get_trace_moments(shadows::Array{<:AbstractShadow, 2}, kth_moments::Vector{Int})
    #TODO: The trace moment function is invariant under cyclic permutations. We can exploit this to reduce the number of evaluations.
    n_ru, n_m = size(shadows)
    p = Float64[]  # Initialize the vector for trace moments
    n_shadows = n_ru * n_m

    # Validate kth_moments
    @assert all(kth_moments .>= 1) "Only integer valued moments Tr[ρ^k] with k >= 1 can be computed."
    @assert all(kth_moments .<= n_shadows) "The number of shadows must be >= the largest moment k."

    # Precompute total evaluations
    total_evaluations = 0
    for k in kth_moments
        num_permutations = prod(n_ru - i for i in 0:(k - 1))  # Number of row permutations
        num_cartesian_products = n_m^k  # Cartesian product over columns
        total_evaluations += num_permutations * num_cartesian_products
    end

    # Issue warning if total evaluations exceed threshold
    if total_evaluations>10000
        @warn "Total number of trace_product function evaluations to estimate all $kth_moments moments equals $total_evaluations."
    end

    for k in kth_moments
        est = []  # Store estimates for this k
        for r in permutations(1:n_ru, k)  # Permutations over rows
            for m in CartesianIndices(ntuple(_ -> 1:n_m, k))  # Cartesian product over columns
                # Compute trace product of shadows for this permutation and Cartesian indices
                trace_prod = get_trace_product((shadows[r[i], m[i]] for i in 1:k)...)
                push!(est, trace_prod)
            end
        end
        push!(p, real(mean(est)))  # Average over all combinations
    end

    return p
end

"""
    get_trace_product(shadows...)

Compute the trace of the product of multiple shadows.

# Arguments
- `shadows...`: A variable number of `AbstractShadow` objects to multiply.

# Returns
The trace of the product.
"""
function get_trace_product(shadows::AbstractShadow...)
    result = shadows[1]
    for shadow in shadows[2:end]
        result = multiply(result, shadow)
    end
    return trace(result)
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

# #### Helper functions


# """
#     square(shadow::ITensor)
# """
# function square(shadow::ITensor)
#     Y = multiply(shadow, shadow)
#     return Y
# end

# """
#     multiply(shadow::ITensor, shadow2::ITensor)
# """
# function multiply(shadow::ITensor, shadow2::ITensor)
#     return mapprime(shadow * prime(shadow2), 2, 1)
# end

# """
#     power(shadow::ITensor, n::Int64)
# """
# function power(shadow::ITensor, n::Int64)
#     Y = deepcopy(shadow)
#     for m in 1:n-1
#         Y = multiply(Y, shadow)
#     end
#     return Y
# end

# """
#     trace(shadow::ITensor, ξ::Vector{Index{Int64}})
# """
# function trace(shadow::ITensor, ξ::Vector{Index{Int64}})
#     NA = size(ξ, 1)
#     Y = copy(shadow)
#     for i in 1:NA
#         Y *= δ(ξ[i],ξ[i]')
#     end
#     return Y[]
# end
