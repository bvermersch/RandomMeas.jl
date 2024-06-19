using ITensors

abstract type ClassicalShadow end

mutable struct MPOClassicalShadow <:ClassicalShadow

    site_indices::Vector{Index{Int64}}
    classical_shadow::Vector{MPO}

    function MPOClassicalShadow(site_indices::Vector{Index{Int64}}, classical_shadow::Vector{MPO})
        new(site_indices, classical_shadow)
    end

end

## Explicit constructors for factorized classical shadows to be constructed (<-> in particular factorized classical shadow)

mutable struct DenseClassicalShadow <: ClassicalShadow

    site_indices::Vector{Index{Int64}}
    classical_shadow::Vector{ITensor}

    function DenseClassicalShadow(site_indices::Vector{Index{Int64}}, classical_shadow::Vector{ITensor})
        new(site_indices, classical_shadow)
    end

end


"""
    DenseClassicalShadow(measurement_data::MeasurementData;
                        subystem::Vector{Int64}=collect(1:measurement_data.N),
                        G::Vector{Float64}=ones(Float64,measurement_data.N),
                        number_of_batches::Int64=measurement_data.NU)

Constructs a dense classical shadow object based on the given measurement data.

# Arguments
- `measurement_data::MeasurementData`: The measurement data.
- `subystem::Vector{Int64}`: The subsystem indices of interest. Default is all qubits.
- `G::Vector{Float64}`: The G vectors specifying robust reconstruction for the Hamming tensors. Default is all ones.
- `number_of_batches::Int64`: The number of batches to divide the measurement data into. Default is the number of unitaries.

# Returns
A `DenseClassicalShadow` object.

"""
function DenseClassicalShadow(measurement_data::MeasurementData;subystem::Vector{Int64}=collect(1:measurement_data.N),G::Vector{Float64}=ones(Float64,measurement_data.N),number_of_batches::Int64=measurement_data.NU)

    measurement_data = reduce_to_subystem(measurement_data,subystem) # We reduce the measurement data to the subsystem we are interested in.

    site_indices = measurement_data.site_indices # We get the site indices of the subsystem.

    hamming_tensors = get_hamming_tensor(G, site_indices) # We get the Hamming tensors

    classical_shadow = [ITensor(vcat(site_indices, site_indices')) for _ in 1:number_of_batches] # We initialize the classical shadow.

    batch_size = measurement_data.NU//number_of_batches  # We calculate the batch size. The last batch will contain the remaining unitaries and could be larger.
    for r in 1:measurement_data.NU

        born_probability = get_born_probability(measurement_data[r,:,:])

        #### This part might be wrapped into a helper function
        rho =  2^NA *  ITensor(born_probability, site_indices)
        for i in 1:NA

            s = site_indices[i]
            u = measurement_data.local_unitaries_ITensor[r,i]

            rho *= hamming_tensors[i]
            rho *= δ(s, s', s'')
            ut = u * δ(s'', s)
            ut *= δ(s, s')
            rho = mapprime(ut * rho, 2, 0)
            ut = dag(u) * δ(s'', s)
            rho = mapprime(ut * rho, 2, 1)

        end

        classical_shadow[div(r-1, batch_size) + 1] += rho

    end

    for i in 1:number_of_batches
        classical_shadow[i] = classical_shadow[i] / trace(classical_shadow[i])
    end

    return DenseClassicalShadow(site_indices, classical_shadow)
end


"""
    get_moments(classical_shadow::DenseClassicalShadow, moments::Vector{Int64})

Calculates the moments of the classical shadow.

# Arguments
- `classical_shadow::DenseClassicalShadow`: The dense classical shadow object.
- `moments::Vector{Int64}`: The moments to calculate.

# Returns
A vector of trace moments.

"""
function get_moments(classical_shadow::DenseClassicalShadow, moments::Vector{Int64})

    @assert max(moments) <= length(classical_shadow.classical_shadow) "The maximum trace moment moments must be less or equal than the number of classical shadows."

    site_indices = classical_shadow.site_indices
    classical_shadow = classical_shadow.classical_shadow

    trace_moments = Vector{Float64}()

    for m in moments
        r_a = collect(permutations(1:length(classical_shadow), m)) #m_uplet of n batches
        alpha = length(r_a)
        est = 0
        push!(p, 0)
        for r in r_a
            X = multiply(classical_shadow[r[1]], classical_shadow[r[2]])
            for m1 in 3:m
                X = multiply(X, classical_shadow[r[m1]])
            end
            trace_moments[m-1] += real(trace(X, ξ))
        end
        trace_moments[m-1] /= alpha

    end

    return trace_moments

end

# Helper functions


function get_hamming_tensor(G::Vector{Float64} , site_indices::Vector{Index{Int64}})

    @assert length(G) == length(site_indices)  "The number of G values must be equal to the number of site indices."

    N = length(G)

    hamming_tensors = Array(ITensor, N) # We initialize the Hamming tensors containing information about the robust processing of the data.
    for i in 1:N
        Hamming_matrix = zeros(Float64, (2, 2))
        α  = 3 / (2 * G[i] - 1)
        β  = (G[i] - 2) / (2 * G[i] - 1)
        Hamming_matrix[1, 1] = (α+β)/2
        Hamming_matrix[2, 2] = (α+β)/2
        Hamming_matrix[2, 1] = β/2
        Hamming_matrix[1, 2] = β/2
        hamming_tensors[i] = ITensor(Hamming_matrix, site_indices[i], site_indices[i]'')
    end

end
