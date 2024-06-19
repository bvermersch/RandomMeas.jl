"""
    struct MeasurementData

The `MeasurementData` struct represents a collection of measurement data taken on a quantum system with N qubits. It contains the number of measurement unitaries, the number of measurements per unitary, the number of qubits, the indices of the qubits, the measurement results, and the local unitaries applied to rotate into a certain measurement setting.

# Fields
- `NU::Int`: The number of measurement unitaries.
- `NM::Int`: The number of measurements per unitary.
- `N::Int`: The number of qubits.
- `site_indices::Vector{Index{Int64}}`: The ITensor indices of the qubits.
- `measurement_results::Array{Int, 3}`: The measurement results obtained from projective measurements, an array of integers (0,1) of dimension NU x NM x N.
- `local_unitaries::Array{ITensor, 2}`: The local unitaries applied to rotate into certain measurement settings, an array of dimension NU x N.

# Constructors
- `MeasurementData(measurement_results::Array{Int, 3}, local_unitaries::Array{Matrix{ComplexF64},2}; site_indices::Vector{Index{Int64}} = siteinds("Qubit", size(measurement_results, 3)})`: Constructs a `MeasurementData` object using multiple dispatch.

"""
struct MeasurementData

    NU::Int
    NM::Int
    N::Int
    site_indices::Vector{Index{Int64}}
    measurement_results::Array{Int, 3}
    local_unitaries::Array{ITensor, 2}

    function MeasurementData(measurement_results::Array{Int, 3}, local_unitaries::Array{Matrix{ComplexF64},2}; site_indices::Vector{Index{Int64}} = siteinds("Qubit", size(measurement_results, 3)))

        NU, NM, N = size(measurement_results)
        @assert size(local_unitaries, 1) == NU "Dimension mismatch: local_unitaries first dimension should be $NU"
        @assert size(local_unitaries, 2) == N "Dimension mismatch: local_unitaries second dimension should be $N"
        @assert length(site_indices) == N "Dimension mismatch: site_indices should have length $N"

        local_unitaries_ITensor = Array{ITensor, 2}(undef, NU, N)
        for r in 1:NU
            for i in 1:N
                local_unitaries_ITensor[r, i] = ITensor(local_unitaries[r, i], site_indices[i], prime(site_indices[i]))
            end
        end
        new(NU, NM, N, site_indices, measurement_results, local_unitaries_ITensor)
    end

end



"""
    get_born_probabilities(measurement_data::MeasurementData; site_indices::Vector{Index{Int64}}=measurement_data.site_indices)

Compute the Born probabilities for a given `measurement_data` object.

# Arguments
- `measurement_data::MeasurementData`: The measurement data object containing the measurement results.
- `site_indices::Vector{Index{Int64}}`: (optional) The indices of the sites for which to compute the probabilities. Defaults to `measurement_data.site_indices`.

# Returns
- `probT::Array{ITensor, 2}`: The Born probabilities for each unitary.

# Example

"""
function get_born_probabilities(measurement_data::MeasurementData;site_indices::Vector{Index{Int64}}=measurement_data.site_indices)

    NU, NM, N = size(measurement_data.measurement_results)
    probT = zeros(Float64, NU, 2^N)
    for u in 1:NU
        prob = get_born_probabilities(measurement_data.measurement_results[u,:,:])
        probT[u, :] = TIensor(prob, site_indices)
    end
    return probT
end


function reduce_to_subystem(MeasurementData::measurement_data, subsystem::Vector{Int})
    @assert length(subsystem) <= MeasurementData.N "Dimension mismatch: subsystem should have length less than or equal to $MeasurementData.N"
    @assert all(1 .<= subsystem .<= MeasurementData.N) "Dimension mismatch: subsystem should have elements between 1 and $MeasurementData.N"
    @assert subsystem == sort(subsystem) "Subsystem should be ordered"
    site_indices = MeasurementData.site_indices[subsystem]
    measurement_results = MeasurementData.measurement_results[:, :, subsystem]
    local_unitaries = MeasurementData.local_unitaries[:, subsystem]
    return MeasurementData(measurement_results, local_unitaries, site_indices=site_indices)
end


#### Helper functions
"""
    get_born_probabilities(data::Array{Int8,2})

Construct histogram from randomized measurements as a vector of dimension 2^N
"""
function get_born_probabilities(data::Array{Int8,2})
	NM,N = size(data)
	probf = StatsBase.countmap(eachrow(data))
	prob = zeros(Int64,(2*ones(Int,N))...)
	for (state,val) in probf
		prob[state...] = val
	end
	return prob/NM
end
