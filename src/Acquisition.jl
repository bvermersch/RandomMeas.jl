function simulate_local_measurements(
    state::Union{MPO, MPS},
    NM::Int64;
    mode::Union{String, Nothing} = nothing,
    measurement_settings::Union{LocalUnitaryMeasurementSettings, Nothing} = nothing,
)::MeasurementData{LocalUnitaryMeasurementSettings}

    #TODO Index compability check state and settings
    #TODO  simulate_RandomMeas_dense, ..., require local unitaries to be passed. Thus, we generate a trivial measurement setting object for computational basis measurements. Maybe we should change this.

    N = length(state)
    ξ = [ siteind(state,j; plev=0) for j in 1:N ]  # Extract the site indices of the state

    if measurement_settings == nothing
        measurement_settings = LocalUnitaryMeasurementSettings(N,1;site_indices=ξ,ensemble="CompBasis") # Default to computational basis measurements
    end

    NU = measurement_settings.NU
    local_unitaries = measurement_settings.local_unitaries
    @assert N == measurement_settings.N "Incompatible number of sites"

    # Allocate memory for the measurement results: NU x NM x N
    measurement_results = Array{Int}(undef, NU, NM, N)

    # Loop over measurement settings
    for r in 1:NU
        u = local_unitaries[r, :]  # Extract the unitaries for the r-th measurement setting

        # Perform NM measurements for the current setting
        if mode == "dense"
            measurement_results[r, :, :] .= simulate_local_measurements_dense(state, u, NM)
        else
            measurement_results[r, :, :] .= simulate_local_measurements_TN(state, u, NM)
        end

    end

    # Return the results as a MeasurementData object
    return MeasurementData(
        measurement_results;
        measurement_settings=measurement_settings
    )
end


"""
    simulate_local_measurements_dense(ρ::Union{MPO,MPS}, u::Vector{ITensor}, NM::Int64)

Sample local measurements from a probability vector (2^N dimensional) constracted from an MPS/MPO ρ
"""
function simulate_local_measurements_dense(ρ::Union{MPO,MPS}, u::Vector{ITensor}, NM::Int64)
    if typeof(ρ)==MPS
        ρu = apply(u,ρ)
    else
        ρu = apply(u,ρ;apply_dag=true)
    end
    return get_samples_dense(ρu,NM)
end

"""
    simulate_local_measurements_TN(ρ::MPO, u::Vector{ITensor}, NM::Int64)

Sample lcoal measurements from an MPO representation ρ. The sampling is based from the MPO directly, i.e., is memory-efficient
"""
function simulate_local_measurements_TN(ρ::MPO, u::Vector{ITensor}, NM::Int64)
    ξ = firstsiteinds(ρ;plev=0)
    ρu = apply(u,ρ;apply_dag=true)
    N= length(u)
    data = zeros(Int,NM,N)
    ρu[1] /= trace(ρu, ξ)
    if N > 1
        for m in 1:NM
            data[m, :] = ITensors.sample(ρu)
        end
    else
        s = ξ[1]
        prob = [real(ρu[1][s=>1, s'=>1][]), real(ρu[1][s=>2, s'=>2][])]
        data[:, 1] = StatsBase.sample(1:2, StatsBase.Weights(prob), NM)
    end
    return data
end



"""
    simulate_local_measurements_TN(ψ::MPS, u::Vector{ITensor},NM::Int64)

Sample randomized measurements from an MPS representation ψ. The sampling is based from the MPS directly, i.e is memory-efficient
"""
function simulate_local_measurements_TN(ψ::MPS, u::Vector{ITensor},NM::Int64)
    N = length(ψ)
    data = zeros(Int,NM,N)
    ψu = apply(reverse(u),ψ) #using reverse allows us to maintain orthocenter(ψ)=1 ;)
    for m in 1:NM
        data[m, :] = ITensors.sample(ψu)#[1:NA]
    end
    return data
end


"""
    get_samples_dense(state::Union{MPO,MPS},NM::Int64)

Sample randomized measurements a 2^N probability vector generated from an MPS (pure state) or MPO (mixed state)
"""
function get_samples_dense(state::Union{MPO,MPS},NM::Int64)
    N = length(state)
    data_s = zeros(Int,NM,N)
    #Note: This is borrowed from PastaQ
    Prob = get_Born(state)
    prob = real(array(Prob))
    prob = reshape(prob, 2^N)
    for m in 1:NM
        data = StatsBase.sample(0:(1<<N-1), StatsBase.Weights(prob), 1)
        data_s[m, :] = 1 .+ digits(data[1], base=2, pad=N)
    end

    return data_s
end
