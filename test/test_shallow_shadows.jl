using RandomMeas
using Test

@testset "ShallowShadow Tests" begin
    # Define parameters
    N = 4  # Number of qubits/sites
    NM = 10 # Number of projective measurements
    site_indices = siteinds("Qubit", N)  # Site indices
    depth = 2 #depth of the shallow random circuit
    ψ = random_mps(site_indices)


    # Generate random local unitary measurement setting
    @testset "ShallowMeasurementSetting" begin
        measurement_setting = ShallowUnitaryMeasurementSetting(N,depth;site_indices=site_indices)
        measurement_data_mps = MeasurementData(ψ,NM,measurement_setting;mode=TensorNetwork)
measurement_data_mpo = MeasurementData(outer(ψ',ψ),NM,measurement_setting;mode=TensorNetwork)
measurement_data_dense = MeasurementData(ψ,NM,measurement_setting;mode=Dense)
    end

    @testset "Shallow MeasurementGroup & Shadows" begin
        NU = 10
        depth = 2
        χ = 8
        nsweeps = 4

        println("measurement group")
        measurement_group_mps = MeasurementGroup(ψ,NU,NM; setting_type=ShallowUnitaryMeasurementSetting, depth=depth, mode=TensorNetwork)
        measurement_group_mpo = MeasurementGroup(outer(ψ',ψ),NU,NM; setting_type=ShallowUnitaryMeasurementSetting, depth=depth, mode=TensorNetwork)
        measurement_group_dense = MeasurementGroup(ψ,NU,NM; setting_type=ShallowUnitaryMeasurementSetting, depth=depth, mode=Dense)

        println("mps depolarization")
        shallow_depolarization_mps = get_shallow_depolarization_mps(measurement_group_mps)
        average_shallow_depolarization_mps = get_average_mps(shallow_depolarization_mps,χ,nsweeps)
        s = siteinds("Qubit", N)  # Input Site indices
        ξ = siteinds("Qubit", N)  # Output Site indices


        println("building shallow map")
        shallow_map = get_depolarization_map(average_shallow_depolarization_mps,s,ξ)
        for i in 1:N
            @assert hasinds(shallow_map[i],s[i])
            @assert hasinds(shallow_map[i],s'[i])
            @assert hasinds(shallow_map[i],ξ[i])
            @assert hasinds(shallow_map[i],ξ'[i])
        end

        println("inversing shallow map")
        v = siteinds("Qubit", N)  # Virtual Site indices

        inverse_depolarization_mps_data_init = random_mps(Float64,v;linkdims=χ).data
        #loss = loss_inverse_depolarization_map(inverse_depolarization_mps,shallow_map,s,ξ)
        loss(x) = loss_inverse_depolarization_map(x,shallow_map,v,s,ξ)
        println("initial loss ",loss(inverse_depolarization_mps_data_init))

        optimizer = LBFGS(; maxiter=100, verbosity=1, gradtol = 1e-6)
        loss_and_grad(x) = loss(x),loss'(x)
        inverse_depolarization_mps_data, fs, gs, niter, normgradhistory = optimize(loss_and_grad, inverse_depolarization_mps_data_init, optimizer)
        #@show inverse_depolarization_mps_data
        inverse_depolarization_mps = MPS(inverse_depolarization_mps_data)


        η = siteinds("Qubit", N)  # Virtual Site indices
        inverse_shallow_map = get_depolarization_map(inverse_depolarization_mps,ξ,η)

        #c#ombined_map = [depolarization_map[i]*inverse_depolarization_map[i] for i in 1:N]
        #identity_map =
        #@show inds(flatten(shallow_map))
        #@show inds(flatten(inverse_shallow_map))
    end
end
