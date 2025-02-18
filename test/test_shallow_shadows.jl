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
    measurement_setting = ShallowUnitaryMeasurementSetting(N,depth;site_indices=site_indices)
    measurement_data = MeasurementData(ψ,NM,measurement_setting;mode="MPS/MPO")
    measurement_data = MeasurementData(ψ,NM,measurement_setting;mode="dense")

    # Create MeasurementData
    # measurement_data = MeasurementData(measurement_results; measurement_setting=measurement_setting)
    # measurement_probability = MeasurementProbability(measurement_data)

    # # Constructor 1: From MeasurementData
    # # Test DenseShadow constructor with measurement_probability
    # @testset "Constructor with measurement_probability" begin
    #     shadow = DenseShadow(measurement_probability; G=G)
    #     @test shadow.N == N
    #     @test shadow.site_indices == site_indices
    #     @test isa(shadow.shadow_data, ITensor)
    # end

    # # Test DenseShadow constructor with measurement_data
    # @testset "Constructor with measurement_results" begin
    #     shadow = DenseShadow(measurement_data; G=G)
    #     @test shadow.N == N
    #     @test shadow.site_indices == site_indices
    #     @test isa(shadow.shadow_data, ITensor)
    # end

    # # Test batched dense shadows
    # @testset "Batched Dense Shadows" begin
    #     NU = 10  # Number of random unitaries
    #     NM = 5   # Number of projective measurements per unitary
    #     measurements = Vector{MeasurementData{LocalUnitaryMeasurementSetting}}(undef,NU)
    #     for r in 1:NU
    #         measurement_results = rand(1:2, NM, N)
    #         measurement_setting = LocalUnitaryMeasurementSetting(N; site_indices=site_indices,ensemble="Haar")
    #         measurements[r] = MeasurementData(measurement_results; measurement_setting=measurement_setting)
    #     end
    #     measurement_group = MeasurementGroup(measurements)

    #     batched_shadows = get_dense_shadows(
    #         measurement_group;
    #         G=G,
    #         number_of_ru_batches=3
    #     )

    #     @test length(batched_shadows) == 3
    #     for shadow_batch in batched_shadows
    #         @test isa(shadow_batch, DenseShadow)
    #         @test shadow_batch.N == N
    #     end
    # end
end
