using Test
using ITensors
using StatsBase

include("../src/MeasurementSetting.jl")
include("../src/MeasurementData.jl")
include("../src/MeasurementProbability.jl")
include("../src/MeasurementGroup.jl")

include("../src/AbstractShadows.jl")
include("../src/DenseShadows.jl")


include("../src/Shadows.jl")      # Path to Shadows_Struct.jl
include("../src/Postprocessing.jl")      # Path to Postprocessing.jl

@testset "DenseShadow Tests" begin
    # Define parameters
    N = 4  # Number of qubits/sites
    NM = 10 # Number of projective measurements
    ξ = siteinds("Qubit", N)  # Site indices
    #P = get_Born(measurement_results, ξ)  # Compute Born probabilities
    G = [1.2, 0.8, 1.5, 1.0]  # Example G values for robustness
    measurement_results = rand(1:2, NM, N)

    # Generate random local unitary measurement setting
    measurement_setting = LocalUnitaryMeasurementSetting(N;site_indices=ξ)
    # Create MeasurementData
    measurement_data = MeasurementData(measurement_results; measurement_setting=measurement_setting)
    measurement_probability = MeasurementProbability(measurement_data)

    # Constructor 1: From MeasurementData
    # Test DenseShadow constructor with measurement_probability
    @testset "Constructor with measurement_probability" begin
        shadow = DenseShadow(measurement_probability; G=G)
        @test shadow.N == N
        @test shadow.ξ == ξ
        @test isa(shadow.shadow_data, ITensor)
    end

    # Test DenseShadow constructor with measurement_data
    @testset "Constructor with measurement_results" begin
        shadow = DenseShadow(measurement_data; G=G)
        @test shadow.N == N
        @test shadow.ξ == ξ
        @test isa(shadow.shadow_data, ITensor)
    end

    # Test batched dense shadows
    @testset "Batched Dense Shadows" begin
        NU = 10  # Number of random unitaries
        NM = 5   # Number of projective measurements per unitary
        measurements = Vector{MeasurementData{LocalUnitaryMeasurementSetting}}(undef,NU)
        for r in 1:NU
            measurement_results = rand(1:2, NM, N)
            measurement_setting = LocalUnitaryMeasurementSetting(N; site_indices=ξ,ensemble="Haar")
            measurements[r] = MeasurementData(measurement_results; measurement_setting=measurement_setting)
        end
        measurement_group = MeasurementGroup(measurements)
    
        batched_shadows = get_dense_shadows(
            measurement_group;
            G=G,
            number_of_ru_batches=3
        )

        @test length(batched_shadows) == 3
        for shadow_batch in batched_shadows
            @test isa(shadow_batch, DenseShadow)
            @test shadow_batch.N == N
        end
    end
end