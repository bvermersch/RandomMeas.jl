using Test
using ITensors
using StatsBase

include("../src/MeasurementSettings.jl")
include("../src/MeasurementData.jl")
include("../src/AbstractShadows.jl")
include("../src/DenseShadows.jl")


include("../src/Shadows.jl")      # Path to Shadows_Struct.jl
include("../src/Postprocessing.jl")      # Path to Postprocessing.jl

@testset "DenseShadow Tests" begin
    # Define parameters
    N = 4  # Number of qubits/sites
    ξ = siteinds("Qubit", N)  # Site indices
    local_unitaries = [op("RandomUnitary", ξ[i]) for i in 1:N]  # Generate random unitaries
    measurement_results = rand(1:2, 10, N)  # Simulated binary results (1, 2 for Julia indexing)
    P = get_Born(measurement_results, ξ)  # Compute Born probabilities
    G = [1.2, 0.8, 1.5, 1.0]  # Example G values for robustness

    # Test DenseShadow constructor with ITensor P
    @testset "Constructor with ITensor P" begin
        shadow = DenseShadow(P, local_unitaries; G=G)
        @test shadow.N == N
        @test shadow.ξ == ξ
        @test isa(shadow.shadow_data, ITensor)
    end

    # Test DenseShadow constructor with measurement_results
    @testset "Constructor with measurement_results" begin
        shadow = DenseShadow(measurement_results, local_unitaries; G=G)
        @test shadow.N == N
        @test shadow.ξ == ξ
        @test isa(shadow.shadow_data, ITensor)
    end

    # Test batched dense shadows
    @testset "Batched Dense Shadows" begin
        NU = 10  # Number of random unitaries
        NM = 5   # Number of projective measurements per unitary
        measurement_data = MeasurementData(
            rand(1:2, NU, NM, N),
            measurement_settings=LocalUnitaryMeasurementSettings(N, NU)
        )
        batched_shadows = get_dense_shadows(
            measurement_data;
            G=G,
            number_of_ru_batches=3,
            number_of_projective_measurement_batches=2
        )

        @test size(batched_shadows) == (3, 2)
        for shadow_batch in batched_shadows
            @test isa(shadow_batch, DenseShadow)
            @test shadow_batch.N == N
        end
    end
end


@testset "Dense Shadows Comparison Tests" begin
    # Parameters
    N = 4   # Number of qubits/sites
    NU = 5  # Number of random unitaries
    NM = 10 # Number of projective measurements
    G = fill(1.0, N) # G values (no measurement errors)

    # Generate measurement settings
    measurement_settings = LocalUnitaryMeasurementSettings(N, NU, ensemble="Haar")

    # Simulate measurement results
    ξ = measurement_settings.site_indices
    local_unitaries = measurement_settings.local_unitaries
    measurement_results = rand(0:1, NU, NM, N) .+ 1 # Ensure 1, 2 indexing
    measurement_data = MeasurementData(measurement_results; measurement_settings=measurement_settings)

    # Compute dense shadows using the new method
    dense_shadows = get_dense_shadows(measurement_data; G=G)

    # Compare with old method for each unitary
    for r in 1:NU
        # Compute P as the average over all projective measurements
        P = get_Born(measurement_results[r, :, :], ξ)

        # Compute shadow using the old method
        old_shadow = get_shadow(P, ξ, local_unitaries[r, :]; G=G)

        # Extract shadow from the new method
        new_shadow = dense_shadows[r, 1].shadow_data

        # Compare the two shadows
        @test isapprox(new_shadow, old_shadow, atol=1e-10)
    end
end
