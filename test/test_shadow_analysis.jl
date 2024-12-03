using Test
using ITensors

# Include necessary files for the Shadows module
include("../src/MeasurementSettings.jl")
include("../src/MeasurementData.jl")
include("../src/AbstractShadows.jl")
include("../src/FactorizedShadows.jl")
include("../src/DenseShadows.jl")
include("../src/Postprocessing.jl")      # Path to Postprocessing.jl

# Set up test parameters
N = 4  # Number of qubits
NU = 5  # Number of unitaries
NM = 50  # Number of measurements per unitary
site_indices = siteinds("Qubit", N)
measurement_settings = LocalUnitaryMeasurementSettings(N, NU; site_indices=site_indices)
measurement_results = rand(1:2, NU, NM, N)  # Random binary measurement results
measurement_data = MeasurementData(measurement_results; measurement_settings=measurement_settings)

@testset "Shadows Tests" begin
    # Generate shadows
    dense_shadows = get_dense_shadows(measurement_data)
    factorized_shadows = get_factorized_shadows(measurement_data)

    @show size(dense_shadows)
    @show size(factorized_shadows)

    # Generate random observable (for expectation testing)
    observable = outer(random_mps(site_indices)', random_mps(site_indices))

    # Timing for DenseShadow Expectation (Method 1)
    expect_dense = 0.0
    dense_time = @elapsed begin
        for r in 1:NU
            for m in 1:1
                expect_dense += get_expect_shadow(observable, dense_shadows[r, m])
            end
        end
        expect_dense /= NU
    end

    # Timing for FactorizedShadow Expectation (Method 2)
    expect_factorized = 0.0
    factorized_time = @elapsed begin
        for r in 1:NU
            for m in 1:NM
                expect_factorized += get_expect_shadow(observable, factorized_shadows[r, m])
            end
        end
        expect_factorized /= NU * NM
    end

    # Timing for batched DenseShadow Expectation (Method 3)
    dense_batch_time = @elapsed begin
        expect_shadow_dense_2 = get_expect_shadow(observable, dense_shadows)
    end

    # Timing for batched FactorizedShadow Expectation (Method 4)
    factorized_batch_time = @elapsed begin
        expect_shadow_factorized_2 = get_expect_shadow(observable, factorized_shadows)
    end

    # Display results
    @show expect_shadow_dense_2
    @show expect_shadow_factorized_2
    @show expect_factorized
    @show expect_dense

    # Timing results
    println("\nTiming Results:")
    println("DenseShadow (individual): $dense_time seconds")
    println("FactorizedShadow (individual): $factorized_time seconds")
    println("DenseShadow (batched): $dense_batch_time seconds")
    println("FactorizedShadow (batched): $factorized_batch_time seconds")

    # Tests
    @test isapprox(expect_factorized, expect_dense, atol=1e-10)
    @test isapprox(expect_shadow_dense_2, expect_dense, atol=1e-10)
    @test isapprox(expect_shadow_factorized_2, expect_dense, atol=1e-10)
end
