using Test
#using RandomMeas  # Replace with the actual module name
using ITensors

# Include necessary files for the Shadows module
include("../src/MeasurementSettings.jl")
include("../src/MeasurementData.jl")
include("../src/Shadows_Struct.jl")
include("../src/Postprocessing.jl")      # Path to Postprocessing.jl

# Set up test parameters
N = 4  # Number of qubits
NU = 5  # Number of unitaries
NM = 50  # Number of measurements per unitary
site_indices = siteinds("Qubit", N)
measurement_settings = LocalUnitaryMeasurementSettings(N, NU;site_indices=site_indices)
measurement_results = rand(1:2, NU, NM, N)  # Random binary measurement results
measurement_data = MeasurementData(measurement_results; measurement_settings=measurement_settings)

@testset "Shadows Tests" begin
    # Generate shadows
    dense_shadows = get_dense_shadows(measurement_data)
    factorized_shadows = get_factorized_shadows(measurement_data)

    @show size(dense_shadows)
    @show size(factorized_shadows)

    # Generate random observable (for expectation testing)
    observable = outer(random_mps(site_indices)',random_mps(site_indices))

    expect_dense = 0.0
    # Test get_expectation for DenseShadow
    @testset "DenseShadow Expectation" begin
        for r in 1:NU
            for m in 1:1
                expect_dense += get_expect_shadow(observable, dense_shadows[r, m])
            end
        end
        expect_dense /= NU
    end

    expect_factorized = 0.0
    # Test get_expectation for FactorizedShadow
    @testset "FactorizedShadow Expectation" begin
        for r in 1:NU
            for m in 1:NM
                expect_factorized += get_expect_shadow(observable, factorized_shadows[r, m])
            end
        end
        expect_factorized /= NU * NM
    end

    expect_shadow_dense_2 = get_expect_shadow(observable, dense_shadows)
    expect_shadow_factorized_2 = get_expect_shadow(observable, factorized_shadows)

    @show expect_shadow_dense_2
    @show expect_shadow_factorized_2
    @show expect_factorized
    @show expect_dense

    @test isapprox(expect_factorized, expect_dense, atol=1e-10)
    @test isapprox(expect_shadow_dense_2, expect_dense, atol=1e-10)
    @test isapprox(expect_shadow_factorized_2, expect_dense, atol=1e-10)
end
