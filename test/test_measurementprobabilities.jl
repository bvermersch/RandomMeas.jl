using Test
using ITensors,ITensorMPS
using StatsBase
using RandomMeas



@testset "MeasurementProbabilities Tests" begin
    # Test parameters
    N = 4  # Number of qubits
    NU = 5  # Number of unitary settings
    NM = 50  # Number of measurements per setting
    site_indices = siteinds("Qubit", N)

    # Generate random measurement results
    measurement_results = rand(1:2, NU, NM, N)

    # Generate random local unitary measurement settings
    measurement_settings = sample_local_random_unitaries(N, NU;site_indices=site_indices)

    # Create MeasurementData
    measurement_data = MeasurementData(measurement_results; measurement_settings=measurement_settings)

    # Constructor 1: From MeasurementData
    @testset "Constructor from MeasurementData" begin
        measurement_probabilities = MeasurementProbabilities(measurement_data)
        @test measurement_probabilities.N == N
        @test measurement_probabilities.NU == NU
        @test length(measurement_probabilities.measurement_probabilities) == NU
        @test measurement_probabilities.measurement_settings === measurement_settings
    end

    # Constructor 2: From state (MPS) and settings
    @testset "Constructor from MPS and settings" begin
        ψ = random_mps(site_indices)
        measurement_probabilities = MeasurementProbabilities(ψ, measurement_settings)
        @test measurement_probabilities.N == N
        @test measurement_probabilities.NU == NU
        @test length(measurement_probabilities.measurement_probabilities) == NU
        @test measurement_probabilities.measurement_settings === measurement_settings
    end

    # Constructor 3: From state (MPO) and settings
    @testset "Constructor from MPO and settings" begin
        ρ = outer(random_mps(site_indices)', random_mps(site_indices))
        measurement_probabilities = MeasurementProbabilities(ρ, measurement_settings)
        @test measurement_probabilities.N == N
        @test measurement_probabilities.NU == NU
        @test length(measurement_probabilities.measurement_probabilities) == NU
        @test measurement_probabilities.measurement_settings === measurement_settings
    end

    # Test individual Born probabilities
    @testset "Born probabilities" begin
        P = get_Born(measurement_results[1, :, :], site_indices)
        @test P isa ITensor
        @test sum(array(P)) ≈ 1.0  # Ensure normalization
    end
end
