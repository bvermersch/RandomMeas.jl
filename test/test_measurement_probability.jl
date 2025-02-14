using RandomMeas
using Test


@testset "MeasurementProbability Tests" begin
    # Test parameters
    N = 2  # Number of qubits
    NM = 50  # Number of measurements per setting
    site_indices = siteinds("Qubit", N)

    # Generate random measurement results
    measurement_results = rand(1:2, NM, N)

    # Generate random local unitary measurement setting
    measurement_setting = LocalUnitaryMeasurementSetting(N;site_indices=site_indices)

    # Create MeasurementData
    measurement_data = MeasurementData(measurement_results; measurement_setting=measurement_setting)

    # Constructor 1: From MeasurementData
    @testset "Constructor from MeasurementData" begin
        measurement_probability = MeasurementProbability(measurement_data)
        @test measurement_probability.N == N
        @test measurement_probability.measurement_setting === measurement_setting
    end

    # Constructor 2: From state (MPS) and setting
    @testset "Constructor from MPS and setting" begin
        ψ = random_mps(site_indices)
        measurement_probability = MeasurementProbability(ψ, measurement_setting)
        @test measurement_probability.N == N
        @test measurement_probability.measurement_setting === measurement_setting
    end

    # Constructor 3: From state (MPO) and setting
    @testset "Constructor from MPO and setting" begin
        ρ = outer(random_mps(site_indices)', random_mps(site_indices))
        measurement_probability = MeasurementProbability(ρ, measurement_setting)
        @test measurement_probability.N == N
        @test measurement_probability.measurement_setting === measurement_setting
    end

     # Constructor 4: From state (MPS) without setting
     @testset "Constructor from MPS without setting" begin
        ψ = random_mps(site_indices)
        measurement_probability = MeasurementProbability(ψ)
        @test measurement_probability.N == N
    end

end
