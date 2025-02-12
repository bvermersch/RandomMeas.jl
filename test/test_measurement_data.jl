using Test
using ITensors
using NPZ
include("../src/MeasurementSetting.jl")
include("../src/MeasurementData.jl")

# Test file for MeasurementData.jl
@testset "MeasurementData Tests" begin
    # Mock data setup
    N = 4  # Number of sites
    NM = 10  # Number of measurements per setting

    # Generate random binary measurement results
    measurement_results = rand(1:2, NM, N)

    # Mock measurement setting
    measurement_setting = LocalUnitaryMeasurementSetting(N, ensemble="Haar")

    # Test 1: Creating MeasurementData with setting
    @testset "With Measurement Setting" begin
        data_with_setting = MeasurementData(measurement_results; measurement_setting=measurement_setting)

        @test data_with_setting.N == N
        @test data_with_setting.NM == NM
        @test data_with_setting.measurement_results == measurement_results
        @test data_with_setting.measurement_setting === measurement_setting
    end

    # Test 2: Creating MeasurementData without setting
    @testset "Without Measurement Setting" begin
        data_without_setting = MeasurementData(measurement_results)

        @test data_without_setting.N == N
        @test data_without_setting.NM == NM
        @test data_without_setting.measurement_results == measurement_results
        @test data_without_setting.measurement_setting === nothing
    end

    # # Test 3: Importing MeasurementData with unitaries
    # @testset "Importing MeasurementData with Unitaries" begin
    #     # Mock export of unitary settings
    #     unitaries_file = "test_unitaries.npy"
    #     export_unitaries(measurement_settings, unitaries_file)

    #     # Mock export of measurement results
    #     results_file = "test_results.npy"
    #     npzwrite(results_file, measurement_results)

    #     # Import data
    #     imported_data = import_measurement_data(results_file; unitaries_path=unitaries_file)

    #     @test imported_data.N == N
    #     @test imported_data.NU == NU
    #     @test imported_data.NM == NM
    #     @test imported_data.measurement_results == measurement_results
    #     @test imported_data.measurement_settings.N == measurement_settings.N
    #     @test imported_data.measurement_settings.NU == measurement_settings.NU

    #     # Cleanup
    #     rm(unitaries_file, force=true)
    #     rm(results_file, force=true)
    # end

    # # Test 4: Importing MeasurementData without unitaries
    # @testset "Importing MeasurementData without Unitaries" begin
    #     # Mock export of measurement results
    #     results_file = "test_results.npy"
    #     npzwrite(results_file, measurement_results)

    #     # Import data without unitaries
    #     imported_data = import_measurement_data(results_file)

    #     @test imported_data.N == N
    #     @test imported_data.NU == NU
    #     @test imported_data.NM == NM
    #     @test imported_data.measurement_results == measurement_results
    #     @test imported_data.measurement_settings === nothing

    #     # Cleanup
    #     rm(results_file, force=true)
    # end

    # # Test 5: Importing only measurement results
    # @testset "Importing Only Measurement Results" begin
    #     # Mock export of measurement results
    #     results_file = "test_results.npy"
    #     npzwrite(results_file, measurement_results)

    #     # Import data with predefined settings
    #     imported_data = import_measurement_results(results_file; measurement_settings=measurement_settings)

    #     @test imported_data.N == N
    #     @test imported_data.NU == NU
    #     @test imported_data.NM == NM
    #     @test imported_data.measurement_results == measurement_results
    #     @test imported_data.measurement_settings === measurement_settings

    #     # Cleanup
    #     rm(results_file, force=true)
    # end
end
