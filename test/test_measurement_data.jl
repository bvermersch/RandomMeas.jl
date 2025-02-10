using Test
using ITensors
using NPZ
include("../src/MeasurementSettings.jl")
include("../src/MeasurementData.jl")

# Test file for MeasurementData.jl
@testset "MeasurementData Tests" begin
    # Mock data setup
    N = 4  # Number of sites
    NU = 3  # Number of measurement settings
    NM = 10  # Number of measurements per setting

    # Generate random binary measurement results
    measurement_results = rand(Bool, NU, NM, N)

    # Mock measurement settings
    measurement_settings = LocalUnitaryMeasurementSettings(N, NU, ensemble="Haar")

    # Test 1: Creating MeasurementData with settings
    @testset "With Measurement Settings" begin
        data_with_settings = MeasurementData(measurement_results; measurement_settings=measurement_settings)

        @test data_with_settings.N == N
        @test data_with_settings.NU == NU
        @test data_with_settings.NM == NM
        @test data_with_settings.measurement_results == measurement_results
        @test data_with_settings.measurement_settings === measurement_settings
    end

    # Test 2: Creating MeasurementData without settings
    @testset "Without Measurement Settings" begin
        data_without_settings = MeasurementData(measurement_results)

        @test data_without_settings.N == N
        @test data_without_settings.NU == NU
        @test data_without_settings.NM == NM
        @test data_without_settings.measurement_results == measurement_results
        @test data_without_settings.measurement_settings === nothing
    end

    # Test 3: Importing MeasurementData with unitaries
    @testset "Importing MeasurementData with Unitaries" begin
        # Mock export of unitary settings
        unitaries_file = "test_unitaries.npy"
        export_unitaries(measurement_settings, unitaries_file)

        # Mock export of measurement results
        results_file = "test_results.npy"
        npzwrite(results_file, measurement_results)

        # Import data
        imported_data = import_measurement_data(results_file; unitaries_path=unitaries_file)

        @test imported_data.N == N
        @test imported_data.NU == NU
        @test imported_data.NM == NM
        @test imported_data.measurement_results == measurement_results
        @test imported_data.measurement_settings.N == measurement_settings.N
        @test imported_data.measurement_settings.NU == measurement_settings.NU

        # Cleanup
        rm(unitaries_file, force=true)
        rm(results_file, force=true)
    end

    # Test 4: Importing MeasurementData without unitaries
    @testset "Importing MeasurementData without Unitaries" begin
        # Mock export of measurement results
        results_file = "test_results.npy"
        npzwrite(results_file, measurement_results)

        # Import data without unitaries
        imported_data = import_measurement_data(results_file)

        @test imported_data.N == N
        @test imported_data.NU == NU
        @test imported_data.NM == NM
        @test imported_data.measurement_results == measurement_results
        @test imported_data.measurement_settings === nothing

        # Cleanup
        rm(results_file, force=true)
    end

    # Test 5: Importing only measurement results
    @testset "Importing Only Measurement Results" begin
        # Mock export of measurement results
        results_file = "test_results.npy"
        npzwrite(results_file, measurement_results)

        # Import data with predefined settings
        imported_data = import_measurement_results(results_file; measurement_settings=measurement_settings)

        @test imported_data.N == N
        @test imported_data.NU == NU
        @test imported_data.NM == NM
        @test imported_data.measurement_results == measurement_results
        @test imported_data.measurement_settings === measurement_settings

        # Cleanup
        rm(results_file, force=true)
    end
end
