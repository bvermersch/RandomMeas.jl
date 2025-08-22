using RandomMeas
using Test


# Test file for MeasurementData.jl
@testset "MeasurementData Tests" begin
    # Mock data setup
    N = 4  # Number of sites
    NM = 10  # Number of measurements per setting

    # Generate random binary measurement results
    measurement_results = rand(1:2, NM, N)
    # Mock measurement setting
    measurement_setting = LocalUnitaryMeasurementSetting(N, ensemble=Haar)

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

    # Test 3: Sampling MeasurementData from a pure theory state ψ
    @testset "Sampling MeasurementData from a pure theory state ψ " begin
        ξ = measurement_setting.site_indices
        ψ = random_mps(ξ)

        #dense mode
        data = MeasurementData(ψ,NM,measurement_setting,mode=Dense)
        @test data.N == N
        @test data.NM == NM
        @test data.measurement_setting === measurement_setting

        #mps mode
        data = MeasurementData(ψ,NM,measurement_setting;mode=TensorNetwork)
        @test data.N == N
        @test data.NM == NM
        @test data.measurement_setting === measurement_setting
    end

    # Test 4: Sampling MeasurementData from a mixed state ρ
    @testset "Sampling MeasurementData from a mixed state ρ " begin
        ξ = measurement_setting.site_indices
        ψ = random_mps(ξ)
        ρ = outer(ψ',ψ)

        #dense mode
        data = MeasurementData(ρ,NM,measurement_setting;mode=Dense)
        @test data.N == N
        @test data.NM == NM
        @test data.measurement_setting === measurement_setting

        #mps mode
        data = MeasurementData(ρ,NM,measurement_setting;mode=TensorNetwork)
        @test data.N == N
        @test data.NM == NM
        @test data.measurement_setting === measurement_setting
    end

    @testset "MeasurementData Import/Export Tests" begin
        # Setup parameters
        N = 4       # Number of sites (qubits)
        NM = 10     # Number of measurements per setting

        # Generate random binary measurement results (values 1 or 2)
        measurement_results = rand(1:2, NM, N)

        # Create a measurement setting (using the Identity ensemble for clarity)
        setting = LocalUnitaryMeasurementSetting(N; ensemble=Haar)

            data_with_setting = MeasurementData(measurement_results; measurement_setting=setting)

            @test data_with_setting.N == N
            @test data_with_setting.NM == NM
            @test data_with_setting.measurement_results == measurement_results
            @test data_with_setting.measurement_setting === setting

        # Export and then import the data with setting
        tmp_dir = mktempdir()
        tmp_file = joinpath(tmp_dir, "data_with_setting.npz")
        export_MeasurementData(data_with_setting, tmp_file)
        imported_data = import_MeasurementData(tmp_file; predefined_setting=setting)

        @test imported_data.N == data_with_setting.N
        @test imported_data.NM == data_with_setting.NM
        @test imported_data.measurement_results == data_with_setting.measurement_results

        @show imported_data.measurement_setting

        if imported_data.measurement_setting !== nothing
            @test imported_data.measurement_setting.site_indices == setting.site_indices
        else
            @test false
        end

        rm(tmp_dir, recursive=true)

        # Test 2: MeasurementData without measurement setting
            data_without_setting = MeasurementData(measurement_results)
            @test data_without_setting.N == N
            @test data_without_setting.NM == NM
            @test data_without_setting.measurement_results == measurement_results
            @test data_without_setting.measurement_setting === nothing

        # Export and import the data without setting
        tmp_dir2 = mktempdir()
        tmp_file2 = joinpath(tmp_dir2, "data_without_setting.npz")
        export_MeasurementData(data_without_setting, tmp_file2)
        imported_data2 = import_MeasurementData(tmp_file2)

        @test imported_data2.N == data_without_setting.N
        @test imported_data2.NM == data_without_setting.NM
        @test imported_data2.measurement_results == data_without_setting.measurement_results
        @test imported_data2.measurement_setting === nothing

        rm(tmp_dir2, recursive=true)
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
