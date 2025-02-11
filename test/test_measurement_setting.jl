using Test
using ITensors
using NPZ

# Include the MeasurementSettings module
include("../src/MeasurementSetting.jl")

@testset "MeasurementSetting Tests" begin

    N = 4   # Number of sites

    # Generate measurement settings with Haar unitaries
    measurement_setting = LocalUnitaryMeasurementSetting(N)

    @testset "Test 1: Creating Measurement Setting" begin
        @test measurement_setting.N == N
        @test length(measurement_setting.site_indices) == N

        # Validate dimensions of local unitaries
        @test size(measurement_setting.local_unitary, 1) == N
    end

    # @testset "Test 2: Exporting and Importing Unitaries" begin
    #     export_filepath = "test_unitaries.npy"

    #     # Export unitaries
    #     export_unitaries(measurement_settings, export_filepath)
    #     @test isfile(export_filepath)

    #     # Import unitaries
    #     imported_settings = import_unitaries(export_filepath)
    #     @test imported_settings.N == measurement_settings.N
    #     @test imported_settings.NU == measurement_settings.NU

    #     # Validate individual unitaries
    #     for r in 1:NU
    #         for n in 1:N
    #             original_unitary = Array(measurement_settings.local_unitaries[r, n], measurement_settings.site_indices[n]', measurement_settings.site_indices[n])
    #             imported_unitary = Array(imported_settings.local_unitaries[r, n], imported_settings.site_indices[n]', imported_settings.site_indices[n])
    #             @test isapprox(original_unitary, imported_unitary, atol=1e-10)
    #         end
    #     end

    #     # Cleanup
    #     rm(export_filepath, force=true)
    # end

    # @testset "Test 3: Importing Unitaries with Predefined Indices" begin
    #     # Export unitaries
    #     export_filepath = "test_unitaries.npy"
    #     export_unitaries(measurement_settings, export_filepath)

    #     # Define custom site indices
    #     custom_site_indices = siteinds("Qubit", N)

    #     # Import unitaries with predefined indices
    #     imported_with_indices = import_unitaries(export_filepath, site_indices=custom_site_indices)
    #     @test imported_with_indices.site_indices == custom_site_indices

    #     # Validate individual unitaries
    #     for r in 1:NU
    #         for n in 1:N
    #             original_unitary = Array(measurement_settings.local_unitaries[r, n], measurement_settings.site_indices[n]', measurement_settings.site_indices[n])
    #             imported_unitary = Array(imported_with_indices.local_unitaries[r, n], custom_site_indices[n]', custom_site_indices[n])
    #             @test isapprox(original_unitary, imported_unitary, atol=1e-10)
    #         end
    #     end

    #     # Cleanup
    #     rm(export_filepath, force=true)
    # end

    # @testset "Test 4: Edge Cases" begin
    #     # Recreate the required `measurement_settings` for this test
    #     #measurement_settings = LocalUnitaryMeasurementSettings(N, NU, ensemble="Haar")

    #     @test_throws AssertionError begin
    #         invalid_unitaries = Array{ITensor, 2}(undef, NU, N + 1)  # Invalid size
    #         LocalUnitaryMeasurementSettings(N, NU, invalid_unitaries, measurement_settings.site_indices)
    #     end

    #     # Test missing site indices (valid case)
    #     export_filepath = "test_unitaries.npy"
    #     export_unitaries(measurement_settings, export_filepath)
    #     imported_with_generated_indices = import_unitaries(export_filepath, site_indices=nothing)
    #     @test length(imported_with_generated_indices.site_indices) == N
    #     rm(export_filepath, force=true)
    # end

    @testset "Test 5: Different Ensembles" begin
        ensembles = ["Haar", "Pauli", "Identity"]
        for ensemble in ensembles
            setting = LocalUnitaryMeasurementSetting(N; ensemble=ensemble)
            @test setting.N == N
            @test length(setting.site_indices) == N
        end
    end

end

println("All tests completed successfully!")
