using RandomMeas
using Test

# Test file for MeasurementData.jl
@testset "MeasurementGroup Tests" begin
    # Mock data setup
    N = 4  # Number of sites
    NM = 10  # Number of measurements per setting
    ξ = siteinds("Qubit", N)  # Site indices
    measurement_results = rand(1:2, NM, N)
    measurement_setting = LocalUnitaryMeasurementSetting(N;site_indices=ξ, ensemble="Haar")
    data1 = MeasurementData(measurement_results; measurement_setting=measurement_setting)
    measurement_results = rand(1:2, NM, N)
    measurement_setting = LocalUnitaryMeasurementSetting(N;site_indices=ξ, ensemble="Haar")
    data2 = MeasurementData(measurement_results; measurement_setting=measurement_setting)
    measurements = [data1,data2]



    # Test 1: Creating MeasurementGroup with setting
    @testset "With Measurement Setting" begin
        group = MeasurementGroup(measurements)

        @test group.N == N
        @test group.NU == 2
        @test group.NM == NM
        @test group.measurements[1] == data1
    end

     # Test 2: Creating MeasurementGroup from an MPS dense mode
     @testset "Creating MeasurementGroup from an MPS dense mode" begin
        NU = 10
        ψ = random_mps(ξ; linkdims=3);
        group = MeasurementGroup(ψ,NU,NM;mode="dense")
        @test group.N == N
        @test group.NU == NU
        @test group.NM == NM
    end

    # Test 3: Creating MeasurementGroup from an MPS | MPS/MPO mode
         @testset "Creating MeasurementGroup from an MPS | MPS/MPO mode" begin
            NU = 10
            ψ = random_mps(ξ; linkdims=3);
            group = MeasurementGroup(ψ,NU,NM;mode="MPS/MPO")
            @test group.N == N
            @test group.NU == NU
            @test group.NM == NM
    end

    # Test 4: Creating MeasurementGroup from an MPO | dense mode
        @testset "Creating MeasurementGroup from an MPO | dense mode" begin
            NU = 10
            ψ = random_mps(ξ; linkdims=3);
            ρ  = outer(ψ',ψ)
            group = MeasurementGroup(ρ,NU,NM;mode="dense")
            @test group.N == N
            @test group.NU == NU
            @test group.NM == NM
    end

       # Test5 : Creating MeasurementGroup from an MPO | MPS/MPO mode
       @testset "Creating MeasurementGroup from an MPO | MPS/MPO mode" begin
        NU = 10
        ψ = random_mps(ξ; linkdims=3);
        ρ  = outer(ψ',ψ)
        group = MeasurementGroup(ρ,NU,NM;mode="MPS/MPO")
        @test group.N == N
        @test group.NU == NU
        @test group.NM == NM
end


     # Test 6: reduce_to_subsystem
     @testset "With Measurement Setting" begin
        group = MeasurementGroup(measurements)
        reduced_group = reduce_to_subsystem(group,collect(1:2))

        @test reduced_group.N == 2
        @test reduced_group.NU == 2
        @test reduced_group.NM == NM
        #@test reduced_group.measurements[1] == reduce_to_subsystem(data1,collect(1:2))
    end

    @testset "MeasurementGroup Import/Export Tests" begin
        # Setup parameters
        N = 4       # Number of sites (qubits)
        NM = 10     # Number of measurements per setting
        NU = 3      # Number of MeasurementData objects in the group

        # Generate default site indices
        ξ = siteinds("Qubit", N)

        # Create a vector of MeasurementData objects with individual measurement settings.
        # Each measurement setting is assumed to be either a LocalUnitaryMeasurementSetting
        # or a ComputationalBasisMeasurementSetting. Here, we use LocalUnitaryMeasurementSetting.
        measurements = Vector{MeasurementData{LocalUnitaryMeasurementSetting}}(undef, NU)
        predefined_settings = Vector{LocalUnitaryMeasurementSetting}(undef, NU)
        for i in 1:NU
            results = rand(1:2, NM, N)  # Generate random measurement results (binary values: 1 or 2)
            setting = LocalUnitaryMeasurementSetting(N; site_indices=ξ, ensemble="Haar")
            measurements[i] = MeasurementData(results; measurement_setting=setting)
            predefined_settings[i] = setting
        end

        # Construct the original MeasurementGroup from the measurements vector.
        group_original = MeasurementGroup(measurements)

        # Export the MeasurementGroup to a temporary NPZ file.
        tmp_dir = mktempdir()
        tmp_file = joinpath(tmp_dir, "data_with_setting.npz")

        export_MeasurementGroup(group_original, tmp_file)

        # Import the MeasurementGroup using the vector of predefined settings.
        group_imported = import_MeasurementGroup(tmp_file; predefined_settings=predefined_settings, site_indices=ξ)

        # Verify that the imported group matches the original.
        @test group_imported.N == group_original.N
        @test group_imported.NM == group_original.NM
        @test group_imported.NU == group_original.NU
        @test group_imported.measurements[1].measurement_results == group_original.measurements[1].measurement_results

        # If measurement settings are present, check that the imported settings' local unitaries match.
        if group_imported.measurements[1].measurement_setting !== nothing
            imported_setting = group_imported.measurements[1].measurement_setting
            original_setting = group_original.measurements[1].measurement_setting
            for j in 1:N
                A_imported = Array(imported_setting.local_unitary[j], imported_setting.site_indices[j]', imported_setting.site_indices[j])
                A_original = Array(original_setting.local_unitary[j], original_setting.site_indices[j]', original_setting.site_indices[j])
                @test isapprox(A_imported, A_original, atol=1e-10)
            end
        else
            @test false
        end

        rm(tmp_dir, recursive=true)

        # Test the case where measurement_setting is nothing.
        measurements_no_setting = Vector{MeasurementData{Nothing}}(undef, NU)
        for i in 1:NU
            results = rand(1:2, NM, N)
            measurements_no_setting[i] = MeasurementData(results)  # measurement_setting defaults to nothing
        end
        group_no_setting = MeasurementGroup(measurements_no_setting)

        tmp_dir = mktempdir()
        tmp_file = joinpath(tmp_dir, "data_with_setting2.npz")

        export_MeasurementGroup(group_no_setting, tmp_file)
        group_no_setting_imported = import_MeasurementGroup(tmp_file)

        @test group_no_setting_imported.N == group_no_setting.N
        @test group_no_setting_imported.NM == group_no_setting.NM
        @test group_no_setting_imported.NU == group_no_setting.NU
        @test group_no_setting_imported.measurements[1].measurement_setting === nothing

        rm(tmp_dir, recursive=true)
    end


  end
