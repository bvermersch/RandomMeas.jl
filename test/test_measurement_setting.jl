using Test
using ITensors,ITensorMPS
using NPZ

# Include the MeasurementSetting module (adjust the relative path if needed)
include("../src/Structures.jl")
include("../src/MeasurementSetting.jl")

@testset "MeasurementSetting Tests" begin
    # Define the number of sites (qubits)
    N = 4

    # Test 1: Creating a Measurement Setting with the default ("Haar") ensemble
    @testset "Test 1: Create LocalUnitaryMeasurementSetting (Haar)" begin
        setting = LocalUnitaryMeasurementSetting(N)
        @test setting.N == N
        @test length(setting.site_indices) == N
        @test length(setting.local_unitary) == N
        for j in 1:N
            @test isa(setting.local_unitary[j], ITensor)
            # Ensure each ITensor has two indices (the bra and ket for that site)
            inds_j = inds(setting.local_unitary[j])
            @test length(inds_j) == 2
        end
    end

    # Test 2: Creating settings with different ensembles ("Haar", "Pauli", "Identity")
    @testset "Test 2: Different Ensembles" begin
        ensembles = ["Haar", "Pauli", "Identity"]
        for ensemble in ensembles
            setting = LocalUnitaryMeasurementSetting(N; ensemble=ensemble)
            @test setting.N == N
            @test length(setting.site_indices) == N
            @test length(setting.local_unitary) == N
            for j in 1:N
                @test isa(setting.local_unitary[j], ITensor)
            end
        end
    end

    # Test 3: Error Handling with Invalid Site Indices Length
    @testset "Test 3: Error Handling for Invalid Site Indices" begin
        # Create a valid setting first
        valid_setting = LocalUnitaryMeasurementSetting(N)
        # Remove one site index (simulate invalid input)
        bad_site_indices = valid_setting.site_indices[1:end-1]
        @test_throws AssertionError begin
            LocalUnitaryMeasurementSetting(N, valid_setting.local_unitary, bad_site_indices)
        end
    end

    # Test 4: Reducing a Measurement Setting to a Subsystem
    @testset "Test 4: Reduce to Subsystem" begin
        setting = LocalUnitaryMeasurementSetting(N)
        # Choose a subsystem (e.g., the first two sites)
        subsystem = [1, 2]
        reduced_setting = reduce_to_subsystem(setting, subsystem)
        @test reduced_setting.N == length(subsystem)
        @test length(reduced_setting.site_indices) == length(subsystem)
        @test length(reduced_setting.local_unitary) == length(subsystem)
    end

    # Test 5: Constructing a Setting from a Unitary Array
    @testset "Test 5: Create from Unitary Array" begin
        # Create an N×2×2 array where each 2×2 slice is the identity matrix.
        unitary_array = zeros(ComplexF64, N, 2, 2)
        for n in 1:N
            unitary_array[n, :, :] = [1 0; 0 1]
        end
        # Let the constructor generate site indices automatically
        setting_from_array = LocalUnitaryMeasurementSetting(unitary_array; site_indices=nothing)
        @test setting_from_array.N == N
        @test length(setting_from_array.site_indices) == N
        @test length(setting_from_array.local_unitary) == N
        # Optionally check that each ITensor matches the identity matrix (within numerical precision)
        for n in 1:N
            # Convert the ITensor to an Array; note that the indices might appear in a different order,
            # so we use isapprox with a tolerance.
            A = Array(setting_from_array.local_unitary[n], setting_from_array.site_indices[n]', setting_from_array.site_indices[n])
            @test isapprox(A, [1 0; 0 1], atol=1e-10)
        end
    end

    @testset "Test 6: Assertion Checks for LocalUnitaryMeasurementSetting" begin
        # Assume N = 2 for simplicity
        N = 2

        # Generate valid site indices using the helper function siteinds (make sure siteinds is defined and imported)
        valid_site_indices = siteinds("Qubit", N)

        # Create a valid local_unitary vector using get_rotation (this should produce ITensors with the correct indices)
        valid_local_unitary = [get_rotation(valid_site_indices[i], "Haar") for i in 1:N]

        # -- Valid Case --
        @testset "Valid ITensors" begin
            # This should pass without errors.
            setting = LocalUnitaryMeasurementSetting(N, valid_local_unitary, valid_site_indices)
            @test setting.N == N
            @test length(setting.local_unitary) == N
        end

        # -- Test 1: ITensor with the wrong number of indices --
        @testset "Invalid number of indices" begin
            # Create an ITensor with only one index
            bad_itensor = ITensor(valid_site_indices[1])
            bad_local_unitary = copy(valid_local_unitary)
            bad_local_unitary[1] = bad_itensor
            @test_throws AssertionError begin
                LocalUnitaryMeasurementSetting(N, bad_local_unitary, valid_site_indices)
            end
        end

        # -- Test 2: ITensor that does not contain the required unprimed and primed indices --
        @testset "Invalid indices in ITensor" begin
            # Create a dummy index that is different from valid_site_indices[1]
            wrong_index = Index(2, "Wrong")
            # Construct an ITensor with wrong_index and its primed version
            it_wrong = ITensor(wrong_index, prime(wrong_index))
            bad_local_unitary = copy(valid_local_unitary)
            bad_local_unitary[1] = it_wrong
            @test_throws AssertionError begin
                LocalUnitaryMeasurementSetting(N, bad_local_unitary, valid_site_indices)
            end
        end

        # -- Test 3: ITensor with indices in reversed order (should be acceptable) --
        @testset "Reversed order indices" begin
            # Manually construct an ITensor with reversed indices order:
            # It must contain both valid_site_indices[1] and prime(valid_site_indices[1]), regardless of order.
            it_reversed = ITensor(prime(valid_site_indices[1]), valid_site_indices[1])
            good_local_unitary = copy(valid_local_unitary)
            good_local_unitary[1] = it_reversed
            # This construction should pass.
            setting = LocalUnitaryMeasurementSetting(N, good_local_unitary, valid_site_indices)
            @test setting.N == N
        end
    end
end



println("All MeasurementSetting tests completed successfully!")
