using RandomMeas
using Test
using Random
using LinearAlgebra

@testset "MeasurementSetting Tests" begin
    # Define the number of sites (qubits)
    N = 4

    # Test 1: Creating a Measurement Setting with the default ("Haar") ensemble
    @testset "Test 1: Create LocalUnitaryMeasurementSetting (Haar)" begin
        setting = LocalUnitaryMeasurementSetting(N)
        @test setting.N == N
        @test length(setting.site_indices) == N
        @test length(setting.basis_transformation) == N
        for j in 1:N
            @test isa(setting.basis_transformation[j], ITensor)
            # Ensure each ITensor has two indices (the bra and ket for that site)
            inds_j = inds(setting.basis_transformation[j])
            @test length(inds_j) == 2
        end
    end

    # Test 2: Creating settings with different ensembles ("Haar", "Pauli", "Identity")
    @testset "Test 2: Different Ensembles" begin
        ensembles = [Haar, Pauli, Identity]
        for ensemble in ensembles
            setting = LocalUnitaryMeasurementSetting(N; ensemble=ensemble)
            @test setting.N == N
            @test length(setting.site_indices) == N
            @test length(setting.basis_transformation) == N
            for j in 1:N
                @test isa(setting.basis_transformation[j], ITensor)
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
            LocalUnitaryMeasurementSetting(N, valid_setting.basis_transformation, bad_site_indices)
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
        @test length(reduced_setting.basis_transformation) == length(subsystem)
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
        @test length(setting_from_array.basis_transformation) == N
        # Optionally check that each ITensor matches the identity matrix (within numerical precision)
        for n in 1:N
            # Convert the ITensor to an Array; note that the indices might appear in a different order,
            # so we use isapprox with a tolerance.
            A = Array(setting_from_array.basis_transformation[n], setting_from_array.site_indices[n]', setting_from_array.site_indices[n])
            @test isapprox(A, [1 0; 0 1], atol=1e-10)
        end
    end

    @testset "Test 6: Assertion Checks for LocalUnitaryMeasurementSetting" begin
        # Assume N = 2 for simplicity
        N = 2

        # Generate valid site indices using the helper function siteinds (make sure siteinds is defined and imported)
        valid_site_indices = siteinds("Qubit", N)

        # Create a valid basis_transformation vector using get_rotation (this should produce ITensors with the correct indices)
        valid_basis_transformation = [get_rotation(valid_site_indices[i], Haar) for i in 1:N]

        # -- Valid Case --
        @testset "Valid ITensors" begin
            # This should pass without errors.
            setting = LocalUnitaryMeasurementSetting(N, valid_basis_transformation, valid_site_indices)
            @test setting.N == N
            @test length(setting.basis_transformation) == N
        end

        # -- Test 1: ITensor with the wrong number of indices --
        @testset "Invalid number of indices" begin
                    # Create an ITensor with only one index
        bad_itensor = ITensor(valid_site_indices[1])
        bad_basis_transformation = copy(valid_basis_transformation)
        bad_basis_transformation[1] = bad_itensor
        @test_throws AssertionError begin
            LocalUnitaryMeasurementSetting(N, bad_basis_transformation, valid_site_indices)
        end
        end

        # -- Test 2: ITensor that does not contain the required unprimed and primed indices --
        @testset "Invalid indices in ITensor" begin
                    # Create a dummy index that is different from valid_site_indices[1]
        wrong_index = Index(2, "Wrong")
        # Construct an ITensor with wrong_index and its primed version
        it_wrong = ITensor(wrong_index, prime(wrong_index))
        bad_basis_transformation = copy(valid_basis_transformation)
        bad_basis_transformation[1] = it_wrong
        @test_throws AssertionError begin
            LocalUnitaryMeasurementSetting(N, bad_basis_transformation, valid_site_indices)
        end
        end

        # -- Test 3: ITensor with indices in reversed order (should be acceptable) --
        @testset "Reversed order indices" begin
                    # Manually construct an ITensor with reversed indices order:
        # It must contain both valid_site_indices[1] and prime(valid_site_indices[1]), regardless of order.
        it_reversed = ITensor(prime(valid_site_indices[1]), valid_site_indices[1])
        good_basis_transformation = copy(valid_basis_transformation)
        good_basis_transformation[1] = it_reversed
        # This construction should pass.
        setting = LocalUnitaryMeasurementSetting(N, good_basis_transformation, valid_site_indices)
            @test setting.N == N
        end
    end

    @testset "ComputationalBasisMeasurementSetting Tests" begin
        # Define number of sites (qubits)
        N = 3

        # Generate site indices using the helper function (assumes siteinds is defined)
        site_indices = siteinds("Qubit", N)

        # Create a ComputationalBasisMeasurementSetting object
        comp_setting = ComputationalBasisMeasurementSetting(N, site_indices)

        # Check that the number of sites is correct and the vectors have length N
        @test comp_setting.N == N
        @test length(comp_setting.site_indices) == N
        @test length(comp_setting.basis_transformation) == N

    end

    @testset "Test 7: Import/Export Unitary" begin
        # Create a local unitary measurement setting
        setting = LocalUnitaryMeasurementSetting(N; ensemble=Haar)

        # Create a temporary file for export.
        tmp_dir = mktempdir()
        tmp_file = joinpath(tmp_dir, "tempfile.npz")

        # Export the setting to the temporary file.
        export_LocalUnitaryMeasurementSetting(setting, tmp_file)

        # Import the setting from the temporary file.
        imported_setting = import_LocalUnitaryMeasurementSetting(tmp_file; site_indices=setting.site_indices)

        # Check that the imported setting has the same dimensions.
        @test imported_setting.N == setting.N
        @test length(imported_setting.site_indices) == N
        @test length(imported_setting.basis_transformation) == N

        # For each site, check that the exported and then imported ITensor (converted to an Array)
        # matches the expected identity matrix.
        for i in 1:N
            # Convert both the original and imported ITensors to arrays using the same indices.
            original_array = Array(setting.basis_transformation[i], setting.site_indices[i]', setting.site_indices[i])
            imported_array = Array(imported_setting.basis_transformation[i], imported_setting.site_indices[i]', imported_setting.site_indices[i])
            @test isapprox(original_array, imported_array, atol=1e-10)
        end

        # Clean up the temporary file.
        rm(tmp_dir, recursive=true)
    end

    @testset "Test 8: OpenQASM Export" begin
        Nq = 3
        ξq = siteinds("Qubit", Nq)

        # Local unitary setting with identity gates.
        unitary_array = zeros(ComplexF64, Nq, 2, 2)
        for n in 1:Nq
            unitary_array[n, :, :] = [1 0; 0 1]
        end
        local_setting = LocalUnitaryMeasurementSetting(unitary_array; site_indices=ξq)

        qasm_local = to_OpenQASM(local_setting)
        @test occursin("OPENQASM 2.0;", qasm_local)
        @test occursin("include \"qelib1.inc\";", qasm_local)
        @test occursin("qreg q[$Nq];", qasm_local)
        @test occursin("creg c[$Nq];", qasm_local)
        @test count(line -> occursin("u3(", line), split(qasm_local, '\n')) == Nq
        @test count(line -> occursin("measure q[", line), split(qasm_local, '\n')) == Nq

        # Computational basis emits no unitary gates.
        comp_setting = ComputationalBasisMeasurementSetting(Nq; site_indices=ξq)
        qasm_comp = to_OpenQASM(comp_setting)
        @test occursin("OPENQASM 2.0;", qasm_comp)
        @test count(line -> occursin("u3(", line), split(qasm_comp, '\n')) == 0
        @test count(line -> occursin("measure q[", line), split(qasm_comp, '\n')) == Nq

        # Optional omission of measurements.
        qasm_no_meas = to_OpenQASM(local_setting; include_measurements=false)
        @test !occursin("measure q[", qasm_no_meas)
        @test !occursin("creg c[", qasm_no_meas)

        # File export
        tmp_dir = mktempdir()
        tmp_file = joinpath(tmp_dir, "setting.qasm")
        export_OpenQASM(local_setting, tmp_file)
        @test isfile(tmp_file)
        @test read(tmp_file, String) == qasm_local
        rm(tmp_dir, recursive=true)
    end

    @testset "Test 9: u3 Angle Reconstruction" begin
        # OpenQASM u3 gate matrix convention.
        u3_matrix(θ, ϕ, λ) = ComplexF64[
            cos(θ / 2) -exp(1im * λ) * sin(θ / 2);
            exp(1im * ϕ) * sin(θ / 2) exp(1im * (ϕ + λ)) * cos(θ / 2)
        ]

        # Compare up to an overall global phase.
        function phase_invariant_error(U::AbstractMatrix{<:Complex}, V::AbstractMatrix{<:Complex})
            α = angle(sum(conj.(V) .* U))
            return norm(U - exp(1im * α) * V)
        end

        fixed_cases = [
            ("I", ComplexF64[1 0; 0 1]),
            ("X", ComplexF64[0 1; 1 0]),
            ("Y", ComplexF64[0 -im; im 0]),
            ("Z", ComplexF64[1 0; 0 -1]),
            ("H", (1 / sqrt(2)) * ComplexF64[1 1; 1 -1]),
            ("S", ComplexF64[1 0; 0 im]),
            ("T", ComplexF64[1 0; 0 exp(1im * π / 4)]),
        ]

        for (_, U) in fixed_cases
            θ, ϕ, λ = RandomMeas._u3_angles_from_unitary(U)
            U_reconstructed = u3_matrix(θ, ϕ, λ)
            @test phase_invariant_error(U, U_reconstructed) ≤ 1e-12
        end

        # Haar-random single-qubit unitaries using the standard QR construction.
        Random.seed!(7)
        n_haar_samples = 500
        for _ in 1:n_haar_samples
            A = randn(ComplexF64, 2, 2)
            F = qr(A)
            Q = Matrix(F.Q)
            R = Matrix(F.R)
            d = diag(R)
            phases = map(x -> iszero(x) ? (1.0 + 0im) : x / abs(x), d)
            U = Q * Diagonal(phases)

            θ, ϕ, λ = RandomMeas._u3_angles_from_unitary(U)
            U_reconstructed = u3_matrix(θ, ϕ, λ)
            @test phase_invariant_error(U, U_reconstructed) ≤ 1e-12
        end
    end

end



println("All MeasurementSetting tests completed successfully!")
