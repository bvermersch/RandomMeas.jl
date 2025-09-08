using RandomMeas
using Test

# Test parameters
N = 4  # Number of qubits/sites
NM = 5  # Number of projective measurements

# Generate measurement settings
measurement_setting = LocalUnitaryMeasurementSetting(N, ensemble=Haar)

# Generate random measurement results
measurement_results = rand(1:2, NM, N)  # Random binary results in the range [1, 2]

# Create a MeasurementData object
measurement_data = MeasurementData(measurement_results; measurement_setting=measurement_setting)

# Testing FactorizedShadow constructor
@testset "FactorizedShadow Tests" begin
        for m in 1:NM
            local_unitary = measurement_setting.basis_transformation
            data = measurement_results[m, :]

            # Construct FactorizedShadow
            shadow = FactorizedShadow(data, local_unitary)
            @test shadow.N == N
            @test length(shadow.shadow_data) == N
        end
end

# Testing factorized shadow generation
@testset "Batch FactorizedShadow Tests" begin
    G = fill(1.0, N)  # No error correction
    shadows = get_factorized_shadows(measurement_data; G=G)

    @test length(shadows) == NM

        for m in 1:NM
            shadow = shadows[m]
            @test shadow.N == N
            @test length(shadow.shadow_data) == N
        end
end

# Test with non-uniform G
@testset "FactorizedShadow with G" begin
    G = [1.0, 0.9, 1.1, 1.2]  # Example of non-uniform G values
    local_unitary = measurement_setting.basis_transformation
        for m in 1:NM
            data = measurement_results[m, :]

            # Construct FactorizedShadow
            shadow = FactorizedShadow(data, local_unitary; G=G)
            @test shadow.N == N
            @test length(shadow.shadow_data) == N
        end
end

# Test edge cases
@testset "FactorizedShadow Edge Cases" begin
    # Invalid G length
    @test_throws AssertionError FactorizedShadow(measurement_results[1, :], measurement_setting.basis_transformation; G=[1.0, 0.9])

    # Invalid measurement results length
    @test_throws AssertionError FactorizedShadow([1, 2], measurement_setting.basis_transformation)
end



println("All tests passed!")
