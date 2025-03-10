using RandomMeas
using Test

# Test parameters
N = 4         # Number of qubits
NU = 5        # Number of unitaries
NM = 6       # Number of measurements per unitary
k_values = 1:4  # Trace moments to test
ξ = siteinds("Qubit", N)

# Generate measurement settings and data
ψ1 = random_mps(ξ)
measurements = Vector{MeasurementData{LocalUnitaryMeasurementSetting}}(undef,NU)
for r in 1:NU
    measurement_setting_r = LocalUnitaryMeasurementSetting(N; site_indices=ξ,ensemble="Haar")
    measurements[r] = MeasurementData(ψ1,NM,measurement_setting_r;mode="dense")
end
measurement_group = MeasurementGroup(measurements)

@testset "Trace Moments Tests" begin
    # Compute dense shadows
    dense_shadows_4_batches = get_dense_shadows(measurement_group, number_of_ru_batches=4)
    dense_shadows_nu_batches = get_dense_shadows(measurement_group, number_of_ru_batches=NU)
    # Compute factorized shadows
    factorized_shadows = get_factorized_shadows(measurement_group)

    # Test for each moment k
    for k in k_values
        @testset "Trace Moment k=$k" begin
            # Compute trace moments
            @time dense_moments_4_batches = get_trace_moment(dense_shadows_4_batches, k)
            @time dense_moments_nu_batches = get_trace_moment(dense_shadows_nu_batches, k)
            @time factorized_moments = get_trace_moment(factorized_shadows, k)

            # Display results for manual inspection
            @show dense_moments_4_batches
            @show dense_moments_nu_batches
            @show factorized_moments

            # Compare dense and factorized results
            @test isapprox(dense_moments_nu_batches, factorized_moments; atol=1e-10)
        end
    end
end
