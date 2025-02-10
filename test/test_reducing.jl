using Test
using ITensors

# Include necessary files for the Shadows module
include("../src/MeasurementSettings.jl")
include("../src/MeasurementData.jl")
include("../src/AbstractShadows.jl")
include("../src/DenseShadows.jl")
include("../src/FactorizedShadows.jl")
include("../src/Postprocessing.jl")

@testset "Partial Trace and Consistency Tests" begin
    # Parameters
    N = 6  # Total number of qubits
    NU = 5  # Number of random unitaries
    NM = 50  # Number of measurements per unitary
    subsystem = [1, 3, 5]  # Subsystem to retain
    site_indices = siteinds("Qubit", N)

    # Generate random pure state and measurement data
    measurement_settings = LocalUnitaryMeasurementSettings(N, NU; site_indices=site_indices)
    measurement_results = rand(1:2, NU, NM, N)  # Random binary measurement results
    measurement_data = MeasurementData(measurement_results; measurement_settings=measurement_settings)

    @testset "Direct Shadow Reduction" begin
        # Generate shadows
        dense_shadows = get_dense_shadows(measurement_data)
        factorized_shadows = get_factorized_shadows(measurement_data)

        # Reduce shadows directly
        dense_reduced_shadows = [
            partial_trace(shadow, subsystem) for shadow in dense_shadows
        ]
        factorized_reduced_shadows = [
            partial_trace(shadow, subsystem; assume_unit_trace=false) for shadow in factorized_shadows
        ]

        # Ensure all reduced shadows have correct subsystem size
        @test all([shadow.N == length(subsystem) for shadow in dense_reduced_shadows])
        @test all([shadow.N == length(subsystem) for shadow in factorized_reduced_shadows])
    end

    @testset "Measurement Data Reduction" begin
        # Reduce measurement data
        reduced_measurement_data = reduce_to_subsystem(measurement_data, subsystem)

        # Ensure dimensions are correct
        @test reduced_measurement_data.N == length(subsystem)
        @test size(reduced_measurement_data.measurement_results) == (NU, NM, length(subsystem))
        @test reduced_measurement_data.measurement_settings.N == length(subsystem)
        @test reduced_measurement_data.measurement_settings.site_indices == site_indices[subsystem]

        # Generate shadows from reduced data
        dense_shadows_reduced_data = get_dense_shadows(reduced_measurement_data)
        factorized_shadows_reduced_data = get_factorized_shadows(reduced_measurement_data)

        # Ensure all shadows from reduced data have correct subsystem size
        @test all([shadow.N == length(subsystem) for shadow in dense_shadows_reduced_data])
        @test all([shadow.N == length(subsystem) for shadow in factorized_shadows_reduced_data])
    end

    @testset "Consistency Between Reduction Methods" begin
        # Generate shadows and reduce both directly and from reduced data
        dense_shadows = get_dense_shadows(measurement_data)
        factorized_shadows = get_factorized_shadows(measurement_data)

        dense_reduced_shadows = partial_trace(dense_shadows, subsystem)
        factorized_reduced_shadows = partial_trace(factorized_shadows, subsystem; assume_unit_trace=false)

        reduced_measurement_data = reduce_to_subsystem(measurement_data, subsystem)
        dense_shadows_reduced_data = get_dense_shadows(reduced_measurement_data)
        factorized_shadows_reduced_data = get_factorized_shadows(reduced_measurement_data)

        @test size(dense_reduced_shadows) == size(dense_shadows_reduced_data)
        @test size(factorized_reduced_shadows) == size(factorized_shadows_reduced_data)


        for idx in eachindex(dense_shadows)
            # Compare Dense Shadows
            dense_direct = dense_reduced_shadows[idx]
            dense_from_data = dense_shadows_reduced_data[idx]
            @test isapprox(dense_direct.shadow_data, dense_from_data.shadow_data, atol=1e-10)
        end

        for idx in eachindex(factorized_shadows)
            # Compare Factorized Shadows
            factorized_direct = factorized_reduced_shadows[idx]
            factorized_from_data = factorized_shadows_reduced_data[idx]
            @test all(isapprox.(
                factorized_direct.shadow_data,
                factorized_from_data.shadow_data,
                atol=1e-10
            ))
        end

    end
end
