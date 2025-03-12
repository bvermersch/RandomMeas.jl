using RandomMeas
using Test

# Set up test parameters
N = 2  # Number of qubits
NU = 100  # Number of unitaries
NM = 100  # Number of measurements per unitary
ξ = siteinds("Qubit", N)
ψ1 = random_mps(ξ)
ψ2 = random_mps(ξ)
observable = outer(ψ2', ψ2)
exact_value = inner(ψ1',observable,ψ1)
measurements = Vector{MeasurementData{LocalUnitaryMeasurementSetting}}(undef,NU)
for r in 1:NU
    measurement_setting_ = LocalUnitaryMeasurementSetting(N; site_indices=ξ,ensemble="Haar")
    measurements[r] = MeasurementData(ψ1,NM,measurement_setting_;mode="dense")
end
measurement_group = MeasurementGroup(measurements)


@testset "Shadows Tests" begin
    
    
    batch_shadow = get_dense_shadows(measurement_group;number_of_ru_batches=1)
    expect_batch = get_expect_shadow(observable,batch_shadow)

    expect_dense = 0.
    for r in 1:NU
        measurement_data = measurement_group.measurements[r]
        dense_shadow = DenseShadow(measurement_data)
        expect_dense += get_expect_shadow(observable,dense_shadow)/NU
    end

    expect_factorized = 0.
    for r in 1:NU
        measurement_data = measurement_group.measurements[r]
        factorized_shadows = get_factorized_shadows(measurement_data)
        expect_factorized += get_expect_shadow(observable,factorized_shadows)/NU 
    end
   
    # Display results
    @show exact_value
    @show expect_batch
    @show expect_dense
    @show expect_factorized

    # Tests
    @test isapprox(expect_dense, expect_batch, atol=1e-10)
    @test isapprox(expect_factorized, expect_batch, atol=1e-10)
end
