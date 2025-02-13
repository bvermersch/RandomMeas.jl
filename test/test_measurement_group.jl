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



    # Test 1: Creating MeasurementData with setting
    @testset "With Measurement Setting" begin
        group = MeasurementGroup(measurements)

        @test group.N == N
        @test group.NU == 2
        @test group.NM == NM
        @test group.measurements[1] == data1
    end

  end
