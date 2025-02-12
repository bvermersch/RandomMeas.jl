using Test
using ITensors
using NPZ
include("../src/MeasurementSetting.jl")
include("../src/MeasurementData.jl")
include("../src/MeasurementGroup.jl")

# Test file for MeasurementData.jl
@testset "MeasurementGroup Tests" begin
    # Mock data setup
    N = 4  # Number of sites
    NM = 10  # Number of measurements per setting

    measurement_results = rand(1:2, NM, N)
    measurement_setting = LocalUnitaryMeasurementSetting(N, ensemble="Haar")
    data1 = MeasurementData(measurement_results; measurement_setting=measurement_setting)
    measurement_results = rand(1:2, NM, N)
    measurement_setting = LocalUnitaryMeasurementSetting(N, ensemble="Haar")
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
