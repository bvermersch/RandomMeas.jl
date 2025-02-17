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
     @testset "With Measurement Setting" begin
        NU = 10
        ψ = random_mps(ξ; linkdims=3);
        group = MeasurementGroup(ψ,NU,NM;mode="dense")
        @test group.N == N
        @test group.NU == NU
        @test group.NM == NM
    end

    # Test 3: Creating MeasurementGroup from an MPS | MPS/MPO mode
         @testset "With Measurement Setting" begin
            NU = 10
            ψ = random_mps(ξ; linkdims=3);
            group = MeasurementGroup(ψ,NU,NM;mode="MPS/MPO")
            @test group.N == N
            @test group.NU == NU
            @test group.NM == NM
    end

     # Test 3: reduce_to_subsystem
     @testset "With Measurement Setting" begin
        group = MeasurementGroup(measurements)
        reduced_group = reduce_to_subsystem(group,collect(1:2))

        @test reduced_group.N == 2
        @test reduced_group.NU == 2
        @test reduced_group.NM == NM      
        #@test reduced_group.measurements[1] == reduce_to_subsystem(data1,collect(1:2))
    end

  end
