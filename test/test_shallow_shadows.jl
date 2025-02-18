using RandomMeas
using Test

@testset "ShallowShadow Tests" begin
    # Define parameters
    N = 4  # Number of qubits/sites
    NM = 10 # Number of projective measurements
    site_indices = siteinds("Qubit", N)  # Site indices
    depth = 2 #depth of the shallow random circuit
    ψ = random_mps(site_indices)


    # Generate random local unitary measurement setting
    @testset "ShallowMeasurementSetting" begin
        measurement_setting = ShallowUnitaryMeasurementSetting(N,depth;site_indices=site_indices)
        measurement_data_mps = MeasurementData(ψ,NM,measurement_setting;mode="MPS/MPO")
        measurement_data_mpo = MeasurementData(outer(ψ',ψ),NM,measurement_setting;mode="MPS/MPO")
        measurement_data_dense = MeasurementData(ψ,NM,measurement_setting;mode="dense")
    end

    @testset "Shallow MeasurementGroup" begin
        NU = 10
        depth = 2
        measurement_group_mps = MeasurementGroup(ψ,NU,NM,depth;mode="MPS/MPO")
        measurement_group_mpo = MeasurementGroup(outer(ψ',ψ),NU,NM,depth;mode="MPS/MPO")
        measurement_group_dense = MeasurementGroup(ψ,NU,NM,depth;mode="dense")
    
        depolarization_vectors = get_depolarization_vectors(measurement_group_mps::MeasurementGroup{ShallowUnitaryMeasurementSetting})
    end
end
