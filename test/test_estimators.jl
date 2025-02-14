using RandomMeas
using Test

# Test parameters
N = 2      # Number of qubits
NU = 200        # Number of unitaries
NM = 100     # Number of measurements per unitary
ξ = siteinds("Qubit", N)

# Generate measurement settings and data
ψ1 = random_mps(ξ)
ψ2 = random_mps(ξ)
measurements1 = Vector{MeasurementData{LocalUnitaryMeasurementSetting}}(undef,NU)
measurements2 = Vector{MeasurementData{LocalUnitaryMeasurementSetting}}(undef,NU)
for r in 1:NU
    measurement_setting = LocalUnitaryMeasurementSetting(N; site_indices=ξ,ensemble="Haar")
    measurements1[r] = MeasurementData(ψ1,NM;mode="dense", measurement_setting=measurement_setting)
    measurements2[r] = MeasurementData(ψ2,NM;mode="dense", measurement_setting=measurement_setting)
end
measurement_group1= MeasurementGroup(measurements1)
measurement_group2= MeasurementGroup(measurements2)

exact_fidelity = abs(inner(ψ1,ψ2))^2

@testset "Purity and Overlap Tests" begin
    fidelity = get_fidelity(measurement_group1,measurement_group2)
    purity1 = get_purity(measurement_group1)
    overlap = get_overlap(measurement_group1,measurement_group2)

    @show purity1
    @show overlap
    @show fidelity
    @show exact_fidelity
end

@testset "XEB Test" begin
    measurements = MeasurementData(ψ1,NM;mode="dense")
    XEB = get_XEB(ψ1,measurements)
    self_XEB = get_selfXEB(ψ1)
    @show XEB
    @show self_XEB
end
