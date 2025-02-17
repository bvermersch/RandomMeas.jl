using RandomMeas
using Test

# Test parameters
N = 4      # Number of qubits
ξ = siteinds("Qubit", N)

# Generate measurement settings and data
ψ = random_mps(ComplexF64,ξ;linkdims=3)
ϕ = random_mps(ComplexF64,ξ;linkdims=3)
ρ = 0.7*outer(ψ',ψ)+0.3*outer(ϕ',ϕ)

@testset "Full state Trace Moments" begin
    @show get_trace(ρ)
    @show get_trace_moment(ρ,2)
    @show get_trace_moment(ρ,3)
end

@testset "Subsystem state Trace Moments" begin
    #First method
    purity_1 = get_trace_moment(ψ,2,collect(1:2))
    trrho3_1 = get_trace_moment(ψ,3,collect(1:2))
    #Second method
    ρA = reduce_to_subsystem(ψ,collect(1:2))
    purity_2 = get_trace_moment(ρA,2)
    trrho3_2 = get_trace_moment(ρA,3)
    @show purity_1
    @show purity_2
    @show trrho3_1
    @show trrho3_2
end

@testset "PartialTranspose Moments" begin
    #First method
    ρT = partial_transpose(ρ,collect(1:2))

    purity = get_trace_moment(ρ,2)
    @show purity

    p2 = get_trace_moment(ρT,2)
    @show p2

    trrho3 = get_trace_moment(ρ,3)
    @show trrho3

    p3 = get_trace_moment(ρT,3)
    @show p3
end
