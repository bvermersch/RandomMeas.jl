using Test
using ITensors
include("../src/MeasurementSettings.jl")
include("../src/MeasurementData.jl")
include("../src/Shadows_Struct.jl")

# Test parameters
const N = 4  # Number of qubits/sites
const NU = 3  # Number of measurement settings
const NM = 5  # Number of projective measurements

# Generate measurement settings
measurement_settings = LocalUnitaryMeasurementSettings(N, NU, ensemble="Haar")

# Generate random measurement results
measurement_results = rand(1:2, NU, NM, N)  # Random binary results in the range [1, 2]

# Create a MeasurementData object
measurement_data = MeasurementData(measurement_results; measurement_settings=measurement_settings)

# Testing FactorizedShadow constructor
@testset "FactorizedShadow Tests" begin
    for r in 1:NU
        for m in 1:NM
            local_unitaries = measurement_settings.local_unitaries[r, :]
            data = measurement_results[r, m, :]

            # Construct FactorizedShadow
            shadow = FactorizedShadow(data, local_unitaries)
            @show shadow.N == N
            @test length(shadow.factorized_shadow) == N
        end
    end
end

# Testing factorized shadow generation
@testset "Batch FactorizedShadow Tests" begin
    G_vec = fill(1.0, N)  # No error correction
    shadows = get_factorized_shadows(measurement_data; G_vec=G_vec)

    @test size(shadows) == (NU, NM)

    for r in 1:NU
        for m in 1:NM
            shadow = shadows[r, m]
            @test shadow.N == N
            @test length(shadow.factorized_shadow) == N
        end
    end
end

# Test with non-uniform G_vec
@testset "FactorizedShadow with G_vec" begin
    G_vec = [1.0, 0.9, 1.1, 1.2]  # Example of non-uniform G values
    for r in 1:NU
        for m in 1:NM
            local_unitaries = measurement_settings.local_unitaries[r, :]
            data = measurement_results[r, m, :]

            # Construct FactorizedShadow
            shadow = FactorizedShadow(data, local_unitaries; G_vec=G_vec)
            @test shadow.N == N
            @test length(shadow.factorized_shadow) == N
        end
    end
end

# Test edge cases
@testset "FactorizedShadow Edge Cases" begin
    # Invalid G_vec length
    @test_throws AssertionError FactorizedShadow(measurement_results[1, 1, :], measurement_settings.local_unitaries[1, :]; G_vec=[1.0, 0.9])

    # Invalid measurement results length
    @test_throws AssertionError FactorizedShadow([1, 2], measurement_settings.local_unitaries[1, :])
end


function get_shadow_factorized_old(data::Array{Int}, ξ::Vector{Index{Int64}}, u::Vector{ITensor};G_vec::Union{Nothing,Vector{Float64}}=nothing)
    N = length(u)
    ρ = Vector{ITensor}()
    for i in 1:N
        if G_vec ===nothing
            α = 3
            β = -1
        else
            α = 3 / (2 * G_vec[i] - 1)
            β = (G_vec[i] - 2) / (2 * G_vec[i] - 1)
        end
        #u*_{s',s}|s'><s'|=u^dag_{s,s'}|s'><s'|
        ψ = dag(u[i]) * onehot(ξ[i]' => data[i])
        push!(ρ, α * ψ' * dag(ψ) + β * δ(ξ[i], ξ[i]'))
    end
    return ρ
end

@testset "FactorizedShadow Backwards compability" begin
    # Define parameters
    N = 4  # Number of qubits/sites
    ξ = siteinds("Qubit", N)
    local_unitaries = [op("RandomUnitary", ξ[i]) for i in 1:N]
    measurement_results = rand(1:2, N)  # Simulate binary results (1, 2 for Julia indexing)
    G_vec = [1.2, 0.8, 1.5, 1.0]  # Example G values

    # Test FactorizedShadow constructor
    shadow = FactorizedShadow(measurement_results, local_unitaries; G_vec=G_vec)

    @test shadow.N == N
    @test length(shadow.factorized_shadow) == N
    @test shadow.ξ == ξ
    @test shadow.G == G_vec

    # Compare outputs of get_shadow_factorized and FactorizedShadow
    factorized_shadow_manual = get_shadow_factorized_old(measurement_results, ξ, local_unitaries; G_vec=G_vec)
    @test length(factorized_shadow_manual) == length(shadow.factorized_shadow)

    for i in 1:N
        @test isapprox(
            Array(factorized_shadow_manual[i], ξ[i]', ξ[i]),
            Array(shadow.factorized_shadow[i], ξ[i]', ξ[i]),
            atol=1e-10
        )
    end

    println("FactorizedShadow matches get_shadow_factorized outputs.")
end

println("All tests passed!")
