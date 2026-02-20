# benchmarks/run_benchmarks.jl
using Pkg
using Random
using RandomMeas
using BenchmarkTools
using Statistics

using InteractiveUtils
using LinearAlgebra

println("===== Environment =====")
versioninfo(verbose=false)

println("\n===== Threads =====")
println("JULIA_NUM_THREADS = ", Threads.nthreads())
println("BLAS threads      = ", BLAS.get_num_threads())

Pkg.status()

# -----------------------
# Defaults (as requested). Override via ENV if desired.
# -----------------------
N  = parse(Int, get(ENV, "RM_BENCH_N",  "50"))
NU = parse(Int, get(ENV, "RM_BENCH_NU", "200"))
NM = parse(Int, get(ENV, "RM_BENCH_NM", "100"))
χ  = parse(Int, get(ENV, "RM_BENCH_CHI","2"))

NA = parse(Int, get(ENV, "RM_BENCH_NA", "6"))
A  = collect(1:NA)

Kobs  = parse(Int, get(ENV, "RM_BENCH_KOBS", "5"))           # number of observables
k_vec = parse.(Int, split(get(ENV, "RM_BENCH_KVEC", "2,3,4"), ","))
kmax = maximum(k_vec)
nbatch_mom = 4 * kmax # often sufficient in practise

println("\n===== Benchmark configuration =====")
println("N=$N, NU=$NU, NM=$NM, χ=$χ, NA=$NA, Kobs=$Kobs, k_vec=$k_vec, nbatch_mom=$nbatch_mom")
println("Subsystem A = 1:$NA (reduced density matrix moments on A only)")

# BenchmarkTools settings
BenchmarkTools.DEFAULT_PARAMETERS.samples = 20
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 60.0

# reproducible RNG for internal random choices (observables)
rng = MersenneTwister(1)

# -----------------------
# 1) Randomized measurement settings (as in manuscript)
# -----------------------
measurement_settings = [LocalUnitaryMeasurementSetting(N) for _ in 1:NU]

# -----------------------
# 2) Classical simulation of randomized measurements (as in manuscript)
# -----------------------
site_indices = siteinds("Qubit", N)
ψ = random_mps(site_indices; linkdims=χ)

println("\n===== Stage 1: simulate randomized measurements to generate a MeasurementGroup(ψ, settings, NM) =====")

_ = MeasurementGroup(ψ, measurement_settings[1:2], 2)
@btime MeasurementGroup(ψ, measurement_settings, NM)

# Reduce to subsystem A for dense + moments + fair factorized comparison on same support
measurement_group = MeasurementGroup(ψ, measurement_settings, NM)
measurement_group_A = reduce_to_subsystem(measurement_group, A)

println("\n===== Stage 2: build shadows on subsystem A =====")

println("\n-- build factorized shadows (NU × NM) --")
_ = get_factorized_shadows(measurement_group_A); GC.gc()
@btime get_factorized_shadows($measurement_group_A);

println("\n-- build average dense shadow estimating observables --")
_ = get_dense_shadows(measurement_group_A; number_of_ru_batches=1); GC.gc()
@btime get_dense_shadows($measurement_group_A; number_of_ru_batches=1);

println("\n-- build dense shadows for moments (nbatch = $nbatch_mom) --")
_ = get_dense_shadows(measurement_group_A; number_of_ru_batches=nbatch_mom); GC.gc()
@btime get_dense_shadows($measurement_group_A; number_of_ru_batches=$nbatch_mom);

# Build once for estimator benchmarks
shF_A = get_factorized_shadows(measurement_group_A)  # Array{FactorizedShadow,2} size (NU,NM)
shD_A = get_dense_shadows(measurement_group_A; number_of_ru_batches=1) # Vector{DenseShadow} length NU

# -----------------------
# Helpers: random local Pauli MPOs on subsystem A
# -----------------------
function random_pauli_mpos(rng::AbstractRNG, siteinds::Vector{Index{Int64}}, K::Int; p_identity=0.5)
    N = length(siteinds)
    out = Vector{MPO}(undef, K)
    for k in 1:K
        ops = Vector{String}(undef, N)
        for j in 1:N
            if rand(rng) < p_identity
                ops[j] = "I"
            else
                ops[j] = rand(rng, ("X","Y","Z"))
            end
        end
        out[k] = MPO(ComplexF64, siteinds, ops)
    end
    return out
end

siteA = measurement_group_A.measurements[1].measurement_setting.site_indices
Os = random_pauli_mpos(rng, siteA, Kobs; p_identity=0.5)

estimate_all(shadows, ops) = [real(get_expect_shadow(O, shadows)) for O in ops]

println("\n===== Stage 3: observable estimation on A (mean) =====")
println("\n-- estimate $Kobs MPO expectations (factorized) --")
@btime estimate_all($shF_A, $Os);

println("\n-- estimate $Kobs MPO expectations (dense) --")
@btime estimate_all($shD_A, $Os);

# -----------------------
# Stage 4: trace moments Tr[ρ_A^k] (mean; no MoM)
# get_trace_moments expects Array{<:AbstractShadow,2}.
# Factorized: already (NU,NM).
# Dense: Vector{DenseShadow} (length NU). Reshape to (NU,1).
# -----------------------

# Build once for estimator benchmarks
shD_A = get_dense_shadows(measurement_group_A; number_of_ru_batches=nbatch_mom) # Vector{DenseShadow} length NU

# Warmup compile
_ = get_trace_moments(shD_A, k_vec)

println("\n===== Stage 4: trace moments Tr[ρ_A^k] on A (mean) with dense batch shadows (nbatch = $nbatch_mom) =====")

@btime get_trace_moments($shD_A, $k_vec)

println("\n===== Done =====")
