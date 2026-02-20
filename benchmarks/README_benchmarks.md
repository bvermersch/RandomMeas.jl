# Benchmarks

This directory contains two lightweight benchmark entry points:

- `example_benchmarks.jl` — a **single-run** benchmark that exercises the full workflow
  (simulate randomized measurements → build shadows on a subsystem → estimate observables and trace moments).
- `scaling_benchmarks.ipynb` — an **interactive scalability notebook** that sweeps key parameters
  (system size, bond dimension, number of settings/shots, subsystem size) and produces plots/CSV outputs.

Both are intended for **reproducible, version-pinned performance benchmarks** 
---

## 1. `example_benchmarks.jl`

### What it measures

Given defaults (override via environment variables):

1. **Classical simulation of randomized measurements**
   - Samples randomized measurement settings (`LocalUnitaryMeasurementSetting`).
   - Simulates outcomes on an MPS `ψ` via `MeasurementGroup(ψ, settings, NM)`.

2. **Shadow construction on subsystem A**
   - Factorized shadows: `get_factorized_shadows(reduce_to_subsystem(...))`
   - Dense shadows (two modes):
     - `number_of_ru_batches=1` (a single dense object suitable for observable estimation)
     - `number_of_ru_batches=nbatch_mom` (dense batching suitable for trace-moment estimation)

3. **Post-processing (mean estimators; no median-of-means)**
   - Observable estimation: `get_expect_shadow` on `Kobs` random local Pauli MPOs on A.
   - Trace moments: `get_trace_moments` for `k_vec` on A (dense batching).

Timing is reported with `BenchmarkTools.@btime` (including allocations).

### How to run

From the repository root:

```bash
julia --project benchmarks/example_benchmarks.jl
```

### Parameters (ENV overrides)

```bash
RM_BENCH_N=50        # total qubits
RM_BENCH_NU=200      # number of random settings
RM_BENCH_NM=100      # shots per setting
RM_BENCH_CHI=2       # MPS bond dimension
RM_BENCH_NA=6        # subsystem size |A|
RM_BENCH_KOBS=5      # number of local Pauli MPOs to estimate on A
RM_BENCH_KVEC=2,3,4  # trace moments to estimate
```

Example:

```bash
RM_BENCH_N=50 RM_BENCH_NU=500 RM_BENCH_NM=250 RM_BENCH_CHI=4 RM_BENCH_NA=8 julia --project benchmarks/example_benchmarks.jl
```

---

## 2. `scaling_benchmarks.ipynb`

### What it measures

The notebook is organized into two parts:

1. **Simulation scaling** (MPS-based measurement simulation)
   - Sweeps one parameter at a time for `MeasurementGroup(ψ, settings, NM)`:
     - `N` (system size), `χ` (bond dimension), `NU` (settings), `NM` (shots)
   - Uses `@elapsed` with a small number of repeats and reports the median.

2. **Post-processing scaling on a subsystem A**
   - Shadow construction vs `NA`:
     - factorized and dense (1 batch)
     - optionally records `Base.summarysize` for the constructed shadow objects
   - Observable estimation vs `NA` (estimator-only timing, shadows pre-built)
   - Trace moments vs `NA` (dense-only by default; factorized moments are typically impractical beyond small `NA`)

### How to run

Open the notebook in your preferred environment (VSCode, IJulia/Jupyter) with the project activated:

```julia
using Pkg; Pkg.activate("."); Pkg.instantiate()
```

Then run cells top-to-bottom.

### Parameters (ENV overrides)

The notebook reads `RM_*` environment variables (see the first parameter cell). Common ones:

```bash
RM_N=50 RM_NU=200 RM_NM=100 RM_CHI=4
RM_KOBS=5 RM_KVEC=2,3,4
RM_NBATCH_MOM=16     # dense RU-batches for trace moments (heuristic: ~O(kmax))
RM_REPS=2            # repeats for @elapsed medians
```

Sweep controls (examples):

```bash
RM_SWEEP_N="20,30,40,50,60"
RM_SWEEP_CHI="2,4,8,12"
RM_SWEEP_NU="50,100,200,400"
RM_SWEEP_NM="25,50,100,200"
RM_SWEEP_NA_BUILD="4,6,8,10,12"
RM_SWEEP_NA_OBS="4,6,8,10,12"
RM_SWEEP_NA_MOM="4,6,8"
```

### Outputs

The notebook produces plots inline and (optionally) writes CSV files to:

```
benchmarks/results/
```

CSV export is helpful for comparing:
- different machines,
- different Julia/RandomMeas versions,
- different branches (e.g. `main` vs `dev_*`).

---

## Notes on interpretation

- **Machine- and version-dependence:** always report the printed `versioninfo()` and `Pkg.status()`.
- **Dense vs factorized trade-offs:** dense objects scale exponentially with subsystem size `NA`; factorized objects scale linearly in `N` but may be expensive for small `N` for polynomial functionals because U-statistics scale with the number of snapshots.
