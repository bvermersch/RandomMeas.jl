# RandomMeas: The randomized measurement toolbox in Julia

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://bvermersch.github.io/RandomMeas.jl/dev/)
[![Build Status](https://github.com/bvermersch/RandomMeas.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/bvermersch/RandomMeas.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

This package provides efficient routines for sampling, simulating, and post-processing randomized measurements, including classical shadows, to extract properties of many-body quantum states and processes.
RandomMeas relies heavily on [ITensors.jl](https://itensor.github.io/ITensors.jl/dev/).

<img src="Pics/RandomMeas.png" alt="drawing" width="500"/>.

## Install

In a Julia terminal, install the package RandomMeas

```julia
julia> ]
pkg> add RandomMeas
```

## Documentation

[dev](https://bvermersch.github.io/RandomMeas.jl/dev/) -- documentation of the in-development version.

## Presentation

1) Routines to prepare on QPUs/simulate the data acquisition

 ```julia
using RandomMeas
N = 3#number of qubits
χ = 2 #bon dimension of a Matrix-Product-State
ξ = siteinds("Qubit", N)
ψ = random_mps(ξ; linkdims=χ);

 
NU=200 #Number of measurement settings
NM=100 #Number of projective measurements per setting
measurement_group = MeasurementGroup(ψ,NU,NM;mode="dense");
 ```

2) Postprocessing routines for randomized measurements, eg to get the purity

 ```julia
p_estimated = zeros(N)
p_exact = zeros(N)
for NA in 1:N
        p_estimated[NA] = get_purity(measurement_group, collect(1:NA)) #Estimated value
        p_exact[NA] = get_trace_moment(ψ,2,collect(1:NA)) #Exact value
end
println("Exact purities ", p_estimated);
println("Estimated purities ",p_exact);
 ```

3) Interface with matrix-product-states simulations with ITensors.jl to simulate large-scale randomized measurements protocols.

4) Jupyter notebooks to present various recent case studies.

## Examples with Jupyter notebooks

### Classical shadows

1) [Energy/Energy variance measurements with classical shadows](examples/EnergyVarianceMeasurements.ipynb)

2) [Robust Shadow tomography](examples/RobustShadowTomography.ipynb)

3) [Process Shadow tomography](examples/ProcessShadowTomography.ipynb)

4) [Classical shadows with shallow circuits](examples/ShallowShadows.ipynb)

5) [Virtual distillation](examples/VirtualDistillation.ipynb)

### Quantum benchmark

6) [Cross-Entropy/Self-Cross entropy benchmarking](examples/CrossEntropyBenchmarking.ipynb)

7) [Fidelities from common randomized measurements](examples/FidelityCommonRandomizedMeasurements.ipynb)

8) [Cross-Platform verification](examples/CrossPlatform.ipynb)

### Entanglement

9) [Entanglement entropy of pure states"](examples/PureStateEntanglement.ipynb)

10) [Analyzing the experimental data of Brydges et al, Science 2019](examples/BrydgesScience2019.ipynb)

11) [Surface code and the measurement of the topological entanglement entropy](examples/EntanglementSurfaceCode.ipynb)

12) [Mixed-state entanglement: The p3-PPT condition and batch shadows](examples/MixedStateEntanglement.ipynb)

### Miscellanous

13) [Noisy circuit simulations with tensor networks](examples/CircuitNoiseSimulations.ipynb)

14) [Estimating statistical error bars via Jackknife resampling](examples/ErrorBars.ipynb)

15) [Executing randomized measurements on IBM's quantum computers](examples/RandomizedMeasurementsQiskit.ipynb)

16) [Postprocessing randomized measurements from IBM's quantum computers](examples/RandomizedMeasurementsQiskitPostprocessing.ipynb)

## Acknowledgments

The development of this code has been supported by University Grenoble Alpes, CNRS, Agence National de la Recherche (ANR) under the programs QRand (ANR-20-CE47-0005), and via the research programs Plan France 2030 EPIQ (ANR-22-PETQ-0007), QUBITAF (ANR-22-PETQ-0004) and HQI (ANR-22-PNCQ-0002), and Quobly. Furthermore, we acknowledge support from the Walter Burke Institue and the ETHZ-PSI Quantum Computing Hub.
