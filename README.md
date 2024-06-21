# RandomMeas: The randomized measurements toolbox in julia

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://bvermersch.github.io/RandomMeas.jl/dev/)
[![Build Status](https://github.com/bvermersch/RandomMeas.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/bvermersch/RandomMeas.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

This package presents efficient routines for testing and postprocessing randomized measurements, in order to estimate physical properties in quantum computers.

RandomMeas relies heavily on ITensors. Some examples use the packages PastaQ and MAT

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
 using ITensors,ITensorMPS
 using RandomMeas
 N  = 2
 ψ = random_mps(siteinds("Qubit", 2*N); linkdims=2^N);
 ρ,ξ = reduce_dm(ψ,1,N)
 

 nu=100 #number of random unitaries
 NM=100 #number of projective measurements
 data = zeros(Int8,(nu,NM,N))
 for r in 1:nu
     #generate Haar-random single qubit rotations
     u = get_rotations(ξ,1)
     #acquire RM measurements
     data[r,:,:] = get_RandomMeas(ρ,u,NM)
 end
 ```

2) Postprocessing routines for randomized measurements, eg to get the purity

 ```julia
 purity_e = get_purity_hamming(data,ξ)
    println("estimated purity ", purity_e)
    println("exact purity ", get_purity(ρ))
 ```

3) Interface with matrix-product-states simulations with ITensors.jl to simulate large-scale randomized measurements protocols.

4) Jupyter notebooks to present various recent case studies.

## Examples with Jupyter notebooks

1) [Energy/Energy variance measurements with classical shadows](https://github.com/bvermersch/RandomMeas.jl/blob/main/examples/EnergyVarianceMeasurements.ipynb)

2) [Fidelities from common randomized measurements](https://github.com/bvermersch/RandomMeas.jl/blob/main/examples/FidelityCommonRandomizedMeasurements.ipynb)

3) [Cross-Platform verication](https://github.com/bvermersch/RandomMeas.jl/blob/main/examples/CrossPlatform.ipynb)

4) [Entanglement Entropy of pure states & The page curve"](https://github.com/bvermersch/RandomMeas.jl/blob/main/examples/PureStateEntanglement.ipynb)

5) [Mixed-state entanglement: The p3-PPT condition and batch shadows](https://github.com/bvermersch/RandomMeas.jl/blob/main/examples/MixedStateEntanglement.ipynb)

6) [Analyzing the experimental data of Brydges et al, Science 2019](https://github.com/bvermersch/RandomMeas.jl/blob/main/examples/BrydgesScience2019.ipynb)

7) [Robust Shadow tomography](https://github.com/bvermersch/RandomMeas.jl/blob/main/examples/RobustShadowTomography.ipynb)

8) [Process Shadow tomography](https://github.com/bvermersch/RandomMeas.jl/blob/main/examples/ProcessShadowTomography.ipynb)

9) [Virtual distillation](https://github.com/bvermersch/RandomMeas.jl/blob/main/examples/VirtualDistillation.ipynb)

10) [Cross-Entropy/Self-Cross entropy benchmarking](https://github.com/bvermersch/RandomMeas.jl/blob/main/examples/CrossEntropy.ipynb)

11) [Noisy circuit simulations with tensor networks](https://github.com/bvermersch/RandomMeas.jl/blob/main/examples/CircuitNoiseSimulations.ipynb)

12) [Classical shadows with shallow circuits](https://github.com/bvermersch/RandomMeas.jl/blob/main/examples/ShallowShadows.ipynb)

13) [Surface code and the measurement of the topological entanglement entropy](https://github.com/bvermersch/RandomMeas.jl/blob/main/examples/EntanglementSurfaceCode.ipynb)