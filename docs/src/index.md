```@meta
CurrentModule = RandomMeas
```

# RandomMeas

Documentation for [RandomMeas.jl](https://github.com/bvermersch/RandomMeas.jl): The randomized measurements toolbox in julia

```@contents
Depth = 2:3
```
## Structures

```@autodocs
Modules = [RandomMeas]
Pages = ["MeasurementStructures.jl","ShadowStructures.jl"]
```

## Data acquisition routines

```@autodocs
Modules = [RandomMeas]
Pages = ["MeasurementStructures.jl","ShadowStructures.jl","MeasurementSetting.jl","MeasurementData.jl","MeasurementGroup.jl","MeasurementProbability.jl"]
```

## Postprocessing routines (excluding classical shadows)

```@autodocs
Modules = [RandomMeas]
Pages = ["Estimators.jl"]
```

## Postprocessing routines for classical shadows

```@autodocs
Modules = [RandomMeas]
Pages = ["AbstractShadows.jl","FactorizedShadows.jl", "DenseShadows.jl","ShallowShadows.jl"]
```

## Routines for simulating quantum circuits

```@autodocs
Modules = [RandomMeas]
Pages = ["Circuits.jl"]
```

## Additional useful routines for ITensor

```@autodocs
Modules = [RandomMeas]
Pages = ["TensorNetworkUtilities.jl"]
```


## Examples

### Classical shadows

1) [Energy/Energy variance measurements with classical shadows](https://github.com/bvermersch/RandomMeas.jl/blob/main/examples/EnergyVarianceMeasurements.ipynb)

2) [Robust Shadow tomography](https://github.com/bvermersch/RandomMeas.jl/blob/main/examples/RobustShadowTomography.ipynb)

3) [Process Shadow tomography](https://github.com/bvermersch/RandomMeas.jl/blob/main/examples/ProcessShadowTomography.ipynb)

4) [Classical shadows with shallow circuits](https://github.com/bvermersch/RandomMeas.jl/blob/main/examples/ShallowShadows.ipynb)

5) [Virtual distillation](https://github.com/bvermersch/RandomMeas.jl/blob/main/examples/VirtualDistillation.ipynb)

### Quantum benchmark

6) [Cross-Entropy/Self-Cross entropy benchmarking](https://github.com/bvermersch/RandomMeas.jl/blob/main/examples/CrossEntropyBenchmarking.ipynb)

7) [Fidelities from common randomized measurements](https://github.com/bvermersch/RandomMeas.jl/blob/main/examples/FidelityCommonRandomizedMeasurements.ipynb)

8) [Cross-Platform verification](https://github.com/bvermersch/RandomMeas.jl/blob/main/examples/CrossPlatform.ipynb)

### Entanglement

9) [Entanglement entropy of pure states"](https://github.com/bvermersch/RandomMeas.jl/blob/main/examples/PureStateEntanglement.ipynb)

10) [Analyzing the experimental data of Brydges et al, Science 2019](https://github.com/bvermersch/RandomMeas.jl/blob/main/examples/BrydgesScience2019.ipynb)

11) [Surface code and the measurement of the topological entanglement entropy](https://github.com/bvermersch/RandomMeas.jl/blob/main/examples/EntanglementSurfaceCode.ipynb)

12) [Mixed-state entanglement: The p3-PPT condition and batch shadows](https://github.com/bvermersch/RandomMeas.jl/blob/main/examples/MixedStateEntanglement.ipynb)

### Miscellanous

13) [Noisy circuit simulations with tensor networks](https://github.com/bvermersch/RandomMeas.jl/blob/main/examples/CircuitNoiseSimulations.ipynb)

14) [Estimating statistical error bars via Jackknife resampling](https://github.com/bvermersch/RandomMeas.jl/blob/main/examples/ErrorBars.ipynb)

15) [Executing randomized measurements on IBM's quantum computers](https://github.com/bvermersch/RandomMeas.jl/blob/main/examples/RandomizedMeasurementsQiskit.ipynb)
