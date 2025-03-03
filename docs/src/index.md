```@meta
CurrentModule = RandomMeas
```

# RandomMeas

Documentation for [RandomMeas.jl](https://github.com/bvermersch/RandomMeas.jl): The randomized measurements toolbox in julia

```@contents
Depth = 2:3
```

## Data acquisition routines

```@autodocs
Modules = [RandomMeas]
Pages = ["Structures.jl","MeasurementSetting.jl"]
```

## Data storage and handling routines

```@autodocs
Modules = [RandomMeas]
Pages = ["MeasurementData.jl","MeasurementGroup.jl","MeasurementProbabilities.jl"]
```

## Postprocessing routines (excluding classical shadows)

```@autodocs
Modules = [RandomMeas]
Pages = ["MeasurementProbabilities.jl", "Estimators.jl"]
```

## Postprocessing routines for classical shadows

```@autodocs
Modules = [RandomMeas]
Pages = ["Shadows.jl","FactorizedShadows.jl", "DenseShadows.jl","ShallowShadows.jl"]
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
