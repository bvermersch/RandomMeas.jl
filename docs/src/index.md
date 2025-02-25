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
