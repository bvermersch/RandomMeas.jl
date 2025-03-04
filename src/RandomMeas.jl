"""
    RandomMeas

RandomMeas is a Julia package for measurement-based quantum information processing.
It provides a framework to construct measurement settings, acquire measurement data,
compute classical shadows (both factorized and dense), and estimate various properties of
quantum many-body states and processes.

# Package Organization

1. **Imports:**
   Loads external dependencies and global utilities from `imports.jl`.

2. **Core Structures:**
   Defines fundamental types and data structures (in `MeasurementsStructures.jl` and `ShadowStructures.jl`).

3. **Measurement Modules:**
   - `MeasurementSetting.jl`
   - `MeasurementData.jl`
   - `MeasurementProbability.jl`
   - `MeasurementGroup.jl`
   These files handle the construction and validation of measurement settings and data.

4. **Classical Shadows:**
   - `AbstractShadows.jl`
   - `FactorizedShadows.jl`
   - `DenseShadows.jl`
   - `ShallowShadows.jl`
   Implements classical shadow techniques.

5. **Additional Functionality:**
   - `Estimators.jl`
   - `TensorNetworkUtilities.jl`
   - `Circuits.jl`
   Provide estimation functions (not involving classical shadows), utilities for tensor network states, circuit tools, and protocols for shallow shadows.

6. **Exports:**
   Re-exports public symbols via `exports.jl`, forming the package’s public API.

# Usage

To use RandomMeas in your Julia project, add it to your environment and then:

```julia
using RandomMeas
```

This gives you access to functionality for creating measurement settings, acquiring data,
computing shadows, and performing quantum state estimation.

# Authors and License

- **Authors:** Benoit Vermersch and Andreas Elben
- **License:** Apache 2.0

For further details and examples, please refer to the package’s documentation.
"""

module RandomMeas

# ---------------------------------------------------------------------------
# Load External Dependencies and Global Utilities.
# ---------------------------------------------------------------------------
include("imports.jl")

# ---------------------------------------------------------------------------
# Load Core Data Structures and Types.
# ---------------------------------------------------------------------------
include("MeasurementStructures.jl")
include("ShadowStructures.jl")


# ---------------------------------------------------------------------------
# Load Measurement Modules.
# These files define measurement settings, data, probabilities, and groups.
# ---------------------------------------------------------------------------
include("MeasurementSetting.jl")
include("MeasurementData.jl")
include("MeasurementProbability.jl")
include("MeasurementGroup.jl")

# ---------------------------------------------------------------------------
# Load Classical Shadows Modules.
# These files contain methods to compute classical shadows and use them for
# property estimation.
# ---------------------------------------------------------------------------
include("AbstractShadows.jl")
include("FactorizedShadows.jl")
include("DenseShadows.jl")
include("ShallowShadows.jl")
# ---------------------------------------------------------------------------
# Load Estimator Modules.
# This file implement various estimators which do not utilize classical shadows.
# (e.g. Purity estimation Brydges et al., Cross-Entropy benchmarking... )
# ---------------------------------------------------------------------------

include("Estimators.jl")

# ---------------------------------------------------------------------------
# Load Additional Modules.
# These files contain functions to evaluate quantum information properties of
# MPS/MPO and circuit utilities
# ---------------------------------------------------------------------------
include("TensorNetworkUtilities.jl")
include("Circuits.jl")

# ---------------------------------------------------------------------------
# Export Public API.
# ---------------------------------------------------------------------------
include("exports.jl")

end # module RandomMeas
