"""
    Exports.jl

This file re-exports the public symbols of the RandomMeas package, serving as the main public API. It leverages Reexport.jl to expose selected types and functions from both internal modules and external dependencies.

# Overview

The exported symbols are organized into the following groups:

- **Measurement Settings:**
  Contains types that define various measurement settings used to configure randomized measurements.
  - *Types:* `MeasurementSetting`, `LocalMeasurementSetting`, `LocalUnitaryMeasurementSetting`, `ComputationalBasisMeasurementSetting`, `ShallowUnitaryMeasurementSetting`

- **Measurement Data Structures:**
  Provides types for handling experimental measurement data, associated probabilities, and groups of measurements.
  - *Types:* `MeasurementData`, `MeasurementProbability`, `MeasurementGroup`

- **Classical Shadows API:**
  Implements core functions for constructing and manipulating classical shadows for quantum states and processes.
  - *Functions:* `get_expect_shadow`, `get_trace_moment`, `get_trace_moments`, `get_trace_product`, `multiply`, `trace`, `partial_trace`, `partial_transpose`
  - *Types:* `AbstractShadow`, along with concrete implementations such as `DenseShadow`, `FactorizedShadow`, and `ShallowShadow`

- **Dense and Factorized Shadows:**
  Provides routines for creating and converting between different shadow representations.
  - *Functions:* `get_dense_shadows`, `get_factorized_shadows`, `convert_to_dense_shadow`

  - **Shallow Shadows:**
  Implements methods specific to constructing and using shallow shadows, a resource-efficient alternative to full classical shadows suitable for near-term devices.
  - *Types:* `ShallowShadow`
  - *Functions:* `get_shallow_depolarization_mps`, `get_depolarization_map`, `loss_inverse_depolarization_map`, `apply_map`, `get_shallow_shadows`


- **Tensor Network Utilities:**
  Offers functions for quantum state analysis, including site index extraction, trace computation, Born probability MPS creation, and XEB calculations.
  - *Functions:* `get_siteinds`, `get_trace`, `get_Born_MPS`, `get_selfXEB`, `flatten`, `get_average_mps`

- **Estimators:**
  Contains statistical tools for estimating key quantum information metrics such as fidelity, purity, and overlaps.
  - *Functions:* `get_h_tensor`, `get_fidelity`, `get_overlap`, `get_purity`, `get_XEB`

- **Circuits and Noise:**
  Provides tools for generating random circuits and simulating noise, including depolarization and random Pauli or magnetic field layers.
  - *Functions:* `apply_depo_channel`, `random_circuit`, `random_Pauli_layer`, `random_magnetic_field_layer`

Overall, these symbols form the core building blocks for performing randomized measurements, constructing classical shadows (and their shallow variants), and analyzing quantum measurement data.

For more details on each function and type, please refer to the respective documentation in the source files.
"""

using Reexport
@reexport using ITensors,ITensorMPS,ProgressMeter,OptimKit

export

##############################
# Measurement Settings
##############################

MeasurementSetting,
LocalMeasurementSetting,
LocalUnitaryMeasurementSetting,
ComputationalBasisMeasurementSetting,
ShallowUnitaryMeasurementSetting,

get_rotation,
reduce_to_subsystem,
export_LocalUnitaryMeasurementSetting,
import_LocalUnitaryMeasurementSetting,
import_MeasurementData,
export_MeasurementData,
import_MeasurementGroup,
export_MeasurementGroup,

##############################
# Measurement Probability
##############################
MeasurementProbability,


##############################
# Measurement Data
##############################
MeasurementData,

##############################
# Measurement Group
##############################
MeasurementGroup,

##############################
# Abstract Shadows API
##############################
AbstractShadow,
get_expect_shadow,
get_trace_moment,
get_trace_moments,
get_trace_product,
multiply,
trace,
partial_trace,
partial_transpose,

##############################
# Dense and Factorized Shadows
##############################
FactorizedShadow,
DenseShadow,
get_dense_shadows,
get_factorized_shadows,
convert_to_dense_shadow,

##############################
# Shallow Shadows
##############################
ShallowShadow,
get_shallow_depolarization_mps,
get_depolarization_map,
loss_inverse_depolarization_map,
apply_map,
get_shallow_shadows,

##############################
# Estimators
##############################
get_h_tensor,
get_fidelity,
get_overlap,
get_purity,
get_XEB,
get_calibration_vector,

##############################
# Tensor Network Utilities
##############################
get_siteinds,
get_trace_moment,
get_trace,
get_Born_MPS,
get_selfXEB,
flatten,
get_average_mps,


##############################å
# Circuits and Noise
##############################
apply_depo_channel,
random_circuit,
random_Pauli_layer,
random_magnetic_field_layer
