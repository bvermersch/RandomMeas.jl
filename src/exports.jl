"""
    Exports.jl

This file re-exports the public symbols of the RandomMeas package.
It uses Reexport.jl to expose selected types and functions from both internal modules
and external dependencies, forming the public API of RandomMeas.
"""

using Reexport
@reexport using ITensors,ITensorMPS,ProgressMeter,OptimKit

export

#Structures
MeasurementSetting,
LocalMeasurementSetting,
LocalUnitaryMeasurementSetting,
ComputationalBasisMeasurementSetting,
ShallowUnitaryMeasurementSetting,

MeasurementProbability,
MeasurementData,
MeasurementGroup,

AbstractShadow,
FactorizedShadow,
DenseShadow,
ShallowShadow,

#Methods for MeasurementSetting
get_rotation,
reduce_to_subsystem,
export_LocalUnitaryMeasurementSetting,
import_LocalUnitaryMeasurementSetting,
import_MeasurementData,
export_MeasurementData,
import_MeasurementGroup,
export_MeasurementGroup,

#Methods for MeasurementProbability


#Methods for MeasurementData

#import_measurement_data,
#export_measurement_data,

#MeasurementGroup

#AbstractShadows
get_expect_shadow,
get_trace_moment,
get_trace_moments,
get_trace_product,
multiply,
trace,
partial_trace,
partial_transpose,

#DenseShadows
get_dense_shadows,
# get_purity_dense_shadows,

#FactorizedShadows
get_factorized_shadows,
convert_to_dense_shadow,


#TensorNetworkUtilities.jl
get_siteinds,
get_trace_moment,
get_trace,
get_Born_MPS,
get_selfXEB,
partial_transpose,
flatten,
get_average_mps,


#Estimators.jl
 get_h_tensor,
 get_fidelity,
 get_overlap,
 get_purity,
 get_XEB,


#Circuits.jl
apply_depo_channel,
random_circuit,
random_Pauli_layer,
random_magnetic_field_layer,

#ShallowShadows.jl,
get_shallow_depolarization_mps,
get_depolarization_map,
loss_inverse_depolarization_map,
apply_map,
get_shallow_shadows
# get_inverse_depolarization_vector,
# apply_inverse_channel
