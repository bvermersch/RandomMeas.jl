# Copyright (c) 2024 Benoît Vermersch and Andreas Elben
# SPDX-License-Identifier: Apache-2.0
# http://www.apache.org/licenses/LICENSE-2.0

"""
RandomMeas.jl - Type Definitions

Enums for type-safe parameter specification throughout RandomMeas.jl
"""

"""
UnitaryEnsemble - Unitary Ensemble Type

Enum for specifying the type of unitary ensemble.

# Values
- `Haar`: Haar-random unitaries
- `Pauli`: Random Pauli rotations
- `Identity`: Identity transformation
"""
@enum UnitaryEnsemble begin
    Haar
    Pauli
    Identity
end

"""
SimulationMode - Simulation Method Type

Enum for specifying the simulation method.

# Values
- `Dense`: Dense matrix representation
- `TensorNetwork`: Tensor network (MPS/MPO) representation
"""
@enum SimulationMode begin
    Dense
    TensorNetwork
end
