# Copyright The QAX Developers.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

'''
class StateVectorCircuit:
    """

    """
    def __init__(self, input_states: Ket) -> None:
        self.input_states = input_states

    def add(self, gate: Gate) -> None:
        """
        Adds a quantum gate to the circuit.

        Args:
            gate (Gate): The quantum gate to add.
        """
        self.states = Ket(jnp.dot(gate.data, self.input_states.data))

    def run(self) -> Ket:
        """
        Runs the quantum circuit and returns the final state.

        Returns:
            Ket: The final quantum state after applying all gates.
        """
        return self.states

class DensityMatrixCircuit:
    """
    Represents a quantum circuit using density matrices.

    """
    def __init__(self, input_states: jnp.ndarray) -> None:
        self.input_states = input_states

    def apply(self, superoperator: jnp.ndarray) -> None:

class Gate(Operator):
    def __init__(self, name: str, matrix: jax.Array | np.ndarray):
        """
        Initializes a Gate.

        Args:
            name (str): The name of the gate (e.g., "H", "CNOT", "P(π/4)").
            matrix (jax.Array | np.ndarray): The unitary matrix representing the gate.
        """
        super().__init__(array=matrix)
        self.name = name

    def __repr__(self) -> str:
        """Defines the string representation when print() is called."""
        return f"Gate(name='{self.name}', shape={self.shape}, dtype={self.dtype})"

class Measurement:
    def __init__(self, operator: Operator):
        """

        """
'''
