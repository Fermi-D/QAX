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

"""
QAX: High-Performance Quantum System Simulation Library using JAX

QAX provides efficient tools for simulating quantum systems including:
- Time evolution of quantum states
- Quantum control optimization
- Quantum circuit simulation
"""

__version__ = "0.0.1"

# Core quantum objects
from qax.core import Ket, Bra, Operator

# Display utilities
from qax.plotter.sympy_display import setup_printing

__all__ = [
    # Version
    "__version__",
    # Core objects
    "Ket",
    "Bra",
    "Operator",
    # Display
    "setup_printing",
]
