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

"""Quantum state generation functions."""

import jax.numpy as jnp
from jax.scipy.linalg import expm
from .core import Ket, Operator
from . import operators as ops


def qubit(n: int) -> Ket:
    """
    Generate an initial n-qubit state |00...0⟩ in a Hilbert space of dimension 2^n.

    Args:
        n (int): Number of qubits.

    Returns:
        Ket: A 2D array (2^n, 1) representing the n-qubit state |00...0⟩.
    """
    mat = jnp.zeros(2**n, dtype=jnp.complex64).at[0].set(1.0)
    return Ket(mat)


def vacuum(dim: int) -> Ket:
    """
    Generate a vacuum state |0⟩ in a Hilbert space of dimension dim.

    Args:
        dim (int): The dimension of the Hilbert space.

    Returns:
        Ket: A 2D array (dim, 1) representing the vacuum state |0⟩.
    """
    mat = jnp.zeros(dim, dtype=jnp.complex64).at[0].set(1.0)
    return Ket(mat)


def fock(n_photon: int, dim: int) -> Ket:
    """
    Generate a Fock state |n⟩ in a Hilbert space of dimension dim.

    Args:
        n_photon (int): The Fock state index (0-based).
        dim (int): The dimension of the Hilbert space.

    Returns:
        Ket: A 2D array (dim, 1) representing the Fock state |n⟩.

    Raises:
        ValueError: If n_photon >= dim.
    """
    if n_photon >= dim:
        raise ValueError(
            f"n_photon ({n_photon}) must be less than the dimension ({dim})."
        )
    mat = jnp.zeros(dim, dtype=jnp.complex64).at[n_photon].set(1.0)
    return Ket(mat)


def coherent(alpha: complex, dim: int) -> Ket:
    """
    Generate a coherent state |α⟩ in a Hilbert space of dimension dim.

    The coherent state is defined as: |α⟩ = D(α)|0⟩
    where D(α) is the displacement operator.

    Args:
        alpha (complex): The coherent state amplitude.
        dim (int): The dimension of the Hilbert space.

    Returns:
        Ket: A 2D array (dim, 1) representing the coherent state |α⟩.
    """
    return ops.displacement(alpha, dim) @ vacuum(dim)


def squeezed(z: complex, dim: int) -> Ket:
    """
    Generate a squeezed vacuum state in a Hilbert space of dimension dim.

    The squeezed state is defined as: S(z)|0⟩
    where S(z) is the squeeze operator.

    Args:
        z (complex): The squeeze parameter.
        dim (int): The dimension of the Hilbert space.

    Returns:
        Ket: A 2D array (dim, 1) representing the squeezed state.
    """
    return ops.squeeze(z, dim) @ vacuum(dim)


def position(x: float, dim: int) -> Ket:
    """
    Generate a position eigenstate |x⟩ in a Hilbert space of dimension dim.

    Args:
        x (float): Position eigenvalue.
        dim (int): The dimension of the Hilbert space.

    Returns:
        Ket: A 2D array (dim, 1) representing the position state |x⟩.
    """
    coff_1 = jnp.pi ** (1 / 4)
    coff_2 = jnp.exp(0.5 * x**2)

    # Create the operator (a† - √2 x I)
    a_dag = ops.creation(dim)
    identity = ops.identity(dim)
    op_arg = a_dag - jnp.sqrt(2) * x * identity

    # Compute exp(-0.5 * op_arg^2)
    op_squared = op_arg @ op_arg
    exponential = Operator(expm(-0.5 * op_squared.data))

    # Apply to vacuum state
    state = coff_1 * coff_2 * (exponential @ vacuum(dim))
    return state


def momentum(p: float, dim: int) -> Ket:
    """
    Generate a momentum eigenstate |p⟩ in a Hilbert space of dimension dim.

    Args:
        p (float): Momentum eigenvalue.
        dim (int): The dimension of the Hilbert space.

    Returns:
        Ket: A 2D array (dim, 1) representing the momentum state |p⟩.
    """
    coff_1 = jnp.pi ** (1 / 4)
    coff_2 = jnp.exp(0.5 * p**2)

    # Create the operator (a + i√2 p I)
    a = ops.annihilation(dim)
    identity = ops.identity(dim)
    op_arg = a + 1j * jnp.sqrt(2) * p * identity

    # Compute exp(-0.5 * op_arg^2)
    op_squared = op_arg @ op_arg
    exponential = Operator(expm(-0.5 * op_squared.data))

    # Apply to vacuum state
    state = coff_1 * coff_2 * (exponential @ vacuum(dim))
    return state
