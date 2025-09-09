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

import jax.numpy as jnp
from jax.scipy.linalg import expm
from .core import Operator


def identity(dim: int) -> Operator:
    """
    Returns the identity operator.

    Args:
      dim (int): The dimension of Hilbert space.

    Returns:
      Operator: A 'dim'x'dim' matrix of the identity operator with complex64 precision.
    """
    mat = jnp.identity(dim, dtype=jnp.complex64)
    return Operator(mat)


def pauli_x() -> Operator:
    """
    Returns the Pauli X operator.
    The Pauli X operator is a 2x2 matrix defined as:
    [[0, 1],
     [1, 0]]

    Returns:
      Operator: A 2x2 matrix of the Pauli X operator with complex64 precision.
    """
    mat = jnp.array([[0 + 0j, 1 + 0j], [1 + 0j, 0 + 0j]], dtype=jnp.complex64)
    return Operator(mat)


def pauli_y() -> Operator:
    """
    Returns the Pauli Y operator.
    The Pauli Y operator is a 2x2 matrix defined as:
    [[0, -i],
     [i, 0]]

    Returns:
      Operator: A 2x2 matrix of the Pauli Y operator with complex64 precision.
    """
    mat = jnp.array([[0 + 0j, 0 - 1j], [0 + 1j, 0 + 0j]], dtype=jnp.complex64)
    return Operator(mat)


def pauli_z() -> Operator:
    """
    Returns the Pauli Z operator.
    The Pauli Z operator is a 2x2 matrix defined as:
    [[1, 0],
     [0, -1]]

    Returns:
      Operator: A 2x2 matrix of the Pauli Z operator with complex64 precision.
    """
    mat = jnp.array([[1 + 0j, 0 + 0j], [0 + 0j, -1 + 0j]], dtype=jnp.complex64)
    return Operator(mat)


def raising() -> Operator:
    """
    Returns the raising operator.
    The raising operator is a 2x2 matrix defined as:
    [[0, 1],
     [0, 0]]

    Returns:
      Operator: A 2x2 matrix of the raising operator with complex64 precision.
    """
    mat = jnp.array([[0.0, 1.0], [0.0, 0.0]], dtype=jnp.complex64)
    return Operator(mat)


def lowering() -> Operator:
    """
    Returns the lowering operator.
    The lowering operator is a 2x2 matrix defined as:
    [[0, 0],
     [1, 0]]

    Returns:
      Operator: A 2x2 matrix of the lowering operator with complex64 precision.
    """
    mat = jnp.array([[0.0, 0.0], [1.0, 0.0]], dtype=jnp.complex64)
    return Operator(mat)


def annihilation(dim: int) -> Operator:
    """
    Returns the annihilation operator.

    Args:
      dim (int): The dimension of Hilbert space.

    Returns:
      Operator: A 'dim'x'dim' matrix of the annihilation operator with complex64 precision.
    """
    mat = jnp.diag(jnp.sqrt(jnp.arange(1, dim, dtype=jnp.complex64)), k=1)
    return Operator(mat)


def creation(dim: int) -> Operator:
    """
    Returns the creation operator

    Args:
      dim (int): The dimension of Hilbert space.

    Returns:
      Operator: A 'dim'x'dim' matrix of the creation operator with complex64 precision.
    """
    mat = jnp.diag(jnp.sqrt(jnp.arange(1, dim, dtype=jnp.complex64)), k=-1)
    return Operator(mat)


def number(dim: int) -> Operator:
    """
    Returns the number operator.

    Args:
      dim (int): The dimension of Hilbert space.

    Returns:
      Operator: A 'dim'x'dim' matrix of the number operator with complex64 precision.
    """
    mat = jnp.diag(jnp.arange(0, dim, dtype=jnp.complex64))
    return Operator(mat)


def position(dim: int) -> Operator:
    """
    Returns the position quadrature operator (dimensionless).
    Defined as x = (a + a_dag) / sqrt(2).

    Args:
        dim (int): The dimension of Hilbert space.

    Returns:
        Operator: A 'dim'x'dim' matrix of the position operator.
    """
    a_dag = creation(dim)
    a = annihilation(dim)
    return (a + a_dag) / jnp.sqrt(2)


def momentum(dim: int) -> Operator:
    """
    Returns the momentum quadrature operator (dimensionless).
    Defined as p = i * (a_dag - a) / sqrt(2) to be Hermitian.

    Args:
        dim (int): The dimension of Hilbert space.

    Returns:
        Operator: A 'dim'x'dim' matrix of the momentum operator.
    """
    a_dag = creation(dim)
    a = annihilation(dim)
    return 1j * (a_dag - a) / jnp.sqrt(2)


def displacement(alpha: complex, dim: int) -> Operator:
    """
    Returns the displacement operator D(alpha).
    This operator displaces a quantum state in phase space,
    generating a coherent state when applied to the vacuum.
    D(alpha) = exp(alpha * a_dag - alpha_conj * a)

    Args:
        alpha (complex): The complex displacement amplitude.
        dim (int): The dimension of Hilbert space.

    Returns:
        Operator: The 'dim'x'dim' displacement operator.
    """
    a_dag = creation(dim)
    a = annihilation(dim)
    arg = alpha * a_dag - jnp.conjugate(alpha) * a
    return Operator(expm(arg.data))


def squeeze(z: complex, dim: int) -> Operator:
    """
    Returns the squeeze operator S(z).
    This operator squeezes a quantum state in phase space,
    generating a squeezed state when applied to the vacuum.
    S(z) = exp(0.5 * (z_conj * a^2 - z * (a_dag)^2))

    Args:
        z (complex): The complex squeeze factor.
        dim (int): The dimension of Hilbert space.

    Returns:
        Operator: The 'dim'x'dim' squeeze operator.
    """
    a_dag = creation(dim)
    a = annihilation(dim)
    arg = (jnp.conjugate(z) / 2) * (a @ a) - (z / 2) * (a_dag @ a_dag)
    return Operator(expm(arg.data))
