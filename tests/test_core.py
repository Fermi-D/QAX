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

"""Tests for qax.core module."""

import pytest
import numpy as np
import jax.numpy as jnp
from qax.core import Ket, Bra, Operator


class TestKet:
    """Test cases for Ket class."""

    def test_init_from_list(self):
        """Test Ket initialization from list."""
        ket = Ket([1, 0])
        assert ket.shape == (2, 1)
        assert ket.dtype == jnp.complex64

    def test_init_from_numpy(self):
        """Test Ket initialization from numpy array."""
        arr = np.array([1 + 1j, 0])
        ket = Ket(arr)
        assert ket.shape == (2, 1)
        assert jnp.allclose(ket.data[0, 0], 1 + 1j)

    def test_init_column_vector(self):
        """Test that 2D column vector is accepted."""
        arr = jnp.array([[1], [0]])
        ket = Ket(arr)
        assert ket.shape == (2, 1)

    def test_init_invalid_shape(self):
        """Test that invalid shapes raise ValueError."""
        with pytest.raises(ValueError):
            Ket(jnp.array([[1, 2], [3, 4]]))  # 2x2 matrix

    def test_dagger(self):
        """Test dagger operation returns Bra."""
        ket = Ket([1, 0])
        bra = ket.dagger()
        assert isinstance(bra, Bra)
        assert bra.shape == (1, 2)

    def test_addition(self):
        """Test Ket addition."""
        ket1 = Ket([1, 0])
        ket2 = Ket([0, 1])
        ket3 = ket1 + ket2
        assert jnp.allclose(ket3.data, jnp.array([[1], [1]]))

    def test_scalar_multiplication(self):
        """Test scalar multiplication."""
        ket = Ket([1, 0])
        ket2 = 2 * ket
        assert jnp.allclose(ket2.data, jnp.array([[2], [0]]))


class TestBra:
    """Test cases for Bra class."""

    def test_init_from_list(self):
        """Test Bra initialization from list."""
        bra = Bra([1, 0])
        assert bra.shape == (1, 2)
        assert bra.dtype == jnp.complex64

    def test_dagger(self):
        """Test dagger operation returns Ket."""
        bra = Bra([1, 0])
        ket = bra.dagger()
        assert isinstance(ket, Ket)
        assert ket.shape == (2, 1)

    def test_inner_product(self):
        """Test Bra @ Ket inner product."""
        bra = Bra([1, 0])
        ket = Ket([1, 0])
        result = bra @ ket
        assert jnp.isclose(result, 1.0)

    def test_bra_operator_product(self):
        """Test Bra @ Operator."""
        bra = Bra([1, 0])
        op = Operator([[1, 0], [0, -1]])
        result = bra @ op
        assert isinstance(result, Bra)
        assert jnp.allclose(result.data, jnp.array([[1, 0]]))


class TestOperator:
    """Test cases for Operator class."""

    def test_init_from_list(self):
        """Test Operator initialization from list."""
        op = Operator([[1, 0], [0, 1]])
        assert op.shape == (2, 2)
        assert op.dtype == jnp.complex64

    def test_init_invalid_shape(self):
        """Test that non-square matrices raise ValueError."""
        with pytest.raises(ValueError):
            Operator([[1, 2, 3], [4, 5, 6]])  # 2x3 matrix

    def test_dagger(self):
        """Test Hermitian conjugate."""
        op = Operator([[1, 1j], [-1j, 1]])
        op_dag = op.dagger()
        expected = jnp.array([[1, 1j], [-1j, 1]])
        assert jnp.allclose(op_dag.data, expected)

    def test_is_hermitian(self):
        """Test Hermitian check."""
        # Pauli X is Hermitian
        pauli_x = Operator([[0, 1], [1, 0]])
        assert pauli_x.is_hermitian()

        # Non-Hermitian matrix
        non_herm = Operator([[1, 2], [3, 4]])
        assert not non_herm.is_hermitian()

    def test_trace(self):
        """Test trace calculation."""
        op = Operator([[2, 0], [0, 3]])
        assert jnp.isclose(op.trace(), 5.0)

    def test_operator_ket_product(self):
        """Test Operator @ Ket."""
        op = Operator([[0, 1], [1, 0]])  # Pauli X
        ket = Ket([1, 0])
        result = op @ ket
        assert isinstance(result, Ket)
        assert jnp.allclose(result.data, jnp.array([[0], [1]]))

    def test_operator_operator_product(self):
        """Test Operator @ Operator."""
        op1 = Operator([[0, 1], [1, 0]])  # Pauli X
        op2 = Operator([[0, -1j], [1j, 0]])  # Pauli Y
        result = op1 @ op2
        assert isinstance(result, Operator)
        expected = jnp.array([[1j, 0], [0, -1j]])  # i * Pauli Z
        assert jnp.allclose(result.data, expected)


class TestQuantumOperations:
    """Test quantum mechanical operations between classes."""

    def test_outer_product(self):
        """Test Ket @ Bra outer product."""
        ket = Ket([1, 0])
        bra = Bra([1, 0])
        op = ket @ bra
        assert isinstance(op, Operator)
        assert jnp.allclose(op.data, jnp.array([[1, 0], [0, 0]]))

    def test_expectation_value_workflow(self):
        """Test expectation value calculation workflow."""
        # |ψ⟩ = |0⟩
        ket = Ket([1, 0])
        # H = σ_z
        op = Operator([[1, 0], [0, -1]])
        # ⟨ψ|H|ψ⟩
        bra = ket.dagger()
        temp = op @ ket
        expectation = bra @ temp
        assert jnp.isclose(expectation, 1.0)
