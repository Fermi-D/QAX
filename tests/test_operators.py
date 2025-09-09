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

"""Tests for qax.operators module."""

import jax.numpy as jnp
from qax import operators


class TestPauliOperators:
    """Test Pauli operators properties."""

    def test_pauli_x_matrix(self):
        """Test Pauli X matrix values."""
        pauli_x = operators.pauli_x()
        expected = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex64)
        assert jnp.allclose(pauli_x.data, expected)

    def test_pauli_y_matrix(self):
        """Test Pauli Y matrix values."""
        pauli_y = operators.pauli_y()
        expected = jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex64)
        assert jnp.allclose(pauli_y.data, expected)

    def test_pauli_z_matrix(self):
        """Test Pauli Z matrix values."""
        pauli_z = operators.pauli_z()
        expected = jnp.array([[1, 0], [0, -1]], dtype=jnp.complex64)
        assert jnp.allclose(pauli_z.data, expected)

    def test_pauli_hermitian(self):
        """Test that all Pauli matrices are Hermitian."""
        assert operators.pauli_x().is_hermitian()
        assert operators.pauli_y().is_hermitian()
        assert operators.pauli_z().is_hermitian()

    def test_pauli_anticommutation(self):
        """Test Pauli anticommutation relations: {σ_i, σ_j} = 2δ_ij I."""
        pauli_x = operators.pauli_x()
        pauli_y = operators.pauli_y()
        pauli_z = operators.pauli_z()
        identity = operators.identity(2)

        # {σ_x, σ_y} = 0
        anticomm_xy = pauli_x @ pauli_y + pauli_y @ pauli_x
        assert jnp.allclose(anticomm_xy.data, jnp.zeros((2, 2)))

        # {σ_x, σ_z} = 0
        anticomm_xz = pauli_x @ pauli_z + pauli_z @ pauli_x
        assert jnp.allclose(anticomm_xz.data, jnp.zeros((2, 2)))

        # {σ_y, σ_z} = 0
        anticomm_yz = pauli_y @ pauli_z + pauli_z @ pauli_y
        assert jnp.allclose(anticomm_yz.data, jnp.zeros((2, 2)))

        # {σ_x, σ_x} = 2I
        anticomm_xx = pauli_x @ pauli_x + pauli_x @ pauli_x
        assert jnp.allclose(anticomm_xx.data, 2 * identity.data)

        # {σ_y, σ_y} = 2I
        anticomm_yy = pauli_y @ pauli_y + pauli_y @ pauli_y
        assert jnp.allclose(anticomm_yy.data, 2 * identity.data)

        # {σ_z, σ_z} = 2I
        anticomm_zz = pauli_z @ pauli_z + pauli_z @ pauli_z
        assert jnp.allclose(anticomm_zz.data, 2 * identity.data)

    def test_pauli_commutation(self):
        """Test Pauli commutation relations: [σ_i, σ_j] = 2iε_ijk σ_k."""
        pauli_x = operators.pauli_x()
        pauli_y = operators.pauli_y()
        pauli_z = operators.pauli_z()

        # [σ_x, σ_y] = 2iσ_z
        comm_xy = pauli_x @ pauli_y - pauli_y @ pauli_x
        expected = 2j * pauli_z
        assert jnp.allclose(comm_xy.data, expected.data)


class TestSpinOperators:
    """Test spin raising and lowering operators."""

    def test_raising_operator(self):
        """Test raising operator σ_+."""
        sigma_plus = operators.raising()
        expected = jnp.array([[0, 1], [0, 0]], dtype=jnp.complex64)
        assert jnp.allclose(sigma_plus.data, expected)

    def test_lowering_operator(self):
        """Test lowering operator σ_-."""
        sigma_minus = operators.lowering()
        expected = jnp.array([[0, 0], [1, 0]], dtype=jnp.complex64)
        assert jnp.allclose(sigma_minus.data, expected)

    def test_raising_lowering_relation(self):
        """Test σ_+ = (σ_x + iσ_y)/2 and σ_- = (σ_x - iσ_y)/2."""
        sigma_x = operators.pauli_x()
        sigma_y = operators.pauli_y()
        sigma_plus = operators.raising()
        sigma_minus = operators.lowering()

        expected_plus = (sigma_x + 1j * sigma_y) / 2
        expected_minus = (sigma_x - 1j * sigma_y) / 2

        assert jnp.allclose(sigma_plus.data, expected_plus.data)
        assert jnp.allclose(sigma_minus.data, expected_minus.data)


class TestBosonicOperators:
    """Test bosonic creation and annihilation operators."""

    def test_annihilation_operator(self):
        """Test annihilation operator matrix elements."""
        dim = 5
        a = operators.annihilation(dim)

        # Check a|n⟩ = √n|n-1⟩
        for n in range(1, dim):
            expected = jnp.sqrt(n)
            assert jnp.isclose(a.data[n - 1, n], expected)

    def test_creation_operator(self):
        """Test creation operator matrix elements."""
        dim = 5
        a_dag = operators.creation(dim)

        # Check a†|n⟩ = √(n+1)|n+1⟩
        for n in range(dim - 1):
            expected = jnp.sqrt(n + 1)
            assert jnp.isclose(a_dag.data[n + 1, n], expected)

    def test_creation_annihilation_dagger(self):
        """Test that a† = (a)†."""
        dim = 5
        a = operators.annihilation(dim)
        a_dag = operators.creation(dim)
        assert jnp.allclose(a_dag.data, a.dagger().data)

    def test_number_operator(self):
        """Test number operator n = a†a."""
        dim = 5
        n = operators.number(dim)
        a = operators.annihilation(dim)
        a_dag = operators.creation(dim)

        n_computed = a_dag @ a
        assert jnp.allclose(n.data, n_computed.data, atol=1e-6)

    def test_commutation_relation(self):
        """Test [a, a†] = I (with truncation effects at the boundary)."""
        dim = 5
        a = operators.annihilation(dim)
        a_dag = operators.creation(dim)
        expected = jnp.eye(dim, dtype=jnp.complex64)
        expected = expected.at[dim - 1, dim - 1].set(1 - dim)

        commutator = a @ a_dag - a_dag @ a
        assert jnp.allclose(commutator.data, expected, atol=1e-6)


class TestQuadratureOperators:
    """Test position and momentum quadrature operators."""

    def test_position_hermitian(self):
        """Test that position operator is Hermitian."""
        dim = 10
        x = operators.position(dim)
        assert x.is_hermitian()

    def test_momentum_hermitian(self):
        """Test that momentum operator is Hermitian."""
        dim = 10
        p = operators.momentum(dim)
        assert p.is_hermitian()

    def test_quadrature_commutation(self):
        """Test [x, p] = i."""
        dim = 20  # Use larger dimension for better approximation
        x = operators.position(dim)
        p = operators.momentum(dim)

        commutator = x @ p - p @ x
        expected = 1j * jnp.eye(dim, dtype=jnp.complex64)
        expected = expected.at[dim - 1, dim - 1].set(1j * (1 - dim))
        assert jnp.allclose(commutator.data, expected, atol=1e-4)


class TestDisplacementOperator:
    """Test displacement operator."""

    def test_displacement_unitary(self):
        """Test that D(α) is unitary."""
        alpha = 1.0 + 0.5j
        dim = 10
        D = operators.displacement(alpha, dim)
        D_dag = D.dagger()
        identity = operators.identity(dim)

        # D†D = I
        assert jnp.allclose((D_dag @ D).data, identity.data, atol=1e-4)
        # DD† = I
        assert jnp.allclose((D @ D_dag).data, identity.data, atol=1e-4)

    def test_displacement_identity(self):
        """Test D(0) = I."""
        dim = 10
        D = operators.displacement(0, dim)
        identity = operators.identity(dim)
        assert jnp.allclose(D.data, identity.data)

    def test_displacement_inverse(self):
        """Test D(-α) = D†(α)."""
        alpha = 0.5 + 0.3j
        dim = 10
        D_alpha = operators.displacement(alpha, dim)
        D_minus_alpha = operators.displacement(-alpha, dim)
        assert jnp.allclose(D_minus_alpha.data, D_alpha.dagger().data, atol=1e-5)


class TestSqueezeOperator:
    """Test squeeze operator."""

    def test_squeeze_unitary(self):
        """Test that S(z) is unitary."""
        z = 0.5 + 0.2j
        dim = 10
        S = operators.squeeze(z, dim)
        S_dag = S.dagger()
        identity = operators.identity(dim)

        # S†S = I
        assert jnp.allclose((S_dag @ S).data, identity.data, atol=1e-4)
        # SS† = I
        assert jnp.allclose((S @ S_dag).data, identity.data, atol=1e-4)

    def test_squeeze_identity(self):
        """Test S(0) = I."""
        dim = 10
        S = operators.squeeze(0, dim)
        identity = operators.identity(dim)
        assert jnp.allclose(S.data, identity.data)

    def test_squeeze_inverse(self):
        """Test S(-z) = S†(z)."""
        z = 0.3 + 0.1j
        dim = 10
        S_z = operators.squeeze(z, dim)
        S_minus_z = operators.squeeze(-z, dim)
        assert jnp.allclose(S_minus_z.data, S_z.dagger().data, atol=1e-5)
