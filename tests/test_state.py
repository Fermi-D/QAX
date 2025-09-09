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

"""Tests for qax.state module."""

import pytest
import jax.numpy as jnp
from qax import state
from qax import operators as ops


class TestQubitStates:
    """Test qubit state generation."""

    def test_single_qubit(self):
        """Test single qubit state |0⟩."""
        psi = state.qubit(1)
        assert psi.shape == (2, 1)
        assert jnp.isclose(psi.data[0, 0], 1.0)
        assert jnp.isclose(psi.data[1, 0], 0.0)

    def test_two_qubits(self):
        """Test two-qubit state |00⟩."""
        psi = state.qubit(2)
        assert psi.shape == (4, 1)
        assert jnp.isclose(psi.data[0, 0], 1.0)
        assert jnp.allclose(psi.data[1:, 0], 0.0)

    def test_n_qubits(self):
        """Test n-qubit state normalization."""
        for n in range(1, 5):
            psi = state.qubit(n)
            assert psi.shape == (2**n, 1)
            # Check normalization
            norm = jnp.linalg.norm(psi.data)
            assert jnp.isclose(norm, 1.0)


class TestFockStates:
    """Test Fock state generation."""

    def test_vacuum_state(self):
        """Test vacuum state |0⟩."""
        dim = 10
        psi = state.vacuum(dim)
        assert psi.shape == (dim, 1)
        assert jnp.isclose(psi.data[0, 0], 1.0)
        assert jnp.allclose(psi.data[1:, 0], 0.0)

    def test_fock_state(self):
        """Test Fock state |n⟩."""
        dim = 10
        n = 3
        psi = state.fock(n, dim)
        assert psi.shape == (dim, 1)
        assert jnp.isclose(psi.data[n, 0], 1.0)
        # Check all other elements are zero
        for i in range(dim):
            if i != n:
                assert jnp.isclose(psi.data[i, 0], 0.0)

    def test_fock_state_error(self):
        """Test that Fock state raises error for n >= dim."""
        dim = 5
        with pytest.raises(ValueError):
            state.fock(5, dim)  # n = dim
        with pytest.raises(ValueError):
            state.fock(10, dim)  # n > dim

    def test_fock_state_normalization(self):
        """Test Fock state normalization."""
        dim = 10
        for n in range(dim):
            psi = state.fock(n, dim)
            norm = jnp.linalg.norm(psi.data)
            assert jnp.isclose(norm, 1.0)


class TestCoherentStates:
    """Test coherent state generation."""

    def test_coherent_vacuum(self):
        """Test that coherent state with α=0 is vacuum."""
        dim = 10
        psi = state.coherent(0.0, dim)
        vacuum = state.vacuum(dim)
        assert jnp.allclose(psi.data, vacuum.data, atol=1e-6)

    def test_coherent_normalization(self):
        """Test coherent state normalization."""
        dim = 20
        alphas = [0.5, 1.0 + 0.5j, 2.0j]
        for alpha in alphas:
            psi = state.coherent(alpha, dim)
            norm = jnp.linalg.norm(psi.data)
            assert jnp.isclose(norm, 1.0, atol=1e-5)

    def test_coherent_mean_photon_number(self):
        """Test mean photon number ⟨n⟩ = |α|²."""
        dim = 30
        alpha = 1.5 + 1.0j
        psi = state.coherent(alpha, dim)
        n_op = ops.number(dim)

        # Calculate expectation value ⟨n⟩
        mean_n = jnp.real((psi.dagger() @ n_op @ psi))
        expected_n = jnp.abs(alpha) ** 2
        assert jnp.isclose(
            mean_n, expected_n, rtol=0.1
        )  # 10% tolerance due to truncation


class TestSqueezedStates:
    """Test squeezed state generation."""

    def test_squeezed_vacuum(self):
        """Test that squeezed state with z=0 is vacuum."""
        dim = 10
        psi = state.squeezed(0.0, dim)
        vacuum = state.vacuum(dim)
        assert jnp.allclose(psi.data, vacuum.data, atol=1e-6)

    def test_squeezed_normalization(self):
        """Test squeezed state normalization."""
        dim = 20
        z_values = [0.2, 0.3 + 0.1j, 0.5j]
        for z in z_values:
            psi = state.squeezed(z, dim)
            norm = jnp.linalg.norm(psi.data)
            assert jnp.isclose(norm, 1.0, atol=1e-5)

    def test_squeezed_even_fock_components(self):
        """Test that squeezed vacuum has only even Fock components."""
        dim = 20
        z = 0.3
        psi = state.squeezed(z, dim)

        # Check odd Fock components are near zero
        for n in range(1, dim, 2):  # odd indices
            assert jnp.abs(psi.data[n, 0]) < 1e-10


class TestPositionMomentumStates:
    """Test position and momentum eigenstate generation."""

    def test_position_normalization(self):
        """Test position state normalization."""
        dim = 30
        x_values = [-1.0, 0.0, 1.0, 2.0]
        for x in x_values:
            psi = state.position(x, dim)
            norm = jnp.linalg.norm(psi.data)
            # Position eigenstates are not perfectly normalized in finite dimension
            # but should be close to 1
            assert jnp.isclose(norm, 1.0, rtol=0.1)

    def test_momentum_normalization(self):
        """Test momentum state normalization."""
        dim = 30
        p_values = [-1.0, 0.0, 1.0, 2.0]
        for p in p_values:
            psi = state.momentum(p, dim)
            norm = jnp.linalg.norm(psi.data)
            # Momentum eigenstates are not perfectly normalized in finite dimension
            # but should be close to 1
            assert jnp.isclose(norm, 1.0, rtol=0.1)

    def test_position_expectation(self):
        """Test position expectation value for position eigenstate."""
        dim = 40
        x_test = 0.5
        psi = state.position(x_test, dim)
        x_op = ops.position(dim)

        # Calculate ⟨x⟩
        mean_x = jnp.real((psi.dagger() @ x_op @ psi))
        assert jnp.isclose(mean_x, x_test, atol=0.1)

    def test_momentum_expectation(self):
        """Test momentum expectation value for momentum eigenstate."""
        dim = 40
        p_test = 0.5
        psi = state.momentum(p_test, dim)
        p_op = ops.momentum(dim)

        # Calculate ⟨p⟩
        mean_p = jnp.real((psi.dagger() @ p_op @ psi))
        assert jnp.isclose(mean_p, p_test, atol=0.1)

    def test_position_momentum_fourier(self):
        """Test approximate Fourier transform relation between position and momentum."""
        dim = 30

        # Position state at x=0 should be roughly uniform in momentum space
        psi_x0 = state.position(0.0, dim)

        # Check that amplitude is roughly constant across Fock basis
        # (which approximates momentum basis for x=0)
        amplitudes = jnp.abs(psi_x0.data[: dim // 2, 0])  # Check first half
        mean_amp = jnp.mean(amplitudes)
        std_amp = jnp.std(amplitudes)
        # Relative standard deviation should be small
        assert std_amp / mean_amp < 0.5
