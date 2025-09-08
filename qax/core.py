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
import jax
import jax.numpy as jnp
import numpy as np
import sympy
from typing import Union, Any
from jaxtyping import Array, Complex

from .plotter.sympy_display import _display_resizable_matrix


class Ket:
    """Represents a quantum state ket |ψ⟩ as a column vector."""

    data: Array[Complex, "n 1"]

    def __init__(
        self, array: Union[Array[Complex, "n 1"], Array[Complex, "n"], np.ndarray]
    ) -> None:
        """
        Initializes the Ket vector.
        If a 1D array is passed, it is automatically converted to a column vector.
        """
        if not isinstance(array, jax.Array):
            array = jnp.asarray(array, dtype=jnp.complex64)

        if array.ndim == 1:
            array = array.reshape(-1, 1)  # Ensure it's a column vector

        if not (array.ndim == 2 and array.shape[1] == 1):
            raise ValueError(
                f"Ket vector must be a column (Nx1) vector, but got shape {array.shape}"
            )

        self.data = array

    def info(self) -> None:
        """Displays key information about the ket."""
        print("Type: state vector (ket)")
        print(f"Shape: {self.shape}")
        print(f"Dtype: {self.dtype}")
        print(f"Norm: {jnp.linalg.norm(self.data):.6f}")
        _display_resizable_matrix(sympy.Matrix(np.asarray(jnp.round(self.data, 3))))

    def dagger(self) -> Bra:
        """Returns the Hermitian conjugate (bra) of this ket."""
        return Bra(jnp.conjugate(self.data.T))

    def __matmul__(self, other: Bra) -> Operator:
        """Outer product: Ket @ Bra -> Operator."""
        if isinstance(other, Bra):
            return Operator(self.data @ other.data)
        return NotImplemented

    def __rmatmul__(self, other: Operator) -> Ket:
        """Allows `Operator @ Ket`."""
        if isinstance(other, Operator):
            return Ket(other.data @ self.data)
        return NotImplemented

    def _repr_latex_(self) -> str:
        sympy_matrix = sympy.Matrix(np.asarray(jnp.round(self.data, 3)))
        return f"$${sympy.latex(sympy_matrix, mat_str='pmatrix')}$$"

    def __repr__(self) -> str:
        return f"Ket(shape={self.shape}, dtype={self.dtype})\n{self.data}"

    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape

    @property
    def dtype(self) -> Any:
        return self.data.dtype

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, key: Any) -> Array[Complex, "..."]:
        return self.data[key]

    def __add__(self, other: Ket) -> Ket:
        if isinstance(other, Ket):
            return Ket(self.data + other.data)
        return NotImplemented

    def __sub__(self, other: Ket) -> Ket:
        if isinstance(other, Ket):
            return Ket(self.data - other.data)
        return NotImplemented

    def __mul__(self, scalar: Union[complex, float, Array]) -> Ket:
        return Ket(self.data * scalar)

    def __rmul__(self, scalar: Union[complex, float, Array]) -> Ket:
        return Ket(scalar * self.data)

    def __truediv__(self, scalar: Union[complex, float, Array]) -> Ket:
        return Ket(self.data / scalar)


class Bra:
    """Represents a quantum state bra ⟨ψ| as a row vector."""

    data: Array[Complex, "1 n"]

    def __init__(
        self, array: Union[Array[Complex, "1 n"], Array[Complex, "n"], np.ndarray]
    ) -> None:
        """
        Initializes the Bra.
        If a 1D array is passed, it is automatically converted to a row vector.
        """
        if not isinstance(array, jax.Array):
            array = jnp.asarray(array, dtype=jnp.complex64)
        if array.ndim == 1:
            array = array.reshape(1, -1)  # Ensure it's a row vector
        if not (array.ndim == 2 and array.shape[0] == 1):
            raise ValueError(
                f"Bra vector must be a row (1xN) vector, but got shape {array.shape}"
            )
        self.data = array

    def dagger(self) -> Ket:
        """Returns the Hermitian conjugate (ket) of this bra."""
        return Ket(jnp.conjugate(self.data.T))

    def __matmul__(self, other: Union[Ket, Operator]) -> Union[Array[Complex, ""], Bra]:
        """
        - Bra @ Ket -> Inner product (scalar as a 0-dim JAX array)
        - Bra @ Operator -> new Bra
        """
        if isinstance(other, Ket):
            return (self.data @ other.data)[0, 0]  # Returns a scalar value
        elif isinstance(other, Operator):
            return Bra(self.data @ other.data)
        return NotImplemented

    def info(self) -> None:
        """Displays key information about the bra."""
        print("Type: state vector (bra)")
        print(f"Shape: {self.shape}")
        print(f"Dtype: {self.dtype}")
        print(f"Norm: {jnp.linalg.norm(self.data):.6f}")
        _display_resizable_matrix(sympy.Matrix(np.asarray(jnp.round(self.data, 3))))

    def __repr__(self) -> str:
        return f"Bra(shape={self.shape}, dtype={self.dtype})\n{self.data}"

    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape

    @property
    def dtype(self) -> Any:
        return self.data.dtype

    def __len__(self) -> int:
        return self.data.shape[1]

    def __getitem__(self, key: Any) -> Array[Complex, "..."]:
        return self.data[key]

    def __add__(self, other: Bra) -> Bra:
        if isinstance(other, Bra):
            return Bra(self.data + other.data)
        return NotImplemented

    def __sub__(self, other: Bra) -> Bra:
        if isinstance(other, Bra):
            return Bra(self.data - other.data)
        return NotImplemented

    def __mul__(self, scalar: Union[complex, float, Array]) -> Bra:
        return Bra(self.data * scalar)

    def __rmul__(self, scalar: Union[complex, float, Array]) -> Bra:
        return Bra(scalar * self.data)

    def __truediv__(self, scalar: Union[complex, float, Array]) -> Bra:
        return Bra(self.data / scalar)


class Operator:
    """A wrapper class for a JAX array representing a quantum operator (matrix)."""

    data: Array[Complex, "n n"]

    def __init__(self, array: Union[Array[Complex, "n n"], np.ndarray]) -> None:
        if not isinstance(array, jax.Array):
            array = jnp.asarray(array, dtype=jnp.complex64)
        if array.ndim != 2 or array.shape[0] != array.shape[1]:
            raise ValueError(
                f"Operator must be initialized with a square 2D array, but got shape {array.shape}"
            )
        self.data = array

    def info(self) -> None:
        """
        Displays key information about the operator.
        Displayed values are rounded to 3 decimal places.
        """
        print("Type: operator")
        print(f"Shape: {self.shape}")
        print(f"Dtype: {self.dtype}")
        print(f"Hermitian: {self.is_hermitian()}")

        display_array = jnp.round(self.data, decimals=3)
        sympy_matrix = sympy.Matrix(np.asarray(display_array))
        _display_resizable_matrix(sympy_matrix)

    def _repr_latex_(self) -> str:
        """Rich display in Jupyter, rounded to 3 decimal places."""
        sympy_matrix = sympy.Matrix(np.asarray(jnp.round(self.data, 3)))
        latex_str = sympy.latex(sympy_matrix)
        resizable_str = latex_str.replace(
            r"\begin{bmatrix}", r"\left[ \begin{matrix}"
        ).replace(r"\end{bmatrix}", r"\end{matrix} \right]")
        return f"$${resizable_str}$$"

    def __repr__(self) -> str:
        return f"Operator(dtype={self.dtype}, shape={self.shape})\n{self.data}"

    def dagger(self) -> Operator:
        """Returns the Hermitian conjugate of this operator."""
        return Operator(jnp.conj(jnp.transpose(self.data)))

    def trace(self) -> Array[Complex, ""]:
        """Returns the trace of the operator."""
        return jnp.trace(self.data)

    def is_hermitian(self, tol: float = 1e-6) -> bool:
        """Checks if the operator is Hermitian."""
        return jnp.allclose(self.data, self.dagger().data, atol=tol)

    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape

    @property
    def dtype(self) -> Any:
        return self.data.dtype

    def __add__(self, other: Operator) -> Operator:
        if isinstance(other, Operator):
            return Operator(self.data + other.data)
        return NotImplemented

    def __sub__(self, other: Operator) -> Operator:
        if isinstance(other, Operator):
            return Operator(self.data - other.data)
        return NotImplemented

    def __mul__(self, scalar: Union[complex, float, Array]) -> Operator:
        return Operator(self.data * scalar)

    def __rmul__(self, scalar: Union[complex, float, Array]) -> Operator:
        return Operator(scalar * self.data)

    def __truediv__(self, scalar: Union[complex, float, Array]) -> Operator:
        if scalar == 0:
            raise ZeroDivisionError("Cannot divide an Operator by zero.")
        return Operator(self.data / scalar)

    def __matmul__(self, other: Union[Operator, Ket]) -> Union[Operator, Ket]:
        if isinstance(other, Operator):
            return Operator(self.data @ other.data)
        elif isinstance(other, Ket):
            return Ket(self.data @ other.data)
        return NotImplemented
