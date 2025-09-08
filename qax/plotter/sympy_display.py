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

import sympy
from IPython.display import display, Math


def setup_printing(mat_str: str = "pmatrix") -> None:
    """
    Initializes SymPy's pretty printing for Jupyter.
    'pmatrix' (parentheses) is often preferred for vectors.
    'bmatrix' (square brackets) for operators.

    Args:
        mat_str (str): The matrix environment string for LaTeX printing.
                       e.g., 'pmatrix', 'bmatrix'.
    """
    sympy.init_printing(mat_str=mat_str)


def _display_resizable_matrix(sympy_matrix_obj: sympy.Matrix) -> None:
    """Helper function to display a SymPy matrix with resizable brackets."""
    latex_str = sympy.latex(sympy_matrix_obj)
    if r"\begin{pmatrix}" in latex_str:
        resizable_str = latex_str.replace(
            r"\begin{pmatrix}", r"\left( \begin{matrix}"
        ).replace(r"\end{pmatrix}", r"\end{matrix} \right)")
    elif r"\begin{bmatrix}" in latex_str:
        resizable_str = latex_str.replace(
            r"\begin{bmatrix}", r"\left[ \begin{matrix}"
        ).replace(r"\end{bmatrix}", r"\end{matrix} \right]")
    else:
        resizable_str = latex_str
    display(Math(resizable_str))
