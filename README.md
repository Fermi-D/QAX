# QAX

Contributors : [R. Maekura](https://github.com/Fermi-D)

![python version](https://img.shields.io/badge/python-3.11%2B-purple) [![license: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-purple)](https://github.com/Fermi-D/QAX/blob/main/LICENSE)

## What is QAX?

QAX is a high-performance library for simulating quantum systems, including time evolution, quantum control, and quantum circuits. Built on top of JAX, it leverages automatic differentiation and just-in-time compilation for efficient multi-GPU computations.

### Key Features
- 🚀 **High Performance**: Multi-GPU support with JAX backend
- 🔄 **Time Evolution**: Efficient quantum dynamics simulation
- 🎛️ **Quantum Control**: Optimal control algorithms for quantum systems
- 🔌 **Circuit Simulation**: Gate-based quantum circuit emulation
- 🔬 **Differentiable**: Full support for automatic differentiation
- 📊 **Visualization**: Built-in plotting utilities for quantum states and dynamics

> [!WARNING]
> This library is under active development and some APIs and solvers are still finding their footing. While most of the library is stable, new releases might introduce breaking changes.

## Installation

### From Source (Development)
```shell
git clone https://github.com/Fermi-D/QAX.git
cd QAX
uv venv
uv sync --dev
```

> [!Note]
> Installation via the `pip` command will be supported in a future update.

### Requirements
- Python >= 3.11
- JAX >= 0.4.20
- NumPy >= 1.24.0

For GPU support:
```shell
uv add jax[cuda12]  # For CUDA 12.x
```

## Quick Start

```python
import qax
import jax.numpy as jnp

# Example: Quantum state evolution
# (Code examples will be added as the library develops)
```

## Tutorial

Comprehensive tutorials are available in the `examples/` directory:
- Basic quantum state manipulation
- Quantum dynamics simulations
- Quantum control optimization
- Circuit construction and simulation

## Documentation

Full documentation is available at [GitHub Pages](https://fermi-d.github.io/QAX/) (coming soon).

## Comparison to Other Libraries
<!--
| Feature | QAX | Qiskit | QuTiP | PennyLane | Dynamiqs | CUDA-Q |
|---------|-----|---------|-------|-----------|----------|---------|
| JAX Backend | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ |
| Multi-GPU | ✅ | Limited | ❌ | Limited | ✅ | ✅ |
| Auto-diff | ✅ | Limited | ❌ | ✅ | ✅ | ❌ |
| Time Evolution | ✅ | Limited | ✅ | Limited | ✅ | Limited |
| Quantum Control | ✅ | ❌ | ✅ | Limited | ✅ | ❌ |
| Circuit Simulation | ✅ | ✅ | Limited | ✅ | Limited | ✅ |
| Open Systems | ✅ | Limited | ✅ | ❌ | ✅ | ❌ |
| GPU Acceleration | Native | Limited | ❌ | Limited | Native | Native |
| C++ Backend | ❌ | ✅ | Partial | ❌ | ❌ | ✅ |
-->
## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## Citation

If you use QAX in your research, please cite:
```bibtex
@software{qax2025,
  author = {Maekura, R.},
  title = {QAX: High-Performance Quantum System Simulation with JAX},
  year = {2025},
  url = {https://github.com/Fermi-D/QAX}
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This project was developed as part of doctoral research in quantum computing.
