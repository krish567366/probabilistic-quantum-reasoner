# Getting Started

## Installation

The Probabilistic Quantum Reasoner can be installed via pip with optional dependencies for different quantum backends.

### Basic Installation

For classical simulation only:

```bash
pip install probabilistic-quantum-reasoner
```

### Quantum Backend Support

To use IBM Quantum hardware via Qiskit:

```bash
pip install probabilistic-quantum-reasoner[qiskit]
```

To use PennyLane for variational quantum algorithms:

```bash
pip install probabilistic-quantum-reasoner[pennylane]
```

For all quantum backends:

```bash
pip install probabilistic-quantum-reasoner[quantum]
```

### Development Installation

For development and contributing:

```bash
pip install probabilistic-quantum-reasoner[dev]
```

This includes testing, documentation, and code quality tools.

## System Requirements

- Python 3.10 or higher
- NumPy 1.21+
- SciPy 1.7+
- NetworkX 2.6+

### Optional Requirements

- Qiskit 0.45+ (for IBM Quantum backend)
- PennyLane 0.30+ (for variational algorithms)
- Matplotlib 3.5+ (for visualization)

## Verification

Verify your installation:

```python
import probabilistic_quantum_reasoner as pqr
print(f"Version: {pqr.__version__}")

# Test basic functionality
from probabilistic_quantum_reasoner.backends import ClassicalSimulator
backend = ClassicalSimulator()
print("Classical simulator ready!")

# Test quantum backends (if installed)
try:
    from probabilistic_quantum_reasoner.backends import QiskitBackend
    qiskit_backend = QiskitBackend()
    print("Qiskit backend available!")
except ImportError:
    print("Qiskit backend not installed")

try:
    from probabilistic_quantum_reasoner.backends import PennyLaneBackend
    pennylane_backend = PennyLaneBackend()
    print("PennyLane backend available!")
except ImportError:
    print("PennyLane backend not installed")
```

## Next Steps

- Continue with the [Quick Start Guide](quickstart.md)
- Explore [Basic Examples](examples.md)
- Read the [Architecture Overview](../architecture/overview.md)
