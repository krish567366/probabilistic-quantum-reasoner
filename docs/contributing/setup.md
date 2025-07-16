# Contributing to Probabilistic Quantum Reasoner

We welcome contributions from the quantum computing and AI communities! This guide will help you get started.

## Development Setup

### Prerequisites

- Python 3.10 or higher
- Git
- Virtual environment tool (venv, conda, etc.)

### Clone and Setup

```bash
# Clone the repository
git clone https://github.com/quantum-ai/probabilistic-quantum-reasoner.git
cd probabilistic-quantum-reasoner

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Verify Installation

```bash
# Run tests
pytest

# Check code formatting
black --check .
pylint probabilistic_quantum_reasoner

# Type checking
mypy probabilistic_quantum_reasoner
```

## Development Workflow

### 1. Create Feature Branch

```bash
git checkout -b feature/quantum-causal-discovery
```

### 2. Make Changes

Follow our coding standards:

- **Type hints**: All functions must have type annotations
- **Docstrings**: Google-style docstrings for all public APIs
- **Tests**: Comprehensive test coverage for new features
- **Black formatting**: Automatic code formatting

### 3. Run Tests

```bash
# Run all tests
pytest

# Run specific test files
pytest tests/test_operators.py

# Run with coverage
pytest --cov=probabilistic_quantum_reasoner --cov-report=html

# Run quantum-specific tests (requires quantum backends)
pytest -m quantum
```

### 4. Check Code Quality

```bash
# Format code
black .
isort .

# Lint code
pylint probabilistic_quantum_reasoner

# Type check
mypy probabilistic_quantum_reasoner
```

### 5. Commit Changes

```bash
git add .
git commit -m "feat: add quantum causal discovery algorithm"
```

We use conventional commits format:

- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation
- `test:` for tests
- `refactor:` for refactoring

### 6. Submit Pull Request

- Push to your fork
- Create pull request with description
- Ensure CI passes
- Request review

## Coding Standards

### Code Style

We use Black for code formatting with these settings:

```toml
[tool.black]
line-length = 88
target-version = ['py310']
include = '\.pyi?$'
```

### Type Annotations

All functions must have complete type annotations:

```python
from typing import Dict, List, Optional, Union
import numpy as np

def quantum_inference(
    network: QuantumBayesianNetwork,
    evidence: Dict[str, str],
    query_nodes: List[str]
) -> InferenceResult:
    """Perform quantum inference on Bayesian network."""
    pass
```

### Docstring Format

Use Google-style docstrings:

```python
def add_quantum_node(
    self,
    node_id: str,
    outcome_space: List[str],
    initial_amplitudes: Optional[np.ndarray] = None,
    name: Optional[str] = None
) -> QuantumNode:
    """Add a quantum node to the network.
    
    Args:
        node_id: Unique identifier for the node.
        outcome_space: List of possible outcomes.
        initial_amplitudes: Complex amplitudes for quantum superposition.
        name: Human-readable name for the node.
        
    Returns:
        The created quantum node.
        
    Raises:
        ValueError: If node_id already exists.
        QuantumStateError: If amplitudes are not normalized.
        
    Example:
        >>> network = QuantumBayesianNetwork("test", backend)
        >>> node = network.add_quantum_node(
        ...     "weather", 
        ...     ["sunny", "rainy"],
        ...     np.array([0.8, 0.6], dtype=complex)
        ... )
    """
```

### Error Handling

Use custom exceptions for quantum-specific errors:

```python
from probabilistic_quantum_reasoner.core.exceptions import (
    QuantumStateError,
    EntanglementError,
    BackendError
)

def normalize_quantum_state(amplitudes: np.ndarray) -> np.ndarray:
    """Normalize quantum state amplitudes."""
    norm_squared = np.sum(np.abs(amplitudes) ** 2)
    
    if norm_squared == 0:
        raise QuantumStateError("Cannot normalize zero state")
        
    return amplitudes / np.sqrt(norm_squared)
```

## Testing Guidelines

### Test Structure

Organize tests by component:

```bash
tests/
├── conftest.py              # Test fixtures and utilities
├── test_operators.py        # Quantum operator tests
├── test_nodes.py           # Node type tests
├── test_network.py         # Network functionality tests
├── test_inference.py       # Inference algorithm tests
├── test_backends.py        # Backend tests
└── integration/            # Integration tests
    ├── test_examples.py    # Example tests
    └── test_workflows.py   # End-to-end tests
```

### Quantum Test Utilities

Use our quantum testing utilities:

```python
from tests.conftest import (
    assert_quantum_state_equal,
    assert_probability_distribution_valid,
    QuantumTestUtils
)

def test_quantum_gate():
    """Test quantum gate application."""
    gate = QuantumGate.hadamard()
    state = np.array([1, 0], dtype=complex)
    result = gate.apply(state)
    
    expected = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
    assert_quantum_state_equal(result, expected)
```

### Test Markers

Use pytest markers to categorize tests:

```python
import pytest

@pytest.mark.quantum
def test_with_quantum_backend():
    """Test requiring quantum backend."""
    pass

@pytest.mark.slow
def test_expensive_computation():
    """Long-running test."""
    pass

@pytest.mark.integration
def test_end_to_end_workflow():
    """Integration test."""
    pass
```

### Parametrized Tests

Use parametrization for comprehensive testing:

```python
@pytest.mark.parametrize("angle", [0, np.pi/4, np.pi/2, np.pi])
def test_rotation_gate(angle):
    """Test rotation gate with different angles."""
    gate = QuantumGate.rotation_x(angle)
    # Test implementation
```

## Documentation

### Building Documentation

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build documentation
mkdocs serve

# Build for production
mkdocs build
```

### Documentation Standards

- Use MkDocs with Material theme
- Include code examples in docstrings
- Add mathematical notation with MathJax
- Create tutorials for new features

### Adding Examples

When adding new examples:

1. Create example file in `probabilistic_quantum_reasoner/examples/`
2. Add comprehensive docstrings and comments
3. Create documentation page in `docs/examples/`
4. Add tests in `tests/integration/test_examples.py`

## Performance Considerations

### Classical Simulation Optimization

- Use NumPy vectorized operations
- Minimize state vector copying
- Implement lazy evaluation where possible

```python
# Good: Vectorized operation
amplitudes = np.array([...], dtype=complex)
probabilities = np.abs(amplitudes) ** 2

# Avoid: Element-wise loops
probabilities = [abs(amp) ** 2 for amp in amplitudes]
```

### Quantum Backend Efficiency

- Minimize quantum circuit depth
- Batch quantum operations
- Use approximate algorithms for large networks

### Memory Management

- Clean up large quantum states
- Use sparse representations when appropriate
- Implement gradient checkpointing for large networks

## Release Process

### Version Management

We use semantic versioning (semver):
- Major: Breaking API changes
- Minor: New features, backwards compatible
- Patch: Bug fixes

### Release Checklist

1. Update version in `pyproject.toml` and `setup.py`
2. Update `CHANGELOG.md`
3. Run full test suite
4. Build documentation
5. Create release tag
6. Deploy to PyPI

### Continuous Integration

Our CI pipeline includes:
- Multi-platform testing (Linux, macOS, Windows)
- Multiple Python versions (3.10, 3.11, 3.12)
- Code quality checks
- Documentation building
- Security scanning

## Getting Help

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and discussions
- **Documentation**: Comprehensive guides and API reference
- **Email**: quantum-reasoner@example.com for sensitive issues

## Recognition

Contributors are recognized in:
- `CONTRIBUTORS.md` file
- Release notes
- Documentation acknowledgments
- Conference presentations

Thank you for contributing to the quantum AI community! 🚀
