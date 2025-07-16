# Testing Guidelines

This document outlines the testing strategy and guidelines for the Probabilistic Quantum Reasoner project.

## Testing Philosophy

We follow a comprehensive testing approach that includes:

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test interactions between components
- **End-to-End Tests**: Test complete workflows
- **Performance Tests**: Benchmark and validate performance
- **Quantum Tests**: Validate quantum-specific functionality

## Test Structure

### Test Organization

```
tests/
├── conftest.py                    # Shared fixtures and configuration
├── unit/                          # Unit tests
│   ├── test_core/
│   │   ├── test_reasoner.py
│   │   ├── test_networks.py
│   │   └── test_nodes.py
│   ├── test_backends/
│   │   ├── test_classical.py
│   │   ├── test_qiskit.py
│   │   └── test_pennylane.py
│   └── test_inference/
│       ├── test_belief_propagation.py
│       └── test_variational.py
├── integration/                   # Integration tests
│   ├── test_end_to_end.py
│   ├── test_backend_switching.py
│   └── test_causal_workflows.py
├── performance/                   # Performance tests
│   ├── test_benchmarks.py
│   └── test_scalability.py
└── quantum/                       # Quantum-specific tests
    ├── test_quantum_networks.py
    ├── test_entanglement.py
    └── test_quantum_algorithms.py
```

## Writing Tests

### Test Fixtures

Use pytest fixtures for reusable test components:

```python
# tests/conftest.py
import pytest
import numpy as np
from probabilistic_quantum_reasoner import ProbabilisticQuantumReasoner
from probabilistic_quantum_reasoner.networks import BayesianNetwork
from probabilistic_quantum_reasoner.nodes import DiscreteNode, QuantumNode

@pytest.fixture
def simple_classical_network():
    """Create a simple classical Bayesian network for testing."""
    network = BayesianNetwork(name="Simple Test Network")
    
    # Weather node
    weather = DiscreteNode(
        name="Weather",
        states=["sunny", "rainy"],
        prior=[0.7, 0.3]
    )
    
    # Mood node (depends on weather)
    mood = DiscreteNode(
        name="Mood",
        states=["happy", "sad"],
        parents=[weather],
        cpt=np.array([
            [0.8, 0.2],  # sunny -> happy/sad
            [0.3, 0.7]   # rainy -> happy/sad
        ])
    )
    
    network.add_nodes([weather, mood])
    return network

@pytest.fixture
def simple_quantum_network():
    """Create a simple quantum network for testing."""
    network = BayesianNetwork(name="Simple Quantum Network")
    
    # Quantum node in superposition
    qubit = QuantumNode(
        name="Qubit",
        num_qubits=1,
        initial_state="superposition"
    )
    
    network.add_node(qubit)
    return network

@pytest.fixture
def classical_reasoner():
    """Create a classical reasoner for testing."""
    return ProbabilisticQuantumReasoner(backend="classical")

@pytest.fixture(params=["classical", "qiskit"])
def reasoner(request):
    """Parametrized reasoner fixture for testing multiple backends."""
    if request.param == "qiskit":
        pytest.importorskip("qiskit")
    
    return ProbabilisticQuantumReasoner(backend=request.param)

@pytest.fixture
def sample_evidence():
    """Sample evidence for testing."""
    return {"Weather": "sunny"}

@pytest.fixture
def sample_query():
    """Sample query for testing."""
    return ["Mood"]
```

### Unit Test Examples

#### Testing Core Components

```python
# tests/unit/test_core/test_networks.py
import pytest
import numpy as np
from probabilistic_quantum_reasoner.networks import BayesianNetwork
from probabilistic_quantum_reasoner.nodes import DiscreteNode

class TestBayesianNetwork:
    """Test suite for BayesianNetwork class."""
    
    def test_network_creation(self):
        """Test basic network creation."""
        network = BayesianNetwork(name="Test Network")
        assert network.name == "Test Network"
        assert len(network.nodes) == 0
        assert len(network.edges) == 0
    
    def test_add_single_node(self):
        """Test adding a single node to network."""
        network = BayesianNetwork()
        
        node = DiscreteNode(
            name="TestNode",
            states=["state1", "state2"],
            prior=[0.6, 0.4]
        )
        
        network.add_node(node)
        
        assert len(network.nodes) == 1
        assert "TestNode" in network.nodes
        assert network.nodes["TestNode"] == node
    
    def test_add_multiple_nodes(self):
        """Test adding multiple nodes at once."""
        network = BayesianNetwork()
        
        node1 = DiscreteNode(name="Node1", states=["a", "b"], prior=[0.5, 0.5])
        node2 = DiscreteNode(name="Node2", states=["x", "y"], prior=[0.3, 0.7])
        
        network.add_nodes([node1, node2])
        
        assert len(network.nodes) == 2
        assert "Node1" in network.nodes
        assert "Node2" in network.nodes
    
    def test_node_dependencies(self, simple_classical_network):
        """Test that node dependencies are correctly established."""
        network = simple_classical_network
        
        weather_node = network.nodes["Weather"]
        mood_node = network.nodes["Mood"]
        
        assert len(mood_node.parents) == 1
        assert mood_node.parents[0] == weather_node
        assert len(network.edges) == 1
    
    def test_invalid_node_addition(self):
        """Test that invalid node additions raise appropriate errors."""
        network = BayesianNetwork()
        
        # Add initial node
        node1 = DiscreteNode(name="Node1", states=["a", "b"], prior=[0.5, 0.5])
        network.add_node(node1)
        
        # Try to add node with same name
        node2 = DiscreteNode(name="Node1", states=["x", "y"], prior=[0.3, 0.7])
        
        with pytest.raises(ValueError, match="already exists"):
            network.add_node(node2)
    
    def test_network_validation(self, simple_classical_network):
        """Test network validation."""
        network = simple_classical_network
        
        # Valid network should pass validation
        assert network.validate()
        
        # Create invalid network (circular dependency)
        weather = network.nodes["Weather"]
        mood = network.nodes["Mood"]
        
        # This would create a cycle: Weather -> Mood -> Weather
        weather.parents = [mood]
        
        with pytest.raises(ValueError, match="circular"):
            network.validate()
```

#### Testing Inference Algorithms

```python
# tests/unit/test_inference/test_belief_propagation.py
import pytest
import numpy as np
from probabilistic_quantum_reasoner.inference import BeliefPropagation

class TestBeliefPropagation:
    """Test suite for Belief Propagation algorithm."""
    
    def test_exact_inference(self, simple_classical_network, classical_reasoner):
        """Test exact inference on simple network."""
        network = simple_classical_network
        
        # Test inference without evidence
        result = classical_reasoner.infer(
            network=network,
            query=["Mood"],
            evidence={}
        )
        
        # Check result structure
        assert "Mood" in result
        assert "happy" in result["Mood"]
        assert "sad" in result["Mood"]
        
        # Check probabilities sum to 1
        prob_sum = sum(result["Mood"].values())
        assert abs(prob_sum - 1.0) < 1e-10
        
        # Check that probabilities are positive
        for prob in result["Mood"].values():
            assert prob >= 0
    
    def test_inference_with_evidence(self, simple_classical_network, classical_reasoner):
        """Test inference with evidence."""
        network = simple_classical_network
        
        # Test with sunny weather evidence
        result_sunny = classical_reasoner.infer(
            network=network,
            query=["Mood"],
            evidence={"Weather": "sunny"}
        )
        
        # Test with rainy weather evidence
        result_rainy = classical_reasoner.infer(
            network=network,
            query=["Mood"],
            evidence={"Weather": "rainy"}
        )
        
        # With sunny weather, should be more likely to be happy
        assert result_sunny["Mood"]["happy"] > result_rainy["Mood"]["happy"]
        assert result_sunny["Mood"]["sad"] < result_rainy["Mood"]["sad"]
    
    def test_multiple_queries(self, simple_classical_network, classical_reasoner):
        """Test inference with multiple query variables."""
        network = simple_classical_network
        
        result = classical_reasoner.infer(
            network=network,
            query=["Weather", "Mood"],
            evidence={}
        )
        
        assert "Weather" in result
        assert "Mood" in result
        
        # Check that each variable's probabilities sum to 1
        for var in ["Weather", "Mood"]:
            prob_sum = sum(result[var].values())
            assert abs(prob_sum - 1.0) < 1e-10
    
    def test_convergence(self):
        """Test that belief propagation converges."""
        bp = BeliefPropagation(max_iterations=100, tolerance=1e-6)
        
        # Test with a more complex network
        network = self._create_complex_network()
        
        result = bp.infer(
            network=network,
            query=["Target"],
            evidence={}
        )
        
        # Should converge and return valid probabilities
        assert isinstance(result, dict)
        assert "Target" in result
    
    def _create_complex_network(self):
        """Helper method to create a more complex test network."""
        from probabilistic_quantum_reasoner.networks import BayesianNetwork
        from probabilistic_quantum_reasoner.nodes import DiscreteNode
        
        network = BayesianNetwork(name="Complex Test Network")
        
        # Create chain of dependencies: A -> B -> C -> D
        node_a = DiscreteNode(name="A", states=["0", "1"], prior=[0.5, 0.5])
        
        node_b = DiscreteNode(
            name="B", 
            states=["0", "1"],
            parents=[node_a],
            cpt=np.array([[0.8, 0.2], [0.3, 0.7]])
        )
        
        node_c = DiscreteNode(
            name="C",
            states=["0", "1"], 
            parents=[node_b],
            cpt=np.array([[0.9, 0.1], [0.2, 0.8]])
        )
        
        node_d = DiscreteNode(
            name="Target",
            states=["0", "1"],
            parents=[node_c],
            cpt=np.array([[0.7, 0.3], [0.4, 0.6]])
        )
        
        network.add_nodes([node_a, node_b, node_c, node_d])
        return network
```

#### Testing Quantum Components

```python
# tests/quantum/test_quantum_networks.py
import pytest
import numpy as np
from probabilistic_quantum_reasoner.nodes import QuantumNode
from probabilistic_quantum_reasoner.quantum_ops import HadamardGate, CNOTGate

@pytest.mark.quantum
class TestQuantumNetworks:
    """Test suite for quantum network functionality."""
    
    @pytest.fixture
    def quantum_reasoner(self):
        """Create quantum reasoner (skip if backends unavailable)."""
        qiskit = pytest.importorskip("qiskit")
        from probabilistic_quantum_reasoner import ProbabilisticQuantumReasoner
        return ProbabilisticQuantumReasoner(backend="qiskit")
    
    def test_single_qubit_operations(self, quantum_reasoner):
        """Test single qubit quantum operations."""
        from probabilistic_quantum_reasoner.networks import BayesianNetwork
        
        network = BayesianNetwork(name="Single Qubit Test")
        
        # Create qubit in |0⟩ state
        qubit = QuantumNode(
            name="Qubit",
            num_qubits=1,
            initial_state="zero"
        )
        
        network.add_node(qubit)
        
        # Measure without any operations - should always be |0⟩
        measurements = []
        for _ in range(100):
            result = quantum_reasoner.measure(network, ["Qubit"])
            measurements.append(result["Qubit"])
        
        # All measurements should be |0⟩ (or equivalent)
        assert all(m in ["0", "zero", False] for m in measurements)
    
    def test_hadamard_superposition(self, quantum_reasoner):
        """Test Hadamard gate creates superposition."""
        from probabilistic_quantum_reasoner.networks import BayesianNetwork
        
        network = BayesianNetwork(name="Hadamard Test")
        
        # Create qubit and apply Hadamard
        qubit = QuantumNode(
            name="Qubit",
            num_qubits=1,
            initial_state="zero",
            quantum_operations=[HadamardGate(qubit=0)]
        )
        
        network.add_node(qubit)
        
        # Measure many times - should get ~50/50 distribution
        measurements = []
        for _ in range(1000):
            result = quantum_reasoner.measure(network, ["Qubit"])
            measurements.append(result["Qubit"])
        
        # Count 0s and 1s
        zero_count = sum(1 for m in measurements if m in ["0", "zero", False])
        one_count = sum(1 for m in measurements if m in ["1", "one", True])
        
        total = zero_count + one_count
        zero_ratio = zero_count / total
        
        # Should be approximately 50% (within 5% tolerance)
        assert 0.45 <= zero_ratio <= 0.55
    
    def test_entanglement(self, quantum_reasoner):
        """Test quantum entanglement between qubits."""
        from probabilistic_quantum_reasoner.networks import BayesianNetwork
        
        network = BayesianNetwork(name="Entanglement Test")
        
        # Create Bell state: |00⟩ + |11⟩
        qubit_a = QuantumNode(name="QubitA", num_qubits=1, initial_state="zero")
        qubit_b = QuantumNode(name="QubitB", num_qubits=1, initial_state="zero")
        
        bell_state = QuantumNode(
            name="BellState",
            num_qubits=2,
            parents=[qubit_a, qubit_b],
            quantum_operations=[
                HadamardGate(qubit=0),
                CNOTGate(control_qubit=0, target_qubit=1)
            ]
        )
        
        network.add_nodes([qubit_a, qubit_b, bell_state])
        
        # Measure correlations
        correlations = []
        for _ in range(1000):
            result = quantum_reasoner.measure(network, ["QubitA", "QubitB"])
            a_result = result["QubitA"]
            b_result = result["QubitB"]
            
            # Convert to binary for comparison
            a_binary = 1 if a_result in ["1", "one", True] else 0
            b_binary = 1 if b_result in ["1", "one", True] else 0
            
            correlations.append(a_binary == b_binary)
        
        # Bell state should have perfect correlation
        correlation_rate = np.mean(correlations)
        assert correlation_rate > 0.95  # Allow for measurement noise
    
    @pytest.mark.slow
    def test_quantum_inference_scaling(self, quantum_reasoner):
        """Test quantum inference scales reasonably with system size."""
        import time
        
        times = []
        qubit_counts = [1, 2, 3, 4]
        
        for num_qubits in qubit_counts:
            network = self._create_quantum_network(num_qubits)
            
            start_time = time.time()
            
            # Perform multiple measurements
            for _ in range(10):
                quantum_reasoner.measure(network, [f"Qubit_{i}" for i in range(num_qubits)])
            
            end_time = time.time()
            times.append(end_time - start_time)
        
        # Time should not grow exponentially (for small qubit counts)
        # This is a basic sanity check
        assert times[-1] < times[0] * 100  # Allow 100x slowdown max
    
    def _create_quantum_network(self, num_qubits):
        """Helper to create quantum network with specified number of qubits."""
        from probabilistic_quantum_reasoner.networks import BayesianNetwork
        
        network = BayesianNetwork(name=f"Quantum Network {num_qubits} qubits")
        
        qubits = []
        for i in range(num_qubits):
            qubit = QuantumNode(
                name=f"Qubit_{i}",
                num_qubits=1,
                initial_state="superposition"
            )
            qubits.append(qubit)
        
        network.add_nodes(qubits)
        return network
```

### Integration Tests

```python
# tests/integration/test_end_to_end.py
import pytest
from probabilistic_quantum_reasoner import ProbabilisticQuantumReasoner
from probabilistic_quantum_reasoner.examples import WeatherMoodNetwork

class TestEndToEnd:
    """End-to-end integration tests."""
    
    @pytest.mark.parametrize("backend", ["classical", "qiskit"])
    def test_complete_workflow(self, backend):
        """Test complete workflow from network creation to inference."""
        if backend == "qiskit":
            pytest.importorskip("qiskit")
        
        # Create example network
        weather_network = WeatherMoodNetwork()
        network = weather_network.create_network()
        
        # Create reasoner
        reasoner = ProbabilisticQuantumReasoner(backend=backend)
        
        # Perform inference
        result = reasoner.infer(
            network=network,
            query=["Mood"],
            evidence={"Weather": "sunny"}
        )
        
        # Validate result
        assert isinstance(result, dict)
        assert "Mood" in result
        assert abs(sum(result["Mood"].values()) - 1.0) < 1e-10
    
    def test_backend_switching(self):
        """Test switching between different backends."""
        network = WeatherMoodNetwork().create_network()
        
        # Start with classical
        reasoner = ProbabilisticQuantumReasoner(backend="classical")
        classical_result = reasoner.infer(
            network=network,
            query=["Mood"],
            evidence={"Weather": "sunny"}
        )
        
        # Switch to quantum (if available)
        try:
            reasoner.set_backend("qiskit")
            quantum_result = reasoner.infer(
                network=network,
                query=["Mood"],
                evidence={"Weather": "sunny"}
            )
            
            # Results should be similar (within tolerance)
            for state in classical_result["Mood"]:
                classical_prob = classical_result["Mood"][state]
                quantum_prob = quantum_result["Mood"][state]
                assert abs(classical_prob - quantum_prob) < 0.1
        
        except ImportError:
            pytest.skip("Quantum backends not available")
    
    def test_error_handling(self):
        """Test error handling in complete workflows."""
        reasoner = ProbabilisticQuantumReasoner(backend="classical")
        network = WeatherMoodNetwork().create_network()
        
        # Test invalid query
        with pytest.raises(ValueError):
            reasoner.infer(
                network=network,
                query=["NonexistentNode"],
                evidence={}
            )
        
        # Test invalid evidence
        with pytest.raises(ValueError):
            reasoner.infer(
                network=network,
                query=["Mood"],
                evidence={"NonexistentNode": "value"}
            )
```

## Performance Testing

### Benchmarking

```python
# tests/performance/test_benchmarks.py
import pytest
import time
import numpy as np
from probabilistic_quantum_reasoner import ProbabilisticQuantumReasoner

@pytest.mark.performance
class TestBenchmarks:
    """Performance benchmark tests."""
    
    @pytest.mark.parametrize("network_size", [10, 25, 50])
    @pytest.mark.parametrize("backend", ["classical"])
    def test_inference_scaling(self, network_size, backend):
        """Test inference time scaling with network size."""
        network = self._create_test_network(network_size)
        reasoner = ProbabilisticQuantumReasoner(backend=backend)
        
        # Warm up
        reasoner.infer(network=network, query=["node_0"], evidence={})
        
        # Benchmark
        start_time = time.time()
        
        for _ in range(10):  # Multiple runs for average
            result = reasoner.infer(
                network=network,
                query=["node_0"],
                evidence={}
            )
        
        end_time = time.time()
        avg_time = (end_time - start_time) / 10
        
        # Performance assertions (adjust based on acceptable performance)
        if network_size == 10:
            assert avg_time < 0.1  # 100ms for small network
        elif network_size == 25:
            assert avg_time < 0.5  # 500ms for medium network
        elif network_size == 50:
            assert avg_time < 2.0  # 2s for large network
    
    def test_memory_usage(self):
        """Test memory usage stays within bounds."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create large network
        network = self._create_test_network(100)
        reasoner = ProbabilisticQuantumReasoner(backend="classical")
        
        # Perform inference
        reasoner.infer(network=network, query=["node_0"], evidence={})
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (adjust threshold as needed)
        assert memory_increase < 500  # Less than 500MB increase
    
    def _create_test_network(self, size):
        """Create test network of specified size."""
        from probabilistic_quantum_reasoner.networks import BayesianNetwork
        from probabilistic_quantum_reasoner.nodes import DiscreteNode
        
        network = BayesianNetwork(name=f"Test Network Size {size}")
        
        # Create chain of nodes
        previous_node = None
        for i in range(size):
            if i == 0:
                # Root node
                node = DiscreteNode(
                    name=f"node_{i}",
                    states=["true", "false"],
                    prior=[0.5, 0.5]
                )
            else:
                # Dependent node
                node = DiscreteNode(
                    name=f"node_{i}",
                    states=["true", "false"],
                    parents=[previous_node],
                    cpt=np.array([[0.8, 0.2], [0.3, 0.7]])
                )
            
            network.add_node(node)
            previous_node = node
        
        return network
```

## Test Configuration

### pytest Configuration

```python
# tests/conftest.py (additional configuration)
import pytest
import warnings

def pytest_configure(config):
    """Configure pytest."""
    # Add custom markers
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "quantum: marks tests that require quantum backends"
    )
    config.addinivalue_line(
        "markers", "performance: marks performance benchmark tests"
    )
    config.addinivalue_line(
        "markers", "integration: marks integration tests"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU acceleration"
    )

def pytest_collection_modifyitems(config, items):
    """Modify test collection."""
    # Skip quantum tests if backends not available
    try:
        import qiskit
        quantum_available = True
    except ImportError:
        quantum_available = False
    
    if not quantum_available:
        skip_quantum = pytest.mark.skip(reason="Quantum backends not available")
        for item in items:
            if "quantum" in item.keywords:
                item.add_marker(skip_quantum)

@pytest.fixture(autouse=True)
def suppress_warnings():
    """Suppress known warnings during tests."""
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
```

## Running Tests

### Basic Test Commands

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=probabilistic_quantum_reasoner --cov-report=html

# Run specific test categories
pytest -m "not slow"           # Skip slow tests
pytest -m quantum              # Only quantum tests
pytest -m "performance"        # Only performance tests

# Run tests in parallel
pytest -n auto                # Use all CPU cores

# Run with verbose output
pytest -v

# Stop on first failure
pytest -x

# Run specific test file
pytest tests/unit/test_core/test_reasoner.py

# Run specific test function
pytest tests/unit/test_core/test_reasoner.py::test_basic_functionality
```

### Continuous Integration

Example GitHub Actions workflow:

```yaml
# .github/workflows/test.yml
name: Test Suite

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.10, 3.11, 3.12]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev,test]"
    
    - name: Run tests
      run: |
        pytest --cov=probabilistic_quantum_reasoner --cov-report=xml
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

## Best Practices

### Test Writing Guidelines

1. **Clear test names**: Use descriptive names that explain what is being tested
2. **Arrange-Act-Assert**: Structure tests with clear setup, execution, and verification
3. **Test one thing**: Each test should verify one specific behavior
4. **Use fixtures**: Leverage pytest fixtures for reusable test components
5. **Mock external dependencies**: Use mocking for external services or hardware
6. **Test edge cases**: Include tests for boundary conditions and error cases

### Coverage Guidelines

- Aim for **90%+ code coverage** for core components
- **100% coverage** for critical inference algorithms
- Focus on **meaningful coverage** rather than just line coverage
- Test both **happy paths** and **error conditions**

### Performance Testing

- Include **performance benchmarks** in the test suite
- Set **reasonable performance thresholds** based on hardware
- Test **memory usage** and **resource consumption**
- Benchmark **different backends** and configurations

This comprehensive testing strategy ensures the reliability, performance, and correctness of the Probabilistic Quantum Reasoner across all supported platforms and use cases.
