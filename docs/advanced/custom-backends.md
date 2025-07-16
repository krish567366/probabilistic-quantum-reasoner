# Custom Backend Development

This guide walks you through creating custom backends for the Probabilistic Quantum Reasoner, enabling integration with new quantum hardware, simulators, or specialized classical computing resources.

## Backend Architecture Overview

The backend system in PQR follows a layered architecture:

```mermaid
Application Layer
    ↓
ProbabilisticQuantumReasoner
    ↓  
Backend Interface
    ↓
Backend Implementation
    ↓
Hardware/Simulator Layer
```

## Base Backend Classes

### Backend Interface

All backends must implement the base `Backend` interface:

```python
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import numpy as np

class Backend(ABC):
    """Abstract base class for all backends."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.backend_type = "unknown"
        self.supports_quantum = False
        self.supports_classical = True
        self._initialize()
    
    @abstractmethod
    def _initialize(self):
        """Initialize backend-specific resources."""
        pass
    
    @abstractmethod
    def infer(self, network, query: List[str], evidence: Dict[str, Any]) -> Dict[str, Any]:
        """Perform inference on the network."""
        pass
    
    @abstractmethod
    def measure(self, network, nodes: List[str]) -> Dict[str, Any]:
        """Measure specified nodes in the network."""
        pass
    
    def validate_network(self, network) -> bool:
        """Validate that the network is compatible with this backend."""
        return True
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Return backend capabilities."""
        return {
            "backend_type": self.backend_type,
            "supports_quantum": self.supports_quantum,
            "supports_classical": self.supports_classical,
            "max_qubits": getattr(self, 'max_qubits', None),
            "available_gates": getattr(self, 'available_gates', [])
        }
```

### Quantum Backend Base Class

For quantum backends, extend the `QuantumBackend` class:

```python
from probabilistic_quantum_reasoner.backends import Backend

class QuantumBackend(Backend):
    """Base class for quantum backends."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.supports_quantum = True
        self.max_qubits = 30  # Default limit
        self.available_gates = ["H", "X", "Y", "Z", "CNOT", "RX", "RY", "RZ"]
        self.shots = self.config.get("shots", 1000)
        self.noise_model = self.config.get("noise_model", None)
    
    @abstractmethod
    def create_circuit(self, num_qubits: int):
        """Create a quantum circuit with specified number of qubits."""
        pass
    
    @abstractmethod
    def execute_circuit(self, circuit, shots: Optional[int] = None):
        """Execute a quantum circuit and return measurement results."""
        pass
    
    @abstractmethod
    def get_state_vector(self, circuit):
        """Get the quantum state vector of a circuit."""
        pass
    
    def apply_noise_model(self, circuit):
        """Apply noise model to the circuit."""
        if self.noise_model:
            return self.noise_model.apply(circuit)
        return circuit
```

## Implementing a Custom Quantum Backend

### Example: Custom Quantum Simulator

Let's create a custom quantum backend using a simple matrix-based simulator:

```python
import numpy as np
from typing import Dict, List, Any, Optional
from probabilistic_quantum_reasoner.backends import QuantumBackend

class CustomQuantumSimulator(QuantumBackend):
    """Custom quantum simulator backend using numpy matrices."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.backend_type = "custom_quantum_simulator"
        self.max_qubits = 20  # Reasonable limit for numpy simulation
        
        # Quantum gate definitions
        self._define_gates()
        
    def _initialize(self):
        """Initialize the custom simulator."""
        self.circuit_cache = {}
        self.execution_count = 0
        
    def _define_gates(self):
        """Define quantum gates as numpy matrices."""
        # Pauli gates
        self.gates = {
            "I": np.array([[1, 0], [0, 1]], dtype=complex),
            "X": np.array([[0, 1], [1, 0]], dtype=complex),
            "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
            "Z": np.array([[1, 0], [0, -1]], dtype=complex),
            "H": np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2),
        }
        
        # Two-qubit gates
        self.gates["CNOT"] = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0], 
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=complex)
    
    def create_circuit(self, num_qubits: int):
        """Create a custom quantum circuit."""
        if num_qubits > self.max_qubits:
            raise ValueError(f"Too many qubits: {num_qubits} > {self.max_qubits}")
        
        return CustomQuantumCircuit(num_qubits, self)
    
    def execute_circuit(self, circuit, shots: Optional[int] = None):
        """Execute the quantum circuit."""
        if shots is None:
            shots = self.shots
        
        # Get final state vector
        state_vector = self.get_state_vector(circuit)
        
        # Simulate measurements
        probabilities = np.abs(state_vector) ** 2
        
        # Sample from probability distribution
        num_states = len(probabilities)
        measurements = []
        
        for _ in range(shots):
            outcome = np.random.choice(num_states, p=probabilities)
            # Convert to bitstring
            bitstring = format(outcome, f'0{circuit.num_qubits}b')
            measurements.append(bitstring)
        
        # Count outcomes
        counts = {}
        for measurement in measurements:
            counts[measurement] = counts.get(measurement, 0) + 1
        
        self.execution_count += 1
        return counts
    
    def get_state_vector(self, circuit):
        """Compute the state vector by applying all gates."""
        # Start with |00...0⟩ state
        state = np.zeros(2 ** circuit.num_qubits, dtype=complex)
        state[0] = 1.0
        
        # Apply each gate in sequence
        for gate_info in circuit.gates:
            state = self._apply_gate(state, gate_info, circuit.num_qubits)
        
        return state
    
    def _apply_gate(self, state, gate_info, num_qubits):
        """Apply a single gate to the state vector."""
        gate_name, qubits, params = gate_info
        
        if gate_name in ["X", "Y", "Z", "H"]:
            # Single-qubit gate
            qubit = qubits[0]
            gate_matrix = self.gates[gate_name]
            full_matrix = self._construct_full_matrix(gate_matrix, qubit, num_qubits)
            return full_matrix @ state
            
        elif gate_name == "CNOT":
            # Two-qubit gate
            control, target = qubits
            gate_matrix = self.gates["CNOT"]
            full_matrix = self._construct_two_qubit_matrix(
                gate_matrix, control, target, num_qubits
            )
            return full_matrix @ state
            
        elif gate_name.startswith("R"):
            # Rotation gate
            angle = params[0] if params else 0
            qubit = qubits[0]
            
            if gate_name == "RX":
                gate_matrix = np.array([
                    [np.cos(angle/2), -1j*np.sin(angle/2)],
                    [-1j*np.sin(angle/2), np.cos(angle/2)]
                ], dtype=complex)
            elif gate_name == "RY":
                gate_matrix = np.array([
                    [np.cos(angle/2), -np.sin(angle/2)],
                    [np.sin(angle/2), np.cos(angle/2)]
                ], dtype=complex)
            elif gate_name == "RZ":
                gate_matrix = np.array([
                    [np.exp(-1j*angle/2), 0],
                    [0, np.exp(1j*angle/2)]
                ], dtype=complex)
            
            full_matrix = self._construct_full_matrix(gate_matrix, qubit, num_qubits)
            return full_matrix @ state
        
        else:
            raise ValueError(f"Unknown gate: {gate_name}")
    
    def _construct_full_matrix(self, gate_matrix, target_qubit, num_qubits):
        """Construct full matrix for single-qubit gate."""
        matrices = []
        for i in range(num_qubits):
            if i == target_qubit:
                matrices.append(gate_matrix)
            else:
                matrices.append(self.gates["I"])
        
        # Tensor product of all matrices
        result = matrices[0]
        for matrix in matrices[1:]:
            result = np.kron(result, matrix)
        
        return result
    
    def _construct_two_qubit_matrix(self, gate_matrix, control, target, num_qubits):
        """Construct full matrix for two-qubit gate."""
        # This is a simplified implementation
        # For a full implementation, need to handle arbitrary control/target positions
        dim = 2 ** num_qubits
        result = np.eye(dim, dtype=complex)
        
        # Apply CNOT logic manually (simplified for demonstration)
        for i in range(dim):
            bitstring = format(i, f'0{num_qubits}b')
            bits = [int(b) for b in bitstring]
            
            if bits[control] == 1:  # Control is |1⟩
                # Flip target bit
                new_bits = bits.copy()
                new_bits[target] = 1 - new_bits[target]
                j = int(''.join(map(str, new_bits)), 2)
                
                # Swap amplitudes
                result[i, i] = 0
                result[i, j] = 1
        
        return result
    
    def infer(self, network, query: List[str], evidence: Dict[str, Any]) -> Dict[str, Any]:
        """Perform inference using the custom simulator."""
        # Convert network to quantum circuits
        circuits = self._network_to_circuits(network, query, evidence)
        
        results = {}
        for var_name, circuit in circuits.items():
            # Execute circuit
            counts = self.execute_circuit(circuit)
            
            # Convert to probabilities
            total_shots = sum(counts.values())
            probabilities = {}
            
            for state, count in counts.items():
                # Map quantum state to variable value
                var_value = self._quantum_state_to_value(state, var_name)
                prob = count / total_shots
                
                if var_value in probabilities:
                    probabilities[var_value] += prob
                else:
                    probabilities[var_value] = prob
            
            results[var_name] = probabilities
        
        return results
    
    def measure(self, network, nodes: List[str]) -> Dict[str, Any]:
        """Measure specified nodes."""
        # Create measurement circuit
        circuit = self._create_measurement_circuit(network, nodes)
        
        # Execute once to get single measurement
        counts = self.execute_circuit(circuit, shots=1)
        bitstring = list(counts.keys())[0]
        
        # Map to node values
        results = {}
        for i, node_name in enumerate(nodes):
            bit_value = bitstring[i]
            results[node_name] = self._bit_to_value(bit_value, node_name)
        
        return results
    
    def _network_to_circuits(self, network, query, evidence):
        """Convert network to quantum circuits (simplified)."""
        # This is a placeholder - real implementation would be more complex
        circuits = {}
        
        for var_name in query:
            circuit = self.create_circuit(num_qubits=2)  # Simplified
            
            # Add gates based on network structure
            circuit.add_gate("H", [0])  # Example gate
            
            circuits[var_name] = circuit
        
        return circuits
    
    def _create_measurement_circuit(self, network, nodes):
        """Create circuit for measuring specific nodes."""
        num_qubits = len(nodes)
        circuit = self.create_circuit(num_qubits)
        
        # Add gates based on network (simplified)
        for i in range(num_qubits):
            circuit.add_gate("H", [i])
        
        return circuit
    
    def _quantum_state_to_value(self, state, var_name):
        """Map quantum state to variable value."""
        # Simplified mapping
        return "state_0" if state[0] == '0' else "state_1"
    
    def _bit_to_value(self, bit, node_name):
        """Map bit value to node value."""
        return "value_0" if bit == '0' else "value_1"

class CustomQuantumCircuit:
    """Custom quantum circuit implementation."""
    
    def __init__(self, num_qubits: int, backend):
        self.num_qubits = num_qubits
        self.backend = backend
        self.gates = []  # List of (gate_name, qubits, params)
    
    def add_gate(self, gate_name: str, qubits: List[int], params: Optional[List[float]] = None):
        """Add a gate to the circuit."""
        if params is None:
            params = []
        
        # Validate qubits
        for qubit in qubits:
            if qubit >= self.num_qubits:
                raise ValueError(f"Qubit {qubit} out of range")
        
        # Validate gate
        if gate_name not in self.backend.available_gates:
            raise ValueError(f"Gate {gate_name} not available")
        
        self.gates.append((gate_name, qubits, params))
    
    def add_hadamard(self, qubit: int):
        """Add Hadamard gate."""
        self.add_gate("H", [qubit])
    
    def add_cnot(self, control: int, target: int):
        """Add CNOT gate."""
        self.add_gate("CNOT", [control, target])
    
    def add_rotation(self, gate_type: str, qubit: int, angle: float):
        """Add rotation gate."""
        self.add_gate(gate_type, [qubit], [angle])
    
    def depth(self) -> int:
        """Return circuit depth."""
        return len(self.gates)
    
    def copy(self):
        """Create a copy of the circuit."""
        new_circuit = CustomQuantumCircuit(self.num_qubits, self.backend)
        new_circuit.gates = self.gates.copy()
        return new_circuit
```

## Classical Backend Implementation

### Example: GPU-Accelerated Classical Backend

```python
import numpy as np
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

from probabilistic_quantum_reasoner.backends import Backend

class GPUClassicalBackend(Backend):
    """GPU-accelerated classical backend using CuPy."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.backend_type = "gpu_classical"
        self.supports_classical = True
        
        if not CUPY_AVAILABLE:
            raise ImportError("CuPy not available for GPU acceleration")
        
        self.device_id = self.config.get("device_id", 0)
        self.memory_pool = cp.get_default_memory_pool()
        
    def _initialize(self):
        """Initialize GPU resources."""
        cp.cuda.Device(self.device_id).use()
        print(f"Using GPU device {self.device_id}")
        
        # Pre-allocate common arrays
        self._setup_memory_pools()
    
    def _setup_memory_pools(self):
        """Setup memory pools for efficient GPU memory management."""
        # Pre-allocate common matrix sizes
        common_sizes = [10, 100, 1000]
        self.matrix_cache = {}
        
        for size in common_sizes:
            self.matrix_cache[size] = cp.zeros((size, size), dtype=cp.float32)
    
    def infer(self, network, query: List[str], evidence: Dict[str, Any]) -> Dict[str, Any]:
        """Perform GPU-accelerated inference."""
        
        # Convert network to GPU arrays
        gpu_factors = self._network_to_gpu_factors(network)
        
        # Apply evidence
        gpu_factors = self._apply_evidence_gpu(gpu_factors, evidence)
        
        # Perform variable elimination on GPU
        results = {}
        for var_name in query:
            marginal = self._compute_marginal_gpu(gpu_factors, var_name)
            
            # Convert back to CPU and normalize
            marginal_cpu = cp.asnumpy(marginal)
            marginal_cpu = marginal_cpu / np.sum(marginal_cpu)
            
            # Convert to probability dictionary
            var_states = network.nodes[var_name].states
            prob_dict = {state: float(marginal_cpu[i]) 
                        for i, state in enumerate(var_states)}
            
            results[var_name] = prob_dict
        
        return results
    
    def _network_to_gpu_factors(self, network):
        """Convert network factors to GPU arrays."""
        gpu_factors = []
        
        for node_name, node in network.nodes.items():
            if hasattr(node, 'cpt'):
                # Convert CPT to GPU array
                cpt_gpu = cp.asarray(node.cpt, dtype=cp.float32)
                gpu_factors.append({
                    'name': node_name,
                    'variables': [node_name] + [p.name for p in node.parents],
                    'factor': cpt_gpu
                })
            elif hasattr(node, 'prior'):
                # Convert prior to GPU array
                prior_gpu = cp.asarray(node.prior, dtype=cp.float32)
                gpu_factors.append({
                    'name': node_name,
                    'variables': [node_name],
                    'factor': prior_gpu
                })
        
        return gpu_factors
    
    def _apply_evidence_gpu(self, gpu_factors, evidence):
        """Apply evidence using GPU operations."""
        for factor in gpu_factors:
            for var_name in factor['variables']:
                if var_name in evidence:
                    # Reduce factor by fixing evidence variable
                    factor['factor'] = self._fix_variable_gpu(
                        factor['factor'], var_name, evidence[var_name], factor
                    )
        
        return gpu_factors
    
    def _fix_variable_gpu(self, factor_gpu, var_name, value, factor_info):
        """Fix a variable to a specific value using GPU operations."""
        # This is a simplified implementation
        # Real implementation would handle arbitrary factor shapes
        
        var_index = factor_info['variables'].index(var_name)
        value_index = self._get_value_index(var_name, value)
        
        # Select the slice corresponding to the evidence
        # This is a simplified selection - real implementation more complex
        if var_index == 0:
            return factor_gpu[value_index, :]
        else:
            return factor_gpu[:, value_index]
    
    def _compute_marginal_gpu(self, gpu_factors, target_var):
        """Compute marginal distribution using GPU operations."""
        
        # Find factors involving target variable
        relevant_factors = [f for f in gpu_factors if target_var in f['variables']]
        
        if not relevant_factors:
            raise ValueError(f"No factors found for variable {target_var}")
        
        # Start with first factor
        result = relevant_factors[0]['factor']
        
        # Multiply with other factors (simplified)
        for factor in relevant_factors[1:]:
            result = self._multiply_factors_gpu(result, factor['factor'])
        
        # Sum out non-target variables (simplified)
        # Real implementation would handle arbitrary marginalization
        if result.ndim > 1:
            result = cp.sum(result, axis=tuple(range(1, result.ndim)))
        
        return result
    
    def _multiply_factors_gpu(self, factor1, factor2):
        """Multiply two factors using GPU operations."""
        # This is a simplified multiplication
        # Real implementation would handle arbitrary factor combinations
        
        if factor1.shape == factor2.shape:
            return factor1 * factor2
        else:
            # Broadcast and multiply
            return cp.multiply.outer(factor1, factor2).flatten()
    
    def measure(self, network, nodes: List[str]) -> Dict[str, Any]:
        """Perform measurement (classical sampling)."""
        # Infer joint distribution
        joint_result = self.infer(network, nodes, {})
        
        # Sample from joint distribution
        # Simplified - real implementation would compute actual joint
        results = {}
        for node_name in nodes:
            node_probs = joint_result[node_name]
            states = list(node_probs.keys())
            probs = list(node_probs.values())
            
            # Sample
            sampled_state = np.random.choice(states, p=probs)
            results[node_name] = sampled_state
        
        return results
    
    def _get_value_index(self, var_name, value):
        """Get index of value in variable's state space."""
        # This would be implemented based on network structure
        return 0  # Simplified
    
    def get_memory_usage(self):
        """Get current GPU memory usage."""
        return {
            'used_bytes': self.memory_pool.used_bytes(),
            'total_bytes': self.memory_pool.total_bytes()
        }
    
    def clear_cache(self):
        """Clear GPU memory cache."""
        self.memory_pool.free_all_blocks()
```

## Backend Registration and Integration

### Registering Custom Backends

```python
from probabilistic_quantum_reasoner.backends import BackendRegistry

# Register the custom backends
BackendRegistry.register("custom_quantum", CustomQuantumSimulator)
BackendRegistry.register("gpu_classical", GPUClassicalBackend)

# Use the custom backend
from probabilistic_quantum_reasoner import ProbabilisticQuantumReasoner

# Create reasoner with custom quantum backend
reasoner = ProbabilisticQuantumReasoner(
    backend="custom_quantum",
    backend_config={
        "shots": 2000,
        "max_qubits": 15
    }
)

# Or create directly
custom_backend = CustomQuantumSimulator(config={"shots": 5000})
reasoner = ProbabilisticQuantumReasoner(backend=custom_backend)
```

### Backend Factory Pattern

```python
class BackendFactory:
    """Factory for creating backends with different configurations."""
    
    @staticmethod
    def create_backend(backend_type: str, **kwargs):
        """Create backend with specified configuration."""
        
        if backend_type == "custom_quantum_optimized":
            return CustomQuantumSimulator(config={
                "shots": kwargs.get("shots", 10000),
                "optimization_level": kwargs.get("optimization", 3),
                "memory_limit": kwargs.get("memory", "8GB")
            })
        
        elif backend_type == "gpu_classical_high_performance":
            return GPUClassicalBackend(config={
                "device_id": kwargs.get("gpu_id", 0),
                "precision": kwargs.get("precision", "float32"),
                "batch_size": kwargs.get("batch_size", 1000)
            })
        
        else:
            raise ValueError(f"Unknown backend type: {backend_type}")

# Use factory
backend = BackendFactory.create_backend(
    "custom_quantum_optimized",
    shots=20000,
    optimization=2
)
```

## Testing Custom Backends

### Unit Tests

```python
import unittest
from probabilistic_quantum_reasoner.networks import BayesianNetwork
from probabilistic_quantum_reasoner.nodes import DiscreteNode

class TestCustomQuantumSimulator(unittest.TestCase):
    """Test suite for custom quantum simulator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.backend = CustomQuantumSimulator()
        self.simple_network = self._create_simple_network()
    
    def _create_simple_network(self):
        """Create a simple test network."""
        network = BayesianNetwork()
        
        node_a = DiscreteNode(
            name="A",
            states=["true", "false"],
            prior=[0.6, 0.4]
        )
        
        network.add_node(node_a)
        return network
    
    def test_circuit_creation(self):
        """Test quantum circuit creation."""
        circuit = self.backend.create_circuit(num_qubits=3)
        self.assertEqual(circuit.num_qubits, 3)
        self.assertEqual(len(circuit.gates), 0)
    
    def test_gate_addition(self):
        """Test adding gates to circuit."""
        circuit = self.backend.create_circuit(num_qubits=2)
        circuit.add_hadamard(0)
        circuit.add_cnot(0, 1)
        
        self.assertEqual(len(circuit.gates), 2)
        self.assertEqual(circuit.gates[0][0], "H")
        self.assertEqual(circuit.gates[1][0], "CNOT")
    
    def test_state_vector_computation(self):
        """Test state vector computation."""
        circuit = self.backend.create_circuit(num_qubits=1)
        circuit.add_hadamard(0)
        
        state = self.backend.get_state_vector(circuit)
        
        # Should be |+⟩ = (|0⟩ + |1⟩)/√2
        expected = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
        np.testing.assert_allclose(state, expected, atol=1e-10)
    
    def test_circuit_execution(self):
        """Test circuit execution and measurement."""
        circuit = self.backend.create_circuit(num_qubits=1)
        circuit.add_hadamard(0)
        
        counts = self.backend.execute_circuit(circuit, shots=1000)
        
        # Should get roughly 50/50 split
        total_counts = sum(counts.values())
        self.assertEqual(total_counts, 1000)
        
        # Check that we get both '0' and '1' outcomes
        self.assertIn('0', counts)
        self.assertIn('1', counts)
    
    def test_bell_state(self):
        """Test Bell state creation and measurement."""
        circuit = self.backend.create_circuit(num_qubits=2)
        circuit.add_hadamard(0)
        circuit.add_cnot(0, 1)
        
        counts = self.backend.execute_circuit(circuit, shots=1000)
        
        # Should get only '00' and '11' outcomes
        self.assertIn('00', counts)
        self.assertIn('11', counts)
        self.assertNotIn('01', counts)
        self.assertNotIn('10', counts)
    
    def test_error_handling(self):
        """Test error handling."""
        # Test qubit out of range
        with self.assertRaises(ValueError):
            circuit = self.backend.create_circuit(num_qubits=2)
            circuit.add_hadamard(2)
        
        # Test too many qubits
        with self.assertRaises(ValueError):
            self.backend.create_circuit(num_qubits=100)

class TestGPUClassicalBackend(unittest.TestCase):
    """Test suite for GPU classical backend."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not CUPY_AVAILABLE:
            self.skipTest("CuPy not available")
        
        self.backend = GPUClassicalBackend()
    
    def test_initialization(self):
        """Test backend initialization."""
        self.assertEqual(self.backend.backend_type, "gpu_classical")
        self.assertTrue(self.backend.supports_classical)
    
    def test_memory_management(self):
        """Test GPU memory management."""
        initial_usage = self.backend.get_memory_usage()
        
        # Perform some operations
        large_array = cp.zeros((1000, 1000))
        
        final_usage = self.backend.get_memory_usage()
        self.assertGreater(final_usage['used_bytes'], initial_usage['used_bytes'])
        
        # Clear cache
        self.backend.clear_cache()

# Run tests
if __name__ == "__main__":
    unittest.main()
```

### Integration Tests

```python
def test_backend_integration():
    """Test integration with the main reasoner."""
    
    # Create test network
    network = BayesianNetwork()
    
    node_a = DiscreteNode(
        name="Weather",
        states=["sunny", "rainy"],
        prior=[0.7, 0.3]
    )
    
    node_b = DiscreteNode(
        name="Mood",
        states=["happy", "sad"],
        parents=[node_a],
        cpt=np.array([
            [0.8, 0.2],  # sunny -> happy/sad
            [0.3, 0.7]   # rainy -> happy/sad
        ])
    )
    
    network.add_nodes([node_a, node_b])
    
    # Test with custom backend
    custom_backend = CustomQuantumSimulator()
    reasoner = ProbabilisticQuantumReasoner(backend=custom_backend)
    
    # Perform inference
    result = reasoner.infer(
        network=network,
        query=["Mood"],
        evidence={"Weather": "sunny"}
    )
    
    print("Integration test result:")
    print(f"P(Mood|Weather=sunny): {result}")
    
    # Test measurement
    measurement = reasoner.measure(
        network=network,
        nodes=["Weather", "Mood"]
    )
    
    print(f"Sample measurement: {measurement}")

# Run integration test
test_backend_integration()
```

## Performance Optimization

### Caching and Memoization

```python
from functools import lru_cache
import hashlib

class OptimizedCustomBackend(CustomQuantumSimulator):
    """Optimized version with caching."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.circuit_cache = {}
        self.state_cache = {}
    
    def get_state_vector(self, circuit):
        """Cached state vector computation."""
        # Create circuit hash
        circuit_hash = self._hash_circuit(circuit)
        
        if circuit_hash in self.state_cache:
            return self.state_cache[circuit_hash]
        
        # Compute state vector
        state = super().get_state_vector(circuit)
        
        # Cache result
        self.state_cache[circuit_hash] = state
        
        return state
    
    def _hash_circuit(self, circuit):
        """Create hash of circuit for caching."""
        circuit_str = f"{circuit.num_qubits}_{circuit.gates}"
        return hashlib.md5(circuit_str.encode()).hexdigest()
    
    @lru_cache(maxsize=1000)
    def _apply_single_gate_cached(self, state_hash, gate_info, num_qubits):
        """Cached gate application."""
        # Convert hash back to state (simplified)
        state = self._hash_to_state(state_hash)
        return self._apply_gate(state, gate_info, num_qubits)
    
    def clear_caches(self):
        """Clear all caches."""
        self.circuit_cache.clear()
        self.state_cache.clear()
        self._apply_single_gate_cached.cache_clear()
```

### Parallel Processing

```python
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor

class ParallelQuantumBackend(CustomQuantumSimulator):
    """Parallel quantum backend for multiple circuit execution."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.num_workers = config.get("num_workers", mp.cpu_count())
    
    def execute_circuits_parallel(self, circuits, shots_per_circuit=None):
        """Execute multiple circuits in parallel."""
        
        if shots_per_circuit is None:
            shots_per_circuit = [self.shots] * len(circuits)
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [
                executor.submit(self.execute_circuit, circuit, shots)
                for circuit, shots in zip(circuits, shots_per_circuit)
            ]
            
            results = [future.result() for future in futures]
        
        return results
    
    def infer_parallel(self, networks, queries, evidences):
        """Perform inference on multiple networks in parallel."""
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [
                executor.submit(self.infer, network, query, evidence)
                for network, query, evidence in zip(networks, queries, evidences)
            ]
            
            results = [future.result() for future in futures]
        
        return results
```

## Deployment and Distribution

### Packaging Custom Backends

```python
# setup.py for custom backend package
from setuptools import setup, find_packages

setup(
    name="pqr-custom-backends",
    version="0.1.0",
    description="Custom backends for Probabilistic Quantum Reasoner",
    packages=find_packages(),
    install_requires=[
        "probabilistic-quantum-reasoner>=0.1.0",
        "numpy>=1.21.0",
        "cupy-cuda11x>=10.0.0; platform_system!='Darwin'",
    ],
    extras_require={
        "gpu": ["cupy-cuda11x>=10.0.0"],
        "dev": ["pytest>=7.0.0", "pytest-cov>=4.0.0"],
    },
    entry_points={
        "pqr.backends": [
            "custom_quantum = pqr_custom_backends:CustomQuantumSimulator",
            "gpu_classical = pqr_custom_backends:GPUClassicalBackend",
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: Other/Proprietary License"
        "Programming Language :: Python :: 3.10+",
    ],
    python_requires=">=3.10",
)
```

### Docker Deployment

```dockerfile
# Dockerfile for custom backend deployment
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cuda-toolkit-11-8 \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt .
RUN pip install -r requirements.txt

# Install custom backends
COPY . /app/pqr-custom-backends
WORKDIR /app/pqr-custom-backends
RUN pip install .

# Set up environment
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTHONPATH=/app

# Run tests
RUN python -m pytest tests/

ENTRYPOINT ["python", "-m", "pqr_custom_backends.server"]
```

## Best Practices

### Performance Guidelines

1. **Minimize state vector computations** - cache when possible
2. **Use appropriate precision** - float32 vs float64 trade-offs
3. **Batch operations** - process multiple circuits together
4. **Memory management** - clear unused arrays and circuits
5. **Parallel execution** - leverage multi-core and GPU resources

### Error Handling

1. **Validate inputs** early and provide clear error messages
2. **Handle hardware failures** gracefully with fallback options
3. **Monitor resource usage** and prevent memory leaks
4. **Log operations** for debugging and performance analysis

### Testing Strategy

1. **Unit tests** for individual components
2. **Integration tests** with the main reasoner
3. **Performance benchmarks** against existing backends  
4. **Stress tests** with large networks and long circuits
5. **Hardware-specific tests** for GPU and quantum devices

This guide provides a comprehensive foundation for developing custom backends that integrate seamlessly with the Probabilistic Quantum Reasoner framework.
