# Quantum Nodes

This guide provides detailed information about quantum nodes, their properties, and how to work with quantum superposition in probabilistic reasoning.

## Understanding Quantum Nodes

Quantum nodes represent variables that exist in quantum superposition, allowing them to be in multiple states simultaneously with complex probability amplitudes.

### Quantum Superposition

Unlike classical variables that have definite values, quantum nodes can exist in superposition:

```python
import numpy as np
from probabilistic_quantum_reasoner import QuantumBayesianNetwork
from probabilistic_quantum_reasoner.backends import ClassicalSimulator

backend = ClassicalSimulator()
network = QuantumBayesianNetwork("QuantumDemo", backend)

# Classical node - definite state
classical_coin = network.add_stochastic_node(
    "classical_coin",
    outcome_space=["heads", "tails"]
)
classical_coin.prior_distribution = np.array([0.5, 0.5])

# Quantum node - superposition state  
quantum_coin = network.add_quantum_node(
    "quantum_coin",
    outcome_space=["heads", "tails"],
    initial_amplitudes=np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
)

print(f"Classical probabilities: {classical_coin.prior_distribution}")
print(f"Quantum amplitudes: {quantum_coin.quantum_state.amplitudes}")
print(f"Quantum probabilities: {quantum_coin.probability_distribution}")
```

### Complex Amplitudes

Quantum amplitudes are complex numbers that encode both magnitude and phase:

```python
# Amplitude with phase
phase = np.pi / 4
quantum_node = network.add_quantum_node(
    "phase_node",
    outcome_space=["state_0", "state_1"],
    initial_amplitudes=np.array([
        1/np.sqrt(2),  # Real amplitude
        np.exp(1j * phase) / np.sqrt(2)  # Complex amplitude with phase
    ])
)

amplitudes = quantum_node.quantum_state.amplitudes
print(f"Amplitude magnitudes: {np.abs(amplitudes)}")
print(f"Amplitude phases: {np.angle(amplitudes)}")
print(f"Born rule probabilities: {np.abs(amplitudes)**2}")
```

## Quantum State Operations

### Applying Quantum Gates

Transform quantum states using unitary operators:

```python
from probabilistic_quantum_reasoner.core.operators import QuantumGate

# Create quantum node in |0⟩ state
qubit = network.add_quantum_node(
    "qubit",
    outcome_space=[0, 1],
    initial_amplitudes=np.array([1, 0], dtype=complex)
)

print(f"Initial state: {qubit.quantum_state.amplitudes}")

# Apply Hadamard gate to create superposition
hadamard = QuantumGate.hadamard()
network.apply_quantum_operator("qubit", hadamard)
print(f"After Hadamard: {qubit.quantum_state.amplitudes}")

# Apply Pauli-X gate (bit flip)
pauli_x = QuantumGate.pauli_x()
network.apply_quantum_operator("qubit", pauli_x)
print(f"After Pauli-X: {qubit.quantum_state.amplitudes}")

# Apply rotation gate
rotation_y = QuantumGate.rotation_y(np.pi/3)
network.apply_quantum_operator("qubit", rotation_y)
print(f"After Y rotation: {qubit.quantum_state.amplitudes}")
```

### Custom Quantum Operations

Create and apply custom unitary operations:

```python
from probabilistic_quantum_reasoner.core.operators import UnitaryOperator

# Custom 2x2 unitary matrix
custom_matrix = np.array([
    [0.6, 0.8],
    [0.8, -0.6]
], dtype=complex)

# Verify it's unitary
assert np.allclose(custom_matrix @ custom_matrix.conj().T, np.eye(2))

# Create and apply custom operator
custom_operator = UnitaryOperator(custom_matrix)
network.apply_quantum_operator("qubit", custom_operator)
print(f"After custom operation: {qubit.quantum_state.amplitudes}")
```

## Multi-State Quantum Nodes

### Qutrit Example (3-Level System)

```python
# Three-level quantum system
qutrit = network.add_quantum_node(
    "qutrit",
    outcome_space=["low", "medium", "high"],
    initial_amplitudes=np.array([1, 1, 1], dtype=complex) / np.sqrt(3)
)

print(f"Qutrit state: {qutrit.quantum_state.amplitudes}")
print(f"Qutrit probabilities: {qutrit.probability_distribution}")

# Custom 3x3 unitary (discrete Fourier transform)
dft_3 = np.array([
    [1, 1, 1],
    [1, np.exp(2j*np.pi/3), np.exp(4j*np.pi/3)],
    [1, np.exp(4j*np.pi/3), np.exp(8j*np.pi/3)]
], dtype=complex) / np.sqrt(3)

dft_operator = UnitaryOperator(dft_3)
network.apply_quantum_operator("qutrit", dft_operator)
print(f"After DFT: {qutrit.quantum_state.amplitudes}")
```

### Many-Level Systems

```python
# High-dimensional quantum node
n_levels = 8
high_dim_node = network.add_quantum_node(
    "high_dim",
    outcome_space=list(range(n_levels)),
    initial_amplitudes=np.ones(n_levels, dtype=complex) / np.sqrt(n_levels)
)

# Apply random unitary transformation
from probabilistic_quantum_reasoner.core.operators import QuantumGate
random_unitary = QuantumGate.random_unitary(n_levels)
network.apply_quantum_operator("high_dim", random_unitary)

print(f"High-dimensional state dimension: {high_dim_node.quantum_state.dimension}")
```

## Quantum Measurements

### Computational Basis Measurement

```python
from probabilistic_quantum_reasoner.core.operators import MeasurementOperator

# Create measurement operator
computational_measurement = MeasurementOperator.computational_basis(2)

# Measure quantum node in superposition
superposition_node = network.add_quantum_node(
    "measurement_demo",
    outcome_space=[0, 1],
    initial_amplitudes=np.array([0.6, 0.8], dtype=complex)
)

# Perform measurement
result = network.measure_node("measurement_demo")
print(f"Measurement outcome: {result['outcome']}")
print(f"Measurement probability: {result['probability']}")
print(f"Post-measurement state: {result['post_measurement_state']}")

# State has collapsed
print(f"Node state after measurement: {superposition_node.quantum_state.amplitudes}")
```

### Observable Measurements

```python
# Pauli-Z measurement (energy eigenvalues)
pauli_z_measurement = MeasurementOperator.pauli_z()

# Create node in superposition
energy_node = network.add_quantum_node(
    "energy",
    outcome_space=["low_energy", "high_energy"],
    initial_amplitudes=np.array([0.8, 0.6], dtype=complex)
)

# Measure energy
energy_result = pauli_z_measurement.measure(energy_node.quantum_state.amplitudes)
print(f"Energy measurement: {energy_result}")

# Pauli-X measurement (spin in X direction)
pauli_x_measurement = MeasurementOperator.pauli_x()
spin_result = pauli_x_measurement.measure(energy_node.quantum_state.amplitudes)
print(f"Spin-X measurement: {spin_result}")
```

### Expectation Values

```python
# Compute expectation values without measurement
def compute_expectation_value(node, observable_matrix):
    """Compute expectation value ⟨ψ|O|ψ⟩."""
    amplitudes = node.quantum_state.amplitudes
    return np.real(np.conj(amplitudes) @ observable_matrix @ amplitudes)

# Pauli-Z expectation value
pauli_z = np.array([[1, 0], [0, -1]], dtype=complex)
z_expectation = compute_expectation_value(energy_node, pauli_z)
print(f"⟨Z⟩ = {z_expectation}")

# Pauli-X expectation value  
pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
x_expectation = compute_expectation_value(energy_node, pauli_x)
print(f"⟨X⟩ = {x_expectation}")

# Pauli-Y expectation value
pauli_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
y_expectation = compute_expectation_value(energy_node, pauli_y)
print(f"⟨Y⟩ = {y_expectation}")
```

## Quantum Entanglement

### Creating Entangled States

```python
# Two quantum nodes to be entangled
alice = network.add_quantum_node(
    "alice",
    outcome_space=[0, 1],
    initial_amplitudes=np.array([1, 0], dtype=complex)
)

bob = network.add_quantum_node(
    "bob",
    outcome_space=[0, 1], 
    initial_amplitudes=np.array([1, 0], dtype=complex)
)

# Create entanglement (Bell state)
network.entangle([alice, bob])

# Check joint quantum state
joint_state = network.get_joint_quantum_state([alice, bob])
print(f"Joint entangled state: {joint_state.amplitudes}")

# This should be |00⟩ + |11⟩ (Bell state)
expected_bell = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=complex)
print(f"Expected Bell state: {expected_bell}")
```

### Measuring Entangled Systems

```python
# Measure Alice's qubit
alice_result = network.measure_node("alice")
print(f"Alice measured: {alice_result['outcome']}")

# Bob's state is now correlated
bob_state_after = bob.quantum_state.amplitudes
print(f"Bob's state after Alice's measurement: {bob_state_after}")

# Bob's measurement will be perfectly correlated
bob_result = network.measure_node("bob")
print(f"Bob measured: {bob_result['outcome']}")
print(f"Correlation: {alice_result['outcome'] == bob_result['outcome']}")
```

### Multi-Particle Entanglement

```python
# Create GHZ state (three-particle entanglement)
particle1 = network.add_quantum_node("particle1", [0, 1])
particle2 = network.add_quantum_node("particle2", [0, 1])  
particle3 = network.add_quantum_node("particle3", [0, 1])

# Entangle all three particles
network.entangle([particle1, particle2, particle3])

# Apply operations to create GHZ state |000⟩ + |111⟩
hadamard = QuantumGate.hadamard()
cnot = QuantumGate.controlled_not()

# H on first particle, then CNOT chain
network.apply_quantum_operator("particle1", hadamard)
network.apply_two_qubit_gate("particle1", "particle2", cnot)
network.apply_two_qubit_gate("particle2", "particle3", cnot)

ghz_state = network.get_joint_quantum_state([particle1, particle2, particle3])
print(f"GHZ state: {ghz_state.amplitudes}")
```

## Quantum Decoherence

### Modeling Decoherence

```python
# Simulate decoherence effects
def apply_decoherence(node, decoherence_rate, time_step):
    """Apply simple decoherence model."""
    amplitudes = node.quantum_state.amplitudes
    
    # Exponential decay of off-diagonal elements (T2 process)
    decay_factor = np.exp(-time_step / decoherence_rate)
    
    # Keep population (diagonal) terms, decay coherence (off-diagonal)
    probabilities = np.abs(amplitudes) ** 2
    new_amplitudes = np.sqrt(probabilities)
    
    # Add small amount of phase noise
    phase_noise = np.random.normal(0, 0.1, len(amplitudes))
    new_amplitudes = new_amplitudes * np.exp(1j * phase_noise * decay_factor)
    
    node.quantum_state.amplitudes = new_amplitudes
    node.quantum_state.normalize()

# Example: decoherence in quantum node
decoherent_node = network.add_quantum_node(
    "decoherent",
    outcome_space=[0, 1],
    initial_amplitudes=np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
)

print(f"Before decoherence: {decoherent_node.quantum_state.amplitudes}")

# Apply decoherence over time
for t in range(5):
    apply_decoherence(decoherent_node, decoherence_rate=10.0, time_step=1.0)
    print(f"After {t+1} time steps: {decoherent_node.quantum_state.amplitudes}")
```

## Quantum State Analysis

### State Properties

```python
def analyze_quantum_state(node):
    """Analyze properties of a quantum state."""
    amplitudes = node.quantum_state.amplitudes
    probabilities = np.abs(amplitudes) ** 2
    
    # Purity (how mixed vs pure the state is)
    purity = np.sum(probabilities ** 2)
    
    # Entropy (measure of uncertainty)
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
    
    # Maximum coherence (off-diagonal terms)
    coherence = np.sum(np.abs(amplitudes[np.triu_indices(len(amplitudes), k=1)]))
    
    return {
        "purity": purity,
        "entropy": entropy,
        "coherence": coherence,
        "dimension": len(amplitudes)
    }

# Analyze different quantum states
pure_state = network.add_quantum_node("pure", [0, 1], np.array([1, 0], dtype=complex))
mixed_state = network.add_quantum_node("mixed", [0, 1], np.array([0.7, 0.7], dtype=complex))
superposition = network.add_quantum_node("super", [0, 1], np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex))

for node_name, node in [("pure", pure_state), ("mixed", mixed_state), ("superposition", superposition)]:
    analysis = analyze_quantum_state(node)
    print(f"{node_name} state analysis: {analysis}")
```

### Quantum Fidelity

```python
def quantum_fidelity(state1, state2):
    """Compute fidelity between two quantum states."""
    amplitudes1 = state1.quantum_state.amplitudes
    amplitudes2 = state2.quantum_state.amplitudes
    return abs(np.vdot(amplitudes1, amplitudes2)) ** 2

# Compare quantum states
state_a = network.add_quantum_node("state_a", [0, 1], np.array([1, 0], dtype=complex))
state_b = network.add_quantum_node("state_b", [0, 1], np.array([0, 1], dtype=complex))
state_c = network.add_quantum_node("state_c", [0, 1], np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex))

print(f"Fidelity(A, B): {quantum_fidelity(state_a, state_b)}")  # Orthogonal states
print(f"Fidelity(A, C): {quantum_fidelity(state_a, state_c)}")  # Partial overlap
print(f"Fidelity(A, A): {quantum_fidelity(state_a, state_a)}")  # Identical states
```

## Best Practices

### State Preparation

```python
# Good: Normalized amplitudes
good_amplitudes = np.array([0.6, 0.8], dtype=complex)
good_amplitudes /= np.linalg.norm(good_amplitudes)

# Bad: Unnormalized amplitudes (will be auto-normalized but prints warning)
bad_amplitudes = np.array([1, 1], dtype=complex)  # Should be [1/√2, 1/√2]

# Always verify normalization
assert abs(np.sum(np.abs(good_amplitudes)**2) - 1.0) < 1e-10
```

### Memory Management

```python
# Monitor quantum state sizes
def check_memory_usage(network):
    """Check memory usage of quantum states."""
    total_amplitudes = 0
    
    for node_id, node in network.nodes.items():
        if hasattr(node, 'quantum_state'):
            n_amplitudes = len(node.quantum_state.amplitudes)
            total_amplitudes += n_amplitudes
            print(f"Node {node_id}: {n_amplitudes} amplitudes")
    
    memory_mb = total_amplitudes * 16 / 1024 / 1024  # 16 bytes per complex number
    print(f"Total quantum memory: {memory_mb:.2f} MB")
    
    return memory_mb

# Check before creating large quantum systems
memory_usage = check_memory_usage(network)
if memory_usage > 1000:  # 1 GB limit
    print("Warning: Large memory usage detected")
```

### Performance Optimization

```python
# Efficient quantum operations
class QuantumNodeManager:
    def __init__(self, network):
        self.network = network
        self._cached_states = {}
    
    def batch_apply_operations(self, operations):
        """Apply multiple quantum operations efficiently."""
        for node_id, operator in operations:
            self.network.apply_quantum_operator(node_id, operator)
    
    def cache_quantum_state(self, node_id):
        """Cache quantum state for fast access."""
        node = self.network.nodes[node_id]
        self._cached_states[node_id] = node.quantum_state.amplitudes.copy()
    
    def restore_quantum_state(self, node_id):
        """Restore cached quantum state."""
        if node_id in self._cached_states:
            node = self.network.nodes[node_id]
            node.quantum_state.amplitudes = self._cached_states[node_id].copy()
```

## Common Pitfalls

1. **Unnormalized States**: Always ensure amplitudes are normalized
2. **Phase Ignoring**: Remember that quantum phases matter for interference
3. **Measurement Timing**: Quantum states collapse after measurement
4. **Memory Scaling**: State space grows exponentially with system size
5. **Classical Thinking**: Quantum superposition is not just probability

## Next Steps

- Learn about [Inference Methods](inference.md)
- Explore [Causal Reasoning](causal.md) with quantum variables
- See [Variational Methods](variational.md) for optimization
- Check [Examples](../examples/quantum-xor.md) for practical applications
