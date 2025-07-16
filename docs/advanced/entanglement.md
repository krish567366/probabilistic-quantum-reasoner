# Quantum Entanglement in Probabilistic Reasoning

This guide explores the role of quantum entanglement in probabilistic reasoning and how to leverage it effectively within the Probabilistic Quantum Reasoner framework.

## Understanding Quantum Entanglement

Quantum entanglement is a fundamental quantum mechanical phenomenon where quantum systems become correlated in such a way that the quantum state of each system cannot be described independently. In probabilistic reasoning, entanglement can create complex dependencies that go beyond classical correlations.

### Mathematical Foundation

For two qubits A and B, an entangled state cannot be written as a product state:

$$|\psi\rangle_{AB} \neq |\psi\rangle_A \otimes |\psi\rangle_B$$

The most famous example is the Bell state:

$$|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$$

This state exhibits perfect correlation: measuring qubit A in the computational basis instantly determines the measurement outcome of qubit B, regardless of the physical distance between them.

## Creating Entangled Networks

### Basic Entanglement

```python
import numpy as np
from probabilistic_quantum_reasoner import ProbabilisticQuantumReasoner
from probabilistic_quantum_reasoner.networks import BayesianNetwork
from probabilistic_quantum_reasoner.nodes import QuantumNode
from probabilistic_quantum_reasoner.quantum_ops import HadamardGate, CNOTGate

def create_bell_state_network():
    """Create a network with Bell state entanglement."""
    
    network = BayesianNetwork(name="Bell State Network")
    
    # First qubit - will be put in superposition
    qubit_a = QuantumNode(
        name="QubitA",
        num_qubits=1,
        initial_state="zero"  # |0⟩
    )
    
    # Second qubit - initially in |0⟩ 
    qubit_b = QuantumNode(
        name="QubitB",
        num_qubits=1,
        initial_state="zero"
    )
    
    # Entanglement creation node
    bell_state = QuantumNode(
        name="BellState",
        num_qubits=2,
        parents=[qubit_a, qubit_b],
        quantum_operations=[
            HadamardGate(qubit=0),  # Put first qubit in superposition
            CNOTGate(control_qubit=0, target_qubit=1)  # Create entanglement
        ]
    )
    
    network.add_nodes([qubit_a, qubit_b, bell_state])
    
    return network

# Create and test Bell state
bell_network = create_bell_state_network()
reasoner = ProbabilisticQuantumReasoner(backend="qiskit")

# Measure correlations
correlations = []
for _ in range(1000):
    measurement = reasoner.measure(
        network=bell_network,
        nodes=["QubitA", "QubitB"]
    )
    
    # Check if measurements are correlated
    a_result = measurement["QubitA"]
    b_result = measurement["QubitB"]
    correlations.append(a_result == b_result)

correlation_rate = np.mean(correlations)
print(f"Correlation rate: {correlation_rate:.3f}")
print("Perfect correlation expected: 1.000")
```

### Multi-Qubit Entanglement

```python
def create_ghz_state_network(num_qubits=3):
    """Create a GHZ (Greenberger-Horne-Zeilinger) state network."""
    
    network = BayesianNetwork(name=f"GHZ-{num_qubits} Network")
    
    # Create individual qubit nodes
    qubits = []
    for i in range(num_qubits):
        qubit = QuantumNode(
            name=f"Qubit_{i}",
            num_qubits=1,
            initial_state="zero"
        )
        qubits.append(qubit)
    
    # Create GHZ state: |000...⟩ + |111...⟩
    operations = [HadamardGate(qubit=0)]  # Put first qubit in superposition
    
    # CNOT gates to entangle all qubits with the first one
    for i in range(1, num_qubits):
        operations.append(CNOTGate(control_qubit=0, target_qubit=i))
    
    ghz_state = QuantumNode(
        name="GHZ_State",
        num_qubits=num_qubits,
        parents=qubits,
        quantum_operations=operations
    )
    
    network.add_nodes(qubits + [ghz_state])
    
    return network

# Test GHZ state correlations
ghz_network = create_ghz_state_network(num_qubits=4)

# Measure all qubits multiple times
measurements = []
for _ in range(1000):
    result = reasoner.measure(
        network=ghz_network,
        nodes=[f"Qubit_{i}" for i in range(4)]
    )
    
    # Check if all qubits have the same value
    values = [result[f"Qubit_{i}"] for i in range(4)]
    all_same = len(set(values)) == 1
    measurements.append(all_same)

ghz_correlation = np.mean(measurements)
print(f"GHZ correlation rate: {ghz_correlation:.3f}")
```

## Entanglement in Causal Networks

### Quantum Causal Models

```python
from probabilistic_quantum_reasoner.nodes import DiscreteNode

def create_quantum_causal_network():
    """Create a causal network with quantum entanglement."""
    
    network = BayesianNetwork(name="Quantum Causal Network")
    
    # Classical cause variable
    cause = DiscreteNode(
        name="Cause",
        states=["present", "absent"],
        prior=[0.3, 0.7]
    )
    
    # Quantum mediator variables (entangled)
    mediator_a = QuantumNode(
        name="MediatorA",
        num_qubits=1,
        initial_state="zero"
    )
    
    mediator_b = QuantumNode(
        name="MediatorB", 
        num_qubits=1,
        initial_state="zero"
    )
    
    # Create entanglement between mediators
    entangled_mediators = QuantumNode(
        name="EntangledMediators",
        num_qubits=2,
        parents=[mediator_a, mediator_b],
        quantum_operations=[
            HadamardGate(qubit=0),
            CNOTGate(control_qubit=0, target_qubit=1)
        ]
    )
    
    # Quantum-influenced effects
    effect_a = DiscreteNode(
        name="EffectA",
        states=["positive", "negative"],
        parents=[cause, entangled_mediators],
        # CPT depends on both classical cause and quantum measurement
        cpt=np.array([
            # Cause=present, Mediators=|00⟩
            [0.8, 0.2],
            # Cause=present, Mediators=|11⟩ 
            [0.9, 0.1],
            # Cause=absent, Mediators=|00⟩
            [0.3, 0.7],
            # Cause=absent, Mediators=|11⟩
            [0.4, 0.6]
        ])
    )
    
    effect_b = DiscreteNode(
        name="EffectB",
        states=["high", "low"],
        parents=[cause, entangled_mediators],
        cpt=np.array([
            # Symmetric effects due to entanglement
            [0.7, 0.3],  # Cause=present, Mediators=|00⟩
            [0.85, 0.15], # Cause=present, Mediators=|11⟩
            [0.2, 0.8],   # Cause=absent, Mediators=|00⟩ 
            [0.35, 0.65]  # Cause=absent, Mediators=|11⟩
        ])
    )
    
    network.add_nodes([
        cause, mediator_a, mediator_b, entangled_mediators,
        effect_a, effect_b
    ])
    
    return network

# Analyze causal relationships with entanglement
causal_network = create_quantum_causal_network()

# Test causal intervention
print("Causal Analysis with Quantum Entanglement:")
print("-" * 45)

# Without intervention
result_observational = reasoner.infer(
    network=causal_network,
    query=["EffectA", "EffectB"],
    evidence={}
)

print("Observational (no intervention):")
print(f"P(EffectA=positive): {result_observational['EffectA']['positive']:.3f}")
print(f"P(EffectB=high): {result_observational['EffectB']['high']:.3f}")

# With causal intervention on Cause
result_intervention = reasoner.infer(
    network=causal_network,
    query=["EffectA", "EffectB"],
    evidence={"Cause": "present"}
)

print("\nWith intervention (Cause=present):")
print(f"P(EffectA=positive): {result_intervention['EffectA']['positive']:.3f}")
print(f"P(EffectB=high): {result_intervention['EffectB']['high']:.3f}")
```

## Measuring Entanglement

### Entanglement Quantification

```python
from probabilistic_quantum_reasoner.metrics import (
    compute_concurrence,
    compute_entanglement_entropy,
    compute_quantum_mutual_information
)

def measure_network_entanglement(network, reasoner):
    """Measure entanglement properties of a quantum network."""
    
    # Get quantum state of the system
    state_vector = reasoner.get_state_vector(network)
    
    # Compute various entanglement measures
    measures = {}
    
    # Concurrence (for two-qubit systems)
    if network.num_qubits == 2:
        measures["concurrence"] = compute_concurrence(state_vector)
    
    # Entanglement entropy
    measures["entanglement_entropy"] = compute_entanglement_entropy(
        state_vector, 
        subsystem_size=1
    )
    
    # Quantum mutual information
    measures["quantum_mutual_info"] = compute_quantum_mutual_information(
        state_vector,
        subsystem_a=[0],
        subsystem_b=[1]
    )
    
    return measures

# Measure Bell state entanglement
bell_measures = measure_network_entanglement(bell_network, reasoner)
print(f"Bell State Entanglement Measures:")
print(f"Concurrence: {bell_measures['concurrence']:.3f}")
print(f"Entanglement Entropy: {bell_measures['entanglement_entropy']:.3f}")
print(f"Quantum Mutual Info: {bell_measures['quantum_mutual_info']:.3f}")

# Compare with separable state
separable_network = create_separable_network()
separable_measures = measure_network_entanglement(separable_network, reasoner)
print(f"\nSeparable State Entanglement Measures:")
print(f"Concurrence: {separable_measures['concurrence']:.3f}")
print(f"Entanglement Entropy: {separable_measures['entanglement_entropy']:.3f}")
```

### Dynamic Entanglement Analysis

```python
def analyze_entanglement_dynamics(network, time_steps=100):
    """Analyze how entanglement changes over time."""
    
    entanglement_history = []
    
    for t in range(time_steps):
        # Evolve the system (add noise, decoherence, etc.)
        network.evolve_time_step(dt=0.1)
        
        # Measure current entanglement
        measures = measure_network_entanglement(network, reasoner)
        entanglement_history.append(measures["concurrence"])
    
    return entanglement_history

# Analyze entanglement dynamics
import matplotlib.pyplot as plt

# Create network with decoherence
from probabilistic_quantum_reasoner.noise import AmplitudeDamping

noisy_network = create_bell_state_network()
noisy_network.add_noise_model(AmplitudeDamping(probability=0.02))

dynamics = analyze_entanglement_dynamics(noisy_network, time_steps=50)

plt.figure(figsize=(10, 6))
plt.plot(dynamics, 'b-', linewidth=2)
plt.xlabel('Time Steps')
plt.ylabel('Concurrence')
plt.title('Entanglement Decay Due to Decoherence')
plt.grid(True, alpha=0.3)
plt.show()
```

## Entanglement-Based Inference

### Quantum Advantage through Entanglement

```python
def demonstrate_quantum_advantage():
    """Demonstrate quantum advantage using entanglement."""
    
    # Classical network (no entanglement)
    classical_network = BayesianNetwork(name="Classical Network")
    
    var_a = DiscreteNode(
        name="VarA",
        states=["0", "1"],
        prior=[0.5, 0.5]
    )
    
    var_b = DiscreteNode(
        name="VarB",
        states=["0", "1"], 
        prior=[0.5, 0.5]
    )
    
    # Classical correlation (limited)
    corr_ab = DiscreteNode(
        name="CorrAB",
        states=["same", "different"],
        parents=[var_a, var_b],
        cpt=np.array([
            [0.8, 0.2],  # A=0, B=0 -> mostly same
            [0.3, 0.7],  # A=0, B=1 -> mostly different
            [0.3, 0.7],  # A=1, B=0 -> mostly different  
            [0.8, 0.2]   # A=1, B=1 -> mostly same
        ])
    )
    
    classical_network.add_nodes([var_a, var_b, corr_ab])
    
    # Quantum network (with entanglement)
    quantum_network = create_bell_state_network()
    
    # Compare inference accuracy
    test_cases = [
        {"evidence": {"VarA": "0"}, "query": "VarB"},
        {"evidence": {"VarB": "1"}, "query": "VarA"},
        {"evidence": {}, "query": "CorrAB"}
    ]
    
    print("Quantum Advantage Demonstration:")
    print("-" * 40)
    
    classical_reasoner = ProbabilisticQuantumReasoner(backend="classical")
    quantum_reasoner = ProbabilisticQuantumReasoner(backend="qiskit")
    
    for i, test in enumerate(test_cases):
        print(f"\nTest Case {i+1}: Evidence = {test['evidence']}")
        
        # Classical inference
        classical_result = classical_reasoner.infer(
            network=classical_network,
            query=[test['query']],
            evidence=test['evidence']
        )
        
        # Quantum inference (approximate due to measurement)
        quantum_measurements = []
        for _ in range(100):
            measurement = quantum_reasoner.measure(
                network=quantum_network,
                nodes=["QubitA", "QubitB"]
            )
            quantum_measurements.append(measurement)
        
        # Analyze quantum correlations
        quantum_correlation = np.mean([
            m["QubitA"] == m["QubitB"] for m in quantum_measurements
        ])
        
        print(f"Classical correlation strength: {classical_result}")
        print(f"Quantum correlation strength: {quantum_correlation:.3f}")
```

## Advanced Entanglement Techniques

### Quantum Error Correction with Entanglement

```python
def create_quantum_error_correction_network():
    """Create a network with quantum error correction using entanglement."""
    
    network = BayesianNetwork(name="Quantum Error Correction")
    
    # Logical qubit (encoded in 3 physical qubits)
    physical_qubits = []
    for i in range(3):
        qubit = QuantumNode(
            name=f"PhysicalQubit_{i}",
            num_qubits=1,
            initial_state="zero"
        )
        physical_qubits.append(qubit)
    
    # Logical qubit encoding (repetition code)
    logical_qubit = QuantumNode(
        name="LogicalQubit",
        num_qubits=3,
        parents=physical_qubits,
        quantum_operations=[
            # Encode logical |0⟩ as |000⟩ and logical |1⟩ as |111⟩
            CNOTGate(control_qubit=0, target_qubit=1),
            CNOTGate(control_qubit=0, target_qubit=2)
        ]
    )
    
    # Error detection nodes
    syndrome_1 = DiscreteNode(
        name="Syndrome_1",
        states=["no_error", "error"],
        parents=[logical_qubit],
        # Detects error between qubits 0 and 1
        cpt=np.array([
            [0.95, 0.05],  # No error case
            [0.1, 0.9]     # Error case
        ])
    )
    
    syndrome_2 = DiscreteNode(
        name="Syndrome_2", 
        states=["no_error", "error"],
        parents=[logical_qubit],
        # Detects error between qubits 1 and 2
        cpt=np.array([
            [0.95, 0.05],  # No error case
            [0.1, 0.9]     # Error case
        ])
    )
    
    # Error correction decision
    correction = DiscreteNode(
        name="Correction",
        states=["none", "qubit_0", "qubit_1", "qubit_2"],
        parents=[syndrome_1, syndrome_2],
        cpt=np.array([
            # S1=no_error, S2=no_error -> no correction
            [0.9, 0.033, 0.033, 0.033],
            # S1=no_error, S2=error -> correct qubit 2
            [0.1, 0.1, 0.1, 0.7],
            # S1=error, S2=no_error -> correct qubit 0
            [0.1, 0.7, 0.1, 0.1],
            # S1=error, S2=error -> correct qubit 1
            [0.1, 0.1, 0.7, 0.1]
        ])
    )
    
    network.add_nodes([
        *physical_qubits, logical_qubit,
        syndrome_1, syndrome_2, correction
    ])
    
    return network

# Test error correction
error_correction_network = create_quantum_error_correction_network()

# Simulate error and correction
print("Quantum Error Correction with Entanglement:")
print("-" * 45)

# Inject random errors
for trial in range(10):
    # Add noise to physical qubits
    for qubit in physical_qubits:
        if np.random.random() < 0.1:  # 10% error rate
            qubit.apply_pauli_x()  # Bit flip error
    
    # Run error detection and correction
    result = reasoner.infer(
        network=error_correction_network,
        query=["Syndrome_1", "Syndrome_2", "Correction"],
        evidence={}
    )
    
    correction_needed = max(result["Correction"], key=result["Correction"].get)
    
    print(f"Trial {trial+1}: Correction = {correction_needed}")
```

### Entanglement Swapping

```python
def create_entanglement_swapping_network():
    """Create a network demonstrating entanglement swapping."""
    
    network = BayesianNetwork(name="Entanglement Swapping")
    
    # Two independent Bell pairs
    # Pair 1: Qubits A and B
    qubit_a = QuantumNode(name="QubitA", num_qubits=1, initial_state="zero")
    qubit_b = QuantumNode(name="QubitB", num_qubits=1, initial_state="zero")
    
    bell_pair_1 = QuantumNode(
        name="BellPair1",
        num_qubits=2,
        parents=[qubit_a, qubit_b],
        quantum_operations=[
            HadamardGate(qubit=0),
            CNOTGate(control_qubit=0, target_qubit=1)
        ]
    )
    
    # Pair 2: Qubits C and D
    qubit_c = QuantumNode(name="QubitC", num_qubits=1, initial_state="zero")
    qubit_d = QuantumNode(name="QubitD", num_qubits=1, initial_state="zero")
    
    bell_pair_2 = QuantumNode(
        name="BellPair2",
        num_qubits=2,
        parents=[qubit_c, qubit_d],
        quantum_operations=[
            HadamardGate(qubit=0),
            CNOTGate(control_qubit=0, target_qubit=1)
        ]
    )
    
    # Bell measurement on qubits B and C (swapping operation)
    bell_measurement = QuantumNode(
        name="BellMeasurement",
        num_qubits=2,
        parents=[bell_pair_1, bell_pair_2],
        quantum_operations=[
            # Perform Bell measurement on qubits B and C
            CNOTGate(control_qubit=1, target_qubit=2),  # B -> C
            HadamardGate(qubit=1),                      # H on B
            # Measurement in computational basis
        ]
    )
    
    # Result: Qubits A and D become entangled
    swapped_pair = QuantumNode(
        name="SwappedPair",
        num_qubits=2,
        parents=[bell_measurement],
        # The entanglement is now between A and D
    )
    
    network.add_nodes([
        qubit_a, qubit_b, qubit_c, qubit_d,
        bell_pair_1, bell_pair_2, bell_measurement, swapped_pair
    ])
    
    return network

# Test entanglement swapping
swapping_network = create_entanglement_swapping_network()

# Verify that A and D are entangled after swapping
print("Entanglement Swapping Results:")
print("-" * 30)

correlations_ad = []
for _ in range(1000):
    measurement = reasoner.measure(
        network=swapping_network,
        nodes=["QubitA", "QubitD"]
    )
    
    correlations_ad.append(measurement["QubitA"] == measurement["QubitD"])

correlation_rate_ad = np.mean(correlations_ad)
print(f"A-D correlation after swapping: {correlation_rate_ad:.3f}")
print("Expected correlation: ~1.000 (perfect entanglement)")
```

## Applications of Entanglement

### Quantum Cryptography

```python
def create_quantum_key_distribution_network():
    """Create a network for quantum key distribution using entanglement."""
    
    network = BayesianNetwork(name="Quantum Key Distribution")
    
    # Alice's qubit
    alice_qubit = QuantumNode(
        name="AliceQubit",
        num_qubits=1,
        initial_state="zero"
    )
    
    # Bob's qubit
    bob_qubit = QuantumNode(
        name="BobQubit",
        num_qubits=1,
        initial_state="zero"
    )
    
    # Entangled pair generation
    entangled_pair = QuantumNode(
        name="EntangledPair",
        num_qubits=2,
        parents=[alice_qubit, bob_qubit],
        quantum_operations=[
            HadamardGate(qubit=0),
            CNOTGate(control_qubit=0, target_qubit=1)
        ]
    )
    
    # Alice's measurement choice
    alice_basis = DiscreteNode(
        name="AliceBasis",
        states=["computational", "hadamard"],
        prior=[0.5, 0.5]
    )
    
    # Bob's measurement choice
    bob_basis = DiscreteNode(
        name="BobBasis",
        states=["computational", "hadamard"],
        prior=[0.5, 0.5]
    )
    
    # Measurement outcomes
    alice_outcome = DiscreteNode(
        name="AliceOutcome",
        states=["0", "1"],
        parents=[entangled_pair, alice_basis],
        # CPT depends on entangled state and measurement basis
        cpt=np.array([
            # Entangled state, computational basis
            [0.5, 0.5],  # Random for entangled qubits
            # Entangled state, Hadamard basis
            [0.5, 0.5]   # Also random but correlated
        ])
    )
    
    bob_outcome = DiscreteNode(
        name="BobOutcome",
        states=["0", "1"],
        parents=[entangled_pair, bob_basis, alice_basis, alice_outcome],
        # Bob's outcome correlated with Alice's when bases match
        cpt=np.array([
            # Matching bases -> perfect correlation
            [1.0, 0.0],  # If Alice gets 0, Bob gets 0
            [0.0, 1.0],  # If Alice gets 1, Bob gets 1
            # Non-matching bases -> random
            [0.5, 0.5],  # Random correlation
            [0.5, 0.5]
        ])
    )
    
    # Eavesdropping detection
    eavesdropper = DiscreteNode(
        name="Eavesdropper",
        states=["absent", "present"],
        prior=[0.9, 0.1]  # Assume low probability of eavesdropping
    )
    
    # Security check
    security_check = DiscreteNode(
        name="SecurityCheck",
        states=["secure", "compromised"],
        parents=[alice_outcome, bob_outcome, alice_basis, bob_basis, eavesdropper],
        # Security depends on correlation when bases match
        cpt=np.array([
            # No eavesdropper, matching bases, matching outcomes
            [0.95, 0.05],
            # No eavesdropper, matching bases, different outcomes  
            [0.1, 0.9],
            # Eavesdropper present
            [0.3, 0.7],
            [0.3, 0.7]
        ])
    )
    
    network.add_nodes([
        alice_qubit, bob_qubit, entangled_pair,
        alice_basis, bob_basis, alice_outcome, bob_outcome,
        eavesdropper, security_check
    ])
    
    return network

# Test quantum key distribution
qkd_network = create_quantum_key_distribution_network()

print("Quantum Key Distribution Security Analysis:")
print("-" * 45)

# Simulate key distribution protocol
for round_num in range(5):
    result = reasoner.infer(
        network=qkd_network,
        query=["SecurityCheck", "AliceOutcome", "BobOutcome"],
        evidence={"AliceBasis": "computational", "BobBasis": "computational"}
    )
    
    security_prob = result["SecurityCheck"]["secure"]
    
    print(f"Round {round_num+1}:")
    print(f"  Security probability: {security_prob:.3f}")
    print(f"  Alice outcome: {result['AliceOutcome']}")
    print(f"  Bob outcome: {result['BobOutcome']}")
```

## Best Practices for Entanglement

### Entanglement Preservation

1. **Minimize decoherence**: Use short circuit depths
2. **Error correction**: Implement quantum error correction
3. **Careful measurement**: Avoid premature measurements
4. **Noise mitigation**: Use error mitigation techniques

### Performance Optimization

1. **Efficient encodings**: Use compact entangled state representations
2. **Parallel processing**: Leverage quantum parallelism
3. **Smart routing**: Optimize entanglement distribution
4. **Resource management**: Track and allocate quantum resources

### Debugging Entangled Networks

```python
def debug_entanglement(network, reasoner):
    """Debug entanglement in a quantum network."""
    
    print("Entanglement Debug Information:")
    print("-" * 35)
    
    # Check network structure
    quantum_nodes = [node for node in network.nodes.values() 
                    if hasattr(node, 'num_qubits')]
    
    print(f"Number of quantum nodes: {len(quantum_nodes)}")
    print(f"Total qubits: {sum(node.num_qubits for node in quantum_nodes)}")
    
    # Measure entanglement properties
    for node in quantum_nodes:
        if hasattr(node, 'parents') and node.parents:
            print(f"\nNode: {node.name}")
            print(f"  Parents: {[p.name for p in node.parents]}")
            print(f"  Qubits: {node.num_qubits}")
            
            # Check for entangling operations
            entangling_ops = [op for op in node.quantum_operations 
                            if hasattr(op, 'control_qubit')]
            print(f"  Entangling operations: {len(entangling_ops)}")
    
    # Measure actual entanglement
    try:
        measures = measure_network_entanglement(network, reasoner)
        print(f"\nMeasured entanglement:")
        for measure, value in measures.items():
            print(f"  {measure}: {value:.3f}")
    except Exception as e:
        print(f"Error measuring entanglement: {e}")

# Debug example networks
debug_entanglement(bell_network, reasoner)
debug_entanglement(ghz_network, reasoner)
```

## Conclusion

Quantum entanglement provides powerful capabilities for probabilistic reasoning:

1. **Enhanced correlations** beyond classical limits
2. **Quantum advantage** in specific inference tasks  
3. **Novel algorithms** for optimization and sampling
4. **Security applications** in cryptography and communication

The Probabilistic Quantum Reasoner framework provides tools to create, manipulate, and reason with entangled quantum states, enabling exploration of quantum-enhanced probabilistic models.
