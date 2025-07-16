# Quantum Logic: XOR Gate Implementation

This example demonstrates how to implement quantum logic gates using the Probabilistic Quantum Reasoner, specifically focusing on the quantum XOR gate and its classical probabilistic equivalent.

## Overview

The XOR (exclusive OR) gate is a fundamental logical operation that outputs true when exactly one of its inputs is true. In quantum computing, we can implement XOR gates using quantum superposition and entanglement, while in classical probabilistic reasoning, we model it using conditional probability distributions.

## Classical Probabilistic XOR

Let's start with a classical probabilistic implementation of an XOR gate:

```python
import numpy as np
from probabilistic_quantum_reasoner import ProbabilisticQuantumReasoner
from probabilistic_quantum_reasoner.networks import BayesianNetwork
from probabilistic_quantum_reasoner.nodes import DiscreteNode

def create_classical_xor():
    """Create a classical probabilistic XOR network."""
    
    # Create the network
    network = BayesianNetwork(name="Classical XOR")
    
    # Define input nodes A and B
    node_a = DiscreteNode(
        name="A",
        states=["false", "true"],
        prior=[0.5, 0.5]  # Uniform distribution
    )
    
    node_b = DiscreteNode(
        name="B", 
        states=["false", "true"],
        prior=[0.5, 0.5]  # Uniform distribution
    )
    
    # Define XOR output node
    # XOR is true when exactly one input is true
    xor_cpt = np.array([
        # A=false, B=false -> XOR=false (prob=1.0)
        [1.0, 0.0],
        # A=false, B=true -> XOR=true (prob=1.0)  
        [0.0, 1.0],
        # A=true, B=false -> XOR=true (prob=1.0)
        [0.0, 1.0],
        # A=true, B=true -> XOR=false (prob=1.0)
        [1.0, 0.0]
    ])
    
    node_xor = DiscreteNode(
        name="XOR",
        states=["false", "true"],
        parents=[node_a, node_b],
        cpt=xor_cpt
    )
    
    # Add nodes to network
    network.add_nodes([node_a, node_b, node_xor])
    
    return network

# Create and test the classical XOR
classical_xor = create_classical_xor()
reasoner = ProbabilisticQuantumReasoner(backend="classical")

# Test all input combinations
test_cases = [
    ({"A": "false", "B": "false"}, "false"),
    ({"A": "false", "B": "true"}, "true"),
    ({"A": "true", "B": "false"}, "true"), 
    ({"A": "true", "B": "true"}, "false")
]

print("Classical XOR Truth Table:")
print("A\tB\tXOR\tProbability")
print("-" * 30)

for evidence, expected in test_cases:
    result = reasoner.infer(
        network=classical_xor,
        query=["XOR"],
        evidence=evidence
    )
    prob_true = result["XOR"]["true"]
    predicted = "true" if prob_true > 0.5 else "false"
    print(f"{evidence['A']}\t{evidence['B']}\t{predicted}\t{prob_true:.3f}")
```

## Quantum XOR Implementation

Now let's implement a quantum version using superposition:

```python
from probabilistic_quantum_reasoner.nodes import QuantumNode
from probabilistic_quantum_reasoner.quantum_ops import HadamardGate, CNOTGate, PauliXGate

def create_quantum_xor():
    """Create a quantum XOR network using superposition."""
    
    network = BayesianNetwork(name="Quantum XOR")
    
    # Create quantum input nodes in superposition
    node_a = QuantumNode(
        name="A_quantum",
        num_qubits=1,
        initial_state="superposition"  # |+⟩ = (|0⟩ + |1⟩)/√2
    )
    
    node_b = QuantumNode(
        name="B_quantum", 
        num_qubits=1,
        initial_state="superposition"  # |+⟩ = (|0⟩ + |1⟩)/√2
    )
    
    # Create entangled XOR output using CNOT operations
    # This creates the quantum equivalent of XOR logic
    node_xor = QuantumNode(
        name="XOR_quantum",
        num_qubits=1,
        parents=[node_a, node_b],
        quantum_operations=[
            # Apply CNOT with A as control, XOR as target
            CNOTGate(control_qubit=0, target_qubit=2),
            # Apply CNOT with B as control, XOR as target  
            CNOTGate(control_qubit=1, target_qubit=2)
        ]
    )
    
    network.add_nodes([node_a, node_b, node_xor])
    
    return network

# Create quantum XOR
quantum_xor = create_quantum_xor()
quantum_reasoner = ProbabilisticQuantumReasoner(backend="qiskit")

print("\nQuantum XOR with Superposition:")
print("Measuring XOR output from superposed inputs...")

# Measure the quantum XOR multiple times
measurements = []
for i in range(1000):
    result = quantum_reasoner.measure(
        network=quantum_xor,
        nodes=["XOR_quantum"]
    )
    measurements.append(result["XOR_quantum"])

# Analyze results
false_count = measurements.count("false")
true_count = measurements.count("true")
total = len(measurements)

print(f"Results from {total} measurements:")
print(f"False: {false_count} ({false_count/total:.3f})")
print(f"True: {true_count} ({true_count/total:.3f})")
```

## Quantum-Classical Hybrid XOR

We can also create a hybrid approach that combines quantum and classical reasoning:

```python
def create_hybrid_xor():
    """Create a hybrid quantum-classical XOR network."""
    
    network = BayesianNetwork(name="Hybrid XOR")
    
    # Classical probabilistic inputs
    classical_a = DiscreteNode(
        name="classical_A",
        states=["false", "true"],
        prior=[0.3, 0.7]  # Biased towards true
    )
    
    # Quantum superposition input
    quantum_b = QuantumNode(
        name="quantum_B",
        num_qubits=1,
        initial_state="superposition"
    )
    
    # Hybrid XOR node that combines classical and quantum inputs
    hybrid_xor = DiscreteNode(
        name="hybrid_XOR",
        states=["false", "true"],
        parents=[classical_a, quantum_b],
        # CPT handles quantum measurement outcomes
        cpt=np.array([
            # classical_A=false, quantum_B=|0⟩ -> XOR=false
            [1.0, 0.0],
            # classical_A=false, quantum_B=|1⟩ -> XOR=true
            [0.0, 1.0],
            # classical_A=true, quantum_B=|0⟩ -> XOR=true
            [0.0, 1.0],
            # classical_A=true, quantum_B=|1⟩ -> XOR=false
            [1.0, 0.0]
        ])
    )
    
    network.add_nodes([classical_a, quantum_b, hybrid_xor])
    
    return network

# Test hybrid XOR
hybrid_xor = create_hybrid_xor()
hybrid_reasoner = ProbabilisticQuantumReasoner(backend="pennylane")

print("\nHybrid Quantum-Classical XOR:")
results = []

for i in range(100):
    # Each inference involves quantum measurement
    result = hybrid_reasoner.infer(
        network=hybrid_xor,
        query=["hybrid_XOR"],
        evidence={}  # No evidence, let quantum measurement determine B
    )
    results.append(result["hybrid_XOR"]["true"])

avg_prob = np.mean(results)
std_prob = np.std(results)

print(f"Average P(XOR=true): {avg_prob:.3f} ± {std_prob:.3f}")
print(f"Expected (classical A biased 0.7): ~0.35")
```

## Advanced: Noisy Quantum XOR

Real quantum systems have noise. Let's model a noisy quantum XOR:

```python
from probabilistic_quantum_reasoner.noise import DepolarizingNoise, AmplitudeDamping

def create_noisy_quantum_xor(noise_level=0.1):
    """Create a quantum XOR with realistic noise."""
    
    network = BayesianNetwork(name="Noisy Quantum XOR")
    
    # Noisy quantum inputs
    node_a = QuantumNode(
        name="noisy_A",
        num_qubits=1,
        initial_state="superposition",
        noise_model=DepolarizingNoise(probability=noise_level)
    )
    
    node_b = QuantumNode(
        name="noisy_B",
        num_qubits=1, 
        initial_state="superposition",
        noise_model=AmplitudeDamping(probability=noise_level)
    )
    
    # XOR with noise affecting the computation
    node_xor = QuantumNode(
        name="noisy_XOR",
        num_qubits=1,
        parents=[node_a, node_b],
        quantum_operations=[
            CNOTGate(control_qubit=0, target_qubit=2),
            CNOTGate(control_qubit=1, target_qubit=2)
        ],
        noise_model=DepolarizingNoise(probability=noise_level)
    )
    
    network.add_nodes([node_a, node_b, node_xor])
    
    return network

# Compare different noise levels
noise_levels = [0.0, 0.05, 0.1, 0.2, 0.3]

print("\nNoisy Quantum XOR Analysis:")
print("Noise Level\tP(XOR=true)\tFidelity")
print("-" * 35)

for noise in noise_levels:
    noisy_xor = create_noisy_quantum_xor(noise)
    noisy_reasoner = ProbabilisticQuantumReasoner(backend="qiskit")
    
    # Measure multiple times
    measurements = []
    for _ in range(500):
        result = noisy_reasoner.measure(
            network=noisy_xor,
            nodes=["noisy_XOR"]
        )
        measurements.append(result["noisy_XOR"] == "true")
    
    prob_true = np.mean(measurements)
    # Ideal probability for superposition XOR is 0.5
    fidelity = 1 - 2 * abs(prob_true - 0.5)
    
    print(f"{noise:.2f}\t\t{prob_true:.3f}\t\t{fidelity:.3f}")
```

## Performance Analysis

Let's analyze the computational complexity and accuracy:

```python
import time
import matplotlib.pyplot as plt

def benchmark_xor_implementations():
    """Benchmark different XOR implementations."""
    
    implementations = {
        "Classical": create_classical_xor,
        "Quantum": create_quantum_xor,
        "Hybrid": create_hybrid_xor
    }
    
    results = {}
    
    for name, create_func in implementations.items():
        print(f"\nBenchmarking {name} XOR...")
        
        # Time network creation
        start_time = time.time()
        network = create_func()
        creation_time = time.time() - start_time
        
        # Choose appropriate backend
        if name == "Classical":
            reasoner = ProbabilisticQuantumReasoner(backend="classical")
        elif name == "Quantum":
            reasoner = ProbabilisticQuantumReasoner(backend="qiskit")
        else:  # Hybrid
            reasoner = ProbabilisticQuantumReasoner(backend="pennylane")
        
        # Time inference
        inference_times = []
        for _ in range(10):
            start_time = time.time()
            
            if name == "Classical":
                result = reasoner.infer(
                    network=network,
                    query=["XOR"],
                    evidence={"A": "true", "B": "false"}
                )
            else:
                result = reasoner.measure(
                    network=network,
                    nodes=[f"{name.lower()}_XOR" if name == "Hybrid" else "XOR_quantum"]
                )
            
            inference_times.append(time.time() - start_time)
        
        avg_inference_time = np.mean(inference_times)
        
        results[name] = {
            "creation_time": creation_time,
            "avg_inference_time": avg_inference_time,
            "total_time": creation_time + avg_inference_time
        }
        
        print(f"  Creation time: {creation_time:.4f}s")
        print(f"  Avg inference time: {avg_inference_time:.4f}s")
    
    return results

# Run benchmark
benchmark_results = benchmark_xor_implementations()

# Plot results
names = list(benchmark_results.keys())
creation_times = [benchmark_results[name]["creation_time"] for name in names]
inference_times = [benchmark_results[name]["avg_inference_time"] for name in names]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Creation time comparison
ax1.bar(names, creation_times, color=['blue', 'red', 'green'])
ax1.set_ylabel('Time (seconds)')
ax1.set_title('Network Creation Time')
ax1.set_yscale('log')

# Inference time comparison  
ax2.bar(names, inference_times, color=['blue', 'red', 'green'])
ax2.set_ylabel('Time (seconds)')
ax2.set_title('Average Inference Time')
ax2.set_yscale('log')

plt.tight_layout()
plt.savefig('xor_performance_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nPerformance Summary:")
for name, metrics in benchmark_results.items():
    print(f"{name}: {metrics['total_time']:.4f}s total")
```

## Theoretical Analysis

### Quantum Advantage

The quantum XOR implementation demonstrates several key quantum computing concepts:

1. **Superposition**: Input qubits exist in superposition states
2. **Entanglement**: CNOT gates create entangled states  
3. **Interference**: Quantum amplitudes can interfere constructively/destructively
4. **Measurement**: Quantum states collapse to classical outputs

### Complexity Analysis

- **Classical XOR**: O(1) time, O(n) space for CPT storage
- **Quantum XOR**: O(1) quantum gates, O(2^n) classical simulation overhead  
- **Hybrid XOR**: Combines benefits of both approaches

### Error Analysis

Quantum implementations are susceptible to:
- **Decoherence**: Loss of quantum coherence over time
- **Gate errors**: Imperfect quantum gate operations
- **Measurement errors**: Incorrect state readout

## Applications

### Cryptography

Quantum XOR gates are fundamental for:
- Quantum key distribution protocols
- Random number generation
- One-time pad encryption

### Machine Learning

XOR problems demonstrate:
- Non-linear classification capabilities
- Feature entanglement in quantum neural networks
- Hybrid quantum-classical optimization

### Error Correction

XOR operations are crucial for:
- Quantum error correction codes
- Syndrome detection and correction
- Fault-tolerant quantum computation

## Best Practices

1. **Use classical simulation for debugging** before quantum hardware
2. **Implement noise models** to simulate realistic conditions
3. **Validate results** against analytical expectations
4. **Monitor quantum resource usage** (gates, qubits, time)
5. **Consider hybrid approaches** for practical applications

## Next Steps

- Explore multi-qubit XOR implementations
- Implement quantum Fourier transform for XOR
- Study quantum error correction for XOR circuits
- Develop variational quantum XOR algorithms

This example provides a comprehensive foundation for understanding quantum logic operations within the Probabilistic Quantum Reasoner framework.
