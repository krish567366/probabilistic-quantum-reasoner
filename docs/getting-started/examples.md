# Basic Examples

This page provides simple examples to get you started with the Probabilistic Quantum Reasoner library.

## Simple Quantum Network

Let's start with a basic quantum Bayesian network:

```python
import numpy as np
from probabilistic_quantum_reasoner import QuantumBayesianNetwork
from probabilistic_quantum_reasoner.backends import ClassicalSimulator

# Create a quantum Bayesian network
backend = ClassicalSimulator()
network = QuantumBayesianNetwork("SimpleExample", backend)

# Add a quantum node in superposition
weather = network.add_quantum_node(
    "weather",
    outcome_space=["sunny", "rainy"],
    initial_amplitudes=np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
)

# Add a classical stochastic node
mood = network.add_stochastic_node(
    "mood",
    outcome_space=["happy", "sad"]
)

# Connect them causally
network.add_edge(weather, mood)

# Perform inference
result = network.infer(evidence={"weather": "sunny"})
print(f"Mood distribution given sunny weather: {result.marginal_probabilities['mood']}")
```

## Quantum Superposition Example

Demonstrate quantum superposition effects:

```python
# Create a node in superposition
quantum_coin = network.add_quantum_node(
    "coin",
    outcome_space=["heads", "tails"],
    initial_amplitudes=np.array([0.8, 0.6], dtype=complex)  # Biased quantum coin
)

# Measure the quantum state
measurement_result = network.measure_node("coin")
print(f"Coin measurement: {measurement_result}")

# The superposition collapses after measurement
post_measurement = network.infer(query_nodes=["coin"])
print(f"Post-measurement distribution: {post_measurement.marginal_probabilities['coin']}")
```

## Entanglement Example

Create entangled quantum nodes:

```python
# Add two quantum nodes
alice = network.add_quantum_node(
    "alice_bit",
    outcome_space=[0, 1],
    initial_amplitudes=np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
)

bob = network.add_quantum_node(
    "bob_bit", 
    outcome_space=[0, 1],
    initial_amplitudes=np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
)

# Entangle them
network.entangle([alice, bob])

# Now measuring one affects the other
alice_result = network.measure_node("alice_bit")
bob_distribution = network.infer(query_nodes=["bob_bit"])

print(f"Alice measured: {alice_result}")
print(f"Bob's distribution after Alice's measurement: {bob_distribution.marginal_probabilities['bob_bit']}")
```

## Causal Intervention

Perform do-calculus interventions:

```python
# Original network inference
original = network.infer(query_nodes=["mood"])
print(f"Natural mood distribution: {original.marginal_probabilities['mood']}")

# Intervention: force weather to be sunny
intervention_result = network.intervene(
    interventions={"weather": "sunny"},
    query_nodes=["mood"]
)
print(f"Mood under intervention do(weather=sunny): {intervention_result.marginal_probabilities['mood']}")
```

## Hybrid Reasoning

Combine quantum and classical reasoning:

```python
# Add hybrid node (part quantum, part classical)
energy = network.add_hybrid_node(
    "energy_level",
    outcome_space=["low", "medium", "high"],
    mixing_parameter=0.7  # 70% quantum, 30% classical
)

# Connect weather and mood to energy
network.add_edge(weather, energy)
network.add_edge(mood, energy)

# Inference considers both quantum and classical effects
result = network.infer(
    evidence={"weather": "sunny", "mood": "happy"},
    query_nodes=["energy_level"]
)
print(f"Energy level distribution: {result.marginal_probabilities['energy_level']}")
```

## Next Steps

- Explore more complex examples in the [Examples](../examples/weather-mood.md) section
- Learn about [building networks](../guide/networks.md) in detail
- Understand [quantum nodes](../guide/quantum-nodes.md) and their properties
- Try [causal reasoning](../guide/causal.md) with interventions

## Common Patterns

### Pattern 1: Weather Prediction
Combine meteorological data with quantum uncertainty modeling.

### Pattern 2: Medical Diagnosis  
Use quantum superposition to model diagnostic uncertainty.

### Pattern 3: Financial Risk
Model market volatility using quantum probability distributions.

### Pattern 4: Game Theory
Analyze strategic decisions with quantum Nash equilibria.

!!! tip "Pro Tip"
    Start with classical simulation backends for development, then switch to quantum hardware for production runs.

!!! warning "Memory Usage"
    Quantum state spaces grow exponentially. For n qubits, you need 2^n complex amplitudes in memory.
