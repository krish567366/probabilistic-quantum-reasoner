# Building Networks

This guide shows you how to construct and configure quantum Bayesian networks for various reasoning tasks.

## Network Creation

### Basic Network Setup

Start by creating a quantum Bayesian network with your chosen backend:

```python
from probabilistic_quantum_reasoner import QuantumBayesianNetwork
from probabilistic_quantum_reasoner.backends import ClassicalSimulator

# Initialize backend
backend = ClassicalSimulator(seed=42)

# Create network
network = QuantumBayesianNetwork("MyNetwork", backend)
```

### Network Properties

Access network properties and structure:

```python
# Network information
print(f"Network name: {network.name}")
print(f"Number of nodes: {len(network.nodes)}")
print(f"Number of edges: {len(network.edges)}")
print(f"Backend type: {type(network.backend).__name__}")

# List all nodes
for node_id, node in network.nodes.items():
    print(f"Node {node_id}: {type(node).__name__}")
```

## Node Types

### Quantum Nodes

Quantum nodes represent variables in quantum superposition:

```python
import numpy as np

# Simple binary quantum node
weather = network.add_quantum_node(
    node_id="weather",
    outcome_space=["sunny", "rainy"],
    name="Weather Condition",
    initial_amplitudes=np.array([0.6, 0.8], dtype=complex)  # Not normalized
)

# Multi-state quantum node
energy = network.add_quantum_node(
    node_id="energy_level", 
    outcome_space=["low", "medium", "high"],
    name="Energy Level",
    initial_amplitudes=np.array([1, 1, 1], dtype=complex) / np.sqrt(3)
)

# Access quantum properties
print(f"Weather amplitudes: {weather.quantum_state.amplitudes}")
print(f"Weather probabilities: {weather.probability_distribution}")
print(f"Energy state dimension: {energy.quantum_state.dimension}")
```

**Key Features:**

- Complex amplitude vectors
- Quantum superposition states
- Unitary evolution capability
- Born rule measurement

### Stochastic Nodes

Classical probabilistic nodes with standard probability distributions:

```python
# Binary stochastic node
mood = network.add_stochastic_node(
    node_id="mood",
    outcome_space=["happy", "sad"],
    name="Mood State"
)

# Set prior distribution
mood.prior_distribution = np.array([0.7, 0.3])  # 70% happy, 30% sad

# Multi-valued stochastic node
activity = network.add_stochastic_node(
    node_id="activity",
    outcome_space=["work", "leisure", "sleep"],
    name="Daily Activity"
)

# Uniform prior
activity.prior_distribution = np.ones(3) / 3
```

### Hybrid Nodes

Combine quantum and classical reasoning:

```python
# Hybrid node mixing quantum and classical behavior
decision = network.add_hybrid_node(
    node_id="decision",
    outcome_space=["option_a", "option_b", "option_c"],
    name="Decision Making",
    mixing_parameter=0.8  # 80% quantum, 20% classical
)

# Access hybrid components
quantum_part = decision.quantum_component
classical_part = decision.classical_component
mixing_ratio = decision.mixing_parameter

print(f"Mixing parameter: {mixing_ratio}")
print(f"Hybrid distribution: {decision.get_hybrid_distribution()}")
```

## Network Structure

### Adding Edges

Create causal relationships between nodes:

```python
# Simple parent-child relationship
network.add_edge(weather, mood)  # Weather influences mood

# Multiple parents
network.add_edge(weather, activity)
network.add_edge(mood, activity)  # Both weather and mood influence activity

# Check network structure
print(f"Weather children: {network.get_children('weather')}")
print(f"Activity parents: {network.get_parents('activity')}")
```

### Conditional Dependencies

Set up conditional probability tables for classical dependencies:

```python
from probabilistic_quantum_reasoner.core.nodes import ConditionalProbabilityTable

# Create CPT for mood given weather
mood_cpt = ConditionalProbabilityTable(
    child_outcomes=["happy", "sad"],
    parent_outcomes=[["sunny", "rainy"]]  # One parent: weather
)

# Set conditional probabilities
# P(mood=happy | weather=sunny) = 0.9
mood_cpt.set_probability("happy", ("sunny",), 0.9)
mood_cpt.set_probability("sad", ("sunny",), 0.1)

# P(mood=happy | weather=rainy) = 0.3  
mood_cpt.set_probability("happy", ("rainy",), 0.3)
mood_cpt.set_probability("sad", ("rainy",), 0.7)

# Assign CPT to node
mood.conditional_probability_table = mood_cpt
```

### Multi-Parent Dependencies

Handle nodes with multiple parents:

```python
# Activity depends on both weather and mood
activity_cpt = ConditionalProbabilityTable(
    child_outcomes=["work", "leisure", "sleep"],
    parent_outcomes=[["sunny", "rainy"], ["happy", "sad"]]  # Two parents
)

# Set all conditional probabilities
activity_cpt.set_probability("work", ("sunny", "happy"), 0.6)
activity_cpt.set_probability("leisure", ("sunny", "happy"), 0.3)
activity_cpt.set_probability("sleep", ("sunny", "happy"), 0.1)

activity_cpt.set_probability("work", ("sunny", "sad"), 0.2)
activity_cpt.set_probability("leisure", ("sunny", "sad"), 0.3)
activity_cpt.set_probability("sleep", ("sunny", "sad"), 0.5)

# Continue for all parent combinations...
activity_cpt.set_probability("work", ("rainy", "happy"), 0.3)
activity_cpt.set_probability("leisure", ("rainy", "happy"), 0.6)
activity_cpt.set_probability("sleep", ("rainy", "happy"), 0.1)

activity_cpt.set_probability("work", ("rainy", "sad"), 0.1)
activity_cpt.set_probability("leisure", ("rainy", "sad"), 0.2)
activity_cpt.set_probability("sleep", ("rainy", "sad"), 0.7)

activity.conditional_probability_table = activity_cpt
```

## Quantum-Specific Features

### Quantum Entanglement

Create entangled quantum nodes:

```python
# Add two quantum nodes
alice_measurement = network.add_quantum_node(
    "alice",
    outcome_space=[0, 1],
    initial_amplitudes=np.array([1, 0], dtype=complex)
)

bob_measurement = network.add_quantum_node(
    "bob", 
    outcome_space=[0, 1],
    initial_amplitudes=np.array([1, 0], dtype=complex)
)

# Entangle them (creates Bell state)
network.entangle([alice_measurement, bob_measurement])

# Check entanglement
entangled_pairs = network.get_entangled_groups()
print(f"Entangled groups: {entangled_pairs}")
```

### Quantum Operations

Apply quantum operations to nodes:

```python
from probabilistic_quantum_reasoner.core.operators import QuantumGate

# Apply Hadamard gate to create superposition
hadamard = QuantumGate.hadamard()
network.apply_quantum_operator("weather", hadamard)

# Apply rotation gates
rotation_y = QuantumGate.rotation_y(np.pi/4)
network.apply_quantum_operator("energy_level", rotation_y)

# Check quantum state after operations
print(f"Weather after Hadamard: {weather.quantum_state.amplitudes}")
```

### Quantum Measurements

Perform quantum measurements:

```python
# Measure a quantum node
measurement_result = network.measure_node("weather")

print(f"Measurement outcome: {measurement_result['outcome']}")
print(f"Measurement probability: {measurement_result['probability']}")
print(f"Post-measurement state: {measurement_result['post_measurement_state']}")

# The quantum state collapses after measurement
print(f"Weather state after measurement: {weather.quantum_state.amplitudes}")
```

## Network Validation

### Structure Validation

Check network consistency:

```python
# Validate network structure
validation_result = network.validate()

if validation_result["valid"]:
    print("Network structure is valid")
else:
    print(f"Validation errors: {validation_result['errors']}")

# Check for cycles
has_cycles = network.has_cycles()
print(f"Network has cycles: {has_cycles}")

# Check connectivity
connected_components = network.get_connected_components()
print(f"Connected components: {len(connected_components)}")
```

### Probabilistic Consistency

Verify probability distributions:

```python
# Check if all CPTs are properly normalized
for node_id, node in network.nodes.items():
    if hasattr(node, 'conditional_probability_table'):
        cpt = node.conditional_probability_table
        if not cpt.is_normalized():
            print(f"Warning: CPT for {node_id} is not normalized")
            cpt.normalize()

# Check quantum state normalization
for node_id, node in network.nodes.items():
    if hasattr(node, 'quantum_state'):
        state = node.quantum_state
        norm = np.sum(np.abs(state.amplitudes) ** 2)
        if abs(norm - 1.0) > 1e-10:
            print(f"Warning: Quantum state for {node_id} is not normalized")
            state.normalize()
```

## Network Serialization

### Save and Load Networks

```python
# Save network to file
network.save("my_network.json")

# Load network from file
loaded_network = QuantumBayesianNetwork.load("my_network.json", backend)

# Export to different formats
network.export_to_graphml("network.graphml")  # For visualization
network.export_to_dot("network.dot")         # For Graphviz
```

### Network Copying

```python
# Create a copy of the network
network_copy = network.copy()

# Deep copy with independent quantum states
network_deep_copy = network.copy(deep=True)

# Copy with different backend
new_backend = QiskitBackend("aer_simulator")
network_with_new_backend = network.copy(backend=new_backend)
```

## Advanced Patterns

### Temporal Networks

Model temporal dependencies:

```python
# Time-series network
for t in range(5):  # 5 time steps
    weather_t = network.add_quantum_node(
        f"weather_t{t}",
        outcome_space=["sunny", "rainy"],
        initial_amplitudes=np.array([1, 1], dtype=complex) / np.sqrt(2)
    )
    
    if t > 0:
        # Add temporal dependency
        prev_weather = network.nodes[f"weather_t{t-1}"]
        network.add_edge(prev_weather, weather_t)
```

### Hierarchical Networks

Create hierarchical structures:

```python
# High-level concepts
climate = network.add_quantum_node("climate", ["tropical", "temperate", "arctic"])

# Mid-level features
season = network.add_stochastic_node("season", ["spring", "summer", "fall", "winter"])
geography = network.add_stochastic_node("geography", ["coastal", "inland", "mountain"])

# Low-level observations
temperature = network.add_hybrid_node("temperature", ["cold", "mild", "warm", "hot"])
humidity = network.add_hybrid_node("humidity", ["low", "medium", "high"])

# Build hierarchy
network.add_edge(climate, season)
network.add_edge(climate, geography)
network.add_edge(season, temperature)
network.add_edge(geography, temperature)
network.add_edge(geography, humidity)
```

### Modular Networks

Combine multiple sub-networks:

```python
# Create sub-networks
weather_module = create_weather_network(backend)
mood_module = create_mood_network(backend)
activity_module = create_activity_network(backend)

# Combine into larger network
combined_network = QuantumBayesianNetwork.combine([
    weather_module, mood_module, activity_module
], name="CombinedNetwork")

# Add cross-module connections
combined_network.add_edge("weather_module.weather", "mood_module.mood")
combined_network.add_edge("mood_module.mood", "activity_module.activity")
```

## Best Practices

### Design Guidelines

1. **Start Simple**: Begin with small networks and add complexity gradually
2. **Validate Early**: Check network structure before adding complex dependencies  
3. **Use Appropriate Node Types**: Choose quantum vs classical nodes based on uncertainty characteristics
4. **Monitor Memory**: Quantum state spaces grow exponentially
5. **Test Incrementally**: Validate each component before combining

### Performance Tips

```python
# Efficient network construction
class NetworkBuilder:
    def __init__(self, backend):
        self.backend = backend
        self.network = QuantumBayesianNetwork("EfficientNetwork", backend)
    
    def add_batch_nodes(self, node_specs):
        """Add multiple nodes efficiently."""
        for spec in node_specs:
            if spec["type"] == "quantum":
                self.network.add_quantum_node(**spec["params"])
            elif spec["type"] == "stochastic":
                self.network.add_stochastic_node(**spec["params"])
            elif spec["type"] == "hybrid":
                self.network.add_hybrid_node(**spec["params"])
    
    def build(self):
        """Finalize and return network."""
        self.network.validate()
        return self.network
```

### Common Pitfalls

1. **Unormalized States**: Always ensure quantum amplitudes are normalized
2. **Inconsistent CPTs**: Verify conditional probability tables sum to 1
3. **Circular Dependencies**: Check for cycles before inference
4. **Memory Overflow**: Monitor exponential growth in quantum state spaces
5. **Backend Mismatch**: Ensure operations are supported by chosen backend

## Next Steps

- Learn about [Quantum Nodes](quantum-nodes.md) in detail
- Explore [Inference Methods](inference.md)
- See [Causal Reasoning](causal.md) examples
- Check [API Reference](../api/core.md) for complete documentation
