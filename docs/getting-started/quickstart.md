# Quick Start Guide

This guide will walk you through creating your first quantum Bayesian network and performing inference.

## Your First Quantum Network

Let's build a simple network that models the relationship between weather and mood using quantum superposition.

### Step 1: Import Required Components

```python
import numpy as np
from probabilistic_quantum_reasoner import QuantumBayesianNetwork
from probabilistic_quantum_reasoner.backends import ClassicalSimulator
```

### Step 2: Create the Network

```python
# Initialize with classical simulator (no quantum hardware needed)
backend = ClassicalSimulator()
network = QuantumBayesianNetwork("WeatherMoodModel", backend)
```

### Step 3: Add Quantum Variables

```python
# Weather as a quantum variable in superposition
weather = network.add_quantum_node(
    "weather",
    outcome_space=["sunny", "cloudy", "rainy"],
    name="Weather Conditions",
    # Initial amplitudes represent quantum uncertainty
    initial_amplitudes=np.array([0.6, 0.5, 0.4], dtype=complex)
)

# Mood influenced by weather (classical variable)
mood = network.add_stochastic_node(
    "mood",
    outcome_space=["happy", "neutral", "sad"],
    name="Emotional State"
)
```

### Step 4: Define Relationships

```python
# Weather influences mood
network.add_edge(weather, mood)

# Set conditional probabilities for mood given weather
mood_cpt = {
    ("sunny",): {"happy": 0.8, "neutral": 0.15, "sad": 0.05},
    ("cloudy",): {"happy": 0.3, "neutral": 0.5, "sad": 0.2},
    ("rainy",): {"happy": 0.1, "neutral": 0.3, "sad": 0.6}
}
network.set_conditional_probability_table(mood, mood_cpt)
```

### Step 5: Perform Inference

```python
# Query: What's the probability of being happy?
result = network.infer(query_nodes=["mood"])
print("Unconditional mood probabilities:")
for state, prob in result.marginal_probabilities["mood"].items():
    print(f"  P(mood={state}) = {prob:.3f}")

# Conditional inference: Given it's sunny, what's P(happy)?
sunny_result = network.infer(
    evidence={"weather": "sunny"},
    query_nodes=["mood"]
)
print(f"\nP(mood=happy|weather=sunny) = {sunny_result.marginal_probabilities['mood']['happy']:.3f}")
```

### Step 6: Quantum Effects

```python
# Demonstrate quantum superposition
print("\nQuantum weather state:")
weather_state = network.get_quantum_state("weather")
for i, outcome in enumerate(["sunny", "cloudy", "rainy"]):
    amplitude = weather_state.amplitudes[i]
    probability = abs(amplitude) ** 2
    print(f"  |{outcome}⟩: amplitude={amplitude:.3f}, P={probability:.3f}")
```

## Expected Output

```mermaid
Unconditional mood probabilities:
  P(mood=happy) = 0.542
  P(mood=neutral) = 0.312
  P(mood=sad) = 0.146

P(mood=happy|weather=sunny) = 0.800

Quantum weather state:
  |sunny⟩: amplitude=0.600+0.000j, P=0.360
  |cloudy⟩: amplitude=0.500+0.000j, P=0.250
  |rainy⟩: amplitude=0.400+0.000j, P=0.160
```

## Key Concepts Demonstrated

### Quantum Superposition

The weather variable exists in a superposition of all possible states, with amplitudes determining the probability of each outcome.

### Hybrid Reasoning

Classical probabilistic reasoning (conditional probability tables) combined with quantum uncertainty representation.

### Born Rule

Quantum probabilities follow the Born rule: P(outcome) = |amplitude|²

## Next Steps

Now that you've created your first quantum Bayesian network, explore:

1. **[Quantum Entanglement](../guide/quantum-nodes.md#entanglement)**: Create correlated quantum variables
2. **[Causal Inference](../guide/causal.md)**: Perform interventions and counterfactual reasoning
3. **[Advanced Examples](examples.md)**: More complex real-world scenarios
4. **[Quantum Backends](../architecture/backends.md)**: Using real quantum hardware

## Common Patterns

### Evidence-Based Reasoning

```python
# Multiple evidence variables
evidence = {"weather": "rainy", "temperature": "cold"}
result = network.infer(evidence=evidence, query_nodes=["mood", "activity"])
```

### Batch Inference

```python
# Multiple queries at once
queries = ["mood", "activity", "energy_level"]
result = network.infer(query_nodes=queries)
```

### Temporal Reasoning

```python
# Add time-dependent variables
morning_weather = network.add_quantum_node("morning_weather", ...)
evening_mood = network.add_stochastic_node("evening_mood", ...)
network.add_edge(morning_weather, evening_mood)
```

Continue with [Basic Examples](examples.md) to see more sophisticated use cases.
