# Inference Methods

This guide covers the various inference algorithms available in the Probabilistic Quantum Reasoner for extracting information from quantum-classical hybrid networks.

## Overview of Inference

Inference in quantum Bayesian networks involves computing probability distributions over query variables given evidence, leveraging both classical probabilistic reasoning and quantum amplitude-based computation.

## Basic Inference

### Simple Queries

```python
from probabilistic_quantum_reasoner import QuantumBayesianNetwork
from probabilistic_quantum_reasoner.backends import ClassicalSimulator
import numpy as np

# Create example network
backend = ClassicalSimulator()
network = QuantumBayesianNetwork("InferenceDemo", backend)

# Add nodes
weather = network.add_quantum_node(
    "weather",
    outcome_space=["sunny", "rainy"],
    initial_amplitudes=np.array([0.8, 0.6], dtype=complex)
)

mood = network.add_stochastic_node("mood", outcome_space=["happy", "sad"])
activity = network.add_hybrid_node("activity", outcome_space=["indoor", "outdoor"], mixing_parameter=0.7)

# Add edges
network.add_edge(weather, mood)
network.add_edge(mood, activity)

# Basic inference without evidence
result = network.infer(query_nodes=["mood", "activity"])
print(f"Mood distribution: {result.marginal_probabilities['mood']}")
print(f"Activity distribution: {result.marginal_probabilities['activity']}")
```

### Conditional Inference

```python
# Inference with evidence
evidence = {"weather": "sunny"}
conditional_result = network.infer(
    query_nodes=["mood", "activity"],
    evidence=evidence
)

print(f"Mood given sunny weather: {conditional_result.marginal_probabilities['mood']}")
print(f"Activity given sunny weather: {conditional_result.marginal_probabilities['activity']}")

# Multiple evidence variables
multi_evidence = {"weather": "sunny", "mood": "happy"}
multi_result = network.infer(
    query_nodes=["activity"],
    evidence=multi_evidence
)
print(f"Activity given sunny weather and happy mood: {multi_result.marginal_probabilities['activity']}")
```

### Joint Distributions

```python
# Joint probability distributions
joint_result = network.infer(
    query_nodes=["weather", "mood"],
    return_joint=True
)

print("Joint distribution P(weather, mood):")
for weather_val in ["sunny", "rainy"]:
    for mood_val in ["happy", "sad"]:
        joint_prob = joint_result.joint_probabilities.get((weather_val, mood_val), 0)
        print(f"P(weather={weather_val}, mood={mood_val}) = {joint_prob:.3f}")
```

## Belief Propagation

### Quantum Message Passing

```python
# Use belief propagation algorithm explicitly
bp_result = network.infer(
    query_nodes=["activity"],
    evidence={"weather": "rainy"},
    method="belief_propagation",
    max_iterations=50,
    convergence_threshold=1e-6
)

print(f"Belief propagation result: {bp_result.marginal_probabilities['activity']}")
print(f"Converged: {bp_result.converged}")
print(f"Iterations: {bp_result.iterations}")
```

### Message Analysis

```python
# Access detailed message passing information
detailed_result = network.infer(
    query_nodes=["mood"],
    method="belief_propagation",
    return_messages=True
)

# Examine messages between nodes
for edge, message in detailed_result.messages.items():
    parent, child = edge
    print(f"Message from {parent} to {child}:")
    print(f"  Amplitudes: {message.amplitudes}")
    print(f"  Probabilities: {np.abs(message.amplitudes)**2}")
```

## Variational Inference

### Variational Quantum Eigensolver (VQE)

```python
# Use variational inference for complex distributions
variational_result = network.infer(
    query_nodes=["weather", "mood"],
    method="variational",
    max_iterations=100,
    optimization_method="L-BFGS-B"
)

print(f"Variational inference result: {variational_result.marginal_probabilities}")
print(f"Final cost: {variational_result.final_cost}")
print(f"Optimization success: {variational_result.optimization_success}")
```

### Circuit Parameters

```python
# Access optimized quantum circuit parameters
if hasattr(variational_result, 'optimal_parameters'):
    params = variational_result.optimal_parameters
    print(f"Optimized circuit parameters: {params}")
    
    # Visualize parameter evolution
    if hasattr(variational_result, 'parameter_history'):
        import matplotlib.pyplot as plt
        
        history = variational_result.parameter_history
        plt.figure(figsize=(10, 6))
        for i, param_trace in enumerate(history.T):
            plt.plot(param_trace, label=f'Parameter {i}')
        plt.xlabel('Optimization Step')
        plt.ylabel('Parameter Value')
        plt.title('Variational Parameter Evolution')
        plt.legend()
        plt.show()
```

### Custom Ansatz

```python
# Define custom variational ansatz
def custom_ansatz(parameters, n_qubits):
    """Custom variational circuit ansatz."""
    from probabilistic_quantum_reasoner.core.operators import QuantumGate
    
    circuit = []
    
    # Layer 1: Individual rotations
    for i in range(n_qubits):
        circuit.append(('RY', i, parameters[i]))
    
    # Layer 2: Entangling gates
    for i in range(n_qubits - 1):
        circuit.append(('CNOT', i, i + 1))
    
    # Layer 3: Final rotations
    for i in range(n_qubits):
        circuit.append(('RZ', i, parameters[n_qubits + i]))
    
    return circuit

# Use custom ansatz in inference
custom_result = network.infer(
    query_nodes=["weather", "mood"],
    method="variational",
    ansatz=custom_ansatz,
    n_parameters=4  # 2 qubits × 2 parameters each
)
```

## Sampling-Based Inference

### Quantum Sampling

```python
# Sampling-based approximate inference
sampling_result = network.infer(
    query_nodes=["mood", "activity"],
    evidence={"weather": "sunny"},
    method="sampling",
    n_samples=1000
)

print(f"Sampling result: {sampling_result.marginal_probabilities}")
print(f"Sample variance: {sampling_result.sample_variance}")
print(f"Effective sample size: {sampling_result.effective_sample_size}")
```

### Markov Chain Monte Carlo

```python
# MCMC sampling for complex posterior distributions
mcmc_result = network.infer(
    query_nodes=["weather", "mood", "activity"],
    method="mcmc",
    n_samples=5000,
    burn_in=1000,
    thin=5
)

print(f"MCMC marginals: {mcmc_result.marginal_probabilities}")
print(f"Acceptance rate: {mcmc_result.acceptance_rate}")
print(f"Chain convergence: {mcmc_result.r_hat}")  # Gelman-Rubin statistic
```

### Importance Sampling

```python
# Importance sampling with quantum proposal distribution
importance_result = network.infer(
    query_nodes=["activity"],
    evidence={"weather": "sunny"},
    method="importance_sampling",
    n_samples=2000,
    proposal="quantum_uniform"  # Use quantum superposition as proposal
)

print(f"Importance sampling result: {importance_result.marginal_probabilities}")
print(f"Effective sample size: {importance_result.effective_sample_size}")
```

## Exact Inference

### Quantum Amplitude Enumeration

```python
# Exact inference by enumerating all quantum amplitudes
exact_result = network.infer(
    query_nodes=["weather", "mood"],
    method="exact",
    enumerate_amplitudes=True
)

print(f"Exact marginals: {exact_result.marginal_probabilities}")
print(f"Computation time: {exact_result.computation_time}")
print(f"Memory usage: {exact_result.memory_usage_mb} MB")
```

### Junction Tree Algorithm

```python
# Use junction tree for efficient exact inference
if network.is_tree_decomposable():
    junction_result = network.infer(
        query_nodes=["activity"],
        method="junction_tree"
    )
    print(f"Junction tree result: {junction_result.marginal_probabilities}")
else:
    print("Network is not tree-decomposable, using alternative method")
```

## Approximate Inference

### Loopy Belief Propagation

```python
# Loopy belief propagation for networks with cycles
loopy_result = network.infer(
    query_nodes=["mood"],
    method="loopy_belief_propagation",
    max_iterations=100,
    damping_factor=0.5  # Damping to improve convergence
)

print(f"Loopy BP result: {loopy_result.marginal_probabilities}")
print(f"Converged: {loopy_result.converged}")
```

### Mean Field Approximation

```python
# Mean field variational approximation
mean_field_result = network.infer(
    query_nodes=["weather", "mood", "activity"],
    method="mean_field",
    max_iterations=50
)

print(f"Mean field marginals: {mean_field_result.marginal_probabilities}")
print(f"ELBO (Evidence Lower Bound): {mean_field_result.elbo}")
```

## Quantum-Specific Inference

### Grover-Enhanced Search

```python
# Use Grover's algorithm for probabilistic search
def search_condition(state_dict):
    """Search condition: weather is sunny AND mood is happy."""
    return state_dict.get("weather") == "sunny" and state_dict.get("mood") == "happy"

grover_result = network.infer(
    query_nodes=["weather", "mood"],
    method="grover_search",
    search_condition=search_condition,
    max_iterations=None  # Optimal number of iterations
)

print(f"Grover search result: {grover_result.marginal_probabilities}")
print(f"Search success probability: {grover_result.success_probability}")
```

### Quantum Fourier Transform

```python
# QFT-based inference for periodic patterns
qft_result = network.infer(
    query_nodes=["weather"],
    method="quantum_fourier_transform",
    basis="frequency"  # Frequency domain analysis
)

print(f"QFT frequency components: {qft_result.frequency_amplitudes}")
print(f"Dominant frequencies: {qft_result.dominant_frequencies}")
```

## Inference Configuration

### Performance Tuning

```python
# Configure inference algorithms for performance
config = {
    "belief_propagation": {
        "max_iterations": 100,
        "convergence_threshold": 1e-8,
        "parallel_messages": True
    },
    "variational": {
        "optimizer": "Adam",
        "learning_rate": 0.01,
        "batch_size": 32
    },
    "sampling": {
        "n_samples": 10000,
        "parallel_chains": 4,
        "adaptive_step_size": True
    }
}

# Apply configuration
network.set_inference_config(config)

# Run with optimized settings
optimized_result = network.infer(
    query_nodes=["activity"],
    evidence={"weather": "rainy"},
    method="belief_propagation"
)
```

### Memory Management

```python
# Monitor and control memory usage during inference
memory_config = {
    "max_memory_mb": 4096,  # 4 GB limit
    "state_compression": True,
    "lazy_evaluation": True,
    "garbage_collection": "aggressive"
}

network.set_memory_config(memory_config)

# Memory-efficient inference
efficient_result = network.infer(
    query_nodes=["weather", "mood", "activity"],
    method="variational",
    memory_efficient=True
)

print(f"Peak memory usage: {efficient_result.peak_memory_mb} MB")
```

## Inference Diagnostics

### Convergence Analysis

```python
# Detailed convergence diagnostics
diagnostic_result = network.infer(
    query_nodes=["mood"],
    method="belief_propagation",
    return_diagnostics=True
)

diagnostics = diagnostic_result.diagnostics
print(f"Convergence rate: {diagnostics['convergence_rate']}")
print(f"Message fidelities: {diagnostics['message_fidelities']}")
print(f"Quantum coherence: {diagnostics['quantum_coherence']}")

# Plot convergence
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(diagnostics['cost_history'])
plt.title('Cost Function')
plt.xlabel('Iteration')

plt.subplot(1, 3, 2)
plt.plot(diagnostics['fidelity_history'])
plt.title('State Fidelity')
plt.xlabel('Iteration')

plt.subplot(1, 3, 3)
plt.plot(diagnostics['coherence_history'])
plt.title('Quantum Coherence')
plt.xlabel('Iteration')

plt.tight_layout()
plt.show()
```

### Uncertainty Quantification

```python
# Quantify uncertainty in inference results
uncertainty_result = network.infer(
    query_nodes=["activity"],
    evidence={"weather": "sunny"},
    method="sampling",
    uncertainty_quantification=True,
    confidence_level=0.95
)

activity_dist = uncertainty_result.marginal_probabilities["activity"]
activity_ci = uncertainty_result.confidence_intervals["activity"]

print("Activity distribution with uncertainty:")
for outcome, prob in activity_dist.items():
    ci_lower, ci_upper = activity_ci[outcome]
    print(f"  P(activity={outcome}) = {prob:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]")
```

## Inference Comparison

### Algorithm Benchmarking

```python
# Compare different inference algorithms
algorithms = ["belief_propagation", "variational", "sampling", "exact"]
evidence = {"weather": "sunny"}
query = ["mood", "activity"]

results = {}
for algorithm in algorithms:
    try:
        start_time = time.time()
        result = network.infer(
            query_nodes=query,
            evidence=evidence,
            method=algorithm
        )
        end_time = time.time()
        
        results[algorithm] = {
            "marginals": result.marginal_probabilities,
            "time": end_time - start_time,
            "memory": result.memory_usage_mb if hasattr(result, 'memory_usage_mb') else 0
        }
    except Exception as e:
        print(f"{algorithm} failed: {e}")

# Compare results
print("Algorithm comparison:")
for algorithm, data in results.items():
    print(f"\n{algorithm.upper()}:")
    print(f"  Time: {data['time']:.3f} seconds")
    print(f"  Memory: {data['memory']:.1f} MB")
    print(f"  Activity marginal: {data['marginals']['activity']}")
```

## Best Practices

### Algorithm Selection

```python
def select_inference_algorithm(network, query_nodes, evidence=None):
    """Automatically select best inference algorithm."""
    
    # Small networks: use exact inference
    if len(network.nodes) <= 10:
        return "exact"
    
    # Tree-structured networks: use belief propagation
    elif network.is_tree():
        return "belief_propagation"
    
    # Many quantum nodes: use variational methods
    elif network.count_quantum_nodes() > 5:
        return "variational"
    
    # Large classical networks: use sampling
    elif len(network.nodes) > 50:
        return "sampling"
    
    # Default: loopy belief propagation
    else:
        return "loopy_belief_propagation"

# Auto-select algorithm
optimal_algorithm = select_inference_algorithm(network)
auto_result = network.infer(
    query_nodes=["activity"],
    evidence={"weather": "sunny"},
    method=optimal_algorithm
)
```

### Error Handling

```python
# Robust inference with error handling
def robust_inference(network, query_nodes, evidence=None, fallback_methods=None):
    """Perform inference with automatic fallback methods."""
    
    if fallback_methods is None:
        fallback_methods = ["belief_propagation", "variational", "sampling"]
    
    for method in fallback_methods:
        try:
            result = network.infer(
                query_nodes=query_nodes,
                evidence=evidence,
                method=method,
                timeout=60  # 1 minute timeout
            )
            
            # Validate result
            if result.is_valid():
                return result, method
                
        except Exception as e:
            print(f"Method {method} failed: {e}")
            continue
    
    raise RuntimeError("All inference methods failed")

# Use robust inference
try:
    final_result, used_method = robust_inference(
        network, 
        query_nodes=["mood"],
        evidence={"weather": "rainy"}
    )
    print(f"Successfully used method: {used_method}")
    print(f"Result: {final_result.marginal_probabilities}")
    
except RuntimeError as e:
    print(f"Inference failed: {e}")
```

## Next Steps

- Learn about [Causal Reasoning](causal.md) and interventions
- Explore [Variational Methods](variational.md) in detail
- See practical [Examples](../examples/weather-mood.md)
- Check [API Reference](../api/inference.md) for complete documentation
