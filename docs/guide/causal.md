# Causal Reasoning

This guide covers causal inference, interventions, and counterfactual reasoning in quantum Bayesian networks using do-calculus and quantum causal models.

## Introduction to Quantum Causality

Quantum causal reasoning extends classical causal inference to quantum systems, allowing for interventions on quantum variables and counterfactual analysis with quantum superposition.

## Basic Causal Operations

### Do-Calculus Interventions

```python
from probabilistic_quantum_reasoner import QuantumBayesianNetwork
from probabilistic_quantum_reasoner.backends import ClassicalSimulator
import numpy as np

# Create causal network
backend = ClassicalSimulator()
network = QuantumBayesianNetwork("CausalDemo", backend)

# Causal chain: Weather → Mood → Activity
weather = network.add_quantum_node(
    "weather",
    outcome_space=["sunny", "rainy"],
    initial_amplitudes=np.array([0.8, 0.6], dtype=complex)
)

mood = network.add_stochastic_node("mood", outcome_space=["happy", "sad"])
activity = network.add_stochastic_node("activity", outcome_space=["indoor", "outdoor"])

# Add causal edges
network.add_edge(weather, mood)
network.add_edge(mood, activity)

# Observational distribution
observational = network.infer(query_nodes=["activity"])
print(f"Observational P(activity): {observational.marginal_probabilities['activity']}")

# Interventional distribution: do(weather = sunny)
interventional = network.intervene(
    interventions={"weather": "sunny"},
    query_nodes=["activity"]
)
print(f"Interventional P(activity | do(weather=sunny)): {interventional.marginal_probabilities['activity']}")
```

### Quantum Interventions

```python
# Intervention on quantum node with specific amplitude pattern
quantum_intervention = network.intervene(
    interventions={
        "weather": {
            "type": "quantum",
            "amplitudes": np.array([1, 0], dtype=complex)  # Force to |sunny⟩
        }
    },
    query_nodes=["mood", "activity"]
)

print(f"Quantum intervention result: {quantum_intervention.marginal_probabilities}")

# Intervention with superposition state
superposition_intervention = network.intervene(
    interventions={
        "weather": {
            "type": "quantum", 
            "amplitudes": np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
        }
    },
    query_nodes=["activity"]
)

print(f"Superposition intervention: {superposition_intervention.marginal_probabilities}")
```

## Counterfactual Reasoning

### Classical Counterfactuals

```python
# Factual scenario: Weather was rainy, mood was sad
factual_evidence = {
    "weather": "rainy",
    "mood": "sad"
}

# Counterfactual question: What if weather had been sunny instead?
counterfactual_result = network.counterfactual_inference(
    factual_evidence=factual_evidence,
    counterfactual_intervention={"weather": "sunny"},
    query_nodes=["mood", "activity"]
)

print("Counterfactual Analysis:")
print(f"Factual mood: {counterfactual_result['factual_outcome']['mood']}")
print(f"Counterfactual mood: {counterfactual_result['counterfactual_outcome']['mood']}")
print(f"Causal effect on mood: {counterfactual_result['causal_effect']['mood']}")
```

### Quantum Counterfactuals

```python
# Quantum counterfactual: What if we had prepared a different quantum state?
quantum_counterfactual = network.counterfactual_inference(
    factual_evidence={"weather": "rainy"},
    counterfactual_intervention={
        "weather": {
            "type": "quantum",
            "amplitudes": np.array([0.6, 0.8], dtype=complex)
        }
    },
    query_nodes=["mood"]
)

print("Quantum Counterfactual:")
print(f"Quantum causal effect: {quantum_counterfactual['quantum_causal_effect']}")
```

## Causal Discovery

### Quantum Causal Structure Learning

```python
# Learn causal structure from quantum data
def learn_quantum_causal_structure(data, nodes):
    """Learn causal DAG from quantum measurement data."""
    
    from probabilistic_quantum_reasoner.causal import QuantumCausalDiscovery
    
    discoverer = QuantumCausalDiscovery()
    
    # Score different causal structures
    candidate_structures = discoverer.enumerate_candidate_dags(nodes)
    
    best_structure = None
    best_score = -np.inf
    
    for structure in candidate_structures:
        score = discoverer.score_structure(structure, data)
        if score > best_score:
            best_score = score
            best_structure = structure
    
    return best_structure, best_score

# Example usage with simulated data
simulated_data = network.simulate_data(n_samples=1000)
learned_structure, score = learn_quantum_causal_structure(
    simulated_data, 
    ["weather", "mood", "activity"]
)

print(f"Learned causal structure: {learned_structure}")
print(f"Structure score: {score}")
```

### Constraint-Based Discovery

```python
# Quantum PC algorithm for causal discovery
def quantum_pc_algorithm(data, alpha=0.05):
    """Quantum version of PC algorithm for causal discovery."""
    
    nodes = list(data.columns)
    n_nodes = len(nodes)
    
    # Initialize complete graph
    adjacency_matrix = np.ones((n_nodes, n_nodes)) - np.eye(n_nodes)
    
    # Phase 1: Remove edges using conditional independence tests
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            if adjacency_matrix[i, j] == 1:
                # Test conditional independence with quantum correlations
                p_value = quantum_conditional_independence_test(
                    data, nodes[i], nodes[j], conditioning_set=[]
                )
                
                if p_value > alpha:
                    adjacency_matrix[i, j] = 0
                    adjacency_matrix[j, i] = 0
    
    # Phase 2: Orient edges using quantum causal principles
    oriented_graph = orient_quantum_edges(adjacency_matrix, nodes, data)
    
    return oriented_graph

def quantum_conditional_independence_test(data, var1, var2, conditioning_set):
    """Test conditional independence using quantum mutual information."""
    
    # Compute quantum mutual information
    joint_entropy = compute_quantum_entropy(data[[var1, var2] + conditioning_set])
    marginal_entropy1 = compute_quantum_entropy(data[[var1] + conditioning_set])
    marginal_entropy2 = compute_quantum_entropy(data[[var2] + conditioning_set])
    conditioning_entropy = compute_quantum_entropy(data[conditioning_set]) if conditioning_set else 0
    
    quantum_mi = marginal_entropy1 + marginal_entropy2 - joint_entropy - conditioning_entropy
    
    # Convert to p-value (simplified)
    p_value = np.exp(-abs(quantum_mi))
    
    return p_value

# Apply quantum PC algorithm
discovered_graph = quantum_pc_algorithm(simulated_data)
print(f"Discovered causal graph: {discovered_graph}")
```

## Causal Effect Identification

### Backdoor Criterion

```python
# Check backdoor criterion for causal identification
def check_backdoor_criterion(network, treatment, outcome, adjustment_set):
    """Check if adjustment set satisfies backdoor criterion."""
    
    # Find all paths from treatment to outcome
    all_paths = network.find_all_paths(treatment, outcome)
    
    # Check each path for backdoor property
    for path in all_paths:
        if is_backdoor_path(path, treatment, outcome):
            # Check if adjustment set blocks this path
            if not is_path_blocked(path, adjustment_set, network):
                return False, f"Backdoor path not blocked: {path}"
    
    # Check that adjustment set doesn't contain descendants of treatment
    treatment_descendants = network.get_descendants(treatment)
    if any(var in treatment_descendants for var in adjustment_set):
        return False, "Adjustment set contains treatment descendants"
    
    return True, "Backdoor criterion satisfied"

# Example: Check if {mood} satisfies backdoor criterion for weather→activity
is_valid, message = check_backdoor_criterion(
    network, 
    treatment="weather", 
    outcome="activity", 
    adjustment_set=["mood"]
)

print(f"Backdoor criterion check: {is_valid}")
print(f"Message: {message}")
```

### Front-door Criterion

```python
# Apply front-door criterion when backdoor is not available
def frontdoor_adjustment(network, treatment, outcome, mediator):
    """Compute causal effect using front-door adjustment."""
    
    # Step 1: Compute P(mediator | do(treatment))
    mediator_given_treatment = {}
    for treatment_val in network.nodes[treatment].outcome_space:
        result = network.intervene(
            interventions={treatment: treatment_val},
            query_nodes=[mediator]
        )
        mediator_given_treatment[treatment_val] = result.marginal_probabilities[mediator]
    
    # Step 2: Compute P(outcome | do(mediator), treatment)
    outcome_given_mediator_treatment = {}
    for treatment_val in network.nodes[treatment].outcome_space:
        for mediator_val in network.nodes[mediator].outcome_space:
            result = network.intervene(
                interventions={mediator: mediator_val},
                evidence={treatment: treatment_val},
                query_nodes=[outcome]
            )
            outcome_given_mediator_treatment[(treatment_val, mediator_val)] = \
                result.marginal_probabilities[outcome]
    
    # Step 3: Marginalize to get P(outcome | do(treatment))
    causal_effect = {}
    for treatment_val in network.nodes[treatment].outcome_space:
        effect_dist = {}
        for outcome_val in network.nodes[outcome].outcome_space:
            total_prob = 0
            for mediator_val in network.nodes[mediator].outcome_space:
                prob_mediator = mediator_given_treatment[treatment_val][mediator_val]
                prob_outcome = outcome_given_mediator_treatment[(treatment_val, mediator_val)][outcome_val]
                total_prob += prob_mediator * prob_outcome
            effect_dist[outcome_val] = total_prob
        causal_effect[treatment_val] = effect_dist
    
    return causal_effect

# Example front-door adjustment
if network.satisfies_frontdoor_criterion("weather", "activity", "mood"):
    frontdoor_effect = frontdoor_adjustment(network, "weather", "activity", "mood")
    print(f"Front-door causal effect: {frontdoor_effect}")
```

## Quantum Causal Models

### Quantum Structural Causal Models

```python
class QuantumStructuralCausalModel:
    """Quantum extension of structural causal models."""
    
    def __init__(self, network):
        self.network = network
        self.structural_equations = {}
        self.noise_terms = {}
    
    def add_structural_equation(self, variable, equation, noise_distribution):
        """Add structural equation for a variable."""
        self.structural_equations[variable] = equation
        self.noise_terms[variable] = noise_distribution
    
    def add_quantum_structural_equation(self, variable, quantum_equation, noise_operator):
        """Add quantum structural equation."""
        self.structural_equations[variable] = {
            "type": "quantum",
            "equation": quantum_equation,
            "noise": noise_operator
        }
    
    def simulate_intervention(self, interventions, n_samples=1000):
        """Simulate data under interventions."""
        
        # Create modified structural equations
        modified_equations = self.structural_equations.copy()
        
        for var, value in interventions.items():
            if isinstance(value, dict) and value.get("type") == "quantum":
                # Quantum intervention
                modified_equations[var] = {
                    "type": "quantum_intervention",
                    "amplitudes": value["amplitudes"]
                }
            else:
                # Classical intervention
                modified_equations[var] = {
                    "type": "classical_intervention",
                    "value": value
                }
        
        # Generate samples using modified equations
        samples = []
        for _ in range(n_samples):
            sample = self._generate_sample(modified_equations)
            samples.append(sample)
        
        return samples
    
    def _generate_sample(self, equations):
        """Generate single sample from structural equations."""
        sample = {}
        
        # Topological ordering
        ordered_vars = self.network.topological_sort()
        
        for var in ordered_vars:
            equation = equations[var]
            
            if equation.get("type") == "quantum":
                # Quantum structural equation
                parents = self.network.get_parents(var)
                parent_values = {p: sample[p] for p in parents if p in sample}
                
                # Apply quantum equation
                quantum_state = equation["equation"](parent_values)
                
                # Add quantum noise
                noisy_state = equation["noise"].apply(quantum_state)
                
                # Measure to get classical value
                measurement_result = self.network.measure_quantum_state(noisy_state)
                sample[var] = measurement_result
                
            elif equation.get("type") == "quantum_intervention":
                # Quantum intervention
                amplitudes = equation["amplitudes"]
                measurement_result = self.network.measure_amplitudes(amplitudes)
                sample[var] = measurement_result
                
            elif equation.get("type") == "classical_intervention":
                # Classical intervention
                sample[var] = equation["value"]
                
            else:
                # Classical structural equation
                parents = self.network.get_parents(var)
                parent_values = {p: sample[p] for p in parents if p in sample}
                
                # Apply classical equation with noise
                noise = self.noise_terms[var].sample()
                sample[var] = equation(parent_values, noise)
        
        return sample

# Create quantum SCM
qscm = QuantumStructuralCausalModel(network)

# Add structural equations
qscm.add_quantum_structural_equation(
    "weather",
    lambda parents: np.array([0.8, 0.6], dtype=complex),  # Intrinsic weather distribution
    QuantumGate.phase(np.pi/8)  # Quantum noise
)

qscm.add_structural_equation(
    "mood", 
    lambda parents, noise: "happy" if (parents.get("weather") == "sunny" and noise > 0.2) else "sad",
    lambda: np.random.uniform(0, 1)  # Uniform noise
)

# Simulate intervention
intervention_data = qscm.simulate_intervention(
    interventions={"weather": {"type": "quantum", "amplitudes": np.array([1, 0], dtype=complex)}},
    n_samples=1000
)

print(f"Intervention simulation complete: {len(intervention_data)} samples")
```

## Mediation Analysis

### Quantum Mediation

```python
def quantum_mediation_analysis(network, treatment, mediator, outcome):
    """Perform mediation analysis with quantum mediator."""
    
    # Natural Direct Effect (NDE): effect not through mediator
    nde = {}
    for treatment_val in network.nodes[treatment].outcome_space:
        # Fix mediator at its natural value when treatment = reference
        reference_val = network.nodes[treatment].outcome_space[0]
        
        mediator_natural = network.infer(
            evidence={treatment: reference_val},
            query_nodes=[mediator]
        )
        
        # Sample mediator value from natural distribution
        mediator_sample = max(
            mediator_natural.marginal_probabilities[mediator].items(),
            key=lambda x: x[1]
        )[0]
        
        # Compute outcome under intervention
        nde_result = network.intervene(
            interventions={treatment: treatment_val, mediator: mediator_sample},
            query_nodes=[outcome]
        )
        
        nde[treatment_val] = nde_result.marginal_probabilities[outcome]
    
    # Natural Indirect Effect (NIE): effect through mediator
    nie = {}
    for treatment_val in network.nodes[treatment].outcome_space:
        reference_val = network.nodes[treatment].outcome_space[0]
        
        # Mediator under treatment
        mediator_treated = network.infer(
            evidence={treatment: treatment_val},
            query_nodes=[mediator]
        )
        
        # If mediator is quantum, preserve superposition
        if hasattr(network.nodes[mediator], 'quantum_state'):
            # Quantum mediation preserves superposition
            mediator_amplitudes = network.get_quantum_mediator_amplitudes(
                treatment_val, mediator
            )
            
            nie_result = network.quantum_mediation_effect(
                treatment=reference_val,
                mediator_amplitudes=mediator_amplitudes,
                outcome=outcome
            )
        else:
            # Classical mediation
            mediator_sample = max(
                mediator_treated.marginal_probabilities[mediator].items(),
                key=lambda x: x[1]
            )[0]
            
            nie_result = network.intervene(
                interventions={treatment: reference_val, mediator: mediator_sample},
                query_nodes=[outcome]
            )
        
        nie[treatment_val] = nie_result.marginal_probabilities[outcome]
    
    # Total Effect = NDE + NIE
    total_effect = {}
    for treatment_val in network.nodes[treatment].outcome_space:
        te_result = network.intervene(
            interventions={treatment: treatment_val},
            query_nodes=[outcome]
        )
        total_effect[treatment_val] = te_result.marginal_probabilities[outcome]
    
    return {
        "natural_direct_effect": nde,
        "natural_indirect_effect": nie,
        "total_effect": total_effect,
        "proportion_mediated": compute_proportion_mediated(nde, nie, total_effect)
    }

# Perform quantum mediation analysis
mediation_results = quantum_mediation_analysis(
    network, 
    treatment="weather", 
    mediator="mood", 
    outcome="activity"
)

print("Mediation Analysis Results:")
for effect_type, effects in mediation_results.items():
    print(f"{effect_type}: {effects}")
```

## Causal Inference from Quantum Data

### Quantum Bootstrap

```python
def quantum_bootstrap_causal_effect(network, treatment, outcome, n_bootstrap=1000):
    """Bootstrap confidence intervals for quantum causal effects."""
    
    bootstrap_effects = []
    
    for _ in range(n_bootstrap):
        # Sample quantum states with noise
        noisy_network = network.add_quantum_noise(noise_level=0.1)
        
        # Compute causal effect
        causal_result = noisy_network.intervene(
            interventions={treatment: "sunny"},  # Example intervention
            query_nodes=[outcome]
        )
        
        effect_size = causal_result.marginal_probabilities[outcome]["outdoor"]
        bootstrap_effects.append(effect_size)
    
    # Compute confidence interval
    bootstrap_effects = np.array(bootstrap_effects)
    ci_lower = np.percentile(bootstrap_effects, 2.5)
    ci_upper = np.percentile(bootstrap_effects, 97.5)
    
    return {
        "point_estimate": np.mean(bootstrap_effects),
        "confidence_interval": (ci_lower, ci_upper),
        "bootstrap_distribution": bootstrap_effects
    }

# Bootstrap causal effect estimation
bootstrap_result = quantum_bootstrap_causal_effect(
    network, "weather", "activity"
)

print(f"Causal effect estimate: {bootstrap_result['point_estimate']:.3f}")
print(f"95% CI: {bootstrap_result['confidence_interval']}")
```

## Advanced Causal Topics

### Causal Sufficiency

```python
def test_causal_sufficiency(network, observed_variables):
    """Test if observed variables are causally sufficient."""
    
    # Test vanishing tetrad differences for linear quantum systems
    tetrads = []
    
    for i in range(len(observed_variables)):
        for j in range(i+1, len(observed_variables)):
            for k in range(j+1, len(observed_variables)):
                for l in range(k+1, len(observed_variables)):
                    vars_quartet = [observed_variables[i], observed_variables[j], 
                                  observed_variables[k], observed_variables[l]]
                    
                    # Compute quantum tetrad difference
                    tetrad_diff = compute_quantum_tetrad_difference(network, vars_quartet)
                    tetrads.append(tetrad_diff)
    
    # Test if tetrads are close to zero
    max_tetrad = max(abs(t) for t in tetrads)
    is_sufficient = max_tetrad < 0.1  # Threshold for sufficiency
    
    return {
        "causally_sufficient": is_sufficient,
        "max_tetrad_difference": max_tetrad,
        "all_tetrads": tetrads
    }

def compute_quantum_tetrad_difference(network, variables):
    """Compute tetrad difference for quantum variables."""
    # Simplified quantum tetrad computation
    a, b, c, d = variables
    
    # Quantum covariances
    cov_ab_cd = compute_quantum_covariance(network, [a, b], [c, d])
    cov_ac_bd = compute_quantum_covariance(network, [a, c], [b, d])
    cov_ad_bc = compute_quantum_covariance(network, [a, d], [b, c])
    
    return cov_ab_cd - cov_ac_bd - cov_ad_bc

# Test causal sufficiency
sufficiency_test = test_causal_sufficiency(
    network, ["weather", "mood", "activity"]
)
print(f"Causal sufficiency test: {sufficiency_test}")
```

## Best Practices

### Causal Modeling Guidelines

1. **Clear Causal Assumptions**: Explicitly state causal assumptions and domain knowledge
2. **Quantum-Classical Separation**: Distinguish between quantum and classical causal mechanisms  
3. **Temporal Ordering**: Ensure causes precede effects in time
4. **Confounding Control**: Identify and control for confounding variables
5. **Sensitivity Analysis**: Test robustness to modeling assumptions

### Common Pitfalls

1. **Confusing Association and Causation**: Correlation ≠ Causation in quantum systems too
2. **Ignoring Quantum Measurement Effects**: Measurement can affect causal relationships
3. **Post-Treatment Bias**: Don't condition on variables affected by treatment
4. **Quantum Decoherence**: Account for decoherence in causal models
5. **Simpson's Paradox**: Aggregation can reverse causal effects

## Next Steps

- Learn about [Variational Methods](variational.md) for causal inference optimization
- Explore practical [Examples](../examples/prisoners-dilemma.md) of causal reasoning
- See [Advanced Topics](../advanced/entanglement.md) for quantum causal discovery
- Check [API Reference](../api/inference.md) for causal inference functions
