# Variational Methods

This guide covers variational quantum algorithms for optimization and inference in quantum Bayesian networks, including VQE, QAOA, and hybrid classical-quantum optimization.

## Introduction to Variational Quantum Algorithms

Variational methods combine parametrized quantum circuits with classical optimization to solve complex inference and optimization problems that are intractable for classical methods alone.

## Variational Quantum Eigensolver (VQE)

### Basic VQE Implementation

```python
from probabilistic_quantum_reasoner import QuantumBayesianNetwork
from probabilistic_quantum_reasoner.backends import PennyLaneBackend
from probabilistic_quantum_reasoner.inference.variational import VQE
import numpy as np

# Create network with PennyLane backend for VQE
backend = PennyLaneBackend("default.qubit", shots=None)  # Analytic gradients
network = QuantumBayesianNetwork("VQEExample", backend)

# Add quantum nodes
energy_state = network.add_quantum_node(
    "energy",
    outcome_space=["ground", "excited"],
    initial_amplitudes=np.array([1, 0], dtype=complex)
)

spin_state = network.add_quantum_node(
    "spin", 
    outcome_space=["up", "down"],
    initial_amplitudes=np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
)

# Entangle the qubits
network.entangle([energy_state, spin_state])

# Define Hamiltonian for the system
def create_hamiltonian():
    """Create Hamiltonian operator for two-qubit system."""
    # H = -J * Z⊗Z - h * (X⊗I + I⊗X)
    J = 1.0  # Coupling strength
    h = 0.5  # External field
    
    # Pauli matrices
    I = np.eye(2)
    X = np.array([[0, 1], [1, 0]])
    Z = np.array([[1, 0], [0, -1]])
    
    # Two-qubit Hamiltonian
    ZZ = np.kron(Z, Z)
    XI = np.kron(X, I) 
    IX = np.kron(I, X)
    
    H = -J * ZZ - h * (XI + IX)
    return H

hamiltonian = create_hamiltonian()

# Create VQE instance
vqe = VQE(network, hamiltonian)

# Define variational ansatz
def ansatz(parameters, n_qubits=2):
    """Create variational ansatz circuit."""
    theta1, theta2, theta3, theta4 = parameters
    
    circuit = [
        # Layer 1: Single qubit rotations
        ('RY', 0, theta1),
        ('RY', 1, theta2),
        
        # Entangling layer
        ('CNOT', 0, 1),
        
        # Layer 2: More rotations
        ('RZ', 0, theta3),
        ('RZ', 1, theta4)
    ]
    
    return circuit

# Run VQE optimization
vqe_result = vqe.optimize(
    ansatz=ansatz,
    n_parameters=4,
    optimizer="Adam",
    max_iterations=200,
    learning_rate=0.1
)

print(f"VQE Ground State Energy: {vqe_result['ground_state_energy']:.6f}")
print(f"Optimal Parameters: {vqe_result['optimal_parameters']}")
print(f"Convergence: {vqe_result['converged']}")
```

### Advanced VQE Features

```python
# VQE with custom cost function
def custom_vqe_cost(parameters, network, target_distribution):
    """Custom VQE cost function to match target distribution."""
    
    # Apply variational circuit
    state = network.apply_variational_circuit(ansatz(parameters))
    
    # Compute current probability distribution
    current_probs = np.abs(state.amplitudes) ** 2
    
    # KL divergence from target
    epsilon = 1e-10
    kl_div = np.sum(target_distribution * np.log(
        (target_distribution + epsilon) / (current_probs + epsilon)
    ))
    
    # Add regularization
    regularization = 0.01 * np.sum(parameters ** 2)
    
    return kl_div + regularization

# Target distribution (e.g., thermal state)
target_dist = np.array([0.6, 0.2, 0.15, 0.05])  # |00⟩, |01⟩, |10⟩, |11⟩

custom_vqe_result = vqe.optimize(
    cost_function=lambda params: custom_vqe_cost(params, network, target_dist),
    n_parameters=4,
    optimizer="L-BFGS-B",
    max_iterations=100
)

print(f"Custom VQE result: {custom_vqe_result}")
```

## Quantum Approximate Optimization Algorithm (QAOA)

### QAOA for Inference Problems

```python
from probabilistic_quantum_reasoner.inference.variational import QAOA

# Create QAOA instance for inference optimization
qaoa = QAOA(network, p_layers=3)

# Define problem Hamiltonian based on inference objective
def create_inference_hamiltonian(network, evidence, query_nodes):
    """Create problem Hamiltonian for inference task."""
    
    # Convert inference problem to Ising model
    n_vars = len(query_nodes)
    n_qubits = n_vars
    
    # Coupling terms based on network structure
    J = np.zeros((n_qubits, n_qubits))
    h = np.zeros(n_qubits)
    
    for i, node1 in enumerate(query_nodes):
        for j, node2 in enumerate(query_nodes):
            if j > i and network.has_edge(node1, node2):
                # Coupling strength based on conditional probabilities
                J[i, j] = compute_coupling_strength(network, node1, node2, evidence)
        
        # External field based on evidence
        h[i] = compute_external_field(network, node1, evidence)
    
    # Construct Hamiltonian matrix
    H = construct_ising_hamiltonian(J, h)
    return H

def compute_coupling_strength(network, node1, node2, evidence):
    """Compute coupling strength between nodes."""
    # Simplified: use mutual information
    joint_dist = network.infer(
        query_nodes=[node1, node2],
        evidence=evidence
    )
    
    return mutual_information(joint_dist.joint_probabilities)

def compute_external_field(network, node, evidence):
    """Compute external field for node given evidence."""
    # Bias based on evidence
    marginal = network.infer(
        query_nodes=[node],
        evidence=evidence
    )
    
    # Convert to bias term
    probs = list(marginal.marginal_probabilities[node].values())
    return np.log(probs[0] / probs[1]) if len(probs) >= 2 else 0.0

# Run QAOA for inference
inference_hamiltonian = create_inference_hamiltonian(
    network, 
    evidence={"energy": "ground"}, 
    query_nodes=["spin"]
)

qaoa_result = qaoa.optimize(
    problem_hamiltonian=inference_hamiltonian,
    max_iterations=50,
    optimizer="COBYLA"
)

print(f"QAOA Inference Result: {qaoa_result['optimal_state']}")
print(f"QAOA Parameters: {qaoa_result['optimal_parameters']}")
```

### Multi-Layer QAOA

```python
# Deep QAOA with many layers
deep_qaoa = QAOA(network, p_layers=10)

# Adaptive parameter initialization
def adaptive_parameter_initialization(p_layers):
    """Initialize QAOA parameters adaptively."""
    
    # Linear interpolation strategy
    gamma_schedule = np.linspace(0, np.pi, p_layers)
    beta_schedule = np.linspace(0, np.pi/2, p_layers)
    
    return np.concatenate([gamma_schedule, beta_schedule])

initial_params = adaptive_parameter_initialization(10)

deep_qaoa_result = deep_qaoa.optimize(
    problem_hamiltonian=inference_hamiltonian,
    initial_parameters=initial_params,
    max_iterations=200,
    optimizer="Adam",
    learning_rate=0.01,
    adaptive_step_size=True
)

print(f"Deep QAOA convergence: {deep_qaoa_result['converged']}")
print(f"Final cost: {deep_qaoa_result['final_cost']}")
```

## Hybrid Classical-Quantum Optimization

### Quantum-Classical Expectation-Maximization

```python
class QuantumEM:
    """Quantum-enhanced Expectation-Maximization algorithm."""
    
    def __init__(self, network, backend):
        self.network = network
        self.backend = backend
        
    def fit(self, data, max_iterations=100, tolerance=1e-6):
        """Fit quantum Bayesian network using quantum EM."""
        
        log_likelihood_history = []
        
        for iteration in range(max_iterations):
            # E-step: Quantum expectation computation
            expected_sufficient_stats = self.quantum_e_step(data)
            
            # M-step: Classical parameter maximization
            old_parameters = self.get_parameters()
            self.m_step(expected_sufficient_stats)
            new_parameters = self.get_parameters()
            
            # Compute log-likelihood
            log_likelihood = self.compute_log_likelihood(data)
            log_likelihood_history.append(log_likelihood)
            
            # Check convergence
            param_change = np.linalg.norm(new_parameters - old_parameters)
            if param_change < tolerance:
                break
        
        return {
            "converged": param_change < tolerance,
            "iterations": iteration + 1,
            "log_likelihood_history": log_likelihood_history,
            "final_log_likelihood": log_likelihood_history[-1]
        }
    
    def quantum_e_step(self, data):
        """Quantum expectation step using superposition."""
        
        sufficient_stats = {}
        
        for data_point in data:
            # Create quantum superposition over hidden variables
            hidden_vars = self.network.get_hidden_variables()
            
            # Quantum inference for expectations
            for hidden_var in hidden_vars:
                posterior = self.network.infer(
                    query_nodes=[hidden_var],
                    evidence=data_point,
                    method="variational"
                )
                
                # Accumulate sufficient statistics
                if hidden_var not in sufficient_stats:
                    sufficient_stats[hidden_var] = {}
                
                for value, prob in posterior.marginal_probabilities[hidden_var].items():
                    if value not in sufficient_stats[hidden_var]:
                        sufficient_stats[hidden_var][value] = 0
                    sufficient_stats[hidden_var][value] += prob
        
        return sufficient_stats
    
    def m_step(self, sufficient_stats):
        """Classical maximization step."""
        
        # Update parameters based on sufficient statistics
        for node_id, node in self.network.nodes.items():
            if hasattr(node, 'conditional_probability_table'):
                # Update CPT parameters
                self.update_cpt_parameters(node, sufficient_stats)
            elif hasattr(node, 'quantum_state'):
                # Update quantum amplitudes
                self.update_quantum_parameters(node, sufficient_stats)

# Example usage
quantum_em = QuantumEM(network, backend)

# Generate synthetic data
synthetic_data = network.simulate_data(n_samples=1000)

# Fit model
em_result = quantum_em.fit(synthetic_data, max_iterations=50)
print(f"Quantum EM converged: {em_result['converged']}")
print(f"Final log-likelihood: {em_result['final_log_likelihood']:.3f}")
```

### Variational Autoencoders with Quantum Layers

```python
class QuantumVariationalAutoencoder:
    """Quantum variational autoencoder for probabilistic modeling."""
    
    def __init__(self, network, latent_dim=4):
        self.network = network
        self.latent_dim = latent_dim
        self.encoder_parameters = None
        self.decoder_parameters = None
    
    def encode(self, data, parameters):
        """Quantum encoder: data → latent quantum state."""
        
        # Classical preprocessing
        processed_data = self.preprocess_data(data)
        
        # Quantum encoding circuit
        def encoding_circuit(data_vec, params):
            n_qubits = self.latent_dim
            circuit = []
            
            # Data encoding
            for i, data_val in enumerate(data_vec[:n_qubits]):
                circuit.append(('RY', i, data_val * np.pi))
            
            # Parameterized encoding
            param_idx = 0
            for layer in range(2):  # 2 encoding layers
                for i in range(n_qubits):
                    circuit.append(('RZ', i, params[param_idx]))
                    param_idx += 1
                
                # Entangling layer
                for i in range(n_qubits - 1):
                    circuit.append(('CNOT', i, i + 1))
            
            return circuit
        
        # Apply encoding circuit
        encoded_state = self.network.apply_quantum_circuit(
            encoding_circuit(processed_data, parameters)
        )
        
        return encoded_state
    
    def decode(self, latent_state, parameters):
        """Quantum decoder: latent quantum state → data reconstruction."""
        
        # Quantum decoding circuit
        def decoding_circuit(params):
            n_qubits = self.latent_dim
            circuit = []
            
            param_idx = 0
            for layer in range(2):  # 2 decoding layers
                for i in range(n_qubits):
                    circuit.append(('RY', i, params[param_idx]))
                    param_idx += 1
                
                for i in range(n_qubits - 1):
                    circuit.append(('CNOT', i, i + 1))
            
            # Measurement layer
            for i in range(n_qubits):
                circuit.append(('RZ', i, params[param_idx]))
                param_idx += 1
            
            return circuit
        
        # Apply decoding circuit
        decoded_state = self.network.apply_quantum_circuit_to_state(
            decoding_circuit(parameters), latent_state
        )
        
        # Convert to classical reconstruction
        reconstruction = np.abs(decoded_state.amplitudes) ** 2
        return reconstruction
    
    def elbo_loss(self, data, encoder_params, decoder_params):
        """Evidence Lower BOund loss function."""
        
        # Encode data to latent distribution
        latent_state = self.encode(data, encoder_params)
        
        # Decode latent state
        reconstruction = self.decode(latent_state, decoder_params)
        
        # Reconstruction loss
        reconstruction_loss = np.mean((data - reconstruction) ** 2)
        
        # KL divergence regularization (latent prior)
        uniform_prior = np.ones(2**self.latent_dim) / (2**self.latent_dim)
        latent_probs = np.abs(latent_state.amplitudes) ** 2
        
        epsilon = 1e-10
        kl_divergence = np.sum(latent_probs * np.log(
            (latent_probs + epsilon) / (uniform_prior + epsilon)
        ))
        
        # Total ELBO
        beta = 0.1  # KL weight
        elbo = reconstruction_loss + beta * kl_divergence
        
        return elbo, reconstruction_loss, kl_divergence
    
    def train(self, training_data, n_epochs=100, learning_rate=0.01):
        """Train quantum VAE."""
        
        # Initialize parameters
        n_encoder_params = 2 * self.latent_dim * 2  # 2 layers, 2 rotations per qubit
        n_decoder_params = 3 * self.latent_dim * 2  # 2 layers + measurement
        
        encoder_params = np.random.uniform(0, 2*np.pi, n_encoder_params)
        decoder_params = np.random.uniform(0, 2*np.pi, n_decoder_params)
        
        loss_history = []
        
        for epoch in range(n_epochs):
            epoch_loss = 0
            
            for data_point in training_data:
                # Compute loss and gradients
                loss, recon_loss, kl_loss = self.elbo_loss(
                    data_point, encoder_params, decoder_params
                )
                
                # Compute gradients (finite differences for simplicity)
                encoder_grad = self.compute_gradient(
                    lambda p: self.elbo_loss(data_point, p, decoder_params)[0],
                    encoder_params
                )
                
                decoder_grad = self.compute_gradient(
                    lambda p: self.elbo_loss(data_point, encoder_params, p)[0],
                    decoder_params
                )
                
                # Update parameters
                encoder_params -= learning_rate * encoder_grad
                decoder_params -= learning_rate * decoder_grad
                
                epoch_loss += loss
            
            avg_loss = epoch_loss / len(training_data)
            loss_history.append(avg_loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Average Loss: {avg_loss:.6f}")
        
        self.encoder_parameters = encoder_params
        self.decoder_parameters = decoder_params
        
        return {
            "loss_history": loss_history,
            "final_loss": loss_history[-1],
            "encoder_parameters": encoder_params,
            "decoder_parameters": decoder_params
        }

# Example: Train quantum VAE
qvae = QuantumVariationalAutoencoder(network, latent_dim=4)

# Generate training data
training_data = [network.sample() for _ in range(100)]

# Train model
vae_result = qvae.train(training_data, n_epochs=50, learning_rate=0.02)
print(f"QVAE training complete. Final loss: {vae_result['final_loss']:.6f}")
```

## Optimization Techniques

### Gradient-Free Optimization

```python
# Evolutionary optimization for quantum circuits
from scipy.optimize import differential_evolution

def evolutionary_vqe(network, hamiltonian, n_parameters):
    """VQE using evolutionary optimization."""
    
    def cost_function(parameters):
        """Cost function for evolutionary optimizer."""
        circuit = ansatz(parameters)
        state = network.apply_variational_circuit(circuit)
        energy = np.real(np.conj(state.amplitudes) @ hamiltonian @ state.amplitudes)
        return energy
    
    # Define bounds for parameters
    bounds = [(0, 2*np.pi) for _ in range(n_parameters)]
    
    # Run differential evolution
    result = differential_evolution(
        cost_function,
        bounds,
        maxiter=200,
        popsize=15,
        seed=42
    )
    
    return {
        "optimal_parameters": result.x,
        "optimal_energy": result.fun,
        "success": result.success,
        "n_evaluations": result.nfev
    }

# Example usage
evo_result = evolutionary_vqe(network, hamiltonian, n_parameters=4)
print(f"Evolutionary VQE result: {evo_result}")
```

### Gradient-Based Optimization with Parameter Shift

```python
def parameter_shift_gradient(network, cost_function, parameters, shift=np.pi/2):
    """Compute gradients using parameter shift rule."""
    
    gradients = np.zeros_like(parameters)
    
    for i in range(len(parameters)):
        # Shift parameter forward
        params_plus = parameters.copy()
        params_plus[i] += shift
        cost_plus = cost_function(params_plus)
        
        # Shift parameter backward
        params_minus = parameters.copy()
        params_minus[i] -= shift
        cost_minus = cost_function(params_minus)
        
        # Compute gradient
        gradients[i] = (cost_plus - cost_minus) / 2
    
    return gradients

def gradient_descent_vqe(network, hamiltonian, n_parameters, learning_rate=0.1, max_iterations=100):
    """VQE with gradient descent using parameter shift rule."""
    
    # Initialize parameters
    parameters = np.random.uniform(0, 2*np.pi, n_parameters)
    
    def vqe_cost(params):
        circuit = ansatz(params)
        state = network.apply_variational_circuit(circuit)
        return np.real(np.conj(state.amplitudes) @ hamiltonian @ state.amplitudes)
    
    cost_history = []
    
    for iteration in range(max_iterations):
        # Compute cost and gradient
        cost = vqe_cost(parameters)
        gradient = parameter_shift_gradient(network, vqe_cost, parameters)
        
        # Update parameters
        parameters -= learning_rate * gradient
        
        cost_history.append(cost)
        
        if iteration % 20 == 0:
            print(f"Iteration {iteration}, Cost: {cost:.6f}")
    
    return {
        "optimal_parameters": parameters,
        "optimal_cost": cost_history[-1],
        "cost_history": cost_history
    }

# Run gradient-based VQE
grad_vqe_result = gradient_descent_vqe(network, hamiltonian, n_parameters=4)
print(f"Gradient VQE final cost: {grad_vqe_result['optimal_cost']:.6f}")
```

## Performance Analysis

### Convergence Monitoring

```python
def analyze_convergence(cost_history, window_size=10):
    """Analyze convergence of variational optimization."""
    
    if len(cost_history) < window_size:
        return {"converged": False, "reason": "Insufficient iterations"}
    
    # Moving average
    moving_avg = np.convolve(cost_history, np.ones(window_size)/window_size, mode='valid')
    
    # Convergence criteria
    final_variance = np.var(moving_avg[-window_size:])
    relative_change = abs(moving_avg[-1] - moving_avg[-window_size]) / abs(moving_avg[-window_size])
    
    converged = final_variance < 1e-8 and relative_change < 1e-6
    
    return {
        "converged": converged,
        "final_variance": final_variance,
        "relative_change": relative_change,
        "moving_average": moving_avg
    }

# Analyze VQE convergence
convergence_analysis = analyze_convergence(grad_vqe_result['cost_history'])
print(f"Convergence analysis: {convergence_analysis}")
```

### Barren Plateau Detection

```python
def detect_barren_plateau(network, cost_function, n_parameters, n_samples=100):
    """Detect barren plateau in variational landscape."""
    
    gradient_variances = []
    
    for _ in range(n_samples):
        # Random parameter initialization
        random_params = np.random.uniform(0, 2*np.pi, n_parameters)
        
        # Compute gradient
        gradient = parameter_shift_gradient(network, cost_function, random_params)
        
        # Compute gradient variance
        grad_variance = np.var(gradient)
        gradient_variances.append(grad_variance)
    
    avg_gradient_variance = np.mean(gradient_variances)
    
    # Barren plateau threshold (problem-dependent)
    barren_threshold = 1e-6
    
    return {
        "barren_plateau_detected": avg_gradient_variance < barren_threshold,
        "average_gradient_variance": avg_gradient_variance,
        "gradient_variances": gradient_variances
    }

# Check for barren plateaus
plateau_analysis = detect_barren_plateau(
    network, 
    lambda params: vqe_cost(params),  # Define vqe_cost appropriately
    n_parameters=4
)
print(f"Barren plateau analysis: {plateau_analysis}")
```

## Best Practices

### Circuit Design

1. **Expressivity vs Trainability**: Balance circuit depth with trainability
2. **Problem-Tailored Ansätze**: Design circuits specific to problem structure
3. **Parameter Initialization**: Use informed initialization strategies
4. **Regularization**: Add constraints to prevent overfitting

### Optimization Strategies

1. **Multi-Scale Optimization**: Combine global and local optimization
2. **Adaptive Learning Rates**: Adjust learning rates during training
3. **Ensemble Methods**: Use multiple random initializations
4. **Early Stopping**: Prevent overfitting with validation monitoring

## Next Steps

- Explore practical [Examples](../examples/quantum-xor.md) using variational methods
- Learn about [Advanced Topics](../advanced/optimization.md) in quantum optimization
- See [API Reference](../api/inference.md) for variational algorithm documentation
- Check [Performance Optimization](../advanced/optimization.md) for scaling guidelines
