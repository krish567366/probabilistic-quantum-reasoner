# Performance Optimization

This guide covers advanced techniques for optimizing the performance of the Probabilistic Quantum Reasoner across different hardware platforms and use cases.

## Performance Analysis Framework

### Profiling Tools

```python
import time
import memory_profiler
import cProfile
import pstats
from contextlib import contextmanager
from probabilistic_quantum_reasoner import ProbabilisticQuantumReasoner

class PerformanceProfiler:
    """Comprehensive performance profiling for PQR."""
    
    def __init__(self):
        self.measurements = {}
        self.memory_usage = {}
        
    @contextmanager
    def profile_time(self, operation_name):
        """Profile execution time of an operation."""
        start_time = time.perf_counter()
        start_memory = memory_profiler.memory_usage()[0]
        
        try:
            yield
        finally:
            end_time = time.perf_counter()
            end_memory = memory_profiler.memory_usage()[0]
            
            self.measurements[operation_name] = {
                'execution_time': end_time - start_time,
                'memory_delta': end_memory - start_memory,
                'peak_memory': max(memory_profiler.memory_usage())
            }
    
    def profile_function(self, func, *args, **kwargs):
        """Profile a function with detailed statistics."""
        profiler = cProfile.Profile()
        
        start_time = time.perf_counter()
        profiler.enable()
        
        result = func(*args, **kwargs)
        
        profiler.disable()
        end_time = time.perf_counter()
        
        # Analyze profile data
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        
        return {
            'result': result,
            'total_time': end_time - start_time,
            'stats': stats,
            'top_functions': stats.get_stats_profile().func_profiles
        }
    
    def benchmark_inference(self, network, reasoner, num_trials=100):
        """Benchmark inference performance."""
        times = []
        memory_usage = []
        
        for _ in range(num_trials):
            with self.profile_time("inference_trial"):
                result = reasoner.infer(
                    network=network,
                    query=list(network.nodes.keys())[:3],  # First 3 nodes
                    evidence={}
                )
            
            times.append(self.measurements["inference_trial"]["execution_time"])
            memory_usage.append(self.measurements["inference_trial"]["memory_delta"])
        
        return {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'mean_memory': np.mean(memory_usage),
            'total_memory': np.sum(memory_usage)
        }

# Usage example
profiler = PerformanceProfiler()

# Profile network creation
with profiler.profile_time("network_creation"):
    network = create_large_network(num_nodes=50)

# Profile inference
reasoner = ProbabilisticQuantumReasoner(backend="classical")
benchmark_results = profiler.benchmark_inference(network, reasoner)

print(f"Average inference time: {benchmark_results['mean_time']:.3f}s")
print(f"Memory usage: {benchmark_results['mean_memory']:.1f} MB")
```

### Memory Optimization

```python
import gc
import weakref
from typing import Dict, Any, Optional

class MemoryOptimizer:
    """Memory optimization utilities for PQR."""
    
    def __init__(self):
        self.cache_size_limit = 1000  # Maximum cached items
        self.weak_references = weakref.WeakSet()
        
    def optimize_network_memory(self, network):
        """Optimize memory usage of a Bayesian network."""
        
        # Convert dense arrays to sparse where appropriate
        for node_name, node in network.nodes.items():
            if hasattr(node, 'cpt'):
                node.cpt = self._sparsify_cpt(node.cpt)
            
            if hasattr(node, 'parents'):
                # Use weak references for parent relationships
                node._parent_refs = [weakref.ref(p) for p in node.parents]
        
        # Compact string representations
        self._intern_node_names(network)
        
        return network
    
    def _sparsify_cpt(self, cpt, threshold=1e-6):
        """Convert dense CPT to sparse representation if beneficial."""
        import scipy.sparse as sp
        
        if isinstance(cpt, np.ndarray):
            # Check sparsity
            zero_ratio = np.count_nonzero(cpt < threshold) / cpt.size
            
            if zero_ratio > 0.5:  # More than 50% zeros
                return sp.csr_matrix(cpt)
        
        return cpt
    
    def _intern_node_names(self, network):
        """Intern string names to save memory."""
        name_pool = {}
        
        for node in network.nodes.values():
            # Intern node name
            if node.name not in name_pool:
                name_pool[node.name] = sys.intern(node.name)
            node.name = name_pool[node.name]
            
            # Intern state names
            if hasattr(node, 'states'):
                for i, state in enumerate(node.states):
                    if state not in name_pool:
                        name_pool[state] = sys.intern(state)
                    node.states[i] = name_pool[state]
    
    def cleanup_unused_objects(self):
        """Force cleanup of unused objects."""
        # Clear weak references to deleted objects
        self.weak_references = weakref.WeakSet([
            obj for obj in self.weak_references if obj is not None
        ])
        
        # Force garbage collection
        collected = gc.collect()
        print(f"Garbage collected {collected} objects")
        
        return collected
    
    def monitor_memory_usage(self, func, *args, **kwargs):
        """Monitor memory usage during function execution."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Initial memory
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Execute function
        result = func(*args, **kwargs)
        
        # Final memory
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        return {
            'result': result,
            'memory_delta': final_memory - initial_memory,
            'peak_memory': final_memory
        }

# Memory optimization example
optimizer = MemoryOptimizer()

# Optimize existing network
optimized_network = optimizer.optimize_network_memory(large_network)

# Monitor memory during inference
memory_result = optimizer.monitor_memory_usage(
    reasoner.infer,
    network=optimized_network,
    query=["target_node"],
    evidence={}
)

print(f"Memory delta: {memory_result['memory_delta']:.1f} MB")
```

## Quantum Circuit Optimization

### Circuit Depth Reduction

```python
from probabilistic_quantum_reasoner.optimization import CircuitOptimizer

class QuantumCircuitOptimizer:
    """Optimize quantum circuits for better performance."""
    
    def __init__(self):
        self.optimization_passes = [
            'single_qubit_merge',
            'commutative_cancellation', 
            'redundant_gate_removal',
            'circuit_depth_reduction'
        ]
    
    def optimize_circuit(self, circuit, optimization_level=3):
        """Optimize quantum circuit based on optimization level."""
        
        optimized_circuit = circuit.copy()
        
        if optimization_level >= 1:
            optimized_circuit = self._merge_single_qubit_gates(optimized_circuit)
        
        if optimization_level >= 2:
            optimized_circuit = self._cancel_commutative_gates(optimized_circuit)
            optimized_circuit = self._remove_redundant_gates(optimized_circuit)
        
        if optimization_level >= 3:
            optimized_circuit = self._reduce_circuit_depth(optimized_circuit)
            optimized_circuit = self._optimize_two_qubit_gates(optimized_circuit)
        
        return optimized_circuit
    
    def _merge_single_qubit_gates(self, circuit):
        """Merge consecutive single-qubit gates."""
        new_gates = []
        gate_buffer = {}  # qubit -> list of gates
        
        for gate_info in circuit.gates:
            gate_name, qubits, params = gate_info
            
            if len(qubits) == 1:  # Single-qubit gate
                qubit = qubits[0]
                if qubit not in gate_buffer:
                    gate_buffer[qubit] = []
                gate_buffer[qubit].append(gate_info)
            else:
                # Flush single-qubit gate buffer
                for q, gates in gate_buffer.items():
                    if gates:
                        merged_gate = self._merge_single_qubit_sequence(gates)
                        if merged_gate:
                            new_gates.append(merged_gate)
                gate_buffer.clear()
                
                # Add two-qubit gate
                new_gates.append(gate_info)
        
        # Flush remaining gates
        for q, gates in gate_buffer.items():
            if gates:
                merged_gate = self._merge_single_qubit_sequence(gates)
                if merged_gate:
                    new_gates.append(merged_gate)
        
        circuit.gates = new_gates
        return circuit
    
    def _merge_single_qubit_sequence(self, gate_sequence):
        """Merge a sequence of single-qubit gates into one."""
        if not gate_sequence:
            return None
        
        # Compute combined rotation matrix
        combined_matrix = np.eye(2, dtype=complex)
        qubit = gate_sequence[0][1][0]
        
        for gate_name, qubits, params in gate_sequence:
            gate_matrix = self._get_gate_matrix(gate_name, params)
            combined_matrix = gate_matrix @ combined_matrix
        
        # Convert back to rotation parameters
        if np.allclose(combined_matrix, np.eye(2)):
            return None  # Identity operation, can be removed
        
        # Decompose into rotation gates (simplified)
        return self._matrix_to_rotation_gates(combined_matrix, qubit)
    
    def _cancel_commutative_gates(self, circuit):
        """Cancel commuting gates that result in identity."""
        new_gates = []
        i = 0
        
        while i < len(circuit.gates):
            gate = circuit.gates[i]
            
            # Look for canceling pair
            if i + 1 < len(circuit.gates):
                next_gate = circuit.gates[i + 1]
                
                if self._gates_cancel(gate, next_gate):
                    i += 2  # Skip both gates
                    continue
            
            new_gates.append(gate)
            i += 1
        
        circuit.gates = new_gates
        return circuit
    
    def _gates_cancel(self, gate1, gate2):
        """Check if two gates cancel each other."""
        gate1_name, qubits1, params1 = gate1
        gate2_name, qubits2, params2 = gate2
        
        # Same gate on same qubits
        if gate1_name == gate2_name and qubits1 == qubits2:
            # Check if they're inverses
            if gate1_name in ['X', 'Y', 'Z', 'H']:
                return True  # Self-inverse gates
            elif gate1_name.startswith('R') and params1 and params2:
                return abs(params1[0] + params2[0]) < 1e-10  # Opposite rotations
        
        return False
    
    def _reduce_circuit_depth(self, circuit):
        """Reduce circuit depth by parallelizing commuting gates."""
        # Group gates by dependency
        dependency_graph = self._build_dependency_graph(circuit)
        
        # Topological sort to find parallelizable operations
        parallel_layers = self._extract_parallel_layers(dependency_graph)
        
        # Reconstruct circuit with optimized ordering
        new_gates = []
        for layer in parallel_layers:
            new_gates.extend(layer)
        
        circuit.gates = new_gates
        return circuit
    
    def _build_dependency_graph(self, circuit):
        """Build dependency graph for circuit gates."""
        dependencies = {}
        qubit_last_gate = {}  # qubit -> last gate index
        
        for i, (gate_name, qubits, params) in enumerate(circuit.gates):
            dependencies[i] = []
            
            # Add dependencies on previous gates affecting same qubits
            for qubit in qubits:
                if qubit in qubit_last_gate:
                    dependencies[i].append(qubit_last_gate[qubit])
                qubit_last_gate[qubit] = i
        
        return dependencies
    
    def _extract_parallel_layers(self, dependency_graph):
        """Extract layers of parallelizable gates."""
        layers = []
        remaining_gates = set(dependency_graph.keys())
        
        while remaining_gates:
            # Find gates with no remaining dependencies
            ready_gates = []
            for gate_idx in remaining_gates:
                if all(dep not in remaining_gates for dep in dependency_graph[gate_idx]):
                    ready_gates.append(gate_idx)
            
            if not ready_gates:
                # Circular dependency or error
                ready_gates = list(remaining_gates)
            
            layers.append(ready_gates)
            remaining_gates -= set(ready_gates)
        
        return layers

# Circuit optimization example
circuit_optimizer = QuantumCircuitOptimizer()

# Create test circuit
circuit = backend.create_circuit(num_qubits=4)
circuit.add_hadamard(0)
circuit.add_hadamard(0)  # Redundant - will be removed
circuit.add_cnot(0, 1)
circuit.add_rotation("RY", 2, np.pi/4)
circuit.add_rotation("RY", 2, -np.pi/4)  # Cancels previous rotation

print(f"Original circuit depth: {circuit.depth()}")

# Optimize circuit
optimized_circuit = circuit_optimizer.optimize_circuit(circuit, optimization_level=3)

print(f"Optimized circuit depth: {optimized_circuit.depth()}")
```

### Noise-Aware Optimization

```python
class NoiseAwareOptimizer:
    """Optimize circuits considering quantum noise."""
    
    def __init__(self, noise_model):
        self.noise_model = noise_model
        self.gate_fidelities = self._compute_gate_fidelities()
    
    def _compute_gate_fidelities(self):
        """Compute fidelities for different gates under noise."""
        fidelities = {}
        
        # Single-qubit gate fidelities
        fidelities['H'] = 0.999
        fidelities['X'] = 0.999
        fidelities['Y'] = 0.999
        fidelities['Z'] = 0.9995  # Virtual Z-gate, higher fidelity
        fidelities['RX'] = 0.998
        fidelities['RY'] = 0.998
        fidelities['RZ'] = 0.9995
        
        # Two-qubit gate fidelities
        fidelities['CNOT'] = 0.99
        fidelities['CZ'] = 0.992
        
        return fidelities
    
    def optimize_for_noise(self, circuit):
        """Optimize circuit considering noise characteristics."""
        optimized_circuit = circuit.copy()
        
        # Replace gates with higher-fidelity equivalents
        optimized_circuit = self._replace_low_fidelity_gates(optimized_circuit)
        
        # Minimize two-qubit gates
        optimized_circuit = self._minimize_two_qubit_gates(optimized_circuit)
        
        # Add error correction if beneficial
        optimized_circuit = self._add_error_correction(optimized_circuit)
        
        return optimized_circuit
    
    def _replace_low_fidelity_gates(self, circuit):
        """Replace gates with higher-fidelity alternatives."""
        new_gates = []
        
        for gate_name, qubits, params in circuit.gates:
            if gate_name == 'Y':
                # Replace Y with Z-RX-Z sequence (higher fidelity)
                new_gates.extend([
                    ('Z', qubits, []),
                    ('RX', qubits, [np.pi]),
                    ('Z', qubits, [])
                ])
            elif gate_name == 'CNOT' and len(qubits) == 2:
                # Check if CZ is available and has higher fidelity
                if self.gate_fidelities.get('CZ', 0) > self.gate_fidelities.get('CNOT', 0):
                    # Convert CNOT to CZ + single-qubit gates
                    control, target = qubits
                    new_gates.extend([
                        ('H', [target], []),
                        ('CZ', qubits, []),
                        ('H', [target], [])
                    ])
                else:
                    new_gates.append((gate_name, qubits, params))
            else:
                new_gates.append((gate_name, qubits, params))
        
        circuit.gates = new_gates
        return circuit
    
    def _minimize_two_qubit_gates(self, circuit):
        """Minimize the number of two-qubit gates."""
        # Count current two-qubit gates
        two_qubit_count = sum(1 for gate_name, qubits, _ in circuit.gates 
                             if len(qubits) == 2)
        
        if two_qubit_count <= 5:  # Already minimal
            return circuit
        
        # Apply more aggressive single-qubit optimizations
        optimized = circuit.copy()
        
        # Try to decompose multi-qubit gates differently
        optimized = self._decompose_for_minimal_cnots(optimized)
        
        return optimized
    
    def _add_error_correction(self, circuit):
        """Add error correction if circuit is long enough to benefit."""
        circuit_depth = circuit.depth()
        
        if circuit_depth > 100:  # Only for deep circuits
            # Add simple repetition code
            return self._add_repetition_code(circuit)
        
        return circuit
    
    def estimate_circuit_fidelity(self, circuit):
        """Estimate overall circuit fidelity under noise."""
        total_fidelity = 1.0
        
        for gate_name, qubits, params in circuit.gates:
            gate_fidelity = self.gate_fidelities.get(gate_name, 0.95)
            total_fidelity *= gate_fidelity
        
        # Account for decoherence time
        circuit_time = self._estimate_circuit_time(circuit)
        decoherence_factor = np.exp(-circuit_time / self.noise_model.t1_time)
        
        return total_fidelity * decoherence_factor

# Noise-aware optimization example
from probabilistic_quantum_reasoner.noise import DepolarizingNoise

noise_model = DepolarizingNoise(probability=0.01)
noise_optimizer = NoiseAwareOptimizer(noise_model)

# Optimize for noise
noisy_circuit = create_test_circuit()
optimized_for_noise = noise_optimizer.optimize_for_noise(noisy_circuit)

original_fidelity = noise_optimizer.estimate_circuit_fidelity(noisy_circuit)
optimized_fidelity = noise_optimizer.estimate_circuit_fidelity(optimized_for_noise)

print(f"Original fidelity: {original_fidelity:.4f}")
print(f"Optimized fidelity: {optimized_fidelity:.4f}")
print(f"Improvement: {(optimized_fidelity/original_fidelity - 1)*100:.1f}%")
```

## Classical Inference Optimization

### Sparse Matrix Operations

```python
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

class SparseInferenceEngine:
    """Inference engine optimized for sparse networks."""
    
    def __init__(self, sparsity_threshold=0.1):
        self.sparsity_threshold = sparsity_threshold
        self.sparse_cache = {}
    
    def optimize_network_for_sparsity(self, network):
        """Convert network to sparse representation."""
        
        for node_name, node in network.nodes.items():
            if hasattr(node, 'cpt'):
                # Convert CPT to sparse if beneficial
                cpt = node.cpt
                sparsity = np.count_nonzero(cpt) / cpt.size
                
                if sparsity < self.sparsity_threshold:
                    node.cpt = sp.csr_matrix(cpt)
                    print(f"Converted {node_name} CPT to sparse (sparsity: {sparsity:.3f})")
        
        return network
    
    def sparse_variable_elimination(self, network, query_vars, evidence):
        """Perform variable elimination using sparse operations."""
        
        # Convert factors to sparse matrices
        sparse_factors = self._convert_factors_to_sparse(network, evidence)
        
        # Determine elimination order
        elimination_order = self._compute_elimination_order(network, query_vars)
        
        # Eliminate variables one by one
        for var in elimination_order:
            if var not in query_vars:
                sparse_factors = self._eliminate_variable_sparse(sparse_factors, var)
        
        # Compute final marginals
        return self._compute_marginals_sparse(sparse_factors, query_vars)
    
    def _convert_factors_to_sparse(self, network, evidence):
        """Convert network factors to sparse representation."""
        factors = []
        
        for node_name, node in network.nodes.items():
            if hasattr(node, 'cpt'):
                factor = node.cpt
                
                # Apply evidence
                if node_name in evidence:
                    factor = self._apply_evidence_to_factor(factor, evidence[node_name])
                
                # Convert to sparse if not already
                if not sp.issparse(factor):
                    factor = sp.csr_matrix(factor)
                
                factors.append({
                    'factor': factor,
                    'variables': [node_name] + [p.name for p in node.parents],
                    'name': node_name
                })
        
        return factors
    
    def _eliminate_variable_sparse(self, factors, var):
        """Eliminate a variable using sparse operations."""
        # Find factors involving the variable
        relevant_factors = [f for f in factors if var in f['variables']]
        other_factors = [f for f in factors if var not in f['variables']]
        
        if not relevant_factors:
            return factors
        
        # Multiply relevant factors
        product_factor = relevant_factors[0]['factor']
        product_variables = relevant_factors[0]['variables']
        
        for factor_info in relevant_factors[1:]:
            product_factor = self._multiply_sparse_factors(
                product_factor, factor_info['factor']
            )
            product_variables = list(set(product_variables + factor_info['variables']))
        
        # Sum out the variable
        marginalized_factor = self._marginalize_sparse(product_factor, var, product_variables)
        marginalized_variables = [v for v in product_variables if v != var]
        
        # Add marginalized factor to other factors
        other_factors.append({
            'factor': marginalized_factor,
            'variables': marginalized_variables,
            'name': f'marginalized_{var}'
        })
        
        return other_factors
    
    def _multiply_sparse_factors(self, factor1, factor2):
        """Multiply two sparse factors."""
        # This is a simplified implementation
        # Real implementation would handle arbitrary tensor operations
        
        if factor1.shape == factor2.shape:
            return factor1.multiply(factor2)
        else:
            # Handle different shapes through broadcasting
            return sp.kron(factor1, factor2)
    
    def _marginalize_sparse(self, factor, var, variables):
        """Marginalize out a variable from sparse factor."""
        # Simplified marginalization for sparse matrices
        # Real implementation would handle arbitrary tensor marginalization
        
        var_index = variables.index(var)
        
        if factor.ndim == 2 and var_index == 0:
            return sp.csr_matrix(factor.sum(axis=0))
        elif factor.ndim == 2 and var_index == 1:
            return sp.csr_matrix(factor.sum(axis=1))
        else:
            # For higher dimensions, convert to dense temporarily
            dense_factor = factor.toarray()
            marginalized = np.sum(dense_factor, axis=var_index)
            return sp.csr_matrix(marginalized)

# Sparse optimization example
sparse_engine = SparseInferenceEngine(sparsity_threshold=0.15)

# Optimize network for sparsity
sparse_network = sparse_engine.optimize_network_for_sparsity(large_network)

# Perform sparse inference
start_time = time.time()
sparse_result = sparse_engine.sparse_variable_elimination(
    sparse_network, 
    query_vars=["target"],
    evidence={"obs1": "value1"}
)
sparse_time = time.time() - start_time

print(f"Sparse inference time: {sparse_time:.3f}s")
```

### Parallel Classical Inference

```python
import concurrent.futures
import multiprocessing as mp
from functools import partial

class ParallelInferenceEngine:
    """Parallel inference engine for classical networks."""
    
    def __init__(self, num_workers=None):
        self.num_workers = num_workers or mp.cpu_count()
        self.chunk_size = 1000  # For batch operations
    
    def parallel_variable_elimination(self, network, queries, evidences):
        """Perform variable elimination in parallel for multiple queries."""
        
        # Create worker function
        worker_func = partial(self._single_inference_worker, network)
        
        # Execute in parallel
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [
                executor.submit(worker_func, query, evidence)
                for query, evidence in zip(queries, evidences)
            ]
            
            results = []
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
        
        return results
    
    def _single_inference_worker(self, network, query, evidence):
        """Worker function for single inference task."""
        from probabilistic_quantum_reasoner.inference import BeliefPropagation
        
        bp = BeliefPropagation()
        return bp.infer(network, query, evidence)
    
    def parallel_factor_multiplication(self, factors_list):
        """Multiply factors in parallel."""
        
        if len(factors_list) <= 2:
            return self._multiply_factors_sequential(factors_list)
        
        # Divide factors into chunks
        chunk_size = max(2, len(factors_list) // self.num_workers)
        chunks = [factors_list[i:i+chunk_size] 
                 for i in range(0, len(factors_list), chunk_size)]
        
        # Multiply within chunks in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [
                executor.submit(self._multiply_factors_sequential, chunk)
                for chunk in chunks
            ]
            
            chunk_results = [future.result() for future in futures]
        
        # Multiply chunk results sequentially
        return self._multiply_factors_sequential(chunk_results)
    
    def _multiply_factors_sequential(self, factors):
        """Multiply a list of factors sequentially."""
        if not factors:
            return None
        
        result = factors[0]
        for factor in factors[1:]:
            result = np.multiply(result, factor)
        
        return result
    
    def parallel_sampling(self, network, num_samples, batch_size=None):
        """Generate samples in parallel."""
        
        if batch_size is None:
            batch_size = max(1, num_samples // self.num_workers)
        
        # Create batches
        batches = []
        remaining = num_samples
        while remaining > 0:
            current_batch = min(batch_size, remaining)
            batches.append(current_batch)
            remaining -= current_batch
        
        # Generate samples in parallel
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [
                executor.submit(self._generate_samples_batch, network, batch_size)
                for batch_size in batches
            ]
            
            all_samples = []
            for future in futures:
                all_samples.extend(future.result())
        
        return all_samples
    
    def _generate_samples_batch(self, network, num_samples):
        """Generate a batch of samples."""
        samples = []
        
        for _ in range(num_samples):
            sample = {}
            
            # Topological sampling
            for node_name in self._topological_sort(network):
                node = network.nodes[node_name]
                
                if hasattr(node, 'sample'):
                    # Sample from node distribution
                    value = node.sample(evidence=sample)
                    sample[node_name] = value
                else:
                    # Default sampling
                    if hasattr(node, 'states'):
                        value = np.random.choice(node.states)
                        sample[node_name] = value
            
            samples.append(sample)
        
        return samples
    
    def _topological_sort(self, network):
        """Topological sort of network nodes."""
        # Simplified topological sort
        sorted_nodes = []
        in_degree = {name: 0 for name in network.nodes}
        
        # Compute in-degrees
        for node_name, node in network.nodes.items():
            if hasattr(node, 'parents'):
                in_degree[node_name] = len(node.parents)
        
        # Find nodes with no incoming edges
        queue = [name for name, degree in in_degree.items() if degree == 0]
        
        while queue:
            current = queue.pop(0)
            sorted_nodes.append(current)
            
            # Update in-degrees
            current_node = network.nodes[current]
            if hasattr(current_node, 'children'):
                for child in current_node.children:
                    in_degree[child.name] -= 1
                    if in_degree[child.name] == 0:
                        queue.append(child.name)
        
        return sorted_nodes

# Parallel inference example
parallel_engine = ParallelInferenceEngine(num_workers=4)

# Multiple queries in parallel
queries = [["node1"], ["node2"], ["node3"]]
evidences = [{}, {"obs": "val1"}, {"obs": "val2"}]

parallel_results = parallel_engine.parallel_variable_elimination(
    network, queries, evidences
)

print(f"Processed {len(queries)} queries in parallel")

# Parallel sampling
samples = parallel_engine.parallel_sampling(network, num_samples=10000)
print(f"Generated {len(samples)} samples in parallel")
```

## Hardware-Specific Optimizations

### GPU Acceleration

```python
import cupy as cp
import numpy as np

class GPUOptimizedReasoner:
    """GPU-optimized quantum reasoner."""
    
    def __init__(self, device_id=0):
        cp.cuda.Device(device_id).use()
        self.device_id = device_id
        self.stream = cp.cuda.Stream()
        
    def gpu_accelerated_inference(self, network, query, evidence):
        """Perform inference using GPU acceleration."""
        
        with self.stream:
            # Convert network to GPU arrays
            gpu_factors = self._network_to_gpu(network)
            
            # Apply evidence on GPU
            gpu_factors = self._apply_evidence_gpu(gpu_factors, evidence)
            
            # Perform inference on GPU
            result = self._infer_on_gpu(gpu_factors, query)
            
            # Transfer result back to CPU
            cpu_result = cp.asnumpy(result)
        
        return cpu_result
    
    def _network_to_gpu(self, network):
        """Transfer network to GPU memory."""
        gpu_factors = []
        
        for node_name, node in network.nodes.items():
            if hasattr(node, 'cpt'):
                gpu_cpt = cp.asarray(node.cpt)
                gpu_factors.append({
                    'name': node_name,
                    'cpt': gpu_cpt,
                    'variables': [node_name] + [p.name for p in node.parents]
                })
        
        return gpu_factors
    
    def batch_quantum_simulation(self, circuits, shots_per_circuit):
        """Simulate multiple quantum circuits in parallel on GPU."""
        
        # Batch state vector computations
        batch_states = []
        
        for circuit in circuits:
            state = self._compute_state_vector_gpu(circuit)
            batch_states.append(state)
        
        # Stack states for batch processing
        batch_tensor = cp.stack(batch_states)
        
        # Batch measurement simulation
        results = self._batch_measure_gpu(batch_tensor, shots_per_circuit)
        
        return results
    
    def _compute_state_vector_gpu(self, circuit):
        """Compute state vector on GPU."""
        num_qubits = circuit.num_qubits
        state = cp.zeros(2**num_qubits, dtype=cp.complex128)
        state[0] = 1.0  # |00...0⟩
        
        # Apply gates using GPU operations
        for gate_name, qubits, params in circuit.gates:
            gate_matrix = self._get_gate_matrix_gpu(gate_name, params)
            state = self._apply_gate_gpu(state, gate_matrix, qubits, num_qubits)
        
        return state
    
    def _apply_gate_gpu(self, state, gate_matrix, qubits, num_qubits):
        """Apply quantum gate on GPU."""
        # This is a simplified implementation
        # Real implementation would use tensor contractions
        
        if len(qubits) == 1:
            # Single-qubit gate
            full_matrix = self._construct_full_matrix_gpu(gate_matrix, qubits[0], num_qubits)
            return full_matrix @ state
        else:
            # Multi-qubit gate (simplified)
            return gate_matrix @ state  # Placeholder
    
    def _batch_measure_gpu(self, batch_states, shots_per_circuit):
        """Perform batch measurements on GPU."""
        batch_results = []
        
        for i, state in enumerate(batch_states):
            # Compute probabilities
            probabilities = cp.abs(state) ** 2
            
            # Sample on GPU
            samples = cp.random.choice(
                len(probabilities), 
                size=shots_per_circuit[i],
                p=probabilities
            )
            
            # Convert to measurement outcomes
            outcomes = {}
            for sample in samples:
                bitstring = format(int(sample), f'0{int(cp.log2(len(state)))}b')
                outcomes[bitstring] = outcomes.get(bitstring, 0) + 1
            
            batch_results.append(outcomes)
        
        return batch_results

# GPU optimization example
gpu_reasoner = GPUOptimizedReasoner(device_id=0)

# Batch circuit simulation
test_circuits = [create_test_circuit(i) for i in range(10)]
shots_list = [1000] * 10

gpu_results = gpu_reasoner.batch_quantum_simulation(test_circuits, shots_list)
print(f"Simulated {len(test_circuits)} circuits on GPU")
```

### FPGA Acceleration

```python
class FPGAAcceleratedBackend:
    """FPGA-accelerated backend for specific operations."""
    
    def __init__(self, fpga_device="/dev/fpga0"):
        self.fpga_device = fpga_device
        self.fpga_available = self._check_fpga_availability()
        
    def _check_fpga_availability(self):
        """Check if FPGA device is available."""
        try:
            import os
            return os.path.exists(self.fpga_device)
        except:
            return False
    
    def fpga_accelerated_matrix_multiplication(self, matrices):
        """Perform matrix multiplication on FPGA."""
        
        if not self.fpga_available:
            # Fallback to CPU
            return self._cpu_matrix_multiplication(matrices)
        
        # FPGA-specific implementation
        return self._fpga_multiply_matrices(matrices)
    
    def _fpga_multiply_matrices(self, matrices):
        """FPGA matrix multiplication implementation."""
        # This would interface with actual FPGA hardware
        # Using placeholder implementation
        
        result = matrices[0]
        for matrix in matrices[1:]:
            # Simulate FPGA operation with optimized CPU code
            result = np.dot(result, matrix)
        
        return result
    
    def fpga_quantum_simulation(self, circuit):
        """Simulate quantum circuit on FPGA."""
        
        if not self.fpga_available:
            return None
        
        # FPGA-specific quantum simulation
        # This would use dedicated quantum simulation cores on FPGA
        return self._fpga_simulate_circuit(circuit)

# FPGA example
fpga_backend = FPGAAcceleratedBackend()

if fpga_backend.fpga_available:
    print("FPGA acceleration available")
else:
    print("FPGA not available, using CPU fallback")
```

## Benchmark Results

### Performance Comparison

```python
def comprehensive_benchmark():
    """Comprehensive performance benchmark."""
    
    # Test configurations
    configurations = [
        {"backend": "classical", "optimization": "none"},
        {"backend": "classical", "optimization": "sparse"},
        {"backend": "classical", "optimization": "parallel"},
        {"backend": "qiskit", "optimization": "none"},
        {"backend": "qiskit", "optimization": "circuit_opt"},
        {"backend": "gpu_classical", "optimization": "gpu"},
    ]
    
    # Test networks of different sizes
    network_sizes = [10, 25, 50, 100]
    
    results = {}
    
    for config in configurations:
        config_name = f"{config['backend']}_{config['optimization']}"
        results[config_name] = {}
        
        for size in network_sizes:
            print(f"Benchmarking {config_name} with {size} nodes...")
            
            # Create test network
            network = create_test_network(size)
            
            # Setup reasoner with configuration
            reasoner = setup_reasoner(config)
            
            # Benchmark inference
            times = []
            for _ in range(10):  # 10 trials
                start_time = time.time()
                result = reasoner.infer(
                    network=network,
                    query=list(network.nodes.keys())[:3],
                    evidence={}
                )
                end_time = time.time()
                times.append(end_time - start_time)
            
            results[config_name][size] = {
                'mean_time': np.mean(times),
                'std_time': np.std(times),
                'min_time': np.min(times)
            }
    
    return results

# Run comprehensive benchmark
benchmark_results = comprehensive_benchmark()

# Display results
print("\nBenchmark Results:")
print("="*50)

for config, size_results in benchmark_results.items():
    print(f"\n{config}:")
    for size, metrics in size_results.items():
        print(f"  {size} nodes: {metrics['mean_time']:.3f}s ± {metrics['std_time']:.3f}s")
```

## Best Practices Summary

### Performance Optimization Guidelines

1. **Profile First**: Always profile before optimizing
2. **Start with Algorithms**: Algorithmic improvements usually beat micro-optimizations
3. **Use Appropriate Data Structures**: Sparse vs dense, GPU vs CPU arrays
4. **Cache Wisely**: Cache expensive computations but manage memory
5. **Parallelize When Possible**: Use all available cores and accelerators
6. **Optimize for Your Use Case**: Different applications need different optimizations

### Memory Management

1. **Monitor Memory Usage**: Track memory consumption during inference
2. **Use Weak References**: Prevent memory leaks in large networks
3. **Clear Caches Regularly**: Implement cache eviction policies
4. **Optimize Data Types**: Use appropriate precision (float32 vs float64)

### Hardware Utilization

1. **Match Backend to Hardware**: Use GPU backends for GPU systems
2. **Batch Operations**: Process multiple items together when possible
3. **Minimize Data Transfer**: Keep data on accelerators when possible
4. **Use Vendor Optimizations**: Leverage hardware-specific libraries

This comprehensive guide provides the foundation for optimizing PQR performance across different hardware platforms and use cases.
