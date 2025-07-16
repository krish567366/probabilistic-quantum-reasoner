# 🧠 Inference API Reference

This section documents the inference engines and algorithms used for probabilistic reasoning in quantum-classical hybrid networks.

---

## 🔄 Core Inference Engine

::: probabilistic_quantum_reasoner.inference.engine.QuantumInferenceEngine
    options:
      show_source: false
      merge_init_into_class: true
      show_root_heading: true
      show_root_toc_entry: false

---

## 🧠 Variational Inference

::: probabilistic_quantum_reasoner.inference.variational.VariationalQuantumInference
    options:
      show_source: false
      merge_init_into_class: true
      show_root_heading: true
      show_root_toc_entry: false

::: probabilistic_quantum_reasoner.inference.variational.PennyLaneCircuit
    options:
      show_source: false
      show_root_heading: true
      show_root_toc_entry: false

::: probabilistic_quantum_reasoner.inference.variational.QiskitCircuit
    options:
      show_source: false
      show_root_heading: true
      show_root_toc_entry: false

::: probabilistic_quantum_reasoner.inference.variational.Optimizer
    options:
      show_source: false
      show_root_heading: true
      show_root_toc_entry: false

::: probabilistic_quantum_reasoner.inference.variational.AdamOptimizer
    options:
      show_source: false
      show_root_heading: true
      show_root_toc_entry: false

::: probabilistic_quantum_reasoner.inference.variational.SGDOptimizer
    options:
      show_source: false
      show_root_heading: true
      show_root_toc_entry: false

---

## 🧠 Belief Propagation

::: probabilistic_quantum_reasoner.inference.belief_propagation.QuantumBeliefPropagation
    options:
      show_source: false
      show_root_heading: true
      show_root_toc_entry: false

---

## 🧠 Causal Inference

::: probabilistic_quantum_reasoner.inference.causal.QuantumCausalInference
    options:
      show_source: false
      show_root_heading: true
      show_root_toc_entry: false

---

## 🧪 Example Usage

### 📌 Basic Inference

```python
from probabilistic_quantum_reasoner.inference.engine import QuantumInferenceEngine
engine = QuantumInferenceEngine(network)
result = engine.infer(query_nodes=["A", "B"])
