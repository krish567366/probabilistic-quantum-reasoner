# 🔌 API Reference: Backends

This section documents the backend interfaces — both classical and quantum — that power simulation, execution, and quantum-classical hybrid computation in the **Probabilistic Quantum Reasoner**.

---

## 🧠 Backend Base Classes

### Abstract Backend Interface

::: probabilistic_quantum_reasoner.backends.simulator.Backend
    options:
      show_source: false
      merge_init_into_class: true
      show_root_heading: true
      show_root_toc_entry: false

### Classical Simulator Backend

::: probabilistic_quantum_reasoner.backends.simulator.ClassicalSimulator
    options:
      show_source: false
      merge_init_into_class: true
      show_root_heading: true
      show_root_toc_entry: false

---

## ⚛️ Quantum Backends

### Qiskit Backend

::: probabilistic_quantum_reasoner.backends.qiskit_backend.QiskitBackend
    options:
      show_source: false
      merge_init_into_class: true
      show_root_heading: true
      show_root_toc_entry: false

### PennyLane Backend

::: probabilistic_quantum_reasoner.backends.pennylane_backend.PennyLaneBackend
    options:
      show_source: false
      merge_init_into_class: true
      show_root_heading: true
      show_root_toc_entry: false

---

## 🧪 Noise Modeling

### Noise Model Base Class

::: probabilistic_quantum_reasoner.backends.simulator.NoiseModel
    options:
      show_source: false
      merge_init_into_class: true
      show_root_heading: true
      show_root_toc_entry: false
