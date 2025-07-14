# Probabilistic Quantum Reasoner

[![PyPI](https://img.shields.io/pypi/v/probabilistic-quantum-reasoner.svg?label=PyPI&color=blue&logo=python&logoColor=white)](https://pypi.org/project/probabilistic-quantum-reasoner/)
[![Documentation Status](https://readthedocs.org/projects/probabilistic-quantum-reasoner/badge/?version=latest)](https://krish567366.github.io/probabilistic-quantum-reasoner/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

A **quantum-classical hybrid reasoning engine** for uncertainty-aware AI inference, fusing quantum probabilistic graphical models (QPGMs) with classical probabilistic logic.

## 🎯 Overview

The Probabilistic Quantum Reasoner implements a novel approach to AI reasoning by encoding knowledge using **quantum amplitude distributions** over Hilbert space, modeling uncertainty through entanglement and non-commutative conditional graphs, and enabling hybrid **Quantum Bayesian Networks** with causal, counterfactual, and abductive reasoning capabilities.

## 🧩 Key Features

- **Quantum Bayesian Networks**: Hybrid classical-quantum probabilistic graphical models
- **Quantum Belief Propagation**: Unitary message passing with amplitude-weighted inference
- **Causal Quantum Reasoning**: Do-calculus analog for quantum intervention logic
- **Multiple Backends**: Support for Qiskit, PennyLane, and classical simulation
- **Uncertainty Modeling**: Entanglement-based uncertainty representation
- **Counterfactual Reasoning**: Quantum counterfactuals using unitary interventions

## 🚀 Quick Start

### Installation

```bash
pip install probabilistic-quantum-reasoner
```

For development with extra features:

```bash
pip install probabilistic-quantum-reasoner[dev,docs,extras]
```

### Basic Usage

```python
from probabilistic_quantum_reasoner import QuantumBayesianNetwork
from probabilistic_quantum_reasoner.backends import QiskitBackend

# Create a quantum Bayesian network
qbn = QuantumBayesianNetwork(backend=QiskitBackend())

# Add quantum and classical nodes
weather = qbn.add_quantum_node("weather", ["sunny", "rainy"])
mood = qbn.add_stochastic_node("mood", ["happy", "sad"])

# Create entangled relationship
qbn.add_edge(weather, mood)
qbn.entangle([weather, mood])

# Perform quantum inference
result = qbn.infer(evidence={"weather": "sunny"})
print(f"Mood probabilities: {result}")

# Quantum intervention (do-calculus)
intervention_result = qbn.intervene("weather", "rainy")
print(f"Mood under intervention: {intervention_result}")
```

## 🧬 Mathematical Foundation

The library implements quantum probabilistic reasoning using:

- **Tensor Product Spaces**: Joint state representation as |ψ⟩ = Σᵢⱼ αᵢⱼ|iⱼ⟩
- **Amplitude Manipulation**: Via Kraus operators and parameterized unitaries
- **Density Matrix Operations**: Mixed state inference through partial tracing
- **Non-commutative Conditional Probability**: P_Q(A|B) ≠ P_Q(B|A) in general

## 📖 Documentation

- **[API Reference](https://krish567366.github.io/probabilistic-quantum-reasoner/api-reference/)**
- **[Architecture Guide](https://krish567366.github.io/probabilistic-quantum-reasoner/architecture/)**
- **[Examples & Tutorials](https://krish567366.github.io/probabilistic-quantum-reasoner/examples/)**

## 🧪 Examples

### Quantum XOR Reasoning
```python
# Create entangled XOR gate reasoning
qbn = QuantumBayesianNetwork()
a = qbn.add_quantum_node("A", [0, 1])
b = qbn.add_quantum_node("B", [0, 1])
xor = qbn.add_quantum_node("XOR", [0, 1])

qbn.add_quantum_xor_relationship(a, b, xor)
result = qbn.infer(evidence={"A": 1, "B": 0})
```

### Weather-Mood Causal Graph
```python
# Hybrid classical-quantum causal modeling
from probabilistic_quantum_reasoner.examples import WeatherMoodExample

example = WeatherMoodExample()
causal_effect = example.estimate_causal_effect("weather", "mood")
counterfactual = example.counterfactual_query("What if it was sunny?")
```

## 🛠️ Architecture

```
probabilistic_quantum_reasoner/
├── core/                    # Core network structures
│   ├── network.py          # QuantumBayesianNetwork
│   ├── nodes.py            # Quantum/Stochastic/Hybrid nodes
│   └── operators.py        # Quantum operators and gates
├── inference/              # Reasoning engines
│   ├── engine.py           # Main inference engine
│   ├── causal.py           # Causal reasoning
│   ├── belief_propagation.py
│   └── variational.py      # Variational quantum inference
├── backends/               # Backend implementations
│   ├── qiskit_backend.py
│   ├── pennylane_backend.py
│   └── simulator.py
└── examples/               # Example implementations
```

## 🔬 Research Applications

- **AGI Inference Scaffolds**: Uncertainty-aware reasoning for autonomous systems
- **Quantum Explainable AI (Q-XAI)**: Interpretable quantum decision making
- **Counterfactual Analysis**: "What-if" scenarios in quantum superposition
- **Epistemic Uncertainty Modeling**: Non-classical uncertainty representation

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use this library in your research, please cite:

```bibtex
@software{bajpai2025quantum,
  title={Probabilistic Quantum Reasoner: A Hybrid Quantum-Classical Reasoning Engine},
  author={Bajpai, Krishna},
  year={2025},
  url={https://github.com/krish567366/probabilistic-quantum-reasoner}
}
```

## 👨‍💻 Author

**Krishna Bajpai**
- Email: bajpaikrishna715@gmail.com
- GitHub: [@krish567366](https://github.com/krish567366)

## 🙏 Acknowledgments

- Quantum computing community for foundational algorithms
- Classical probabilistic reasoning research
- Open source quantum computing frameworks (Qiskit, PennyLane)
