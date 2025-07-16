# API Reference: Examples

This section provides detailed API documentation for the example implementations in the **Probabilistic Quantum Reasoner**.

## Example Networks

::: probabilistic_quantum_reasoner.examples.WeatherMoodExample
    options:
      show_source: false
      show_root_heading: true
      show_root_toc_entry: false
      merge_init_into_class: true

::: probabilistic_quantum_reasoner.examples.QuantumXORExample
    options:
      show_source: false
      show_root_heading: true
      show_root_toc_entry: false
      merge_init_into_class: true

::: probabilistic_quantum_reasoner.examples.QuantumPrisonersDilemmaExample
    options:
      show_source: false
      show_root_heading: true
      show_root_toc_entry: false
      merge_init_into_class: true

## Example Usage

### Weather-Mood Network

```python
from probabilistic_quantum_reasoner.examples import WeatherMoodExample

example = WeatherMoodExample()
network = example.create_network()
network.infer(query_nodes=["mood"], evidence={"weather": "sunny"})
```

### Quantum XOR Network

```python
from probabilistic_quantum_reasoner.examples import QuantumXORExample

xor_example = QuantumXORExample()
network = xor_example.create_network()
result = network.infer(query_nodes=["output"])
```

### Quantum Prisoner's Dilemma

```python
from probabilistic_quantum_reasoner.examples import QuantumPrisonersDilemmaExample

qpd = QuantumPrisonersDilemmaExample()
analysis = qpd.run_complete_game_analysis()
print(qpd.generate_game_report())
```
