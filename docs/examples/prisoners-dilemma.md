# Game Theory: Prisoner's Dilemma with Quantum Strategies

This example explores the famous Prisoner's Dilemma using quantum game theory concepts within the Probabilistic Quantum Reasoner framework. We'll demonstrate how quantum superposition and entanglement can change the strategic landscape of classical games.

## Classical Prisoner's Dilemma

First, let's implement the classical version to establish a baseline:

```python
import numpy as np
from probabilistic_quantum_reasoner import ProbabilisticQuantumReasoner
from probabilistic_quantum_reasoner.networks import BayesianNetwork
from probabilistic_quantum_reasoner.nodes import DiscreteNode, QuantumNode
from probabilistic_quantum_reasoner.quantum_ops import HadamardGate, CNOTGate, ParameterizedGate

def create_classical_prisoners_dilemma():
    """Create a classical Prisoner's Dilemma game."""
    
    network = BayesianNetwork(name="Classical Prisoner's Dilemma")
    
    # Player strategies: Cooperate or Defect
    player1_strategy = DiscreteNode(
        name="Player1_Strategy",
        states=["cooperate", "defect"],
        prior=[0.5, 0.5]  # Initially random strategy
    )
    
    player2_strategy = DiscreteNode(
        name="Player2_Strategy", 
        states=["cooperate", "defect"],
        prior=[0.5, 0.5]
    )
    
    # Payoff matrix for Player 1
    # Rows: Player1's choice, Columns: Player2's choice
    # (C,C): 3,3  (C,D): 0,5  (D,C): 5,0  (D,D): 1,1
    player1_payoff_cpt = np.array([
        # P1=C, P2=C -> Payoff levels [low, medium, high]
        [0.0, 0.0, 1.0],  # Reward for mutual cooperation (3 points)
        # P1=C, P2=D -> Sucker's payoff
        [1.0, 0.0, 0.0],  # Lowest payoff (0 points)
        # P1=D, P2=C -> Temptation payoff  
        [0.0, 0.0, 1.0],  # Highest payoff (5 points) - actually should be higher
        # P1=D, P2=D -> Punishment
        [0.0, 1.0, 0.0]   # Low payoff (1 point)
    ])
    
    player1_payoff = DiscreteNode(
        name="Player1_Payoff",
        states=["low", "medium", "high"],
        parents=[player1_strategy, player2_strategy],
        cpt=player1_payoff_cpt
    )
    
    # Symmetric payoff for Player 2
    player2_payoff_cpt = np.array([
        # P1=C, P2=C -> Both get reward
        [0.0, 0.0, 1.0],
        # P1=C, P2=D -> P2 gets temptation
        [0.0, 0.0, 1.0],  
        # P1=D, P2=C -> P2 gets sucker's payoff
        [1.0, 0.0, 0.0],
        # P1=D, P2=D -> Both get punishment
        [0.0, 1.0, 0.0]
    ])
    
    player2_payoff = DiscreteNode(
        name="Player2_Payoff",
        states=["low", "medium", "high"], 
        parents=[player1_strategy, player2_strategy],
        cpt=player2_payoff_cpt
    )
    
    network.add_nodes([
        player1_strategy, player2_strategy,
        player1_payoff, player2_payoff
    ])
    
    return network

def analyze_classical_game():
    """Analyze the classical Prisoner's Dilemma outcomes."""
    
    network = create_classical_prisoners_dilemma()
    reasoner = ProbabilisticQuantumReasoner(backend="classical")
    
    strategies = ["cooperate", "defect"]
    outcomes = {}
    
    print("Classical Prisoner's Dilemma Analysis:")
    print("P1 Strategy\tP2 Strategy\tP1 Payoff\tP2 Payoff")
    print("-" * 50)
    
    for p1_strat in strategies:
        for p2_strat in strategies:
            evidence = {
                "Player1_Strategy": p1_strat,
                "Player2_Strategy": p2_strat
            }
            
            result = reasoner.infer(
                network=network,
                query=["Player1_Payoff", "Player2_Payoff"],
                evidence=evidence
            )
            
            # Get most likely payoff
            p1_payoff = max(result["Player1_Payoff"], key=result["Player1_Payoff"].get)
            p2_payoff = max(result["Player2_Payoff"], key=result["Player2_Payoff"].get)
            
            outcomes[(p1_strat, p2_strat)] = (p1_payoff, p2_payoff)
            
            print(f"{p1_strat[:4]}\t\t{p2_strat[:4]}\t\t{p1_payoff}\t{p2_payoff}")
    
    return outcomes

# Run classical analysis
classical_outcomes = analyze_classical_game()
```

## Quantum Prisoner's Dilemma

Now let's implement the quantum version where players can use quantum strategies:

```python
def create_quantum_prisoners_dilemma(entanglement_strength=0.5):
    """
    Create a quantum Prisoner's Dilemma with entangled initial state.
    
    Args:
        entanglement_strength: Amount of initial entanglement between players
    """
    
    network = BayesianNetwork(name="Quantum Prisoner's Dilemma")
    
    # Quantum strategy nodes - players can be in superposition
    player1_quantum = QuantumNode(
        name="Player1_Quantum",
        num_qubits=1,
        initial_state="zero"  # Start in |0⟩ (cooperate)
    )
    
    player2_quantum = QuantumNode(
        name="Player2_Quantum", 
        num_qubits=1,
        initial_state="zero"  # Start in |0⟩ (cooperate)
    )
    
    # Entanglement node - creates initial correlation
    entanglement_node = QuantumNode(
        name="Entanglement",
        num_qubits=2,
        parents=[player1_quantum, player2_quantum],
        quantum_operations=[
            # Create entangled state based on strength parameter
            HadamardGate(qubit=0),  # Put player 1 in superposition
            ParameterizedGate(
                gate_type="RY",  # Rotation around Y-axis
                qubit=1,
                parameter=entanglement_strength * np.pi
            ),
            CNOTGate(control_qubit=0, target_qubit=1)  # Entangle players
        ]
    )
    
    # Quantum strategy application nodes
    player1_strategy_quantum = QuantumNode(
        name="Player1_Strategy_Quantum",
        num_qubits=1,
        parents=[entanglement_node],
        quantum_operations=[
            # Player 1 can apply quantum strategy
            ParameterizedGate(
                gate_type="RY",
                qubit=0, 
                parameter="strategy_param_1"  # Learned parameter
            )
        ]
    )
    
    player2_strategy_quantum = QuantumNode(
        name="Player2_Strategy_Quantum",
        num_qubits=1,
        parents=[entanglement_node],
        quantum_operations=[
            # Player 2 can apply quantum strategy
            ParameterizedGate(
                gate_type="RY",
                qubit=1,
                parameter="strategy_param_2"  # Learned parameter
            )
        ]
    )
    
    # Measurement nodes - convert quantum states to classical outcomes
    player1_measurement = DiscreteNode(
        name="Player1_Action",
        states=["cooperate", "defect"],
        parents=[player1_strategy_quantum],
        # Measurement probability based on quantum amplitudes
        cpt=np.array([
            [1.0, 0.0],  # |0⟩ -> cooperate with probability 1
            [0.0, 1.0]   # |1⟩ -> defect with probability 1  
        ])
    )
    
    player2_measurement = DiscreteNode(
        name="Player2_Action",
        states=["cooperate", "defect"],
        parents=[player2_strategy_quantum],
        cpt=np.array([
            [1.0, 0.0],  # |0⟩ -> cooperate
            [0.0, 1.0]   # |1⟩ -> defect
        ])
    )
    
    # Payoff calculation (same as classical)
    player1_payoff = DiscreteNode(
        name="Player1_Quantum_Payoff",
        states=["low", "medium", "high"],
        parents=[player1_measurement, player2_measurement],
        cpt=np.array([
            [0.0, 0.0, 1.0],  # (C,C) -> high
            [1.0, 0.0, 0.0],  # (C,D) -> low
            [0.0, 0.0, 1.0],  # (D,C) -> high (temptation)
            [0.0, 1.0, 0.0]   # (D,D) -> medium
        ])
    )
    
    player2_payoff = DiscreteNode(
        name="Player2_Quantum_Payoff", 
        states=["low", "medium", "high"],
        parents=[player1_measurement, player2_measurement],
        cpt=np.array([
            [0.0, 0.0, 1.0],  # (C,C) -> high
            [0.0, 0.0, 1.0],  # (C,D) -> high (temptation)
            [1.0, 0.0, 0.0],  # (D,C) -> low
            [0.0, 1.0, 0.0]   # (D,D) -> medium
        ])
    )
    
    network.add_nodes([
        player1_quantum, player2_quantum, entanglement_node,
        player1_strategy_quantum, player2_strategy_quantum,
        player1_measurement, player2_measurement,
        player1_payoff, player2_payoff
    ])
    
    return network

def simulate_quantum_game(entanglement_levels, num_trials=1000):
    """Simulate quantum Prisoner's Dilemma for different entanglement levels."""
    
    results = {}
    
    for entanglement in entanglement_levels:
        print(f"\nSimulating with entanglement strength: {entanglement:.2f}")
        
        network = create_quantum_prisoners_dilemma(entanglement)
        reasoner = ProbabilisticQuantumReasoner(backend="qiskit")
        
        cooperation_rate = 0
        mutual_cooperation_rate = 0
        
        for trial in range(num_trials):
            # Measure quantum strategies
            measurement = reasoner.measure(
                network=network,
                nodes=["Player1_Action", "Player2_Action"]
            )
            
            p1_action = measurement["Player1_Action"]
            p2_action = measurement["Player2_Action"]
            
            if p1_action == "cooperate":
                cooperation_rate += 0.5  # Each player contributes 0.5
            if p2_action == "cooperate":
                cooperation_rate += 0.5
                
            if p1_action == "cooperate" and p2_action == "cooperate":
                mutual_cooperation_rate += 1
        
        cooperation_rate /= num_trials
        mutual_cooperation_rate /= num_trials
        
        results[entanglement] = {
            "cooperation_rate": cooperation_rate,
            "mutual_cooperation_rate": mutual_cooperation_rate
        }
        
        print(f"  Cooperation rate: {cooperation_rate:.3f}")
        print(f"  Mutual cooperation rate: {mutual_cooperation_rate:.3f}")
    
    return results

# Simulate for different entanglement levels
entanglement_levels = [0.0, 0.25, 0.5, 0.75, 1.0]
quantum_results = simulate_quantum_game(entanglement_levels)
```

## Evolutionary Quantum Strategies

Let's implement an evolutionary approach where quantum strategies adapt over time:

```python
from probabilistic_quantum_reasoner.optimization import VariationalOptimizer
from probabilistic_quantum_reasoner.metrics import expected_payoff

class QuantumStrategyEvolution:
    """Evolve quantum strategies for the Prisoner's Dilemma."""
    
    def __init__(self, population_size=20, mutation_rate=0.1):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generation = 0
        
        # Initialize random strategy parameters
        self.population = np.random.uniform(
            0, 2*np.pi, 
            size=(population_size, 2)  # [strategy_param_1, strategy_param_2]
        )
        
        self.fitness_history = []
    
    def evaluate_fitness(self, strategy_params):
        """Evaluate fitness of a strategy pair against the population."""
        
        total_payoff = 0
        num_games = 50
        
        for _ in range(num_games):
            # Create network with these strategy parameters
            network = create_quantum_prisoners_dilemma()
            
            # Set strategy parameters
            network.set_parameters({
                "strategy_param_1": strategy_params[0],
                "strategy_param_2": strategy_params[1]
            })
            
            reasoner = ProbabilisticQuantumReasoner(backend="qiskit")
            
            # Play against random opponent from population
            opponent_idx = np.random.randint(self.population_size)
            opponent_params = self.population[opponent_idx]
            
            # Update network with opponent parameters
            network.set_parameters({
                "strategy_param_2": opponent_params[1]
            })
            
            # Measure outcome
            result = reasoner.infer(
                network=network,
                query=["Player1_Quantum_Payoff"]
            )
            
            # Calculate expected payoff
            payoff = (
                result["Player1_Quantum_Payoff"]["low"] * 0 +
                result["Player1_Quantum_Payoff"]["medium"] * 1 + 
                result["Player1_Quantum_Payoff"]["high"] * 3
            )
            
            total_payoff += payoff
        
        return total_payoff / num_games
    
    def evolve_generation(self):
        """Evolve the population for one generation."""
        
        # Evaluate fitness for all individuals
        fitness_scores = []
        for individual in self.population:
            fitness = self.evaluate_fitness(individual)
            fitness_scores.append(fitness)
        
        fitness_scores = np.array(fitness_scores)
        self.fitness_history.append(np.mean(fitness_scores))
        
        # Selection (tournament selection)
        new_population = []
        for _ in range(self.population_size):
            # Tournament selection
            tournament_size = 3
            tournament_indices = np.random.choice(
                self.population_size, tournament_size, replace=False
            )
            tournament_fitness = fitness_scores[tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            
            # Mutation
            parent = self.population[winner_idx].copy()
            if np.random.random() < self.mutation_rate:
                mutation = np.random.normal(0, 0.1, size=2)
                parent += mutation
                parent = np.clip(parent, 0, 2*np.pi)  # Keep in valid range
            
            new_population.append(parent)
        
        self.population = np.array(new_population)
        self.generation += 1
        
        return np.mean(fitness_scores), np.max(fitness_scores)
    
    def run_evolution(self, generations=50):
        """Run evolution for specified number of generations."""
        
        print(f"Running quantum strategy evolution for {generations} generations...")
        print("Generation\tAvg Fitness\tMax Fitness\tBest Strategy")
        print("-" * 60)
        
        for gen in range(generations):
            avg_fitness, max_fitness = self.evolve_generation()
            
            # Find best strategy
            fitness_scores = [self.evaluate_fitness(ind) for ind in self.population]
            best_idx = np.argmax(fitness_scores)
            best_strategy = self.population[best_idx]
            
            if gen % 10 == 0:  # Print every 10 generations
                print(f"{gen:3d}\t\t{avg_fitness:.3f}\t\t{max_fitness:.3f}\t\t"
                      f"[{best_strategy[0]:.2f}, {best_strategy[1]:.2f}]")
        
        return self.population[best_idx]

# Run evolutionary optimization
evolution = QuantumStrategyEvolution(population_size=20, mutation_rate=0.15)
best_strategy = evolution.run_evolution(generations=100)

print(f"\nBest evolved strategy: [{best_strategy[0]:.3f}, {best_strategy[1]:.3f}]")
print(f"Strategy interpretation:")
print(f"  Player 1 cooperation probability: {np.cos(best_strategy[0]/2)**2:.3f}")
print(f"  Player 2 cooperation probability: {np.cos(best_strategy[1]/2)**2:.3f}")
```

## Nash Equilibrium Analysis

Let's analyze the Nash equilibria in the quantum game:

```python
def find_quantum_nash_equilibria(grid_resolution=20):
    """Find Nash equilibria in the quantum Prisoner's Dilemma."""
    
    # Create parameter grid
    param_range = np.linspace(0, 2*np.pi, grid_resolution)
    payoff_matrix_p1 = np.zeros((grid_resolution, grid_resolution))
    payoff_matrix_p2 = np.zeros((grid_resolution, grid_resolution))
    
    print("Computing payoff matrix for Nash equilibrium analysis...")
    
    for i, p1_param in enumerate(param_range):
        for j, p2_param in enumerate(param_range):
            
            # Create network with specific parameters
            network = create_quantum_prisoners_dilemma()
            network.set_parameters({
                "strategy_param_1": p1_param,
                "strategy_param_2": p2_param
            })
            
            reasoner = ProbabilisticQuantumReasoner(backend="classical")
            
            # Calculate expected payoffs
            result = reasoner.infer(
                network=network,
                query=["Player1_Quantum_Payoff", "Player2_Quantum_Payoff"]
            )
            
            # Convert to numerical payoffs
            p1_payoff = (
                result["Player1_Quantum_Payoff"]["low"] * 0 +
                result["Player1_Quantum_Payoff"]["medium"] * 1 +
                result["Player1_Quantum_Payoff"]["high"] * 3
            )
            
            p2_payoff = (
                result["Player2_Quantum_Payoff"]["low"] * 0 +
                result["Player2_Quantum_Payoff"]["medium"] * 1 +
                result["Player2_Quantum_Payoff"]["high"] * 3
            )
            
            payoff_matrix_p1[i, j] = p1_payoff
            payoff_matrix_p2[i, j] = p2_payoff
    
    # Find Nash equilibria (best response dynamics)
    nash_equilibria = []
    
    for i in range(grid_resolution):
        for j in range(grid_resolution):
            # Check if (i,j) is a Nash equilibrium
            
            # Player 1's best response to player 2's strategy j
            p1_best_response = np.argmax(payoff_matrix_p1[:, j])
            
            # Player 2's best response to player 1's strategy i  
            p2_best_response = np.argmax(payoff_matrix_p2[i, :])
            
            # Nash equilibrium if both are best responding
            if p1_best_response == i and p2_best_response == j:
                nash_equilibria.append((
                    param_range[i], 
                    param_range[j],
                    payoff_matrix_p1[i, j],
                    payoff_matrix_p2[i, j]
                ))
    
    return nash_equilibria, payoff_matrix_p1, payoff_matrix_p2

# Find Nash equilibria
nash_points, p1_payoffs, p2_payoffs = find_quantum_nash_equilibria()

print(f"\nFound {len(nash_points)} Nash equilibria:")
print("P1 Strategy\tP2 Strategy\tP1 Payoff\tP2 Payoff\tCooperation Prob")
print("-" * 70)

for p1_strat, p2_strat, p1_pay, p2_pay in nash_points:
    # Convert to cooperation probabilities
    p1_coop_prob = np.cos(p1_strat/2)**2
    p2_coop_prob = np.cos(p2_strat/2)**2
    
    print(f"{p1_strat:.3f}\t\t{p2_strat:.3f}\t\t{p1_pay:.3f}\t\t{p2_pay:.3f}\t\t"
          f"({p1_coop_prob:.3f}, {p2_coop_prob:.3f})")
```

## Quantum Advantage Analysis

Let's analyze when quantum strategies provide an advantage:

```python
import matplotlib.pyplot as plt

def analyze_quantum_advantage():
    """Analyze quantum advantage in Prisoner's Dilemma."""
    
    # Compare classical vs quantum outcomes
    classical_network = create_classical_prisoners_dilemma()
    quantum_network = create_quantum_prisoners_dilemma(entanglement_strength=0.5)
    
    classical_reasoner = ProbabilisticQuantumReasoner(backend="classical")
    quantum_reasoner = ProbabilisticQuantumReasoner(backend="qiskit")
    
    # Classical Nash equilibrium (Defect, Defect)
    classical_result = classical_reasoner.infer(
        network=classical_network,
        query=["Player1_Payoff", "Player2_Payoff"],
        evidence={"Player1_Strategy": "defect", "Player2_Strategy": "defect"}
    )
    
    classical_payoff = (
        classical_result["Player1_Payoff"]["medium"] * 1  # Both get 1 point
    )
    
    # Quantum strategies with optimized parameters
    quantum_network.set_parameters({
        "strategy_param_1": np.pi/4,  # Quantum strategy
        "strategy_param_2": np.pi/4
    })
    
    quantum_trials = 1000
    quantum_payoffs = []
    
    for _ in range(quantum_trials):
        result = quantum_reasoner.infer(
            network=quantum_network,
            query=["Player1_Quantum_Payoff"]
        )
        
        payoff = (
            result["Player1_Quantum_Payoff"]["low"] * 0 +
            result["Player1_Quantum_Payoff"]["medium"] * 1 +
            result["Player1_Quantum_Payoff"]["high"] * 3
        )
        quantum_payoffs.append(payoff)
    
    avg_quantum_payoff = np.mean(quantum_payoffs)
    quantum_advantage = avg_quantum_payoff - classical_payoff
    
    print(f"\nQuantum Advantage Analysis:")
    print(f"Classical Nash equilibrium payoff: {classical_payoff:.3f}")
    print(f"Quantum strategy average payoff: {avg_quantum_payoff:.3f}")
    print(f"Quantum advantage: {quantum_advantage:.3f}")
    print(f"Relative improvement: {quantum_advantage/classical_payoff*100:.1f}%")
    
    # Plot payoff distributions
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(quantum_payoffs, bins=30, alpha=0.7, color='blue', label='Quantum')
    plt.axvline(classical_payoff, color='red', linestyle='--', label='Classical Nash')
    plt.xlabel('Payoff')
    plt.ylabel('Frequency')
    plt.title('Payoff Distribution')
    plt.legend()
    
    # Entanglement strength analysis
    plt.subplot(1, 2, 2)
    entanglement_strengths = np.linspace(0, 1, 20)
    avg_payoffs = []
    
    for strength in entanglement_strengths:
        network = create_quantum_prisoners_dilemma(strength)
        network.set_parameters({
            "strategy_param_1": np.pi/4,
            "strategy_param_2": np.pi/4
        })
        
        payoffs = []
        for _ in range(100):
            result = quantum_reasoner.infer(
                network=network,
                query=["Player1_Quantum_Payoff"]
            )
            payoff = (
                result["Player1_Quantum_Payoff"]["low"] * 0 +
                result["Player1_Quantum_Payoff"]["medium"] * 1 +
                result["Player1_Quantum_Payoff"]["high"] * 3
            )
            payoffs.append(payoff)
        
        avg_payoffs.append(np.mean(payoffs))
    
    plt.plot(entanglement_strengths, avg_payoffs, 'b-', label='Quantum')
    plt.axhline(classical_payoff, color='red', linestyle='--', label='Classical Nash')
    plt.xlabel('Entanglement Strength')
    plt.ylabel('Average Payoff')
    plt.title('Payoff vs Entanglement')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('quantum_advantage_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return quantum_advantage

# Analyze quantum advantage
advantage = analyze_quantum_advantage()
```

## Applications and Extensions

### Mechanism Design

Quantum games can be used for:

- **Auction design**: Quantum bidding strategies
- **Voting systems**: Quantum voting protocols  
- **Resource allocation**: Quantum fair division

### Multi-Player Games

Extension to n-player scenarios:

```python
def create_multiplayer_quantum_game(num_players=4):
    """Create a multi-player quantum game."""
    
    network = BayesianNetwork(name=f"{num_players}-Player Quantum Game")
    
    # Create quantum strategy nodes for each player
    players = []
    for i in range(num_players):
        player = QuantumNode(
            name=f"Player_{i}_Quantum",
            num_qubits=1,
            initial_state="superposition"
        )
        players.append(player)
    
    # Create entanglement between all players
    entanglement = QuantumNode(
        name="Global_Entanglement",
        num_qubits=num_players,
        parents=players,
        quantum_operations=[
            # Create GHZ state for maximum entanglement
            HadamardGate(qubit=0)
        ] + [
            CNOTGate(control_qubit=0, target_qubit=i) 
            for i in range(1, num_players)
        ]
    )
    
    # Add measurement and payoff nodes...
    # (Implementation details omitted for brevity)
    
    return network
```

### Behavioral Economics

Quantum games can model:

- **Irrationality**: Quantum superposition of choices
- **Bounded rationality**: Decoherence effects
- **Social preferences**: Entanglement between players

## Conclusion

This example demonstrates how quantum mechanics can fundamentally change game theory:

1. **Quantum strategies** expand the strategy space beyond classical pure/mixed strategies
2. **Entanglement** creates new forms of correlation between players
3. **Superposition** allows players to be in multiple strategic states simultaneously
4. **Measurement** introduces probabilistic outcomes even with deterministic strategies

The Probabilistic Quantum Reasoner provides a framework for exploring these quantum game-theoretic concepts and their applications to economics, computer science, and social sciences.
