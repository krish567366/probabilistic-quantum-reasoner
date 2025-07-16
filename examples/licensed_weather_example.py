#!/usr/bin/env python3
"""
Licensed Example: Weather Prediction with Quantum Reasoning

This example demonstrates the QuantumMeta license integration in the
Probabilistic Quantum Reasoner. All quantum operations require valid
license activation.

Contact: bajpaikrishna715@gmail.com for licensing information.
"""

try:
    # Import will automatically validate core license
    from probabilistic_quantum_reasoner import (
        QuantumBayesianNetwork,
        QuantumNode,
        StochasticNode,
        QiskitBackend
    )
    from probabilistic_quantum_reasoner.licensing import display_license_info, get_machine_id
    
    print("🎉 Successfully imported Probabilistic Quantum Reasoner!")
    print(f"🔧 Your Machine ID: {get_machine_id()}")
    print("\n" + "="*80)
    
    # Display license information
    display_license_info()
    
    print("\n" + "="*80)
    print("📊 Creating a simple weather prediction network...")
    
    # Create network (requires core license)
    network = QuantumBayesianNetwork(name="Weather Prediction Network")
    print("✅ Network created successfully!")
    
    # Add classical node (requires basic_networks feature)
    weather = network.add_node("weather", "stochastic", outcome_space=["sunny", "rainy"])
    print("✅ Added classical weather node!")
    
    try:
        # Try to add quantum node (requires pro license with quantum_nodes feature)
        quantum_mood = network.add_node("quantum_mood", "quantum", outcome_space=["happy", "sad"])
        print("✅ Added quantum mood node!")
        
        # Try quantum entanglement (requires pro license with entanglement feature)
        network.entangle(["weather", "quantum_mood"])
        print("✅ Created quantum entanglement!")
        
        # Try quantum inference (requires pro license with quantum_inference feature)
        result = network.infer(["quantum_mood"], evidence={"weather": "sunny"})
        print("✅ Performed quantum inference!")
        print(f"📈 Results: {result}")
        
    except Exception as e:
        print(f"⚠️  Advanced quantum features not available: {e}")
        print("📧 Contact bajpaikrishna715@gmail.com to upgrade your license!")
    
    print("\n" + "="*80)
    print("🎯 Example completed successfully!")
    
except Exception as e:
    print(f"❌ License validation failed: {e}")
    print("\n" + "="*80)
    print("📋 To use Probabilistic Quantum Reasoner:")
    print("1. Contact bajpaikrishna715@gmail.com")
    print("2. Include your machine ID in the license request")
    print("3. Specify which license tier you need:")
    print("   • CORE: Basic networks and classical inference")
    print("   • PRO: Quantum operations, entanglement, causal reasoning")
    print("   • ENTERPRISE: Advanced backends, distributed inference")
    print("\n⚠️  No bypass or development mode available")
