#!/usr/bin/env python3
"""
License Information and Activation Script

This script provides information about QuantumMeta licensing for the
Probabilistic Quantum Reasoner and helps users understand licensing requirements.

Contact: bajpaikrishna715@gmail.com for licensing information.
"""

import sys
import platform
import uuid
import socket
import hashlib

def generate_machine_id():
    """Generate machine ID for license binding."""
    # Collect machine-specific information
    info_components = [
        platform.node(),  # Computer name
        platform.machine(),  # Machine type
        platform.processor(),  # Processor info
    ]
    
    # Add MAC address if available
    try:
        mac = ':'.join(['{:02x}'.format((uuid.getnode() >> elements) & 0xff) 
                       for elements in range(0, 2*6, 2)][::-1])
        info_components.append(mac)
    except:
        pass
    
    # Add hostname
    try:
        hostname = socket.gethostname()
        info_components.append(hostname)
    except:
        pass
    
    # Create hash from collected information
    machine_info = '|'.join(filter(None, info_components))
    machine_hash = hashlib.sha256(machine_info.encode()).hexdigest()[:16]
    
    return machine_hash.upper()

def display_system_info():
    """Display system information for license purposes."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        PROBABILISTIC QUANTUM REASONER                        ║
║                           LICENSING INFORMATION                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║ This software requires a valid QuantumMeta license to operate.              ║
║                                                                              ║""")
    
    machine_id = generate_machine_id()
    print(f"║ 🔧 MACHINE ID: {machine_id:<52} ║")
    
    print(f"""║                                                                              ║
║ 📧 TO OBTAIN A LICENSE:                                                     ║
║    Contact: bajpaikrishna715@gmail.com                                      ║
║    Include your Machine ID in the license request                           ║
║                                                                              ║
║ 🔑 AVAILABLE LICENSE TIERS:                                                 ║
║                                                                              ║
║    💡 CORE LICENSE ($199/year)                                              ║
║       • Basic Bayesian networks                                             ║
║       • Classical probabilistic inference                                   ║
║       • Standard node types                                                 ║
║       • Documentation and support                                           ║
║                                                                              ║
║    ⚛️  PRO LICENSE ($799/year)                                               ║
║       • All Core features                                                   ║
║       • Quantum nodes and operations                                        ║
║       • Quantum entanglement                                                ║
║       • Quantum inference algorithms                                        ║
║       • Causal reasoning and interventions                                  ║
║       • Counterfactual analysis                                             ║
║       • Quantum backend support (Qiskit, PennyLane)                        ║
║                                                                              ║
║    🏢 ENTERPRISE LICENSE ($2999/year)                                       ║
║       • All Pro features                                                    ║
║       • Advanced quantum backends                                           ║
║       • Distributed inference                                               ║
║       • Custom operators and gates                                          ║
║       • Performance optimization tools                                      ║
║       • Priority support and consulting                                     ║
║       • Multi-user team licensing                                           ║
║                                                                              ║
║ 🎓 ACADEMIC DISCOUNTS:                                                      ║
║    • Students: 50% discount with valid student ID                           ║
║    • Academic institutions: 30% discount                                    ║
║    • Research projects: Custom pricing available                            ║
║                                                                              ║
║ ⚠️  IMPORTANT SECURITY FEATURES:                                            ║
║    • 24-hour grace period for new installations                             ║
║    • Machine ID binding for license security                                ║
║    • Encrypted license validation                                           ║
║    • Tamper-proof license enforcement                                       ║
║                                                                              ║
║ 📞 SUPPORT:                                                                 ║
║    • Email: bajpaikrishna715@gmail.com                                      ║
║    • Response time: 24-48 hours                                             ║
║    • License activation assistance                                          ║
║    • Technical support included                                             ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

def display_license_features():
    """Display detailed feature comparison."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                              FEATURE COMPARISON                             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ FEATURE                                    │  CORE  │  PRO   │ ENTERPRISE   ║
╠════════════════════════════════════════════╪════════╪════════╪══════════════╣
║ Basic Bayesian Networks                    │   ✅   │   ✅   │      ✅      ║
║ Classical Inference                        │   ✅   │   ✅   │      ✅      ║
║ Stochastic Nodes                          │   ✅   │   ✅   │      ✅      ║
║ Documentation & Support                    │   ✅   │   ✅   │      ✅      ║
║                                            │        │        │              ║
║ Quantum Nodes                              │   ❌   │   ✅   │      ✅      ║
║ Quantum Inference                          │   ❌   │   ✅   │      ✅      ║
║ Entanglement Operations                    │   ❌   │   ✅   │      ✅      ║
║ Causal Reasoning                           │   ❌   │   ✅   │      ✅      ║
║ Counterfactual Analysis                    │   ❌   │   ✅   │      ✅      ║
║ Quantum Backends (Qiskit/PennyLane)       │   ❌   │   ✅   │      ✅      ║
║                                            │        │        │              ║
║ Advanced Quantum Backends                  │   ❌   │   ❌   │      ✅      ║
║ Distributed Inference                      │   ❌   │   ❌   │      ✅      ║
║ Custom Operators                           │   ❌   │   ❌   │      ✅      ║
║ Performance Optimization                   │   ❌   │   ❌   │      ✅      ║
║ Multi-user Team Licensing                  │   ❌   │   ❌   │      ✅      ║
║ Priority Support                           │   ❌   │   ❌   │      ✅      ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

def check_license_status():
    """Check current license status."""
    print("\n🔍 Checking License Status...")
    
    try:
        # Try to import and check license
        from probabilistic_quantum_reasoner.licensing import get_license_manager, display_license_info
        
        manager = get_license_manager()
        licensed_features = manager.get_licensed_features()
        
        print("✅ Valid license found!")
        display_license_info()
        
    except ImportError:
        print("❌ Probabilistic Quantum Reasoner not installed")
        print("   Install with: pip install probabilistic-quantum-reasoner")
        
    except Exception as e:
        if "license" in str(e).lower():
            print("❌ No valid license found")
            print(f"   Error: {e}")
        else:
            print(f"❌ Error checking license: {e}")

def main():
    """Main license information script."""
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "info":
            display_system_info()
        elif command == "features":
            display_license_features()
        elif command == "status":
            check_license_status()
        elif command == "machine-id":
            machine_id = generate_machine_id()
            print(f"Machine ID: {machine_id}")
        else:
            print(f"Unknown command: {command}")
            print("Available commands: info, features, status, machine-id")
    else:
        display_system_info()
        display_license_features()
        check_license_status()

if __name__ == "__main__":
    main()
