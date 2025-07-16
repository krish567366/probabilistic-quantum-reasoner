#!/usr/bin/env python3
"""
License Enforcement Test

This script tests the comprehensive license enforcement in the
Probabilistic Quantum Reasoner to ensure no bypass mechanisms exist.
"""

import sys
import traceback
from typing import List, Dict

def test_license_enforcement():
    """Test comprehensive license enforcement."""
    
    print("🔒 Testing QuantumMeta License Enforcement")
    print("="*80)
    
    test_results = []
    
    # Test 1: Package Import License Check
    print("\n📦 Test 1: Package Import License Validation")
    try:
        import probabilistic_quantum_reasoner
        test_results.append(("Package Import", "FAILED - Should require license"))
        print("❌ Package imported without license validation!")
    except Exception as e:
        if "license" in str(e).lower():
            test_results.append(("Package Import", "PASSED - License required"))
            print("✅ Package correctly requires license on import")
        else:
            test_results.append(("Package Import", f"FAILED - Unexpected error: {e}"))
            print(f"❌ Unexpected error: {e}")
    
    # Test 2: Core Network Creation
    print("\n🌐 Test 2: Core Network Creation License")
    try:
        from probabilistic_quantum_reasoner import QuantumBayesianNetwork
        network = QuantumBayesianNetwork()
        test_results.append(("Network Creation", "FAILED - Should require license"))
        print("❌ Network created without license validation!")
    except Exception as e:
        if "license" in str(e).lower():
            test_results.append(("Network Creation", "PASSED - License required"))
            print("✅ Network creation correctly requires license")
        else:
            test_results.append(("Network Creation", f"FAILED - Unexpected error: {e}"))
            print(f"❌ Unexpected error: {e}")
    
    # Test 3: Quantum Node Creation
    print("\n⚛️  Test 3: Quantum Node Creation License")
    try:
        from probabilistic_quantum_reasoner import QuantumNode
        node = QuantumNode("test", "test", ["0", "1"])
        test_results.append(("Quantum Node", "FAILED - Should require pro license"))
        print("❌ Quantum node created without pro license validation!")
    except Exception as e:
        if "license" in str(e).lower() or "feature" in str(e).lower():
            test_results.append(("Quantum Node", "PASSED - License required"))
            print("✅ Quantum node correctly requires pro license")
        else:
            test_results.append(("Quantum Node", f"FAILED - Unexpected error: {e}"))
            print(f"❌ Unexpected error: {e}")
    
    # Test 4: Backend Access
    print("\n🔧 Test 4: Quantum Backend Access License")
    try:
        from probabilistic_quantum_reasoner import QiskitBackend
        backend = QiskitBackend()
        test_results.append(("Quantum Backend", "FAILED - Should require pro license"))
        print("❌ Quantum backend created without pro license validation!")
    except Exception as e:
        if "license" in str(e).lower() or "feature" in str(e).lower():
            test_results.append(("Quantum Backend", "PASSED - License required"))
            print("✅ Quantum backend correctly requires pro license")
        else:
            test_results.append(("Quantum Backend", f"FAILED - Unexpected error: {e}"))
            print(f"❌ Unexpected error: {e}")
    
    # Test 5: Causal Inference
    print("\n🔗 Test 5: Causal Inference License")
    try:
        from probabilistic_quantum_reasoner.inference import QuantumCausalInference
        causal = QuantumCausalInference(None)
        test_results.append(("Causal Inference", "FAILED - Should require pro license"))
        print("❌ Causal inference created without pro license validation!")
    except Exception as e:
        if "license" in str(e).lower() or "feature" in str(e).lower():
            test_results.append(("Causal Inference", "PASSED - License required"))
            print("✅ Causal inference correctly requires pro license")
        else:
            test_results.append(("Causal Inference", f"FAILED - Unexpected error: {e}"))
            print(f"❌ Unexpected error: {e}")
    
    # Test 6: Development Mode Bypass Check
    print("\n🚧 Test 6: Development Mode Bypass Check")
    import os
    
    # Try various development mode environment variables
    dev_vars = [
        "QUANTUMMETA_DEV",
        "QUANTUM_DEV_MODE",
        "PQR_DEV_MODE",
        "DEBUG",
        "DEVELOPMENT",
        "BYPASS_LICENSE"
    ]
    
    bypass_attempts = []
    for var in dev_vars:
        original_value = os.environ.get(var)
        try:
            # Set development variable
            os.environ[var] = "1"
            
            # Try to import and use
            import importlib
            if 'probabilistic_quantum_reasoner' in sys.modules:
                del sys.modules['probabilistic_quantum_reasoner']
            
            try:
                import probabilistic_quantum_reasoner
                from probabilistic_quantum_reasoner import QuantumBayesianNetwork
                network = QuantumBayesianNetwork()
                bypass_attempts.append(f"FAILED - {var} bypassed license")
            except Exception:
                bypass_attempts.append(f"PASSED - {var} did not bypass license")
        
        except Exception as e:
            bypass_attempts.append(f"ERROR - {var}: {e}")
        
        finally:
            # Restore original value
            if original_value is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = original_value
    
    bypass_test_passed = all("PASSED" in attempt for attempt in bypass_attempts)
    test_results.append(("Development Bypass", "PASSED" if bypass_test_passed else "FAILED"))
    
    for attempt in bypass_attempts:
        print(f"   {attempt}")
    
    # Test 7: Machine ID Generation
    print("\n🔧 Test 7: Machine ID Generation")
    try:
        from probabilistic_quantum_reasoner.licensing import get_machine_id
        machine_id = get_machine_id()
        if machine_id and len(machine_id) > 8:
            test_results.append(("Machine ID", "PASSED - Valid machine ID"))
            print(f"✅ Machine ID generated: {machine_id}")
        else:
            test_results.append(("Machine ID", "FAILED - Invalid machine ID"))
            print("❌ Invalid machine ID generated")
    except Exception as e:
        test_results.append(("Machine ID", f"FAILED - {e}"))
        print(f"❌ Machine ID generation failed: {e}")
    
    # Test Summary
    print("\n" + "="*80)
    print("📊 LICENSE ENFORCEMENT TEST SUMMARY")
    print("="*80)
    
    passed_tests = 0
    for test_name, result in test_results:
        status = "✅ PASSED" if "PASSED" in result else "❌ FAILED"
        print(f"{test_name:<25} {status}")
        if "PASSED" in result:
            passed_tests += 1
    
    print(f"\n📈 Results: {passed_tests}/{len(test_results)} tests passed")
    
    if passed_tests == len(test_results):
        print("🔒 LICENSE ENFORCEMENT: SECURE ✅")
        print("   No bypass mechanisms detected")
        print("   All operations correctly require valid licenses")
    else:
        print("⚠️  LICENSE ENFORCEMENT: VULNERABILITIES DETECTED ❌")
        print("   Some operations may not properly enforce licensing")
    
    print("\n📧 For licensing: bajpaikrishna715@gmail.com")
    print("🔧 Machine ID:", end=" ")
    try:
        from probabilistic_quantum_reasoner.licensing import get_machine_id
        print(get_machine_id())
    except:
        print("Unable to retrieve")


if __name__ == "__main__":
    test_license_enforcement()
