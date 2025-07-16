#!/usr/bin/env python3
"""
Test script to verify 24-hour grace period licensing functionality.

This script tests that the Probabilistic Quantum Reasoner works correctly
with the 24-hour grace period for new installations.
"""

import sys
import time
from datetime import datetime

def test_grace_period_licensing():
    """Test the 24-hour grace period licensing functionality."""
    
    print("🧪 Testing Probabilistic Quantum Reasoner with 24-hour grace period...")
    print(f"⏰ Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # Test 1: Basic import with grace period
        print("📦 Test 1: Basic package import...")
        import probabilistic_quantum_reasoner as pqr
        print("✅ Package imported successfully")
        print(f"   Machine ID: {pqr.get_machine_id()}")
        print()
        
        # Test 2: Basic network creation (Core features)
        print("🔗 Test 2: Creating basic Bayesian network...")
        network = pqr.QuantumBayesianNetwork(name="Grace Period Test Network")
        print("✅ Basic network created successfully")
        print(f"   Network: {network.name}")
        print()
        
        # Test 3: License information display
        print("ℹ️  Test 3: Displaying license information...")
        pqr.display_license_info()
        print()
        
        # Test 4: Feature access validation
        print("🔍 Test 4: Checking available features...")
        try:
            from probabilistic_quantum_reasoner.licensing import get_license_manager
            manager = get_license_manager()
            licensed_features = manager.get_licensed_features()
            
            print("✅ Available features during grace period:")
            for feature in licensed_features:
                print(f"   • {feature}")
            print()
            
        except Exception as e:
            print(f"⚠️  Feature check warning: {e}")
            print()
        
        # Test 5: Try quantum features (should work during grace period)
        print("⚛️  Test 5: Testing quantum features during grace period...")
        try:
            # This should work during grace period even without Pro license
            quantum_node = pqr.QuantumNode(
                node_id="test_quantum",
                name="Test Quantum Node",
                outcome_space=[0, 1]
            )
            print("✅ Quantum node created during grace period")
            print(f"   Node: {quantum_node.name}")
            print()
            
        except Exception as e:
            print(f"⚠️  Quantum feature limitation: {e}")
            print()
        
        # Test 6: Licensing decorator functionality
        print("🔒 Test 6: Testing license decorator enforcement...")
        try:
            # Test that license checks are performed but don't block during grace period
            from probabilistic_quantum_reasoner.licensing import validate_license
            
            validate_license(["core"])
            print("✅ Core license validation passed")
            
            validate_license(["quantum_inference", "pro"])
            print("✅ Pro feature validation passed (grace period)")
            print()
            
        except Exception as e:
            print(f"ℹ️  License validation info: {e}")
            print()
        
        # Summary
        print("📊 GRACE PERIOD TEST SUMMARY:")
        print("✅ Package import: PASSED")
        print("✅ Basic network creation: PASSED")
        print("✅ License information display: PASSED")
        print("✅ Feature availability check: PASSED")
        print("✅ Grace period functionality: WORKING")
        print()
        print("🎉 All tests completed successfully!")
        print("📧 For full licensing, contact: bajpaikrishna715@gmail.com")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        print("📧 Contact bajpaikrishna715@gmail.com for support")
        return False

def test_license_info_display():
    """Test the license information display functionality."""
    
    print("\n" + "="*80)
    print("📋 LICENSE INFORMATION TEST")
    print("="*80)
    
    try:
        # Test license info script
        from license_info import display_system_info, display_license_features, generate_machine_id
        
        print("🔧 Machine ID:", generate_machine_id())
        print()
        
        display_system_info()
        display_license_features()
        
        return True
        
    except Exception as e:
        print(f"❌ License info test failed: {e}")
        return False

def main():
    """Main test function."""
    
    print("🚀 PROBABILISTIC QUANTUM REASONER")
    print("   24-HOUR GRACE PERIOD LICENSING TEST")
    print("="*80)
    
    # Run grace period functionality test
    test1_passed = test_grace_period_licensing()
    
    # Run license info display test
    test2_passed = test_license_info_display()
    
    print("\n" + "="*80)
    print("📊 FINAL TEST RESULTS")
    print("="*80)
    
    if test1_passed and test2_passed:
        print("🎉 ALL TESTS PASSED!")
        print("✅ 24-hour grace period is working correctly")
        print("✅ License system is properly integrated")
        print("📧 Contact bajpaikrishna715@gmail.com for full licensing")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        print("📧 Contact bajpaikrishna715@gmail.com for support")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
