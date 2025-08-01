#!/usr/bin/env python3
"""Test script to verify NumPy 2.2.4 compatibility with the minimal dopamine imports."""

def test_numpy_compatibility():
    """Test that all minimal imports work with NumPy 2.2+."""
    try:
        import numpy as np
        print(f"NumPy version: {np.__version__}")
        
        # Test numpy version compatibility
        from packaging import version
        if version.parse(np.__version__) >= version.parse("2.2.0"):
            print("✅ NumPy 2.2+ detected")
        else:
            print(f"⚠️ NumPy version {np.__version__} is older than 2.2.0")
        
        # Test core imports
        print("\nTesting core imports...")
        from dopamine.jax.agents.edqn.edqn_agent import JaxEnhancedDQNAgent, train
        print("  ✅ EDQN agent import successful")
        
        from dopamine.metrics import statistics_instance
        print("  ✅ Statistics instance import successful")
        
        import dopamine.jax.agents.edqn.edqn_agent
        print("  ✅ EDQN module import successful")
        
        import dopamine.jax.agents.dqn.dqn_agent
        print("  ✅ DQN module import successful")
        
        import dopamine.jax.replay_memory.replay_buffer
        print("  ✅ Replay buffer import successful")
        
        # Test key dependencies
        print("\nTesting key dependencies...")
        import jax
        print(f"  ✅ JAX {jax.__version__}")
        
        import flax
        print(f"  ✅ Flax {flax.__version__}")
        
        print(f"\n🎉 All tests passed! Ready to use with NumPy {np.__version__}")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Run 'pip install -r requirements.txt' to install dependencies")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    test_numpy_compatibility()