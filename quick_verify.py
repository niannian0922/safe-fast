"""
Quick Installation Verification Script

Run this after fixing the installation to ensure everything is working.
This is a simplified version that focuses on the essential imports.

Usage:
    python quick_verify.py
"""

import sys

def test_imports():
    """Test importing essential packages."""
    print("🧪 Testing essential imports...")
    print("-" * 40)
    
    # Test results storage
    results = {}
    
    # Test JAX (core requirement)
    print("Testing JAX...", end=" ")
    try:
        import jax
        import jax.numpy as jnp
        # Quick functionality test
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.sum(x)
        results['jax'] = f"✅ JAX {jax.__version__} (devices: {len(jax.devices())})"
    except Exception as e:
        results['jax'] = f"❌ JAX failed: {str(e)[:50]}..."
    print(results['jax'])
    
    # Test Flax (neural networks)
    print("Testing Flax...", end=" ")
    try:
        import flax
        from flax import linen as nn
        results['flax'] = f"✅ Flax {flax.__version__}"
    except Exception as e:
        results['flax'] = f"❌ Flax failed: {str(e)[:50]}..."
    print(results['flax'])
    
    # Test Optax (optimizers)
    print("Testing Optax...", end=" ")
    try:
        import optax
        results['optax'] = f"✅ Optax {optax.__version__}"
    except Exception as e:
        results['optax'] = f"❌ Optax failed: {str(e)[:50]}..."
    print(results['optax'])
    
    # Test ML Collections (config)
    print("Testing ML Collections...", end=" ")
    try:
        import ml_collections
        config = ml_collections.ConfigDict()
        config.test_value = 42
        results['ml_collections'] = "✅ ML Collections working"
    except Exception as e:
        results['ml_collections'] = f"❌ ML Collections failed: {str(e)[:50]}..."
    print(results['ml_collections'])
    
    # Test NumPy (basic scientific computing)
    print("Testing NumPy...", end=" ")
    try:
        import numpy as np
        # Test basic functionality
        arr = np.array([1, 2, 3])
        result = np.sum(arr)
        results['numpy'] = f"✅ NumPy {np.__version__}"
    except Exception as e:
        results['numpy'] = f"❌ NumPy failed: {str(e)[:50]}..."
    print(results['numpy'])
    
    # Test optional packages
    optional_packages = [
        ('jraph', 'import jraph'),
        ('qpax', 'import qpax'),
        ('chex', 'import chex')
    ]
    
    print("\nTesting optional packages...")
    for name, import_cmd in optional_packages:
        print(f"Testing {name}...", end=" ")
        try:
            exec(import_cmd)
            results[name] = f"✅ {name} working"
        except Exception as e:
            results[name] = f"⚠️ {name} not available (optional)"
        print(results[name])
    
    return results


def test_jax_functionality():
    """Test basic JAX functionality including gradients."""
    print("\n🔬 Testing JAX functionality...")
    print("-" * 40)
    
    try:
        import jax
        import jax.numpy as jnp
        from jax import grad, jit
        
        # Test basic operations
        print("Testing basic operations...", end=" ")
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.sum(x**2)
        print(f"✅ Sum of squares: {y}")
        
        # Test gradients
        print("Testing gradients...", end=" ")
        def simple_function(x):
            return jnp.sum(x**2)
        
        grad_fn = grad(simple_function)
        gradient = grad_fn(x)
        print(f"✅ Gradient computed: {gradient}")
        
        # Test JIT compilation
        print("Testing JIT compilation...", end=" ")
        jit_fn = jit(simple_function)
        result = jit_fn(x)
        print(f"✅ JIT result: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ JAX functionality test failed: {e}")
        return False


def test_configuration_system():
    """Test our configuration system."""
    print("\n⚙️ Testing configuration system...")
    print("-" * 40)
    
    try:
        # Try to import our config (this tests the project structure)
        sys.path.append('.')
        from configs.default_config import get_minimal_config
        
        config = get_minimal_config()
        print(f"✅ Configuration loaded")
        print(f"   Physics timestep: {config.physics.dt}")
        print(f"   Number of agents: {config.env.num_agents}")
        return True
        
    except ImportError as e:
        print(f"⚠️ Config system not available: {e}")
        print("   This is expected if you haven't set up the full project yet")
        return False
    except Exception as e:
        print(f"❌ Config system failed: {e}")
        return False


def main():
    """Main verification function."""
    print("🔍 QUICK INSTALLATION VERIFICATION")
    print("=" * 50)
    print(f"Python version: {sys.version}")
    print()
    
    # Test imports
    import_results = test_imports()
    
    # Test JAX functionality
    jax_ok = test_jax_functionality()
    
    # Test configuration system
    config_ok = test_configuration_system()
    
    # Summary
    print("\n📊 VERIFICATION SUMMARY")
    print("=" * 50)
    
    essential_packages = ['jax', 'flax', 'optax', 'ml_collections', 'numpy']
    essential_working = sum(1 for pkg in essential_packages if '✅' in import_results.get(pkg, ''))
    
    print(f"Essential packages: {essential_working}/{len(essential_packages)} working")
    print(f"JAX functionality: {'✅ Working' if jax_ok else '❌ Failed'}")
    print(f"Configuration system: {'✅ Working' if config_ok else '⚠️ Not available'}")
    
    if essential_working == len(essential_packages) and jax_ok:
        print("\n🎉 VERIFICATION SUCCESSFUL!")
        print("You can now run the main validation script:")
        print("   python main.py")
        return True
    else:
        print("\n⚠️ VERIFICATION INCOMPLETE")
        print("Some essential components are not working properly.")
        print("Please check the installation guide for troubleshooting.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)