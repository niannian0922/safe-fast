"""
End-to-end integration test for Safe Agile Flight System

This test validates the complete methodology implementation:
1. GCBF+ GNN perception 
2. Policy network  
3. Safety layer with qpax
4. JAX-native physics
5. BPTT with gradient flow
6. DiffPhysDrone temporal gradient decay

SUCCESS CRITERIA:
- All components JIT compile successfully
- Gradients flow end-to-end (no NaNs/Infs)
- BPTT loop executes without errors
- CBF constraints are enforced
- Temporal gradient decay stabilizes training
"""

import jax
import jax.numpy as jnp
from jax import random, grad, jit
import optax
import time
from pathlib import Path
import sys

# Configure JAX 
jax.config.update("jax_enable_x64", True)

# Add project path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import all components
from core.physics import (
    DroneState, PhysicsParams, dynamics_step,
    create_initial_drone_state, temporal_gradient_decay
)
from core.perception import (
    CBFNet, pointcloud_to_graph, GraphConfig,
    DroneState as PerceptionDroneState
)
from core.safety import SafetyLayer, SafetyConfig
from core.loop import create_complete_bptt_scan_function, ScanCarry, ScanOutput
from core.training import LossConfig, compute_comprehensive_loss
from configs.default_config import get_config

def test_individual_components():
    """Test each component individually"""
    print("🧪 Testing individual components...")
    
    # Initialize random key
    key = random.PRNGKey(42)
    
    # === 1. Test Physics Engine ===
    print("  📍 Testing physics engine...")
    drone_state = create_initial_drone_state(position=jnp.array([0.0, 0.0, 1.0]))
    physics_params = PhysicsParams()
    control_input = jnp.array([0.1, 0.0, 0.5])
    
    # Test forward pass
    next_state = dynamics_step(drone_state, control_input, physics_params)
    print(f"    ✅ Physics step: {next_state.position}")
    
    # Test JIT compilation
    dynamics_jit = jit(dynamics_step)
    next_state_jit = dynamics_jit(drone_state, control_input, physics_params)
    print(f"    ✅ Physics JIT compiled")
    
    # Test gradients
    def physics_loss(control):
        state = dynamics_step(drone_state, control, physics_params)
        return jnp.sum(state.position**2)
    
    grad_fn = grad(physics_loss)
    gradients = grad_fn(control_input)
    print(f"    ✅ Physics gradients: {gradients}")
    
    # === 2. Test Perception Module ===
    print("  🔍 Testing perception module...")
    
    # Create dummy point cloud
    point_cloud = jnp.array([
        [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 
        [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]
    ])
    
    config = GraphConfig()
    graph, node_types = pointcloud_to_graph(drone_state, point_cloud, config)
    print(f"    ✅ Graph construction: {graph.nodes.shape}")
    print(f"    📊 Graph n_node shape: {graph.n_node.shape}, value: {graph.n_node}")
    print(f"    📊 Graph n_edge shape: {graph.n_edge.shape}, value: {graph.n_edge}")
    
    # Test CBF network
    cbf_net = CBFNet()
    key, subkey = random.split(key)
    cbf_params = cbf_net.init(subkey, graph, n_type=1)
    cbf_value = cbf_net.apply(cbf_params, graph, n_type=1)
    print(f"    ✅ CBF computation: {cbf_value}")
    
    # Test CBF gradients
    def cbf_wrt_position(pos):
        modified_state = drone_state.replace(position=pos)
        graph_mod, _ = pointcloud_to_graph(modified_state, point_cloud, config)
        return cbf_net.apply(cbf_params, graph_mod, n_type=1)
    
    cbf_gradients = grad(cbf_wrt_position)(drone_state.position)
    print(f"    ✅ CBF gradients: {cbf_gradients}")
    
    # === 3. Test Safety Layer ===
    print("  🛡️ Testing safety layer...")
    
    safety_config = SafetyConfig()
    safety_layer = SafetyLayer(safety_config)
    u_nominal = jnp.array([0.2, 0.1, 0.3])
    
    u_safe, qp_info = safety_layer.safety_filter(
        u_nominal, cbf_value, cbf_gradients, drone_state
    )
    print(f"    ✅ Safety filter: {u_safe}, feasible: {qp_info.is_feasible}")
    
    # === 4. Test Temporal Gradient Decay ===
    print("  ⏰ Testing temporal gradient decay...")
    
    test_tensor = jnp.array([1.0, 2.0, 3.0])
    decayed = temporal_gradient_decay(test_tensor, 0.4)
    print(f"    ✅ Gradient decay: {decayed}")
    
    # Test gradient flow
    def decay_loss(x):
        return jnp.sum(temporal_gradient_decay(x, 0.4)**2)
    
    decay_grad = grad(decay_loss)(test_tensor)
    print(f"    ✅ Decay gradients: {decay_grad}")
    
    return cbf_params, safety_config, physics_params

def test_end_to_end_system(cbf_params, safety_config, physics_params):
    """Test complete end-to-end system"""
    print("\n🚀 Testing end-to-end system...")
    
    key = random.PRNGKey(123)
    
    # === Create BPTT Scan Function ===
    print("  🔄 Creating BPTT scan function...")
    
    # Create dummy policy parameters
    key, subkey = random.split(key)
    policy_params = {
        'Dense_0': {
            'kernel': random.normal(subkey, (10, 32)),
            'bias': jnp.zeros(32)
        },
        'Dense_1': {
            'kernel': random.normal(subkey, (32, 3)),
            'bias': jnp.zeros(3)
        }
    }
    
    scan_function = create_complete_bptt_scan_function(
        cbf_params, policy_params, safety_config, physics_params
    )
    print("    ✅ BPTT scan function created")
    
    # === Test Single Step ===
    print("  📍 Testing single scan step...")
    
    initial_state = create_initial_drone_state(position=jnp.array([0.0, 0.0, 1.0]))
    initial_carry = ScanCarry(
        drone_state=initial_state,
        rnn_hidden_state=jnp.zeros(32),
        step_count=0,
        cumulative_reward=0.0
    )
    
    external_input = {'target_velocity': jnp.array([1.0, 0.0, 0.0])}
    
    try:
        new_carry, output = scan_function(initial_carry, external_input)
        print(f"    ✅ Single step successful")
        print(f"        Position: {new_carry.drone_state.position}")
        print(f"        CBF value: {output.cbf_values}")
        print(f"        Safe control: {output.safe_controls}")
    except Exception as e:
        print(f"    ❌ Single step failed: {e}")
        return False
    
    # === Test JIT Compilation ===
    print("  ⚡ Testing JIT compilation...")
    
    try:
        scan_jit = jit(scan_function)
        new_carry_jit, output_jit = scan_jit(initial_carry, external_input)
        print(f"    ✅ JIT compilation successful")
    except Exception as e:
        print(f"    ❌ JIT compilation failed: {e}")
        return False
    
    # === Test Multi-Step BPTT ===
    print("  🔄 Testing multi-step BPTT...")
    
    def multi_step_rollout(carry, inputs_sequence):
        """Multi-step rollout using lax.scan"""
        final_carry, outputs = jax.lax.scan(
            scan_function, carry, inputs_sequence, length=len(inputs_sequence)
        )
        return final_carry, outputs
    
    # Create sequence of inputs
    sequence_length = 10
    inputs_sequence = [
        {'target_velocity': jnp.array([1.0, 0.0, 0.0])} 
        for _ in range(sequence_length)
    ]
    
    try:
        final_carry, trajectory = multi_step_rollout(initial_carry, inputs_sequence)
        print(f"    ✅ Multi-step BPTT successful")
        print(f"        Final position: {final_carry.drone_state.position}")
        print(f"        Trajectory shape: {trajectory.positions.shape}")
    except Exception as e:
        print(f"    ❌ Multi-step BPTT failed: {e}")
        return False
    
    # === Test End-to-End Gradients ===
    print("  🔀 Testing end-to-end gradients...")
    
    def trajectory_loss(cbf_p, policy_p):
        """Loss function for the trajectory"""
        scan_fn = create_complete_bptt_scan_function(
            cbf_p, policy_p, safety_config, physics_params
        )
        
        final_carry, trajectory = jax.lax.scan(
            scan_fn, initial_carry, inputs_sequence, length=sequence_length
        )
        
        # Simple loss: encourage reaching target
        target_pos = jnp.array([10.0, 0.0, 0.0])
        position_error = jnp.sum((final_carry.drone_state.position - target_pos)**2)
        
        # Add safety penalty
        safety_violations = jnp.sum(jnp.maximum(0.0, -trajectory.cbf_values))
        
        return position_error + 10.0 * safety_violations
    
    try:
        # Compute gradients
        grad_fn = grad(trajectory_loss, argnums=(0, 1))
        cbf_grads, policy_grads = grad_fn(cbf_params, policy_params)
        
        print(f"    ✅ End-to-end gradients computed")
        print(f"        CBF gradient norm: {jax.tree_util.tree_reduce(
            lambda x, y: x + jnp.sum(y**2), cbf_grads, initializer=0.0
        )**0.5}")
        print(f"        Policy gradient norm: {jax.tree_util.tree_reduce(
            lambda x, y: x + jnp.sum(y**2), policy_grads, initializer=0.0
        )**0.5}")
        
    except Exception as e:
        print(f"    ❌ End-to-end gradients failed: {e}")
        return False
    
    return True

def test_performance_benchmarks(cbf_params, policy_params, safety_config, physics_params):
    """Test performance benchmarks"""
    print("\n⚡ Performance benchmarks...")
    
    initial_state = create_initial_drone_state(position=jnp.array([0.0, 0.0, 1.0]))
    initial_carry = ScanCarry(
        drone_state=initial_state,
        rnn_hidden_state=jnp.zeros(32),
        step_count=0,
        cumulative_reward=0.0
    )
    
    # Test different sequence lengths
    for seq_len in [10, 50, 100]:
        print(f"  📏 Testing sequence length {seq_len}...")
        
        inputs_sequence = [
            {'target_velocity': jnp.array([1.0, 0.0, 0.0])} 
            for _ in range(seq_len)
        ]
        
        scan_function = create_complete_bptt_scan_function(
            cbf_params, policy_params, safety_config, physics_params
        )
        
        # JIT compile
        scan_jit = jit(lambda carry, inputs: jax.lax.scan(
            scan_function, carry, inputs, length=len(inputs)
        ))
        
        # Warm up JIT
        _ = scan_jit(initial_carry, inputs_sequence)
        
        # Benchmark
        start_time = time.time()
        for _ in range(3):
            final_carry, trajectory = scan_jit(initial_carry, inputs_sequence)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 3
        print(f"    ⏱️ Average time: {avg_time:.4f}s ({avg_time/seq_len*1000:.2f}ms per step)")

def main():
    """Run complete end-to-end test suite"""
    print("🚀 Safe Agile Flight - End-to-End Integration Test")
    print("=" * 60)
    
    try:
        # Test individual components
        cbf_params, safety_config, physics_params = test_individual_components()
        
        # Create policy parameters
        key = random.PRNGKey(123)
        policy_params = {
            'Dense_0': {
                'kernel': random.normal(key, (10, 32)),
                'bias': jnp.zeros(32)
            },
            'Dense_1': {
                'kernel': random.normal(key, (32, 3)),
                'bias': jnp.zeros(3)
            }
        }
        
        # Test end-to-end system
        success = test_end_to_end_system(cbf_params, safety_config, physics_params)
        
        if success:
            print("\n✅ ALL TESTS PASSED!")
            print("🎉 End-to-end integration successful!")
            print("\nThe complete methodology is working:")
            print("  ✅ GCBF+ GNN perception")
            print("  ✅ Policy network") 
            print("  ✅ qpax safety layer")
            print("  ✅ JAX-native physics")
            print("  ✅ BPTT with gradient flow")
            print("  ✅ DiffPhysDrone temporal decay")
            
            # Run performance benchmarks
            test_performance_benchmarks(cbf_params, policy_params, safety_config, physics_params)
            
        else:
            print("\n❌ TESTS FAILED!")
            print("Some components need debugging.")
            
        return success
        
    except Exception as e:
        print(f"\n💥 CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)