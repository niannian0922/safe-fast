#!/usr/bin/env python3
"""
Debug script for single step execution
"""

import jax
import jax.numpy as jnp
from jax import random, jit
import traceback

# Set JAX to show full tracebacks
import os
os.environ['JAX_TRACEBACK_FILTERING'] = 'off'

# Import components
from core.physics import create_initial_drone_state, PhysicsParams
from core.perception import GraphConfig
from core.safety import SafetyConfig
from core.loop import create_complete_bptt_scan_function, ScanCarry

def debug_single_step():
    """Debug the single step execution"""
    print("🔍 Debug: Single step execution...")
    
    try:
        # Initialize parameters
        key = random.PRNGKey(42)
        
        # Physics parameters
        physics_params = PhysicsParams()
        
        # Safety configuration
        safety_config = SafetyConfig()
        
        # Create actual CBF parameters by initializing the network
        from core.perception import CBFNet, pointcloud_to_graph, GraphConfig
        
        # Create sample data for initialization
        drone_state = create_initial_drone_state(position=jnp.array([0.0, 0.0, 1.0]))
        point_cloud = jnp.array([
            [2.0, 1.0, 0.5], [1.5, -1.0, 0.8], 
            [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 
            [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]
        ])
        config = GraphConfig()
        graph, node_types = pointcloud_to_graph(drone_state, point_cloud, config)
        
        # Initialize CBF network with proper parameters
        cbf_net = CBFNet()
        key, subkey = random.split(key)
        cbf_params = cbf_net.init(subkey, graph, n_type=1)
        print("    ✅ CBF parameters properly initialized")
        
        # Create actual policy parameters by initializing the network
        from core.policy import PolicyNetworkRNN, PolicyParams
        
        policy_config = PolicyParams(
            hidden_dims=(32, 32),
            use_rnn=True,
            rnn_hidden_size=16
        )
        policy_net = PolicyNetworkRNN(params=policy_config)
        
        # Create sample observation for initialization (RNN expects batch dimension)
        sample_obs = jnp.zeros((1, 10))  # [batch_size, obs_dim]
        sample_rnn_state = jnp.zeros(16)
        
        key, subkey = random.split(key)
        policy_params = policy_net.init(subkey, sample_obs, sample_rnn_state)
        print("    ✅ Policy parameters properly initialized")
        
        print("    ✅ Parameters created")
        
        # Create scan function
        scan_function = create_complete_bptt_scan_function(
            cbf_params, policy_params, safety_config, physics_params
        )
        print("    ✅ Scan function created")
        
        # Create initial state
        initial_state = create_initial_drone_state(position=jnp.array([0.0, 0.0, 1.0]))
        initial_carry = ScanCarry(
            drone_state=initial_state,
            rnn_hidden_state=jnp.zeros(16),  # Match RNN hidden size
            step_count=0,
            cumulative_reward=0.0
        )
        print("    ✅ Initial carry created")
        
        # External input
        external_input = {'target_velocity': jnp.array([1.0, 0.0, 0.0])}
        print("    ✅ External input created")
        
        # Execute single step
        print("    🚀 Executing single step...")
        new_carry, output = scan_function(initial_carry, external_input)
        print("    ✅ Single step completed successfully!")
        print(f"        New position: {new_carry.drone_state.position}")
        
    except Exception as e:
        print(f"    ❌ Single step failed: {e}")
        print("Full traceback:")
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    debug_single_step()