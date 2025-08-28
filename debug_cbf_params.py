#!/usr/bin/env python3
"""
Debug CBF parameters structure
"""

import jax
import jax.numpy as jnp
from jax import random

# Import components
from core.physics import create_initial_drone_state
from core.perception import CBFNet, pointcloud_to_graph, GraphConfig

def debug_cbf_params():
    """Debug the CBF parameters structure"""
    print("🔍 Debug: CBF parameters structure...")
    
    # Initialize random key
    key = random.PRNGKey(42)
    
    # Create sample data
    drone_state = create_initial_drone_state(position=jnp.array([0.0, 0.0, 1.0]))
    point_cloud = jnp.array([
        [2.0, 1.0, 0.5], [1.5, -1.0, 0.8], 
        [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 
        [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]
    ])
    config = GraphConfig()
    
    # Create graph
    graph, node_types = pointcloud_to_graph(drone_state, point_cloud, config)
    print(f"Graph nodes shape: {graph.nodes.shape}")
    
    # Create CBF network
    cbf_net = CBFNet()
    
    # Initialize parameters
    key, subkey = random.split(key)
    cbf_params = cbf_net.init(subkey, graph, n_type=1)
    
    print("CBF parameters structure:")
    import pprint
    pprint.pprint(cbf_params, depth=5)
    
    # Test forward pass
    cbf_value = cbf_net.apply(cbf_params, graph, n_type=1)
    print(f"CBF value: {cbf_value}")
    
    return cbf_params

if __name__ == "__main__":
    debug_cbf_params()