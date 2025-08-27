"""
Perception Module for Safe Agile Flight System

This module integrates GCBF+ GNN architecture with LiDAR point cloud processing
for single-agent CBF-based safe control.

Key Components:
1. Point cloud to graph conversion (pointcloud_to_graph)
2. GNN-based CBF value and gradient computation
3. JAX native implementation with JIT compilation support

Integration from GCBF+ codebase:
- GNN architecture from gcbfplus/nn/gnn.py
- Graph construction from gcbfplus/utils/graph.py
- CBF network from gcbfplus/algo/module/cbf.py

Adaptations:
- Multi-agent neighbor discovery -> single-agent LiDAR processing
- cvxpylayers -> qpax (handled in safety.py)
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import struct
import jraph
import functools as ft
from typing import Tuple, Callable, Optional, NamedTuple
from dataclasses import dataclass

# JAX utilities
from jax import vmap, jit
from jax.lax import top_k

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@struct.dataclass
class DroneState:
    """Drone state representation"""
    position: jnp.ndarray  # (3,) - World position [x, y, z]
    velocity: jnp.ndarray  # (3,) - World velocity [vx, vy, vz]
    orientation: jnp.ndarray  # (3, 3) - Rotation matrix R_world_to_body
    angular_velocity: jnp.ndarray  # (3,) - Angular velocity [wx, wy, wz]

@dataclass(frozen=True)
class GraphConfig:
    """Graph construction parameters"""
    k_neighbors: int = 8  # KNN neighbor count
    max_range: float = 5.0  # Maximum sensing range
    min_points: int = 10  # Minimum point cloud size
    max_points: int = 1000  # Maximum point cloud size (for memory control)
    ego_node_features: int = 10  # Ego node feature dimension
    obstacle_node_features: int = 3  # Obstacle node feature dimension
    edge_features: int = 4  # Edge feature dimension

# =============================================================================
# POINT CLOUD TO GRAPH CONVERSION
# =============================================================================

def compute_pairwise_distances(points1: jnp.ndarray, points2: jnp.ndarray) -> jnp.ndarray:
    """
    Compute pairwise distances between two point sets
    JAX-native implementation with vmap support
    
    Args:
        points1: (N, 3) - First point set
        points2: (M, 3) - Second point set
    Returns:
        distances: (N, M) - Pairwise distance matrix
    """
    # Vectorized distance computation: ||p1 - p2||_2
    diff = points1[:, None, :] - points2[None, :, :]  # (N, M, 3)
    distances = jnp.linalg.norm(diff, axis=2)  # (N, M)
    return distances

def filter_points_by_range(points: jnp.ndarray, drone_pos: jnp.ndarray, max_range: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Filter point cloud by distance range from drone
    """
    distances = jnp.linalg.norm(points - drone_pos, axis=1)
    mask = distances <= max_range
    # Use jnp.where to avoid boolean indexing issues
    valid_indices = jnp.where(mask, size=points.shape[0], fill_value=0)[0]
    valid_points = points[valid_indices]
    return valid_points, mask

def find_knn_edges(drone_pos: jnp.ndarray, obstacle_points: jnp.ndarray, k: int) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Build KNN edges for ego-centric graph construction
    JAX/JIT compatible version with fixed-size outputs
    
    Args:
        drone_pos: (3,) - Drone position
        obstacle_points: (N, 3) - Obstacle point cloud
        k: KNN neighbor count (must be static for JIT compilation)
        
    Returns:
        senders: (fixed_size,) - Sender node indices (padded with -1 for invalid)
        receivers: (fixed_size,) - Receiver node indices (padded with -1 for invalid) 
        edge_features: (fixed_size, 4) - Edge features (padded with zeros for invalid)
    """
    n_obstacles = obstacle_points.shape[0]
    n_total = n_obstacles + 1  # +1 for ego drone node
    
    # Combined position array: [drone, obstacle1, obstacle2, ...]
    all_positions = jnp.concatenate([drone_pos[None, :], obstacle_points], axis=0)
    
    # Compute all pairwise distances
    distances = compute_pairwise_distances(all_positions, all_positions)
    
    # Mask diagonal to exclude self-connections
    distances_masked = jnp.where(jnp.eye(n_total), jnp.inf, distances)
    
    # Use static k value
    k_use = min(k, n_total - 1)  # Static computation
    
    # Get k nearest neighbors for each node
    _, top_k_indices = jax.vmap(lambda row: jax.lax.top_k(-row, k_use))(distances_masked)
    
    # Create fixed-size edge arrays 
    max_edges = n_total * k_use
    
    # Create all potential edges (dense format)
    all_senders = jnp.repeat(jnp.arange(n_total), k_use)
    all_receivers = top_k_indices.flatten()
    
    # Compute all edge features
    sender_positions = all_positions[all_senders]
    receiver_positions = all_positions[all_receivers]
    rel_positions = receiver_positions - sender_positions
    edge_distances = jnp.linalg.norm(rel_positions, axis=1, keepdims=True)
    all_edge_features = jnp.concatenate([edge_distances, rel_positions], axis=1)
    
    # Create validity mask (but don't use for indexing)
    # Edge is valid if sender != receiver
    validity_mask = (all_senders != all_receivers) & (all_receivers < n_total)
    
    # Instead of dynamic filtering, use fixed-size arrays and mark invalid edges
    # Invalid edges will have sender/receiver = -1 and zero features
    final_senders = jnp.where(validity_mask, all_senders, -1)
    final_receivers = jnp.where(validity_mask, all_receivers, -1)
    final_features = jnp.where(validity_mask[:, None], all_edge_features, 0.0)
    
    return final_senders, final_receivers, final_features

def pointcloud_to_graph(drone_state: DroneState, point_cloud: jnp.ndarray, config: GraphConfig) -> Tuple[jraph.GraphsTuple, jnp.ndarray]:
    """
    Convert LiDAR point cloud to jraph.GraphsTuple for GNN processing
    
    This is the core adaptation from GCBF+'s multi-agent neighbor discovery
    to single-agent LiDAR processing (replacing multi-agent environment).
    
    Args:
        drone_state: Current drone state
        point_cloud: (N, 3) LiDAR point cloud in drone body frame
        config: Graph construction configuration
    
    Returns:
        graph: jraph.GraphsTuple - Graph structure for GNN
        node_types: Node type array for processing
        
    Graph structure:
        - Node 0: Ego drone node with state features
        - Nodes 1~N: Obstacle nodes (LiDAR points) with position features
        - Edges: Spatial connectivity based on KNN
    """
    # 1. Transform points to world frame
    # point_cloud is in body frame, transform to world frame
    world_points = (drone_state.orientation.T @ point_cloud.T).T + drone_state.position
    
    # 2. Filter points by range (simplified approach)
    distances = jnp.linalg.norm(world_points - drone_state.position, axis=1)
    valid_mask = distances <= config.max_range
    
    # Use padding approach instead of filtering to avoid dynamic shapes
    n_total_points = world_points.shape[0]
    valid_points = world_points  # Keep all points for now
    n_valid = n_total_points
    if n_valid < config.min_points:
        # Pad with dummy points if too few
        dummy_points = jnp.zeros((config.min_points, 3))
        valid_points = dummy_points
        n_valid = config.min_points
    elif n_valid > config.max_points:
        # Subsample if too many points
        key = jax.random.PRNGKey(0)  # Use proper JAX random API
        indices = jax.random.choice(
            key, n_valid, (config.max_points,), replace=False
        )
        valid_points = valid_points[indices]
        n_valid = config.max_points
    
    # 3. Construct node features
    # Ego node features: [position(3), velocity(3), angular_velocity(3), forward_direction(3)]
    ego_features = jnp.concatenate([
        drone_state.position,
        drone_state.velocity, 
        drone_state.angular_velocity,
        drone_state.orientation[:, 0]  # Forward direction vector
    ])  # (12,) -> truncate to ego_node_features dimension
    ego_features = ego_features[:config.ego_node_features]
    
    # Obstacle node features: relative position in drone body frame
    obstacle_features = (drone_state.orientation @ (valid_points - drone_state.position).T).T
    obstacle_features = obstacle_features[:, :config.obstacle_node_features]  # Keep only x,y,z
    
    # Combine all node features (pad ego features to match obstacle feature dimension)
    ego_features_padded = jnp.pad(ego_features, (0, max(0, config.obstacle_node_features - len(ego_features))))[:config.obstacle_node_features]
    all_node_features = jnp.concatenate([
        ego_features_padded[None, :],  # (1, obstacle_node_features)
        obstacle_features  # (n_valid, obstacle_node_features)
    ], axis=0)
    
    # 5. Build edges with fixed-size arrays
    senders, receivers, edge_features = find_knn_edges(
        drone_state.position, valid_points, config.k_neighbors
    )
    
    # Filter out invalid edges (marked with -1) without dynamic slicing
    valid_edge_mask = (senders >= 0) & (receivers >= 0)
    
    # Count valid edges
    n_valid_edges = jnp.sum(valid_edge_mask.astype(jnp.int32))
    
    # For GraphsTuple, we'll use the full arrays but jraph can handle -1 indices
    # The GNN will ignore edges with negative indices
    
    # 6. Node type classification
    # 0 = ego drone, 1 = obstacle
    node_types = jnp.concatenate([
        jnp.array([0]),  # ego node
        jnp.ones(n_valid)  # obstacle nodes
    ]).astype(jnp.int32)
    
    # 7. Create GraphsTuple - use full arrays, jraph handles invalid edges gracefully
    n_nodes = n_valid + 1
    n_total_edges = len(senders)  # Include both valid and invalid edges
    
    graph = jraph.GraphsTuple(
        n_node=jnp.array([n_nodes]),  # Single graph
        n_edge=jnp.array([n_total_edges]),  # Total edges (including invalid)
        nodes=all_node_features,  # (n_nodes, node_features)
        edges=edge_features,  # (n_total_edges, edge_features) 
        globals=None,
        senders=senders,  # (n_total_edges,) - includes -1 for invalid
        receivers=receivers,  # (n_total_edges,) - includes -1 for invalid
    )
    
    return graph, node_types

# =============================================================================  
# GNN ARCHITECTURE (Based on GCBF+)
# =============================================================================

def default_nn_init():
    """Default network initialization following GCBF+ convention"""
    return nn.initializers.xavier_uniform()

class MLP(nn.Module):
    """Multi-layer perceptron from GCBF+ codebase"""
    hid_sizes: Tuple[int, ...]
    act: Callable = nn.relu
    act_final: bool = False
    
    @nn.compact
    def __call__(self, x):
        for i, size in enumerate(self.hid_sizes):
            x = nn.Dense(size, kernel_init=default_nn_init())(x)
            if i < len(self.hid_sizes) - 1 or self.act_final:
                x = self.act(x)
        return x

class GNNLayer(nn.Module):
    """
    Single GNN layer adapted from GCBF+
    Implements message passing with attention-based aggregation
    """
    msg_dim: int
    out_dim: int
    hid_size_msg: Tuple[int, ...] = (256, 256)
    hid_size_aggr: Tuple[int, ...] = (128, 128)  
    hid_size_update: Tuple[int, ...] = (256, 256)
    
    @nn.compact
    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        def message_fn(edge_feats, sender_feats, receiver_feats):
            """Message function: combines edge + sender + receiver features"""
            # Handle invalid edges (sender or receiver = -1) by masking
            feats = jnp.concatenate([edge_feats, sender_feats, receiver_feats], axis=-1)
            feats = MLP(self.hid_size_msg, act=nn.relu, act_final=False)(feats)
            feats = nn.Dense(self.msg_dim, kernel_init=default_nn_init())(feats)
            return feats
        
        def update_fn(node_feats, aggr_msgs):
            """Node update function: combines node features + aggregated messages"""  
            feats = jnp.concatenate([node_feats, aggr_msgs], axis=-1)
            feats = MLP(self.hid_size_update, act=nn.relu, act_final=False)(feats)
            feats = nn.Dense(self.out_dim, kernel_init=default_nn_init())(feats)
            return feats
            
        def aggregate_fn(msgs, segment_ids, num_segments):
            """Attention-based message aggregation (from GCBF+ implementation)"""
            gate_feats = MLP(self.hid_size_aggr, act=nn.relu, act_final=False)(msgs)
            gate_feats = nn.Dense(1, kernel_init=default_nn_init())(gate_feats).squeeze(-1)
            
            # Softmax attention
            attn = jraph.segment_softmax(gate_feats, segment_ids=segment_ids, num_segments=num_segments)
            
            # Weighted aggregation
            aggr_msg = jraph.segment_sum(
                attn[:, None] * msgs, segment_ids=segment_ids, num_segments=num_segments
            )
            return aggr_msg
        
        # Apply message passing using jraph primitives
        updated_nodes = jraph.GraphNetwork(
            update_node_fn=lambda nodes, sent_msgs, received_msgs, globals_: update_fn(nodes, received_msgs),
            update_edge_fn=None,  # Keep edges unchanged
            update_global_fn=None,  # No global features
            aggregate_edges_for_nodes_fn=aggregate_fn,
            aggregate_nodes_for_globals_fn=None,
            aggregate_edges_for_globals_fn=None,
        )(
            graph._replace(
                edges=message_fn(graph.edges, 
                                graph.nodes[graph.senders], 
                                graph.nodes[graph.receivers])
            )
        ).nodes
        
        return graph._replace(nodes=updated_nodes)

class GNN(nn.Module):
    """
    Multi-layer GNN from GCBF+ with attention-based message passing
    Adapted for single-agent LiDAR processing
    """
    msg_dim: int = 128
    out_dim: int = 128  
    n_layers: int = 3
    hid_size_msg: Tuple[int, ...] = (256, 256)
    hid_size_aggr: Tuple[int, ...] = (128, 128)
    hid_size_update: Tuple[int, ...] = (256, 256)
    
    @nn.compact  
    def __call__(self, graph: jraph.GraphsTuple) -> jnp.ndarray:
        current_graph = graph
        
        for i in range(self.n_layers):
            out_dim = self.out_dim if i == self.n_layers - 1 else self.msg_dim
            layer = GNNLayer(
                msg_dim=self.msg_dim,
                out_dim=out_dim,
                hid_size_msg=self.hid_size_msg,
                hid_size_aggr=self.hid_size_aggr,
                hid_size_update=self.hid_size_update
            )
            current_graph = layer(current_graph)
            
        return current_graph.nodes

class CBFNet(nn.Module):
    """
    CBF network combining GNN + MLP head
    Adapted from GCBF+ CBF module implementation
    """
    gnn_msg_dim: int = 128
    gnn_out_dim: int = 128
    gnn_layers: int = 3
    head_sizes: Tuple[int, ...] = (256, 256)
    
    @nn.compact
    def __call__(self, graph: jraph.GraphsTuple, node_types: jnp.ndarray) -> jnp.ndarray:
        # GNN processing
        gnn = GNN(
            msg_dim=self.gnn_msg_dim,
            out_dim=self.gnn_out_dim, 
            n_layers=self.gnn_layers
        )
        node_embeddings = gnn(graph)  # (n_nodes, gnn_out_dim)
        
        # Extract ego drone embedding without boolean indexing
        # We know ego node is always at index 0 by construction
        ego_embedding = node_embeddings[0:1, :]  # (1, gnn_out_dim) - slice instead of boolean mask
        
        # MLP head for CBF prediction
        head = MLP(hid_sizes=self.head_sizes, act=nn.relu, act_final=False)
        x = head(ego_embedding)
        
        # Final CBF value with tanh activation (bounded output)
        cbf_raw = nn.Dense(1, kernel_init=default_nn_init())(x)
        cbf_value = nn.tanh(cbf_raw)
        
        # Return scalar instead of (1,) shape
        return cbf_value.squeeze()  # Remove all dimensions of size 1

# =============================================================================
# PERCEPTION MODULE
# =============================================================================

class PerceptionModule:
    """
    Complete perception pipeline for CBF computation
    """
    
    def __init__(self, config: GraphConfig):
        self.config = config
        self.cbf_net = CBFNet()
        
    def get_cbf_and_gradients(self, params, drone_state: DroneState, point_cloud: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute CBF value and gradients from point cloud input
        
        Args:
            params: Network parameters
            drone_state: Current drone state
            point_cloud: LiDAR point cloud (N, 3)
            
        Returns:
            cbf_value: CBF value (scalar)
            cbf_gradients: CBF gradients w.r.t. drone position
        """
        def cbf_fn(state):
            graph, node_types = pointcloud_to_graph(state, point_cloud, self.config)
            return self.cbf_net.apply(params, graph, node_types)
        
        # Compute CBF value
        cbf_value = cbf_fn(drone_state)
        
        # Compute gradients w.r.t. drone position
        grad_fn = jax.grad(lambda state: cbf_fn(state).sum())  # sum() for scalar output
        cbf_gradients = grad_fn(drone_state)
        
        return cbf_value, cbf_gradients

# =============================================================================
# JIT-COMPATIBLE WRAPPER FUNCTIONS
# =============================================================================

@ft.partial(jit, static_argnums=(2,))  # config is static
def pointcloud_to_graph_jit(drone_state: DroneState, point_cloud: jnp.ndarray, config: GraphConfig) -> Tuple[jraph.GraphsTuple, jnp.ndarray]:
    """JIT-compatible version of pointcloud_to_graph with static config"""
    return pointcloud_to_graph(drone_state, point_cloud, config)

@ft.partial(jit, static_argnums=(2,))  # k is static  
def find_knn_edges_jit(drone_pos: jnp.ndarray, obstacle_points: jnp.ndarray, k: int) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """JIT-compatible version of find_knn_edges with static k"""
    return find_knn_edges(drone_pos, obstacle_points, k)

# =============================================================================
# FACTORY FUNCTIONS AND UTILITIES
# =============================================================================

def create_default_perception_module():
    """Create perception module with default parameters"""
    config = GraphConfig()
    return PerceptionModule(config)

def init_cbf_network(rng_key, input_graph: jraph.GraphsTuple, node_types: jnp.ndarray):
    """Initialize CBF network parameters"""
    cbf_net = CBFNet()
    return cbf_net.init(rng_key, input_graph, node_types)

# Core function for integration with loop.py scan_function
@jit
def get_cbf_from_pointcloud(params, drone_state: DroneState, point_cloud: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Optimized function for CBF computation in training loop
    Used by loop.py scan_function
    JIT-compiled for performance
    """
    config = GraphConfig()
    graph, node_types = pointcloud_to_graph(drone_state, point_cloud, config)
    
    cbf_net = CBFNet()
    cbf_value = cbf_net.apply(params, graph, node_types)
    
    # Compute gradients w.r.t. drone position (needed for QP constraints)
    def cbf_wrt_position(pos):
        # Add numerical stability checks
        pos_clipped = jnp.clip(pos, -10.0, 10.0)  # Prevent extreme positions
        
        modified_state = DroneState(
            position=pos_clipped,
            velocity=drone_state.velocity,
            orientation=drone_state.orientation,
            angular_velocity=drone_state.angular_velocity
        )
        graph_mod, node_types_mod = pointcloud_to_graph(modified_state, point_cloud, config)
        cbf_raw = cbf_net.apply(params, graph_mod, node_types_mod)
        
        # Apply numerical stability: prevent extreme CBF values
        cbf_stable = jnp.clip(cbf_raw, -5.0, 5.0)
        return cbf_stable
    
    # Compute gradients with clipping
    cbf_grad_raw = jax.grad(lambda pos: cbf_wrt_position(pos).sum())(drone_state.position)
    
    # Apply gradient clipping for numerical stability
    grad_norm = jnp.linalg.norm(cbf_grad_raw)
    max_grad_norm = 10.0  # Maximum allowed gradient norm
    
    cbf_grad = jnp.where(
        grad_norm > max_grad_norm,
        cbf_grad_raw * (max_grad_norm / (grad_norm + 1e-8)),
        cbf_grad_raw
    )
    
    return cbf_value, cbf_grad

# =============================================================================
# TESTING AND VALIDATION
# =============================================================================

def test_pointcloud_to_graph():
    """Test point cloud to graph conversion"""
    # Create test data
    drone_state = DroneState(
        position=jnp.array([0.0, 0.0, 1.0]),
        velocity=jnp.array([1.0, 0.0, 0.0]),
        orientation=jnp.eye(3),
        angular_velocity=jnp.zeros(3)
    )
    
    # Simple obstacle point cloud
    point_cloud = jnp.array([
        [1.0, 0.0, 0.0],  # Front obstacle
        [0.0, 1.0, 0.0],  # Right obstacle  
        [-1.0, 0.0, 0.0], # Rear obstacle
        [0.0, -1.0, 0.0], # Left obstacle
        [0.0, 0.0, 1.0],  # Top obstacle
    ])
    
    config = GraphConfig()
    graph, node_types = pointcloud_to_graph(drone_state, point_cloud, config)
    
    print(f"Number of nodes: {graph.n_node}")
    print(f"Number of edges: {graph.n_edge}")
    print(f"Node features shape: {graph.nodes.shape}")
    print(f"Edge features shape: {graph.edges.shape}")
    print(f"Node types: {node_types}")
    
    return graph, node_types

if __name__ == "__main__":
    # Run basic test
    print("Testing point cloud to graph conversion...")
    test_graph, test_node_types = test_pointcloud_to_graph()
    print("Test completed!")