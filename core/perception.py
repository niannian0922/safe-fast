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
    
    Edge types:
    1. Ego drone to k nearest obstacles
    2. Each obstacle to k nearest neighbors (including drone)
    
    Args:
        drone_pos: (3,) - Drone position
        obstacle_points: (N, 3) - Obstacle point cloud
        k: KNN neighbor count
        
    Returns:
        senders: (M,) - Sender node indices
        receivers: (M,) - Receiver node indices  
        edge_features: (M, edge_dim) - Edge features
    """
    n_obstacles = obstacle_points.shape[0]
    n_total = n_obstacles + 1  # +1 for ego drone node
    
    # Combined position array: [drone, obstacle1, obstacle2, ...]
    all_positions = jnp.concatenate([drone_pos[None, :], obstacle_points], axis=0)
    
    # Compute all pairwise distances
    distances = compute_pairwise_distances(all_positions, all_positions)
    
    # Mask diagonal to exclude self-connections
    # Set diagonal to infinity to exclude from KNN
    distances_masked = jnp.where(jnp.eye(n_total), jnp.inf, distances)
    
    # Find k nearest neighbors using top_k (negate for smallest distances)
    k_actual = jnp.minimum(k, n_total - 1)  # Ensure k doesn't exceed available neighbors
    # Apply top_k along each row separately using vmap
    def find_top_k_for_row(distances_row):
        _, indices = jax.lax.top_k(-distances_row, k_actual)
        return indices
    
    top_k_indices = jax.vmap(find_top_k_for_row)(distances_masked)
    
    # Build edge lists
    senders_list = []
    receivers_list = []
    edge_features_list = []
    
    for i in range(n_total):
        neighbors = top_k_indices[i]  # (k,) 
        for j in range(k_actual):
            neighbor_idx = neighbors[j]
            # Only add edge if neighbor is valid and different from sender
            if neighbor_idx != i and distances[i, neighbor_idx] < jnp.inf:
                senders_list.append(i)
                receivers_list.append(neighbor_idx)
                
                # Edge features: [distance, relative_position (3D)]
                rel_pos = all_positions[neighbor_idx] - all_positions[i]
                distance = distances[i, neighbor_idx]
                edge_feat = jnp.concatenate([jnp.array([distance]), rel_pos])
                edge_features_list.append(edge_feat)
    
    # Convert to arrays
    senders = jnp.array(senders_list) if senders_list else jnp.array([], dtype=jnp.int32)
    receivers = jnp.array(receivers_list) if receivers_list else jnp.array([], dtype=jnp.int32)
    edge_features = jnp.stack(edge_features_list) if edge_features_list else jnp.zeros((0, 4))
    
    return senders, receivers, edge_features

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
        indices = jax.random.choice(
            jax.random.PRNGKey(0), n_valid, (config.max_points,), replace=False
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
    
    # 4. Build edges
    senders, receivers, edge_features = find_knn_edges(
        drone_state.position, valid_points, config.k_neighbors
    )
    
    # 5. Node type classification
    # 0 = ego drone, 1 = obstacle
    node_types = jnp.concatenate([
        jnp.array([0]),  # ego node
        jnp.ones(n_valid)  # obstacle nodes
    ]).astype(jnp.int32)
    
    # 6. Create GraphsTuple
    n_nodes = n_valid + 1
    n_edges = len(senders)
    
    graph = jraph.GraphsTuple(
        n_node=jnp.array([n_nodes]),  # Single graph
        n_edge=jnp.array([n_edges]),  # Single graph
        nodes=all_node_features,  # (n_nodes, node_features)
        edges=edge_features,  # (n_edges, edge_features) 
        globals=None,
        senders=senders,  # (n_edges,)
        receivers=receivers,  # (n_edges,)
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
        
        # Extract ego drone embedding (only ego node type == 0)
        ego_mask = node_types == 0
        ego_embedding = node_embeddings[ego_mask]  # Should be exactly one ego node
        
        # MLP head for CBF prediction
        head = MLP(hid_sizes=self.head_sizes, act=nn.relu, act_final=False)
        x = head(ego_embedding)
        
        # Final CBF value with tanh activation (bounded output)
        cbf_value = nn.tanh(nn.Dense(1, kernel_init=default_nn_init())(x))
        
        return cbf_value.squeeze(-1)  # Remove last dimension

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
def get_cbf_from_pointcloud(params, drone_state: DroneState, point_cloud: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Optimized function for CBF computation in training loop
    Used by loop.py scan_function
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