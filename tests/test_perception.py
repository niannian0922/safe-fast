"""
Unit tests for perception module (Stage 3 validation)

This test suite validates:
1. Point cloud to graph conversion functionality
2. JIT compilation compatibility
3. GNN-based CBF computation
4. Gradient computation correctness
5. Integration with JAX transformations

Test focuses on verifying the core adaptation from GCBF+'s multi-agent
neighbor discovery to single-agent LiDAR processing.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import jraph
import pytest
from jax import random, grad, jit

# Import modules under test
from core.perception import (
    DroneState, GraphConfig, 
    pointcloud_to_graph, 
    compute_pairwise_distances,
    find_knn_edges,
    CBFNet, GNN, MLP,
    get_cbf_from_pointcloud,
    init_cbf_network
)

class TestPointCloudToGraph:
    """Test point cloud to graph conversion functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.config = GraphConfig(
            k_neighbors=4,
            max_range=10.0,
            min_points=5,
            max_points=50,
            obstacle_node_features=3
        )
        
        self.drone_state = DroneState(
            position=jnp.array([0.0, 0.0, 1.0]),
            velocity=jnp.array([1.0, 0.0, 0.0]),
            orientation=jnp.eye(3),
            angular_velocity=jnp.zeros(3)
        )
        
        self.point_cloud = jnp.array([
            [1.0, 0.0, 0.0],   # Front obstacle
            [0.0, 1.0, 0.0],   # Right obstacle
            [-1.0, 0.0, 0.0],  # Rear obstacle
            [0.0, -1.0, 0.0],  # Left obstacle
            [0.0, 0.0, 1.0],   # Top obstacle
            [2.0, 2.0, 0.0],   # Distant obstacle
        ])

    def test_pairwise_distances(self):
        """Test pairwise distance computation"""
        points1 = jnp.array([[0., 0., 0.], [1., 1., 1.]])
        points2 = jnp.array([[1., 0., 0.], [0., 1., 0.]])
        
        distances = compute_pairwise_distances(points1, points2)
        
        # Verify shape
        assert distances.shape == (2, 2)
        
        # Verify specific distances
        expected_d00 = 1.0  # Distance from [0,0,0] to [1,0,0]
        expected_d11 = jnp.sqrt(2.0)  # Distance from [1,1,1] to [0,1,0]
        
        assert jnp.allclose(distances[0, 0], expected_d00)
        assert jnp.allclose(distances[1, 1], expected_d11)
        
        print("✅ Pairwise distances computation: PASSED")

    def test_knn_edges(self):
        """Test KNN edge construction"""
        drone_pos = jnp.array([0., 0., 0.])
        obstacles = jnp.array([
            [1., 0., 0.],
            [0., 1., 0.],
            [2., 0., 0.],
        ])
        
        senders, receivers, edge_features = find_knn_edges(drone_pos, obstacles, k=2)
        
        # Should have edges for each node to its k nearest neighbors
        assert len(senders) > 0
        assert len(senders) == len(receivers)
        assert len(senders) == edge_features.shape[0]
        
        # Edge features should have correct dimension (distance + 3D relative position)
        assert edge_features.shape[1] == 4
        
        # All edge indices should be valid
        n_total = len(obstacles) + 1  # +1 for drone
        assert jnp.all(senders < n_total)
        assert jnp.all(receivers < n_total)
        
        print("✅ KNN edge construction: PASSED")

    def test_pointcloud_to_graph_structure(self):
        """Test graph structure creation"""
        graph, node_types = pointcloud_to_graph(
            self.drone_state, self.point_cloud, self.config
        )
        
        # Verify graph structure
        assert isinstance(graph, jraph.GraphsTuple)
        assert graph.n_node.shape == (1,)  # Single graph
        assert graph.n_edge.shape == (1,)
        
        # Verify node count (drone + obstacles)
        expected_nodes = len(self.point_cloud) + 1
        assert graph.n_node[0] == expected_nodes
        
        # Verify node features shape
        assert graph.nodes.shape[0] == expected_nodes
        assert graph.nodes.shape[1] == self.config.obstacle_node_features
        
        # Verify node types
        assert node_types.shape[0] == expected_nodes
        assert node_types[0] == 0  # Ego node
        assert jnp.all(node_types[1:] == 1)  # Obstacle nodes
        
        print("✅ Graph structure creation: PASSED")

    def test_jit_compilation(self):
        """Test JIT compilation of pointcloud_to_graph"""
        # Function should already be JIT compiled
        jit_fn = jit(pointcloud_to_graph, static_argnums=2)
        
        # Test compilation
        graph_jit, node_types_jit = jit_fn(
            self.drone_state, self.point_cloud, self.config
        )
        
        # Compare with non-JIT version
        graph_regular, node_types_regular = pointcloud_to_graph(
            self.drone_state, self.point_cloud, self.config
        )
        
        # Results should be identical
        assert jnp.allclose(graph_jit.nodes, graph_regular.nodes)
        assert jnp.array_equal(node_types_jit, node_types_regular)
        
        print("✅ JIT compilation: PASSED")

    def test_edge_cases(self):
        """Test edge cases for graph construction"""
        # Test with very few points
        small_cloud = jnp.array([[1., 0., 0.]])
        graph_small, _ = pointcloud_to_graph(
            self.drone_state, small_cloud, self.config
        )
        
        # Should pad to min_points
        assert graph_small.n_node[0] >= self.config.min_points + 1
        
        # Test with many points (should subsample)
        config_large = GraphConfig(max_points=3)
        large_cloud = jnp.random.normal(0, 1, (20, 3))
        graph_large, _ = pointcloud_to_graph(
            self.drone_state, large_cloud, config_large
        )
        
        # Should limit to max_points + 1 (drone)
        assert graph_large.n_node[0] == config_large.max_points + 1
        
        print("✅ Edge cases handling: PASSED")


class TestGNNArchitecture:
    """Test GNN-based CBF computation"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.rng_key = random.PRNGKey(42)
        self.config = GraphConfig()
        
        # Create test graph
        self.drone_state = DroneState(
            position=jnp.array([0.0, 0.0, 1.0]),
            velocity=jnp.array([1.0, 0.0, 0.0]),
            orientation=jnp.eye(3),
            angular_velocity=jnp.zeros(3)
        )
        
        point_cloud = jnp.array([
            [1., 0., 0.], [0., 1., 0.], [-1., 0., 0.], [0., -1., 0.]
        ])
        
        self.graph, self.node_types = pointcloud_to_graph(
            self.drone_state, point_cloud, self.config
        )

    def test_mlp_module(self):
        """Test MLP building block"""
        mlp = MLP(hid_sizes=(64, 32), act=jax.nn.relu)
        
        # Initialize and test
        test_input = jnp.ones((4, 10))
        params = mlp.init(self.rng_key, test_input)
        output = mlp.apply(params, test_input)
        
        assert output.shape == (4, 32)  # Last layer size
        assert not jnp.any(jnp.isnan(output))
        
        print("✅ MLP module: PASSED")

    def test_gnn_forward_pass(self):
        """Test GNN forward pass"""
        gnn = GNN(msg_dim=64, out_dim=32, n_layers=2)
        
        # Initialize parameters
        params = gnn.init(self.rng_key, self.graph)
        
        # Forward pass
        node_embeddings = gnn.apply(params, self.graph)
        
        # Verify output shape
        expected_shape = (self.graph.n_node[0], 32)
        assert node_embeddings.shape == expected_shape
        assert not jnp.any(jnp.isnan(node_embeddings))
        
        print("✅ GNN forward pass: PASSED")

    def test_cbf_net(self):
        """Test CBF network end-to-end"""
        cbf_net = CBFNet(gnn_layers=2, head_sizes=(128, 64))
        
        # Initialize parameters
        params = cbf_net.init(self.rng_key, self.graph, self.node_types)
        
        # Forward pass
        cbf_value = cbf_net.apply(params, self.graph, self.node_types)
        
        # Should output single scalar for ego node
        assert cbf_value.shape == ()  # Scalar
        assert not jnp.isnan(cbf_value)
        assert -1.0 <= cbf_value <= 1.0  # Bounded by tanh
        
        print("✅ CBF network: PASSED")

    def test_cbf_gradients(self):
        """Test CBF gradient computation"""
        cbf_net = CBFNet()
        params = cbf_net.init(self.rng_key, self.graph, self.node_types)
        
        def cbf_fn(drone_state):
            graph, node_types = pointcloud_to_graph(
                drone_state, jnp.array([[1., 0., 0.]]), self.config
            )
            return cbf_net.apply(params, graph, node_types)
        
        # Compute gradients w.r.t. drone position
        grad_fn = grad(lambda state: cbf_fn(state).sum())
        gradients = grad_fn(self.drone_state)
        
        # Gradients should exist and be finite
        assert hasattr(gradients, 'position')
        assert gradients.position.shape == (3,)
        assert jnp.all(jnp.isfinite(gradients.position))
        
        print("✅ CBF gradient computation: PASSED")


class TestIntegrationFunctions:
    """Test integration functions for training loop"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.rng_key = random.PRNGKey(123)
        self.drone_state = DroneState(
            position=jnp.array([0.0, 0.0, 1.0]),
            velocity=jnp.array([0.5, 0.0, 0.0]),
            orientation=jnp.eye(3),
            angular_velocity=jnp.zeros(3)
        )
        self.point_cloud = jnp.array([
            [2., 0., 0.], [0., 2., 0.], [-2., 0., 0.]
        ])

    def test_cbf_initialization(self):
        """Test CBF network parameter initialization"""
        graph, node_types = pointcloud_to_graph(
            self.drone_state, self.point_cloud, GraphConfig()
        )
        
        params = init_cbf_network(self.rng_key, graph, node_types)
        
        # Should return valid parameter dictionary
        assert isinstance(params, dict)
        assert 'params' in params
        
        print("✅ CBF network initialization: PASSED")

    def test_get_cbf_from_pointcloud(self):
        """Test optimized CBF computation for training loop"""
        # Initialize network
        graph, node_types = pointcloud_to_graph(
            self.drone_state, self.point_cloud, GraphConfig()
        )
        params = init_cbf_network(self.rng_key, graph, node_types)
        
        # Test function
        cbf_value, cbf_grad = get_cbf_from_pointcloud(
            params, self.drone_state, self.point_cloud
        )
        
        # Verify outputs
        assert cbf_value.shape == ()  # Scalar
        assert cbf_grad.shape == (3,)  # 3D gradient
        assert jnp.all(jnp.isfinite(cbf_value))
        assert jnp.all(jnp.isfinite(cbf_grad))
        
        print("✅ CBF computation from point cloud: PASSED")

    def test_jit_compatibility(self):
        """Test JIT compilation of integrated functions"""
        # Setup
        graph, node_types = pointcloud_to_graph(
            self.drone_state, self.point_cloud, GraphConfig()
        )
        params = init_cbf_network(self.rng_key, graph, node_types)
        
        # JIT compile the function
        jit_cbf_fn = jit(get_cbf_from_pointcloud)
        
        # Test compilation and execution
        cbf_value_jit, cbf_grad_jit = jit_cbf_fn(
            params, self.drone_state, self.point_cloud
        )
        
        # Compare with regular version
        cbf_value_reg, cbf_grad_reg = get_cbf_from_pointcloud(
            params, self.drone_state, self.point_cloud
        )
        
        assert jnp.allclose(cbf_value_jit, cbf_value_reg)
        assert jnp.allclose(cbf_grad_jit, cbf_grad_reg)
        
        print("✅ JIT compilation compatibility: PASSED")

    def test_batch_processing(self):
        """Test batched processing capabilities"""
        # Create batch of drone states
        batch_size = 4
        batch_positions = jnp.array([
            [0., 0., 1.],
            [1., 0., 1.],
            [0., 1., 1.],
            [1., 1., 1.]
        ])
        
        # Initialize once
        graph, node_types = pointcloud_to_graph(
            self.drone_state, self.point_cloud, GraphConfig()
        )
        params = init_cbf_network(self.rng_key, graph, node_types)
        
        # Process batch using vmap
        def single_cbf_fn(pos):
            state = DroneState(
                position=pos,
                velocity=self.drone_state.velocity,
                orientation=self.drone_state.orientation,
                angular_velocity=self.drone_state.angular_velocity
            )
            return get_cbf_from_pointcloud(params, state, self.point_cloud)
        
        batch_cbf_fn = jax.vmap(single_cbf_fn)
        cbf_values, cbf_grads = batch_cbf_fn(batch_positions)
        
        # Verify batch outputs
        assert cbf_values.shape == (batch_size,)
        assert cbf_grads.shape == (batch_size, 3)
        assert jnp.all(jnp.isfinite(cbf_values))
        assert jnp.all(jnp.isfinite(cbf_grads))
        
        print("✅ Batch processing: PASSED")


def run_all_tests():
    """Run comprehensive test suite"""
    print("🔍 Starting Perception Module Test Suite...")
    print("=" * 60)
    
    # Test point cloud to graph conversion
    print("\n📊 Testing Point Cloud to Graph Conversion...")
    test_pc_to_graph = TestPointCloudToGraph()
    test_pc_to_graph.setup_method()
    test_pc_to_graph.test_pairwise_distances()
    test_pc_to_graph.test_knn_edges()
    test_pc_to_graph.test_pointcloud_to_graph_structure()
    test_pc_to_graph.test_jit_compilation()
    test_pc_to_graph.test_edge_cases()
    
    # Test GNN architecture
    print("\n🧠 Testing GNN Architecture...")
    test_gnn = TestGNNArchitecture()
    test_gnn.setup_method()
    test_gnn.test_mlp_module()
    test_gnn.test_gnn_forward_pass()
    test_gnn.test_cbf_net()
    test_gnn.test_cbf_gradients()
    
    # Test integration functions
    print("\n🔗 Testing Integration Functions...")
    test_integration = TestIntegrationFunctions()
    test_integration.setup_method()
    test_integration.test_cbf_initialization()
    test_integration.test_get_cbf_from_pointcloud()
    test_integration.test_jit_compatibility()
    test_integration.test_batch_processing()
    
    print("\n" + "=" * 60)
    print("🎉 PERCEPTION MODULE: ALL TESTS PASSED!")
    print("✅ Point cloud to graph conversion: VERIFIED")
    print("✅ JIT compilation compatibility: VERIFIED") 
    print("✅ GNN-based CBF computation: VERIFIED")
    print("✅ Gradient computation: VERIFIED")
    print("✅ Integration readiness: VERIFIED")


if __name__ == "__main__":
    run_all_tests()