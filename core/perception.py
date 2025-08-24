"""
感知模块：点云处理和图神经网络
基于GCBF+的GNN架构
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import jraph
from typing import Tuple, NamedTuple, Optional
import chex


class GraphFeatures(NamedTuple):
    """图特征表示"""
    node_features: chex.Array  # [num_nodes, node_feature_dim]
    edge_features: chex.Array  # [num_edges, edge_feature_dim] 
    edges: chex.Array  # [num_edges, 2] - (sender, receiver)
    globals: Optional[chex.Array] = None


def pointcloud_to_graph(drone_position: chex.Array,
                       point_cloud: chex.Array,
                       sensing_radius: float = 5.0,
                       max_neighbors: int = 32) -> jraph.GraphsTuple:
    """
    将点云数据转换为图结构
    
    Args:
        drone_position: 无人机位置 [3]
        point_cloud: 点云数据 [N, 3]
        sensing_radius: 感知半径
        max_neighbors: 最大邻居数
        
    Returns:
        graph: jraph图结构
    """
    
    num_points = point_cloud.shape[0]
    
    # 1. 构建节点
    # 节点包括：无人机节点(1个) + 点云节点(N个)
    drone_node = jnp.array([1.0, 0.0, 0.0])  # [agent, obstacle, goal]
    obstacle_nodes = jnp.array([[0.0, 1.0, 0.0]] * num_points)
    node_types = jnp.concatenate([drone_node[None, :], obstacle_nodes], axis=0)
    
    # 节点位置特征
    all_positions = jnp.concatenate([drone_position[None, :], point_cloud], axis=0)
    
    # 组合节点特征
    node_features = jnp.concatenate([node_types, all_positions], axis=1)  # [N+1, 6]
    
    # 2. 构建边
    # 计算无人机到各点的距离
    distances = jnp.linalg.norm(point_cloud - drone_position, axis=1)
    
    # 选择在感知半径内的点
    in_range_mask = distances < sensing_radius
    in_range_indices = jnp.where(in_range_mask, jnp.arange(num_points), -1)
    
    # 按距离排序，选择最近的max_neighbors个
    sorted_indices = jnp.argsort(distances)
    selected_indices = sorted_indices[:max_neighbors]
    
    # 过滤掉无效索引和超出范围的点
    valid_mask = (selected_indices < num_points) & (distances[selected_indices] < sensing_radius)
    valid_indices = selected_indices[valid_mask]
    
    num_valid_edges = jnp.sum(valid_mask)
    
    # 构建边：无人机(节点0)到选中的障碍物点
    senders = jnp.zeros(num_valid_edges, dtype=jnp.int32)  # 无人机节点索引
    receivers = valid_indices + 1  # 障碍物节点索引（+1因为无人机是节点0）
    
    edges = jnp.stack([senders, receivers], axis=1)
    
    # 3. 边特征
    edge_vectors = point_cloud[valid_indices] - drone_position  # 相对位置向量
    edge_distances = jnp.linalg.norm(edge_vectors, axis=1, keepdims=True)
    edge_features = jnp.concatenate([edge_vectors, edge_distances], axis=1)  # [E, 4]
    
    # 4. 构建jraph图
    graph = jraph.GraphsTuple(
        nodes=node_features,
        edges=edge_features,
        senders=senders,
        receivers=receivers,
        n_node=jnp.array([node_features.shape[0]]),
        n_edge=jnp.array([edges.shape[0]]),
        globals=None
    )
    
    return graph


class GCBFGraphNet(nn.Module):
    """
    基于GCBF+的图神经网络
    实现消息传递和注意力机制
    """
    
    hidden_dim: int = 128
    num_message_passing_steps: int = 3
    output_dim: int = 1  # CBF输出维度
    
    def setup(self):
        # 节点和边的编码器
        self.node_encoder = nn.Dense(self.hidden_dim)
        self.edge_encoder = nn.Dense(self.hidden_dim)
        
        # 消息传递网络
        self.message_nets = [
            nn.Dense(self.hidden_dim) for _ in range(self.num_message_passing_steps)
        ]
        self.update_nets = [
            nn.Dense(self.hidden_dim) for _ in range(self.num_message_passing_steps)
        ]
        
        # 注意力机制
        self.attention_net = nn.Dense(1)
        
        # 输出网络
        self.output_net = nn.Sequential([
            nn.Dense(self.hidden_dim),
            nn.leaky_relu,
            nn.Dense(self.hidden_dim // 2),
            nn.leaky_relu,
            nn.Dense(self.output_dim)
        ])
    
    def __call__(self, graph: jraph.GraphsTuple) -> Tuple[chex.Array, chex.Array]:
        """
        前向传播
        
        Args:
            graph: 输入图
            
        Returns:
            (h, grad_h): CBF值和梯度
        """
        
        # 编码节点和边特征
        node_embeddings = self.node_encoder(graph.nodes)
        edge_embeddings = self.edge_encoder(graph.edges)
        
        # 消息传递
        for i in range(self.num_message_passing_steps):
            # 计算消息
            messages = self.message_nets[i](edge_embeddings)
            
            # 聚合消息（使用注意力机制）
            attention_weights = nn.softmax(self.attention_net(messages))
            aggregated_messages = jraph.segment_sum(
                messages * attention_weights, 
                graph.receivers, 
                num_segments=graph.nodes.shape[0]
            )
            
            # 更新节点表示
            node_embeddings = self.update_nets[i](
                jnp.concatenate([node_embeddings, aggregated_messages], axis=1)
            )
            node_embeddings = nn.leaky_relu(node_embeddings)
        
        # 提取无人机节点的特征（节点0）
        drone_embedding = node_embeddings[0]
        
        # 计算CBF值
        h = self.output_net(drone_embedding)
        
        # 计算CBF梯度（使用自动微分）
        def cbf_fn(pos):
            # 重新构建图，仅改变无人机位置
            modified_nodes = graph.nodes.at[0, 3:6].set(pos)
            modified_graph = graph._replace(nodes=modified_nodes)
            embedding = self.node_encoder(modified_graph.nodes)[0]
            return self.output_net(embedding)
        
        drone_position = graph.nodes[0, 3:6]  # 提取位置部分
        grad_h = jax.grad(lambda pos: cbf_fn(pos).sum())(drone_position)
        
        return h.squeeze(), grad_h


def create_dummy_pointcloud(rng_key: chex.PRNGKey,
                          num_points: int = 50,
                          bounds: float = 10.0) -> chex.Array:
    """创建虚拟点云数据，用于测试"""
    return jax.random.uniform(
        rng_key, 
        (num_points, 3),
        minval=-bounds,
        maxval=bounds
    )


# JIT编译版本
pointcloud_to_graph_jit = jax.jit(pointcloud_to_graph, static_argnames=['max_neighbors'])


def test_perception_pipeline():
    """测试感知流水线"""
    rng_key = jax.random.PRNGKey(42)
    
    # 创建测试数据
    drone_pos = jnp.array([0.0, 0.0, 2.0])
    point_cloud = create_dummy_pointcloud(rng_key, num_points=20)
    
    # 测试点云到图转换
    graph = pointcloud_to_graph(drone_pos, point_cloud)
    
    print(f"图节点数: {graph.nodes.shape[0]}")
    print(f"图边数: {graph.edges.shape[0]}")
    print(f"节点特征维度: {graph.nodes.shape[1]}")
    print(f"边特征维度: {graph.edges.shape[1]}")
    
    # 测试GNN
    gnn = GCBFGraphNet()
    params = gnn.init(rng_key, graph)
    
    h, grad_h = gnn.apply(params, graph)
    
    print(f"CBF值: {h}")
    print(f"CBF梯度: {grad_h}")
    print(f"梯度范数: {jnp.linalg.norm(grad_h)}")
    
    return graph, h, grad_h


if __name__ == "__main__":
    test_perception_pipeline()