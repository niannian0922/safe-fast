"""
感知模块：点云处理和图神经网络 - 完全修复版
基于GCBF+的GNN架构，正确集成CBF计算，确保梯度流通
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import jraph
from typing import Tuple, NamedTuple, Optional, Any, Dict
import chex


def pointcloud_to_graph(drone_position: chex.Array,
                       point_cloud: chex.Array,
                       sensing_radius: float = 5.0,
                       max_neighbors: int = 16) -> jraph.GraphsTuple:
    """
    将点云数据转换为图结构 - 修复版，确保总是有有效边
    
    Args:
        drone_position: 无人机位置 [3]
        point_cloud: 点云数据 [N, 3]
        sensing_radius: 感知半径
        max_neighbors: 最大邻居数
        
    Returns:
        graph: jraph图结构，保证有边连接
    """
    
    num_points = point_cloud.shape[0]
    
    # 1. 构建节点 - 包括无人机节点和障碍物节点
    # 节点特征：[node_type(3), position(3)] = 6维
    drone_node_type = jnp.array([1.0, 0.0, 0.0])  # [agent, obstacle, goal]
    obstacle_node_type = jnp.array([0.0, 1.0, 0.0])
    
    # 组合节点特征
    drone_features = jnp.concatenate([drone_node_type, drone_position])
    obstacle_features = jnp.concatenate([
        jnp.tile(obstacle_node_type[None, :], (num_points, 1)),
        point_cloud
    ], axis=1)
    
    all_nodes = jnp.concatenate([drone_features[None, :], obstacle_features], axis=0)
    
    # 2. 构建边 - 确保总是有边存在
    # 计算无人机到各点的距离
    distances = jnp.linalg.norm(point_cloud - drone_position, axis=1)
    
    # 按距离排序，选择最近的邻居
    sorted_indices = jnp.argsort(distances)
    
    # 确保至少有一条边 - 选择最近的点，即使超出感知半径
    min_edges = jnp.minimum(max_neighbors, num_points)
    min_edges = jnp.maximum(min_edges, 1)  # 至少1条边
    
    selected_indices = sorted_indices[:min_edges]
    selected_distances = distances[selected_indices]
    
    # 应用感知半径过滤，但保留至少一条边
    within_radius_mask = selected_distances < sensing_radius
    num_within_radius = jnp.sum(within_radius_mask)
    
    # 如果没有点在感知半径内，至少保留最近的一个
    final_mask = jnp.where(
        num_within_radius > 0,
        within_radius_mask,
        jnp.arange(min_edges) == 0  # 只保留最近的点
    )
    
    valid_indices = selected_indices[final_mask]
    num_valid_edges = jnp.sum(final_mask)
    
    # 构建边：从无人机(节点0)到障碍物节点
    senders = jnp.zeros(num_valid_edges, dtype=jnp.int32)
    receivers = valid_indices + 1  # +1因为无人机是节点0
    
    # 3. 边特征：[相对位置(3), 距离(1)] = 4维
    relative_positions = point_cloud[valid_indices] - drone_position
    edge_distances = jnp.linalg.norm(relative_positions, axis=1, keepdims=True)
    edge_features = jnp.concatenate([relative_positions, edge_distances], axis=1)
    
    # 4. 构建jraph图
    graph = jraph.GraphsTuple(
        nodes=all_nodes,  # [num_nodes, 6]
        edges=edge_features,  # [num_edges, 4]
        senders=senders,  # [num_edges]
        receivers=receivers,  # [num_edges]
        n_node=jnp.array([all_nodes.shape[0]]),  # [1]
        n_edge=jnp.array([senders.shape[0]]),  # [1]
        globals=None
    )
    
    return graph


class CBFGraphNet(nn.Module):
    """
    基于GCBF+的图神经网络 - 完全修复版
    专门设计用于CBF值和梯度计算，确保梯度流通
    """
    
    hidden_dim: int = 64
    num_layers: int = 2
    
    def setup(self):
        # 编码器
        self.node_encoder = nn.Dense(self.hidden_dim)
        self.edge_encoder = nn.Dense(self.hidden_dim)
        
        # 消息传递网络
        self.message_nets = [
            nn.Dense(self.hidden_dim) for _ in range(self.num_layers)
        ]
        
        # 更新网络
        self.update_nets = [
            nn.Dense(self.hidden_dim) for _ in range(self.num_layers)
        ]
        
        # 输出网络 - 直接输出CBF值
        self.cbf_output = nn.Sequential([
            nn.Dense(self.hidden_dim // 2),
            nn.relu,
            nn.Dense(1)
        ])
    
    def __call__(self, graph: jraph.GraphsTuple) -> float:
        """
        前向传播 - 只输出CBF值，梯度通过JAX自动计算
        
        Args:
            graph: 输入图
            
        Returns:
            h: CBF值 (标量)
        """
        
        # 编码节点和边特征
        nodes = self.node_encoder(graph.nodes)  # [num_nodes, hidden_dim]
        edges = self.edge_encoder(graph.edges)  # [num_edges, hidden_dim]
        
        # 多层消息传递
        for layer_idx in range(self.num_layers):
            # 计算消息
            messages = self.message_nets[layer_idx](edges)
            
            # 聚合消息到接收节点
            aggregated = jraph.segment_sum(
                messages,
                graph.receivers,
                num_segments=graph.nodes.shape[0]
            )
            
            # 更新节点
            nodes = self.update_nets[layer_idx](nodes + aggregated)
            nodes = nn.relu(nodes)
        
        # 提取无人机节点特征(节点0)并计算CBF值
        drone_features = nodes[0]  # [hidden_dim]
        cbf_value = self.cbf_output(drone_features)  # [1]
        
        return cbf_value.squeeze()  # 返回标量


def create_perception_system(config: Dict[str, Any] = None) -> Tuple[CBFGraphNet, Any]:
    """
    创建感知系统 - 修复版
    
    Returns:
        (gnn_model, perception_fn): GNN模型和感知函数
    """
    
    if config is None:
        config = {
            'sensing_radius': 5.0,
            'max_neighbors': 16,
            'hidden_dim': 64,
            'num_layers': 2
        }
    
    # 创建GNN模型
    gnn_model = CBFGraphNet(
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers']
    )
    
    def perception_fn(gnn_params: Any,
                     drone_position: chex.Array,
                     point_cloud: chex.Array) -> Tuple[float, chex.Array]:
        """
        感知函数 - 计算CBF值和梯度，完全JAX兼容
        
        Args:
            gnn_params: GNN参数
            drone_position: 无人机位置 [3]
            point_cloud: 点云 [N, 3]
            
        Returns:
            (h, grad_h): CBF值和梯度
        """
        
        # 构建图
        graph = pointcloud_to_graph(
            drone_position,
            point_cloud,
            config['sensing_radius'],
            config['max_neighbors']
        )
        
        # 定义CBF函数用于梯度计算
        def cbf_fn(pos):
            # 修改图中无人机的位置
            modified_node_features = graph.nodes.at[0, 3:6].set(pos)
            modified_graph = graph._replace(nodes=modified_node_features)
            return gnn_model.apply(gnn_params, modified_graph)
        
        # 计算CBF值
        h = cbf_fn(drone_position)
        
        # 计算CBF梯度
        grad_h = jax.grad(cbf_fn)(drone_position)
        
        return h, grad_h
    
    return gnn_model, perception_fn


def create_dummy_pointcloud(rng_key: chex.PRNGKey,
                          num_points: int = 20,
                          bounds: float = 8.0,
                          min_distance: float = 1.0) -> chex.Array:
    """创建虚拟点云 - 确保点不太近"""
    points = jax.random.uniform(
        rng_key,
        (num_points, 3),
        minval=-bounds,
        maxval=bounds
    )
    
    # 确保点与原点有最小距离
    distances = jnp.linalg.norm(points, axis=1)
    too_close_mask = distances < min_distance
    
    # 将太近的点推远
    directions = points / (distances[:, None] + 1e-8)
    adjusted_points = jnp.where(
        too_close_mask[:, None],
        directions * min_distance,
        points
    )
    
    return adjusted_points


def test_perception_system():
    """测试感知系统 - 完整版"""
    print("🧠 测试感知系统")
    print("=" * 40)
    
    rng_key = jax.random.PRNGKey(42)
    
    # 创建测试数据
    drone_pos = jnp.array([0.0, 0.0, 2.0])
    point_cloud = create_dummy_pointcloud(rng_key, num_points=15)
    
    print(f"无人机位置: {drone_pos}")
    print(f"点云形状: {point_cloud.shape}")
    
    # 测试图构建
    graph = pointcloud_to_graph(drone_pos, point_cloud)
    print(f"图节点数: {graph.nodes.shape[0]}")
    print(f"图边数: {graph.edges.shape[0]}")
    print(f"节点特征维度: {graph.nodes.shape[1]}")
    print(f"边特征维度: {graph.edges.shape[1]}")
    
    # 验证图结构
    assert graph.edges.shape[0] > 0, "图必须有边"
    assert graph.nodes.shape[0] == point_cloud.shape[0] + 1, "节点数错误"
    
    # 创建感知系统
    gnn_model, perception_fn = create_perception_system()
    
    # 初始化参数
    gnn_params = gnn_model.init(rng_key, graph)
    
    print("GNN参数结构:")
    for key, value in jax.tree_util.tree_flatten_with_path(gnn_params)[0]:
        path_str = '.'.join(str(k) for k in key)
        print(f"  {path_str}: {value.shape}")
    
    # 测试感知函数
    h, grad_h = perception_fn(gnn_params, drone_pos, point_cloud)
    
    print(f"\nCBF值: {h:.6f}")
    print(f"CBF梯度: {grad_h}")
    print(f"梯度范数: {jnp.linalg.norm(grad_h):.6f}")
    
    # 验证输出
    assert not jnp.isnan(h), "CBF值不应为NaN"
    assert not jnp.any(jnp.isnan(grad_h)), "CBF梯度不应为NaN"
    assert jnp.linalg.norm(grad_h) > 1e-8, "CBF梯度应该非零"
    
    # 测试梯度流
    print("\n测试GNN梯度流...")
    
    def loss_fn(gnn_params_test):
        h_val, grad_h_val = perception_fn(gnn_params_test, drone_pos, point_cloud)
        return h_val**2 + 0.1 * jnp.sum(grad_h_val**2)
    
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(gnn_params)
    
    def tree_norm(tree):
        return jnp.sqrt(sum(jnp.sum(leaf**2) for leaf in jax.tree_util.tree_leaves(tree)))
    
    grad_norm = tree_norm(grads)
    print(f"GNN参数梯度范数: {grad_norm:.8f}")
    
    assert grad_norm > 1e-6, f"GNN梯度范数过小: {grad_norm}"
    assert not jnp.isnan(grad_norm), "GNN梯度不应包含NaN"
    
    # 测试不同位置的CBF值
    print("\n测试CBF对位置的敏感性...")
    positions = [
        jnp.array([0.0, 0.0, 2.0]),
        jnp.array([1.0, 0.0, 2.0]),  # 稍微移动
        jnp.array([0.0, 1.0, 2.0]),  # y方向移动
    ]
    
    cbf_values = []
    for pos in positions:
        h_pos, _ = perception_fn(gnn_params, pos, point_cloud)
        cbf_values.append(h_pos)
        print(f"位置 {pos} -> CBF: {h_pos:.6f}")
    
    # CBF值应该随位置变化
    cbf_variance = jnp.var(jnp.array(cbf_values))
    print(f"CBF值方差: {cbf_variance:.8f}")
    assert cbf_variance > 1e-6, f"CBF值对位置变化不敏感: {cbf_variance}"
    
    print("\n✅ 感知系统测试通过!")
    return True


if __name__ == "__main__":
    test_perception_system()