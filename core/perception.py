"""
修复的感知模块 - 确保GNN正确计算CBF值和梯度
参考GCBF+论文的图神经网络架构
"""

import jax
import jax.numpy as jnp
import jraph
from flax import linen as nn
from typing import Dict, Tuple, Optional, Callable
import chex
from functools import partial

class GraphAttentionLayer(nn.Module):
    """
    图注意力层 - 参考GCBF+论文的图注意力机制
    满足Definition 1中的条件1和条件2
    """
    hidden_dim: int
    num_heads: int = 4
    dropout_rate: float = 0.1
    sensing_radius: float = 2.0
    
    @nn.compact
    def __call__(self, nodes, edges, adjacency, training=True):
        """
        Args:
            nodes: 节点特征 [n_nodes, node_dim]
            edges: 边特征 [n_nodes, n_nodes, edge_dim]  
            adjacency: 邻接矩阵 [n_nodes, n_nodes]
            training: 训练模式标志
        
        Returns:
            updated_nodes: 更新后的节点特征 [n_nodes, hidden_dim]
            attention_weights: 注意力权重 [n_nodes, n_nodes]
        """
        n_nodes = nodes.shape[0]
        node_dim = nodes.shape[-1]
        edge_dim = edges.shape[-1]
        head_dim = self.hidden_dim // self.num_heads
        
        # 线性投影
        W_q = self.param('W_q', nn.initializers.xavier_uniform(), (node_dim, self.hidden_dim))
        W_k = self.param('W_k', nn.initializers.xavier_uniform(), (node_dim, self.hidden_dim)) 
        W_v = self.param('W_v', nn.initializers.xavier_uniform(), (node_dim, self.hidden_dim))
        W_e = self.param('W_e', nn.initializers.xavier_uniform(), (edge_dim, self.hidden_dim))
        
        # 查询、键、值
        Q = nodes @ W_q  # [n_nodes, hidden_dim]
        K = nodes @ W_k  # [n_nodes, hidden_dim] 
        V = nodes @ W_v  # [n_nodes, hidden_dim]
        
        # 边特征投影
        E = edges @ W_e  # [n_nodes, n_nodes, hidden_dim]
        
        # 多头注意力
        Q = Q.reshape(n_nodes, self.num_heads, head_dim)  # [n_nodes, num_heads, head_dim]
        K = K.reshape(n_nodes, self.num_heads, head_dim)
        V = V.reshape(n_nodes, self.num_heads, head_dim)  
        E = E.reshape(n_nodes, n_nodes, self.num_heads, head_dim)
        
        # 计算注意力分数
        # Q: [n_nodes, num_heads, head_dim]
        # K: [n_nodes, num_heads, head_dim] 
        # 计算 Q @ K^T: [n_nodes, n_nodes, num_heads]
        attention_scores = jnp.einsum('ihd,jhd->ijh', Q, K) / jnp.sqrt(head_dim)
        
        # 添加边特征到注意力分数
        edge_contribution = jnp.einsum('ihd,ijhd->ijh', Q, E) / jnp.sqrt(head_dim)
        attention_scores = attention_scores + edge_contribution
        
        # 应用距离掩码 - 这是满足Definition 1条件的关键
        # 计算节点间距离（假设前3维是位置）
        positions = nodes[:, :3]  # [n_nodes, 3]
        pos_i = positions[:, None, :]  # [n_nodes, 1, 3]
        pos_j = positions[None, :, :]  # [1, n_nodes, 3]  
        distances = jnp.linalg.norm(pos_i - pos_j, axis=-1)  # [n_nodes, n_nodes]
        
        # 距离掩码：超出感知半径的设为-inf
        distance_mask = distances >= self.sensing_radius
        attention_scores = jnp.where(
            distance_mask[:, :, None], 
            -jnp.inf, 
            attention_scores
        )
        
        # 应用邻接掩码
        adjacency_mask = ~adjacency.astype(bool)
        attention_scores = jnp.where(
            adjacency_mask[:, :, None],
            -jnp.inf,
            attention_scores
        )
        
        # 对角掩码（节点不注意自己）
        eye_mask = jnp.eye(n_nodes, dtype=bool)
        attention_scores = jnp.where(
            eye_mask[:, :, None],
            -jnp.inf, 
            attention_scores
        )
        
        # Softmax注意力权重
        attention_weights = jax.nn.softmax(attention_scores, axis=1)  # [n_nodes, n_nodes, num_heads]
        
        # 处理NaN（当所有权重都是-inf时）
        attention_weights = jnp.where(
            jnp.isnan(attention_weights),
            0.0,
            attention_weights
        )
        
        # 应用注意力权重到值
        # V: [n_nodes, num_heads, head_dim]
        # attention_weights: [n_nodes, n_nodes, num_heads] 
        attended_values = jnp.einsum('ijh,jhd->ihd', attention_weights, V)  # [n_nodes, num_heads, head_dim]
        
        # 合并多头
        attended_values = attended_values.reshape(n_nodes, self.hidden_dim)  # [n_nodes, hidden_dim]
        
        # 残差连接和层归一化
        if node_dim == self.hidden_dim:
            output = nn.LayerNorm()(attended_values + nodes)
        else:
            # 维度不匹配时，使用线性投影
            W_res = self.param('W_res', nn.initializers.xavier_uniform(), (node_dim, self.hidden_dim))
            output = nn.LayerNorm()(attended_values + nodes @ W_res)
        
        # Dropout
        if training:
            output = nn.Dropout(rate=self.dropout_rate, deterministic=False)(output)
        
        # 返回平均注意力权重用于可视化
        avg_attention_weights = jnp.mean(attention_weights, axis=-1)  # [n_nodes, n_nodes]
        
        return output, avg_attention_weights

class GCBFGNN(nn.Module):
    """
    GCBF图神经网络 - 参考GCBF+论文架构
    输出CBF值和梯度
    """
    hidden_dims: Tuple[int, ...] = (128, 128, 64)
    num_attention_heads: int = 4
    num_layers: int = 3
    dropout_rate: float = 0.1
    sensing_radius: float = 2.0
    output_cbf_gradients: bool = True
    
    @nn.compact 
    def __call__(self, graph_data: Dict[str, jnp.ndarray], training=True) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        前向传播
        
        Args:
            graph_data: 包含nodes, edges, adjacency的图数据
            training: 训练模式
            
        Returns:
            cbf_values: CBF值 [n_nodes,]
            cbf_gradients: CBF梯度 [n_nodes, state_dim] (如果启用)
        """
        nodes = graph_data['nodes']        # [n_nodes, node_dim]
        edges = graph_data['edges']        # [n_nodes, n_nodes, edge_dim]
        adjacency = graph_data['adjacency']  # [n_nodes, n_nodes]
        
        n_nodes = nodes.shape[0]
        node_dim = nodes.shape[-1]
        
        # 确保输入有效性
        if n_nodes == 0:
            # 空图情况的处理
            empty_cbf_values = jnp.zeros(0)
            empty_cbf_gradients = jnp.zeros((0, node_dim))
            return empty_cbf_values, empty_cbf_gradients
        
        # 初始特征投影
        current_features = nn.Dense(self.hidden_dims[0])(nodes)
        
        # 多层图注意力
        attention_weights_history = []
        for layer_idx, hidden_dim in enumerate(self.hidden_dims):
            layer = GraphAttentionLayer(
                hidden_dim=hidden_dim,
                num_heads=self.num_attention_heads,
                dropout_rate=self.dropout_rate,
                sensing_radius=self.sensing_radius,
                name=f'gat_layer_{layer_idx}'
            )
            
            current_features, attention_weights = layer(
                current_features, edges, adjacency, training=training
            )
            attention_weights_history.append(attention_weights)
            
            # 中间层的非线性激活
            if layer_idx < len(self.hidden_dims) - 1:
                current_features = nn.gelu(current_features)
        
        # CBF值输出头
        cbf_features = nn.Dense(64, name='cbf_hidden')(current_features)
        cbf_features = nn.gelu(cbf_features)
        cbf_features = nn.Dense(32, name='cbf_hidden2')(cbf_features)  
        cbf_features = nn.gelu(cbf_features)
        cbf_values = nn.Dense(1, name='cbf_output')(cbf_features).squeeze(-1)  # [n_nodes,]
        
        if self.output_cbf_gradients:
            # CBF梯度输出头 - 梯度关于状态的前几维（通常是位置和速度）
            state_dim = min(node_dim, 6)  # 通常考虑位置和速度
            grad_features = nn.Dense(64, name='grad_hidden')(current_features)
            grad_features = nn.gelu(grad_features)
            grad_features = nn.Dense(32, name='grad_hidden2')(grad_features)
            grad_features = nn.gelu(grad_features)  
            cbf_gradients = nn.Dense(state_dim, name='grad_output')(grad_features)  # [n_nodes, state_dim]
        else:
            cbf_gradients = jnp.zeros((n_nodes, min(node_dim, 6)))
        
        return cbf_values, cbf_gradients

def construct_graph_from_states_robust(
    states: jnp.ndarray,                # [n_agents, state_dim]
    point_cloud: Optional[jnp.ndarray] = None,  # [n_points, 3] 
    sensing_radius: float = 2.0,
    max_neighbors: int = 10
) -> Dict[str, jnp.ndarray]:
    """
    鲁棒的图构建函数 - 修复版本
    参考GCBF+论文的图构建方法，确保始终产生有效图
    
    Args:
        states: 智能体状态
        point_cloud: LiDAR点云（可选）
        sensing_radius: 感知半径
        max_neighbors: 最大邻居数
        
    Returns:
        graph_data: 图数据字典
    """
    n_agents = states.shape[0]
    state_dim = states.shape[1]
    
    if n_agents == 0:
        # 处理空智能体情况
        return {
            'nodes': jnp.zeros((0, state_dim + 1)),
            'edges': jnp.zeros((0, 0, 4)),
            'adjacency': jnp.zeros((0, 0)),
            'positions': jnp.zeros((0, 3)),
            'n_agents': 0,
            'n_total_nodes': 0
        }
    
    # 提取位置信息
    positions = states[:, :3]  # [n_agents, 3]
    
    # 如果有点云，将其作为额外节点
    if point_cloud is not None and point_cloud.shape[0] > 0:
        n_points = point_cloud.shape[0]
        # 为点云创建特征（位置 + 零填充 + 类型标识）
        point_features = jnp.concatenate([
            point_cloud,                                    # 位置 [n_points, 3] 
            jnp.zeros((n_points, max(0, state_dim - 3))),   # 填充到匹配状态维度
            jnp.ones((n_points, 1)) * 2                     # 类型标识：2=障碍物点
        ], axis=1)
        
        # 合并智能体和点云节点
        all_positions = jnp.concatenate([positions, point_cloud], axis=0)
        agent_features = jnp.concatenate([
            states,
            jnp.ones((n_agents, 1)) * 1  # 类型标识：1=智能体
        ], axis=1)
        node_features = jnp.concatenate([agent_features, point_features], axis=0)
        n_total_nodes = n_agents + n_points
    else:
        # 只有智能体节点
        all_positions = positions
        node_features = jnp.concatenate([
            states,
            jnp.ones((n_agents, 1)) * 1  # 类型标识：1=智能体  
        ], axis=1)
        n_total_nodes = n_agents
    
    # 构建邻接矩阵 - 确保数值稳定性
    pos_i = all_positions[:, None, :]  # [n_total_nodes, 1, 3]
    pos_j = all_positions[None, :, :]  # [1, n_total_nodes, 3]
    distances = jnp.linalg.norm(pos_i - pos_j + 1e-8, axis=-1)  # 添加小的常数避免数值问题
    
    # 邻接关系：在感知半径内且不是自身
    adjacency = (distances < sensing_radius) & (distances > 1e-6)
    
    # 限制邻居数量 - 修复版本
    def limit_neighbors_robust(node_idx):
        node_distances = distances[node_idx]
        node_adjacency = adjacency[node_idx]
        
        # 如果没有邻居，创建全零邻接
        if not jnp.any(node_adjacency):
            return jnp.zeros(n_total_nodes, dtype=bool)
        
        # 找到有效邻居并按距离排序
        valid_neighbors = jnp.where(node_adjacency, node_distances, jnp.inf)
        neighbor_indices = jnp.argsort(valid_neighbors)
        
        # 保留前max_neighbors个最近邻居
        limited_adj = jnp.zeros(n_total_nodes, dtype=bool)
        n_keep = min(max_neighbors, jnp.sum(node_adjacency))
        limited_adj = limited_adj.at[neighbor_indices[:n_keep]].set(
            node_adjacency[neighbor_indices[:n_keep]]
        )
        return limited_adj
    
    adjacency = jax.vmap(limit_neighbors_robust)(jnp.arange(n_total_nodes))
    
    # 构建边特征
    rel_positions = pos_i - pos_j  # [n_total_nodes, n_total_nodes, 3]
    rel_distances = distances[:, :, None]  # [n_total_nodes, n_total_nodes, 1]
    
    # 基础边特征：相对位置 + 相对距离
    edge_features = jnp.concatenate([
        rel_positions,    # [n_total_nodes, n_total_nodes, 3]
        rel_distances     # [n_total_nodes, n_total_nodes, 1]
    ], axis=-1)  # [n_total_nodes, n_total_nodes, 4]
    
    # 如果有速度信息，添加相对速度
    if state_dim >= 6:
        velocities = jnp.concatenate([
            states[:, 3:6],  # 智能体速度
            jnp.zeros((n_total_nodes - n_agents, 3)) if point_cloud is not None and point_cloud.shape[0] > 0 else jnp.empty((0, 3))
        ], axis=0)
        vel_i = velocities[:, None, :]  # [n_total_nodes, 1, 3]
        vel_j = velocities[None, :, :]  # [1, n_total_nodes, 3]
        rel_velocities = vel_i - vel_j  # [n_total_nodes, n_total_nodes, 3]
        
        edge_features = jnp.concatenate([
            edge_features,    # [..., 4]
            rel_velocities    # [..., 3] 
        ], axis=-1)  # [..., 7]
    
    # 确保邻接矩阵是浮点类型
    adjacency = adjacency.astype(jnp.float32)
    
    graph_data = {
        'nodes': node_features,      # [n_total_nodes, feature_dim]
        'edges': edge_features,      # [n_total_nodes, n_total_nodes, edge_dim]
        'adjacency': adjacency,      # [n_total_nodes, n_total_nodes]
        'positions': all_positions,  # [n_total_nodes, 3]
        'n_agents': n_agents,
        'n_total_nodes': n_total_nodes
    }
    
    return graph_data

@jax.jit
def batch_construct_graph_from_states(
    batch_states: jnp.ndarray,         # [batch_size, n_agents, state_dim]
    batch_point_clouds: Optional[jnp.ndarray] = None,  # [batch_size, n_points, 3]
    sensing_radius: float = 2.0,
    max_neighbors: int = 10
) -> Dict[str, jnp.ndarray]:
    """批量处理点云到图的转换 - JIT优化版本"""
    
    def convert_single_batch(states, point_cloud=None):
        return construct_graph_from_states_robust(
            states, point_cloud, sensing_radius, max_neighbors
        )
    
    if batch_point_clouds is None:
        # 没有点云的情况
        batch_graphs = jax.vmap(lambda s: convert_single_batch(s))(batch_states)
    else:
        # 有点云的情况
        batch_graphs = jax.vmap(convert_single_batch)(batch_states, batch_point_clouds)
    
    return batch_graphs

def create_gnn_model(config: Dict) -> GCBFGNN:
    """创建GNN模型"""
    return GCBFGNN(
        hidden_dims=config.get('gnn_hidden_dims', (128, 128, 64)),
        num_attention_heads=config.get('gnn_attention_heads', 4),
        num_layers=config.get('gnn_num_layers', 3),
        dropout_rate=config.get('gnn_dropout_rate', 0.1),
        sensing_radius=config.get('sensing_radius', 2.0),
        output_cbf_gradients=config.get('output_cbf_gradients', True)
    )

def initialize_gnn_params(
    model: GCBFGNN,
    rng_key: jax.random.PRNGKey,
    sample_graph_data: Dict[str, jnp.ndarray]
) -> Dict:
    """初始化GNN参数"""
    
    # 创建样例输入
    dummy_input = {
        'nodes': sample_graph_data['nodes'],
        'edges': sample_graph_data['edges'], 
        'adjacency': sample_graph_data['adjacency']
    }
    
    # 初始化参数
    params = model.init(rng_key, dummy_input, training=False)
    
    return params

def validate_graph_structure(graph_data: Dict[str, jnp.ndarray]) -> bool:
    """验证图结构的完整性"""
    
    required_keys = ['nodes', 'edges', 'adjacency']
    for key in required_keys:
        if key not in graph_data:
            print(f"缺少必要的图数据键: {key}")
            return False
    
    nodes = graph_data['nodes']
    edges = graph_data['edges'] 
    adjacency = graph_data['adjacency']
    
    n_nodes = nodes.shape[0]
    
    if n_nodes == 0:
        print("警告: 空图，但这是合法的")
        return True
    
    # 检查维度一致性
    if edges.shape[:2] != (n_nodes, n_nodes):
        print(f"边特征维度不匹配: {edges.shape[:2]} vs ({n_nodes}, {n_nodes})")
        return False
        
    if adjacency.shape != (n_nodes, n_nodes):
        print(f"邻接矩阵维度不匹配: {adjacency.shape} vs ({n_nodes}, {n_nodes})")
        return False
    
    # 检查数值有效性
    if jnp.any(jnp.isnan(nodes)):
        print("警告: 节点特征包含NaN")
        return False
        
    if jnp.any(jnp.isnan(edges)):
        print("警告: 边特征包含NaN")
        return False
    
    print(f"图结构验证通过: {n_nodes} 节点, {jnp.sum(adjacency):.0f} 条边")
    return True

# 测试函数
def test_gnn_computation():
    """测试GNN计算的正确性"""
    print("测试GNN计算...")
    
    # 创建测试数据
    n_agents = 4
    state_dim = 6
    rng_key = jax.random.PRNGKey(42)
    
    # 随机状态
    states = jax.random.normal(rng_key, (n_agents, state_dim))
    
    # 构建图
    graph_data = construct_graph_from_states_robust(states)
    
    # 验证图结构
    if not validate_graph_structure(graph_data):
        return False
    
    # 创建GNN模型
    config = {'sensing_radius': 2.0}
    gnn_model = create_gnn_model(config)
    
    # 初始化参数
    gnn_params = initialize_gnn_params(gnn_model, rng_key, graph_data)
    
    # 前向传播
    cbf_values, cbf_gradients = gnn_model.apply(gnn_params, graph_data, training=False)
    
    print(f"CBF值: {cbf_values}")
    print(f"CBF值形状: {cbf_values.shape}")
    print(f"CBF梯度形状: {cbf_gradients.shape}")
    
    # 检查输出有效性
    if jnp.any(jnp.isnan(cbf_values)):
        print("错误: CBF值包含NaN")
        return False
        
    if jnp.any(jnp.isnan(cbf_gradients)):
        print("错误: CBF梯度包含NaN")
        return False
    
    print("✅ GNN计算测试通过")
    return True

# 主要导出函数  
__all__ = [
    'GraphAttentionLayer',
    'GCBFGNN', 
    'construct_graph_from_states_robust',
    'batch_construct_graph_from_states',
    'create_gnn_model',
    'initialize_gnn_params',
    'validate_graph_structure',
    'test_gnn_computation'
]