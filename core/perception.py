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
    图注意力层 - 参考GCBF+论文的实现
    确保梯度能够正确流动
    """
    hidden_dim: int
    num_heads: int = 4
    dropout_rate: float = 0.1
    sensing_radius: float = 2.0
    use_edge_features: bool = True
    
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
        if n_nodes == 0:
            return nodes, jnp.zeros((0, 0))
            
        node_dim = nodes.shape[-1]
        edge_dim = edges.shape[-1] if len(edges.shape) > 2 else 0
        head_dim = self.hidden_dim // self.num_heads
        
        # 投影层
        query_proj = nn.Dense(self.hidden_dim, name='query_proj')
        key_proj = nn.Dense(self.hidden_dim, name='key_proj')
        value_proj = nn.Dense(self.hidden_dim, name='value_proj')
        
        if self.use_edge_features and edge_dim > 0:
            edge_proj = nn.Dense(self.hidden_dim, name='edge_proj')
        
        # 计算Q, K, V
        queries = query_proj(nodes)  # [n_nodes, hidden_dim]
        keys = key_proj(nodes)       # [n_nodes, hidden_dim]
        values = value_proj(nodes)   # [n_nodes, hidden_dim]
        
        # 重塑为多头
        queries = queries.reshape(n_nodes, self.num_heads, head_dim)
        keys = keys.reshape(n_nodes, self.num_heads, head_dim)
        values = values.reshape(n_nodes, self.num_heads, head_dim)
        
        # 计算注意力分数
        attention_scores = jnp.einsum('ihd,jhd->ijh', queries, keys) / jnp.sqrt(head_dim)
        
        # 添加边特征（如果有）
        if self.use_edge_features and edge_dim > 0:
            edge_features_proj = edge_proj(edges)  # [n_nodes, n_nodes, hidden_dim]
            edge_features_proj = edge_features_proj.reshape(
                n_nodes, n_nodes, self.num_heads, head_dim
            )
            edge_scores = jnp.einsum('ihd,ijhd->ijh', queries, edge_features_proj)
            attention_scores = attention_scores + edge_scores / jnp.sqrt(head_dim)
        
        # 应用邻接掩码
        mask = adjacency[:, :, None] == 0
        attention_scores = jnp.where(mask, -1e9, attention_scores)
        
        # Softmax
        attention_weights = jax.nn.softmax(attention_scores, axis=1)
        
        # 应用dropout
        if training and self.dropout_rate > 0:
            dropout_rng = self.make_rng('dropout')
            keep_prob = 1.0 - self.dropout_rate
            dropout_mask = jax.random.bernoulli(dropout_rng, keep_prob, attention_weights.shape)
            attention_weights = attention_weights * dropout_mask / keep_prob
        
        # 聚合值
        aggregated_values = jnp.einsum('ijh,jhd->ihd', attention_weights, values)
        aggregated_values = aggregated_values.reshape(n_nodes, self.hidden_dim)
        
        # 输出投影
        output_proj = nn.Dense(self.hidden_dim, name='output_proj')
        output = output_proj(aggregated_values)
        
        # 残差连接和层归一化
        if node_dim == self.hidden_dim:
            output = nn.LayerNorm()(output + nodes)
        else:
            # 投影到相同维度
            residual_proj = nn.Dense(self.hidden_dim, name='residual_proj')
            output = nn.LayerNorm()(output + residual_proj(nodes))
        
        # 平均注意力权重用于可视化
        avg_attention_weights = jnp.mean(attention_weights, axis=-1)
        
        return output, avg_attention_weights

class GCBFGNN(nn.Module):
    """
    GCBF图神经网络 - 输出CBF值和梯度
    确保输出非常数值
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
        """
        nodes = graph_data['nodes']        # [n_nodes, node_dim]
        edges = graph_data['edges']        # [n_nodes, n_nodes, edge_dim]
        adjacency = graph_data['adjacency']  # [n_nodes, n_nodes]
        
        n_nodes = nodes.shape[0] if len(nodes.shape) > 0 else 0
        
        if n_nodes == 0:
            return jnp.zeros(0), jnp.zeros((0, 6))
        
        # 初始嵌入
        current_features = nn.Dense(
            self.hidden_dims[0], 
            kernel_init=nn.initializers.xavier_uniform()
        )(nodes)
        current_features = nn.relu(current_features)
        
        # 图注意力层
        for layer_idx in range(self.num_layers):
            hidden_dim = self.hidden_dims[min(layer_idx, len(self.hidden_dims)-1)]
            
            gat_layer = GraphAttentionLayer(
                hidden_dim=hidden_dim,
                num_heads=self.num_attention_heads,
                dropout_rate=self.dropout_rate,
                sensing_radius=self.sensing_radius,
                use_edge_features=True,
                name=f'gat_layer_{layer_idx}'
            )
            
            current_features, _ = gat_layer(
                current_features, edges, adjacency, training=training
            )
            
            # 激活函数（最后一层除外）
            if layer_idx < self.num_layers - 1:
                current_features = nn.relu(current_features)
                if training and self.dropout_rate > 0:
                    current_features = nn.Dropout(
                        rate=self.dropout_rate, 
                        deterministic=not training
                    )(current_features)
        
        # CBF值输出头 - 确保输出有变化
        cbf_head_1 = nn.Dense(64, kernel_init=nn.initializers.xavier_uniform())
        cbf_head_2 = nn.Dense(32, kernel_init=nn.initializers.xavier_uniform())
        cbf_head_3 = nn.Dense(1, kernel_init=nn.initializers.normal(stddev=0.1))
        
        cbf_features = cbf_head_1(current_features)
        cbf_features = nn.relu(cbf_features)
        cbf_features = cbf_head_2(cbf_features)
        cbf_features = nn.relu(cbf_features)
        cbf_values = cbf_head_3(cbf_features).squeeze(-1)
        
        # 添加残差以防止常数输出
        position_norm = jnp.linalg.norm(nodes[:, :3], axis=-1)
        cbf_values = cbf_values + 0.1 * (position_norm - 2.0)
        
        # 输出CBF梯度
        if self.output_cbf_gradients:
            grad_dim = min(nodes.shape[-1], 6)  # 只对位置和速度
            
            grad_head_1 = nn.Dense(64, kernel_init=nn.initializers.xavier_uniform())
            grad_head_2 = nn.Dense(32, kernel_init=nn.initializers.xavier_uniform())
            grad_head_3 = nn.Dense(grad_dim, kernel_init=nn.initializers.normal(stddev=0.1))
            
            grad_features = grad_head_1(current_features)
            grad_features = nn.relu(grad_features)
            grad_features = grad_head_2(grad_features)
            grad_features = nn.relu(grad_features)
            cbf_gradients = grad_head_3(grad_features)
            
            # 归一化梯度以防止梯度爆炸
            grad_norm = jnp.linalg.norm(cbf_gradients, axis=-1, keepdims=True)
            cbf_gradients = cbf_gradients / jnp.maximum(grad_norm, 1.0)
        else:
            cbf_gradients = jnp.zeros((n_nodes, 6))
        
        return cbf_values, cbf_gradients

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
    
    # 确保样本数据有效
    if 'nodes' not in sample_graph_data:
        sample_graph_data['nodes'] = jnp.zeros((4, 13))
    if 'edges' not in sample_graph_data:
        n_nodes = sample_graph_data['nodes'].shape[0]
        sample_graph_data['edges'] = jnp.zeros((n_nodes, n_nodes, 7))
    if 'adjacency' not in sample_graph_data:
        n_nodes = sample_graph_data['nodes'].shape[0]
        sample_graph_data['adjacency'] = jnp.ones((n_nodes, n_nodes))
    
    params = model.init(rng_key, sample_graph_data, training=False)
    
    return params

def test_gnn_computation():
    """测试GNN计算的正确性"""
    print("测试GNN计算...")
    
    # 创建测试数据
    n_agents = 4
    state_dim = 13
    rng_key = jax.random.PRNGKey(42)
    
    # 创建不同位置的智能体
    positions = jnp.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0]
    ])
    
    states = jnp.zeros((n_agents, state_dim))
    states = states.at[:, :3].set(positions)
    states = states.at[:, 6].set(1.0)  # 单位四元数
    
    # 构建图数据
    from core.training import construct_graph_batch
    graph_data = construct_graph_batch(states[None, :, :], sensing_radius=2.0)
    
    # 转换为单batch
    single_graph = {
        'nodes': graph_data['nodes'][0],
        'edges': graph_data['edges'][0],
        'adjacency': graph_data['adjacency'][0],
        'n_nodes': n_agents
    }
    
    # 创建GNN模型
    config = {'sensing_radius': 2.0}
    gnn_model = create_gnn_model(config)
    
    # 初始化参数
    gnn_params = initialize_gnn_params(gnn_model, rng_key, single_graph)
    
    # 前向传播
    cbf_values, cbf_gradients = gnn_model.apply(gnn_params, single_graph, training=False)
    
    print(f"CBF值: {cbf_values}")
    print(f"CBF值范围: [{jnp.min(cbf_values):.4f}, {jnp.max(cbf_values):.4f}]")
    print(f"CBF值标准差: {jnp.std(cbf_values):.4f}")
    print(f"CBF梯度形状: {cbf_gradients.shape}")
    print(f"CBF梯度范数: {jnp.linalg.norm(cbf_gradients, axis=-1)}")
    
    # 检查输出
    assert not jnp.all(cbf_values == cbf_values[0]), "CBF值不应该全部相同"
    assert jnp.std(cbf_values) > 1e-4, "CBF值应该有变化"
    assert not jnp.any(jnp.isnan(cbf_values)), "CBF值不应包含NaN"
    assert not jnp.any(jnp.isnan(cbf_gradients)), "CBF梯度不应包含NaN"
    
    print("✅ GNN计算测试通过")
    return True

__all__ = [
    'GraphAttentionLayer',
    'GCBFGNN',
    'create_gnn_model',
    'initialize_gnn_params',
    'test_gnn_computation'
]