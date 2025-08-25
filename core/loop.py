"""
修复的BPTT循环模块 - 确保jax.lax.scan正确处理梯度流
参考GCBF+和DiffPhysDrone的时间展开实现
"""

import jax
import jax.numpy as jnp
from typing import Dict, Tuple, Callable, Any, NamedTuple
import chex
from functools import partial

class ScanCarry(NamedTuple):
    """scan函数的carry状态"""
    states: jnp.ndarray           # 当前状态 [batch_size, n_agents, state_dim]
    rnn_hidden: jnp.ndarray       # RNN隐状态（如果使用）
    rng_key: jax.random.PRNGKey   # 随机数种子

class ScanOutput(NamedTuple):
    """scan函数的输出"""
    states: jnp.ndarray           # 状态
    actions: jnp.ndarray          # 动作
    cbf_values: jnp.ndarray       # CBF值
    cbf_derivatives: jnp.ndarray  # CBF导数
    safety_violations: jnp.ndarray # 安全违规
    collision_distances: jnp.ndarray # 碰撞距离

def create_scan_step_fn(
    policy_apply_fn: Callable,
    gnn_apply_fn: Callable,
    physics_step_fn: Callable,
    config: Dict[str, Any]
) -> Callable:
    """
    创建scan步骤函数 - 这是BPTT的核心
    
    Args:
        policy_apply_fn: 策略网络应用函数
        gnn_apply_fn: GNN应用函数  
        physics_step_fn: 物理仿真函数
        config: 配置参数
        
    Returns:
        scan_step: 可被jax.lax.scan调用的步骤函数
    """
    
    def scan_step(carry: ScanCarry, inputs: Dict[str, jnp.ndarray]) -> Tuple[ScanCarry, ScanOutput]:
        """
        单个时间步的计算
        
        Args:
            carry: 前一步的状态
            inputs: 当前步的输入（目标速度、图数据等）
            
        Returns:
            new_carry: 更新后的carry
            outputs: 当前步的输出
        """
        # 解包carry
        current_states = carry.states  # [batch_size, n_agents, state_dim]
        rnn_hidden = carry.rnn_hidden
        rng_key = carry.rng_key
        
        # 解包输入
        target_velocities = inputs['target_velocities']  # [batch_size, n_agents, 3]
        graph_template = inputs.get('graph_template', {})
        policy_params = inputs['policy_params']
        gnn_params = inputs['gnn_params']
        
        # 分割随机数key
        rng_key, action_key, noise_key = jax.random.split(rng_key, 3)
        
        # === 步骤1: 从当前状态构建图数据 ===
        current_graph = construct_graph_from_states(
            current_states, graph_template, config
        )
        
        # === 步骤2: GNN计算CBF值和梯度 ===
        cbf_output = gnn_apply_fn(gnn_params, current_graph)
        
        if isinstance(cbf_output, tuple):
            cbf_values, cbf_grads = cbf_output
        else:
            cbf_values = cbf_output
            # 如果没有显式梯度，计算数值梯度
            cbf_grads = compute_numerical_cbf_gradients(
                gnn_apply_fn, gnn_params, current_states, current_graph, config
            )
        
        # 确保形状正确
        batch_size, n_agents = current_states.shape[:2]
        cbf_values = jnp.reshape(cbf_values, (batch_size, n_agents))
        cbf_grads = jnp.reshape(cbf_grads, (batch_size, n_agents, -1))
        
        # === 步骤3: 策略网络计算动作 ===
        policy_input = prepare_policy_input(
            current_states, target_velocities, cbf_values, rnn_hidden, config
        )
        
        policy_output = policy_apply_fn(policy_params, policy_input)
        
        if isinstance(policy_output, tuple):
            actions, new_rnn_hidden = policy_output
        else:
            actions = policy_output
            new_rnn_hidden = rnn_hidden
            
        # 添加动作噪声（如果在训练模式）
        if config.get('training_mode', True):
            action_noise = jax.random.normal(
                action_key, actions.shape
            ) * config.get('action_noise_std', 0.01)
            actions = actions + action_noise
            
        # 限制动作范围
        actions = jnp.clip(actions, -config.get('max_action', 1.0), config.get('max_action', 1.0))
        
        # === 步骤4: 物理仿真更新状态 ===
        next_states = physics_step_fn(current_states, actions)
        
        # === 步骤5: 计算CBF时间导数 ===
        cbf_derivatives = compute_cbf_time_derivative(
            cbf_values, cbf_grads, current_states, actions, config
        )
        
        # === 步骤6: 安全检查 ===
        collision_distances = compute_collision_distances(current_states)
        safety_violations = collision_distances < config.get('safety_radius', 0.5)
        
        # 创建输出
        outputs = ScanOutput(
            states=current_states,  # 当前状态
            actions=actions,
            cbf_values=cbf_values,
            cbf_derivatives=cbf_derivatives,
            safety_violations=safety_violations.astype(jnp.float32),
            collision_distances=collision_distances
        )
        
        # 创建新的carry
        new_carry = ScanCarry(
            states=next_states,     # 下一时刻状态
            rnn_hidden=new_rnn_hidden,
            rng_key=rng_key
        )
        
        return new_carry, outputs
    
    return scan_step

def construct_graph_from_states(
    states: jnp.ndarray,
    graph_template: Dict,
    config: Dict[str, Any]
) -> Dict[str, jnp.ndarray]:
    """
    从状态构建图数据 - 参考GCBF+的图构建方法
    
    Args:
        states: 智能体状态 [batch_size, n_agents, state_dim]
        graph_template: 图模板
        config: 配置参数
        
    Returns:
        graph_data: 图数据字典
    """
    batch_size, n_agents, state_dim = states.shape
    positions = states[..., :3]  # [batch_size, n_agents, 3]
    
    # 构建智能体间的邻接关系
    sensing_radius = config.get('sensing_radius', 2.0)
    
    # 计算所有智能体对之间的距离
    pos_i = jnp.expand_dims(positions, axis=2)  # [batch, n_agents, 1, 3]
    pos_j = jnp.expand_dims(positions, axis=1)  # [batch, 1, n_agents, 3]
    distances = jnp.linalg.norm(pos_i - pos_j, axis=-1)  # [batch, n_agents, n_agents]
    
    # 创建邻接矩阵（在感知半径内且不是自己）
    adjacency = (distances < sensing_radius) & (distances > 0)
    
    # 构建边特征（相对位置和速度）
    rel_positions = pos_i - pos_j  # [batch, n_agents, n_agents, 3]
    
    if state_dim >= 6:
        velocities = states[..., 3:6]
        vel_i = jnp.expand_dims(velocities, axis=2)
        vel_j = jnp.expand_dims(velocities, axis=1) 
        rel_velocities = vel_i - vel_j  # [batch, n_agents, n_agents, 3]
        edge_features = jnp.concatenate([rel_positions, rel_velocities], axis=-1)
    else:
        edge_features = rel_positions
    
    # 构建节点特征
    node_features = states  # 使用完整状态作为节点特征
    
    graph_data = {
        'nodes': node_features,           # [batch, n_agents, state_dim]
        'edges': edge_features,           # [batch, n_agents, n_agents, edge_dim]  
        'adjacency': adjacency,           # [batch, n_agents, n_agents]
        'n_nodes': n_agents,
        'batch_size': batch_size
    }
    
    return graph_data

def prepare_policy_input(
    states: jnp.ndarray,
    target_velocities: jnp.ndarray, 
    cbf_values: jnp.ndarray,
    rnn_hidden: jnp.ndarray,
    config: Dict[str, Any]
) -> Dict[str, jnp.ndarray]:
    """准备策略网络输入"""
    
    policy_input = {
        'states': states,
        'target_velocities': target_velocities,
        'cbf_values': cbf_values
    }
    
    if rnn_hidden is not None:
        policy_input['rnn_hidden'] = rnn_hidden
        
    return policy_input

def compute_numerical_cbf_gradients(
    gnn_apply_fn: Callable,
    gnn_params: Dict,
    states: jnp.ndarray,
    graph_data: Dict,
    config: Dict[str, Any],
    eps: float = 1e-6
) -> jnp.ndarray:
    """
    计算CBF关于状态的数值梯度
    当GNN没有直接输出梯度时使用
    """
    batch_size, n_agents, state_dim = states.shape
    
    def cbf_fn(states_flat):
        states_reshaped = states_flat.reshape(batch_size, n_agents, state_dim)
        # 重新构建图
        graph = construct_graph_from_states(states_reshaped, {}, config)
        graph.update(graph_data)  # 合并模板数据
        cbf_vals = gnn_apply_fn(gnn_params, graph)
        if isinstance(cbf_vals, tuple):
            cbf_vals = cbf_vals[0]
        return jnp.sum(cbf_vals)  # 标量输出用于求导
    
    # 计算梯度
    states_flat = states.reshape(-1)
    grad_fn = jax.grad(cbf_fn)
    grads_flat = grad_fn(states_flat)
    grads = grads_flat.reshape(batch_size, n_agents, state_dim)
    
    return grads

def compute_cbf_time_derivative(
    cbf_values: jnp.ndarray,
    cbf_grads: jnp.ndarray,
    states: jnp.ndarray, 
    actions: jnp.ndarray,
    config: Dict[str, Any]
) -> jnp.ndarray:
    """
    计算CBF时间导数: dh/dt = ∇h · f(x,u)
    这是连接CBF和动力学的关键
    """
    # 计算系统动力学 f(x,u)
    state_derivatives = compute_system_dynamics(states, actions, config)
    
    # CBF时间导数 = ∇h · ẋ  
    # cbf_grads: [batch, n_agents, state_dim]
    # state_derivatives: [batch, n_agents, state_dim]
    cbf_dot = jnp.sum(cbf_grads * state_derivatives, axis=-1)  # [batch, n_agents]
    
    return cbf_dot

def compute_system_dynamics(
    states: jnp.ndarray, 
    actions: jnp.ndarray,
    config: Dict[str, Any]
) -> jnp.ndarray:
    """
    计算系统动力学 ẋ = f(x,u)
    参考DiffPhysDrone的点质量模型
    """
    batch_size, n_agents, state_dim = states.shape
    
    if state_dim >= 6:
        # 完整状态: [pos, vel]
        positions = states[..., :3]
        velocities = states[..., 3:6]
        
        # 点质量动力学: 
        # ṗ = v
        # v̇ = u (动作为加速度)
        pos_dot = velocities
        vel_dot = actions[..., :3]  # 假设动作为加速度
        
        state_derivatives = jnp.concatenate([pos_dot, vel_dot], axis=-1)
    else:
        # 简化状态：只有位置，速度通过差分近似
        # ṗ = u (动作直接为速度)
        state_derivatives = actions[..., :state_dim]
    
    return state_derivatives

def compute_collision_distances(states: jnp.ndarray) -> jnp.ndarray:
    """计算智能体间的最小碰撞距离"""
    positions = states[..., :3]  # [batch, n_agents, 3]
    
    # 计算成对距离
    pos_i = jnp.expand_dims(positions, axis=2)  # [batch, n_agents, 1, 3]
    pos_j = jnp.expand_dims(positions, axis=1)  # [batch, 1, n_agents, 3] 
    distances = jnp.linalg.norm(pos_i - pos_j, axis=-1)  # [batch, n_agents, n_agents]
    
    # 排除自身距离
    n_agents = distances.shape[-1]
    mask = jnp.eye(n_agents)
    distances = jnp.where(mask, jnp.inf, distances)
    
    # 返回最小距离
    min_distances = jnp.min(distances, axis=-1)  # [batch, n_agents]
    
    return min_distances

@partial(jax.jit, static_argnums=(1, 2))
def execute_bptt_rollout(
    initial_carry: ScanCarry,
    scan_step_fn: Callable,
    seq_length: int,
    inputs_sequence: Dict[str, jnp.ndarray]
) -> Tuple[ScanCarry, ScanOutput]:
    """
    执行BPTT展开 - 使用jax.lax.scan实现高效计算
    
    Args:
        initial_carry: 初始carry状态
        scan_step_fn: scan步骤函数
        seq_length: 序列长度
        inputs_sequence: 输入序列
        
    Returns:
        final_carry: 最终carry状态  
        outputs: 所有时刻的输出
    """
    
    # 使用jax.lax.scan执行时间展开
    final_carry, outputs = jax.lax.scan(
        scan_step_fn,
        initial_carry,
        inputs_sequence,
        length=seq_length
    )
    
    return final_carry, outputs

@partial(jax.jit, static_argnums=(4,))
def create_rollout_inputs(
    target_velocities: jnp.ndarray,  # [batch, seq_len, n_agents, 3]
    policy_params: Dict,
    gnn_params: Dict, 
    graph_template: Dict,
    seq_length: int
) -> Dict[str, jnp.ndarray]:
    """
    创建rollout输入序列
    
    Returns:
        inputs_dict: 每个时刻的输入字典
    """
    
    # 为每个时间步创建输入
    inputs = {
        'target_velocities': target_velocities.transpose(1, 0, 2, 3),  # [seq_len, batch, n_agents, 3]
        'policy_params': policy_params,
        'gnn_params': gnn_params, 
        'graph_template': graph_template
    }
    
    return inputs

def apply_temporal_gradient_decay(
    gradients: Dict,
    decay_factor: float = 0.95
) -> Dict:
    """
    应用时间梯度衰减 - 参考DiffPhysDrone的梯度衰减机制
    
    Args:
        gradients: 梯度字典
        decay_factor: 衰减因子
        
    Returns:
        decayed_gradients: 衰减后的梯度
    """
    
    def decay_grad_tree(grad_tree):
        """递归地对梯度树应用衰减"""
        return jax.tree_util.tree_map(
            lambda x: x * decay_factor, 
            grad_tree
        )
    
    # 对所有梯度应用衰减
    decayed_gradients = jax.tree_util.tree_map(decay_grad_tree, gradients)
    
    return decayed_gradients

@jax.jit  
def rollout_with_gradient_checkpointing(
    initial_states: jnp.ndarray,
    target_velocities: jnp.ndarray,
    policy_params: Dict,
    gnn_params: Dict,
    scan_step_fn: Callable,
    config: Dict[str, Any]
) -> Tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]]:
    """
    执行带梯度检查点的rollout - 节省内存
    
    Returns:
        rollout_data: rollout数据
        metrics: 指标
    """
    batch_size, n_agents, state_dim = initial_states.shape
    seq_length = target_velocities.shape[1]
    
    # 创建初始carry
    initial_carry = ScanCarry(
        states=initial_states,
        rnn_hidden=jnp.zeros((batch_size, n_agents, config.get('rnn_hidden_dim', 64))),
        rng_key=jax.random.PRNGKey(config.get('seed', 0))
    )
    
    # 创建输入序列
    inputs_sequence = create_rollout_inputs(
        target_velocities, policy_params, gnn_params, {}, seq_length
    )
    
    # 应用梯度检查点到scan步骤
    checkpointed_scan_step = jax.checkpoint(scan_step_fn)
    
    # 执行rollout
    final_carry, outputs = execute_bptt_rollout(
        initial_carry, checkpointed_scan_step, seq_length, inputs_sequence
    )
    
    # 整理输出数据
    rollout_data = {
        'states': outputs.states,                    # [seq_len, batch, n_agents, state_dim]
        'actions': outputs.actions,                  # [seq_len, batch, n_agents, action_dim]
        'cbf_values': outputs.cbf_values,           # [seq_len, batch, n_agents]
        'cbf_derivatives': outputs.cbf_derivatives,  # [seq_len, batch, n_agents]
        'collision_distances': outputs.collision_distances,  # [seq_len, batch, n_agents]
        'safety_violations': outputs.safety_violations      # [seq_len, batch, n_agents]
    }
    
    # 计算指标
    metrics = {
        'total_safety_violations': jnp.sum(outputs.safety_violations),
        'avg_collision_distance': jnp.mean(outputs.collision_distances),
        'min_collision_distance': jnp.min(outputs.collision_distances),
        'avg_cbf_value': jnp.mean(outputs.cbf_values),
        'cbf_std': jnp.std(outputs.cbf_values)
    }
    
    return rollout_data, metrics

# 主要导出函数
__all__ = [
    'ScanCarry',
    'ScanOutput', 
    'create_scan_step_fn',
    'execute_bptt_rollout',
    'rollout_with_gradient_checkpointing',
    'apply_temporal_gradient_decay'
]