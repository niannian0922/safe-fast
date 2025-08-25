"""
完全重构的训练模块，解决梯度流问题
参考GCBF+和DiffPhysDrone的实现架构
"""

import jax
import jax.numpy as jnp
import optax
from typing import Dict, Tuple, Any, NamedTuple
from flax import linen as nn
from flax.training import train_state
import chex

class TrainingState(NamedTuple):
    """训练状态结构"""
    policy_state: train_state.TrainState
    gnn_state: train_state.TrainState
    step: int

class CBFLossComponents(NamedTuple):
    """CBF损失组件"""
    cbf_condition_loss: jnp.ndarray  # CBF条件损失
    safe_loss: jnp.ndarray           # 安全状态损失  
    unsafe_loss: jnp.ndarray         # 不安全状态损失
    total_cbf_loss: jnp.ndarray      # CBF总损失

class PhysicsLossComponents(NamedTuple):
    """物理损失组件"""
    velocity_loss: jnp.ndarray       # 速度跟踪损失
    collision_loss: jnp.ndarray      # 碰撞损失
    control_smoothness: jnp.ndarray  # 控制平滑性损失
    total_physics_loss: jnp.ndarray  # 物理总损失

class TotalLossComponents(NamedTuple):
    """总损失组件"""
    cbf_losses: CBFLossComponents
    physics_losses: PhysicsLossComponents
    total_loss: jnp.ndarray
    metrics: Dict[str, jnp.ndarray]

def create_training_state(
    policy_model: nn.Module,
    gnn_model: nn.Module,
    policy_params: Dict,
    gnn_params: Dict,
    learning_rate: float = 1e-4
) -> TrainingState:
    """创建训练状态"""
    
    # 为策略网络创建优化器
    policy_tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate)
    )
    
    # 为GNN创建优化器  
    gnn_tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate)
    )
    
    policy_state = train_state.TrainState.create(
        apply_fn=policy_model.apply,
        params=policy_params,
        tx=policy_tx
    )
    
    gnn_state = train_state.TrainState.create(
        apply_fn=gnn_model.apply,
        params=gnn_params,
        tx=gnn_tx
    )
    
    return TrainingState(
        policy_state=policy_state,
        gnn_state=gnn_state,
        step=0
    )

def compute_cbf_loss(
    cbf_values: jnp.ndarray,
    cbf_derivatives: jnp.ndarray,
    safe_mask: jnp.ndarray,
    unsafe_mask: jnp.ndarray,
    alpha: float = 1.0,
    gamma: float = 0.02
) -> CBFLossComponents:
    """
    计算CBF损失 - 参考GCBF+论文的损失函数设计
    
    Args:
        cbf_values: CBF值 [batch_size, seq_len, n_agents]
        cbf_derivatives: CBF时间导数 [batch_size, seq_len, n_agents]  
        safe_mask: 安全状态掩码
        unsafe_mask: 不安全状态掩码
        alpha: CBF条件参数
        gamma: 严格不等式参数
    """
    chex.assert_rank([cbf_values, cbf_derivatives, safe_mask, unsafe_mask], 3)
    
    # 1. CBF条件损失: h_dot + alpha * h >= 0
    cbf_condition = cbf_derivatives + alpha * cbf_values
    cbf_condition_loss = jnp.mean(
        jax.nn.relu(gamma - cbf_condition) ** 2
    )
    
    # 2. 安全状态损失: 安全时CBF应该 > 0
    safe_loss = jnp.mean(
        safe_mask * jax.nn.relu(gamma - cbf_values) ** 2
    )
    
    # 3. 不安全状态损失: 不安全时CBF应该 < 0  
    unsafe_loss = jnp.mean(
        unsafe_mask * jax.nn.relu(gamma + cbf_values) ** 2
    )
    
    total_cbf_loss = cbf_condition_loss + safe_loss + unsafe_loss
    
    return CBFLossComponents(
        cbf_condition_loss=cbf_condition_loss,
        safe_loss=safe_loss,
        unsafe_loss=unsafe_loss,
        total_cbf_loss=total_cbf_loss
    )

def compute_physics_loss(
    states: jnp.ndarray,
    actions: jnp.ndarray,
    target_velocities: jnp.ndarray,
    collision_distances: jnp.ndarray,
    safety_radius: float = 0.5
) -> PhysicsLossComponents:
    """
    计算物理损失 - 参考DiffPhysDrone的损失设计
    
    Args:
        states: 状态轨迹 [batch_size, seq_len, n_agents, state_dim]
        actions: 动作轨迹 [batch_size, seq_len, n_agents, action_dim]
        target_velocities: 目标速度 [batch_size, seq_len, n_agents, 3]
        collision_distances: 碰撞距离 [batch_size, seq_len, n_agents]
    """
    
    # 从状态中提取位置和速度
    positions = states[..., :3]  # [batch_size, seq_len, n_agents, 3]
    velocities = states[..., 3:6] if states.shape[-1] >= 6 else jnp.zeros_like(positions)
    
    # 1. 速度跟踪损失 (参考DiffPhysDrone的Smooth L1 loss)
    velocity_error = velocities - target_velocities
    velocity_loss = jnp.mean(
        jnp.where(
            jnp.abs(velocity_error) < 1.0,
            0.5 * velocity_error ** 2,
            jnp.abs(velocity_error) - 0.5
        )
    )
    
    # 2. 碰撞损失 (指数惩罚接近)
    collision_loss = jnp.mean(
        jnp.exp(-collision_distances / safety_radius) * 
        jnp.maximum(0, safety_radius - collision_distances) ** 2
    )
    
    # 3. 控制平滑性损失
    action_diff = actions[:, 1:] - actions[:, :-1]
    control_smoothness = jnp.mean(action_diff ** 2)
    
    total_physics_loss = velocity_loss + collision_loss + 0.1 * control_smoothness
    
    return PhysicsLossComponents(
        velocity_loss=velocity_loss,
        collision_loss=collision_loss,
        control_smoothness=control_smoothness,
        total_physics_loss=total_physics_loss
    )

def identify_safe_unsafe_states(
    positions: jnp.ndarray,
    collision_distances: jnp.ndarray,
    safety_radius: float = 0.5,
    danger_threshold: float = 0.8
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    识别安全和不安全状态
    
    Args:
        positions: 位置 [batch_size, seq_len, n_agents, 3]
        collision_distances: 最近碰撞距离 [batch_size, seq_len, n_agents]
        safety_radius: 安全半径
        danger_threshold: 危险阈值倍数
        
    Returns:
        safe_mask: 安全状态掩码
        unsafe_mask: 不安全状态掩码
    """
    
    # 安全状态：距离障碍物/其他智能体足够远
    safe_mask = collision_distances > (safety_radius * danger_threshold)
    
    # 不安全状态：距离障碍物/其他智能体太近
    unsafe_mask = collision_distances < safety_radius
    
    return safe_mask, unsafe_mask

def compute_total_loss_and_metrics(
    training_state: TrainingState,
    batch_data: Dict[str, jnp.ndarray],
    physics_step_fn,
    alpha: float = 1.0,
    cbf_weight: float = 1.0,
    physics_weight: float = 1.0
) -> TotalLossComponents:
    """
    计算总损失和指标 - 这是核心函数，确保梯度正确流动
    """
    
    # 解包输入数据
    initial_states = batch_data['states'][:, 0]  # [batch_size, n_agents, state_dim]
    target_velocities = batch_data['target_velocities']  # [batch_size, seq_len, n_agents, 3]
    graph_data = batch_data['graph_data']  # 图结构数据
    
    batch_size, seq_len = target_velocities.shape[:2]
    n_agents = initial_states.shape[1]
    
    # 前向传播：策略网络 + 物理仿真 + GNN
    def rollout_step(carry, t):
        states, rng_key = carry
        rng_key, subkey = jax.random.split(rng_key)
        
        # 1. 策略网络产生动作
        actions = training_state.policy_state.apply_fn(
            training_state.policy_state.params,
            states,
            target_velocities[:, t]  # 当前时刻目标速度
        )
        
        # 2. 物理仿真更新状态  
        next_states = physics_step_fn(states, actions)
        
        # 3. 计算当前状态的图数据和CBF值
        current_graph = construct_graph_from_states(next_states, graph_data)
        cbf_values, cbf_grads = training_state.gnn_state.apply_fn(
            training_state.gnn_state.params,
            current_graph
        )
        
        # 4. 计算CBF时间导数 (这是关键！)
        cbf_derivatives = compute_cbf_time_derivative(
            cbf_values, cbf_grads, next_states, actions
        )
        
        outputs = {
            'states': next_states,
            'actions': actions, 
            'cbf_values': cbf_values,
            'cbf_derivatives': cbf_derivatives
        }
        
        return (next_states, rng_key), outputs
    
    # 执行时间展开
    rng_key = jax.random.PRNGKey(0)
    _, rollout_outputs = jax.lax.scan(
        rollout_step,
        (initial_states, rng_key),
        jnp.arange(seq_len)
    )
    
    # 提取轨迹数据
    states_trajectory = rollout_outputs['states']  # [seq_len, batch_size, n_agents, state_dim]
    actions_trajectory = rollout_outputs['actions']  # [seq_len, batch_size, n_agents, action_dim]  
    cbf_values_trajectory = rollout_outputs['cbf_values']  # [seq_len, batch_size, n_agents]
    cbf_derivatives_trajectory = rollout_outputs['cbf_derivatives']  # [seq_len, batch_size, n_agents]
    
    # 转换维度为 [batch_size, seq_len, n_agents, ...]
    states_trajectory = jnp.transpose(states_trajectory, (1, 0, 2, 3))
    actions_trajectory = jnp.transpose(actions_trajectory, (1, 0, 2, 3))
    cbf_values_trajectory = jnp.transpose(cbf_values_trajectory, (1, 0, 2))
    cbf_derivatives_trajectory = jnp.transpose(cbf_derivatives_trajectory, (1, 0, 2))
    
    # 计算碰撞距离
    collision_distances = compute_collision_distances(states_trajectory)
    
    # 识别安全/不安全状态
    safe_mask, unsafe_mask = identify_safe_unsafe_states(
        states_trajectory[..., :3], collision_distances
    )
    
    # 计算CBF损失
    cbf_losses = compute_cbf_loss(
        cbf_values_trajectory,
        cbf_derivatives_trajectory, 
        safe_mask,
        unsafe_mask,
        alpha=alpha
    )
    
    # 计算物理损失
    physics_losses = compute_physics_loss(
        states_trajectory,
        actions_trajectory,
        target_velocities,
        collision_distances
    )
    
    # 总损失
    total_loss = cbf_weight * cbf_losses.total_cbf_loss + physics_weight * physics_losses.total_physics_loss
    
    # 计算指标
    metrics = {
        'cbf_condition_loss': cbf_losses.cbf_condition_loss,
        'safe_loss': cbf_losses.safe_loss,
        'unsafe_loss': cbf_losses.unsafe_loss,
        'velocity_loss': physics_losses.velocity_loss,
        'collision_loss': physics_losses.collision_loss,
        'control_smoothness': physics_losses.control_smoothness,
        'safety_violations': jnp.sum(unsafe_mask),
        'avg_cbf_value': jnp.mean(cbf_values_trajectory),
        'min_collision_distance': jnp.min(collision_distances)
    }
    
    return TotalLossComponents(
        cbf_losses=cbf_losses,
        physics_losses=physics_losses,
        total_loss=total_loss,
        metrics=metrics
    )

def construct_graph_from_states(states: jnp.ndarray, graph_template: Dict) -> Dict:
    """从状态构建图数据"""
    # 这需要根据具体的图结构实现
    # 暂时返回模板结构
    return graph_template

def compute_cbf_time_derivative(
    cbf_values: jnp.ndarray,
    cbf_grads: jnp.ndarray, 
    states: jnp.ndarray,
    actions: jnp.ndarray
) -> jnp.ndarray:
    """
    计算CBF时间导数: dh/dt = ∇h · f(x,u)
    这是确保梯度流向GNN的关键函数
    """
    # 从状态计算系统动力学 f(x,u)
    state_derivatives = compute_state_derivatives(states, actions)
    
    # CBF时间导数 = ∇h · ẋ
    cbf_dot = jnp.sum(cbf_grads * state_derivatives, axis=-1)
    
    return cbf_dot

def compute_state_derivatives(states: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
    """计算状态导数 ẋ = f(x,u)"""
    # 简单的点质量模型：ẋ = [v, a]
    positions = states[..., :3]
    velocities = states[..., 3:6] if states.shape[-1] >= 6 else jnp.zeros_like(positions)
    
    # 位置导数 = 速度
    position_derivatives = velocities
    
    # 速度导数 = 动作（加速度）
    velocity_derivatives = actions[..., :3]
    
    return jnp.concatenate([position_derivatives, velocity_derivatives], axis=-1)

def compute_collision_distances(states: jnp.ndarray) -> jnp.ndarray:
    """计算智能体间的最小距离"""
    positions = states[..., :3]  # [batch_size, seq_len, n_agents, 3]
    
    # 计算所有智能体对之间的距离
    positions_expanded_i = jnp.expand_dims(positions, axis=3)  # [batch, seq, n_agents, 1, 3]
    positions_expanded_j = jnp.expand_dims(positions, axis=2)  # [batch, seq, 1, n_agents, 3]
    
    distances = jnp.linalg.norm(positions_expanded_i - positions_expanded_j, axis=-1)  # [batch, seq, n_agents, n_agents]
    
    # 排除自己与自己的距离（设为无穷大）
    n_agents = distances.shape[-1]
    eye_mask = jnp.eye(n_agents)
    distances = jnp.where(eye_mask, jnp.inf, distances)
    
    # 返回每个智能体到最近邻居的距离
    min_distances = jnp.min(distances, axis=-1)  # [batch, seq, n_agents]
    
    return min_distances

@jax.jit
def train_step(
    training_state: TrainingState,
    batch_data: Dict[str, jnp.ndarray], 
    physics_step_fn,
    config: Dict[str, Any]
) -> Tuple[TrainingState, Dict[str, jnp.ndarray]]:
    """
    执行一步训练 - 确保梯度正确计算和传播
    """
    
    def loss_fn(policy_params, gnn_params):
        # 创建临时训练状态用于损失计算
        temp_training_state = TrainingState(
            policy_state=training_state.policy_state.replace(params=policy_params),
            gnn_state=training_state.gnn_state.replace(params=gnn_params),
            step=training_state.step
        )
        
        # 计算损失
        loss_components = compute_total_loss_and_metrics(
            temp_training_state,
            batch_data,
            physics_step_fn,
            alpha=config.get('alpha', 1.0),
            cbf_weight=config.get('cbf_weight', 1.0),
            physics_weight=config.get('physics_weight', 1.0)
        )
        
        return loss_components.total_loss, loss_components
    
    # 计算梯度 - 对两个网络的参数分别求梯度
    grad_fn = jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)
    (loss_value, loss_components), (policy_grads, gnn_grads) = grad_fn(
        training_state.policy_state.params,
        training_state.gnn_state.params
    )
    
    # 更新策略网络
    new_policy_state = training_state.policy_state.apply_gradients(
        grads=policy_grads
    )
    
    # 更新GNN网络
    new_gnn_state = training_state.gnn_state.apply_gradients(
        grads=gnn_grads
    )
    
    # 创建新的训练状态
    new_training_state = TrainingState(
        policy_state=new_policy_state,
        gnn_state=new_gnn_state,
        step=training_state.step + 1
    )
    
    # 计算梯度范数用于监控
    policy_grad_norm = jnp.sqrt(
        sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(policy_grads))
    )
    gnn_grad_norm = jnp.sqrt(
        sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(gnn_grads))
    )
    
    # 添加梯度范数到指标
    metrics = dict(loss_components.metrics)
    metrics.update({
        'total_loss': loss_value,
        'policy_grad_norm': policy_grad_norm,
        'gnn_grad_norm': gnn_grad_norm,
        'learning_step': new_training_state.step
    })
    
    return new_training_state, metrics

# 导出主要函数
__all__ = [
    'TrainingState',
    'CBFLossComponents', 
    'PhysicsLossComponents',
    'TotalLossComponents',
    'create_training_state',
    'compute_total_loss_and_metrics',
    'train_step'
]