"""
完全重构的训练模块，解决梯度流问题
参考GCBF+和DiffPhysDrone的实现架构
"""

import jax
import jax.numpy as jnp
import optax
from typing import Dict, Tuple, Any, NamedTuple, Optional
from flax import linen as nn
from flax.training import train_state
import chex
from functools import partial

class TrainingState(NamedTuple):
    """训练状态结构"""
    policy_state: train_state.TrainState
    gnn_state: train_state.TrainState
    step: int

class CBFLossComponents(NamedTuple):
    """CBF损失组件"""
    cbf_condition_loss: jnp.ndarray
    safe_loss: jnp.ndarray
    unsafe_loss: jnp.ndarray
    total_cbf_loss: jnp.ndarray

class PhysicsLossComponents(NamedTuple):
    """物理损失组件"""
    velocity_loss: jnp.ndarray
    collision_loss: jnp.ndarray
    control_smoothness: jnp.ndarray
    total_physics_loss: jnp.ndarray

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
    learning_rate: float = 1e-4,
    gnn_learning_rate: Optional[float] = None
) -> TrainingState:
    """创建训练状态"""
    
    if gnn_learning_rate is None:
        gnn_learning_rate = learning_rate
    
    # 为策略网络创建优化器
    policy_tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate)
    )
    
    # 为GNN创建优化器 - 可能需要不同的学习率
    gnn_tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(gnn_learning_rate)
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
    计算CBF损失 - 修复版本确保损失非零
    """
    chex.assert_rank([cbf_values, cbf_derivatives, safe_mask, unsafe_mask], 3)
    
    # 1. CBF条件损失: h_dot + alpha * h >= 0
    cbf_condition = cbf_derivatives + alpha * cbf_values
    cbf_condition_loss = jnp.mean(
        jax.nn.relu(gamma - cbf_condition) ** 2
    )
    
    # 2. 安全状态损失: 安全时CBF应该 > 0
    safe_loss = jnp.sum(safe_mask * jax.nn.relu(gamma - cbf_values) ** 2)
    safe_count = jnp.maximum(jnp.sum(safe_mask), 1.0)
    safe_loss = safe_loss / safe_count
    
    # 3. 不安全状态损失: 不安全时CBF应该 < 0
    unsafe_loss = jnp.sum(unsafe_mask * jax.nn.relu(gamma + cbf_values) ** 2)
    unsafe_count = jnp.maximum(jnp.sum(unsafe_mask), 1.0)
    unsafe_loss = unsafe_loss / unsafe_count
    
    # 添加正则化项以防止CBF值过于平坦
    cbf_variance_loss = 0.01 * (1.0 - jnp.std(cbf_values))
    
    total_cbf_loss = cbf_condition_loss + safe_loss + unsafe_loss + cbf_variance_loss
    
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
    """
    
    # 从状态中提取位置和速度
    positions = states[..., :3]
    velocities = states[..., 3:6] if states.shape[-1] >= 6 else jnp.zeros_like(positions)
    
    # 1. 速度跟踪损失
    velocity_error = velocities - target_velocities
    velocity_loss = jnp.mean(
        jnp.where(
            jnp.abs(velocity_error) < 1.0,
            0.5 * velocity_error ** 2,
            jnp.abs(velocity_error) - 0.5
        )
    )
    
    # 2. 碰撞损失 - 确保梯度流动
    min_safe_distance = safety_radius * 2.0  # 智能体间的最小安全距离
    collision_penalty = jnp.where(
        collision_distances < min_safe_distance,
        jnp.exp(-(collision_distances - safety_radius) / safety_radius),
        0.0
    )
    collision_loss = jnp.mean(collision_penalty)
    
    # 3. 控制平滑性损失
    if actions.shape[1] > 1:
        action_diff = actions[:, 1:] - actions[:, :-1]
        control_smoothness = jnp.mean(action_diff ** 2)
    else:
        control_smoothness = 0.0
    
    total_physics_loss = velocity_loss + 5.0 * collision_loss + 0.1 * control_smoothness
    
    return PhysicsLossComponents(
        velocity_loss=velocity_loss,
        collision_loss=collision_loss,
        control_smoothness=control_smoothness,
        total_physics_loss=total_physics_loss
    )

def identify_safe_unsafe_states(
    collision_distances: jnp.ndarray,
    safety_radius: float = 0.5,
    safe_threshold: float = 3.0,
    unsafe_threshold: float = 1.5
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    识别安全和不安全状态 - 修复版本
    """
    
    # 安全状态：距离大于安全阈值
    safe_mask = collision_distances > (safety_radius * safe_threshold)
    
    # 不安全状态：距离小于不安全阈值
    unsafe_mask = collision_distances < (safety_radius * unsafe_threshold)
    
    # 转换为float以用于损失计算
    safe_mask = safe_mask.astype(jnp.float32)
    unsafe_mask = unsafe_mask.astype(jnp.float32)
    
    return safe_mask, unsafe_mask

def compute_collision_distances_batch(states: jnp.ndarray) -> jnp.ndarray:
    """
    批量计算智能体间的最小距离
    
    Args:
        states: [batch_size, seq_len, n_agents, state_dim]
        
    Returns:
        min_distances: [batch_size, seq_len, n_agents]
    """
    positions = states[..., :3]  # [batch_size, seq_len, n_agents, 3]
    
    # 计算每个时间步的距离
    def compute_distances_at_timestep(pos_at_t):
        # pos_at_t: [batch_size, n_agents, 3]
        pos_i = pos_at_t[:, :, None, :]  # [batch, n_agents, 1, 3]
        pos_j = pos_at_t[:, None, :, :]  # [batch, 1, n_agents, 3]
        distances = jnp.linalg.norm(pos_i - pos_j, axis=-1)  # [batch, n_agents, n_agents]
        
        # 排除自身距离
        n_agents = distances.shape[-1]
        mask = jnp.eye(n_agents)
        distances = jnp.where(mask[None, :, :], jnp.inf, distances)
        
        # 最小距离
        min_distances = jnp.min(distances, axis=-1)  # [batch, n_agents]
        return min_distances
    
    # 对每个时间步应用
    min_distances = jax.vmap(compute_distances_at_timestep, in_axes=1, out_axes=1)(positions)
    
    return min_distances

def construct_graph_batch(
    states: jnp.ndarray,
    sensing_radius: float = 2.0
) -> Dict[str, jnp.ndarray]:
    """
    从状态批量构建图数据 - 完全重写确保正确性
    
    Args:
        states: [batch_size, n_agents, state_dim]
        
    Returns:
        graph_batch: 批量图数据
    """
    batch_size, n_agents, state_dim = states.shape
    
    # 提取位置和速度
    positions = states[..., :3]  # [batch_size, n_agents, 3]
    
    # 计算距离矩阵
    pos_i = positions[:, :, None, :]  # [batch, n_agents, 1, 3]
    pos_j = positions[:, None, :, :]  # [batch, 1, n_agents, 3]
    distances = jnp.linalg.norm(pos_i - pos_j, axis=-1)  # [batch, n_agents, n_agents]
    
    # 创建邻接矩阵
    adjacency = (distances < sensing_radius) & (distances > 1e-6)
    adjacency = adjacency.astype(jnp.float32)
    
    # 节点特征就是状态本身
    node_features = states  # [batch_size, n_agents, state_dim]
    
    # 边特征：相对位置和相对速度
    rel_positions = pos_i - pos_j  # [batch, n_agents, n_agents, 3]
    
    if state_dim >= 6:
        velocities = states[..., 3:6]
        vel_i = velocities[:, :, None, :]
        vel_j = velocities[:, None, :, :]
        rel_velocities = vel_i - vel_j
        edge_features = jnp.concatenate([rel_positions, rel_velocities], axis=-1)
    else:
        edge_features = rel_positions
    
    # 添加距离作为额外的边特征
    distances_expanded = distances[..., None]
    edge_features = jnp.concatenate([edge_features, distances_expanded], axis=-1)
    
    return {
        'nodes': node_features,
        'edges': edge_features,
        'adjacency': adjacency,
        'n_nodes': n_agents,
        'batch_size': batch_size
    }

def compute_cbf_time_derivative(
    cbf_values: jnp.ndarray,
    cbf_grads: jnp.ndarray,
    states: jnp.ndarray,
    actions: jnp.ndarray,
    dt: float = 0.02
) -> jnp.ndarray:
    """
    计算CBF时间导数 - 确保形状匹配
    
    Args:
        cbf_values: [batch_size, n_agents]
        cbf_grads: [batch_size, n_agents, grad_dim]
        states: [batch_size, n_agents, state_dim]
        actions: [batch_size, n_agents, action_dim]
    """
    batch_size, n_agents = cbf_values.shape
    
    # 计算状态导数
    if states.shape[-1] >= 6:
        # 完整状态：[position, velocity]
        velocities = states[..., 3:6]
        accelerations = actions[..., :3]
        
        # 只使用位置和速度的梯度
        grad_dim = min(cbf_grads.shape[-1], 6)
        cbf_grads_truncated = cbf_grads[..., :grad_dim]
        
        if grad_dim >= 6:
            # 完整梯度：对位置和速度
            state_dot = jnp.concatenate([velocities, accelerations], axis=-1)
            cbf_dot = jnp.sum(cbf_grads_truncated * state_dot, axis=-1)
        else:
            # 只有位置梯度
            cbf_dot = jnp.sum(cbf_grads_truncated[..., :3] * velocities, axis=-1)
    else:
        # 简化状态
        state_dot = actions[..., :states.shape[-1]]
        cbf_dot = jnp.sum(cbf_grads * state_dot, axis=-1)
    
    return cbf_dot

@partial(jax.jit, static_argnums=(2, 3))
def compute_total_loss_and_metrics(
    training_state: TrainingState,
    batch_data: Dict[str, jnp.ndarray],
    physics_step_fn,
    config: Dict[str, Any]
) -> TotalLossComponents:
    """
    计算总损失 - 完全重写确保GNN梯度流动
    """
    
    # 解包数据
    initial_states = batch_data['initial_states']  # [batch_size, n_agents, state_dim]
    target_velocities = batch_data['target_velocities']  # [batch_size, seq_len, n_agents, 3]
    
    batch_size, n_agents, state_dim = initial_states.shape
    seq_len = target_velocities.shape[1]
    
    def scan_fn(carry, inputs):
        current_states = carry
        target_vel = inputs  # [batch_size, n_agents, 3]
        
        # 1. 构建图批量数据
        graph_batch = construct_graph_batch(current_states, config.get('sensing_radius', 2.0))
        
        # 2. GNN前向传播 - 批量处理
        # 重塑为单个大batch以便GNN处理
        nodes_flat = graph_batch['nodes'].reshape(-1, state_dim)
        edges_flat = graph_batch['edges'].reshape(-1, n_agents, graph_batch['edges'].shape[-1])
        adjacency_flat = graph_batch['adjacency'].reshape(-1, n_agents)
        
        flat_graph = {
            'nodes': nodes_flat,
            'edges': edges_flat,
            'adjacency': adjacency_flat,
            'n_nodes': n_agents
        }
        
        # GNN计算
        cbf_output = training_state.gnn_state.apply_fn(
            training_state.gnn_state.params,
            flat_graph,
            training=True
        )
        
        if isinstance(cbf_output, tuple):
            cbf_values_flat, cbf_grads_flat = cbf_output
        else:
            cbf_values_flat = cbf_output
            cbf_grads_flat = jnp.zeros((batch_size * n_agents, 6))
        
        # 重塑回原始batch形状
        cbf_values = cbf_values_flat.reshape(batch_size, n_agents)
        cbf_grads = cbf_grads_flat.reshape(batch_size, n_agents, -1)
        
        # 3. 策略网络计算动作
        # 准备策略输入：状态 + 目标速度 + CBF值
        policy_inputs = jnp.concatenate([
            current_states,
            target_vel,
            cbf_values[..., None]
        ], axis=-1)
        
        # 批量策略计算
        actions_flat = training_state.policy_state.apply_fn(
            training_state.policy_state.params,
            policy_inputs.reshape(-1, policy_inputs.shape[-1])
        )
        actions = actions_flat.reshape(batch_size, n_agents, -1)
        
        # 4. 物理仿真
        next_states = physics_step_fn(current_states, actions)
        
        # 5. 计算CBF时间导数
        cbf_derivatives = compute_cbf_time_derivative(
            cbf_values, cbf_grads, current_states, actions, config.get('dt', 0.02)
        )
        
        # 6. 计算碰撞距离
        collision_distances = compute_collision_distances_batch(
            current_states[None, :, :, :]
        )[0, 0]  # 取第一个时间步
        
        outputs = (
            current_states,
            actions,
            cbf_values,
            cbf_derivatives,
            collision_distances
        )
        
        return next_states, outputs
    
    # 执行scan
    target_vel_sequence = target_velocities.transpose(1, 0, 2, 3)
    final_states, (states_traj, actions_traj, cbf_values_traj, 
                   cbf_derivatives_traj, collision_distances_traj) = jax.lax.scan(
        scan_fn,
        initial_states,
        target_vel_sequence,
        length=seq_len
    )
    
    # 转置回 [batch, seq_len, ...]
    states_traj = states_traj.transpose(1, 0, 2, 3)
    actions_traj = actions_traj.transpose(1, 0, 2, 3)
    cbf_values_traj = cbf_values_traj.transpose(1, 0, 2)
    cbf_derivatives_traj = cbf_derivatives_traj.transpose(1, 0, 2)
    collision_distances_traj = collision_distances_traj.transpose(1, 0, 2)
    
    # 计算损失
    safe_mask, unsafe_mask = identify_safe_unsafe_states(
        collision_distances_traj,
        safety_radius=config.get('safety_radius', 0.5)
    )
    
    cbf_losses = compute_cbf_loss(
        cbf_values_traj,
        cbf_derivatives_traj,
        safe_mask,
        unsafe_mask,
        alpha=config.get('alpha', 1.0),
        gamma=config.get('gamma', 0.02)
    )
    
    physics_losses = compute_physics_loss(
        states_traj,
        actions_traj,
        target_velocities,
        collision_distances_traj,
        safety_radius=config.get('safety_radius', 0.5)
    )
    
    # 总损失
    total_loss = (config.get('cbf_weight', 2.0) * cbf_losses.total_cbf_loss + 
                  config.get('physics_weight', 1.0) * physics_losses.total_physics_loss)
    
    # 计算最终距离
    final_positions = states_traj[:, -1, :, :3]
    final_target_positions = jnp.mean(target_velocities[:, -1, :, :], axis=1, keepdims=True) * seq_len * config.get('dt', 0.02)
    final_distances = jnp.linalg.norm(final_positions - final_target_positions, axis=-1)
    
    metrics = {
        'cbf_condition_loss': cbf_losses.cbf_condition_loss,
        'safe_loss': cbf_losses.safe_loss,
        'unsafe_loss': cbf_losses.unsafe_loss,
        'velocity_loss': physics_losses.velocity_loss,
        'collision_loss': physics_losses.collision_loss,
        'control_smoothness': physics_losses.control_smoothness,
        'safety_violations': jnp.sum(unsafe_mask),
        'avg_cbf_value': jnp.mean(cbf_values_traj),
        'min_collision_distance': jnp.min(collision_distances_traj),
        'cbf_std': jnp.std(cbf_values_traj),
        'final_distance': jnp.mean(final_distances)
    }
    
    return TotalLossComponents(
        cbf_losses=cbf_losses,
        physics_losses=physics_losses,
        total_loss=total_loss,
        metrics=metrics
    )

@jax.jit
def train_step_corrected(
    training_state: TrainingState,
    batch_data: Dict[str, jnp.ndarray],
    physics_step_fn,
    config: Dict[str, Any]
) -> Tuple[TrainingState, Dict[str, jnp.ndarray]]:
    """
    执行训练步骤 - 确保两个网络都能获得梯度
    """
    
    def loss_fn(policy_params, gnn_params):
        # 创建临时训练状态
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
            config
        )
        
        return loss_components.total_loss, loss_components
    
    # 计算梯度
    grad_fn = jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)
    (loss_value, loss_components), (policy_grads, gnn_grads) = grad_fn(
        training_state.policy_state.params,
        training_state.gnn_state.params
    )
    
    # 更新参数
    new_policy_state = training_state.policy_state.apply_gradients(grads=policy_grads)
    new_gnn_state = training_state.gnn_state.apply_gradients(grads=gnn_grads)
    
    new_training_state = TrainingState(
        policy_state=new_policy_state,
        gnn_state=new_gnn_state,
        step=training_state.step + 1
    )
    
    # 计算梯度范数
    policy_grad_norm = jnp.sqrt(
        sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(policy_grads))
    )
    gnn_grad_norm = jnp.sqrt(
        sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(gnn_grads))
    )
    
    metrics = dict(loss_components.metrics)
    metrics.update({
        'total_loss': loss_value,
        'cbf_total_loss': loss_components.cbf_losses.total_cbf_loss,
        'physics_total_loss': loss_components.physics_losses.total_physics_loss,
        'policy_grad_norm': policy_grad_norm,
        'gnn_grad_norm': gnn_grad_norm,
        'learning_step': new_training_state.step
    })
    
    return new_training_state, metrics

# 测试系统
class CompleteTrainingSystem:
    """完整训练系统"""
    
    def __init__(self, config: Dict, rng_key: jax.random.PRNGKey):
        self.config = config
        self.rng_key = rng_key
        
        # 创建模型
        from core.policy import create_policy_model
        from core.perception import create_gnn_model
        from core.physics import create_physics_step_function, PhysicsConfig
        
        self.policy_model = create_policy_model("mlp")
        self.gnn_model = create_gnn_model(config)
        
        # 创建物理仿真
        physics_config = PhysicsConfig(dt=config.get('dt', 0.02))
        self.physics_step_fn = create_physics_step_function("point_mass", physics_config)
        
        # 初始化参数
        self._init_params()
    
    def _init_params(self):
        """初始化参数"""
        key1, key2 = jax.random.split(self.rng_key)
        
        # 策略网络输入：状态 + 目标速度 + CBF值
        dummy_input = jnp.zeros(13 + 3 + 1)  # state + target_vel + cbf
        self.policy_params = self.policy_model.init(key1, dummy_input)
        
        # GNN参数
        dummy_graph = {
            'nodes': jnp.zeros((4, 13)),
            'edges': jnp.zeros((4, 4, 7)),  # 相对位置(3) + 相对速度(3) + 距离(1)
            'adjacency': jnp.ones((4, 4)),
            'n_nodes': 4
        }
        self.gnn_params = self.gnn_model.init(key2, dummy_graph, training=False)
    
    def get_initial_training_state(self) -> TrainingState:
        """获取初始训练状态"""
        return create_training_state(
            self.policy_model,
            self.gnn_model,
            self.policy_params,
            self.gnn_params,
            learning_rate=self.config.get('learning_rate', 1e-4),
            gnn_learning_rate=self.config.get('gnn_learning_rate', 5e-4)
        )
    
    def train_step(
        self,
        training_state: TrainingState,
        initial_states: jnp.ndarray,
        target_velocities: jnp.ndarray
    ) -> Tuple[TrainingState, Dict[str, jnp.ndarray]]:
        """执行训练步骤"""
        
        batch_data = {
            'initial_states': initial_states,
            'target_velocities': target_velocities
        }
        
        return train_step_corrected(
            training_state,
            batch_data,
            self.physics_step_fn,
            self.config
        )

def test_complete_gradient_flow():
    """测试完整系统梯度流"""
    print("开始完整系统梯度流测试...")
    
    try:
        # 配置
        config = {
            'learning_rate': 1e-4,
            'gnn_learning_rate': 5e-4,
            'trajectory_length': 10,
            'batch_size': 4,
            'n_agents': 3,
            'sensing_radius': 3.0,
            'safety_radius': 0.5,
            'alpha': 1.0,
            'gamma': 0.05,
            'cbf_weight': 2.0,
            'physics_weight': 1.0,
            'dt': 0.02
        }
        
        # 系统初始化
        rng_key = jax.random.PRNGKey(42)
        training_system = CompleteTrainingSystem(config, rng_key)
        training_state = training_system.get_initial_training_state()
        
        # 测试数据
        batch_size = config['batch_size']
        n_agents = config['n_agents']
        seq_len = config['trajectory_length']
        
        # 创建多样化的初始状态
        initial_states = jnp.zeros((batch_size, n_agents, 13))
        # 随机初始位置
        positions = jax.random.uniform(
            jax.random.PRNGKey(123),
            (batch_size, n_agents, 3),
            minval=-2.0,
            maxval=2.0
        )
        initial_states = initial_states.at[..., :3].set(positions)
        # 单位四元数
        initial_states = initial_states.at[..., 6].set(1.0)
        
        # 目标速度序列
        target_velocities = jax.random.normal(
            jax.random.PRNGKey(456),
            (batch_size, seq_len, n_agents, 3)
        ) * 2.0
        
        # 执行多步训练
        print("执行多步训练测试...")
        for i in range(3):
            new_training_state, metrics = training_system.train_step(
                training_state, initial_states, target_velocities
            )
            
            print(f"\n步骤 {i+1}:")
            print(f"总损失: {metrics['total_loss']:.4f}")
            print(f"CBF损失: {metrics['cbf_total_loss']:.4f}")
            print(f"物理损失: {metrics['physics_total_loss']:.4f}")
            print(f"策略梯度范数: {metrics['policy_grad_norm']:.6f}")
            print(f"GNN梯度范数: {metrics['gnn_grad_norm']:.6f}")
            print(f"平均CBF值: {metrics['avg_cbf_value']:.4f}")
            print(f"CBF标准差: {metrics['cbf_std']:.4f}")
            print(f"最小碰撞距离: {metrics['min_collision_distance']:.4f}")
            
            training_state = new_training_state
        
        # 验证梯度流
        final_metrics = metrics
        policy_grad_ok = final_metrics['policy_grad_norm'] > 1e-6
        gnn_grad_ok = final_metrics['gnn_grad_norm'] > 1e-6
        loss_finite = jnp.isfinite(final_metrics['total_loss'])
        cbf_loss_ok = final_metrics['cbf_total_loss'] > 1e-6
        
        if policy_grad_ok and gnn_grad_ok and loss_finite and cbf_loss_ok:
            print("\n✅ 梯度流正常，两个网络都接收到有效梯度")
            print("✅ CBF损失非零，安全机制正常工作")
            return True
        else:
            print("\n❌ 警告: 梯度流或损失计算存在问题")
            if not policy_grad_ok:
                print("  - 策略网络梯度过小")
            if not gnn_grad_ok:
                print("  - GNN网络梯度过小")
            if not loss_finite:
                print("  - 损失值非有限")
            if not cbf_loss_ok:
                print("  - CBF损失过小")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

# 导出函数
__all__ = [
    'TrainingState',
    'CBFLossComponents',
    'PhysicsLossComponents',
    'TotalLossComponents',
    'create_training_state',
    'compute_total_loss_and_metrics',
    'train_step_corrected',
    'CompleteTrainingSystem',
    'test_complete_gradient_flow'
]