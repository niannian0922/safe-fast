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
    识别安全和不安全状态 - 修复版本
    
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

def construct_graph_from_states_corrected(
    states: jnp.ndarray,  # [batch_size, n_agents, state_dim]
    sensing_radius: float = 2.0,
    max_neighbors: int = 10
) -> Dict[str, jnp.ndarray]:
    """
    修复版本：正确地从状态构建图数据
    这是GCBF+的核心 - 必须正确实现
    """
    batch_size, n_agents, state_dim = states.shape
    positions = states[..., :3]  # [batch_size, n_agents, 3]
    
    def build_single_graph(batch_states):
        """为单个batch构建图"""
        batch_positions = batch_states[:, :3]  # [n_agents, 3]
        
        # 计算所有智能体对之间的距离
        pos_i = batch_positions[:, None, :]  # [n_agents, 1, 3]
        pos_j = batch_positions[None, :, :]  # [1, n_agents, 3]
        distances = jnp.linalg.norm(pos_i - pos_j, axis=-1)  # [n_agents, n_agents]
        
        # 创建邻接矩阵
        adjacency = (distances < sensing_radius) & (distances > 0)
        
        # 限制邻居数量
        def limit_neighbors_per_agent(agent_idx):
            agent_distances = distances[agent_idx]
            agent_adjacency = adjacency[agent_idx]
            
            # 找到最近的max_neighbors个邻居
            neighbor_indices = jnp.argsort(agent_distances)
            valid_neighbors = neighbor_indices[1:max_neighbors+1]  # 排除自己
            
            limited_adj = jnp.zeros(n_agents, dtype=bool)
            limited_adj = limited_adj.at[valid_neighbors].set(
                agent_adjacency[valid_neighbors]
            )
            return limited_adj
        
        # 对每个智能体应用邻居限制
        adjacency = jax.vmap(limit_neighbors_per_agent)(jnp.arange(n_agents))
        
        # 构建节点特征 
        node_features = batch_states  # [n_agents, state_dim]
        
        # 构建边特征 (相对位置和速度)
        rel_positions = pos_i - pos_j  # [n_agents, n_agents, 3]
        
        if state_dim >= 6:
            velocities = batch_states[:, 3:6]
            vel_i = velocities[:, None, :]
            vel_j = velocities[None, :, :]
            rel_velocities = vel_i - vel_j  # [n_agents, n_agents, 3]
            edge_features = jnp.concatenate([rel_positions, rel_velocities], axis=-1)
        else:
            edge_features = rel_positions
            
        return {
            'nodes': node_features,
            'edges': edge_features,
            'adjacency': adjacency.astype(jnp.float32),
            'n_nodes': n_agents
        }
    
    # 为每个batch构建图
    batch_graphs = jax.vmap(build_single_graph)(states)
    
    return batch_graphs

def compute_cbf_time_derivative_corrected(
    cbf_values: jnp.ndarray,      # [batch_size, n_agents]
    cbf_grads: jnp.ndarray,       # [batch_size, n_agents, state_dim]
    states: jnp.ndarray,          # [batch_size, n_agents, state_dim] 
    actions: jnp.ndarray,         # [batch_size, n_agents, action_dim]
    dt: float = 0.02
) -> jnp.ndarray:
    """
    修复版本：正确计算CBF时间导数
    这是连接CBF和动力学的关键 - dh/dt = ∇h · f(x,u)
    """
    
    # 计算系统动力学 f(x,u) - 点质量模型
    def compute_dynamics(state, action):
        if state.shape[-1] >= 6:
            # 完整动力学: [position, velocity] -> [velocity, acceleration]
            position = state[:3]
            velocity = state[3:6]
            acceleration = action[:3]  # 动作为期望加速度
            
            state_dot = jnp.concatenate([velocity, acceleration])
        else:
            # 简化动力学: position -> velocity
            state_dot = action[:3]
            
        return state_dot
    
    # 对每个智能体计算状态导数
    state_derivatives = jax.vmap(jax.vmap(compute_dynamics))(states, actions)
    
    # CBF时间导数: dh/dt = ∇h · ẋ
    # cbf_grads: [batch_size, n_agents, state_dim]
    # state_derivatives: [batch_size, n_agents, state_dim]
    cbf_dot = jnp.sum(cbf_grads * state_derivatives, axis=-1)  # [batch_size, n_agents]
    
    return cbf_dot

def compute_total_loss_and_metrics_corrected(
    training_state: TrainingState,
    batch_data: Dict[str, jnp.ndarray],
    physics_step_fn,
    config: Dict[str, Any]
) -> TotalLossComponents:
    """
    修复版本：计算总损失和指标 - 确保梯度正确流动
    """
    
    # 解包输入数据
    initial_states = batch_data['initial_states']  # [batch_size, n_agents, state_dim]
    target_velocities = batch_data['target_velocities']  # [batch_size, seq_len, n_agents, 3]
    
    batch_size, n_agents, state_dim = initial_states.shape
    seq_len = target_velocities.shape[1]
    
    # === 核心修复：正确的时间展开与梯度流 ===
    def single_step(carry, step_input):
        """单步前向传播 - 确保梯度流通"""
        current_states, step = carry
        target_vel = step_input  # [batch_size, n_agents, 3]
        
        # 1. 构建图数据 - 修复版本
        graph_batch = construct_graph_from_states_corrected(
            current_states, 
            sensing_radius=config.get('sensing_radius', 2.0)
        )
        
        # 2. GNN计算CBF - 确保梯度流
        def compute_cbf_for_single_graph(single_graph):
            return training_state.gnn_state.apply_fn(
                training_state.gnn_state.params, 
                single_graph,
                training=True
            )
        
        # 批量处理图数据
        cbf_outputs = jax.vmap(compute_cbf_for_single_graph)(graph_batch)
        cbf_values = cbf_outputs[0] if isinstance(cbf_outputs, tuple) else cbf_outputs
        cbf_grads = cbf_outputs[1] if isinstance(cbf_outputs, tuple) else jnp.zeros_like(current_states)
        
        # 3. 策略网络计算动作
        def compute_action_for_batch(states, target_vel):
            def compute_single_agent_action(state, target):
                return training_state.policy_state.apply_fn(
                    training_state.policy_state.params,
                    jnp.concatenate([state, target])
                )
            
            return jax.vmap(compute_single_agent_action)(states, target_vel)
        
        actions = jax.vmap(compute_action_for_batch)(current_states, target_vel)
        
        # 4. 物理仿真更新状态
        next_states = physics_step_fn(current_states, actions)
        
        # 5. 计算CBF时间导数 - 修复版本
        cbf_derivatives = compute_cbf_time_derivative_corrected(
            cbf_values, cbf_grads, current_states, actions
        )
        
        # 6. 计算碰撞距离
        collision_distances = compute_collision_distances_corrected(current_states)
        
        # 输出本步数据
        step_outputs = {
            'states': current_states,
            'actions': actions,
            'cbf_values': cbf_values,
            'cbf_derivatives': cbf_derivatives,
            'collision_distances': collision_distances
        }
        
        return (next_states, step + 1), step_outputs
    
    # 执行时间展开
    initial_carry = (initial_states, 0)
    target_vel_sequence = jnp.transpose(target_velocities, (1, 0, 2, 3))  # [seq_len, batch, n_agents, 3]
    
    final_carry, trajectory_outputs = jax.lax.scan(
        single_step,
        initial_carry, 
        target_vel_sequence,
        length=seq_len
    )
    
    # 重新整理输出维度: [seq_len, batch, n_agents, ...] -> [batch, seq_len, n_agents, ...]
    states_traj = jnp.transpose(trajectory_outputs['states'], (1, 0, 2, 3))
    actions_traj = jnp.transpose(trajectory_outputs['actions'], (1, 0, 2, 3))
    cbf_values_traj = jnp.transpose(trajectory_outputs['cbf_values'], (1, 0, 2))
    cbf_derivatives_traj = jnp.transpose(trajectory_outputs['cbf_derivatives'], (1, 0, 2))
    collision_distances_traj = jnp.transpose(trajectory_outputs['collision_distances'], (1, 0, 2))
    
    # === 损失计算 ===
    
    # 识别安全/不安全状态
    safe_mask, unsafe_mask = identify_safe_unsafe_states(
        states_traj[..., :3], 
        collision_distances_traj,
        safety_radius=config.get('safety_radius', 0.5)
    )
    
    # CBF损失
    cbf_losses = compute_cbf_loss(
        cbf_values_traj,
        cbf_derivatives_traj,
        safe_mask,
        unsafe_mask,
        alpha=config.get('alpha', 1.0),
        gamma=config.get('gamma', 0.02)
    )
    
    # 物理损失
    physics_losses = compute_physics_loss(
        states_traj,
        actions_traj,
        target_velocities,
        collision_distances_traj,
        safety_radius=config.get('safety_radius', 0.5)
    )
    
    # 总损失
    cbf_weight = config.get('cbf_weight', 1.0)
    physics_weight = config.get('physics_weight', 1.0)
    total_loss = cbf_weight * cbf_losses.total_cbf_loss + physics_weight * physics_losses.total_physics_loss
    
    # 指标
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
        'cbf_values_std': jnp.std(cbf_values_traj),
        'cbf_derivatives_mean': jnp.mean(cbf_derivatives_traj)
    }
    
    return TotalLossComponents(
        cbf_losses=cbf_losses,
        physics_losses=physics_losses,
        total_loss=total_loss,
        metrics=metrics
    )

def compute_collision_distances_corrected(states: jnp.ndarray) -> jnp.ndarray:
    """修复版本：正确计算智能体间的最小距离"""
    positions = states[..., :3]  # [batch_size, n_agents, 3]
    
    # 计算所有智能体对之间的距离
    pos_i = jnp.expand_dims(positions, axis=2)  # [batch, n_agents, 1, 3]
    pos_j = jnp.expand_dims(positions, axis=1)  # [batch, 1, n_agents, 3] 
    distances = jnp.linalg.norm(pos_i - pos_j, axis=-1)  # [batch, n_agents, n_agents]
    
    # 排除自己与自己的距离
    n_agents = distances.shape[-1]
    eye_mask = jnp.eye(n_agents)
    distances = jnp.where(eye_mask, jnp.inf, distances)
    
    # 返回每个智能体到最近邻居的距离
    min_distances = jnp.min(distances, axis=-1)  # [batch, n_agents]
    
    return min_distances

@jax.jit
def train_step_corrected(
    training_state: TrainingState,
    batch_data: Dict[str, jnp.ndarray], 
    physics_step_fn,
    config: Dict[str, Any]
) -> Tuple[TrainingState, Dict[str, jnp.ndarray]]:
    """
    修复版本：执行一步训练 - 确保梯度正确计算和传播
    """
    
    def loss_fn(policy_params, gnn_params):
        """损失函数 - 确保两个网络都能接收到梯度"""
        # 创建临时训练状态
        temp_training_state = TrainingState(
            policy_state=training_state.policy_state.replace(params=policy_params),
            gnn_state=training_state.gnn_state.replace(params=gnn_params),
            step=training_state.step
        )
        
        # 计算损失
        loss_components = compute_total_loss_and_metrics_corrected(
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
    
    # 更新策略网络
    new_policy_state = training_state.policy_state.apply_gradients(grads=policy_grads)
    
    # 更新GNN网络
    new_gnn_state = training_state.gnn_state.apply_gradients(grads=gnn_grads)
    
    # 创建新的训练状态
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
    
    # 整理输出指标
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

# 简化的训练系统测试用例
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
        # 初始化策略网络参数
        key1, key2 = jax.random.split(self.rng_key)
        
        dummy_state = jnp.zeros(13)  # [pos(3) + vel(3) + quat(4) + angvel(3)]
        self.policy_params = self.policy_model.init(key1, dummy_state)
        
        # 初始化GNN参数
        dummy_graph = {
            'nodes': jnp.zeros((4, 13)),
            'edges': jnp.zeros((4, 4, 6)),
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
            learning_rate=self.config.get('learning_rate', 1e-4)
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

# 测试函数
def test_complete_gradient_flow():
    """测试完整系统梯度流"""
    print("开始完整系统梯度流测试...")
    
    try:
        # 配置
        config = {
            'learning_rate': 1e-4,
            'trajectory_length': 5,
            'batch_size': 2,
            'n_agents': 2,
            'sensing_radius': 2.0,
            'safety_radius': 0.5,
            'alpha': 1.0,
            'cbf_weight': 1.0,
            'physics_weight': 0.1,
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
        
        # 随机初始状态
        initial_states = jax.random.normal(
            jax.random.PRNGKey(123), 
            (batch_size, n_agents, 13)
        ) * 0.1
        
        # 随机目标速度
        target_velocities = jax.random.normal(
            jax.random.PRNGKey(456), 
            (batch_size, seq_len, n_agents, 3)
        ) * 0.5
        
        # 执行训练步骤
        print("执行完整训练步骤...")
        new_training_state, metrics = training_system.train_step(
            training_state, initial_states, target_velocities
        )
        
        print("✅ 完整训练步骤执行成功!")
        print(f"总损失: {metrics['total_loss']:.4f}")
        print(f"策略网络梯度范数: {metrics['policy_grad_norm']:.6f}")
        print(f"GNN梯度范数: {metrics['gnn_grad_norm']:.6f}")
        print(f"CBF损失: {metrics['cbf_total_loss']:.4f}")
        print(f"物理损失: {metrics['physics_total_loss']:.4f}")
        print(f"安全违规次数: {metrics['safety_violations']:.1f}")
        print(f"平均CBF值: {metrics['avg_cbf_value']:.4f}")
        
        # 验证梯度流
        policy_grad_ok = metrics['policy_grad_norm'] > 1e-8
        gnn_grad_ok = metrics['gnn_grad_norm'] > 1e-8  # 关键检查
        loss_finite = jnp.isfinite(metrics['total_loss'])
        
        if policy_grad_ok and gnn_grad_ok and loss_finite:
            print("✅ 梯度流正常，两个网络都接收到有效梯度")
            return True
        else:
            print("❌ 警告: 某些网络的梯度异常")
            if not policy_grad_ok:
                print("  - 策略网络梯度异常")
            if not gnn_grad_ok:
                print("  - GNN网络梯度异常") 
            if not loss_finite:
                print("  - 损失值异常")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

# 导出主要函数
__all__ = [
    'TrainingState',
    'CBFLossComponents', 
    'PhysicsLossComponents',
    'TotalLossComponents',
    'create_training_state',
    'compute_total_loss_and_metrics_corrected',
    'train_step_corrected',
    'CompleteTrainingSystem',
    'test_complete_gradient_flow'
]