"""
修复的JAX原生物理引擎模块
参考DiffPhysDrone论文的可微分物理仿真实现
确保梯度能够正确通过物理仿真传播
"""

import jax
import jax.numpy as jnp
from typing import Dict, Tuple, Optional, NamedTuple
import chex
from functools import partial

class PhysicsState(NamedTuple):
    """物理状态结构"""
    position: jnp.ndarray     # 位置 [3,]
    velocity: jnp.ndarray     # 速度 [3,]
    acceleration: jnp.ndarray # 加速度 [3,]
    orientation: jnp.ndarray  # 方向（四元数）[4,] 
    angular_velocity: jnp.ndarray # 角速度 [3,]

class PhysicsConfig(NamedTuple):
    """物理参数配置"""
    dt: float = 0.02                    # 时间步长
    mass: float = 1.0                   # 质量
    drag_coefficient: float = 0.1       # 空气阻力系数
    max_acceleration: float = 10.0      # 最大加速度
    gravity: jnp.ndarray = jnp.array([0.0, 0.0, -9.81])  # 重力加速度
    control_delay: float = 0.05         # 控制延迟
    control_smoothing: float = 0.8      # 控制平滑系数

@jax.jit
def point_mass_dynamics(
    state: jnp.ndarray,        # [state_dim,] 当前状态
    action: jnp.ndarray,       # [action_dim,] 控制输入
    config: PhysicsConfig
) -> jnp.ndarray:
    """
    点质量动力学模型 - 参考DiffPhysDrone的简化动力学
    
    状态表示: [px, py, pz, vx, vy, vz] 
    控制输入: [ax, ay, az] (期望加速度)
    
    动力学方程:
    ṗ = v
    v̇ = a_cmd + gravity - drag(v)
    
    Args:
        state: 当前状态 [6,] = [position, velocity] 
        action: 控制输入 [3,] = [ax, ay, az]
        config: 物理参数配置
        
    Returns:
        next_state: 下一时刻状态 [6,]
    """
    # 解包当前状态
    if state.shape[-1] >= 6:
        position = state[:3]      # [3,]
        velocity = state[3:6]     # [3,]
    else:
        position = state[:3]
        velocity = jnp.zeros(3)   # 如果没有速度状态，假设为0
    
    # 限制控制输入
    action = jnp.clip(action[:3], -config.max_acceleration, config.max_acceleration)
    
    # 空气阻力: F_drag = -k * |v| * v
    speed = jnp.linalg.norm(velocity)
    drag_force = -config.drag_coefficient * speed * velocity
    
    # 总加速度 = 控制输入 + 重力 + 阻力/质量
    total_acceleration = action + config.gravity + drag_force / config.mass
    
    # 数值积分 (Verlet积分提高稳定性)
    dt = config.dt
    
    # 位置更新: p(t+dt) = p(t) + v(t)*dt + 0.5*a(t)*dt^2
    new_position = position + velocity * dt + 0.5 * total_acceleration * dt**2
    
    # 速度更新: v(t+dt) = v(t) + a(t)*dt  
    new_velocity = velocity + total_acceleration * dt
    
    # 速度限制（防止数值爆炸）
    max_speed = 20.0  # 最大速度限制
    speed_new = jnp.linalg.norm(new_velocity)
    new_velocity = jnp.where(
        speed_new > max_speed,
        new_velocity * max_speed / speed_new,
        new_velocity
    )
    
    # 组装新状态
    if state.shape[-1] >= 6:
        next_state = jnp.concatenate([new_position, new_velocity])
    else:
        next_state = new_position
    
    # 如果原状态有更多维度，保持其他维度不变
    if state.shape[-1] > 6:
        next_state = jnp.concatenate([next_state, state[6:]])
    
    return next_state

@jax.jit  
def batch_point_mass_dynamics(
    states: jnp.ndarray,       # [batch_size, n_agents, state_dim]
    actions: jnp.ndarray,      # [batch_size, n_agents, action_dim] 
    config: PhysicsConfig
) -> jnp.ndarray:
    """批量点质量动力学计算"""
    
    # 对每个智能体应用动力学
    def step_single_agent(state, action):
        return point_mass_dynamics(state, action, config)
    
    # 批量处理
    next_states = jax.vmap(jax.vmap(step_single_agent))(states, actions)
    
    return next_states

@jax.jit
def quadrotor_dynamics_simplified(
    state: jnp.ndarray,        # [12,] 四旋翼状态
    action: jnp.ndarray,       # [4,] 控制输入 [thrust, roll, pitch, yaw_rate]
    config: PhysicsConfig
) -> jnp.ndarray:
    """
    简化四旋翼动力学 - 更接近真实无人机
    
    状态: [px, py, pz, vx, vy, vz, roll, pitch, yaw, wx, wy, wz]
    控制: [thrust, roll_cmd, pitch_cmd, yaw_rate_cmd]
    """
    # 解包状态
    position = state[:3]        # [3,]
    velocity = state[3:6]       # [3,]
    euler_angles = state[6:9]   # [roll, pitch, yaw]
    angular_velocity = state[9:12]  # [3,]
    
    # 解包控制
    thrust = action[0]          # 总推力
    roll_cmd = action[1]        # 滚转角指令
    pitch_cmd = action[2]       # 俯仰角指令  
    yaw_rate_cmd = action[3]    # 偏航率指令
    
    # 限制控制输入
    thrust = jnp.clip(thrust, 0.0, 2.0 * config.mass * 9.81)  # 0到2倍重力
    roll_cmd = jnp.clip(roll_cmd, -jnp.pi/4, jnp.pi/4)        # ±45度
    pitch_cmd = jnp.clip(pitch_cmd, -jnp.pi/4, jnp.pi/4)      # ±45度
    yaw_rate_cmd = jnp.clip(yaw_rate_cmd, -2.0, 2.0)          # ±2 rad/s
    
    # 提取当前欧拉角
    roll, pitch, yaw = euler_angles
    
    # 旋转矩阵（从机体系到世界系）
    cos_roll, sin_roll = jnp.cos(roll), jnp.sin(roll)
    cos_pitch, sin_pitch = jnp.cos(pitch), jnp.sin(pitch)
    cos_yaw, sin_yaw = jnp.cos(yaw), jnp.sin(yaw)
    
    # 简化旋转矩阵（ZYX欧拉角）
    R11 = cos_yaw * cos_pitch
    R12 = cos_yaw * sin_pitch * sin_roll - sin_yaw * cos_roll  
    R13 = cos_yaw * sin_pitch * cos_roll + sin_yaw * sin_roll
    R21 = sin_yaw * cos_pitch
    R22 = sin_yaw * sin_pitch * sin_roll + cos_yaw * cos_roll
    R23 = sin_yaw * sin_pitch * cos_roll - cos_yaw * sin_roll
    R31 = -sin_pitch
    R32 = cos_pitch * sin_roll
    R33 = cos_pitch * cos_roll
    
    # 推力向量（机体系z轴方向）
    thrust_world = jnp.array([R13, R23, R33]) * thrust
    
    # 空气阻力
    drag_force = -config.drag_coefficient * velocity * jnp.linalg.norm(velocity)
    
    # 加速度 = (推力 + 重力 + 阻力) / 质量
    acceleration = (thrust_world + config.gravity * config.mass + drag_force) / config.mass
    
    # 位置和速度更新（数值积分）
    dt = config.dt
    new_position = position + velocity * dt + 0.5 * acceleration * dt**2
    new_velocity = velocity + acceleration * dt
    
    # 姿态控制（简化PD控制器）
    kp_angle = 5.0  # 角度比例增益
    kd_angle = 2.0  # 角速度阻尼增益
    
    # 滚转和俯仰控制
    roll_error = roll_cmd - roll
    pitch_error = pitch_cmd - pitch
    
    roll_rate_cmd = kp_angle * roll_error - kd_angle * angular_velocity[0]
    pitch_rate_cmd = kp_angle * pitch_error - kd_angle * angular_velocity[1]
    
    # 角加速度（简化）
    angular_acceleration = jnp.array([
        roll_rate_cmd * 10.0,    # 简化的滚转控制
        pitch_rate_cmd * 10.0,   # 简化的俯仰控制  
        yaw_rate_cmd * 5.0       # 偏航率控制
    ])
    
    # 角速度更新
    new_angular_velocity = angular_velocity + angular_acceleration * dt
    
    # 欧拉角更新（简化运动学）
    new_euler_angles = euler_angles + new_angular_velocity * dt
    
    # 限制欧拉角范围
    new_euler_angles = new_euler_angles.at[2].set(  # yaw角归一化到[-pi, pi]
        jnp.arctan2(jnp.sin(new_euler_angles[2]), jnp.cos(new_euler_angles[2]))
    )
    
    # 组装新状态
    next_state = jnp.concatenate([
        new_position,          # [3,] 
        new_velocity,          # [3,]
        new_euler_angles,      # [3,]
        new_angular_velocity   # [3,]
    ])
    
    return next_state

def create_physics_step_function(
    model_type: str = "point_mass",
    config: Optional[PhysicsConfig] = None
) -> callable:
    """
    创建物理仿真步进函数
    
    Args:
        model_type: 物理模型类型 ("point_mass" 或 "quadrotor")
        config: 物理配置参数
        
    Returns:
        physics_step_fn: 可JIT编译的物理步进函数
    """
    if config is None:
        config = PhysicsConfig()
    
    if model_type == "point_mass":
        @jax.jit
        def physics_step_fn(states, actions):
            return batch_point_mass_dynamics(states, actions, config)
            
    elif model_type == "quadrotor":
        @jax.jit  
        def physics_step_fn(states, actions):
            # 对每个智能体应用四旋翼动力学
            def step_single_quadrotor(state, action):
                return quadrotor_dynamics_simplified(state, action, config)
            
            return jax.vmap(jax.vmap(step_single_quadrotor))(states, actions)
    else:
        raise ValueError(f"未支持的物理模型类型: {model_type}")
    
    return physics_step_fn

@jax.jit
def compute_collision_forces(
    positions: jnp.ndarray,    # [n_agents, 3]
    velocities: jnp.ndarray,   # [n_agents, 3] 
    collision_radius: float = 0.5,
    collision_stiffness: float = 100.0,
    collision_damping: float = 10.0
) -> jnp.ndarray:
    """
    计算智能体间的碰撞力
    
    使用弹簧-阻尼模型处理智能体间碰撞
    
    Returns:
        collision_forces: 碰撞力 [n_agents, 3]
    """
    n_agents = positions.shape[0]
    
    # 计算所有智能体对之间的相对位置和距离
    pos_i = positions[:, None, :]  # [n_agents, 1, 3]
    pos_j = positions[None, :, :]  # [1, n_agents, 3]
    rel_pos = pos_i - pos_j        # [n_agents, n_agents, 3]
    distances = jnp.linalg.norm(rel_pos, axis=-1, keepdims=True)  # [n_agents, n_agents, 1]
    
    # 避免除零
    distances = jnp.maximum(distances, 1e-6)
    unit_vectors = rel_pos / distances  # [n_agents, n_agents, 3]
    
    # 相对速度
    vel_i = velocities[:, None, :]  # [n_agents, 1, 3]
    vel_j = velocities[None, :, :]  # [1, n_agents, 3]
    rel_vel = vel_i - vel_j         # [n_agents, n_agents, 3]
    
    # 计算碰撞力（只在距离小于碰撞半径时）
    penetration = jnp.maximum(0.0, collision_radius - distances.squeeze(-1))  # [n_agents, n_agents]
    
    # 弹性力：F_spring = k * penetration * direction
    spring_forces = collision_stiffness * penetration[:, :, None] * unit_vectors
    
    # 阻尼力：F_damping = -c * (rel_vel · unit_vector) * unit_vector
    relative_velocity_normal = jnp.sum(rel_vel * unit_vectors, axis=-1, keepdims=True)
    damping_forces = -collision_damping * relative_velocity_normal * unit_vectors
    
    # 总碰撞力
    pairwise_forces = spring_forces + damping_forces
    
    # 排除自身作用力
    mask = jnp.eye(n_agents)
    pairwise_forces = pairwise_forces * (1.0 - mask[:, :, None])
    
    # 对每个智能体求和所有作用力
    total_forces = jnp.sum(pairwise_forces, axis=1)  # [n_agents, 3]
    
    return total_forces

@jax.jit
def compute_obstacle_forces(
    positions: jnp.ndarray,     # [n_agents, 3]
    velocities: jnp.ndarray,    # [n_agents, 3]
    obstacle_points: jnp.ndarray,  # [n_obstacles, 3]
    obstacle_radius: float = 0.3,
    obstacle_stiffness: float = 200.0,
    obstacle_damping: float = 20.0
) -> jnp.ndarray:
    """
    计算智能体与障碍物之间的排斥力
    
    Returns:
        obstacle_forces: 障碍物排斥力 [n_agents, 3]
    """
    n_agents = positions.shape[0]
    n_obstacles = obstacle_points.shape[0]
    
    if n_obstacles == 0:
        return jnp.zeros((n_agents, 3))
    
    # 计算智能体到所有障碍物的距离
    agent_pos = positions[:, None, :]      # [n_agents, 1, 3]
    obs_pos = obstacle_points[None, :, :]  # [1, n_obstacles, 3]
    rel_pos = agent_pos - obs_pos          # [n_agents, n_obstacles, 3]
    distances = jnp.linalg.norm(rel_pos, axis=-1, keepdims=True)  # [n_agents, n_obstacles, 1]
    
    # 避免除零
    distances = jnp.maximum(distances, 1e-6)
    unit_vectors = rel_pos / distances     # [n_agents, n_obstacles, 3]
    
    # 计算排斥力（只在距离小于障碍物半径时）
    penetration = jnp.maximum(0.0, obstacle_radius - distances.squeeze(-1))  # [n_agents, n_obstacles]
    
    # 弹性排斥力
    spring_forces = obstacle_stiffness * penetration[:, :, None] * unit_vectors
    
    # 速度阻尼（朝向障碍物的速度分量）
    vel_toward_obs = jnp.sum(
        velocities[:, None, :] * (-unit_vectors), axis=-1, keepdims=True
    )  # [n_agents, n_obstacles, 1]
    damping_forces = obstacle_damping * jnp.maximum(0.0, vel_toward_obs) * unit_vectors
    
    # 总排斥力
    repulsion_forces = spring_forces + damping_forces
    
    # 对每个智能体求和来自所有障碍物的力
    total_forces = jnp.sum(repulsion_forces, axis=1)  # [n_agents, 3]
    
    return total_forces

def create_physics_step_with_collision(
    model_type: str = "point_mass",
    config: Optional[PhysicsConfig] = None,
    enable_agent_collision: bool = True,
    enable_obstacle_collision: bool = True
) -> callable:
    """
    创建包含碰撞处理的物理仿真函数
    
    Returns:
        physics_step_fn: 带碰撞处理的物理步进函数
    """
    if config is None:
        config = PhysicsConfig()
    
    base_physics_fn = create_physics_step_function(model_type, config)
    
    @jax.jit
    def physics_step_with_collision(states, actions, obstacle_points=None):
        """
        带碰撞处理的物理步进
        
        Args:
            states: [batch_size, n_agents, state_dim]
            actions: [batch_size, n_agents, action_dim] 
            obstacle_points: [batch_size, n_obstacles, 3] (可选)
            
        Returns:
            next_states: [batch_size, n_agents, state_dim]
        """
        batch_size, n_agents, state_dim = states.shape
        
        def step_single_batch(batch_states, batch_actions, batch_obstacles=None):
            # 提取位置和速度
            positions = batch_states[:, :3]
            velocities = batch_states[:, 3:6] if state_dim >= 6 else jnp.zeros((n_agents, 3))
            
            # 计算碰撞力
            total_collision_forces = jnp.zeros((n_agents, 3))
            
            if enable_agent_collision:
                agent_forces = compute_collision_forces(positions, velocities)
                total_collision_forces = total_collision_forces + agent_forces
            
            if enable_obstacle_collision and batch_obstacles is not None:
                obstacle_forces = compute_obstacle_forces(positions, velocities, batch_obstacles)
                total_collision_forces = total_collision_forces + obstacle_forces
            
            # 将碰撞力转换为加速度并添加到动作中
            collision_accelerations = total_collision_forces / config.mass
            
            # 修改动作以包含碰撞力
            if batch_actions.shape[-1] >= 3:
                modified_actions = batch_actions.at[:, :3].add(collision_accelerations)
            else:
                modified_actions = collision_accelerations
            
            # 应用基础物理动力学
            next_batch_states = base_physics_fn(
                batch_states[None, :, :], modified_actions[None, :, :]
            )[0]  # 移除添加的batch维度
            
            return next_batch_states
        
        if obstacle_points is None:
            # 没有障碍物的情况
            next_states = jax.vmap(
                lambda s, a: step_single_batch(s, a)
            )(states, actions)
        else:
            # 有障碍物的情况
            next_states = jax.vmap(step_single_batch)(states, actions, obstacle_points)
        
        return next_states
    
    return physics_step_with_collision

# 物理参数验证函数
def validate_physics_config(config: PhysicsConfig) -> bool:
    """验证物理配置参数的合理性"""
    
    checks = [
        (config.dt > 0, "时间步长必须为正数"),
        (config.dt < 0.1, "时间步长不应超过0.1秒"),
        (config.mass > 0, "质量必须为正数"),
        (config.drag_coefficient >= 0, "阻力系数不能为负数"),
        (config.max_acceleration > 0, "最大加速度必须为正数")
    ]
    
    all_passed = True
    for check, message in checks:
        if not check:
            print(f"物理配置验证失败: {message}")
            all_passed = False
    
    if all_passed:
        print("物理配置验证通过")
    
    return all_passed

# 主要导出函数
__all__ = [
    'PhysicsState',
    'PhysicsConfig', 
    'point_mass_dynamics',
    'batch_point_mass_dynamics',
    'quadrotor_dynamics_simplified',
    'create_physics_step_function',
    'create_physics_step_with_collision',
    'compute_collision_forces',
    'compute_obstacle_forces',
    'validate_physics_config'
]