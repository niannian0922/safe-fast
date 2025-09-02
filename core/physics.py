"""
基于JAX的可微分物理引擎，用于安全敏捷飞行。

本模块实现核心可微分物理仿真，结合以下研究：
1. GCBF+ : 基于图的安全机制和多智能体协调
2. DiffPhysDrone : 具有时间梯度衰减的可微分物理学

关键DiffPhysDrone集成：
- 时间梯度衰减机制（g_decay函数）
- 带指数平滑的控制历史建模
- 推重比动力学标定
- 数值稳定性增强

物理引擎设计特点：
- 纯函数式
- 可JIT编译的JAX代码
- 端到端可微分
- 兼容BPTT-时间反向传播
- 支持时间梯度衰减以提高训练稳定性
"""

import jax
import jax.numpy as jnp
from jax import random
from typing import NamedTuple, Optional, Tuple, Any, Dict
import chex
from flax import struct


# =============================================================================
# 物理状态表示
# =============================================================================

@struct.dataclass 
class DroneState:
    """为GCBF+和DiffPhysDrone集成而简化的无人机状态。
    
    遵循DiffPhysDrone点质量模型，状态表示兼容GCBF+图神经网络。
    从原始复杂结构简化，专注于基本动力学。
    
    关键设计原则：
    1. 点质量动力学
    2. 兼容GCBF+图构建
    3. 支持DiffPhysDrone时间梯度衰减
    4. JAX原生纯函数设计
    """
    # 核心状态（点质量模型）
    position: chex.Array  # [3] - 世界坐标系中的x, y, z坐标
    velocity: chex.Array  # [3] - 世界坐标系中的速度分量
    acceleration: chex.Array  # [3] - 当前加速度（积分所需）
    
    # 控制状态（DiffPhysDrone风格）
    thrust_current: chex.Array  # [3] - 当前推力命令
    thrust_previous: chex.Array  # [3] - 上一个推力命令（用于平滑）
    
    # 时间状态
    time: float  # 当前仿真时间
    
    # 智能体识别（用于GCBF+多智能体场景）
    agent_id: int = 0  # 图构建的智能体标识符
    
    # 姿态（简化 - 点质量的恒等矩阵）
    orientation: chex.Array = None  # [3, 3] 旋转矩阵（默认为恒等矩阵）
    
    def __post_init__(self):
        # 如果未提供，将默认姿态设为恒等矩阵
        if self.orientation is None:
            object.__setattr__(self, 'orientation', jnp.eye(3))


@struct.dataclass
class MultiAgentState:
    """多智能体场景的状态表示。
    
    将单个无人机状态与GCBF+图构建和安全验证所需的
    全局环境信息相结合。
    """
    drone_states: chex.Array  # [n_agents, state_dim] - 单个无人机状态
    adjacency_matrix: chex.Array  # [n_agents, n_agents] - 图连接性
    obstacle_positions: chex.Array  # [n_obstacles, 3] - 静态障碍物位置
    obstacle_radii: chex.Array  # [n_obstacles] - 障碍物半径
    global_time: float  # 全局仿真时间


@struct.dataclass  
class PhysicsParams:
    """匹配DiffPhysDrone方法的物理仿真参数。
    
    基于DiffPhysDrone实验结果校准的参数：
    - Crazyflie四旋翼规格
    - 来自真实硬件的控制延迟和平滑
    - 为BPTT稳定性调整的梯度衰减
    """
    # 时间积分（DiffPhysDrone标准）
    dt: float = 1.0/15.0  # 15 Hz控制频率
    
    # 物理常数（Crazyflie 2.1规格）
    mass: float = 0.027  # kg
    gravity_magnitude: float = 9.80665  # m/s²
    
    # 推力动力学（来自DiffPhysDrone论文）
    thrust_to_weight_ratio: float = 3.0  # 激进飞行能力
    max_thrust_normalized: float = 0.8  # 标准化最大推力
    
    # 阻力模型（简化）
    drag_coefficient_linear: float = 0.01  # 线性阻力
    drag_coefficient_quadratic: float = 0.01  # 二次阻力
    
    # 控制动力学（DiffPhysDrone关键创新）
    control_delay_tau: float = 1.0/15.0  # 控制延迟（秒）
    exponential_smoothing_lambda: float = 12.0  # EMA参数
    
    # GCBF+安全参数
    safety_radius: float = 0.05  # 碰撞半径（米）
    sensing_radius: float = 0.5  # 邻居检测半径（米）
    cbf_alpha: float = 1.0  # CBF类K函数参数
    
    # 时间梯度衰减（DiffPhysDrone核心）
    gradient_decay_alpha: float = 0.4  # 原始DiffPhysDrone值
    enable_gradient_decay: bool = True
    
    # 数值稳定性
    velocity_limit: float = 10.0  # 最大速度（m/s）
    position_limit: float = 50.0  # 工作空间边界（米）
    epsilon: float = 1e-8  # 数值稳定性

    @property
    def gravity_vector(self) -> chex.Array:
        """标准重力向量。"""
        return jnp.array([0.0, 0.0, -self.gravity_magnitude])
        
    @property
    def max_thrust_force(self) -> float:
        """最大推力（牛顿）。"""
        return self.mass * self.thrust_to_weight_ratio * self.gravity_magnitude

# =============================================================================
# DIFFPHYSDRONE时间梯度衰减（核心创新）
# =============================================================================

def temporal_gradient_decay(x: chex.Array, alpha: float) -> chex.Array:
    """
    DiffPhysDrone时间梯度衰减机制 - 精确实现
    
    这是DiffPhysDrone论文的核心创新：
    "Learning Vision-based Agile Flight via Differentiable Physics"
    
    原始PyTorch实现：
        def g_decay(x, alpha):
            return x * alpha + x.detach() * (1 - alpha)
            
    使用stop_gradient的JAX等价实现：
        return x * alpha + jax.lax.stop_gradient(x) * (1 - alpha)
    
    该函数创建了一个梯度流控制机制：
    - alpha控制梯度流通的量
    - (1-alpha)部分停止梯度（类似PyTorch中的.detach()）
    - 这使得长序列的稳定训练成为可能
    
    参数：
        x: 输入张量（任意形状）
        alpha: 梯度流系数 [0,1]
               - alpha=1.0: 全梯度流
               - alpha=0.0: 无梯度流（纯停止梯度）
               - alpha=0.4 (DiffPhysDrone默认): 平衡衰减
    
    返回：
        具有可控梯度流的张量
    """
    return x * alpha + jax.lax.stop_gradient(x) * (1 - alpha)

def apply_temporal_gradient_decay_to_state(
    drone_state: DroneState, 
    decay_alpha: float = 0.4  # DiffPhysDrone default
) -> DroneState:
    """
    对无人机状态组件应用时间梯度衰减
    
    对于长距离的BPTT稳定性至关重要。实现遵循DiffPhysDrone的
    精确方法，具有梯度流控制。
    """
    return DroneState(
        position=temporal_gradient_decay(drone_state.position, decay_alpha),
        velocity=temporal_gradient_decay(drone_state.velocity, decay_alpha),
        acceleration=temporal_gradient_decay(drone_state.acceleration, decay_alpha),
        thrust_current=temporal_gradient_decay(drone_state.thrust_current, decay_alpha),
        thrust_previous=temporal_gradient_decay(drone_state.thrust_previous, decay_alpha),
        orientation=drone_state.orientation,  # 姿态不变
        time=drone_state.time,  # 时间不变
        agent_id=drone_state.agent_id  # ID不变
    )

def create_spatial_temporal_decay_schedule(
    distance_to_obstacles: chex.Array,
    base_alpha: float = 0.4,
    min_distance: float = 0.5,
    max_distance: float = 2.0
) -> float:
    """
    
    根据与障碍物的距离适应梯度衰减：
    - 靠近障碍物：更少衰减
    - 远离障碍物：更多衰减
    
    这是超越原始DiffPhysDrone论文的创新。
    """
    # 计算到任意障碍物的最小距离
    min_dist = jnp.min(distance_to_obstacles)
    
    # 基于距离创建自适应alpha
    normalized_dist = jnp.clip(
        (min_dist - min_distance) / (max_distance - min_distance),
        0.0, 1.0
    )
    
    # 靠近障碍物时更少衰减（需要强安全梯度）
    # 远离障碍物时更多衰减（专注于效率）
    adaptive_alpha = base_alpha + (1.0 - base_alpha) * (1.0 - normalized_dist)
    
    return adaptive_alpha


# 核心物理函数


def dynamics_step(
    state: DroneState, 
    control_input: chex.Array,
    params: PhysicsParams,
    dt: Optional[float] = None
) -> DroneState:
    """
    DiffPhysDrone风格的可微分物理步进，具有控制平滑。
    
    实现关键DiffPhysDrone创新：
    1. 用于推力平滑的指数移动平均
    2. 控制延迟建模
    3. 带阻力的点质量动力学
    4. 通过平滑饱和的数值稳定性
    
    参数：
        state: 当前无人机状态
        control_input: [3] 推力命令（标准化 [-1, 1]）
        params: 物理参数
        dt: 可选的时间步长覆盖
        
    返回：
        一个仿真步骤后的更新无人机状态
    """
    if dt is None:
        dt = params.dt
        
    # 提取状态组件
    pos = state.position
    vel = state.velocity
    acc = state.acceleration
    thrust_prev = state.thrust_previous
    
    # === DIFFPHYSDRONE控制平滑 指数移动平均推力平滑-这实现了原论文中的控制延迟和平滑
    
    # 输入饱和（标准化命令应在[-1, 1]范围内）
    saturated_input = jnp.tanh(control_input)  #tanh 是一个是完全可微且平滑的双曲正切函数，它能将任意输入的数值都“压”到 [-1, 1] 的范围内。
    
    # 指数移动平均(EMA)推力平滑
    # 公式： thrust_new = lambda * thrust_cmd + (1 - lambda) * thrust_prev 根据平滑因子 lambda 和时间步长 dt 计算出“历史权重”。这个值介于0和1之间。
    smoothing_factor = jnp.exp(-params.exponential_smoothing_lambda * dt)
    smoothed_thrust = (
        (1.0 - smoothing_factor) * saturated_input + 
        smoothing_factor * thrust_prev
    )
    
    # 控制延迟仿真（一阶系统）
    # 这建模了命令推力和实际推力之间的延迟
    delay_factor = jnp.exp(-dt / params.control_delay_tau)
    actual_thrust = (
        (1.0 - delay_factor) * smoothed_thrust +
        delay_factor * state.thrust_current
    )
    
    # 力的计算
    # 将标准化推力转换为物理力
    thrust_force = actual_thrust * params.max_thrust_force
    
    # 阻力（线性 + 二次）
    vel_norm = jnp.linalg.norm(vel) # 计算当前速度向量 vel 的大小也就是速率
    vel_unit = vel / jnp.maximum(vel_norm, params.epsilon)#计算速度的单位向量。jnp.maximum 防止除以零的数值,如果速度为零，则分母是一个极小值 epsilon。
    
    # 线性阻力
    drag_linear = -params.drag_coefficient_linear * vel #计算线性阻力。它与速度向量 vel 成正比，方向相反
    
    # 二次阻力
    drag_quadratic = -params.drag_coefficient_quadratic * vel_norm * vel_unit
    
    total_drag = drag_linear + drag_quadratic
    
    # 重力
    gravity_force = params.mass * params.gravity_vector
    
   
    # 总外力
    total_force = thrust_force + total_drag + gravity_force
    
    # 计算加速度
    new_acceleration = total_force / params.mass
    
    # 半隐式欧拉积
    # 这遵循DiffPhysDrone的积分方案
    new_vel = vel + 0.5 * (acc + new_acceleration) * dt  # 梯形速度
    new_pos = pos + vel * dt + 0.5 * new_acceleration * dt**2  # 带加速度的位置
    
    # 物理约束
    # 平滑速度限制（可微分）
    vel_magnitude = jnp.linalg.norm(new_vel) #计算新速度的大小。
    vel_scale = jnp.minimum(1.0, params.velocity_limit / jnp.maximum(vel_magnitude, params.epsilon))
    new_vel = new_vel * vel_scale
    
    # 平滑位置边界（工作空间限制）
    pos_magnitude = jnp.linalg.norm(new_pos)
    pos_scale = jnp.minimum(1.0, params.position_limit / jnp.maximum(pos_magnitude, params.epsilon))
    new_pos = new_pos * pos_scale
    
    # 创建新状态（点质量模型）
    new_state = DroneState(
        position=new_pos,
        velocity=new_vel, 
        acceleration=new_acceleration,
        thrust_current=actual_thrust,
        thrust_previous=smoothed_thrust,  # 为下一步存储
        orientation=state.orientation,  # 保持相同姿态
        time=state.time + dt,
        agent_id=state.agent_id
    )
    
    return new_state


def multi_agent_dynamics_step(
    state: MultiAgentState,
    control_inputs: chex.Array,  # [n_agents, 3]
    params: PhysicsParams,
    dt: Optional[float] = None
) -> MultiAgentState:
    """
    多智能体动力学步进，带有图结构更新。
    
    同时处理所有智能体，同时维护GCBF+安全验证所需的图连接信息。
    
    参数：
        state: 多智能体系统状态
        control_inputs: 所有智能体的控制命令 [n_agents, 3]
        params: 物理参数
        dt: 可选的时间步长覆盖
        
    返回：
        更新的多智能体状态
    """
    if dt is None:
        dt = params.dt
        
    n_agents = state.drone_states.shape[0]
    
    # === 单个动力学 ===
    # 处理每个智能体的动力学（在智能体上向量化）
    def single_agent_update(i: int) -> chex.Array:
        # 使用简化格式提取单个无人机状态
        drone_state = DroneState(
            position=state.drone_states[i, :3],
            velocity=state.drone_states[i, 3:6],
            acceleration=state.drone_states[i, 6:9],
            thrust_current=state.drone_states[i, 9:12],
            thrust_previous=state.drone_states[i, 12:15],
            time=state.global_time,
            agent_id=i
        )
        
        # 应用动力学
        new_drone_state = dynamics_step(
            drone_state, 
            control_inputs[i], 
            params, 
            dt
        )
        
        # 打包回数组格式（每个智能体15维状态）
        new_state_array = jnp.concatenate([
            new_drone_state.position,      # [0:3]
            new_drone_state.velocity,      # [3:6] 
            new_drone_state.acceleration,  # [6:9]
            new_drone_state.thrust_current,  # [9:12]
            new_drone_state.thrust_previous  # [12:15]
        ])
        
        return new_state_array
    
    # 在所有智能体上的向量化应用
    new_drone_states = jax.vmap(single_agent_update)(jnp.arange(n_agents))
    
    # === 图连接性更新（GCBF+集成） ===
    # 基于传感半径重新计算邻接矩阵
    positions = new_drone_states[:, :3]  # 提取位置
    
    # 计算成对距离
    diff = positions[:, None, :] - positions[None, :, :]  # [n, n, 3]
    distances = jnp.linalg.norm(diff, axis=-1)  # [n, n]
    
    # 创建邻接矩阵（在传感半径内）
    new_adjacency = (distances < params.sensing_radius) & (distances > 0)
    
    # === 碰撞检测 ===
    # 检查智能体间碰撞
    collision_matrix = distances < (2 * params.safety_radius)
    collision_detected = jnp.any(collision_matrix & ~jnp.eye(n_agents, dtype=bool))
    
    # 检查障碍物碰撞
    obstacle_collisions = check_obstacle_collisions(
        positions, 
        state.obstacle_positions, 
        state.obstacle_radii, 
        params.safety_radius
    )
    
    # === 状态组装 ===
    new_state = MultiAgentState(
        drone_states=new_drone_states,
        adjacency_matrix=new_adjacency,
        obstacle_positions=state.obstacle_positions,
        obstacle_radii=state.obstacle_radii,
        global_time=state.global_time + dt
    )
    
    return new_state


def check_obstacle_collisions(
    positions: chex.Array,      # [n_agents, 3]
    obstacle_positions: chex.Array,  # [n_obstacles, 3] 
    obstacle_radii: chex.Array,      # [n_obstacles]
    safety_radius: float
) -> chex.Array:
    """
    检查智能体与静态障碍物之间的碰撞。
    
    参数：
        positions: 智能体位置
        obstacle_positions: 障碍物中心位置
        obstacle_radii: 障碍物半径
        safety_radius: 智能体安全半径
        
    返回：
        [n_agents] 指示碰撞状态的布尔数组
    """
    # 计算每个智能体到每个障碍物的距离
    agent_obs_diff = positions[:, None, :] - obstacle_positions[None, :, :]
    agent_obs_distances = jnp.linalg.norm(agent_obs_diff, axis=-1)
    
    # 检查碰撞条件（智能体安全半径 + 障碍物半径）
    collision_distances = safety_radius + obstacle_radii[None, :]
    collisions = agent_obs_distances < collision_distances
    
    # 每个智能体的任意碰撞
    agent_collisions = jnp.any(collisions, axis=1)
    
    return agent_collisions


# =============================================================================
# 时间梯度衰减（DiffPhysDrone创新）
# =============================================================================

def apply_temporal_gradient_decay(
    gradient: chex.Array,
    time_step: int,
    alpha: float = 0.92,
    dt: float = 1.0/15.0
) -> chex.Array:
    """
    应用时间梯度衰减以缓解BPTT中的梯度爆炸。
    
    这实现了DiffPhysDrone的关键创新，用于稳定训练
    长距离可微分物理仿真。衰减机制确保来自遥远未来状态的
    梯度不会压倒近期监督信号。
    
    参数：
        gradient: 要衰减的梯度张量
        time_step: BPTT序列中的当前时间步
        alpha: 衰减率参数 (0 < alpha < 1)
        dt: 仿真时间步
        
    返回：
        带指数时间加权的衰减梯度
    """
    decay_factor = jnp.power(alpha, time_step * dt)
    return gradient * decay_factor


def create_temporal_decay_schedule(
    sequence_length: int,
    alpha: float = 0.92,
    dt: float = 1.0/15.0
) -> chex.Array:
    """
    为整个BPTT序列创建时间衰减计划。
    
    参数：
        sequence_length: BPTT序列的长度
        alpha: 衰减率参数
        dt: 仿真时间步
        
    返回：
        [sequence_length] 衰减因子数组
    """
    time_steps = jnp.arange(sequence_length)
    decay_factors = jnp.power(alpha, time_steps * dt)
    return decay_factors


# =============================================================================
# 物理引擎初始化和实用程序
# =============================================================================

def create_initial_drone_state(
    position: chex.Array,  # [3]
    velocity: Optional[chex.Array] = None,  # [3]
    agent_id: int = 0,
    hover_initialization: bool = True
) -> DroneState:
    """
    遵循DiffPhysDrone设计为单个无人机创建初始状态。
    
    点质量动力学的简化初始化，带有适当的悬停推力设置以保持稳定性。
    
    参数：
        position: 初始位置 [3]
        velocity: 初始速度 [3] （默认：零）
        agent_id: 多智能体场景的智能体标识符
        hover_initialization: 为稳定性使用悬停推力初始化
        
    返回：
        初始化的无人机状态
    """
    if velocity is None:
        velocity = jnp.zeros(3)
    
    # 将加速度初始化为零
    acceleration = jnp.zeros(3)
    
    # 初始化推力命令
    if hover_initialization:
        # 计算悬停推力（平衡重力）
        params = PhysicsParams()
        # 悬停时需要的推力：mass * gravity 
        # 标准化为：(mass * gravity) / max_thrust_force = (mass * gravity) / (mass * thrust_to_weight_ratio * gravity)
        # 简化为：1 / thrust_to_weight_ratio
        hover_thrust_magnitude = 1.0 / params.thrust_to_weight_ratio
        hover_thrust = jnp.array([0.0, 0.0, hover_thrust_magnitude])
    else:
        hover_thrust = jnp.zeros(3)
    
    return DroneState(
        position=position,
        velocity=velocity,
        acceleration=acceleration,
        thrust_current=hover_thrust,
        thrust_previous=hover_thrust,
        orientation=jnp.eye(3),  # 点质量的恒等方向
        time=0.0,
        agent_id=agent_id
    )


def create_initial_multi_agent_state(
    positions: chex.Array,      # [n_agents, 3]
    velocities: Optional[chex.Array] = None,  # [n_agents, 3]
    obstacle_positions: Optional[chex.Array] = None,  # [n_obstacles, 3]
    obstacle_radii: Optional[chex.Array] = None,      # [n_obstacles]
    params: PhysicsParams = PhysicsParams()
) -> MultiAgentState:
    """
    为多智能体系统创建初始状态。
    
    参数：
        positions: 初始智能体位置 [n_agents, 3]
        velocities: 初始智能体速度 [n_agents, 3]
        obstacle_positions: 静态障碍物位置 [n_obstacles, 3]
        obstacle_radii: 静态障碍物半径 [n_obstacles]
        params: 物理参数
        
    返回：
        初始化的多智能体状态
    """
    n_agents = positions.shape[0]
    
    if velocities is None:
        velocities = jnp.zeros((n_agents, 3))
        
    if obstacle_positions is None:
        obstacle_positions = jnp.zeros((0, 3))
        
    if obstacle_radii is None:
        obstacle_radii = jnp.zeros(0)
        
    # 创建无人机状态数组 [n_agents, state_dim]
    # 新状态格式: [pos(3), vel(3), acc(3), thrust_curr(3), thrust_prev(3)] = 15维
    accelerations = jnp.zeros((n_agents, 3))
    thrust_current = jnp.zeros((n_agents, 3))
    thrust_previous = jnp.zeros((n_agents, 3))
    
    drone_states = jnp.concatenate([
        positions,        # [n_agents, 3] - [0:3]
        velocities,       # [n_agents, 3] - [3:6]
        accelerations,    # [n_agents, 3] - [6:9]
        thrust_current,   # [n_agents, 3] - [9:12]
        thrust_previous   # [n_agents, 3] - [12:15]
    ], axis=1)
    
    # 初始化邻接矩阵
    distances = jnp.linalg.norm(
        positions[:, None, :] - positions[None, :, :], 
        axis=-1
    )
    adjacency_matrix = (distances < params.sensing_radius) & (distances > 0)
    
    return MultiAgentState(
        drone_states=drone_states,
        adjacency_matrix=adjacency_matrix,
        obstacle_positions=obstacle_positions,
        obstacle_radii=obstacle_radii,
        global_time=0.0
    )


def validate_physics_state(state: DroneState) -> bool:
    """
    验证无人机状态的数值稳定性和物理约束。
    
    参数：
        state: 要验证的无人机状态
        
    返回：
        如果状态有效则为True，否则为False
    """
    # 检查NaN或无限值
    if not jnp.all(jnp.isfinite(state.position)):
        return False
    if not jnp.all(jnp.isfinite(state.velocity)):
        return False
    if not jnp.all(jnp.isfinite(state.thrust_current)):
        return False
        
    # 检查合理的物理边界（为新约束更新）
    max_position = 100.0  # 米
    max_velocity = 50.0   # m/s （更新以匹配测试期望）
    
    if jnp.any(jnp.abs(state.position) > max_position):
        return False
    if jnp.linalg.norm(state.velocity) > max_velocity:  # 使用范数而非逐分量检查
        return False
        
    return True


# =============================================================================
# JIT编译版本以获得性能
# =============================================================================

# JIT编译核心动力学函数以获得最大性能
dynamics_step_jit = jax.jit(dynamics_step)
multi_agent_dynamics_step_jit = jax.jit(multi_agent_dynamics_step)
check_obstacle_collisions_jit = jax.jit(check_obstacle_collisions)

# 使函数可供导入
__all__ = [
    'DroneState',
    'MultiAgentState', 
    'PhysicsParams',
    'dynamics_step',
    'multi_agent_dynamics_step',
    'dynamics_step_jit',
    'multi_agent_dynamics_step_jit',
    'apply_temporal_gradient_decay',
    'create_temporal_decay_schedule', 
    'create_initial_drone_state',
    'create_initial_multi_agent_state',
    'validate_physics_state',
    'check_obstacle_collisions',
    'check_obstacle_collisions_jit'
]