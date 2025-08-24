"""
扩展的BPTT循环，集成所有组件
"""

import jax
import jax.numpy as jnp
from typing import Tuple, NamedTuple, Any, Dict, Optional
import chex

from core.physics import DroneState, DroneParams, dynamics_step
from core.policy import state_to_vector
from core.perception import pointcloud_to_graph, GCBFGraphNet, create_dummy_pointcloud
from core.safety import safety_filter, SafetyParams


class CompleteLoopCarry(NamedTuple):
    """完整scan循环的carry状态"""
    drone_state: DroneState
    rnn_hidden: Optional[chex.Array] = None
    previous_thrust: chex.Array = jnp.zeros(3)
    gnn_state: Optional[Any] = None  # 为未来扩展保留


class CompleteLoopOutput(NamedTuple):
    """完整scan循环的输出"""
    drone_state: DroneState
    u_nom: chex.Array  # 名义控制
    u_safe: chex.Array  # 安全控制
    h: float  # CBF值
    grad_h: chex.Array  # CBF梯度
    reward: float


def complete_scan_function(carry: CompleteLoopCarry,
                         point_cloud: chex.Array,  # [N, 3] 当前时间步的点云
                         policy_params: Any,
                         policy_model: Any,
                         gnn_params: Any,
                         gnn_model: GCBFGraphNet,
                         physics_params: DroneParams,
                         safety_params: SafetyParams,
                         dt: float,
                         use_rnn: bool = False) -> Tuple[CompleteLoopCarry, CompleteLoopOutput]:
    """
    完整的scan函数：感知 -> GNN -> 策略 -> 安全滤波 -> 物理
    
    Args:
        carry: 上一时间步状态
        point_cloud: 当前时间步的点云观测
        其他参数...
        
    Returns:
        (new_carry, output): 新状态和输出信息
    """
    
    current_state = carry.drone_state
    
    # 1. 感知模块：点云 -> 图
    graph = pointcloud_to_graph(
        drone_position=current_state.position,
        point_cloud=point_cloud,
        sensing_radius=5.0,
        max_neighbors=32
    )
    
    # 2. GNN模块：图 -> CBF值和梯度
    h, grad_h = gnn_model.apply(gnn_params, graph)
    
    # 3. 策略模块：状态 -> 名义控制
    state_vector = state_to_vector(current_state)
    
    if use_rnn and carry.rnn_hidden is not None:
        u_nom, new_rnn_hidden = policy_model.apply(
            policy_params, state_vector, carry.rnn_hidden
        )
    else:
        u_nom = policy_model.apply(policy_params, state_vector)
        new_rnn_hidden = carry.rnn_hidden
    
    # 4. 安全滤波层：名义控制 -> 安全控制
    u_safe = safety_filter(
        u_nom=u_nom,
        h=h,
        grad_h=grad_h,
        drone_velocity=current_state.velocity,
        safety_params=safety_params
    )
    
    # 5. 物理仿真：安全控制 -> 新状态
    new_drone_state, actual_thrust = dynamics_step(
        current_state, u_safe, physics_params, dt, carry.previous_thrust
    )
    
    # 6. 计算奖励
    reward = compute_complete_reward(current_state, u_nom, u_safe, h, new_drone_state)
    
    # 构造输出
    new_carry = CompleteLoopCarry(
        drone_state=new_drone_state,
        rnn_hidden=new_rnn_hidden,
        previous_thrust=actual_thrust,
        gnn_state=carry.gnn_state
    )
    
    output = CompleteLoopOutput(
        drone_state=new_drone_state,
        u_nom=u_nom,
        u_safe=u_safe,
        h=h,
        grad_h=grad_h,
        reward=reward
    )
    
    return new_carry, output


def compute_complete_reward(current_state: DroneState,
                          u_nom: chex.Array,
                          u_safe: chex.Array,
                          h: float,
                          next_state: DroneState) -> float:
    """
    计算完整的多目标奖励
    结合效率和安全目标
    """
    
    # 目标位置
    target_position = jnp.array([10.0, 10.0, 5.0])
    
    # 1. 效率损失（来自DiffPhysDrone思想）
    distance_to_target = jnp.linalg.norm(next_state.position - target_position)
    efficiency_reward = -distance_to_target
    
    # 2. 安全损失（来自GCBF+思想）
    # CBF值越大越好（h > 0表示安全）
    safety_reward = 10.0 * h  # 放大安全奖励
    
    # CBF约束满足情况
    h_dot = jnp.dot(grad_h, current_state.velocity)
    cbf_constraint_satisfaction = h_dot + 1.0 * h  # α=1.0
    cbf_reward = 5.0 * jnp.maximum(0.0, cbf_constraint_satisfaction)
    
    # 3. 控制平滑性
    control_deviation = jnp.linalg.norm(u_safe - u_nom)
    smoothness_penalty = -0.1 * control_deviation
    
    # 4. 能耗
    energy_cost = -0.01 * jnp.sum(u_safe**2)
    
    # 5. 边界惩罚
    bounds = 15.0
    boundary_penalty = -10.0 * jnp.sum(
        jnp.maximum(0.0, jnp.abs(next_state.position) - bounds)
    )
    
    total_reward = (
        efficiency_reward + 
        safety_reward + 
        cbf_reward +
        smoothness_penalty + 
        energy_cost + 
        boundary_penalty
    )
    
    return total_reward


def complete_rollout_trajectory(initial_state: DroneState,
                              point_cloud_sequence: chex.Array,  # [T, N, 3]
                              policy_params: Any,
                              policy_model: Any,
                              gnn_params: Any,
                              gnn_model: GCBFGraphNet,
                              physics_params: DroneParams,
                              safety_params: SafetyParams,
                              trajectory_length: int,
                              dt: float,
                              use_rnn: bool = False) -> Tuple[CompleteLoopCarry, CompleteLoopOutput]:
    """
    执行完整的轨迹展开，包含所有组件
    """
    
    # 初始化
    initial_carry = CompleteLoopCarry(
        drone_state=initial_state,
        rnn_hidden=policy_model.init_hidden() if use_rnn and hasattr(policy_model, 'init_hidden') else None,
        previous_thrust=jnp.zeros(3),
        gnn_state=None
    )
    
    # 部分应用参数的scan函数
    def scan_fn(carry, point_cloud):
        return complete_scan_function(
            carry, point_cloud,
            policy_params, policy_model,
            gnn_params, gnn_model,
            physics_params, safety_params,
            dt, use_rnn
        )
    
    # 执行scan
    final_carry, trajectory_outputs = jax.lax.scan(
        scan_fn, initial_carry, point_cloud_sequence, length=trajectory_length
    )
    
    return final_carry, trajectory_outputs


# JIT编译版本
complete_rollout_trajectory_jit = jax.jit(
    complete_rollout_trajectory,
    static_argnames=['trajectory_length', 'dt', 'use_rnn']
)