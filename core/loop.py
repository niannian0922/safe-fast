"""
核心BPTT循环实现 - 修复JIT兼容性
使用jax.lax.scan进行高效的序列处理
"""

import jax
import jax.numpy as jnp
from typing import Tuple, NamedTuple, Any, Optional
import chex

from core.physics import DroneState, DroneParams, dynamics_step
from core.policy import state_to_vector


class LoopCarry(NamedTuple):
    """scan循环的carry状态"""
    drone_state: DroneState
    rnn_hidden: Optional[chex.Array] = None
    previous_thrust: chex.Array = jnp.zeros(3)


class LoopOutput(NamedTuple):
    """scan循环的输出（需要记录的轨迹信息）"""
    drone_state: DroneState
    action: chex.Array
    actual_thrust: chex.Array
    reward: float


def create_scan_function(policy_model: Any,
                        physics_params: DroneParams,
                        dt: float,
                        use_rnn: bool = False):
    """
    创建scan函数，使用闭包避免JIT问题
    """
    
    def scan_function(carry: LoopCarry,
                     x: Any,  # 当前时间步的外部输入（暂时未使用）
                     policy_params: Any) -> Tuple[LoopCarry, LoopOutput]:
        """
        单时间步的scan函数
        """
        
        # 提取当前状态
        current_state = carry.drone_state
        
        # 将状态转换为向量形式
        state_vector = state_to_vector(current_state)
        
        # 策略网络前向传播
        if use_rnn and carry.rnn_hidden is not None:
            action, new_rnn_hidden = policy_model.apply(
                policy_params, state_vector, carry.rnn_hidden
            )
        else:
            action = policy_model.apply(policy_params, state_vector)
            new_rnn_hidden = carry.rnn_hidden
        
        # 物理引擎步进
        new_drone_state, actual_thrust = dynamics_step(
            current_state, action, physics_params, dt, carry.previous_thrust
        )
        
        # 计算即时奖励/损失（简单版本）
        reward = compute_step_reward(current_state, action, new_drone_state)
        
        # 构造新的carry
        new_carry = LoopCarry(
            drone_state=new_drone_state,
            rnn_hidden=new_rnn_hidden,
            previous_thrust=actual_thrust
        )
        
        # 构造输出
        output = LoopOutput(
            drone_state=new_drone_state,
            action=action,
            actual_thrust=actual_thrust,
            reward=reward
        )
        
        return new_carry, output
    
    return scan_function


def compute_step_reward(current_state: DroneState,
                       action: chex.Array,
                       next_state: DroneState) -> float:
    """
    计算单步奖励/损失
    这是一个简化版本，后续会扩展为完整的多目标损失
    """
    # 目标位置（暂时固定）
    target_position = jnp.array([10.0, 10.0, 5.0])
    
    # 距离损失
    distance_to_target = jnp.linalg.norm(next_state.position - target_position)
    distance_reward = -distance_to_target
    
    # 控制成本
    control_cost = -0.01 * jnp.sum(action**2)
    
    # 速度惩罚（避免过快）
    speed_penalty = -0.001 * jnp.sum(next_state.velocity**2)
    
    total_reward = distance_reward + control_cost + speed_penalty
    
    return total_reward


def rollout_trajectory(initial_state: DroneState,
                      policy_params: Any,
                      policy_model: Any,
                      physics_params: DroneParams,
                      trajectory_length: int,
                      dt: float,
                      use_rnn: bool = False,
                      rng_key: chex.PRNGKey = None) -> Tuple[LoopCarry, LoopOutput]:
    """
    执行完整轨迹展开
    """
    
    # 创建scan函数
    scan_fn = create_scan_function(policy_model, physics_params, dt, use_rnn)
    
    # 初始化carry
    initial_carry = LoopCarry(
        drone_state=initial_state,
        rnn_hidden=policy_model.init_hidden() if use_rnn and hasattr(policy_model, 'init_hidden') else None,
        previous_thrust=jnp.zeros(3)
    )
    
    # 外部输入序列（暂时为空）
    xs = jnp.zeros((trajectory_length, 1))
    
    # 部分应用参数的scan函数
    def scan_fn_with_params(carry, x):
        return scan_fn(carry, x, policy_params)
    
    # 执行scan
    final_carry, trajectory_outputs = jax.lax.scan(
        scan_fn_with_params, initial_carry, xs, length=trajectory_length
    )
    
    return final_carry, trajectory_outputs


# JIT编译版本
def create_jit_rollout_fn(policy_model, physics_params, dt, trajectory_length, use_rnn=False):
    """创建JIT编译的rollout函数"""
    
    def rollout_fn(policy_params, initial_state, rng_key):
        return rollout_trajectory(
            initial_state, policy_params, policy_model, physics_params,
            trajectory_length, dt, use_rnn, rng_key
        )
    
    return jax.jit(rollout_fn, static_argnames=[])