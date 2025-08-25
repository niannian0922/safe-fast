"""
核心BPTT循环实现 - 修复JIT兼容性
严格分离设置和计算，确保scan函数只包含纯计算
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


def compute_step_reward(current_state: DroneState,
                       action: chex.Array,
                       next_state: DroneState) -> float:
    """计算单步奖励/损失（纯计算函数）"""
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


def create_rollout_functions(policy_model: Any, 
                           physics_params: DroneParams,
                           dt: float,
                           use_rnn: bool = False):
    """
    创建rollout相关函数
    使用闭包分离设置和计算阶段
    """
    
    def scan_function(carry: LoopCarry,
                     x: Any,  # 外部输入（暂时未使用）
                     policy_params: Any) -> Tuple[LoopCarry, LoopOutput]:
        """
        纯计算的scan函数
        通过闭包捕获模型和物理参数
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
        
        # 计算即时奖励
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
    
    def rollout_trajectory_fn(policy_params: Any,
                            initial_state: DroneState,
                            trajectory_length: int) -> Tuple[LoopCarry, LoopOutput]:
        """
        纯计算的轨迹rollout函数
        """
        
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
            return scan_function(carry, x, policy_params)
        
        # 执行scan
        final_carry, trajectory_outputs = jax.lax.scan(
            scan_fn_with_params, initial_carry, xs, length=trajectory_length
        )
        
        return final_carry, trajectory_outputs
    
    # JIT编译rollout函数
    rollout_trajectory_jit = jax.jit(
        rollout_trajectory_fn, 
        static_argnames=['trajectory_length']
    )
    
    return rollout_trajectory_jit


def rollout_trajectory(initial_state: DroneState,
                      policy_params: Any,
                      policy_model: Any,
                      physics_params: DroneParams,
                      trajectory_length: int,
                      dt: float,
                      use_rnn: bool = False,
                      rng_key: chex.PRNGKey = None) -> Tuple[LoopCarry, LoopOutput]:
    """
    公共接口函数
    在内部处理JIT编译，对外提供简单接口
    """
    
    # 这个函数在每次调用时创建JIT函数，不太高效
    # 但为了保持接口兼容性暂时这样做
    rollout_fn = create_rollout_functions(policy_model, physics_params, dt, use_rnn)
    
    return rollout_fn(policy_params, initial_state, trajectory_length)