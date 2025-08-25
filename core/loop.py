"""
核心BPTT循环实现 - 修复JIT兼容性问题
完全JAX兼容的scan函数实现
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
    previous_thrust: chex.Array = jnp.zeros(3)


class LoopOutput(NamedTuple):
    """scan循环的输出（需要记录的轨迹信息）"""
    drone_state: DroneState
    action: chex.Array
    actual_thrust: chex.Array
    reward: float


def compute_step_reward(current_state: DroneState,
                       action: chex.Array,
                       next_state: DroneState,
                       target_position: chex.Array) -> float:
    """计算单步奖励/损失（纯计算函数）"""
    
    # 1. 距离损失
    distance_to_target = jnp.linalg.norm(next_state.position - target_position)
    distance_reward = -distance_to_target
    
    # 2. 控制成本
    control_cost = -0.01 * jnp.sum(action**2)
    
    # 3. 速度惩罚（避免过快）
    speed_penalty = -0.001 * jnp.sum(next_state.velocity**2)
    
    # 4. 边界惩罚
    bounds = 20.0
    out_of_bounds_penalty = -1.0 * jnp.sum(
        jnp.maximum(0, jnp.abs(next_state.position) - bounds)
    )
    
    total_reward = distance_reward + control_cost + speed_penalty + out_of_bounds_penalty
    
    return total_reward


def create_rollout_functions(policy_model: Any, 
                           physics_params: DroneParams,
                           dt: float):
    """
    创建rollout相关函数 - 完全JAX兼容版本
    使用闭包分离设置和计算阶段
    """
    
    def scan_function(carry: LoopCarry,
                     x: chex.Array,  # 外部输入：[target_position(3)]
                     policy_params: Any) -> Tuple[LoopCarry, LoopOutput]:
        """
        纯计算的scan函数 - 完全JAX兼容
        通过闭包捕获模型和物理参数
        """
        
        # 提取当前状态和目标位置
        current_state = carry.drone_state
        target_position = x  # 目标位置从外部输入获取
        
        # 将状态转换为向量形式
        state_vector = state_to_vector(current_state)
        
        # 策略网络前向传播（简化为MLP，移除RNN逻辑）
        action = policy_model.apply(policy_params, state_vector)
        
        # 物理引擎步进
        new_drone_state, actual_thrust = dynamics_step(
            current_state, action, physics_params, dt, carry.previous_thrust
        )
        
        # 计算即时奖励
        reward = compute_step_reward(current_state, action, new_drone_state, target_position)
        
        # 构造新的carry
        new_carry = LoopCarry(
            drone_state=new_drone_state,
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
                            target_position: chex.Array,
                            trajectory_length: int) -> Tuple[LoopCarry, LoopOutput]:
        """
        纯计算的轨迹rollout函数
        """
        
        # 初始化carry
        initial_carry = LoopCarry(
            drone_state=initial_state,
            previous_thrust=jnp.zeros(3)
        )
        
        # 外部输入序列：每个时间步都使用相同的目标位置
        xs = jnp.tile(target_position, (trajectory_length, 1))  # [T, 3]
        
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
                      target_position: chex.Array,
                      trajectory_length: int,
                      dt: float,
                      use_rnn: bool = False,  # 暂时忽略，简化为MLP
                      rng_key: chex.PRNGKey = None) -> Tuple[LoopCarry, LoopOutput]:
    """
    公共接口函数 - 重新设计为JAX兼容
    
    Args:
        initial_state: 初始状态
        policy_params: 策略参数
        policy_model: 策略模型
        physics_params: 物理参数
        target_position: 目标位置 [3]
        trajectory_length: 轨迹长度
        dt: 时间步长
        use_rnn: 是否使用RNN（暂时忽略）
        rng_key: 随机数种子（暂时未使用）
    """
    
    # 创建rollout函数（在每次调用时创建，暂时这样做）
    rollout_fn = create_rollout_functions(policy_model, physics_params, dt)
    
    return rollout_fn(policy_params, initial_state, target_position, trajectory_length)


class BatchRolloutSystem:
    """
    批量轨迹rollout系统
    预编译JIT函数，提高效率
    """
    
    def __init__(self, policy_model: Any, physics_params: DroneParams, dt: float):
        self.policy_model = policy_model
        self.physics_params = physics_params
        self.dt = dt
        
        # 预编译rollout函数
        self._rollout_fn = create_rollout_functions(policy_model, physics_params, dt)
    
    def rollout_single(self, policy_params: Any,
                      initial_state: DroneState,
                      target_position: chex.Array,
                      trajectory_length: int) -> Tuple[LoopCarry, LoopOutput]:
        """单个轨迹rollout"""
        return self._rollout_fn(policy_params, initial_state, target_position, trajectory_length)
    
    def rollout_batch(self, policy_params: Any,
                     initial_states: DroneState,  # 批量状态
                     target_positions: chex.Array,  # [batch_size, 3]
                     trajectory_length: int) -> Tuple[LoopCarry, LoopOutput]:
        """批量轨迹rollout"""
        # 使用vmap进行批量处理
        batch_rollout_fn = jax.vmap(
            self._rollout_fn, 
            in_axes=(None, 0, 0, None),  # policy_params广播，其他按batch维度
            out_axes=0
        )
        
        return batch_rollout_fn(policy_params, initial_states, target_positions, trajectory_length)


def test_loop_jit_compatibility():
    """测试循环系统的JIT兼容性"""
    print("测试BPTT循环JIT兼容性...")
    
    from core.physics import create_initial_state, create_default_params
    from core.policy import create_policy_model
    
    # 设置阶段
    rng_key = jax.random.PRNGKey(42)
    policy_model = create_policy_model("mlp")
    physics_params = create_default_params()
    dt = 0.02
    
    # 初始化策略参数
    dummy_state = jnp.zeros(13)
    policy_params = policy_model.init(rng_key, dummy_state)
    
    # 测试数据
    initial_state = create_initial_state()
    target_position = jnp.array([5.0, 5.0, 3.0])
    trajectory_length = 10
    
    print("执行基础rollout测试...")
    
    # 测试基础rollout
    final_carry, trajectory_outputs = rollout_trajectory(
        initial_state=initial_state,
        policy_params=policy_params,
        policy_model=policy_model,
        physics_params=physics_params,
        target_position=target_position,
        trajectory_length=trajectory_length,
        dt=dt
    )
    
    print(f"✅ 基础rollout成功")
    print(f"轨迹长度: {trajectory_outputs.drone_state.position.shape[0]}")
    print(f"最终位置: {final_carry.drone_state.position}")
    print(f"奖励范围: [{jnp.min(trajectory_outputs.reward):.3f}, {jnp.max(trajectory_outputs.reward):.3f}]")
    
    # 测试批量系统
    print("测试批量rollout系统...")
    
    batch_system = BatchRolloutSystem(policy_model, physics_params, dt)
    
    # 单个rollout测试
    final_carry_batch, trajectory_outputs_batch = batch_system.rollout_single(
        policy_params, initial_state, target_position, trajectory_length
    )
    
    # 验证一致性
    pos_diff = jnp.linalg.norm(final_carry.drone_state.position - final_carry_batch.drone_state.position)
    print(f"批量vs单个结果差异: {pos_diff:.10f}")
    assert pos_diff < 1e-10, "批量和单个rollout结果应该一致"
    
    # 测试梯度计算
    print("测试梯度计算...")
    
    def loss_fn(params):
        final_c, traj_out = batch_system.rollout_single(
            params, initial_state, target_position, trajectory_length
        )
        # 简单损失：最终距离目标的距离
        return jnp.linalg.norm(final_c.drone_state.position - target_position)
    
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(policy_params)
    
    # 计算梯度范数
    def tree_norm(tree):
        return jnp.sqrt(sum(jnp.sum(leaf**2) for leaf in jax.tree_util.tree_leaves(tree)))
    
    grad_norm = tree_norm(grads)
    print(f"梯度范数: {grad_norm:.8f}")
    
    assert not jnp.isnan(grad_norm), "梯度不应包含NaN"
    assert grad_norm > 1e-8, "梯度应该非零"
    
    print("✅ BPTT循环JIT兼容性测试通过!")
    
    return True


if __name__ == "__main__":
    test_loop_jit_compatibility()