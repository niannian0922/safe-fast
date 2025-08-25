"""
核心BPTT循环实现 - 修复版，集成感知模块和安全层
完全JAX兼容的scan函数实现，支持GNN和CBF
"""

import jax
import jax.numpy as jnp
from typing import Tuple, NamedTuple, Any, Optional
import chex

from core.physics import DroneState, DroneParams, dynamics_step
from core.policy import state_to_vector
from core.safety import safety_filter, SafetyParams


class LoopCarry(NamedTuple):
    """scan循环的carry状态"""
    drone_state: DroneState
    previous_thrust: chex.Array


class LoopOutput(NamedTuple):
    """scan循环的输出（需要记录的轨迹信息）"""
    drone_state: DroneState
    action: chex.Array
    actual_thrust: chex.Array
    reward: float
    cbf_value: float
    safe_control: chex.Array


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


def create_environment_obstacles(rng_key: chex.PRNGKey,
                               num_obstacles: int = 30,
                               bounds: float = 8.0) -> chex.Array:
    """创建环境障碍物点云"""
    return jax.random.uniform(
        rng_key,
        (num_obstacles, 3),
        minval=-bounds,
        maxval=bounds
    )


def create_rollout_functions(policy_model: Any,
                           physics_params: DroneParams,
                           dt: float,
                           perception_fn: Any = None,
                           safety_params: SafetyParams = None,
                           environment_obstacles: chex.Array = None):
    """
    创建rollout相关函数 - 完全JAX兼容版本，集成感知和安全模块
    """
    
    def scan_function_with_perception(carry: LoopCarry,
                                    x: chex.Array,  # [target_position(3)]
                                    policy_params: Any,
                                    gnn_params: Any = None) -> Tuple[LoopCarry, LoopOutput]:
        """
        带感知和安全模块的scan函数 - 完全JAX兼容
        """
        
        # 提取当前状态和目标位置
        current_state = carry.drone_state
        target_position = x
        
        # === 感知模块 ===
        cbf_value = 0.5  # 默认安全值
        grad_cbf = jnp.zeros(3)  # 默认梯度
        
        # 如果有感知模块和GNN参数
        if perception_fn is not None and gnn_params is not None and environment_obstacles is not None:
            try:
                cbf_value, grad_cbf = perception_fn(
                    gnn_params, current_state.position, environment_obstacles
                )
            except:
                # 如果感知模块出错，使用默认值
                cbf_value = 0.5
                grad_cbf = jnp.zeros(3)
        
        # === 策略网络 ===
        state_vector = state_to_vector(current_state)
        nominal_action = policy_model.apply(policy_params, state_vector)
        
        # === 安全层 ===
        safe_action = nominal_action  # 默认使用名义控制
        
        if safety_params is not None:
            try:
                safe_action = safety_filter(
                    u_nom=nominal_action,
                    h=cbf_value,
                    grad_h=grad_cbf,
                    drone_velocity=current_state.velocity,
                    safety_params=safety_params
                )
            except:
                # 如果安全层出错，使用名义控制
                safe_action = nominal_action
        
        # === 物理引擎步进 ===
        new_drone_state, actual_thrust = dynamics_step(
            current_state, safe_action, physics_params, dt, carry.previous_thrust
        )
        
        # === 奖励计算 ===
        reward = compute_step_reward(current_state, safe_action, new_drone_state, target_position)
        
        # 构造新的carry
        new_carry = LoopCarry(
            drone_state=new_drone_state,
            previous_thrust=actual_thrust
        )
        
        # 构造输出
        output = LoopOutput(
            drone_state=new_drone_state,
            action=nominal_action,
            actual_thrust=actual_thrust,
            reward=reward,
            cbf_value=cbf_value,
            safe_control=safe_action
        )
        
        return new_carry, output
    
    def scan_function_basic(carry: LoopCarry,
                          x: chex.Array,
                          policy_params: Any) -> Tuple[LoopCarry, LoopOutput]:
        """
        基础scan函数（不使用感知和安全模块）
        """
        return scan_function_with_perception(carry, x, policy_params, None)
    
    def rollout_trajectory_fn(policy_params: Any,
                            initial_state: DroneState,
                            target_position: chex.Array,
                            trajectory_length: int,
                            gnn_params: Any = None) -> Tuple[LoopCarry, LoopOutput]:
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
        
        # 选择合适的scan函数
        if gnn_params is not None:
            def scan_fn_with_params(carry, x):
                return scan_function_with_perception(carry, x, policy_params, gnn_params)
        else:
            def scan_fn_with_params(carry, x):
                return scan_function_basic(carry, x, policy_params)
        
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
                      gnn_params: Any = None,
                      perception_fn: Any = None,
                      safety_params: SafetyParams = None,
                      environment_obstacles: chex.Array = None) -> Tuple[LoopCarry, LoopOutput]:
    """
    公共接口函数 - 重新设计为JAX兼容，支持完整感知和安全功能
    """
    
    # 创建rollout函数
    rollout_fn = create_rollout_functions(
        policy_model, physics_params, dt, perception_fn, safety_params, environment_obstacles
    )
    
    return rollout_fn(policy_params, initial_state, target_position, trajectory_length, gnn_params)


class BatchRolloutSystem:
    """
    批量轨迹rollout系统 - 修复版，支持感知和安全模块
    预编译JIT函数，提高效率
    """
    
    def __init__(self, 
                 policy_model: Any, 
                 physics_params: DroneParams, 
                 dt: float,
                 perception_fn: Any = None,
                 safety_params: SafetyParams = None,
                 environment_obstacles: chex.Array = None):
        self.policy_model = policy_model
        self.physics_params = physics_params
        self.dt = dt
        self.perception_fn = perception_fn
        self.safety_params = safety_params
        self.environment_obstacles = environment_obstacles
        
        # 预编译rollout函数
        self._rollout_fn = create_rollout_functions(
            policy_model, physics_params, dt, perception_fn, safety_params, environment_obstacles
        )
    
    def rollout_single(self, policy_params: Any,
                      initial_state: DroneState,
                      target_position: chex.Array,
                      trajectory_length: int,
                      gnn_params: Any = None) -> Tuple[LoopCarry, LoopOutput]:
        """单个轨迹rollout"""
        return self._rollout_fn(policy_params, initial_state, target_position, trajectory_length, gnn_params)
    
    def rollout_batch(self, policy_params: Any,
                     initial_states: DroneState,
                     target_positions: chex.Array,
                     trajectory_length: int,
                     gnn_params: Any = None) -> Tuple[LoopCarry, LoopOutput]:
        """批量轨迹rollout"""
        # 使用vmap进行批量处理
        batch_rollout_fn = jax.vmap(
            self._rollout_fn, 
            in_axes=(None, 0, 0, None, None),  # policy_params和gnn_params广播
            out_axes=0
        )
        
        return batch_rollout_fn(policy_params, initial_states, target_positions, trajectory_length, gnn_params)


class CompleteBatchRolloutSystem:
    """
    完整批量rollout系统 - 集成所有组件
    """
    
    def __init__(self,
                 policy_model: Any,
                 physics_params: DroneParams,
                 dt: float,
                 perception_fn: Any,
                 safety_params: SafetyParams,
                 environment_config: dict = None):
        
        self.policy_model = policy_model
        self.physics_params = physics_params
        self.dt = dt
        self.perception_fn = perception_fn
        self.safety_params = safety_params
        
        # 环境配置
        if environment_config is None:
            environment_config = {
                'num_obstacles': 30,
                'obstacle_bounds': 8.0
            }
        self.environment_config = environment_config
        
        # 创建环境障碍物（在设置阶段）
        self.rng_key = jax.random.PRNGKey(42)
        self.environment_obstacles = create_environment_obstacles(
            self.rng_key,
            environment_config['num_obstacles'],
            environment_config['obstacle_bounds']
        )
        
        # 预编译rollout函数
        self._rollout_fn = create_rollout_functions(
            policy_model, physics_params, dt, perception_fn, safety_params, self.environment_obstacles
        )
    
    def rollout_single_complete(self,
                               policy_params: Any,
                               gnn_params: Any,
                               initial_state: DroneState,
                               target_position: chex.Array,
                               trajectory_length: int) -> Tuple[LoopCarry, LoopOutput]:
        """完整的单个轨迹rollout，包含所有组件"""
        return self._rollout_fn(
            policy_params, initial_state, target_position, trajectory_length, gnn_params
        )


def test_loop_jit_compatibility():
    """测试循环系统的JIT兼容性 - 修复版"""
    print("测试BPTT循环JIT兼容性...")
    
    from core.physics import create_initial_state, create_default_params
    from core.policy import create_policy_model
    from core.safety import SafetyParams
    from core.perception import create_perception_system
    
    # 设置阶段
    rng_key = jax.random.PRNGKey(42)
    policy_model = create_policy_model("mlp")
    physics_params = create_default_params()
    safety_params = SafetyParams()
    dt = 0.02
    
    # 创建感知系统
    gnn_model, perception_fn = create_perception_system()
    
    # 初始化参数
    dummy_state = jnp.zeros(13)
    policy_params = policy_model.init(rng_key, dummy_state)
    
    # 初始化GNN参数
    dummy_graph_key = jax.random.split(rng_key, 2)[1]
    from core.perception import pointcloud_to_graph
    dummy_drone_pos = jnp.zeros(3)
    dummy_cloud = jax.random.uniform(dummy_graph_key, (10, 3), minval=-5, maxval=5)
    dummy_graph = pointcloud_to_graph(dummy_drone_pos, dummy_cloud)
    gnn_params = gnn_model.init(dummy_graph_key, dummy_graph)
    
    # 测试数据
    initial_state = create_initial_state()
    target_position = jnp.array([5.0, 5.0, 3.0])
    trajectory_length = 10
    
    print("执行基础rollout测试...")
    
    # 测试基础rollout（不使用GNN）
    basic_system = BatchRolloutSystem(policy_model, physics_params, dt)
    final_carry, trajectory_outputs = basic_system.rollout_single(
        policy_params, initial_state, target_position, trajectory_length
    )
    
    print(f"✅ 基础rollout成功")
    print(f"轨迹长度: {trajectory_outputs.drone_state.position.shape[0]}")
    print(f"最终位置: {final_carry.drone_state.position}")
    print(f"奖励范围: [{jnp.min(trajectory_outputs.reward):.3f}, {jnp.max(trajectory_outputs.reward):.3f}]")
    
    print("测试完整系统rollout...")
    
    # 测试完整系统
    complete_system = CompleteBatchRolloutSystem(
        policy_model, physics_params, dt, perception_fn, safety_params
    )
    
    final_carry_complete, trajectory_outputs_complete = complete_system.rollout_single_complete(
        policy_params, gnn_params, initial_state, target_position, trajectory_length
    )
    
    print(f"✅ 完整rollout成功")
    print(f"CBF值范围: [{jnp.min(trajectory_outputs_complete.cbf_value):.3f}, {jnp.max(trajectory_outputs_complete.cbf_value):.3f}]")
    print(f"安全控制和名义控制差异: {jnp.mean(jnp.linalg.norm(trajectory_outputs_complete.safe_control - trajectory_outputs_complete.action, axis=1)):.6f}")
    
    # 测试梯度计算
    print("测试梯度计算...")
    
    def loss_fn_complete(policy_params, gnn_params):
        final_c, traj_out = complete_system.rollout_single_complete(
            policy_params, gnn_params, initial_state, target_position, trajectory_length
        )
        return jnp.linalg.norm(final_c.drone_state.position - target_position)
    
    grad_fn_complete = jax.grad(loss_fn_complete, argnums=[0, 1])
    policy_grads, gnn_grads = grad_fn_complete(policy_params, gnn_params)
    
    # 计算梯度范数
    def tree_norm(tree):
        return jnp.sqrt(sum(jnp.sum(leaf**2) for leaf in jax.tree_util.tree_leaves(tree)))
    
    policy_grad_norm = tree_norm(policy_grads)
    gnn_grad_norm = tree_norm(gnn_grads)
    
    print(f"策略梯度范数: {policy_grad_norm:.8f}")
    print(f"GNN梯度范数: {gnn_grad_norm:.8f}")
    
    assert not jnp.isnan(policy_grad_norm), "策略梯度不应包含NaN"
    assert not jnp.isnan(gnn_grad_norm), "GNN梯度不应包含NaN"
    assert policy_grad_norm > 1e-8, "策略梯度应该非零"
    assert gnn_grad_norm > 1e-8, "GNN梯度应该非零"
    
    print("✅ BPTT循环JIT兼容性测试通过!")
    
    return True


if __name__ == "__main__":
    test_loop_jit_compatibility()