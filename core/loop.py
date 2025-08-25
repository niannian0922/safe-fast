"""
核心BPTT循环实现 - 完全修复版，确保所有组件正确集成
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
    cbf_gradient: chex.Array
    safe_control: chex.Array
    safety_violation: float


def compute_step_reward(current_state: DroneState,
                       action: chex.Array,
                       next_state: DroneState,
                       target_position: chex.Array,
                       cbf_value: float,
                       safety_violation: float) -> float:
    """计算单步奖励/损失（包含安全奖励）"""
    
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
    
    # 5. 安全奖励 - 关键新增
    safety_reward = 2.0 * jnp.maximum(0, cbf_value)  # 奖励正的CBF值
    safety_penalty = -10.0 * safety_violation  # 惩罚安全违规
    
    total_reward = (distance_reward + control_cost + speed_penalty + 
                   out_of_bounds_penalty + safety_reward + safety_penalty)
    
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
    创建rollout相关函数 - 完全修复版，确保所有组件参与计算
    """
    
    def scan_function_with_full_integration(carry: LoopCarry,
                                          x: chex.Array,  # [target_position(3)]
                                          policy_params: Any,
                                          gnn_params: Any = None) -> Tuple[LoopCarry, LoopOutput]:
        """
        完全集成的scan函数 - 确保所有梯度流通
        """
        
        # 提取当前状态和目标位置
        current_state = carry.drone_state
        target_position = x
        
        # === 感知模块 ===
        cbf_value = 0.1  # 默认不安全值
        grad_cbf = jnp.zeros(3)  # 默认梯度
        
        # 如果有完整的感知系统
        if perception_fn is not None and gnn_params is not None and environment_obstacles is not None:
            # 调用真实的感知函数
            cbf_value, grad_cbf = perception_fn(
                gnn_params, current_state.position, environment_obstacles
            )
        else:
            # 简化的距离基CBF（确保有梯度）
            if environment_obstacles is not None:
                distances = jnp.linalg.norm(
                    environment_obstacles - current_state.position, axis=1
                )
                min_distance = jnp.min(distances)
                cbf_value = min_distance - 1.0  # 安全距离为1米
                
                # 计算梯度（指向最近障碍物）
                closest_idx = jnp.argmin(distances)
                direction = current_state.position - environment_obstacles[closest_idx]
                distance_to_closest = distances[closest_idx]
                grad_cbf = jnp.where(
                    distance_to_closest > 1e-6,
                    direction / jnp.maximum(distance_to_closest, 1e-8),
                    jnp.zeros(3)
                )
        
        # === 策略网络 ===
        state_vector = state_to_vector(current_state)
        nominal_action = policy_model.apply(policy_params, state_vector)
        
        # === 安全层 ===
        safe_action = nominal_action
        safety_violation = 0.0
        
        if safety_params is not None:
            # 使用真实的安全滤波器
            safe_action = safety_filter(
                u_nom=nominal_action,
                h=cbf_value,
                grad_h=grad_cbf,
                drone_velocity=current_state.velocity,
                safety_params=safety_params
            )
            
            # 计算安全违规程度
            safety_violation = jnp.maximum(0.0, -cbf_value)
        
        # === 物理引擎步进 ===
        new_drone_state, actual_thrust = dynamics_step(
            current_state, safe_action, physics_params, dt, carry.previous_thrust
        )
        
        # === 奖励计算 ===
        reward = compute_step_reward(
            current_state, safe_action, new_drone_state, target_position, 
            cbf_value, safety_violation
        )
        
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
            cbf_gradient=grad_cbf,
            safe_control=safe_action,
            safety_violation=safety_violation
        )
        
        return new_carry, output
    
    def rollout_trajectory_fn(policy_params: Any,
                            initial_state: DroneState,
                            target_position: chex.Array,
                            trajectory_length: int,
                            gnn_params: Any = None) -> Tuple[LoopCarry, LoopOutput]:
        """完全集成的轨迹rollout函数"""
        
        # 初始化carry
        initial_carry = LoopCarry(
            drone_state=initial_state,
            previous_thrust=jnp.zeros(3)
        )
        
        # 外部输入序列
        xs = jnp.tile(target_position, (trajectory_length, 1))
        
        # 选择scan函数
        def scan_fn_with_params(carry, x):
            return scan_function_with_full_integration(carry, x, policy_params, gnn_params)
        
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


class CompleteBatchRolloutSystem:
    """
    完全集成的批量rollout系统
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
        
        # 创建环境障碍物
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
        """完整的单个轨迹rollout，确保所有组件参与"""
        return self._rollout_fn(
            policy_params, initial_state, target_position, trajectory_length, gnn_params
        )


def test_complete_integration():
    """测试完整系统集成"""
    print("🔬 测试完整系统集成...")
    
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
    
    # 创建真实的感知系统
    gnn_model, perception_fn = create_perception_system()
    
    # 初始化参数
    dummy_state = jnp.zeros(13)
    policy_params = policy_model.init(rng_key, dummy_state)
    
    # 初始化真实的GNN参数
    from core.perception import pointcloud_to_graph, create_dummy_pointcloud
    dummy_cloud = create_dummy_pointcloud(jax.random.split(rng_key)[1], 10)
    dummy_graph = pointcloud_to_graph(jnp.zeros(3), dummy_cloud)
    gnn_params = gnn_model.init(jax.random.split(rng_key)[1], dummy_graph)
    
    # 测试数据
    initial_state = create_initial_state()
    target_position = jnp.array([5.0, 5.0, 3.0])
    trajectory_length = 10
    
    # 创建完整系统
    complete_system = CompleteBatchRolloutSystem(
        policy_model, physics_params, dt, perception_fn, safety_params
    )
    
    print("执行完整rollout...")
    final_carry, trajectory_outputs = complete_system.rollout_single_complete(
        policy_params, gnn_params, initial_state, target_position, trajectory_length
    )
    
    print(f"✅ 完整rollout成功")
    print(f"CBF值范围: [{jnp.min(trajectory_outputs.cbf_value):.3f}, {jnp.max(trajectory_outputs.cbf_value):.3f}]")
    print(f"安全违规次数: {jnp.sum(trajectory_outputs.safety_violation > 0)}")
    print(f"CBF梯度范数: {jnp.mean(jnp.linalg.norm(trajectory_outputs.cbf_gradient, axis=1)):.6f}")
    
    # 测试完整系统的梯度
    print("测试完整梯度流...")
    
    def complete_loss_fn(policy_params_test, gnn_params_test):
        final_c, traj_out = complete_system.rollout_single_complete(
            policy_params_test, gnn_params_test, initial_state, target_position, trajectory_length
        )
        # 综合损失：距离 + CBF违规
        distance_loss = jnp.linalg.norm(final_c.drone_state.position - target_position)
        safety_loss = jnp.sum(jnp.maximum(0, -traj_out.cbf_value))
        return distance_loss + 5.0 * safety_loss
    
    grad_fn = jax.grad(complete_loss_fn, argnums=[0, 1])
    policy_grads, gnn_grads = grad_fn(policy_params, gnn_params)
    
    # 计算梯度范数
    def tree_norm(tree):
        return jnp.sqrt(sum(jnp.sum(leaf**2) for leaf in jax.tree_util.tree_leaves(tree)))
    
    policy_grad_norm = tree_norm(policy_grads)
    gnn_grad_norm = tree_norm(gnn_grads)
    
    print(f"策略梯度范数: {policy_grad_norm:.8f}")
    print(f"GNN梯度范数: {gnn_grad_norm:.8f}")
    
    # 验证梯度
    assert not jnp.isnan(policy_grad_norm), "策略梯度不应包含NaN"
    assert not jnp.isnan(gnn_grad_norm), "GNN梯度不应包含NaN"
    assert policy_grad_norm > 1e-8, f"策略梯度过小: {policy_grad_norm}"
    assert gnn_grad_norm > 1e-8, f"GNN梯度过小: {gnn_grad_norm}"
    
    print("✅ 完整系统集成测试通过!")
    
    return True


if __name__ == "__main__":
    test_complete_integration()