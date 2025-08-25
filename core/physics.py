"""
JAX原生可微分物理引擎实现 - 完全修复JIT兼容性
基于点质量模型的无人机动力学仿真
"""

import jax
import jax.numpy as jnp
from typing import NamedTuple, Tuple
import chex


class DroneState(NamedTuple):
    """无人机状态表示"""
    position: chex.Array  # [3] - (x, y, z)
    velocity: chex.Array  # [3] - (vx, vy, vz) 
    orientation: chex.Array  # [4] - quaternion (w, x, y, z)
    angular_velocity: chex.Array  # [3] - (wx, wy, wz)


class DroneParams(NamedTuple):
    """无人机物理参数"""
    mass: float = 0.365  # kg
    drag_coefficient: chex.Array = jnp.array([0.1, 0.1, 0.15])  # 空气阻力系数
    max_thrust: float = 13.7  # N
    control_delay: float = 1/15  # 控制延迟 τ
    control_smoothing: float = 12.0  # 指数平滑因子 λ
    gravity: float = 9.81  # m/s^2


def quaternion_to_rotation_matrix(q: chex.Array) -> chex.Array:
    """四元数转旋转矩阵"""
    w, x, y, z = q
    
    # 归一化四元数
    norm = jnp.sqrt(w**2 + x**2 + y**2 + z**2)
    w, x, y, z = w/norm, x/norm, y/norm, z/norm
    
    # 构建旋转矩阵
    R = jnp.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
    ])
    
    return R


def thrust_to_body_acceleration(thrust_command: chex.Array, 
                               orientation: chex.Array,
                               params: DroneParams) -> chex.Array:
    """将推力命令转换为机体加速度 - 完全JAX兼容版本"""
    # 限制推力幅值
    thrust_magnitude = jnp.clip(jnp.linalg.norm(thrust_command), 0.0, params.max_thrust)
    
    # 使用jnp.where替代if语句，避免除零
    thrust_norm = jnp.linalg.norm(thrust_command)
    thrust_direction = jnp.where(
        thrust_norm > 1e-6,
        thrust_command / jnp.maximum(thrust_norm, 1e-8),  # 避免除零
        jnp.array([0.0, 0.0, 1.0])  # 默认向上
    )
    
    thrust_vector = thrust_magnitude * thrust_direction
    
    # 转换到世界坐标系
    R = quaternion_to_rotation_matrix(orientation)
    world_thrust = R @ thrust_vector
    
    # 转换为加速度（除以质量）
    acceleration = world_thrust / params.mass
    
    return acceleration


def apply_air_drag(velocity: chex.Array, params: DroneParams) -> chex.Array:
    """计算空气阻力加速度 - JAX兼容版本"""
    # 二次阻力模型: F_drag = -k * |v| * v
    speed = jnp.linalg.norm(velocity)
    # 使用jnp.where替代if语句
    drag_acceleration = jnp.where(
        speed > 1e-6,
        -params.drag_coefficient * speed * velocity,
        jnp.zeros_like(velocity)
    )
    
    return drag_acceleration


def control_filter(current_thrust: chex.Array,
                  desired_thrust: chex.Array,
                  params: DroneParams,
                  dt: float) -> chex.Array:
    """模拟控制器响应的指数移动平均滤波 - JAX兼容版本"""
    # 一阶低通滤波器模拟控制延迟
    alpha = jnp.exp(-params.control_smoothing * dt)
    filtered_thrust = alpha * current_thrust + (1 - alpha) * desired_thrust
    
    return filtered_thrust


def dynamics_step(state: DroneState, 
                 action: chex.Array,
                 params: DroneParams,
                 dt: float,
                 previous_thrust: chex.Array = None) -> Tuple[DroneState, chex.Array]:
    """
    单步动力学积分 - 完全JAX兼容版本，移除所有Python条件语句
    
    Args:
        state: 当前无人机状态
        action: 控制输入 [3] - 期望推力向量 (ax, ay, az)
        params: 物理参数
        dt: 时间步长
        previous_thrust: 上一步的实际推力（用于控制滤波）
    
    Returns:
        (next_state, actual_thrust): 下一状态和实际施加的推力
    """
    
    # 处理previous_thrust：如果未提供，使用action作为默认值
    # 在JAX中，我们总是传入一个值，所以这里简化处理
    actual_previous_thrust = jnp.where(
        previous_thrust is not None,
        previous_thrust,
        action  # 默认值
    ) if previous_thrust is not None else action
    
    # 控制滤波（模拟控制器响应延迟）
    actual_thrust = control_filter(actual_previous_thrust, action, params, dt)
    
    # 计算推力产生的加速度
    thrust_accel = thrust_to_body_acceleration(actual_thrust, state.orientation, params)
    
    # 重力加速度
    gravity_accel = jnp.array([0.0, 0.0, -params.gravity])
    
    # 空气阻力加速度
    drag_accel = apply_air_drag(state.velocity, params)
    
    # 总加速度
    total_accel = thrust_accel + gravity_accel + drag_accel
    
    # 积分速度和位置（使用改进的欧拉法）
    new_velocity = state.velocity + total_accel * dt
    new_position = state.position + state.velocity * dt + 0.5 * total_accel * dt**2
    
    # 简化的姿态动力学 - 完全移除条件判断
    # 在MVP阶段，我们大幅简化姿态控制
    # 假设无人机总是能快速调整到期望姿态，这里保持当前姿态
    new_orientation = state.orientation
    
    # 角速度动力学（简化处理） - 简单阻尼
    damping_factor = 0.95
    new_angular_velocity = state.angular_velocity * damping_factor
    
    # 构造新状态
    new_state = DroneState(
        position=new_position,
        velocity=new_velocity,
        orientation=new_orientation,
        angular_velocity=new_angular_velocity
    )
    
    return new_state, actual_thrust


# 为scan循环准备的包装函数
def dynamics_step_for_scan(state: DroneState,
                          action: chex.Array,
                          params: DroneParams,
                          dt: float,
                          previous_thrust: chex.Array) -> Tuple[DroneState, chex.Array]:
    """简化版本，用于jax.lax.scan循环"""
    return dynamics_step(state, action, params, dt, previous_thrust)


# JIT编译版本
dynamics_step_jit = jax.jit(dynamics_step, static_argnames=['dt'])
dynamics_step_for_scan_jit = jax.jit(dynamics_step_for_scan, static_argnames=['dt'])


def create_initial_state(position: chex.Array = None,
                        velocity: chex.Array = None) -> DroneState:
    """创建初始状态 - 处理None值"""
    default_position = jnp.zeros(3)
    default_velocity = jnp.zeros(3)
    
    actual_position = default_position if position is None else position
    actual_velocity = default_velocity if velocity is None else velocity
    
    return DroneState(
        position=actual_position,
        velocity=actual_velocity,
        orientation=jnp.array([1.0, 0.0, 0.0, 0.0]),  # 单位四元数
        angular_velocity=jnp.zeros(3)
    )


def create_default_params() -> DroneParams:
    """创建默认物理参数"""
    return DroneParams()


# 测试函数
def test_physics_jit_compatibility():
    """测试物理引擎的JIT兼容性"""
    print("测试JAX物理引擎JIT兼容性...")
    
    state = create_initial_state()
    params = create_default_params()
    action = jnp.array([0.0, 0.0, 5.0])
    dt = 0.01
    
    # 测试基础函数
    new_state, thrust = dynamics_step(state, action, params, dt)
    print(f"✅ 基础物理步进正常")
    print(f"位置变化: {jnp.linalg.norm(new_state.position):.6f}")
    
    # 测试JIT编译
    new_state_jit, thrust_jit = dynamics_step_jit(state, action, params, dt)
    print(f"✅ JIT编译版本正常")
    
    # 验证结果一致性
    pos_diff = jnp.linalg.norm(new_state.position - new_state_jit.position)
    vel_diff = jnp.linalg.norm(new_state.velocity - new_state_jit.velocity)
    
    print(f"位置差异: {pos_diff:.10f}")
    print(f"速度差异: {vel_diff:.10f}")
    
    assert pos_diff < 1e-10, "JIT和非JIT结果应该完全一致"
    assert vel_diff < 1e-10, "速度结果应该完全一致"
    
    # 测试梯度流
    def loss_fn(action_test):
        state_test = create_initial_state()
        new_state_test, _ = dynamics_step(state_test, action_test, params, dt)
        return jnp.sum(new_state_test.position**2)
    
    grad_fn = jax.grad(loss_fn)
    grad_result = grad_fn(action)
    
    print(f"梯度计算正常: {jnp.linalg.norm(grad_result):.8f}")
    assert not jnp.any(jnp.isnan(grad_result)), "梯度不应包含NaN"
    assert jnp.linalg.norm(grad_result) > 1e-8, "梯度应该非零"
    
    print("✅ JAX物理引擎JIT兼容性测试通过!")
    
    return True


if __name__ == "__main__":
    test_physics_jit_compatibility()