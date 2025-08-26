"""
修复的JAX原生物理引擎模块
确保简单且可微分
"""

import jax
import jax.numpy as jnp
from typing import Dict, Tuple, Optional, NamedTuple
import chex
from functools import partial

class PhysicsConfig(NamedTuple):
    """物理参数配置"""
    dt: float = 0.02
    mass: float = 1.0
    drag_coefficient: float = 0.1
    max_acceleration: float = 10.0
    gravity: jnp.ndarray = jnp.array([0.0, 0.0, -9.81])

@jax.jit
def point_mass_dynamics(
    state: jnp.ndarray,
    action: jnp.ndarray,
    config: PhysicsConfig
) -> jnp.ndarray:
    """
    点质量动力学模型 - 简化版本
    
    状态: [px, py, pz, vx, vy, vz, ...]
    动作: [ax, ay, az]
    """
    if state.shape[-1] < 6:
        # 位置only模型
        position = state[:3]
        velocity = action[:3] * config.dt
        new_position = position + velocity
        return new_position
    
    # 完整模型
    position = state[:3]
    velocity = state[3:6]
    
    # 限制加速度
    acceleration = jnp.clip(action[:3], -config.max_acceleration, config.max_acceleration)
    
    # 添加重力和阻力
    drag_force = -config.drag_coefficient * velocity * jnp.linalg.norm(velocity)
    total_acceleration = acceleration + config.gravity + drag_force / config.mass
    
    # 更新
    new_velocity = velocity + total_acceleration * config.dt
    new_position = position + velocity * config.dt + 0.5 * total_acceleration * config.dt**2
    
    # 速度限制
    max_speed = 20.0
    speed = jnp.linalg.norm(new_velocity)
    new_velocity = jnp.where(
        speed > max_speed,
        new_velocity * max_speed / speed,
        new_velocity
    )
    
    # 组装新状态
    new_state = jnp.concatenate([new_position, new_velocity])
    
    # 保留其他状态维度
    if state.shape[-1] > 6:
        new_state = jnp.concatenate([new_state, state[6:]])
    
    return new_state

@jax.jit
def batch_point_mass_dynamics(
    states: jnp.ndarray,
    actions: jnp.ndarray,
    config: PhysicsConfig
) -> jnp.ndarray:
    """批量点质量动力学"""
    return jax.vmap(jax.vmap(lambda s, a: point_mass_dynamics(s, a, config)))(states, actions)

def create_physics_step_function(
    model_type: str = "point_mass",
    config: Optional[PhysicsConfig] = None
) -> callable:
    """创建物理步进函数"""
    if config is None:
        config = PhysicsConfig()
    
    @jax.jit
    def physics_step_fn(states, actions):
        return batch_point_mass_dynamics(states, actions, config)
    
    return physics_step_fn

def test_physics_jit_compatibility():
    """测试物理引擎JIT兼容性"""
    print("测试物理引擎JIT兼容性...")
    
    try:
        config = PhysicsConfig()
        physics_fn = create_physics_step_function("point_mass", config)
        
        # 测试数据
        states = jnp.zeros((2, 3, 13))
        states = states.at[..., 6].set(1.0)  # 单位四元数
        actions = jnp.ones((2, 3, 3)) * 0.1
        
        # JIT编译和执行
        next_states = physics_fn(states, actions)
        
        print(f"✅ 物理引擎JIT测试通过")
        print(f"  - 输入形状: {states.shape}")
        print(f"  - 输出形状: {next_states.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

__all__ = [
    'PhysicsConfig',
    'point_mass_dynamics',
    'batch_point_mass_dynamics',
    'create_physics_step_function',
    'test_physics_jit_compatibility'
]