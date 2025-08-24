"""
物理引擎单元测试
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from core.physics import (
    DroneState, DroneParams, dynamics_step, dynamics_step_jit,
    create_initial_state, create_default_params,
    quaternion_to_rotation_matrix, thrust_to_body_acceleration
)


class TestPhysicsEngine:
    
    def test_create_initial_state(self):
        """测试初始状态创建"""
        state = create_initial_state()
        
        assert state.position.shape == (3,)
        assert state.velocity.shape == (3,)
        assert state.orientation.shape == (4,)
        assert state.angular_velocity.shape == (3,)
        
        # 检查初始值
        np.testing.assert_array_almost_equal(state.position, np.zeros(3))
        np.testing.assert_array_almost_equal(state.velocity, np.zeros(3))
        np.testing.assert_array_almost_equal(state.orientation, np.array([1, 0, 0, 0]))
    
    def test_quaternion_to_rotation_matrix(self):
        """测试四元数到旋转矩阵转换"""
        # 单位四元数应该产生单位矩阵
        q = jnp.array([1.0, 0.0, 0.0, 0.0])
        R = quaternion_to_rotation_matrix(q)
        
        expected = jnp.eye(3)
        np.testing.assert_array_almost_equal(R, expected, decimal=5)
        
        # 检查旋转矩阵性质：R @ R.T = I, det(R) = 1
        np.testing.assert_array_almost_equal(R @ R.T, jnp.eye(3), decimal=5)
        np.testing.assert_almost_equal(jnp.linalg.det(R), 1.0, decimal=5)
    
    def test_thrust_to_body_acceleration(self):
        """测试推力到加速度转换"""
        params = create_default_params()
        orientation = jnp.array([1.0, 0.0, 0.0, 0.0])  # 无旋转
        thrust = jnp.array([0.0, 0.0, params.max_thrust])  # 最大向上推力
        
        accel = thrust_to_body_acceleration(thrust, orientation, params)
        expected_accel = thrust / params.mass
        
        np.testing.assert_array_almost_equal(accel, expected_accel, decimal=5)
    
    def test_dynamics_step_basic(self):
        """测试基本动力学步进"""
        state = create_initial_state()
        params = create_default_params()
        action = jnp.array([0.0, 0.0, params.mass * params.gravity])  # 抵消重力
        dt = 0.01
        
        new_state, actual_thrust = dynamics_step(state, action, params, dt)
        
        # 检查状态形状
        assert new_state.position.shape == (3,)
        assert new_state.velocity.shape == (3,)
        assert actual_thrust.shape == (3,)
        
        # 在抵消重力的情况下，位置和速度应该变化很小
        assert jnp.allclose(new_state.velocity, jnp.zeros(3), atol=1e-2)
    
    def test_dynamics_step_gravity_only(self):
        """测试仅重力作用下的运动"""
        state = create_initial_state()
        params = create_default_params()
        action = jnp.zeros(3)  # 无推力
        dt = 0.01
        
        new_state, _ = dynamics_step(state, action, params, dt)
        
        # 重力作用下，z方向速度应该减小
        expected_vel_z = -params.gravity * dt
        np.testing.assert_almost_equal(new_state.velocity[2], expected_vel_z, decimal=4)
    
    def test_jit_compilation(self):
        """测试JIT编译功能"""
        state = create_initial_state()
        params = create_default_params()
        action = jnp.zeros(3)
        dt = 0.01
        
        # 应该能够成功JIT编译和执行
        new_state, _ = dynamics_step_jit(state, action, params, dt)
        
        assert new_state.position.shape == (3,)
        assert new_state.velocity.shape == (3,)
    
    def test_gradient_flow(self):
        """测试梯度流通性"""
        def loss_fn(action):
            state = create_initial_state()
            params = create_default_params()
            dt = 0.01
            
            new_state, _ = dynamics_step(state, action, params, dt)
            # 简单损失：最终位置的L2范数
            return jnp.sum(new_state.position**2)
        
        action = jnp.array([1.0, 0.0, 0.0])
        
        # 应该能够计算梯度
        loss_and_grad = jax.value_and_grad(loss_fn)
        loss_val, grad = loss_and_grad(action)
        
        assert not jnp.isnan(loss_val)
        assert not jnp.any(jnp.isnan(grad))
        assert grad.shape == (3,)
        assert jnp.sum(jnp.abs(grad)) > 1e-6  # 梯度不应为零
    
    def test_multiple_step_consistency(self):
        """测试多步积分的一致性"""
        state = create_initial_state()
        params = create_default_params()
        action = jnp.array([0.0, 0.0, params.mass * params.gravity * 1.1])  # 略微向上
        dt = 0.01
        n_steps = 10
        
        # 使用数值方法进行多步积分
        current_state = state
        for _ in range(n_steps):
            current_state, _ = dynamics_step(current_state, action, params, dt)
        
        # 检查最终状态的合理性
        assert current_state.position[2] > 0  # 应该向上运动
        assert current_state.velocity[2] > 0  # 向上速度
    
    def test_parameter_effects(self):
        """测试参数对动力学的影响"""
        state = create_initial_state()
        action = jnp.array([0.0, 0.0, 5.0])
        dt = 0.01
        
        # 测试不同质量的影响
        params1 = DroneParams(mass=0.365)  # 默认质量
        params2 = DroneParams(mass=0.730)  # 双倍质量
        
        new_state1, _ = dynamics_step(state, action, params1, dt)
        new_state2, _ = dynamics_step(state, action, params2, dt)
        
        # 质量大的无人机加速度应该更小
        accel1 = new_state1.velocity / dt
        accel2 = new_state2.velocity / dt
        
        # 由于相同的推力和更大的质量，accel2的z分量应该更小
        assert accel2[2] < accel1[2]


if __name__ == "__main__":
    pytest.main([__file__])