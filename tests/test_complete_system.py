#!/usr/bin/env python3
"""
完整系统集成测试 - 修复版本
验证所有组件的协同工作
"""

import pytest
import jax
import jax.numpy as jnp
import time

from core.physics import create_initial_state, create_default_params
from core.perception import create_dummy_pointcloud
from core.safety import SafetyParams
from core.training import (
    CompleteTrainingConfig,
    initialize_complete_training,
    test_complete_gradient_flow,
    test_gradient_flow,
    TrainingConfig
)


class TestCompleteSystem:
    
    def test_minimal_viable_product(self):
        """MVP测试：验证核心技术假设"""
        print("\n" + "="*50)
        print("最小可行产品 (MVP) 测试")
        print("="*50)
        
        # 先测试基础梯度流
        print("1. 测试基础梯度流...")
        basic_success = test_gradient_flow()
        assert basic_success, "基础梯度流测试失败"
        
        # 再测试完整梯度流
        print("\n2. 测试完整系统梯度流...")
        complete_success = test_complete_gradient_flow()
        assert complete_success, "完整梯度流测试失败"
        
        print("✅ MVP验证通过")
    
    def test_component_integration(self):
        """测试各组件集成"""
        print("\n测试组件集成...")
        
        config = CompleteTrainingConfig(trajectory_length=10)
        rng_key = jax.random.PRNGKey(123)
        
        # 初始化所有组件
        (policy_model, gnn_model,
         training_state, policy_optimizer, gnn_optimizer) = initialize_complete_training(config, rng_key)
        
        # 验证参数形状和类型
        assert training_state.policy_params is not None
        assert training_state.gnn_params is not None
        
        print("✅ 组件集成测试通过")
    
    def test_training_step_performance(self):
        """测试训练步骤性能"""
        print("\n测试训练性能...")
        
        # 使用基础配置进行性能测试
        config = TrainingConfig(trajectory_length=20)
        physics_params = create_default_params()
        rng_key = jax.random.PRNGKey(456)
        
        # 初始化基础训练系统
        from core.training import initialize_training, create_loss_and_train_functions
        
        policy_model, training_state, optimizer = initialize_training(config, rng_key)
        loss_fn, train_step_jit = create_loss_and_train_functions(
            config, physics_params, policy_model
        )
        
        # 准备测试数据
        initial_state = create_initial_state()
        target_pos = jnp.array([5.0, 5.0, 3.0])
        
        # 首次执行（包含JIT编译时间）
        start_time = time.time()
        new_training_state, train_info = train_step_jit(
            training_state, optimizer, initial_state, target_pos, rng_key
        )
        first_run_time = time.time() - start_time
        
        # 第二次执行（纯执行时间）
        start_time = time.time()
        new_training_state, train_info = train_step_jit(
            training_state, optimizer, initial_state, target_pos, rng_key
        )
        second_run_time = time.time() - start_time
        
        print(f"首次执行时间 (含JIT): {first_run_time:.3f}s")
        print(f"后续执行时间: {second_run_time:.3f}s")
        
        # 验证输出
        assert 'total_loss' in train_info
        assert 'grad_norm' in train_info
        
        print(f"训练损失: {train_info['total_loss']:.4f}")
        print("✅ 性能测试通过")
    
    def test_safety_constraint_satisfaction(self):
        """测试安全约束满足情况"""
        print("\n测试安全约束...")
        
        from core.safety import safety_filter
        
        # 测试安全滤波器的基本功能
        u_nom = jnp.array([2.0, 1.0, 8.0])
        h_safe = 1.5  # 安全的CBF值
        grad_h = jnp.array([0.1, 0.2, -0.5])  # CBF梯度
        velocity = jnp.array([1.0, 0.5, 0.0])
        
        safe_command = safety_filter(
            u_nom=u_nom,
            h=h_safe,
            grad_h=grad_h,
            drone_velocity=velocity
        )
        
        print(f"名义指令: {u_nom}")
        print(f"安全指令: {safe_command}")
        print(f"CBF值: {h_safe:.4f}")
        print(f"指令修正幅度: {jnp.linalg.norm(safe_command - u_nom):.4f}")
        
        # 基本合理性检查
        assert not jnp.any(jnp.isnan(safe_command)), "安全指令不应包含NaN"
        assert jnp.allclose(safe_command, u_nom, atol=10.0), "安全指令不应偏离名义指令过远"
        
        print("✅ 安全约束测试通过")
    
    def test_multi_step_consistency(self):
        """测试多步一致性"""
        print("\n测试多步一致性...")
        
        from core.loop import rollout_trajectory
        
        config = TrainingConfig(trajectory_length=15)
        physics_params = create_default_params()
        rng_key = jax.random.PRNGKey(999)
        
        # 初始化基础系统
        from core.training import initialize_training
        policy_model, training_state, _ = initialize_training(config, rng_key)
        
        # 执行两次相同的rollout
        initial_state = create_initial_state(
            position=jnp.array([0.0, 0.0, 2.0])
        )
        
        # 第一次rollout
        final_carry1, outputs1 = rollout_trajectory(
            initial_state=initial_state,
            policy_params=training_state.policy_params,
            policy_model=policy_model,
            physics_params=physics_params,
            trajectory_length=config.trajectory_length,
            dt=config.dt
        )
        
        # 第二次rollout（应该得到相同结果）
        final_carry2, outputs2 = rollout_trajectory(
            initial_state=initial_state,
            policy_params=training_state.policy_params,
            policy_model=policy_model,
            physics_params=physics_params,
            trajectory_length=config.trajectory_length,
            dt=config.dt
        )
        
        # 验证一致性
        pos_diff = jnp.linalg.norm(final_carry1.drone_state.position - final_carry2.drone_state.position)
        
        print(f"两次执行的位置差异: {pos_diff:.8f}")
        assert pos_diff < 1e-6, "多次执行结果应该完全一致"
        
        print("✅ 一致性测试通过")


def run_all_tests():
    """运行所有测试"""
    print("🧪 开始完整系统测试")
    print("=" * 60)
    
    test_suite = TestCompleteSystem()
    
    try:
        # 核心MVP测试
        test_suite.test_minimal_viable_product()
        
        # 组件集成测试
        test_suite.test_component_integration()
        
        # 性能测试
        test_suite.test_training_step_performance()
        
        # 安全性测试
        test_suite.test_safety_constraint_satisfaction()
        
        # 一致性测试
        test_suite.test_multi_step_consistency()
        
        print("\n" + "=" * 60)
        print("🎉 所有测试通过！")
        print("✅ 系统已准备就绪，可以开始训练")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    run_all_tests()