#!/usr/bin/env python3
"""
完整系统集成测试
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
    create_complete_training_step,
    test_complete_gradient_flow
)


class TestCompleteSystem:
    
    def test_minimal_viable_product(self):
        """MVP测试：验证核心技术假设"""
        print("\n" + "="*50)
        print("最小可行产品 (MVP) 测试")
        print("="*50)
        
        # 验证梯度流完整性
        success = test_complete_gradient_flow()
        assert success, "梯度流测试失败"
        
        print("✅ MVP验证通过")
    
    def test_component_integration(self):
        """测试各组件集成"""
        print("\n测试组件集成...")
        
        config = CompleteTrainingConfig(trajectory_length=10)
        rng_key = jax.random.PRNGKey(123)
        
        # 初始化所有组件
        (policy_model, gnn_model,
         policy_params, gnn_params,
         policy_optimizer, gnn_optimizer,
         policy_opt_state, gnn_opt_state) = initialize_complete_training(config, rng_key)
        
        # 验证参数形状和类型
        assert policy_params is not None
        assert gnn_params is not None
        
        print("✅ 组件集成测试通过")
    
    def test_training_step_performance(self):
        """测试训练步骤性能"""
        print("\n测试训练性能...")
        
        config = CompleteTrainingConfig(trajectory_length=20, batch_size=4)
        physics_params = create_default_params()
        safety_params = SafetyParams()
        rng_key = jax.random.PRNGKey(456)
        
        # 初始化
        (policy_model, gnn_model,
         policy_params, gnn_params,
         policy_optimizer, gnn_optimizer,
         policy_opt_state, gnn_opt_state) = initialize_complete_training(config, rng_key)
        
        train_step = create_complete_training_step(config, physics_params, safety_params)
        
        # 准备测试数据
        initial_state = create_initial_state()
        point_cloud_seq = jax.random.uniform(
            rng_key, (config.trajectory_length, 20, 3), minval=-3, maxval=3
        )
        target_pos = jnp.array([5.0, 5.0, 3.0])
        target_vel = jnp.array([1.0, 1.0, 0.0])
        
        # 首次执行（包含JIT编译时间）
        start_time = time.time()
        result = train_step(
            policy_params, policy_model,
            gnn_params, gnn_model,
            policy_opt_state, gnn_opt_state,
            policy_optimizer, gnn_optimizer,
            initial_state, point_cloud_seq,
            target_pos, target_vel, rng_key
        )
        first_run_time = time.time() - start_time
        
        # 第二次执行（纯执行时间）
        start_time = time.time()
        result = train_step(
            policy_params, policy_model,
            gnn_params, gnn_model,
            policy_opt_state, gnn_opt_state,
            policy_optimizer, gnn_optimizer,
            initial_state, point_cloud_seq,
            target_pos, target_vel, rng_key
        )
        second_run_time = time.time() - start_time
        
        print(f"首次执行时间 (含JIT): {first_run_time:.3f}s")
        print(f"后续执行时间: {second_run_time:.3f}s")
        
        # 验证输出
        new_policy_params, new_gnn_params, _, _, train_info = result
        
        assert 'total_loss' in train_info
        assert 'policy_grad_norm' in train_info
        assert 'gnn_grad_norm' in train_info
        
        print(f"训练损失: {train_info['total_loss']:.4f}")
        print("✅ 性能测试通过")
    
    def test_safety_constraint_satisfaction(self):
        """测试安全约束满足情况"""
        print("\n测试安全约束...")
        
        from core.safety import safety_filter
        from core.perception import pointcloud_to_graph, GCBFGraphNet
        
        rng_key = jax.random.PRNGKey(789)
        
        # 创建测试场景：接近障碍物
        drone_pos = jnp.array([0.0, 0.0, 1.0])
        # 在前方放置障碍物
        obstacle_points = jnp.array([
            [1.0, 0.0, 1.0],  # 前方1米处
            [0.5, 0.5, 1.0],  # 右前方
            [0.5, -0.5, 1.0]  # 左前方
        ])
        
        # 构建图
        graph = pointcloud_to_graph(drone_pos, obstacle_points)
        
        # 初始化GNN
        gnn_model = GCBFGraphNet()
        gnn_params = gnn_model.init(rng_key, graph)
        
        # 计算CBF值
        h, grad_h = gnn_model.apply(gnn_params, graph)
        
        # 测试安全滤波器
        unsafe_command = jnp.array([5.0, 0.0, 0.0])  # 直接向前冲
        safe_velocity = jnp.array([0.0, 0.0, 0.0])
        
        safe_command = safety_filter(
            u_nom=unsafe_command,
            h=h,
            grad_h=grad_h,
            drone_velocity=safe_velocity
        )
        
        print(f"危险指令: {unsafe_command}")
        print(f"安全指令: {safe_command}")
        print(f"CBF值: {h:.4f}")
        print(f"指令修正幅度: {jnp.linalg.norm(safe_command - unsafe_command):.4f}")
        
        # 安全指令应该减少向前的推力
        assert safe_command[0] < unsafe_command[0], "安全滤波器应该减少危险方向的推力"
        
        print("✅ 安全约束测试通过")
    
    def test_multi_step_consistency(self):
        """测试多步一致性"""
        print("\n测试多步一致性...")
        
        from core.loop import complete_rollout_trajectory
        
        config = CompleteTrainingConfig(trajectory_length=15)
        physics_params = create_default_params()
        safety_params = SafetyParams()
        rng_key = jax.random.PRNGKey(999)
        
        # 初始化
        (policy_model, gnn_model,
         policy_params, gnn_params,
         _, _, _, _) = initialize_complete_training(config, rng_key)
        
        # 执行两次相同的rollout
        initial_state = create_initial_state(
            position=jnp.array([0.0, 0.0, 2.0])
        )
        
        point_cloud_seq = create_dummy_pointcloud(
            rng_key, num_points=15, bounds=4.0
        )[None, :, :].repeat(config.trajectory_length, axis=0)
        
        # 第一次rollout
        final_carry1, outputs1 = complete_rollout_trajectory(
            initial_state=initial_state,
            point_cloud_sequence=point_cloud_seq,
            policy_params=policy_params,
            policy_model=policy_model,
            gnn_params=gnn_params,
            gnn_model=gnn_model,
            physics_params=physics_params,
            safety_params=safety_params,
            trajectory_length=config.trajectory_length,
            dt=config.dt
        )
        
        # 第二次rollout（应该得到相同结果）
        final_carry2, outputs2 = complete_rollout_trajectory(
            initial_state=initial_state,
            point_cloud_sequence=point_cloud_seq,
            policy_params=policy_params,
            policy_model=policy_model,
            gnn_params=gnn_params,
            gnn_model=gnn_model,
            physics_params=physics_params,
            safety_params=safety_params,
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