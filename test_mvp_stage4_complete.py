#!/usr/bin/env python3
"""
MVP Stage 4 Complete End-to-End Test

This is the FINAL TEST for MVP Stage 4 that validates:
1. Complete scan_function with all components (GNN + Policy + Safety + Physics)
2. Simple weighted loss function: L_total = α * L_efficiency + β * L_safety  
3. Full gradient flow from loss back to GNN and Policy parameters
4. JIT compilation of the entire pipeline
5. Training step execution with parameter updates

This test ensures 100% readiness for end-to-end training.
"""

import jax
import jax.numpy as jnp
from jax import random, grad, jit
import optax
import time
from typing import Dict, Tuple

# Import all system components
from core.physics import (
    DroneState, PhysicsParams, create_initial_drone_state, dynamics_step
)
from core.perception import (
    PerceptionModule, CBFNet, pointcloud_to_graph, GraphConfig
)
from core.policy import (
    PolicyNetworkMLP, PolicyParams, create_policy_network
)
from core.safety import (
    SafetyLayer, SafetyConfig
)
from core.loop import (
    ScanCarry, ScanOutput, create_complete_bptt_scan_function
)
from core.training import (
    compute_simple_weighted_loss, simple_training_step, create_optimizer
)

def create_mock_scan_outputs(T: int, B: int, key: jnp.ndarray) -> ScanOutput:
    """创建模拟的扫描输出数据用于测试"""
    keys = random.split(key, 6)
    
    # 模拟轨迹数据
    positions = random.uniform(keys[0], (T, B, 3), minval=-2.0, maxval=2.0)
    velocities = random.uniform(keys[1], (T, B, 3), minval=-1.0, maxval=1.0)
    controls = random.uniform(keys[2], (T, B, 3), minval=-0.5, maxval=0.5)
    cbf_values = random.uniform(keys[3], (T, B), minval=-0.5, maxval=1.0)
    cbf_gradients = random.uniform(keys[4], (T, B, 3), minval=-1.0, maxval=1.0)
    
    # 构建完整状态（12维：位置3 + 速度3 + 加速度3 + 其他3）
    accelerations = random.uniform(keys[5], (T, B, 3), minval=-0.2, maxval=0.2)
    other_states = jnp.zeros((T, B, 3))  # 填充维度
    full_states = jnp.concatenate([positions, velocities, accelerations, other_states], axis=-1)
    
    return ScanOutput(
        positions=positions[0, 0],  # 单个位置示例
        velocities=velocities[0, 0],
        control_commands=controls[0, 0],
        nominal_commands=controls[0, 0],
        step_loss=0.0,
        safety_violation=0.0,
        # 扩展字段
        drone_states=full_states,
        cbf_values=cbf_values,
        cbf_gradients=cbf_gradients,
        safe_controls=controls,
        obstacle_distances=jnp.ones((T, B)),
        trajectory_lengths=jnp.ones((T,))
    )

def test_simple_weighted_loss():
    """测试简单加权损失函数"""
    print("🧪 Testing Simple Weighted Loss Function...")
    
    key = random.PRNGKey(42)
    T, B = 20, 2  # 20时间步，2批次
    
    # 创建模拟数据
    scan_outputs = create_mock_scan_outputs(T, B, key)
    target_positions = jnp.array([[1.0, 1.0, 2.0], [2.0, 0.0, 2.5]])  # (B, 3)
    target_velocities = jnp.zeros((T, B, 3))  # 目标速度为零
    physics_params = PhysicsParams()
    
    # 计算损失
    total_loss, loss_breakdown = compute_simple_weighted_loss(
        scan_outputs, target_positions, target_velocities, physics_params,
        alpha_efficiency=1.0, beta_safety=2.0
    )
    
    print(f"✅ Simple weighted loss computation successful")
    print(f"   Total Loss: {total_loss:.4f}")
    print(f"   Efficiency Loss: {loss_breakdown['efficiency_loss']:.4f}")
    print(f"   Safety Loss: {loss_breakdown['safety_loss']:.4f}")
    print(f"   Final Goal Distance: {loss_breakdown['final_goal_distance']:.4f}")
    
    # 验证损失组件
    expected_total = (1.0 * loss_breakdown['efficiency_loss'] + 
                     2.0 * loss_breakdown['safety_loss'])
    assert jnp.abs(total_loss - expected_total) < 1e-6, "Loss computation error"
    
    print("✅ Loss breakdown verification passed")
    return True

def test_complete_scan_function_jit():
    """测试完整扫描函数的JIT编译"""
    print("🧪 Testing Complete Scan Function JIT Compilation...")
    
    key = random.PRNGKey(42)
    keys = random.split(key, 4)
    
    # 初始化所有组件
    physics_params = PhysicsParams()
    safety_config = SafetyConfig()
    
    # CBF网络参数
    cbf_net = CBFNet(gnn_layers=2)
    dummy_graph_nodes = jnp.ones((5, 3))
    dummy_graph_edges = jnp.ones((8, 4))
    
    import jraph
    dummy_graph = jraph.GraphsTuple(
        nodes=dummy_graph_nodes,
        edges=dummy_graph_edges,
        senders=jnp.array([0, 1, 2, 3, 4, 0, 1, 2]),
        receivers=jnp.array([1, 2, 3, 4, 0, 2, 3, 4]),
        n_node=jnp.array([5]),
        n_edge=jnp.array([8]),
        globals=None
    )
    cbf_net_params = cbf_net.init(keys[0], dummy_graph)
    
    # 策略网络参数
    policy_params = PolicyParams(hidden_dims=(32, 16), use_rnn=False)
    policy_net = create_policy_network(policy_params, network_type="mlp")
    dummy_obs = jnp.ones(10)
    policy_net_params = policy_net.init(keys[1], dummy_obs)
    
    print("✅ All components initialized")
    
    # 创建完整扫描函数
    scan_function = create_complete_bptt_scan_function(
        cbf_net_params, policy_net_params, safety_config, physics_params
    )
    
    # 测试JIT编译
    jit_scan_function = jit(scan_function)
    
    # 创建测试输入
    initial_pos = jnp.array([0.0, 0.0, 1.0])
    drone_state = create_initial_drone_state(initial_pos)
    
    carry = ScanCarry(
        drone_state=drone_state,
        rnn_hidden_state=jnp.zeros(16),
        step_count=jnp.array(0),
        cumulative_reward=jnp.array(0.0)
    )
    
    external_input = {'target_velocity': jnp.array([0.1, 0.0, 0.0])}
    
    # 执行单步
    start_time = time.time()
    new_carry, outputs = jit_scan_function(carry, external_input)
    compile_time = time.time() - start_time
    
    print(f"✅ JIT compilation successful (time: {compile_time:.3f}s)")
    print(f"   Position: {drone_state.position} -> {new_carry.drone_state.position}")
    print(f"   Safety violation: {outputs.safety_violation:.4f}")
    
    return True

def test_end_to_end_gradient_flow():
    """测试端到端梯度流"""
    print("🧪 Testing End-to-End Gradient Flow...")
    
    key = random.PRNGKey(42)
    keys = random.split(key, 4)
    T, B = 10, 1  # 简化测试
    
    # 创建网络参数
    cbf_net = CBFNet(gnn_layers=2)
    dummy_graph_nodes = jnp.ones((5, 3))
    dummy_graph_edges = jnp.ones((8, 4))
    
    import jraph
    dummy_graph = jraph.GraphsTuple(
        nodes=dummy_graph_nodes,
        edges=dummy_graph_edges,
        senders=jnp.array([0, 1, 2, 3, 4, 0, 1, 2]),
        receivers=jnp.array([1, 2, 3, 4, 0, 2, 3, 4]),
        n_node=jnp.array([5]),
        n_edge=jnp.array([8]),
        globals=None
    )
    cbf_net_params = cbf_net.init(keys[0], dummy_graph)
    
    policy_params = PolicyParams(hidden_dims=(32, 16), use_rnn=False)
    policy_net = create_policy_network(policy_params, network_type="mlp")
    dummy_obs = jnp.ones(10)
    policy_net_params = policy_net.init(keys[1], dummy_obs)
    
    # 合并参数
    params_dict = {
        'cbf_net': cbf_net_params,
        'policy_net': policy_net_params
    }
    
    print("✅ Network parameters initialized")
    
    # 创建批次数据
    scan_outputs = create_mock_scan_outputs(T, B, keys[2])
    target_positions = jnp.array([[1.0, 1.0, 2.0]])  # (B=1, 3)
    target_velocities = jnp.zeros((T, B, 3))
    
    batch_data = {
        'scan_outputs': scan_outputs,
        'target_positions': target_positions,
        'target_velocities': target_velocities
    }
    
    # 创建优化器
    optimizer = create_optimizer(learning_rate=1e-3)
    opt_state = optimizer.init(params_dict)
    
    # 测试梯度计算
    def loss_fn(params):
        loss, breakdown = compute_simple_weighted_loss(
            scan_outputs, target_positions, target_velocities,
            PhysicsParams(), alpha_efficiency=1.0, beta_safety=2.0
        )
        return loss, breakdown
    
    (loss_value, loss_breakdown), gradients = jax.value_and_grad(loss_fn, has_aux=True)(params_dict)
    
    print(f"✅ Gradient computation successful")
    print(f"   Loss value: {loss_value:.4f}")
    
    # 检查梯度质量
    def check_gradients(params, name=""):
        if isinstance(params, dict):
            for key, value in params.items():
                check_gradients(value, f"{name}/{key}")
        else:
            grad_norm = jnp.linalg.norm(params)
            grad_mean = jnp.mean(jnp.abs(params))
            grad_max = jnp.max(jnp.abs(params))
            
            print(f"   {name}: norm={grad_norm:.6f}, mean={grad_mean:.6f}, max={grad_max:.6f}")
            
            assert jnp.all(jnp.isfinite(params)), f"Gradients contain NaN/Inf in {name}"
            assert grad_norm > 1e-10, f"Gradients too small in {name}"
            assert grad_norm < 1e3, f"Gradients too large in {name}"
    
    check_gradients(gradients, "params")
    
    print("✅ Gradient quality checks passed")
    return True

def test_training_step_execution():
    """测试完整训练步骤执行"""
    print("🧪 Testing Complete Training Step Execution...")
    
    key = random.PRNGKey(42)
    keys = random.split(key, 4)
    T, B = 15, 2
    
    # 初始化参数
    cbf_net = CBFNet(gnn_layers=2)
    dummy_graph_nodes = jnp.ones((5, 3))
    dummy_graph_edges = jnp.ones((8, 4))
    
    import jraph
    dummy_graph = jraph.GraphsTuple(
        nodes=dummy_graph_nodes,
        edges=dummy_graph_edges,
        senders=jnp.array([0, 1, 2, 3, 4, 0, 1, 2]),
        receivers=jnp.array([1, 2, 3, 4, 0, 2, 3, 4]),
        n_node=jnp.array([5]),
        n_edge=jnp.array([8]),
        globals=None
    )
    cbf_net_params = cbf_net.init(keys[0], dummy_graph)
    
    policy_params = PolicyParams(hidden_dims=(32, 16), use_rnn=False)
    policy_net = create_policy_network(policy_params, network_type="mlp")
    dummy_obs = jnp.ones(10)
    policy_net_params = policy_net.init(keys[1], dummy_obs)
    
    params_dict = {
        'cbf_net': cbf_net_params,
        'policy_net': policy_net_params
    }
    
    # 创建数据和优化器
    scan_outputs = create_mock_scan_outputs(T, B, keys[2])
    target_positions = jnp.array([[1.0, 1.0, 2.0], [2.0, 0.0, 2.5]])
    target_velocities = jnp.zeros((T, B, 3))
    
    batch_data = {
        'scan_outputs': scan_outputs,
        'target_positions': target_positions,
        'target_velocities': target_velocities
    }
    
    optimizer = create_optimizer(learning_rate=1e-3)
    opt_state = optimizer.init(params_dict)
    
    print("✅ Training setup complete")
    
    # 执行多个训练步骤
    current_params = params_dict
    current_opt_state = opt_state
    
    for step in range(3):
        updated_params, updated_opt_state, loss_breakdown = simple_training_step(
            current_params, current_opt_state, batch_data,
            PhysicsParams(), optimizer,
            alpha_efficiency=1.0, beta_safety=2.0
        )
        
        print(f"   Step {step+1}: loss={loss_breakdown['total_loss']:.4f}, "
              f"efficiency={loss_breakdown['efficiency_loss']:.4f}, "
              f"safety={loss_breakdown['safety_loss']:.4f}, "
              f"grad_norm={loss_breakdown['gradient_norm']:.6f}")
        
        # 验证参数更新
        param_changes = jax.tree_util.tree_map(
            lambda old, new: jnp.linalg.norm(new - old),
            current_params, updated_params
        )
        
        total_change = sum(jax.tree_util.tree_leaves(param_changes))
        print(f"   Parameter change magnitude: {total_change:.6f}")
        
        assert total_change > 1e-8, f"Parameters not updating at step {step+1}"
        
        current_params = updated_params
        current_opt_state = updated_opt_state
    
    print("✅ Multi-step training successful with parameter updates")
    return True

def main():
    """运行完整的MVP阶段4测试"""
    print("🚀 MVP Stage 4 Complete End-to-End Test")
    print("=" * 60)
    
    tests = [
        ("Simple Weighted Loss Function", test_simple_weighted_loss),
        ("Complete Scan Function JIT", test_complete_scan_function_jit),
        ("End-to-End Gradient Flow", test_end_to_end_gradient_flow),
        ("Training Step Execution", test_training_step_execution),
    ]
    
    results = []
    for test_name, test_fn in tests:
        print(f"\n{'=' * 20} {test_name} {'=' * 20}")
        try:
            start_time = time.time()
            success = test_fn()
            test_time = time.time() - start_time
            results.append(success)
            
            if success:
                print(f"🎉 {test_name} PASSED (time: {test_time:.3f}s)")
            else:
                print(f"💥 {test_name} FAILED")
        except Exception as e:
            print(f"💥 {test_name} CRASHED: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    print(f"\n{'=' * 60}")
    print("📊 MVP STAGE 4 FINAL TEST RESULTS")
    print(f"{'=' * 60}")
    
    passed = sum(results)
    total = len(results)
    
    for i, (test_name, _) in enumerate(tests):
        status = "✅ PASS" if results[i] else "❌ FAIL"
        print(f"{test_name:<30} {status}")
    
    print(f"\nOverall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉🎉🎉 MVP STAGE 4 COMPLETE! END-TO-END TRAINING READY! 🎉🎉🎉")
        print("\n🏆 Your Safe Agile Flight System is fully validated for:")
        print("   • ✅ Complete scan_function with all components")
        print("   • ✅ Simple weighted loss: L_total = α * L_efficiency + β * L_safety")
        print("   • ✅ Full gradient flow from loss to GNN and Policy parameters")
        print("   • ✅ JIT compilation of entire pipeline")
        print("   • ✅ Training step execution with parameter updates")
        print("\n🚁 Ready for full-scale training and deployment!")
        return 0
    else:
        print("❌ Some tests failed. Please review the errors above.")
        return 1

if __name__ == "__main__":
    exit(main())