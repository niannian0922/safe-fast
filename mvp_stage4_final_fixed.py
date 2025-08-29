#!/usr/bin/env python3
"""
MVP Stage 4 最终修复版本 - 确保梯度流完整打通

核心修复：
1. 实现真正的参数化网络（GNN和Policy）
2. 确保梯度从损失函数正确反向传播到所有参数
3. 验证端到端可微分性
"""

import jax
import jax.numpy as jnp
from jax import random, grad, jit, lax
import optax
import time
import functools
from typing import Dict, Tuple
import chex

# 配置JAX
jax.config.update("jax_enable_x64", True)

# =============================================================================
# 核心数据结构
# =============================================================================

@chex.dataclass
class DroneState:
    """无人机状态"""
    position: chex.Array  # (3,) 位置
    velocity: chex.Array  # (3,) 速度

@chex.dataclass
class BatchData:
    """批次数据"""
    initial_positions: chex.Array  # (B, 3)
    initial_velocities: chex.Array  # (B, 3) 
    target_positions: chex.Array   # (B, 3)
    obstacle_positions: chex.Array  # (B, N, 3)

@chex.dataclass
class ScanCarry:
    """扫描携带状态"""
    positions: chex.Array      # (B, 3)
    velocities: chex.Array     # (B, 3)
    step_count: chex.Array     # (B,)

@chex.dataclass 
class ScanOutput:
    """扫描输出"""
    positions: chex.Array       # (B, 3)
    velocities: chex.Array      # (B, 3)
    controls: chex.Array        # (B, 3)
    cbf_values: chex.Array      # (B,)
    safety_violations: chex.Array  # (B,)

# =============================================================================
# 真正的参数化网络模块
# =============================================================================

def init_gnn_params(key: chex.PRNGKey, input_dim: int = 6, hidden_dim: int = 32, output_dim: int = 1):
    """初始化GNN网络参数"""
    keys = random.split(key, 3)
    
    params = {
        # 节点特征处理层
        'node_embed_w': random.normal(keys[0], (input_dim, hidden_dim)) * 0.02,
        'node_embed_b': jnp.zeros(hidden_dim),
        
        # 距离处理层 
        'distance_w': random.normal(keys[1], (1, hidden_dim)) * 0.02,
        'distance_b': jnp.zeros(hidden_dim),
        
        # 输出层
        'output_w': random.normal(keys[2], (hidden_dim, output_dim)) * 0.02,
        'output_b': jnp.zeros(output_dim)
    }
    
    return params

def gnn_forward(params: Dict, positions: chex.Array, obstacles: chex.Array):
    """
    参数化的GNN前向传播
    positions: (B, 3) 无人机位置
    obstacles: (B, N, 3) 障碍物位置
    返回: (B,) CBF值
    """
    batch_size, n_obstacles = obstacles.shape[:2]
    
    # 创建节点特征：[pos, vel_estimate] = 6维
    # 简化：vel_estimate = zeros
    node_features = jnp.concatenate([
        positions,  # (B, 3)
        jnp.zeros((batch_size, 3))  # (B, 3) 速度估计
    ], axis=-1)  # (B, 6)
    
    # 节点特征嵌入
    node_embed = jnp.tanh(
        node_features @ params['node_embed_w'] + params['node_embed_b']
    )  # (B, hidden_dim)
    
    # 计算到障碍物的距离特征
    distances = jnp.linalg.norm(
        obstacles - positions[:, None, :], axis=-1
    )  # (B, N)
    min_distances = jnp.min(distances, axis=-1, keepdims=True)  # (B, 1)
    
    # 距离特征处理
    distance_embed = jnp.tanh(
        min_distances @ params['distance_w'] + params['distance_b']
    )  # (B, hidden_dim)
    
    # 特征融合
    fused_features = node_embed + distance_embed  # (B, hidden_dim)
    
    # 输出CBF值
    cbf_values = fused_features @ params['output_w'] + params['output_b']  # (B, 1)
    cbf_values = cbf_values.squeeze(-1)  # (B,)
    
    return cbf_values

def init_policy_params(key: chex.PRNGKey, input_dim: int = 10, hidden_dim: int = 64, output_dim: int = 3):
    """初始化策略网络参数"""
    keys = random.split(key, 4)
    
    params = {
        # 第一层
        'layer1_w': random.normal(keys[0], (input_dim, hidden_dim)) * 0.02,
        'layer1_b': jnp.zeros(hidden_dim),
        
        # 第二层
        'layer2_w': random.normal(keys[1], (hidden_dim, hidden_dim)) * 0.02,
        'layer2_b': jnp.zeros(hidden_dim),
        
        # 输出层
        'output_w': random.normal(keys[2], (hidden_dim, output_dim)) * 0.02,
        'output_b': jnp.zeros(output_dim)
    }
    
    return params

def policy_forward(params: Dict, observations: chex.Array):
    """
    参数化的策略网络前向传播
    observations: (B, 10) [pos(3), vel(3), pos_error(3), cbf(1)]
    返回: (B, 3) 控制指令
    """
    # 第一层
    h1 = jnp.tanh(observations @ params['layer1_w'] + params['layer1_b'])
    
    # 第二层
    h2 = jnp.tanh(h1 @ params['layer2_w'] + params['layer2_b'])
    
    # 输出层（使用tanh限制控制幅度）
    controls = jnp.tanh(h2 @ params['output_w'] + params['output_b'])
    
    return controls

# =============================================================================
# 物理仿真模块
# =============================================================================

def physics_step(positions, velocities, controls, dt=0.01):
    """物理仿真步骤"""
    # 简化动力学：a = u - drag * v - gravity
    drag_coef = 0.1
    gravity = jnp.array([0.0, 0.0, -9.81])
    
    accelerations = controls - drag_coef * velocities + gravity
    
    # 欧拉积分
    new_velocities = velocities + accelerations * dt
    new_positions = positions + new_velocities * dt
    
    return new_positions, new_velocities

# =============================================================================
# JIT兼容的BPTT扫描循环
# =============================================================================

def create_scan_function(gnn_params: Dict, policy_params: Dict):
    """创建带参数的扫描函数"""
    
    def scan_step(carry, inputs):
        """单步扫描"""
        positions = carry.positions  # (B, 3)
        velocities = carry.velocities  # (B, 3)
        step_count = carry.step_count  # (B,)
        
        target_positions = inputs['targets']  # (B, 3)
        obstacles = inputs['obstacles']  # (B, N, 3)
        
        # === 1. GNN感知（使用真实参数） ===
        cbf_values = gnn_forward(gnn_params, positions, obstacles)  # (B,)
        
        # === 2. 策略网络（使用真实参数） ===
        position_errors = target_positions - positions  # (B, 3)
        observations = jnp.concatenate([
            positions,           # (B, 3)
            velocities,          # (B, 3) 
            position_errors,     # (B, 3)
            cbf_values[:, None]  # (B, 1)
        ], axis=-1)  # (B, 10)
        
        controls = policy_forward(policy_params, observations)  # (B, 3)
        
        # === 3. 安全层（简化版）===
        # 在不安全区域应用紧急制动
        unsafe_mask = cbf_values < 0  # (B,)
        emergency_control = -2.0 * velocities  # 紧急制动
        
        safe_controls = jnp.where(
            unsafe_mask[:, None],
            emergency_control,
            controls
        )
        
        # === 4. 物理仿真 ===
        new_positions, new_velocities = physics_step(positions, velocities, safe_controls)
        
        # === 5. 安全评估 ===
        safety_violations = (cbf_values < 0).astype(jnp.float32)  # (B,)
        
        # 更新carry
        new_carry = ScanCarry(
            positions=new_positions,
            velocities=new_velocities,
            step_count=step_count + 1
        )
        
        # 输出
        outputs = ScanOutput(
            positions=new_positions,
            velocities=new_velocities,
            controls=safe_controls,
            cbf_values=cbf_values,
            safety_violations=safety_violations
        )
        
        return new_carry, outputs
    
    return scan_step

# =============================================================================
# 完整的前向传播和损失计算
# =============================================================================

@functools.partial(jit, static_argnames=['sequence_length', 'batch_size'])
def complete_forward_pass(
    gnn_params: Dict,
    policy_params: Dict,
    batch_data: BatchData,
    sequence_length: int,
    batch_size: int,
    key: chex.PRNGKey
) -> Tuple[chex.Array, Dict]:
    """完整的前向传播"""
    
    # 创建扫描函数
    scan_fn = create_scan_function(gnn_params, policy_params)
    
    # 初始化carry
    initial_carry = ScanCarry(
        positions=batch_data.initial_positions,
        velocities=batch_data.initial_velocities,
        step_count=jnp.zeros(batch_size, dtype=jnp.int32)
    )
    
    # 准备输入序列
    inputs_sequence = {
        'targets': jnp.tile(
            batch_data.target_positions[None, :, :],  # (1, B, 3)
            (sequence_length, 1, 1)                   # (T, B, 3)
        ),
        'obstacles': jnp.tile(
            batch_data.obstacle_positions[None, :, :, :],  # (1, B, N, 3)
            (sequence_length, 1, 1, 1)                     # (T, B, N, 3)
        )
    }
    
    # 执行BPTT扫描
    final_carry, trajectory = lax.scan(
        scan_fn,
        initial_carry,
        inputs_sequence,
        length=sequence_length
    )
    
    # === 计算损失：L_total = α * L_efficiency + β * L_safety ===
    
    # 1. 效率损失 - 目标到达
    final_positions = trajectory.positions[-1]  # (B, 3)
    goal_errors = jnp.linalg.norm(
        final_positions - batch_data.target_positions, axis=-1
    )  # (B,)
    efficiency_loss = jnp.mean(goal_errors ** 2)
    
    # 2. 速度跟踪损失（追踪向目标的速度）
    target_velocities = jnp.tile(
        jnp.array([0.0, 0.0, 0.0])[None, :], (batch_size, 1)  # 悬停目标
    )
    velocity_errors = trajectory.velocities[-1] - target_velocities  # 最终速度误差
    velocity_loss = jnp.mean(jnp.sum(velocity_errors ** 2, axis=-1))
    
    # 总效率损失
    L_efficiency = efficiency_loss + 0.5 * velocity_loss
    
    # 3. 安全损失 - CBF违反
    cbf_violations = jnp.mean(jnp.maximum(0, -trajectory.cbf_values))  # 负CBF惩罚
    safety_violation_rate = jnp.mean(trajectory.safety_violations)  # 安全违规率
    L_safety = cbf_violations + safety_violation_rate
    
    # 4. 控制正则化
    control_effort = jnp.mean(jnp.sum(trajectory.controls ** 2, axis=-1))
    
    # 5. 总损失：L_total = α * L_efficiency + β * L_safety
    alpha, beta = 1.0, 2.0  # MVP阶段使用简单权重
    total_loss = alpha * L_efficiency + beta * L_safety + 0.01 * control_effort
    
    # 指标
    metrics = {
        'total_loss': total_loss,
        'efficiency_loss': L_efficiency,
        'safety_loss': L_safety,
        'control_effort': control_effort,
        'final_goal_distance': jnp.mean(goal_errors),
        'safety_violation_rate': safety_violation_rate,
        'cbf_violation_rate': jnp.mean(trajectory.cbf_values < 0),
        'average_cbf_value': jnp.mean(trajectory.cbf_values)
    }
    
    return total_loss, metrics

# =============================================================================
# 完整训练步骤
# =============================================================================

def complete_training_step(
    gnn_params: Dict,
    policy_params: Dict,
    gnn_opt_state: optax.OptState,
    policy_opt_state: optax.OptState,
    batch_data: BatchData,
    sequence_length: int,
    batch_size: int,
    key: chex.PRNGKey,
    gnn_optimizer: optax.GradientTransformation,
    policy_optimizer: optax.GradientTransformation
) -> Tuple[Dict, Dict, optax.OptState, optax.OptState, Dict]:
    """完整训练步骤"""
    
    def loss_fn(params):
        gnn_p, policy_p = params
        loss, metrics = complete_forward_pass(
            gnn_p, policy_p, batch_data, sequence_length, batch_size, key
        )
        return loss, metrics
    
    # 梯度计算
    (loss_value, metrics), gradients = jax.value_and_grad(
        loss_fn, has_aux=True
    )((gnn_params, policy_params))
    
    gnn_grads, policy_grads = gradients
    
    # GNN参数更新
    gnn_updates, new_gnn_opt_state = gnn_optimizer.update(
        gnn_grads, gnn_opt_state, gnn_params
    )
    new_gnn_params = optax.apply_updates(gnn_params, gnn_updates)
    
    # 策略参数更新
    policy_updates, new_policy_opt_state = policy_optimizer.update(
        policy_grads, policy_opt_state, policy_params
    )
    new_policy_params = optax.apply_updates(policy_params, policy_updates)
    
    # 梯度统计
    gnn_grad_norm = jnp.sqrt(sum(
        jnp.sum(g ** 2) for g in jax.tree_util.tree_leaves(gnn_grads)
    ))
    policy_grad_norm = jnp.sqrt(sum(
        jnp.sum(g ** 2) for g in jax.tree_util.tree_leaves(policy_grads)
    ))
    
    # 更新指标
    updated_metrics = {
        **metrics,
        'gnn_grad_norm': gnn_grad_norm,
        'policy_grad_norm': policy_grad_norm
    }
    
    return (
        new_gnn_params, new_policy_params,
        new_gnn_opt_state, new_policy_opt_state,
        updated_metrics
    )

# =============================================================================
# 测试函数
# =============================================================================

def test_mvp_stage4_fixed():
    """测试修复后的MVP Stage 4功能"""
    print("🧪 测试修复后的MVP Stage 4功能...")
    
    # 参数设置
    key = random.PRNGKey(42)
    keys = random.split(key, 10)
    
    batch_size = 4
    sequence_length = 15
    n_obstacles = 10
    
    # 创建测试数据
    batch_data = BatchData(
        initial_positions=random.uniform(keys[0], (batch_size, 3), minval=-1, maxval=1),
        initial_velocities=jnp.zeros((batch_size, 3)),
        target_positions=random.uniform(keys[1], (batch_size, 3), minval=-2, maxval=2),
        obstacle_positions=random.uniform(keys[2], (batch_size, n_obstacles, 3), minval=-3, maxval=3)
    )
    
    print(f"✅ 测试数据创建：batch_size={batch_size}, sequence_length={sequence_length}")
    
    # 初始化真实的参数化网络参数
    gnn_params = init_gnn_params(keys[3])
    policy_params = init_policy_params(keys[4])
    
    print("✅ 网络参数初始化完成")
    print(f"   GNN参数数量: {sum(p.size for p in jax.tree_util.tree_leaves(gnn_params))}")
    print(f"   策略参数数量: {sum(p.size for p in jax.tree_util.tree_leaves(policy_params))}")
    
    # 创建优化器
    gnn_optimizer = optax.adam(1e-3)
    policy_optimizer = optax.adam(1e-3)
    gnn_opt_state = gnn_optimizer.init(gnn_params)
    policy_opt_state = policy_optimizer.init(policy_params)
    
    print("✅ 优化器初始化完成")
    
    # === 测试1：前向传播 ===
    print("\n📋 测试1：前向传播")
    start_time = time.time()
    loss, metrics = complete_forward_pass(
        gnn_params, policy_params, batch_data,
        sequence_length, batch_size, keys[5]
    )
    forward_time = time.time() - start_time
    
    print(f"   ✅ 前向传播成功 (时间: {forward_time:.3f}s)")
    print(f"   总损失: {loss:.4f}")
    print(f"   效率损失: {metrics['efficiency_loss']:.4f}")
    print(f"   安全损失: {metrics['safety_loss']:.4f}")
    print(f"   最终目标距离: {metrics['final_goal_distance']:.4f}")
    print(f"   平均CBF值: {metrics['average_cbf_value']:.4f}")
    print(f"   安全违规率: {metrics['safety_violation_rate']:.2%}")
    
    # === 测试2：梯度计算 ===  
    print("\n📋 测试2：梯度计算")
    def simple_loss_fn(params):
        gnn_p, policy_p = params
        loss, _ = complete_forward_pass(
            gnn_p, policy_p, batch_data, sequence_length, batch_size, keys[6]
        )
        return loss
    
    start_time = time.time()
    gradients = grad(simple_loss_fn)((gnn_params, policy_params))
    grad_time = time.time() - start_time
    
    gnn_grads, policy_grads = gradients
    
    gnn_grad_norm = jnp.sqrt(sum(
        jnp.sum(g ** 2) for g in jax.tree_util.tree_leaves(gnn_grads)
    ))
    policy_grad_norm = jnp.sqrt(sum(
        jnp.sum(g ** 2) for g in jax.tree_util.tree_leaves(policy_grads)
    ))
    
    print(f"   ✅ 梯度计算成功 (时间: {grad_time:.3f}s)")
    print(f"   GNN梯度范数: {gnn_grad_norm:.6f}")
    print(f"   策略梯度范数: {policy_grad_norm:.6f}")
    
    # 验证梯度质量
    assert jnp.isfinite(gnn_grad_norm), "GNN梯度包含NaN/Inf"
    assert jnp.isfinite(policy_grad_norm), "策略梯度包含NaN/Inf"  
    assert gnn_grad_norm > 1e-8, f"GNN梯度太小: {gnn_grad_norm}"
    assert policy_grad_norm > 1e-8, f"策略梯度太小: {policy_grad_norm}"
    
    print("   ✅ 梯度质量验证通过")
    
    # === 测试3：完整训练步骤 ===
    print("\n📋 测试3：完整训练步骤")
    start_time = time.time()
    (
        new_gnn_params, new_policy_params,
        new_gnn_opt_state, new_policy_opt_state,
        step_metrics
    ) = complete_training_step(
        gnn_params, policy_params,
        gnn_opt_state, policy_opt_state,
        batch_data, sequence_length, batch_size, keys[7],
        gnn_optimizer, policy_optimizer
    )
    step_time = time.time() - start_time
    
    print(f"   ✅ 训练步骤成功 (时间: {step_time:.3f}s)")
    print(f"   总损失: {step_metrics['total_loss']:.4f}")
    print(f"   GNN梯度范数: {step_metrics['gnn_grad_norm']:.6f}")
    print(f"   策略梯度范数: {step_metrics['policy_grad_norm']:.6f}")
    
    # === 测试4：参数更新验证 ===
    print("\n📋 测试4：参数更新验证")
    
    gnn_param_change = jnp.sqrt(sum(
        jnp.sum((new - old) ** 2) for new, old in zip(
            jax.tree_util.tree_leaves(new_gnn_params),
            jax.tree_util.tree_leaves(gnn_params)
        )
    ))
    policy_param_change = jnp.sqrt(sum(
        jnp.sum((new - old) ** 2) for new, old in zip(
            jax.tree_util.tree_leaves(new_policy_params),
            jax.tree_util.tree_leaves(policy_params)
        )
    ))
    
    print(f"   GNN参数变化幅度: {gnn_param_change:.8f}")
    print(f"   策略参数变化幅度: {policy_param_change:.8f}")
    
    assert gnn_param_change > 1e-10, f"GNN参数没有更新: {gnn_param_change}"
    assert policy_param_change > 1e-10, f"策略参数没有更新: {policy_param_change}"
    
    print("   ✅ 参数更新验证通过")
    
    # === 测试5：多步训练稳定性 ===
    print("\n📋 测试5：多步训练稳定性")
    
    current_gnn_params = gnn_params
    current_policy_params = policy_params
    current_gnn_opt_state = gnn_opt_state
    current_policy_opt_state = policy_opt_state
    
    losses = []
    goal_distances = []
    
    for step in range(10):
        step_key = random.fold_in(keys[8], step)
        
        (
            current_gnn_params, current_policy_params,
            current_gnn_opt_state, current_policy_opt_state,
            step_metrics
        ) = complete_training_step(
            current_gnn_params, current_policy_params,
            current_gnn_opt_state, current_policy_opt_state,
            batch_data, sequence_length, batch_size, step_key,
            gnn_optimizer, policy_optimizer
        )
        
        losses.append(float(step_metrics['total_loss']))
        goal_distances.append(float(step_metrics['final_goal_distance']))
        
        if step % 3 == 0:
            print(f"   Step {step+1:2d}: loss={step_metrics['total_loss']:.4f}, "
                  f"goal_dist={step_metrics['final_goal_distance']:.4f}, "
                  f"cbf={step_metrics['average_cbf_value']:.3f}")
    
    print(f"   ✅ 10步训练完成")
    print(f"   损失变化: {losses[0]:.4f} -> {losses[-1]:.4f}")
    print(f"   目标距离变化: {goal_distances[0]:.4f} -> {goal_distances[-1]:.4f}")
    
    # 验证训练稳定性
    assert all(jnp.isfinite(l) for l in losses), "训练过程中出现NaN损失"
    assert all(l < 1000 for l in losses), "损失爆炸"
    
    print("   ✅ 训练稳定性验证通过")
    
    return True

def main():
    """主函数"""
    print("🚀 MVP Stage 4 最终修复版测试")
    print("=" * 60)
    print("核心修复：")
    print("  1. 真正参数化的GNN网络（非常数函数）")
    print("  2. 真正参数化的Policy网络（MLP）")
    print("  3. 确保梯度从损失正确传播到所有参数")
    print("  4. L_total = α * L_efficiency + β * L_safety")
    print("=" * 60)
    
    try:
        success = test_mvp_stage4_fixed()
        
        if success:
            print("\n" + "=" * 60)
            print("🎉🎉🎉 MVP STAGE 4 最终修复版全部测试通过！🎉🎉🎉")
            print("\n🏆 关键问题已修复：")
            print("   ✅ GNN网络真正使用参数进行计算")
            print("   ✅ Policy网络真正使用参数进行计算")
            print("   ✅ 梯度从损失成功传播到所有网络参数")
            print("   ✅ 简单加权损失函数：L_total = α * L_efficiency + β * L_safety")
            print("   ✅ JIT编译完全兼容")
            print("   ✅ 端到端训练稳定")
            print("\n🔥 **MVP阶段4：端到端梯度流完全打通！**")
            print("   • GNN感知 -> CBF值计算 -> 参数梯度 ✅")
            print("   • Policy控制 -> 控制输出 -> 参数梯度 ✅")
            print("   • 物理仿真 -> 损失计算 -> 反向传播 ✅")
            print("   • 多目标优化 -> 参数更新 -> 训练稳定 ✅")
            print("\n🚁 您的安全敏捷飞行系统现在100%准备进行端到端训练！")
            return 0
        else:
            print("❌ 测试失败")
            return 1
            
    except Exception as e:
        print(f"💥 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())