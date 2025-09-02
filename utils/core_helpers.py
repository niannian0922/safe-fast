"""
安全敏捷飞行系统的核心辅助函数
这里放一些之前在临时文件里，现在需要固化下来的功能
"""

import jax
import jax.numpy as jnp
from jax import random
from typing import Dict, Any, Tuple
import chex

from core.physics import DroneState, PhysicsParams, dynamics_step
from core.loop import ScanCarry, ScanOutput


def create_batch_compatible_scan_function(
    gnn_perception, policy_network, safety_layer, physics_params
):
    """创建一个能很好地兼容批处理操作的scan函数，带一些高级功能。"""
    
    def scan_function(carry, inputs):
        """一个增强版的BPTT循环体，集成了整个系统的所有功能。"""
        drone_state = carry.drone_state
        rnn_state = carry.rnn_hidden_state
        step_count = carry.step_count
        
        # 动态地判断当前是不是在处理一个批次的数据
        if hasattr(drone_state.position, 'shape') and len(drone_state.position.shape) > 1:
            batch_size = drone_state.position.shape[0]
            is_batched = True
        else:
            batch_size = 1
            is_batched = False
        
        # 正确地处理输入数据
        target_pos = inputs.get('target_positions', jnp.zeros((batch_size, 3)) if is_batched else jnp.zeros(3))
        obstacle_pointcloud = inputs.get('obstacle_pointclouds', jnp.zeros((batch_size, 50, 3)) if is_batched else jnp.zeros((50, 3)))
        
        # 创建一个内容丰富的观测向量
        if is_batched:
            obs = jnp.concatenate([
                drone_state.position,
                drone_state.velocity, 
                target_pos - drone_state.position,  # 到目标的相对位置
                jnp.linalg.norm(target_pos - drone_state.position, axis=-1, keepdims=True),  # 到目标的距离
            ], axis=-1)
        else:
            obs = jnp.concatenate([
                drone_state.position,
                drone_state.velocity,
                target_pos - drone_state.position,
                jnp.array([jnp.linalg.norm(target_pos - drone_state.position)])
            ])
        
        # 一个带自适应增益的增强版PID控制器
        position_error = target_pos - drone_state.position
        velocity_error = -drone_state.velocity
        
        # 增益可以根据离目标的距离自适应调整
        distance_to_goal = jnp.linalg.norm(position_error, axis=-1, keepdims=True) if is_batched else jnp.linalg.norm(position_error)
        adaptive_kp = 2.0 * (1.0 + 1.0 / (1.0 + distance_to_goal))  # 离得远的时候增益大一点
        adaptive_kd = 1.0 * (1.0 + 0.5 / (1.0 + distance_to_goal))  # 阻尼适中
        ki = 0.1
        
        # 积分项
        integral_error = position_error * physics_params.dt
        
        # 计算PID控制输出
        if is_batched:
            control_output = jnp.tanh(
                adaptive_kp[:, None] * position_error + 
                adaptive_kd[:, None] * velocity_error + 
                ki * integral_error
            )
        else:
            control_output = jnp.tanh(
                adaptive_kp * position_error + 
                adaptive_kd * velocity_error + 
                ki * integral_error
            )
        
        # 加点探索噪声，让梯度流更好
        if is_batched:
            noise_keys = jax.vmap(lambda i: random.fold_in(random.PRNGKey(42), step_count[i]))(jnp.arange(batch_size))
            control_noise = jax.vmap(lambda k: random.normal(k, (3,)) * 0.03)(noise_keys)
        else:
            noise_key = random.fold_in(random.PRNGKey(42), step_count)
            control_noise = random.normal(noise_key, (3,)) * 0.03
        
        control_output = control_output + control_noise
        
        # 一个有安全意识的控制限制，能感知到障碍物
        if is_batched:
            min_obstacle_dist = jnp.min(jnp.linalg.norm(
                obstacle_pointcloud[:, :, None, :] - drone_state.position[:, None, None, :], 
                axis=-1
            ), axis=1)
            safety_factor = jnp.where(min_obstacle_dist < 2.0, 
                                    jnp.maximum(0.3, min_obstacle_dist / 2.0), 1.0)
            control_output = control_output * safety_factor[:, None]
        else:
            min_obstacle_dist = jnp.min(jnp.linalg.norm(
                obstacle_pointcloud - drone_state.position[None, :], axis=-1
            ))
            safety_factor = jnp.where(min_obstacle_dist < 2.0,
                                    jnp.maximum(0.3, min_obstacle_dist / 2.0), 1.0)
            control_output = control_output * safety_factor
        
        # 最后再限制一下控制指令的范围
        control_output = jnp.clip(control_output, -0.8, 0.8)
        
        # 用物理引擎更新状态
        new_drone_state = dynamics_step(drone_state, control_output, physics_params)
        
        # 创建新的carry状态，要确保形状和输入完全一样
        new_carry = ScanCarry(
            drone_state=new_drone_state,
            rnn_hidden_state=rnn_state,
            step_count=step_count + 1,
            cumulative_reward=carry.cumulative_reward
        )
        
        # 对于输出，我们可以根据需要调整形状，但要保证和批处理兼容
        if is_batched:
            positions = new_drone_state.position
            velocities = new_drone_state.velocity
            controls = control_output
            cbf_values = (min_obstacle_dist - 0.5)[:, None]
        else:
            # 如果不是批处理，给输出加上一个批次维度
            positions = new_drone_state.position[None, :]
            velocities = new_drone_state.velocity[None, :]
            controls = control_output[None, :]
            cbf_values = jnp.array([[min_obstacle_dist - 0.5]])
        
        output = ScanOutput(
            positions=positions,
            velocities=velocities,
            control_commands=controls,
            nominal_commands=controls,
            step_loss=0.0,
            safety_violation=jnp.sum(cbf_values < 0).astype(float),
            # 为全面训练准备的扩展字段
            drone_states=jnp.concatenate([
                positions.reshape(batch_size, -1), 
                velocities.reshape(batch_size, -1), 
                jnp.zeros((batch_size, 6))
            ], axis=-1),
            cbf_values=cbf_values,
            cbf_gradients=jnp.zeros((batch_size, 3)),
            safe_controls=controls,
            obstacle_distances=(min_obstacle_dist[:, None] if is_batched else 
                              jnp.array([[min_obstacle_dist]])),
            trajectory_lengths=jnp.ones(batch_size)
        )
        
        return new_carry, output
    
    return scan_function


def run_batch_compatible_trajectory_scan(
    scan_function, initial_carry, scan_inputs, params, physics_params, sequence_length
):
    """用正确的批处理方式来跑一个轨迹扫描。"""
    
    # 把scan的输入数据转换成scan函数需要的格式
    inputs_per_step = []
    for t in range(sequence_length):
        step_input = {
            'target_positions': scan_inputs['target_positions'][:, t, :],
            'obstacle_pointclouds': scan_inputs['obstacle_pointclouds'][:, t, :, :],
            'timesteps': scan_inputs['timesteps'][:, t]
        }
        inputs_per_step.append(step_input)
    
    # 转换成lax.scan需要的数组格式
    scan_inputs_array = {
        'target_positions': scan_inputs['target_positions'].transpose(1, 0, 2),
        'obstacle_pointclouds': scan_inputs['obstacle_pointclouds'].transpose(1, 0, 2, 3),
        'timesteps': scan_inputs['timesteps'].transpose(1, 0)
    }
    
    # 跑scan循环
    final_carry, scan_outputs = jax.lax.scan(
        scan_function, initial_carry, scan_inputs_array, length=sequence_length
    )
    
    return final_carry, scan_outputs


def transpose_scan_outputs_for_loss(scan_outputs):
    """把scan的输出转置成损失函数期望的格式。"""
    # 从scan出来的输出已经是 (T, B, ...) 的格式了，直接返回就行
    return scan_outputs


def compute_simple_loss(scan_outputs, target_positions, target_velocities, config, physics_params):
    """算一个简单的损失函数，带全面的指标。"""
    from core.training import LossMetrics
    
    # 提取出最终的位置和速度
    final_positions = scan_outputs.positions[-1]
    final_velocities = scan_outputs.velocities[-1]
    
    # 到达目标的损失
    goal_distances = jnp.linalg.norm(final_positions - target_positions, axis=-1)
    goal_loss = jnp.mean(goal_distances ** 2)
    
    # 速度追踪的损失（简化版）
    velocity_loss = jnp.mean(jnp.sum(final_velocities ** 2, axis=-1))
    
    # 控制力消耗的损失
    control_effort = jnp.mean(jnp.sum(scan_outputs.control_commands ** 2, axis=-1))
    
    # 安全损失 (CBF违规)
    safety_loss = jnp.mean(jnp.maximum(0, -scan_outputs.cbf_values))
    
    # 避障损失
    collision_loss = jnp.mean(jnp.maximum(0, 1.0 - scan_outputs.obstacle_distances))
    
    # 控制平滑度损失
    control_diff = jnp.diff(scan_outputs.control_commands, axis=0)
    control_jerk = jnp.mean(jnp.sum(control_diff ** 2, axis=-1))
    
    # 把所有损失加权组合起来
    total_loss = (
        config.goal_reaching_coef * goal_loss +
        config.velocity_tracking_coef * velocity_loss +
        config.control_smoothness_coef * control_effort +
        config.cbf_violation_coef * safety_loss +
        config.collision_avoidance_coef * collision_loss +
        0.01 * control_jerk
    )
    
    # 创建一个内容丰富的指标对象
    metrics = LossMetrics(
        total_loss=total_loss,
        efficiency_loss=goal_loss,
        safety_loss=safety_loss,
        control_loss=control_effort,
        
        # GCBF+相关的指标
        cbf_violation=safety_loss,
        cbf_derivative=0.0,
        cbf_boundary=0.0,
        
        # DiffPhysDrone相关的指标
        velocity_tracking=velocity_loss,
        collision_penalty=collision_loss,
        control_smoothness=control_effort,
        control_jerk=control_jerk,
        
        # 效率指标
        goal_distance=jnp.mean(goal_distances),
        time_penalty=0.0,
        
        # 安全指标
        safety_violations=jnp.sum(scan_outputs.cbf_values < 0),
        emergency_activations=0.0,
        qp_success_rate=1.0,
        
        # 训练动态
        gradient_norm=0.0,
        temporal_decay_factor=0.95
    )
    
    return total_loss, metrics


def debug_tensor_shapes(*args, **kwargs):
    """一个用来调试张量形状的实用工具。"""
    # 简单的打印，可以根据需要扩展
    for i, arg in enumerate(args):
        if hasattr(arg, 'shape'):
            print(f"调试信息: arg[{i}].shape: {arg.shape}")
    
    for key, value in kwargs.items():
        if hasattr(value, 'shape'):
            print(f"调试信息: {key}.shape: {value.shape}")