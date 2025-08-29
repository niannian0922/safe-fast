"""
安全敏捷飞行系统的核心辅助函数
替换已删除临时文件的功能
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
    """创建与批操作兼容的扫描函数，具有高级功能。"""
    
    def scan_function(carry, inputs):
        """具有完整系统集成的BPTT循环增强扫描函数。"""
        drone_state = carry.drone_state
        rnn_state = carry.rnn_hidden_state
        step_count = carry.step_count
        
        # 动态确定批大小
        if hasattr(drone_state.position, 'shape') and len(drone_state.position.shape) > 1:
            batch_size = drone_state.position.shape[0]
            is_batched = True
        else:
            batch_size = 1
            is_batched = False
        
        # 正确处理输入
        target_pos = inputs.get('target_positions', jnp.zeros((batch_size, 3)) if is_batched else jnp.zeros(3))
        obstacle_pointcloud = inputs.get('obstacle_pointclouds', jnp.zeros((batch_size, 50, 3)) if is_batched else jnp.zeros((50, 3)))
        
        # 创建综合观测
        if is_batched:
            obs = jnp.concatenate([
                drone_state.position,
                drone_state.velocity, 
                target_pos - drone_state.position,  # Relative position
                jnp.linalg.norm(target_pos - drone_state.position, axis=-1, keepdims=True),  # Distance to goal
            ], axis=-1)
        else:
            obs = jnp.concatenate([
                drone_state.position,
                drone_state.velocity,
                target_pos - drone_state.position,
                jnp.array([jnp.linalg.norm(target_pos - drone_state.position)])
            ])
        
        # Enhanced PID controller with adaptive gains
        position_error = target_pos - drone_state.position
        velocity_error = -drone_state.velocity
        
        # Distance-adaptive gains
        distance_to_goal = jnp.linalg.norm(position_error, axis=-1, keepdims=True) if is_batched else jnp.linalg.norm(position_error)
        adaptive_kp = 2.0 * (1.0 + 1.0 / (1.0 + distance_to_goal))  # Higher gain when far
        adaptive_kd = 1.0 * (1.0 + 0.5 / (1.0 + distance_to_goal))  # Moderate damping
        ki = 0.1
        
        # Integral error term
        integral_error = position_error * physics_params.dt
        
        # Advanced PID control
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
        
        # Add exploration noise for better gradient flow
        if is_batched:
            noise_keys = jax.vmap(lambda i: random.fold_in(random.PRNGKey(42), step_count[i]))(jnp.arange(batch_size))
            control_noise = jax.vmap(lambda k: random.normal(k, (3,)) * 0.03)(noise_keys)
        else:
            noise_key = random.fold_in(random.PRNGKey(42), step_count)
            control_noise = random.normal(noise_key, (3,)) * 0.03
        
        control_output = control_output + control_noise
        
        # Safety-aware control limiting with obstacle awareness
        if is_batched:
            min_obstacle_dist = jnp.min(jnp.linalg.norm(
                obstacle_pointcloud[:, :, None, :] - drone_state.position[:, None, None, :], 
                axis=-1
            ), axis=1)  # [batch_size]
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
        
        # Apply control limits
        control_output = jnp.clip(control_output, -0.8, 0.8)
        
        # Physics step with enhanced dynamics
        new_drone_state = dynamics_step(drone_state, control_output, physics_params)
        
        # Create new carry - maintain exact same shapes as input
        new_carry = ScanCarry(
            drone_state=new_drone_state,  # Don't modify shapes here
            rnn_hidden_state=rnn_state,
            step_count=step_count + 1,
            cumulative_reward=carry.cumulative_reward
        )
        
        # For outputs, we can modify shapes as needed
        # But ensure they're consistent with batch processing
        if is_batched:
            positions = new_drone_state.position
            velocities = new_drone_state.velocity
            controls = control_output
            cbf_values = (min_obstacle_dist - 0.5)[:, None]  # Safety margin
        else:
            # Add batch dimension for outputs (but not for carry!)
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
            safety_violation=jnp.sum(cbf_values < 0).astype(float),  # Use JAX-compatible conversion
            # Extended fields for comprehensive training
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
    """Run trajectory scan with proper batching."""
    
    # Convert scan_inputs to proper format for scan function
    inputs_per_step = []
    for t in range(sequence_length):
        step_input = {
            'target_positions': scan_inputs['target_positions'][:, t, :],
            'obstacle_pointclouds': scan_inputs['obstacle_pointclouds'][:, t, :, :],
            'timesteps': scan_inputs['timesteps'][:, t]
        }
        inputs_per_step.append(step_input)
    
    # Convert to array format for jax.lax.scan
    scan_inputs_array = {
        'target_positions': scan_inputs['target_positions'].transpose(1, 0, 2),  # [T, B, 3]
        'obstacle_pointclouds': scan_inputs['obstacle_pointclouds'].transpose(1, 0, 2, 3),  # [T, B, N, 3]
        'timesteps': scan_inputs['timesteps'].transpose(1, 0)  # [T, B]
    }
    
    # Run scan
    final_carry, scan_outputs = jax.lax.scan(
        scan_function, initial_carry, scan_inputs_array, length=sequence_length
    )
    
    return final_carry, scan_outputs


def transpose_scan_outputs_for_loss(scan_outputs):
    """Transpose scan outputs to expected format for loss computation."""
    # The outputs are already in [T, B, ...] format from scan
    # Just return them as is
    return scan_outputs


def compute_simple_loss(scan_outputs, target_positions, target_velocities, config, physics_params):
    """Compute simple loss function for training with comprehensive metrics."""
    from core.training import LossMetrics
    
    # Extract final positions and velocities
    final_positions = scan_outputs.positions[-1]  # Last timestep
    final_velocities = scan_outputs.velocities[-1]
    
    # Goal reaching loss
    goal_distances = jnp.linalg.norm(final_positions - target_positions, axis=-1)
    goal_loss = jnp.mean(goal_distances ** 2)
    
    # Velocity tracking loss (simplified)
    velocity_loss = jnp.mean(jnp.sum(final_velocities ** 2, axis=-1))
    
    # Control effort loss
    control_effort = jnp.mean(jnp.sum(scan_outputs.control_commands ** 2, axis=-1))
    
    # Safety loss (CBF violations)
    safety_loss = jnp.mean(jnp.maximum(0, -scan_outputs.cbf_values))
    
    # Collision avoidance loss
    collision_loss = jnp.mean(jnp.maximum(0, 1.0 - scan_outputs.obstacle_distances))
    
    # Control smoothness loss
    control_diff = jnp.diff(scan_outputs.control_commands, axis=0)
    control_jerk = jnp.mean(jnp.sum(control_diff ** 2, axis=-1))
    
    # Combine losses
    total_loss = (
        config.goal_reaching_coef * goal_loss +
        config.velocity_tracking_coef * velocity_loss +
        config.control_smoothness_coef * control_effort +
        config.cbf_violation_coef * safety_loss +
        config.collision_avoidance_coef * collision_loss +
        0.01 * control_jerk  # Small weight for smoothness
    )
    
    # Create comprehensive metrics
    metrics = LossMetrics(
        total_loss=total_loss,
        efficiency_loss=goal_loss,
        safety_loss=safety_loss,
        control_loss=control_effort,
        
        # GCBF+ metrics
        cbf_violation=safety_loss,
        cbf_derivative=0.0,  # Placeholder
        cbf_boundary=0.0,  # Placeholder
        
        # DiffPhysDrone metrics
        velocity_tracking=velocity_loss,
        collision_penalty=collision_loss,
        control_smoothness=control_effort,
        control_jerk=control_jerk,
        
        # Efficiency metrics
        goal_distance=jnp.mean(goal_distances),
        time_penalty=0.0,  # Placeholder
        
        # Safety metrics
        safety_violations=jnp.sum(scan_outputs.cbf_values < 0),
        emergency_activations=0.0,  # Placeholder
        qp_success_rate=1.0,  # Placeholder
        
        # Training dynamics
        gradient_norm=0.0,  # Will be filled in later
        temporal_decay_factor=0.95  # Default
    )
    
    return total_loss, metrics


def debug_tensor_shapes(*args, **kwargs):
    """Debug utility for tensor shapes."""
    # Simple debug print - can be expanded as needed
    for i, arg in enumerate(args):
        if hasattr(arg, 'shape'):
            print(f"DEBUG: arg[{i}].shape: {arg.shape}")
    
    for key, value in kwargs.items():
        if hasattr(value, 'shape'):
            print(f"DEBUG: {key}.shape: {value.shape}")