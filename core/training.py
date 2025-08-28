"""
Complete Training Framework for Safe Agile Flight System

This module implements the comprehensive training methodology combining:
1. GCBF+ (MIT-REALM) CBF loss formulations
2. DiffPhysDrone (SJTU) physics-driven losses
3. Multi-objective optimization with temporal gradient decay
4. JAX-native implementation for maximum performance

Key Components:
- Multi-objective loss function (efficiency + safety + control)
- CBF constraint violations and derivative conditions  
- Physics-driven losses from DiffPhysDrone methodology
- Temporal and spatial gradient decay mechanisms
- MGDA (Multi-Gradient Descent Algorithm) for balanced optimization
"""

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import optax
from typing import Dict, Tuple, NamedTuple, Optional
import chex
from dataclasses import dataclass

from .physics import DroneState, PhysicsParams, dynamics_step
from .perception import PerceptionModule, DroneState as PerceptionDroneState
from .policy import PolicyNetworkMLP  
from .safety import SafetyLayer, SafetyConfig, QSolutionInfo, compute_safety_loss
from .loop import ScanCarry, ScanOutput


# =============================================================================
# LOSS CONFIGURATION AND COMPONENTS
# =============================================================================

@dataclass 
class LossConfig:
    """Configuration for loss function components"""
    # GCBF+ loss coefficients
    cbf_violation_coef: float = 5.0       # CBF constraint violation penalty
    cbf_derivative_coef: float = 3.0      # CBF derivative condition penalty
    cbf_boundary_coef: float = 2.0        # CBF boundary smoothness
    
    # DiffPhysDrone physics-driven losses
    velocity_tracking_coef: float = 1.0   # Target velocity tracking
    collision_avoidance_coef: float = 4.0 # Collision penalty
    control_smoothness_coef: float = 0.1  # Control effort regularization
    control_jerk_coef: float = 0.05       # Control jerk penalty
    
    # Efficiency losses
    goal_reaching_coef: float = 2.0       # Distance to goal penalty
    time_efficiency_coef: float = 0.1     # Time-to-goal penalty
    
    # Safety system losses
    safety_layer_coef: float = 1.0        # Safety filter penalties
    emergency_coef: float = 100.0         # Emergency brake penalty
    
    # Gradient decay parameters
    temporal_decay_alpha: float = 0.95    # Base temporal decay factor
    spatial_decay_enable: bool = True     # Enable spatial adaptation
    spatial_decay_range: float = 2.0      # Distance range for spatial decay


class LossMetrics(NamedTuple):
    """Comprehensive loss metrics for monitoring training"""
    # Total loss components
    total_loss: chex.Array
    efficiency_loss: chex.Array
    safety_loss: chex.Array
    control_loss: chex.Array
    
    # GCBF+ specific losses
    cbf_violation: chex.Array
    cbf_derivative: chex.Array
    cbf_boundary: chex.Array
    
    # DiffPhysDrone specific losses  
    velocity_tracking: chex.Array
    collision_penalty: chex.Array
    control_smoothness: chex.Array
    control_jerk: chex.Array
    
    # Efficiency metrics
    goal_distance: chex.Array
    time_penalty: chex.Array
    
    # Safety metrics
    safety_violations: chex.Array
    emergency_activations: chex.Array
    qp_success_rate: chex.Array
    
    # Training dynamics
    gradient_norm: chex.Array
    temporal_decay_factor: chex.Array


# =============================================================================
# ADVANCED TRAINING FRAMEWORK WITH MULTI-OBJECTIVE OPTIMIZATION
# =============================================================================

class AdvancedTrainingFramework:
    """Advanced training framework with multi-objective optimization and curriculum learning"""
    
    def __init__(self, loss_config: LossConfig, use_curriculum: bool = True):
        self.loss_config = loss_config
        self.use_curriculum = use_curriculum
        self.training_stage = 0
        self.loss_history = {'total': [], 'safety': [], 'efficiency': []}
        self.curriculum_thresholds = {
            'stage_1_to_2': {'min_efficiency': 0.7, 'max_safety_violations': 5},
            'stage_2_to_3': {'min_efficiency': 0.85, 'max_safety_violations': 2}
        }
    
    def compute_comprehensive_loss_with_curriculum(
        self,
        scan_outputs: ScanOutput,
        target_positions: chex.Array,
        target_velocities: chex.Array,
        physics_params: PhysicsParams,
        training_step: int = 0
    ) -> Tuple[chex.Array, LossMetrics, dict]:
        """
        Compute comprehensive loss with curriculum learning adaptation
        
        Three-stage curriculum:
        Stage 0: Focus on basic control and goal reaching (relaxed safety)
        Stage 1: Introduce safety constraints gradually
        Stage 2: Full safety enforcement with efficiency optimization
        """
        # Determine current curriculum stage
        current_stage = self._get_current_curriculum_stage(training_step)
        
        # Adapt loss weights based on curriculum stage
        adapted_config = self._adapt_loss_config(current_stage)
        
        # Compute base loss with adapted configuration
        total_loss, metrics = compute_comprehensive_loss(
            scan_outputs, target_positions, target_velocities,
            adapted_config, physics_params
        )
        
        # Add curriculum-specific components
        curriculum_info = {
            'current_stage': current_stage,
            'stage_progress': self._get_stage_progress(training_step, current_stage),
            'adapted_weights': self._get_weight_summary(adapted_config)
        }
        
        # Update training history
        self._update_training_history(total_loss, metrics)
        
        # Check for stage advancement
        stage_advanced = self._check_stage_advancement(metrics, current_stage)
        if stage_advanced:
            print(f"🎓 Curriculum advanced from stage {current_stage} to {current_stage + 1}")
            self.training_stage = current_stage + 1
            curriculum_info['stage_advanced'] = True
        
        return total_loss, metrics, curriculum_info
    
    def _get_current_curriculum_stage(self, training_step: int) -> int:
        """Determine current curriculum stage"""
        if not self.use_curriculum:
            return 2  # Full training
        
        # Override automatic progression if manually set
        if hasattr(self, 'manual_stage_override'):
            return self.manual_stage_override
        
        # Automatic progression based on training steps
        stage_duration = 3000  # Steps per automatic stage
        automatic_stage = min(2, training_step // stage_duration)
        
        # Use the higher of manual and automatic stage
        return max(self.training_stage, automatic_stage)
    
    def _adapt_loss_config(self, stage: int) -> LossConfig:
        """Adapt loss configuration based on curriculum stage"""
        base_config = self.loss_config
        
        if stage == 0:  # Basic control stage
            return LossConfig(
                cbf_violation_coef=base_config.cbf_violation_coef * 0.2,  # Very relaxed
                cbf_derivative_coef=base_config.cbf_derivative_coef * 0.1,
                cbf_boundary_coef=base_config.cbf_boundary_coef * 0.1,
                velocity_tracking_coef=base_config.velocity_tracking_coef * 1.5,  # Focus on control
                collision_avoidance_coef=base_config.collision_avoidance_coef * 0.3,
                control_smoothness_coef=base_config.control_smoothness_coef * 2.0,  # Encourage smooth control
                goal_reaching_coef=base_config.goal_reaching_coef * 2.0,  # Focus on reaching goals
                safety_layer_coef=base_config.safety_layer_coef * 0.1,
                emergency_coef=base_config.emergency_coef * 0.5,
                temporal_decay_alpha=base_config.temporal_decay_alpha,
                spatial_decay_enable=base_config.spatial_decay_enable,
                spatial_decay_range=base_config.spatial_decay_range
            )
        elif stage == 1:  # Safety-aware stage
            return LossConfig(
                cbf_violation_coef=base_config.cbf_violation_coef * 0.7,  # Moderate safety
                cbf_derivative_coef=base_config.cbf_derivative_coef * 0.6,
                cbf_boundary_coef=base_config.cbf_boundary_coef * 0.6,
                velocity_tracking_coef=base_config.velocity_tracking_coef * 1.2,
                collision_avoidance_coef=base_config.collision_avoidance_coef * 0.8,
                control_smoothness_coef=base_config.control_smoothness_coef * 1.2,
                goal_reaching_coef=base_config.goal_reaching_coef * 1.5,
                safety_layer_coef=base_config.safety_layer_coef * 0.7,
                emergency_coef=base_config.emergency_coef * 0.8,
                temporal_decay_alpha=base_config.temporal_decay_alpha,
                spatial_decay_enable=base_config.spatial_decay_enable,
                spatial_decay_range=base_config.spatial_decay_range
            )
        else:  # Full training stage
            return base_config
    
    def _get_stage_progress(self, training_step: int, current_stage: int) -> float:
        """Get progress within current curriculum stage"""
        stage_duration = 3000
        stage_start = current_stage * stage_duration
        progress = min(1.0, (training_step - stage_start) / stage_duration)
        return progress
    
    def _get_weight_summary(self, config: LossConfig) -> dict:
        """Get summary of current loss weights"""
        return {
            'safety_weight': config.cbf_violation_coef,
            'efficiency_weight': config.goal_reaching_coef,
            'control_weight': config.control_smoothness_coef
        }
    
    def _update_training_history(self, total_loss: chex.Array, metrics: LossMetrics):
        """Update training history for curriculum decisions"""
        self.loss_history['total'].append(float(total_loss))
        self.loss_history['safety'].append(float(metrics.safety_loss))
        self.loss_history['efficiency'].append(float(metrics.efficiency_loss))
        
        # Keep history manageable
        max_history = 1000
        for key in self.loss_history:
            if len(self.loss_history[key]) > max_history:
                self.loss_history[key] = self.loss_history[key][-max_history//2:]
    
    def _check_stage_advancement(self, metrics: LossMetrics, current_stage: int) -> bool:
        """Check if curriculum should advance to next stage"""
        if current_stage >= 2:  # Already at final stage
            return False
        
        # Need sufficient training history
        if len(self.loss_history['total']) < 100:
            return False
        
        # Compute recent performance metrics
        recent_window = 50
        recent_safety_violations = jnp.mean(jnp.array(self.loss_history['safety'][-recent_window:]))
        recent_efficiency = 1.0 / (1.0 + jnp.mean(jnp.array(self.loss_history['efficiency'][-recent_window:])))
        
        # Check advancement criteria
        if current_stage == 0:  # Stage 0 -> 1
            criteria = self.curriculum_thresholds['stage_1_to_2']
            return (recent_efficiency >= criteria['min_efficiency'] * 0.8 and  # Relaxed for stage 1
                   recent_safety_violations <= criteria['max_safety_violations'] * 2.0)
        elif current_stage == 1:  # Stage 1 -> 2
            criteria = self.curriculum_thresholds['stage_2_to_3']
            return (recent_efficiency >= criteria['min_efficiency'] and
                   recent_safety_violations <= criteria['max_safety_violations'])
        
        return False

class MultiObjectiveOptimizer:
    """Multi-objective optimizer using gradient balancing techniques"""
    
    def __init__(self, balance_method: str = 'adaptive_weights'):
        self.balance_method = balance_method
        self.objective_history = {'safety': [], 'efficiency': [], 'control': []}
        self.weight_adaptation_rate = 0.01
        self.current_weights = {'safety': 1.0, 'efficiency': 1.0, 'control': 1.0}
    
    def compute_balanced_loss(
        self,
        safety_loss: chex.Array,
        efficiency_loss: chex.Array,
        control_loss: chex.Array,
        training_step: int = 0
    ) -> Tuple[chex.Array, dict]:
        """Compute balanced multi-objective loss"""
        
        if self.balance_method == 'adaptive_weights':
            return self._adaptive_weight_balancing(safety_loss, efficiency_loss, control_loss)
        elif self.balance_method == 'gradient_cosine':
            return self._gradient_cosine_balancing(safety_loss, efficiency_loss, control_loss)
        elif self.balance_method == 'pareto_efficient':
            return self._pareto_efficient_balancing(safety_loss, efficiency_loss, control_loss)
        else:
            # Simple weighted sum
            weights = self.current_weights
            total_loss = (weights['safety'] * safety_loss + 
                         weights['efficiency'] * efficiency_loss + 
                         weights['control'] * control_loss)
            balance_info = {'method': 'fixed_weights', 'weights': weights}
            return total_loss, balance_info
    
    def _adaptive_weight_balancing(
        self, 
        safety_loss: chex.Array, 
        efficiency_loss: chex.Array, 
        control_loss: chex.Array
    ) -> Tuple[chex.Array, dict]:
        """Adaptive weight balancing based on loss magnitudes"""
        
        # Update objective history
        self.objective_history['safety'].append(float(safety_loss))
        self.objective_history['efficiency'].append(float(efficiency_loss))
        self.objective_history['control'].append(float(control_loss))
        
        # Compute adaptive weights based on recent loss magnitudes
        window_size = min(50, len(self.objective_history['safety']))
        if window_size > 10:
            recent_safety = jnp.mean(jnp.array(self.objective_history['safety'][-window_size:]))
            recent_efficiency = jnp.mean(jnp.array(self.objective_history['efficiency'][-window_size:]))
            recent_control = jnp.mean(jnp.array(self.objective_history['control'][-window_size:]))
            
            # Inverse weighting: give more weight to smaller losses to balance objectives
            total_magnitude = recent_safety + recent_efficiency + recent_control + 1e-6
            target_weights = {
                'safety': (recent_efficiency + recent_control) / (2 * total_magnitude) * 3,
                'efficiency': (recent_safety + recent_control) / (2 * total_magnitude) * 3,
                'control': (recent_safety + recent_efficiency) / (2 * total_magnitude) * 3
            }
            
            # Smooth weight adaptation
            for key in self.current_weights:
                self.current_weights[key] = (
                    (1 - self.weight_adaptation_rate) * self.current_weights[key] +
                    self.weight_adaptation_rate * target_weights[key]
                )
        
        # Compute balanced loss
        weights = self.current_weights
        total_loss = (weights['safety'] * safety_loss + 
                     weights['efficiency'] * efficiency_loss + 
                     weights['control'] * control_loss)
        
        balance_info = {
            'method': 'adaptive_weights',
            'weights': weights,
            'weight_adaptation_rate': self.weight_adaptation_rate
        }
        
        return total_loss, balance_info
    
    def _gradient_cosine_balancing(
        self,
        safety_loss: chex.Array,
        efficiency_loss: chex.Array, 
        control_loss: chex.Array
    ) -> Tuple[chex.Array, dict]:
        """Gradient cosine similarity balancing (simplified implementation)"""
        # This would require gradient computation for full implementation
        # For now, use magnitude-based approximation
        
        loss_magnitudes = jnp.array([safety_loss, efficiency_loss, control_loss])
        
        # Normalize to unit scale
        normalized_losses = loss_magnitudes / (jnp.linalg.norm(loss_magnitudes) + 1e-8)
        
        # Equal weighting with normalization
        equal_weights = jnp.ones(3) / 3.0
        
        # Balance based on deviation from equal contribution
        weights = equal_weights + 0.1 * (equal_weights - normalized_losses)
        weights = jnp.maximum(weights, 0.1)  # Minimum weight
        weights = weights / jnp.sum(weights)  # Normalize
        
        total_loss = jnp.sum(weights * loss_magnitudes)
        
        balance_info = {
            'method': 'gradient_cosine',
            'weights': {'safety': weights[0], 'efficiency': weights[1], 'control': weights[2]},
            'normalized_losses': normalized_losses
        }
        
        return total_loss, balance_info

def compute_cbf_violation_loss(
    h_values: chex.Array,
    h_dots: chex.Array, 
    alpha: float = 1.0
) -> Tuple[chex.Array, Dict]:
    """
    Compute CBF constraint violation loss (from GCBF+ methodology)
    
    CBF constraint: #(x) + �(h(x)) e 0
    
    Args:
        h_values: CBF values at each timestep (T, B)
        h_dots: CBF time derivatives (T, B) 
        alpha: CBF class-K function parameter
        
    Returns:
        loss: CBF violation loss
        metrics: Detailed violation metrics
    """
    # CBF constraint: h_dot + alpha * h >= 0
    cbf_constraint = h_dots + alpha * h_values
    
    # Violation occurs when constraint is negative
    violations = jnp.maximum(0.0, -cbf_constraint)
    violation_loss = jnp.mean(violations ** 2)
    
    # Safety region classification (GCBF+ methodology)
    safe_region = h_values > 0.0  # Positive CBF = safe
    unsafe_region = h_values <= 0.0  # Negative CBF = unsafe
    
    # Penalize positive CBF predictions in unsafe regions
    false_safe_penalty = jnp.mean(
        jnp.where(unsafe_region, jnp.maximum(0.0, h_values) ** 2, 0.0)
    )
    
    # Penalize negative CBF predictions in safe regions  
    false_unsafe_penalty = jnp.mean(
        jnp.where(safe_region, jnp.maximum(0.0, -h_values) ** 2, 0.0)
    )
    
    total_loss = violation_loss + false_safe_penalty + false_unsafe_penalty
    
    metrics = {
        "cbf_violations": jnp.sum(violations > 0.0),
        "violation_magnitude": jnp.mean(violations),
        "false_safe_rate": jnp.mean(unsafe_region & (h_values > 0)),
        "false_unsafe_rate": jnp.mean(safe_region & (h_values < 0)),
        "constraint_satisfaction": jnp.mean(cbf_constraint >= 0)
    }
    
    return total_loss, metrics


def compute_cbf_derivative_loss(
    h_values: chex.Array,
    h_grads: chex.Array,
    drone_states: chex.Array,
    control_inputs: chex.Array,
    physics_params: PhysicsParams
) -> Tuple[chex.Array, Dict]:
    """
    Compute CBF derivative condition loss
    
    Ensures CBF time derivative is computed correctly:
    # = h^T(f(x) + g(x)u)
    
    Args:
        h_values: CBF values (T, B) 
        h_grads: CBF gradients w.r.t. position (T, B, 3)
        drone_states: Full drone states (T, B, state_dim)
        control_inputs: Control inputs (T, B, 3)
        physics_params: Physics parameters
        
    Returns:
        loss: Derivative consistency loss
        metrics: Derivative accuracy metrics
    """
    # Extract position and velocity from drone states
    positions = drone_states[:, :, :3]  # (T, B, 3)
    velocities = drone_states[:, :, 3:6]  # (T, B, 3)
    
    # Simplified dynamics: f(x) = [v, -g], g(x) = [0, I]
    f_dynamics = jnp.concatenate([
        velocities,
        jnp.tile(jnp.array([0.0, 0.0, -9.81]), (h_values.shape[0], h_values.shape[1], 1))
    ], axis=-1)  # (T, B, 6)
    
    g_matrix = jnp.concatenate([
        jnp.zeros((h_values.shape[0], h_values.shape[1], 3, 3)),  # Position part
        jnp.tile(jnp.eye(3), (h_values.shape[0], h_values.shape[1], 1, 1))  # Velocity part  
    ], axis=-2)  # (T, B, 6, 3)
    
    # Extend h_grads to full state space (assume gradient w.r.t. velocity is zero)
    h_grads_full = jnp.concatenate([
        h_grads,  # Position gradients
        jnp.zeros_like(h_grads)  # Velocity gradients (simplified)
    ], axis=-1)  # (T, B, 6)
    
    # Predicted derivative: h^T(f + gu)
    predicted_h_dot = jnp.sum(h_grads_full * f_dynamics, axis=-1) + jnp.sum(
        h_grads_full[:, :, None, :] @ g_matrix * control_inputs[:, :, None, :], axis=(-2, -1)
    )
    
    # Compute actual derivative via finite differences
    dt = physics_params.dt
    actual_h_dot = (h_values[1:] - h_values[:-1]) / dt
    predicted_h_dot_aligned = predicted_h_dot[:-1]  # Align shapes
    
    # Derivative consistency loss
    derivative_error = predicted_h_dot_aligned - actual_h_dot
    derivative_loss = jnp.mean(derivative_error ** 2)
    
    metrics = {
        "derivative_mse": derivative_loss,
        "derivative_mae": jnp.mean(jnp.abs(derivative_error)),
        "prediction_accuracy": 1.0 - jnp.mean(jnp.abs(derivative_error) / (jnp.abs(actual_h_dot) + 1e-8))
    }
    
    return derivative_loss, metrics


# =============================================================================
# PHYSICS-DRIVEN LOSSES (DiffPhysDrone Integration)  
# =============================================================================

def compute_velocity_tracking_loss(
    actual_velocities: chex.Array,
    target_velocities: chex.Array,
    time_weights: Optional[chex.Array] = None
) -> Tuple[chex.Array, Dict]:
    """
    Compute velocity tracking loss (from DiffPhysDrone methodology)
    
    Args:
        actual_velocities: Simulated velocities (T, B, 3)
        target_velocities: Desired velocities (T, B, 3) 
        time_weights: Optional temporal weighting (T, B)
        
    Returns:
        loss: Velocity tracking loss
        metrics: Tracking performance metrics
    """
    velocity_errors = actual_velocities - target_velocities
    
    # Smooth L1 loss (more robust than MSE)
    smooth_l1_loss = jnp.mean(
        jnp.where(
            jnp.abs(velocity_errors) < 1.0,
            0.5 * velocity_errors ** 2,
            jnp.abs(velocity_errors) - 0.5
        )
    )
    
    # Apply temporal weighting if provided
    if time_weights is not None:
        smooth_l1_loss = jnp.mean(time_weights * smooth_l1_loss)
    
    # Component-wise tracking metrics
    velocity_norms = jnp.linalg.norm(velocity_errors, axis=-1)
    
    metrics = {
        "velocity_mse": jnp.mean(velocity_errors ** 2),
        "velocity_mae": jnp.mean(jnp.abs(velocity_errors)),
        "tracking_accuracy": jnp.mean(velocity_norms < 0.5),  # Within 0.5 m/s
        "max_error": jnp.max(velocity_norms)
    }
    
    return smooth_l1_loss, metrics


def compute_collision_avoidance_loss(
    distances_to_obstacles: chex.Array,
    safety_margins: chex.Array,
    velocity_magnitudes: chex.Array
) -> Tuple[chex.Array, Dict]:
    """
    Compute collision avoidance loss with velocity-dependent penalties
    
    Args:
        distances_to_obstacles: Distance to nearest obstacles (T, B)
        safety_margins: Required safety margins (T, B)
        velocity_magnitudes: Current velocity magnitudes (T, B)
        
    Returns:
        loss: Collision avoidance loss  
        metrics: Safety metrics
    """
    # Barrier-like penalty (increases sharply near obstacles)
    clearance = distances_to_obstacles - safety_margins
    
    # Velocity-weighted penalty (higher penalty at high speeds)
    velocity_weights = 1.0 + velocity_magnitudes  # Higher weight for higher speeds
    
    # Soft barrier function
    collision_penalty = jnp.where(
        clearance < 0.5,  # Active within 0.5m of safety boundary
        velocity_weights * jnp.exp(-clearance * 4.0),  # Exponential barrier
        0.0
    )
    
    collision_loss = jnp.mean(collision_penalty)
    
    # Additional quadratic clearance loss (from DiffPhysDrone)
    quadratic_clearance = jnp.where(
        clearance < 1.0,  # Active within 1m
        jnp.maximum(0.0, 1.0 - clearance) ** 2,
        0.0
    )
    
    total_loss = collision_loss + 0.5 * jnp.mean(quadratic_clearance)
    
    metrics = {
        "collision_risk": jnp.mean(clearance < 0.1),
        "safety_violations": jnp.sum(clearance < 0.0),
        "average_clearance": jnp.mean(clearance),
        "min_clearance": jnp.min(clearance)
    }
    
    return total_loss, metrics


def compute_control_regularization_loss(
    control_sequence: chex.Array,
    dt: float
) -> Tuple[chex.Array, Dict]:
    """
    Compute control effort and smoothness regularization
    
    Args:
        control_sequence: Control inputs (T, B, 3)
        dt: Time step size
        
    Returns:
        loss: Combined control regularization loss
        metrics: Control effort metrics
    """
    # Control magnitude penalty
    control_magnitude_loss = jnp.mean(jnp.sum(control_sequence ** 2, axis=-1))
    
    # Control smoothness (acceleration penalty)
    control_diff = jnp.diff(control_sequence, axis=0) / dt
    control_smoothness_loss = jnp.mean(jnp.sum(control_diff ** 2, axis=-1))
    
    # Control jerk penalty (second derivative)
    control_jerk = jnp.diff(control_diff, axis=0) / dt
    control_jerk_loss = jnp.mean(jnp.sum(control_jerk ** 2, axis=-1))
    
    total_loss = control_magnitude_loss + control_smoothness_loss + 0.1 * control_jerk_loss
    
    metrics = {
        "control_magnitude": jnp.mean(jnp.linalg.norm(control_sequence, axis=-1)),
        "control_smoothness": jnp.mean(jnp.linalg.norm(control_diff, axis=-1)),
        "control_jerk": jnp.mean(jnp.linalg.norm(control_jerk, axis=-1)),
        "max_control": jnp.max(jnp.linalg.norm(control_sequence, axis=-1))
    }
    
    return total_loss, metrics


# =============================================================================
# EFFICIENCY AND GOAL-DIRECTED LOSSES
# =============================================================================

def compute_goal_reaching_loss(
    final_positions: chex.Array,
    target_positions: chex.Array,
    trajectory_positions: chex.Array
) -> Tuple[chex.Array, Dict]:
    """
    Compute goal-reaching efficiency loss
    
    Args:
        final_positions: Final positions (B, 3)
        target_positions: Target positions (B, 3)
        trajectory_positions: Full trajectory positions (T, B, 3)
        
    Returns:
        loss: Goal reaching loss
        metrics: Goal reaching metrics
    """
    # Primary objective: reach target position
    final_distance_error = jnp.linalg.norm(final_positions - target_positions, axis=-1)
    goal_reaching_loss = jnp.mean(final_distance_error ** 2)
    
    # Secondary objective: efficient path (minimize trajectory length)
    trajectory_lengths = jnp.sum(
        jnp.linalg.norm(jnp.diff(trajectory_positions, axis=0), axis=-1), axis=0
    )
    direct_distances = jnp.linalg.norm(
        trajectory_positions[-1] - trajectory_positions[0], axis=-1
    )
    path_efficiency = direct_distances / (trajectory_lengths + 1e-8)
    efficiency_loss = jnp.mean((1.0 - path_efficiency) ** 2)
    
    total_loss = goal_reaching_loss + 0.1 * efficiency_loss
    
    metrics = {
        "final_distance_error": jnp.mean(final_distance_error),
        "goal_success_rate": jnp.mean(final_distance_error < 0.5),
        "path_efficiency": jnp.mean(path_efficiency),
        "trajectory_length": jnp.mean(trajectory_lengths)
    }
    
    return total_loss, metrics


# =============================================================================
# TEMPORAL GRADIENT DECAY (DiffPhysDrone)
# =============================================================================

def apply_spatial_temporal_gradient_decay(
    gradients: chex.Array,
    timestep: int,
    distance_to_obstacles: chex.Array,
    config: LossConfig
) -> chex.Array:
    """
    Apply spatial-temporal gradient decay mechanism
    
    Args:
        gradients: Gradients to decay (any shape)
        timestep: Current timestep in trajectory
        distance_to_obstacles: Distance to nearest obstacles
        config: Loss configuration with decay parameters
        
    Returns:
        decayed_gradients: Spatially and temporally decayed gradients
    """
    # Temporal decay (exponential)
    temporal_decay = config.temporal_decay_alpha ** timestep
    
    if config.spatial_decay_enable:
        # Spatial adaptation: stronger decay when far from obstacles
        min_distance = jnp.min(distance_to_obstacles)
        spatial_factor = jnp.minimum(
            1.0, 
            jnp.maximum(0.3, min_distance / config.spatial_decay_range)
        )
        decay_factor = temporal_decay * spatial_factor
    else:
        decay_factor = temporal_decay
    
    return gradients * decay_factor


# =============================================================================
# MULTI-OBJECTIVE LOSS INTEGRATION
# =============================================================================

def compute_comprehensive_loss(
    scan_outputs: ScanOutput,
    target_positions: chex.Array,
    target_velocities: chex.Array,
    config: LossConfig,
    physics_params: PhysicsParams
) -> Tuple[chex.Array, LossMetrics]:
    """
    Compute comprehensive multi-objective loss function
    
    Integrates all loss components from GCBF+ and DiffPhysDrone methodologies
    
    Args:
        scan_outputs: Complete trajectory outputs from scan loop
        target_positions: Target positions (B, 3)
        target_velocities: Target velocities (T, B, 3)
        config: Loss configuration
        physics_params: Physics parameters
        
    Returns:
        total_loss: Combined weighted loss
        metrics: Comprehensive loss metrics
    """
    T, B = scan_outputs.drone_states.shape[:2]
    
    # Extract trajectory components from scan outputs
    positions = scan_outputs.drone_states[:, :, :3]  # (T, B, 3)
    velocities = scan_outputs.drone_states[:, :, 3:6]  # (T, B, 3)
    
    # Handle optional CBF outputs (may not exist in simplified versions)
    h_values = getattr(scan_outputs, 'cbf_values', jnp.zeros((T, B)))
    h_grads = getattr(scan_outputs, 'cbf_gradients', jnp.zeros((T, B, 3)))
    control_inputs = getattr(scan_outputs, 'safe_controls', getattr(scan_outputs, 'controls', jnp.zeros((T, B, 3))))
    distances = getattr(scan_outputs, 'obstacle_distances', jnp.ones((T, B)) * 10.0)  # Safe default
    safety_info = getattr(scan_outputs, 'safety_info', None)
    
    # Initialize loss accumulator
    total_loss = 0.0
    loss_components = {}
    
    # 1. GCBF+ CBF Losses
    # ---------------------------------------------------------
    
    # CBF violation loss
    h_dots = jnp.gradient(h_values, axis=0) / physics_params.dt
    cbf_violation_loss, cbf_metrics = compute_cbf_violation_loss(
        h_values, h_dots, physics_params.cbf_alpha if hasattr(physics_params, 'cbf_alpha') else 1.0
    )
    total_loss += config.cbf_violation_coef * cbf_violation_loss
    loss_components.update(cbf_metrics)
    
    # CBF derivative consistency loss
    cbf_derivative_loss, derivative_metrics = compute_cbf_derivative_loss(
        h_values, h_grads, scan_outputs.drone_states, control_inputs, physics_params
    )
    total_loss += config.cbf_derivative_coef * cbf_derivative_loss
    loss_components.update(derivative_metrics)
    
    # 2. DiffPhysDrone Physics-Driven Losses  
    # ---------------------------------------------------------
    
    # Velocity tracking loss
    velocity_loss, velocity_metrics = compute_velocity_tracking_loss(
        velocities, target_velocities
    )
    total_loss += config.velocity_tracking_coef * velocity_loss
    loss_components.update(velocity_metrics)
    
    # Collision avoidance loss
    velocity_magnitudes = jnp.linalg.norm(velocities, axis=-1)
    safety_margins = jnp.full_like(distances, 0.2)  # 20cm safety margin
    collision_loss, collision_metrics = compute_collision_avoidance_loss(
        distances, safety_margins, velocity_magnitudes
    )
    total_loss += config.collision_avoidance_coef * collision_loss
    loss_components.update(collision_metrics)
    
    # Control regularization
    control_loss, control_metrics = compute_control_regularization_loss(
        control_inputs, physics_params.dt
    )
    total_loss += config.control_smoothness_coef * control_loss
    loss_components.update(control_metrics)
    
    # 3. Efficiency and Goal-Directed Losses
    # ---------------------------------------------------------
    
    # Goal reaching loss
    goal_loss, goal_metrics = compute_goal_reaching_loss(
        positions[-1], target_positions, positions
    )
    total_loss += config.goal_reaching_coef * goal_loss
    loss_components.update(goal_metrics)
    
    # 4. Safety System Losses
    # ---------------------------------------------------------
    
    # Extract safety-related losses from QP solver info
    safety_losses = []
    emergency_count = 0
    qp_success_count = 0
    
    for t in range(T):
        for b in range(B):
            # This would be properly implemented with actual safety_info structure
            # For now, using placeholder logic
            safety_status = 0  # Placeholder
            if safety_status == 3:  # Emergency mode
                emergency_count += 1
                safety_losses.append(config.emergency_coef)
            elif safety_status > 0:  # QP failed
                safety_losses.append(config.safety_layer_coef)
            else:  # Success
                qp_success_count += 1
                safety_losses.append(0.0)
    
    safety_loss = jnp.mean(jnp.array(safety_losses)) if safety_losses else 0.0
    total_loss += safety_loss
    
    # 5. Compile Comprehensive Metrics
    # ---------------------------------------------------------
    
    metrics = LossMetrics(
        total_loss=total_loss,
        efficiency_loss=goal_loss,
        safety_loss=cbf_violation_loss + collision_loss,
        control_loss=control_loss,
        
        # GCBF+ metrics
        cbf_violation=cbf_violation_loss,
        cbf_derivative=cbf_derivative_loss,
        cbf_boundary=0.0,  # Placeholder
        
        # DiffPhysDrone metrics  
        velocity_tracking=velocity_loss,
        collision_penalty=collision_loss,
        control_smoothness=control_loss,
        control_jerk=loss_components.get('control_jerk', 0.0),
        
        # Efficiency metrics
        goal_distance=jnp.mean(jnp.linalg.norm(positions[-1] - target_positions, axis=-1)),
        time_penalty=0.0,  # Placeholder
        
        # Safety metrics
        safety_violations=loss_components.get('safety_violations', 0.0),
        emergency_activations=float(emergency_count) / (T * B),
        qp_success_rate=float(qp_success_count) / (T * B),
        
        # Training dynamics
        gradient_norm=0.0,  # Will be filled by training loop
        temporal_decay_factor=config.temporal_decay_alpha
    )
    
    return total_loss, metrics


# =============================================================================
# MGDA (MULTI-GRADIENT DESCENT ALGORITHM)
# =============================================================================

def mgda_gradient_balancing(
    gradients_dict: Dict[str, chex.Array],
    loss_weights: Dict[str, float]
) -> Tuple[chex.Array, Dict[str, float]]:
    """
    Multi-Gradient Descent Algorithm for balanced multi-objective optimization
    
    Args:
        gradients_dict: Dictionary of gradients for each objective
        loss_weights: Current loss weights
        
    Returns:
        balanced_gradients: Balanced combined gradients
        updated_weights: Updated loss weights
    """
    # Convert gradients to flat arrays for MGDA computation
    flat_gradients = {}
    original_shapes = {}
    
    for name, grad in gradients_dict.items():
        original_shapes[name] = grad.shape
        flat_gradients[name] = grad.flatten()
    
    # Stack gradients into matrix (n_objectives, n_parameters)
    gradient_matrix = jnp.stack([flat_gradients[name] for name in gradients_dict.keys()])
    
    # Compute gram matrix G_ij = <g_i, g_j>
    gram_matrix = gradient_matrix @ gradient_matrix.T
    
    # Solve for optimal weights (simplified Frank-Wolfe algorithm)
    n_objectives = len(gradients_dict)
    current_weights = jnp.array([loss_weights[name] for name in gradients_dict.keys()])
    
    # Project weights onto simplex (ensure they sum to 1)
    current_weights = current_weights / jnp.sum(current_weights)
    
    # Update weights based on gradient conflicts (simplified)
    gradient_conflicts = jnp.diag(gram_matrix) - jnp.sum(
        gram_matrix * current_weights[None, :], axis=1
    )
    
    # Adjust weights to reduce conflicts
    weight_adjustment = 0.01 * gradient_conflicts
    new_weights = current_weights - weight_adjustment
    new_weights = jnp.maximum(0.1, new_weights)  # Minimum weight
    new_weights = new_weights / jnp.sum(new_weights)  # Normalize
    
    # Compute balanced gradients
    balanced_flat_gradients = jnp.sum(new_weights[:, None] * gradient_matrix, axis=0)
    
    # Reshape back to original structure (assuming all same shape for simplicity)
    first_shape = next(iter(original_shapes.values()))
    balanced_gradients = balanced_flat_gradients.reshape(first_shape)
    
    # Update weights dictionary
    updated_weights = dict(zip(gradients_dict.keys(), new_weights))
    
    return balanced_gradients, updated_weights


# =============================================================================
# TRAINING STEP FUNCTIONS
# =============================================================================

def training_step(
    params_dict: Dict,
    optimizer_state: optax.OptState,
    batch_data: Dict,
    config: LossConfig,
    physics_params: PhysicsParams,
    optimizer: optax.GradientTransformation
) -> Tuple[Dict, optax.OptState, LossMetrics]:
    """
    Single training step with complete loss computation and gradient updates
    
    Args:
        params_dict: Model parameters (GNN, Policy, Safety)
        optimizer_state: Optimizer state
        batch_data: Training batch data
        config: Loss configuration
        physics_params: Physics parameters
        
    Returns:
        updated_params: Updated model parameters
        updated_opt_state: Updated optimizer state  
        metrics: Training metrics
    """
    def loss_fn(params):
        # This would integrate with the complete scan loop
        # For now, placeholder implementation
        scan_outputs = batch_data['scan_outputs']  # Placeholder
        target_positions = batch_data['target_positions']
        target_velocities = batch_data['target_velocities']
        
        loss, metrics = compute_comprehensive_loss(
            scan_outputs, target_positions, target_velocities, config, physics_params
        )
        return loss, metrics
    
    # Compute loss and gradients
    (loss, metrics), gradients = jax.value_and_grad(loss_fn, has_aux=True)(params_dict)
    
    # Apply gradient updates using proper optimizer
    updates, new_optimizer_state = optimizer.update(gradients, optimizer_state, params_dict)
    updated_params = optax.apply_updates(params_dict, updates)
    
    # Update metrics with gradient information
    gradient_norm = jnp.sqrt(sum(
        jnp.sum(g ** 2) for g in jax.tree_util.tree_leaves(gradients)
    ))
    
    updated_metrics = metrics._replace(gradient_norm=gradient_norm)
    
    return updated_params, new_optimizer_state, updated_metrics


# =============================================================================
# FACTORY FUNCTIONS AND UTILITIES
# =============================================================================

def create_default_loss_config() -> LossConfig:
    """Create default loss configuration"""
    return LossConfig()


def create_optimizer(learning_rate: float = 1e-3) -> optax.GradientTransformation:
    """Create Adam optimizer with gradient clipping"""
    return optax.chain(
        optax.clip_by_global_norm(1.0),  # Gradient clipping
        optax.adam(learning_rate)
    )


def log_training_metrics(metrics: LossMetrics, step: int):
    """Log comprehensive training metrics"""
    print(f"Step {step}:")
    print(f"  Total Loss: {metrics.total_loss:.6f}")
    print(f"  Efficiency: {metrics.efficiency_loss:.6f}")
    print(f"  Safety: {metrics.safety_loss:.6f}")
    print(f"  CBF Violations: {metrics.cbf_violation:.6f}")
    print(f"  Collision Risk: {metrics.collision_penalty:.6f}")
    print(f"  Goal Distance: {metrics.goal_distance:.3f}m")
    print(f"  QP Success Rate: {metrics.qp_success_rate:.3f}")
    print(f"  Gradient Norm: {metrics.gradient_norm:.6f}")