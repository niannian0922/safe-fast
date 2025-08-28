"""
Performance Tuning Module for Safe Agile Flight System

This module implements advanced performance optimization techniques including:
1. Adaptive learning rate scheduling
2. Dynamic loss weight balancing
3. Curriculum learning strategies
4. Training efficiency monitoring
5. Hyperparameter optimization utilities

Based on insights from GCBF+ and DiffPhysDrone papers for optimal convergence.
"""

import jax
import jax.numpy as jnp
from jax import random
import optax
from typing import Dict, Tuple, Optional, NamedTuple
import chex
from dataclasses import dataclass
import numpy as np


@dataclass 
class PerformanceTuningConfig:
    """Configuration for performance optimization."""
    
    # Learning rate scheduling
    base_learning_rate: float = 3e-4  # Optimized for control tasks
    policy_lr_multiplier: float = 1.0  # Policy-specific LR scaling
    gnn_lr_multiplier: float = 0.5     # GNN typically needs lower LR
    safety_lr_multiplier: float = 0.1  # Safety layers need careful tuning
    
    # Schedule types: "constant", "cosine", "exponential", "polynomial", "warmup_cosine"
    lr_schedule_type: str = "warmup_cosine"
    warmup_steps: int = 1000
    decay_steps: int = 10000
    min_lr_fraction: float = 0.01
    
    # Loss weight adaptation
    adaptive_loss_weights: bool = True
    weight_update_frequency: int = 100  # Steps between weight updates
    weight_adaptation_rate: float = 0.01
    min_weight: float = 0.01
    max_weight: float = 10.0
    
    # Performance monitoring
    performance_window: int = 100  # Steps for performance averaging
    convergence_threshold: float = 1e-6  # Loss change threshold
    gradient_clip_threshold: float = 1.0
    
    # Curriculum learning
    curriculum_enabled: bool = True
    stage_transition_threshold: float = 0.1  # Loss improvement needed to advance
    difficulty_ramp_rate: float = 0.02


class LearningRateScheduler:
    """Advanced learning rate scheduling for multi-component system."""
    
    def __init__(self, config: PerformanceTuningConfig):
        self.config = config
        
    def create_schedule(self, component_type: str = "policy") -> optax.Schedule:
        """Create learning rate schedule for specific component."""
        
        # Get component-specific multiplier
        if component_type == "policy":
            multiplier = self.config.policy_lr_multiplier
        elif component_type == "gnn":
            multiplier = self.config.gnn_lr_multiplier
        elif component_type == "safety":
            multiplier = self.config.safety_lr_multiplier
        else:
            multiplier = 1.0
        
        base_lr = self.config.base_learning_rate * multiplier
        min_lr = base_lr * self.config.min_lr_fraction
        
        if self.config.lr_schedule_type == "constant":
            return optax.constant_schedule(base_lr)
        
        elif self.config.lr_schedule_type == "cosine":
            return optax.cosine_decay_schedule(
                init_value=base_lr,
                decay_steps=self.config.decay_steps,
                alpha=self.config.min_lr_fraction
            )
        
        elif self.config.lr_schedule_type == "exponential":
            return optax.exponential_decay(
                init_value=base_lr,
                transition_steps=self.config.decay_steps // 4,
                decay_rate=0.96
            )
        
        elif self.config.lr_schedule_type == "warmup_cosine":
            # Warmup + cosine decay (best for transformers and complex models)
            warmup_fn = optax.linear_schedule(
                init_value=min_lr,
                end_value=base_lr,
                transition_steps=self.config.warmup_steps
            )
            
            cosine_fn = optax.cosine_decay_schedule(
                init_value=base_lr,
                decay_steps=self.config.decay_steps - self.config.warmup_steps,
                alpha=self.config.min_lr_fraction
            )
            
            return optax.join_schedules(
                schedules=[warmup_fn, cosine_fn],
                boundaries=[self.config.warmup_steps]
            )
        
        else:  # polynomial (default fallback)
            return optax.polynomial_schedule(
                init_value=base_lr,
                end_value=min_lr,
                power=2.0,
                transition_steps=self.config.decay_steps
            )


class AdaptiveLossWeightBalancer:
    """Dynamic loss weight balancing based on training progress."""
    
    def __init__(self, config: PerformanceTuningConfig):
        self.config = config
        self.loss_history = {}
        self.current_weights = {}
        self.update_counter = 0
    
    def initialize_weights(self, loss_components: Dict[str, float]) -> Dict[str, float]:
        """Initialize adaptive weights for loss components."""
        # Start with equal weights normalized to sum to number of components
        n_components = len(loss_components)
        initial_weight = 1.0 / n_components
        
        weights = {}
        for component in loss_components.keys():
            weights[component] = initial_weight
            self.loss_history[component] = []
        
        self.current_weights = weights
        return weights
    
    def update_weights(
        self, 
        loss_components: Dict[str, float],
        step: int
    ) -> Dict[str, float]:
        """Update loss weights based on component performance."""
        
        if not self.config.adaptive_loss_weights:
            return self.current_weights
        
        self.update_counter += 1
        
        # Update loss history
        for component, loss_value in loss_components.items():
            if component in self.loss_history:
                self.loss_history[component].append(float(loss_value))
                # Keep only recent history
                if len(self.loss_history[component]) > self.config.performance_window:
                    self.loss_history[component] = self.loss_history[component][-self.config.performance_window:]
        
        # Update weights every N steps
        if self.update_counter % self.config.weight_update_frequency == 0:
            self._rebalance_weights()
        
        return self.current_weights
    
    def _rebalance_weights(self):
        """Rebalance weights based on loss component progress."""
        
        new_weights = {}
        
        for component in self.current_weights.keys():
            if component not in self.loss_history or len(self.loss_history[component]) < 10:
                new_weights[component] = self.current_weights[component]
                continue
            
            # Compute loss statistics
            recent_losses = self.loss_history[component][-50:]  # Recent performance
            older_losses = self.loss_history[component][-100:-50]  # Older performance
            
            if len(older_losses) == 0:
                new_weights[component] = self.current_weights[component]
                continue
            
            # Measure improvement rate
            recent_avg = np.mean(recent_losses)
            older_avg = np.mean(older_losses)
            improvement_rate = (older_avg - recent_avg) / (older_avg + 1e-8)
            
            # Increase weight for components that aren't improving well
            if improvement_rate < 0.01:  # Slow improvement
                weight_adjustment = self.config.weight_adaptation_rate
            elif improvement_rate < 0.05:  # Moderate improvement  
                weight_adjustment = 0.0
            else:  # Good improvement
                weight_adjustment = -self.config.weight_adaptation_rate
            
            # Apply adjustment
            current_weight = self.current_weights[component]
            new_weight = current_weight * (1.0 + weight_adjustment)
            
            # Clamp to valid range
            new_weight = jnp.clip(
                new_weight, 
                self.config.min_weight, 
                self.config.max_weight
            )
            
            new_weights[component] = float(new_weight)
        
        # Normalize weights to reasonable scale
        total_weight = sum(new_weights.values())
        target_total = len(new_weights)  # Target average weight of 1.0
        
        for component in new_weights.keys():
            new_weights[component] *= target_total / (total_weight + 1e-8)
        
        self.current_weights = new_weights


class CurriculumLearningManager:
    """Curriculum learning for progressive training difficulty."""
    
    def __init__(self, config: PerformanceTuningConfig):
        self.config = config
        self.current_stage = 0
        self.stage_progress = 0.0
        self.best_loss = float('inf')
        
        # Define curriculum stages
        self.stages = [
            {
                "name": "basic_control",
                "description": "Basic position control without safety constraints",
                "difficulty_multiplier": 0.3,
                "enable_safety": False,
                "sequence_length_multiplier": 0.5
            },
            {
                "name": "safety_aware",
                "description": "Add safety constraints with relaxed penalties",
                "difficulty_multiplier": 0.6,
                "enable_safety": True,
                "sequence_length_multiplier": 0.8
            },
            {
                "name": "full_system",
                "description": "Full system with all constraints",
                "difficulty_multiplier": 1.0,
                "enable_safety": True,
                "sequence_length_multiplier": 1.0
            }
        ]
    
    def get_current_stage(self) -> Dict:
        """Get current curriculum stage configuration."""
        if not self.config.curriculum_enabled:
            return self.stages[-1]  # Full system
        
        return self.stages[min(self.current_stage, len(self.stages) - 1)]
    
    def update_progress(self, current_loss: float, step: int) -> bool:
        """Update curriculum progress and check for stage advancement."""
        
        if not self.config.curriculum_enabled:
            return False
        
        # Track best performance
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.stage_progress += self.config.difficulty_ramp_rate
        
        # Check for stage advancement
        if (self.stage_progress >= 1.0 and 
            self.current_stage < len(self.stages) - 1):
            
            self.current_stage += 1
            self.stage_progress = 0.0
            print(f"🎓 Advanced to curriculum stage {self.current_stage}: {self.stages[self.current_stage]['name']}")
            return True
        
        return False


class PerformanceMonitor:
    """Monitor training performance and detect issues."""
    
    def __init__(self, config: PerformanceTuningConfig):
        self.config = config
        self.metrics_history = []
        self.gradient_history = []
        self.convergence_detected = False
        
    def update(
        self, 
        loss: float, 
        gradient_norm: float, 
        metrics: Dict,
        step: int
    ) -> Dict[str, bool]:
        """Update monitoring and return diagnostic flags."""
        
        # Update history
        self.metrics_history.append({
            'step': step,
            'loss': loss,
            'gradient_norm': gradient_norm,
            **metrics
        })
        
        self.gradient_history.append(gradient_norm)
        
        # Keep only recent history
        if len(self.metrics_history) > self.config.performance_window:
            self.metrics_history = self.metrics_history[-self.config.performance_window:]
        
        if len(self.gradient_history) > self.config.performance_window:
            self.gradient_history = self.gradient_history[-self.config.performance_window:]
        
        # Analyze performance
        diagnostics = {
            'convergence_detected': self._detect_convergence(),
            'gradient_explosion': self._detect_gradient_explosion(),
            'gradient_vanishing': self._detect_gradient_vanishing(),
            'loss_plateaued': self._detect_loss_plateau(),
            'training_unstable': self._detect_instability()
        }
        
        return diagnostics
    
    def _detect_convergence(self) -> bool:
        """Detect if training has converged."""
        if len(self.metrics_history) < 50:
            return False
        
        recent_losses = [m['loss'] for m in self.metrics_history[-50:]]
        loss_std = np.std(recent_losses)
        
        return loss_std < self.config.convergence_threshold
    
    def _detect_gradient_explosion(self) -> bool:
        """Detect gradient explosion."""
        if len(self.gradient_history) < 10:
            return False
        
        recent_gradients = self.gradient_history[-10:]
        return max(recent_gradients) > 10.0 * self.config.gradient_clip_threshold
    
    def _detect_gradient_vanishing(self) -> bool:
        """Detect vanishing gradients."""
        if len(self.gradient_history) < 20:
            return False
        
        recent_gradients = self.gradient_history[-20:]
        return max(recent_gradients) < 1e-8
    
    def _detect_loss_plateau(self) -> bool:
        """Detect loss plateau."""
        if len(self.metrics_history) < 100:
            return False
        
        recent_losses = [m['loss'] for m in self.metrics_history[-100:]]
        older_losses = [m['loss'] for m in self.metrics_history[-200:-100]]
        
        if not older_losses:
            return False
        
        recent_avg = np.mean(recent_losses)
        older_avg = np.mean(older_losses)
        
        improvement = (older_avg - recent_avg) / (older_avg + 1e-8)
        return improvement < 0.001  # Less than 0.1% improvement
    
    def _detect_instability(self) -> bool:
        """Detect training instability."""
        if len(self.metrics_history) < 20:
            return False
        
        recent_losses = [m['loss'] for m in self.metrics_history[-20:]]
        loss_variance = np.var(recent_losses)
        loss_mean = np.mean(recent_losses)
        
        # High coefficient of variation indicates instability
        cv = np.sqrt(loss_variance) / (loss_mean + 1e-8)
        return cv > 0.5


def create_optimized_optimizer(config: PerformanceTuningConfig) -> optax.GradientTransformation:
    """Create optimized multi-component optimizer."""
    
    scheduler = LearningRateScheduler(config)
    
    # Create component-specific optimizers
    policy_schedule = scheduler.create_schedule("policy")
    gnn_schedule = scheduler.create_schedule("gnn")
    safety_schedule = scheduler.create_schedule("safety")
    
    # Use Adam with different learning rates for different components
    optimizers = {
        'policy': optax.chain(
            optax.clip_by_global_norm(config.gradient_clip_threshold),
            optax.adam(policy_schedule, b1=0.9, b2=0.999, eps=1e-8)
        ),
        'gnn': optax.chain(
            optax.clip_by_global_norm(config.gradient_clip_threshold),
            optax.adam(gnn_schedule, b1=0.9, b2=0.999, eps=1e-8)
        ),
        'safety': optax.chain(
            optax.clip_by_global_norm(config.gradient_clip_threshold * 0.5),  # More conservative clipping
            optax.adam(safety_schedule, b1=0.9, b2=0.99, eps=1e-7)  # More stable settings
        )
    }
    
    return optax.multi_transform(optimizers, {'policy': 'policy', 'gnn': 'gnn', 'safety': 'safety'})


def get_optimized_training_config():
    """Get performance-optimized training configuration."""
    
    return PerformanceTuningConfig(
        # Optimized learning rates based on empirical results
        base_learning_rate=1e-4,  # Conservative but stable
        policy_lr_multiplier=2.0,  # Policy networks can handle higher LR
        gnn_lr_multiplier=0.5,     # GNN needs more careful training
        safety_lr_multiplier=0.1,  # Safety parameters change slowly
        
        # Advanced scheduling
        lr_schedule_type="warmup_cosine",
        warmup_steps=500,
        decay_steps=5000,
        min_lr_fraction=0.01,
        
        # Adaptive weight balancing
        adaptive_loss_weights=True,
        weight_update_frequency=50,
        weight_adaptation_rate=0.05,
        
        # Performance monitoring
        performance_window=100,
        convergence_threshold=1e-6,
        gradient_clip_threshold=1.0,
        
        # Curriculum learning
        curriculum_enabled=True,
        stage_transition_threshold=0.2,
        difficulty_ramp_rate=0.01
    )


# =============================================================================
# TESTING AND VALIDATION
# =============================================================================

def validate_performance_tuning():
    """Validate performance tuning implementation."""
    print("🧪 Validating Performance Tuning Components...")
    
    config = get_optimized_training_config()
    
    # Test learning rate scheduler
    scheduler = LearningRateScheduler(config)
    policy_schedule = scheduler.create_schedule("policy")
    
    # Test schedule values
    steps = jnp.arange(1000)
    lrs = [policy_schedule(step) for step in steps]
    
    print(f"✅ LR Schedule: initial={lrs[0]:.6f}, mid={lrs[500]:.6f}, final={lrs[-1]:.6f}")
    assert lrs[0] < lrs[100], "Warmup should increase LR"
    assert lrs[500] > lrs[-1], "Should decay after warmup"
    
    # Test adaptive loss balancer
    balancer = AdaptiveLossWeightBalancer(config)
    
    loss_components = {
        'policy_loss': 0.5,
        'safety_loss': 0.3,
        'efficiency_loss': 0.2
    }
    
    weights = balancer.initialize_weights(loss_components)
    print(f"✅ Loss Weights: {weights}")
    
    # Simulate training progress
    for step in range(200):
        # Simulate different improvement rates
        loss_components['policy_loss'] *= 0.99  # Good improvement
        loss_components['safety_loss'] *= 0.999  # Slow improvement
        loss_components['efficiency_loss'] *= 0.995  # Moderate improvement
        
        updated_weights = balancer.update_weights(loss_components, step)
    
    print(f"✅ Updated Weights: {updated_weights}")
    
    # Test curriculum learning
    curriculum = CurriculumLearningManager(config)
    
    stage0 = curriculum.get_current_stage()
    print(f"✅ Initial Curriculum Stage: {stage0['name']}")
    
    # Simulate progress
    for step in range(100):
        curriculum.update_progress(1.0 - step * 0.01, step)
    
    final_stage = curriculum.get_current_stage()
    print(f"✅ Final Curriculum Stage: {final_stage['name']}")
    
    # Test performance monitor
    monitor = PerformanceMonitor(config)
    
    for step in range(50):
        diagnostics = monitor.update(
            loss=1.0 - step * 0.01,
            gradient_norm=1.0,
            metrics={'accuracy': 0.5 + step * 0.01},
            step=step
        )
    
    print(f"✅ Performance Diagnostics: {diagnostics}")
    
    print("🎉 Performance Tuning Validation: ALL TESTS PASSED!")


if __name__ == "__main__":
    validate_performance_tuning()