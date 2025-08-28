"""
Enhanced Policy Networks for Safe Agile Flight System

This module implements production-ready neural network policies that can be
properly trained end-to-end with gradient-based optimization.

Key improvements:
1. Multi-layer perceptron with configurable depth and width
2. Proper initialization strategies for stable training
3. Residual connections for better gradient flow
4. Action history integration for temporal consistency
5. Adaptive control scaling based on drone state
"""

import jax
import jax.numpy as jnp
from jax import random, nn
import flax.linen as nn_linen
from typing import Tuple, Optional, Dict
import chex
from dataclasses import dataclass


@dataclass
class EnhancedPolicyConfig:
    """Configuration for enhanced policy networks."""
    # Architecture
    hidden_dims: Tuple[int, ...] = (512, 256, 128)
    activation: str = "swish"  # Better than ReLU for control tasks
    output_activation: str = "tanh"
    use_batch_norm: bool = True
    dropout_rate: float = 0.1
    
    # Initialization
    kernel_init_scale: float = 1.0
    bias_init_scale: float = 0.1
    output_init_scale: float = 0.1  # Small output initialization
    
    # Control features
    use_residual_connections: bool = True
    use_action_history: bool = True
    history_length: int = 3
    action_smoothing: float = 0.05
    
    # Adaptive scaling
    use_adaptive_scaling: bool = True
    velocity_scaling: bool = True
    distance_scaling: bool = True


class ResidualMLPBlock(nn_linen.Module):
    """Residual MLP block for better gradient flow."""
    
    features: int
    activation: str = "swish"
    use_batch_norm: bool = True
    dropout_rate: float = 0.1
    
    @nn_linen.compact
    def __call__(self, x: chex.Array, training: bool = False) -> chex.Array:
        residual = x
        
        # First layer
        x = nn_linen.Dense(self.features, use_bias=not self.use_batch_norm)(x)
        if self.use_batch_norm:
            x = nn_linen.BatchNorm(use_running_average=not training)(x)
        
        # Activation
        if self.activation == "swish":
            x = nn.swish(x)
        elif self.activation == "gelu":
            x = nn.gelu(x)
        elif self.activation == "relu":
            x = nn.relu(x)
        else:
            x = nn.tanh(x)
        
        # Dropout
        if self.dropout_rate > 0:
            x = nn_linen.Dropout(self.dropout_rate)(x, deterministic=not training)
        
        # Residual connection (if dimensions match)
        if residual.shape[-1] == self.features:
            x = x + residual
        
        return x


class EnhancedPolicyMLP(nn_linen.Module):
    """Enhanced MLP policy network with advanced features."""
    
    config: EnhancedPolicyConfig
    output_dim: int = 3
    
    def setup(self):
        # Feature extraction layers - Flax way
        feature_layers = {}
        for i, features in enumerate(self.config.hidden_dims):
            feature_layers[f'layer_{i}'] = ResidualMLPBlock(
                features=features,
                activation=self.config.activation,
                use_batch_norm=self.config.use_batch_norm,
                dropout_rate=self.config.dropout_rate
            )
        self.feature_layers = feature_layers
        
        # Action history processing
        if self.config.use_action_history:
            self.action_history_processor = nn_linen.Dense(64)
        
        # Output head
        self.output_layer = nn_linen.Dense(
            self.output_dim,
            kernel_init=nn_linen.initializers.variance_scaling(
                self.config.output_init_scale, "fan_in", "truncated_normal"
            )
        )
        
        # Adaptive scaling network
        if self.config.use_adaptive_scaling:
            self.scaling_network = nn_linen.Dense(3)  # Scale per output dimension
    
    def __call__(
        self,
        observations: chex.Array,  # [batch_size, obs_dim]
        action_history: Optional[chex.Array] = None,  # [batch_size, history_len, 3]
        training: bool = True
    ) -> chex.Array:
        """
        Enhanced forward pass with adaptive scaling and history integration.
        
        Args:
            observations: State observations
            action_history: Previous actions for temporal consistency
            training: Training mode flag
            
        Returns:
            Control commands with adaptive scaling
        """
        batch_size = observations.shape[0]
        x = observations
        
        # Process action history if available
        if self.config.use_action_history and action_history is not None:
            # Flatten action history and process
            action_features = self.action_history_processor(
                action_history.reshape(batch_size, -1)
            )
            x = jnp.concatenate([x, action_features], axis=-1)
        
        # Feature extraction through residual blocks
        for layer_name in sorted(self.feature_layers.keys()):
            x = self.feature_layers[layer_name](x, training=training)
        
        # Base control output
        base_control = self.output_layer(x)
        
        # Apply output activation
        if self.config.output_activation == "tanh":
            base_control = nn.tanh(base_control)
        elif self.config.output_activation == "sigmoid":
            base_control = nn.sigmoid(base_control) * 2.0 - 1.0  # Map to [-1, 1]
        
        # Adaptive scaling based on state
        if self.config.use_adaptive_scaling:
            # Extract relevant state features for scaling
            position = observations[:, :3]  # Assume first 3 are position
            velocity = observations[:, 3:6]  # Assume next 3 are velocity
            
            # Distance-based scaling
            if self.config.distance_scaling:
                distance_to_target = jnp.linalg.norm(observations[:, 6:9], axis=1, keepdims=True)  # Target relative pos
                distance_scale = jnp.minimum(2.0, jnp.maximum(0.5, distance_to_target / 2.0))
            else:
                distance_scale = 1.0
            
            # Velocity-based scaling
            if self.config.velocity_scaling:
                velocity_magnitude = jnp.linalg.norm(velocity, axis=1, keepdims=True)
                velocity_scale = jnp.maximum(0.5, 1.0 - velocity_magnitude / 10.0)  # Reduce control at high speed
            else:
                velocity_scale = 1.0
            
            # Learned adaptive scaling
            learned_scale = nn.sigmoid(self.scaling_network(x)) + 0.1  # Scale between 0.1 and 1.1
            
            # Combine scaling factors
            total_scale = distance_scale * velocity_scale * learned_scale
            adaptive_control = base_control * total_scale
        else:
            adaptive_control = base_control
        
        return adaptive_control


class ActionHistoryBuffer:
    """Buffer for maintaining action history for temporal consistency."""
    
    def __init__(self, history_length: int = 3, action_dim: int = 3):
        self.history_length = history_length
        self.action_dim = action_dim
    
    def init_buffer(self, batch_size: int) -> chex.Array:
        """Initialize empty action history buffer."""
        return jnp.zeros((batch_size, self.history_length, self.action_dim))
    
    def update_buffer(
        self, 
        buffer: chex.Array, 
        new_action: chex.Array
    ) -> chex.Array:
        """Update action history buffer with new action."""
        # Shift history and add new action
        new_buffer = jnp.concatenate([
            buffer[:, 1:, :],  # Remove oldest
            new_action[:, None, :]  # Add newest
        ], axis=1)
        return new_buffer


def create_enhanced_policy_network(config: EnhancedPolicyConfig, output_dim: int = 3):
    """Factory function for creating enhanced policy networks."""
    return EnhancedPolicyMLP(config=config, output_dim=output_dim)


def apply_action_smoothing(
    new_action: chex.Array,
    previous_action: chex.Array,
    smoothing_factor: float = 0.05
) -> chex.Array:
    """Apply exponential smoothing to actions for stability."""
    return (1.0 - smoothing_factor) * previous_action + smoothing_factor * new_action


# =============================================================================
# POLICY TRAINING AND EVALUATION UTILITIES
# =============================================================================

def compute_policy_training_loss(
    predicted_actions: chex.Array,
    target_actions: chex.Array,
    action_history: chex.Array,
    config: EnhancedPolicyConfig
) -> Tuple[chex.Array, Dict]:
    """
    Compute comprehensive training loss for policy network.
    
    Args:
        predicted_actions: Network predictions
        target_actions: Target actions (from safety QP or expert)
        action_history: Action history for smoothness terms
        config: Policy configuration
        
    Returns:
        total_loss: Combined loss
        metrics: Loss breakdown
    """
    # Primary action matching loss
    action_mse = jnp.mean((predicted_actions - target_actions) ** 2)
    
    # Action magnitude regularization
    magnitude_loss = jnp.mean(jnp.sum(predicted_actions ** 2, axis=-1))
    
    # Action smoothness loss (if history available)
    if action_history.shape[1] > 1:
        action_diffs = jnp.diff(action_history, axis=1)
        smoothness_loss = jnp.mean(jnp.sum(action_diffs ** 2, axis=-1))
    else:
        smoothness_loss = 0.0
    
    # Adaptive loss weights
    action_weight = 1.0
    magnitude_weight = 0.01
    smoothness_weight = 0.005 if config.action_smoothing > 0 else 0.0
    
    total_loss = (
        action_weight * action_mse +
        magnitude_weight * magnitude_loss +
        smoothness_weight * smoothness_loss
    )
    
    metrics = {
        "action_mse": action_mse,
        "magnitude_loss": magnitude_loss,
        "smoothness_loss": smoothness_loss,
        "total_policy_loss": total_loss,
        "action_accuracy": 1.0 / (1.0 + action_mse)  # Inverse relationship
    }
    
    return total_loss, metrics


def initialize_enhanced_policy(
    config: EnhancedPolicyConfig,
    key: chex.PRNGKey,
    input_dim: int = 9,  # Default observation dimension
    output_dim: int = 3
) -> Tuple[EnhancedPolicyMLP, chex.Array]:
    """
    Initialize enhanced policy network with proper parameter initialization.
    
    Args:
        config: Policy configuration
        key: Random key
        input_dim: Input observation dimension
        output_dim: Control output dimension
        
    Returns:
        policy_network: Initialized network
        params: Network parameters
    """
    # Create network
    policy = create_enhanced_policy_network(config, output_dim)
    
    # Create dummy inputs for initialization
    dummy_obs = jnp.ones((1, input_dim))
    dummy_history = jnp.zeros((1, config.history_length, output_dim))
    
    # Initialize parameters
    params = policy.init(
        key, 
        dummy_obs, 
        action_history=dummy_history if config.use_action_history else None,
        training=False
    )
    
    return policy, params


# =============================================================================
# VALIDATION AND TESTING
# =============================================================================

def validate_enhanced_policy():
    """Validate enhanced policy implementation."""
    print("🧪 Validating Enhanced Policy Network...")
    
    # Create configuration
    config = EnhancedPolicyConfig(
        hidden_dims=(256, 128, 64),
        use_action_history=True,
        use_adaptive_scaling=True
    )
    
    # Initialize policy
    key = random.PRNGKey(42)
    policy, params = initialize_enhanced_policy(config, key, input_dim=9)
    
    # Test forward pass
    batch_size = 4
    obs = random.normal(key, (batch_size, 9))
    history = jnp.zeros((batch_size, config.history_length, 3))
    
    output = policy.apply(params, obs, action_history=history, training=False)
    
    print(f"✅ Forward pass: {obs.shape} -> {output.shape}")
    assert output.shape == (batch_size, 3), f"Expected (4, 3), got {output.shape}"
    
    # Test action history buffer
    buffer = ActionHistoryBuffer(history_length=3, action_dim=3)
    init_buffer = buffer.init_buffer(batch_size)
    new_action = random.normal(key, (batch_size, 3))
    updated_buffer = buffer.update_buffer(init_buffer, new_action)
    
    print(f"✅ Action history: {init_buffer.shape} -> {updated_buffer.shape}")
    assert jnp.allclose(updated_buffer[:, -1, :], new_action), "Latest action should match"
    
    # Test JIT compilation
    jit_policy = jax.jit(policy.apply, static_argnames=['training'])
    jit_output = jit_policy(params, obs, action_history=history, training=False)
    
    print("✅ JIT compilation successful")
    assert jnp.allclose(output, jit_output, rtol=1e-6), "JIT output should match"
    
    # Test gradient computation
    def policy_loss_fn(policy_params):
        pred_actions = policy.apply(policy_params, obs, action_history=history, training=True)
        target_actions = jnp.zeros_like(pred_actions)
        return jnp.mean((pred_actions - target_actions) ** 2)
    
    gradients = jax.grad(policy_loss_fn)(params)
    gradient_norm = jnp.sqrt(sum(
        jnp.sum(g ** 2) for g in jax.tree_util.tree_leaves(gradients)
    ))
    
    print(f"✅ Gradient norm: {gradient_norm:.6f}")
    assert gradient_norm > 1e-8, "Gradients should be meaningful"
    
    print("🎉 Enhanced Policy Network Validation: ALL TESTS PASSED!")


if __name__ == "__main__":
    validate_enhanced_policy()