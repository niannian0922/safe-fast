"""
Policy networks for safe agile flight system.

This module implements neural network policies combining insights from:
1. GCBF+ (MIT-REALM): Distributed safe control with graph neural networks
2. DiffPhysDrone (SJTU): End-to-end vision-based flight control

The policy architecture supports both single-agent and multi-agent scenarios
with recurrent memory for temporal consistency and safety awareness.
"""

import jax
import jax.numpy as jnp
from jax import random
from typing import NamedTuple, Optional, Tuple, Callable, Any
import chex
from flax import linen as nn
from flax import struct
import optax


# =============================================================================
# POLICY STATE REPRESENTATIONS  
# =============================================================================

@struct.dataclass
class PolicyState:
    """State representation for policy networks with memory."""
    rnn_state: chex.Array  # Hidden state for recurrent networks
    step_count: int  # Current step counter for curriculum learning
    action_history: chex.Array  # History of previous actions for smoothness


@struct.dataclass 
class PolicyParams:
    """Parameters for policy network configuration."""
    # Network architecture
    hidden_dims: Tuple[int, ...] = (256, 256)
    rnn_hidden_size: int = 256
    activation: str = "relu"
    use_rnn: bool = True
    
    # Control constraints
    max_thrust: float = 0.8
    thrust_smoothing: float = 0.95  # Exponential smoothing factor
    
    # Safety integration
    enable_cbf_integration: bool = True
    safety_margin: float = 0.1
    
    # Training hyperparameters
    action_penalty_coef: float = 0.01
    smoothness_penalty_coef: float = 0.001


# =============================================================================
# BASE POLICY NETWORK
# =============================================================================

class MLPBlock(nn.Module):
    """Multi-layer perceptron block with configurable activation."""
    
    features: int
    activation: str = "relu"
    use_bias: bool = True
    dropout_rate: float = 0.0
    
    def setup(self):
        self.dense = nn.Dense(self.features, use_bias=self.use_bias)
        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(self.dropout_rate)
        else:
            self.dropout = None
            
    def __call__(self, x: chex.Array, training: bool = False) -> chex.Array:
        x = self.dense(x)
        
        # Apply activation
        if self.activation == "relu":
            x = nn.relu(x)
        elif self.activation == "tanh":
            x = nn.tanh(x)
        elif self.activation == "swish":
            x = nn.swish(x)
        elif self.activation == "gelu":
            x = nn.gelu(x)
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")
            
        # Apply dropout if specified
        if self.dropout is not None:
            x = self.dropout(x, deterministic=not training)
            
        return x


class PolicyNetworkMLP(nn.Module):
    """
    Basic MLP policy network for single-agent control.
    
    Based on DiffPhysDrone's lightweight architecture but enhanced with
    JAX/Flax implementation and improved numerical stability.
    """
    
    params: PolicyParams
    output_dim: int = 3  # 3D thrust commands
    
    def setup(self):
        # Create MLP layers
        self.layers = []
        for i, features in enumerate(self.params.hidden_dims):
            self.layers.append(
                MLPBlock(
                    features=features,
                    activation=self.params.activation,
                    dropout_rate=0.1 if i < len(self.params.hidden_dims) - 1 else 0.0
                )
            )
        
        # Output layer with tanh activation for bounded control
        self.output_layer = nn.Dense(self.output_dim)
    
    def __call__(
        self, 
        observations: chex.Array,  # [batch_size, obs_dim] 
        training: bool = False
    ) -> chex.Array:
        """
        Forward pass through MLP policy.
        
        Args:
            observations: Input observations 
            training: Whether in training mode
            
        Returns:
            Control commands in [-1, 1] range
        """
        x = observations
        
        # Forward through hidden layers
        for layer in self.layers:
            x = layer(x, training=training)
        
        # Output layer
        x = self.output_layer(x)
        
        # Apply tanh for bounded output
        control_output = nn.tanh(x)
        
        return control_output


# =============================================================================
# RECURRENT POLICY NETWORK (DiffPhysDrone-inspired)
# =============================================================================

class PolicyNetworkRNN(nn.Module):
    """
    Recurrent policy network for temporal consistency.
    
    Implements key insights from DiffPhysDrone's CRNN architecture:
    - GRU for temporal memory and planning consistency
    - Lightweight design for real-time deployment  
    - Integrated action smoothing
    """
    
    params: PolicyParams
    output_dim: int = 3
    
    def setup(self):
        # Feature extraction layers
        self.feature_layers = []
        for features in self.params.hidden_dims[:-1]:  # All but last
            self.feature_layers.append(
                MLPBlock(features=features, activation=self.params.activation)
            )
        
        # Recurrent layer (GRU for efficiency)
        self.rnn = nn.GRU(features=self.params.rnn_hidden_size)
        
        # Output projection
        final_hidden_dim = self.params.hidden_dims[-1] if self.params.hidden_dims else self.params.rnn_hidden_size
        self.output_projection = nn.Dense(final_hidden_dim)
        self.control_head = nn.Dense(self.output_dim)
        
        # Action history integration for smoothness
        self.action_history_proj = nn.Dense(self.params.rnn_hidden_size // 4)
    
    def __call__(
        self,
        observations: chex.Array,  # [batch_size, seq_len, obs_dim]
        rnn_state: chex.Array,     # [batch_size, rnn_hidden_size] 
        action_history: Optional[chex.Array] = None,  # [batch_size, history_len, 3]
        training: bool = False
    ) -> Tuple[chex.Array, chex.Array]:
        """
        Forward pass through RNN policy.
        
        Args:
            observations: Input observations sequence
            rnn_state: Previous RNN hidden state
            action_history: Previous action history for smoothing
            training: Training mode flag
            
        Returns:
            (control_commands, new_rnn_state)
        """
        batch_size = observations.shape[0]
        
        # Feature extraction
        x = observations
        for layer in self.feature_layers:
            x = layer(x, training=training)
        
        # Process action history for smoothness (DiffPhysDrone insight)
        if action_history is not None:
            action_features = self.action_history_proj(
                action_history.reshape(batch_size, -1)
            )
            # Combine with current features
            x = jnp.concatenate([x, action_features], axis=-1)
        
        # RNN processing
        rnn_output, new_rnn_state = self.rnn(x[:, None, :], rnn_state)  # Add time dimension
        rnn_output = rnn_output[:, 0, :]  # Remove time dimension
        
        # Output projection
        x = self.output_projection(rnn_output)
        x = nn.relu(x)
        
        # Control head with bounded output
        control_output = self.control_head(x)
        control_output = nn.tanh(control_output)  # Bound to [-1, 1]
        
        return control_output, new_rnn_state


# =============================================================================
# POLICY FACTORY AND UTILITIES
# =============================================================================

def create_policy_network(
    params: PolicyParams,
    network_type: str = "mlp",
    output_dim: int = 3
) -> nn.Module:
    """
    Factory function to create policy networks.
    
    Args:
        params: Policy parameters
        network_type: Type of network ("mlp" or "rnn")  
        output_dim: Output dimension
        
    Returns:
        Policy network instance
    """
    if network_type == "mlp":
        return PolicyNetworkMLP(params=params, output_dim=output_dim)
    elif network_type == "rnn":
        return PolicyNetworkRNN(params=params, output_dim=output_dim)
    else:
        raise ValueError(f"Unknown network type: {network_type}")


def init_policy_state(
    policy_params: PolicyParams,
    rng_key: chex.PRNGKey,
    batch_size: int = 1
) -> PolicyState:
    """Initialize policy state."""
    rnn_state = jnp.zeros((batch_size, policy_params.rnn_hidden_size))
    action_history = jnp.zeros((batch_size, 3, 3))  # Last 3 actions
    
    return PolicyState(
        rnn_state=rnn_state,
        step_count=0,
        action_history=action_history
    )


def apply_control_constraints(
    raw_control: chex.Array,
    params: PolicyParams,
    previous_action: Optional[chex.Array] = None
) -> chex.Array:
    """
    Apply control constraints and smoothing.
    
    Implements control processing from DiffPhysDrone:
    - Thrust magnitude constraints
    - Temporal smoothing for stability
    """
    # Scale to actual thrust range
    control_output = raw_control * params.max_thrust
    
    # Apply exponential smoothing if previous action available
    if previous_action is not None:
        control_output = (
            params.thrust_smoothing * previous_action + 
            (1.0 - params.thrust_smoothing) * control_output
        )
    
    # Enforce hard constraints
    control_output = jnp.clip(control_output, -params.max_thrust, params.max_thrust)
    
    return control_output


# =============================================================================
# POLICY EVALUATION AND UTILITIES
# =============================================================================

@jax.jit
def evaluate_policy_mlp(
    policy: nn.Module,
    params: chex.Array,
    observations: chex.Array,
    training: bool = False
) -> chex.Array:
    """JIT-compiled policy evaluation for MLP."""
    return policy.apply(params, observations, training=training)


@jax.jit 
def evaluate_policy_rnn(
    policy: nn.Module,
    params: chex.Array,
    observations: chex.Array,
    rnn_state: chex.Array,
    action_history: Optional[chex.Array] = None,
    training: bool = False
) -> Tuple[chex.Array, chex.Array]:
    """JIT-compiled policy evaluation for RNN."""
    return policy.apply(
        params, observations, rnn_state, action_history, training=training
    )


def compute_policy_loss_components(
    predicted_actions: chex.Array,
    target_actions: chex.Array,
    action_history: chex.Array,
    params: PolicyParams
) -> Tuple[chex.Array, dict]:
    """
    Compute policy loss components following DiffPhysDrone methodology.
    
    Args:
        predicted_actions: Network output actions
        target_actions: Target actions (from QP solver)
        action_history: Previous actions for smoothness
        params: Policy parameters
        
    Returns:
        (total_loss, loss_dict)
    """
    # Action tracking loss (primary objective)
    action_loss = jnp.mean((predicted_actions - target_actions) ** 2)
    
    # Action magnitude penalty (energy efficiency)
    magnitude_loss = jnp.mean(jnp.sum(predicted_actions ** 2, axis=-1))
    
    # Smoothness penalty (based on action derivatives)
    if action_history.shape[-2] > 1:  # Need at least 2 history steps
        action_derivatives = jnp.diff(action_history, axis=-2)
        smoothness_loss = jnp.mean(jnp.sum(action_derivatives ** 2, axis=-1))
    else:
        smoothness_loss = 0.0
    
    # Combine losses
    total_loss = (
        action_loss + 
        params.action_penalty_coef * magnitude_loss +
        params.smoothness_penalty_coef * smoothness_loss
    )
    
    loss_dict = {
        "action_loss": action_loss,
        "magnitude_loss": magnitude_loss, 
        "smoothness_loss": smoothness_loss,
        "total_loss": total_loss
    }
    
    return total_loss, loss_dict


# =============================================================================
# TESTING AND VALIDATION
# =============================================================================

def validate_policy_implementation():
    """Validate policy network implementation."""
    print("🧪 Validating Policy Network Implementation...")
    
    # Create test parameters
    params = PolicyParams(
        hidden_dims=(128, 64),
        rnn_hidden_size=128,
        use_rnn=True
    )
    
    # Test MLP policy
    mlp_policy = create_policy_network(params, "mlp")
    
    # Initialize parameters
    key = random.PRNGKey(42)
    dummy_obs = jnp.ones((4, 10))  # Batch of 4, obs dim 10
    
    mlp_params = mlp_policy.init(key, dummy_obs)
    mlp_output = mlp_policy.apply(mlp_params, dummy_obs)
    
    print(f"✅ MLP Policy: input {dummy_obs.shape} -> output {mlp_output.shape}")
    assert mlp_output.shape == (4, 3), f"Expected (4, 3), got {mlp_output.shape}"
    
    # Test RNN policy  
    rnn_policy = create_policy_network(params, "rnn")
    rnn_state = jnp.zeros((4, params.rnn_hidden_size))
    
    rnn_params = rnn_policy.init(key, dummy_obs, rnn_state)
    rnn_output, new_rnn_state = rnn_policy.apply(rnn_params, dummy_obs, rnn_state)
    
    print(f"✅ RNN Policy: input {dummy_obs.shape} -> output {rnn_output.shape}")
    assert rnn_output.shape == (4, 3), f"Expected (4, 3), got {rnn_output.shape}"
    assert new_rnn_state.shape == rnn_state.shape, "RNN state shape mismatch"
    
    # Test JIT compilation
    jit_mlp = jax.jit(mlp_policy.apply)
    jit_output = jit_mlp(mlp_params, dummy_obs)
    
    print(f"✅ JIT Compilation: MLP policy compiles successfully")
    assert jnp.allclose(mlp_output, jit_output), "JIT output mismatch"
    
    # Test control constraints
    raw_control = jnp.array([[0.8, -0.6, 1.2], [-0.5, 0.9, -0.3]])
    prev_action = jnp.array([[0.1, -0.1, 0.2], [-0.2, 0.3, -0.1]])
    
    constrained_control = apply_control_constraints(raw_control, params, prev_action)
    print(f"✅ Control Constraints: Applied successfully")
    
    # Verify bounds
    assert jnp.all(jnp.abs(constrained_control) <= params.max_thrust), "Control bounds violated"
    
    print("🎉 Policy Network Validation: ALL TESTS PASSED!")


if __name__ == "__main__":
    validate_policy_implementation()