"""
Simplified policy network for testing
"""

import jax.numpy as jnp
import flax.linen as nn
from flax import struct
import chex


@struct.dataclass
class SimplePolicyParams:
    """Simplified policy parameters for testing"""
    max_thrust: float = 0.8


class SimplePolicyNetworkMLP(nn.Module):
    """Simplified MLP policy network for debugging"""
    
    output_dim: int = 3
    
    def setup(self):
        self.dense1 = nn.Dense(64)
        self.dense2 = nn.Dense(64) 
        self.output_layer = nn.Dense(self.output_dim)
    
    def __call__(self, x: chex.Array, rnn_state=None) -> tuple:
        """
        Forward pass through simplified policy.
        
        Args:
            x: Input observations [batch_size, obs_dim]
            rnn_state: Unused (for compatibility)
            
        Returns:
            (control_output, new_rnn_state)
        """
        # Hidden layers
        x = nn.relu(self.dense1(x))
        x = nn.relu(self.dense2(x))
        
        # Output layer with tanh activation for bounded control
        x = self.output_layer(x)
        x = nn.tanh(x)  # Bound to [-1, 1]
        
        # Scale to appropriate thrust range
        x = x * 0.8  # Scale to [-0.8, 0.8]
        
        return x, None  # Return (control, rnn_state)


def create_simple_policy_network():
    """Create a simple policy network for testing"""
    return SimplePolicyNetworkMLP()