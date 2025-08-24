"""
策略网络实现
基于Flax的MLP和RNN架构
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple, Optional
import chex


class PolicyMLP(nn.Module):
    """
    基础MLP策略网络
    输入: 无人机状态
    输出: 期望控制输入
    """
    hidden_dims: Tuple[int, ...] = (256, 256, 128)
    output_dim: int = 3  # 3D推力向量
    
    @nn.compact
    def __call__(self, state: chex.Array) -> chex.Array:
        """
        Args:
            state: 无人机状态向量 [position(3) + velocity(3) + orientation(4) + angular_vel(3)] = 13维
        
        Returns:
            action: 期望推力向量 [3]
        """
        x = state
        
        # 隐藏层
        for hidden_dim in self.hidden_dims:
            x = nn.Dense(hidden_dim)(x)
            x = nn.leaky_relu(x)
        
        # 输出层
        action = nn.Dense(self.output_dim)(x)
        
        # 限制输出范围（-max_thrust到max_thrust）
        max_thrust = 15.0  # 略大于物理限制，让网络学习
        action = nn.tanh(action) * max_thrust
        
        return action


class PolicyRNN(nn.Module):
    """
    基于RNN的策略网络（为后续扩展准备）
    """
    hidden_size: int = 128
    output_dim: int = 3
    
    def setup(self):
        self.rnn_cell = nn.GRUCell()
        self.output_layer = nn.Dense(self.output_dim)
    
    def __call__(self, 
                 state: chex.Array, 
                 hidden: chex.Array) -> Tuple[chex.Array, chex.Array]:
        """
        Args:
            state: 当前状态 [13]
            hidden: RNN隐藏状态 [hidden_size]
            
        Returns:
            (action, new_hidden): 控制输出和新隐藏状态
        """
        # RNN步进
        new_hidden = self.rnn_cell(hidden, state)
        
        # 输出层
        action = self.output_layer(new_hidden)
        action = nn.tanh(action) * 15.0
        
        return action, new_hidden
    
    def init_hidden(self, batch_size: int = 1) -> chex.Array:
        """初始化隐藏状态"""
        return jnp.zeros((batch_size, self.hidden_size))


def state_to_vector(drone_state) -> chex.Array:
    """将DroneState转换为向量形式，供神经网络使用"""
    return jnp.concatenate([
        drone_state.position,           # [3]
        drone_state.velocity,           # [3]  
        drone_state.orientation,        # [4]
        drone_state.angular_velocity    # [3]
    ])  # 总共13维


def create_policy_model(model_type: str = "mlp") -> nn.Module:
    """创建策略模型"""
    if model_type == "mlp":
        return PolicyMLP()
    elif model_type == "rnn":
        return PolicyRNN()
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# JIT编译的策略函数
@jax.jit
def apply_policy(params, model, state_vector):
    """应用策略模型"""
    return model.apply(params, state_vector)