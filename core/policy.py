"""
策略网络实现 - 修复输入维度匹配
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple, Optional
import chex

class PolicyMLP(nn.Module):
    """
    基础MLP策略网络
    输入: 状态 + 目标速度 + CBF值
    输出: 期望控制输入
    """
    hidden_dims: Tuple[int, ...] = (256, 256, 128)
    output_dim: int = 3  # 3D加速度
    
    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        """
        Args:
            x: 输入向量 [state(13) + target_vel(3) + cbf(1)] = 17维
        
        Returns:
            action: 期望加速度 [3]
        """
        # 隐藏层
        for hidden_dim in self.hidden_dims:
            x = nn.Dense(hidden_dim)(x)
            x = nn.relu(x)
        
        # 输出层
        action = nn.Dense(self.output_dim)(x)
        
        # 限制输出范围
        max_action = 10.0  # m/s^2
        action = nn.tanh(action) * max_action
        
        return action

def create_policy_model(model_type: str = "mlp") -> nn.Module:
    """创建策略模型"""
    if model_type == "mlp":
        return PolicyMLP()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

# JIT编译的策略函数
@jax.jit
def apply_policy(params, model, x):
    """应用策略模型"""
    return model.apply(params, x)

__all__ = [
    'PolicyMLP',
    'create_policy_model',
    'apply_policy'
]