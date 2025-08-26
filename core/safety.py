"""
修复的安全层模块 - 简化版本用于MVP测试
暂时不使用qpax以避免复杂性
"""

import jax
import jax.numpy as jnp
from typing import Dict, Tuple, Optional, NamedTuple
import chex
from functools import partial

class SafetyFilterOutput(NamedTuple):
    """安全过滤器输出"""
    safe_actions: jnp.ndarray
    action_modified: jnp.ndarray
    safety_margin: jnp.ndarray

def simple_safety_filter(
    nominal_actions: jnp.ndarray,
    cbf_values: jnp.ndarray,
    cbf_gradients: jnp.ndarray,
    current_states: jnp.ndarray,
    config: Dict
) -> Tuple[jnp.ndarray, SafetyFilterOutput]:
    """
    简化的安全过滤器 - 用于MVP测试
    
    使用简单的缩放策略而不是QP求解
    """
    batch_size, n_agents, action_dim = nominal_actions.shape
    
    # CBF安全条件: h > threshold
    safety_threshold = config.get('safety_threshold', 0.1)
    unsafe_mask = cbf_values < safety_threshold  # [batch, n_agents]
    
    # 对不安全的智能体缩放动作
    safety_scale = jnp.where(
        unsafe_mask,
        jnp.maximum(0.1, cbf_values / safety_threshold),
        1.0
    )  # [batch, n_agents]
    
    # 应用缩放
    safe_actions = nominal_actions * safety_scale[..., None]
    
    # 限制动作范围
    max_action = config.get('max_action', 1.0)
    safe_actions = jnp.clip(safe_actions, -max_action, max_action)
    
    # 计算安全裕度
    safety_margin = cbf_values - safety_threshold
    
    # 输出信息
    filter_output = SafetyFilterOutput(
        safe_actions=safe_actions,
        action_modified=unsafe_mask.astype(jnp.float32),
        safety_margin=safety_margin
    )
    
    return safe_actions, filter_output

def create_safety_filter(config: Dict):
    """创建安全过滤器函数"""
    
    @jax.jit
    def safety_filter_fn(nominal_actions, cbf_values, cbf_gradients, states):
        return simple_safety_filter(
            nominal_actions, cbf_values, cbf_gradients, states, config
        )
    
    return safety_filter_fn

# 导出函数
__all__ = [
    'SafetyFilterOutput',
    'simple_safety_filter',
    'create_safety_filter'
]