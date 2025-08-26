"""
修复的BPTT循环模块 - 确保jax.lax.scan正确处理梯度流
参考GCBF+和DiffPhysDrone的时间展开实现
"""

import jax
import jax.numpy as jnp
from typing import Dict, Tuple, Callable, Any, NamedTuple
import chex
from functools import partial

class ScanCarry(NamedTuple):
    """scan函数的carry状态"""
    states: jnp.ndarray           # 当前状态 [batch_size, n_agents, state_dim]
    rnn_hidden: jnp.ndarray       # RNN隐状态（如果使用）
    rng_key: jax.random.PRNGKey   # 随机数种子
    step_count: int               # 步数计数

class ScanOutput(NamedTuple):
    """scan函数的输出"""
    states: jnp.ndarray
    actions: jnp.ndarray
    cbf_values: jnp.ndarray
    cbf_derivatives: jnp.ndarray
    safety_violations: jnp.ndarray
    collision_distances: jnp.ndarray
    
def create_scan_step_fn(
    policy_apply_fn: Callable,
    gnn_apply_fn: Callable,
    physics_step_fn: Callable,
    safety_filter_fn: Optional[Callable],
    config: Dict[str, Any]
) -> Callable:
    """
    创建scan步骤函数 - 修复版本确保梯度流
    """
    
    def scan_step(carry: ScanCarry, inputs: Dict[str, jnp.ndarray]) -> Tuple[ScanCarry, ScanOutput]:
        """单个时间步计算"""
        
        # 解包carry
        current_states = carry.states
        rnn_hidden = carry.rnn_hidden
        rng_key = carry.rng_key
        step_count = carry.step_count
        
        # 解包输入
        target_velocities = inputs['target_velocities']
        policy_params = inputs['policy_params']
        gnn_params = inputs['gnn_params']
        
        batch_size, n_agents, state_dim = current_states.shape
        
        # 1. 构建图数据 - 使用正确的函数
        from core.training import construct_graph_batch
        graph_batch = construct_graph_batch(current_states, config.get('sensing_radius', 2.0))
        
        # 2. GNN计算CBF - 批量处理
        nodes_flat = graph_batch['nodes'].reshape(-1, state_dim)
        edges_flat = graph_batch['edges'].reshape(-1, n_agents, graph_batch['edges'].shape[-1])
        adjacency_flat = graph_batch['adjacency'].reshape(-1, n_agents)
        
        flat_graph = {
            'nodes': nodes_flat,
            'edges': edges_flat,
            'adjacency': adjacency_flat,
            'n_nodes': n_agents
        }
        
        cbf_output = gnn_apply_fn(gnn_params, flat_graph)
        
        if isinstance(cbf_output, tuple):
            cbf_values_flat, cbf_grads_flat = cbf_output
        else:
            cbf_values_flat = cbf_output
            cbf_grads_flat = jnp.zeros((batch_size * n_agents, 6))
        
        cbf_values = cbf_values_flat.reshape(batch_size, n_agents)
        cbf_grads = cbf_grads_flat.reshape(batch_size, n_agents, -1)
        
        # 3. 策略网络计算动作
        policy_inputs = jnp.concatenate([
            current_states,
            target_velocities,
            cbf_values[..., None]
        ], axis=-1)
        
        actions_flat = policy_apply_fn(
            policy_params,
            policy_inputs.reshape(-1, policy_inputs.shape[-1])
        )
        nominal_actions = actions_flat.reshape(batch_size, n_agents, -1)
        
        # 4. 安全过滤器（如果提供）
        if safety_filter_fn is not None:
            safe_actions, _ = safety_filter_fn(
                nominal_actions, cbf_values, cbf_grads, current_states
            )
        else:
            safe_actions = nominal_actions
        
        # 5. 添加探索噪声（训练时）
        if config.get('add_exploration_noise', False):
            rng_key, noise_key = jax.random.split(rng_key)
            noise_scale = config.get('exploration_noise_scale', 0.01)
            noise = jax.random.normal(noise_key, safe_actions.shape) * noise_scale
            safe_actions = safe_actions + noise
        
        # 6. 物理仿真
        next_states = physics_step_fn(current_states, safe_actions)
        
        # 7. 计算CBF时间导数
        from core.training import compute_cbf_time_derivative
        cbf_derivatives = compute_cbf_time_derivative(
            cbf_values, cbf_grads, current_states, safe_actions, config.get('dt', 0.02)
        )
        
        # 8. 计算碰撞距离和安全违规
        collision_distances = compute_pairwise_distances(current_states)
        safety_violations = collision_distances < config.get('safety_radius', 0.5)
        
        # 组装输出
        outputs = ScanOutput(
            states=current_states,
            actions=safe_actions,
            cbf_values=cbf_values,
            cbf_derivatives=cbf_derivatives,
            safety_violations=safety_violations.astype(jnp.float32),
            collision_distances=collision_distances
        )
        
        # 创建新carry
        new_carry = ScanCarry(
            states=next_states,
            rnn_hidden=rnn_hidden,  # 暂时未使用RNN
            rng_key=rng_key,
            step_count=step_count + 1
        )
        
        return new_carry, outputs
    
    return scan_step

def compute_pairwise_distances(states: jnp.ndarray) -> jnp.ndarray:
    """
    计算智能体间的成对最小距离
    
    Args:
        states: [batch_size, n_agents, state_dim]
        
    Returns:
        min_distances: [batch_size, n_agents]
    """
    positions = states[..., :3]
    
    # 批量计算距离
    pos_i = positions[:, :, None, :]  # [batch, n_agents, 1, 3]
    pos_j = positions[:, None, :, :]  # [batch, 1, n_agents, 3]
    distances = jnp.linalg.norm(pos_i - pos_j, axis=-1)  # [batch, n_agents, n_agents]
    
    # 排除自身
    n_agents = distances.shape[-1]
    eye_mask = jnp.eye(n_agents)[None, :, :]
    distances = jnp.where(eye_mask, jnp.inf, distances)
    
    # 最小距离
    min_distances = jnp.min(distances, axis=-1)
    
    return min_distances

@partial(jax.jit, static_argnums=(1, 2, 3, 4, 5))
def execute_bptt_rollout(
    initial_carry: ScanCarry,
    scan_step_fn: Callable,
    seq_length: int,
    policy_params: Dict,
    gnn_params: Dict,
    target_vel_sequence: jnp.ndarray
) -> Tuple[ScanCarry, ScanOutput]:
    """
    执行BPTT展开 - 修复版本
    """
    
    # 准备输入序列
    inputs_sequence = {
        'target_velocities': target_vel_sequence,  # [seq_len, batch, n_agents, 3]
        'policy_params': policy_params,
        'gnn_params': gnn_params
    }
    
    # 使用scan执行时间展开
    final_carry, outputs = jax.lax.scan(
        scan_step_fn,
        initial_carry,
        inputs_sequence,
        length=seq_length
    )
    
    return final_carry, outputs

def rollout_with_checkpointing(
    initial_states: jnp.ndarray,
    target_velocities: jnp.ndarray,  # [batch, seq_len, n_agents, 3]
    policy_apply_fn: Callable,
    gnn_apply_fn: Callable,
    physics_step_fn: Callable,
    policy_params: Dict,
    gnn_params: Dict,
    config: Dict[str, Any]
) -> Tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]]:
    """
    带检查点的rollout执行
    """
    batch_size, seq_length, n_agents, _ = target_velocities.shape
    state_dim = initial_states.shape[-1]
    
    # 创建scan步骤函数
    scan_step_fn = create_scan_step_fn(
        policy_apply_fn,
        gnn_apply_fn,
        physics_step_fn,
        safety_filter_fn=None,  # 暂时不使用安全过滤器
        config=config
    )
    
    # 使用检查点
    if config.get('use_gradient_checkpointing', True):
        scan_step_fn = jax.checkpoint(scan_step_fn)
    
    # 初始carry
    initial_carry = ScanCarry(
        states=initial_states,
        rnn_hidden=jnp.zeros((batch_size, n_agents, config.get('rnn_hidden_dim', 64))),
        rng_key=jax.random.PRNGKey(config.get('seed', 0)),
        step_count=0
    )
    
    # 转置目标速度序列
    target_vel_sequence = target_velocities.transpose(1, 0, 2, 3)
    
    # 执行rollout
    final_carry, outputs = execute_bptt_rollout(
        initial_carry,
        scan_step_fn,
        seq_length,
        policy_params,
        gnn_params,
        target_vel_sequence
    )
    
    # 转置输出回 [batch, seq_len, ...]
    rollout_data = {
        'states': outputs.states.transpose(1, 0, 2, 3),
        'actions': outputs.actions.transpose(1, 0, 2, 3),
        'cbf_values': outputs.cbf_values.transpose(1, 0, 2),
        'cbf_derivatives': outputs.cbf_derivatives.transpose(1, 0, 2),
        'collision_distances': outputs.collision_distances.transpose(1, 0, 2),
        'safety_violations': outputs.safety_violations.transpose(1, 0, 2)
    }
    
    # 计算指标
    metrics = {
        'total_safety_violations': jnp.sum(outputs.safety_violations),
        'avg_collision_distance': jnp.mean(outputs.collision_distances),
        'min_collision_distance': jnp.min(outputs.collision_distances),
        'avg_cbf_value': jnp.mean(outputs.cbf_values),
        'cbf_std': jnp.std(outputs.cbf_values),
        'final_step_count': final_carry.step_count
    }
    
    return rollout_data, metrics

def test_loop_jit_compatibility():
    """测试循环模块的JIT兼容性"""
    print("测试循环模块JIT兼容性...")
    
    try:
        # 创建测试数据
        batch_size = 2
        n_agents = 3
        state_dim = 13
        seq_len = 5
        
        initial_states = jnp.zeros((batch_size, n_agents, state_dim))
        initial_states = initial_states.at[..., :3].set(
            jax.random.uniform(jax.random.PRNGKey(42), (batch_size, n_agents, 3))
        )
        initial_states = initial_states.at[..., 6].set(1.0)
        
        target_velocities = jax.random.normal(
            jax.random.PRNGKey(43), (batch_size, seq_len, n_agents, 3)
        )
        
        # 创建虚拟函数
        def dummy_policy(params, x):
            return jnp.zeros((x.shape[0], 3))
        
        def dummy_gnn(params, graph):
            n_nodes = graph['nodes'].shape[0]
            return jnp.ones(n_nodes) * 0.5, jnp.zeros((n_nodes, 6))
        
        def dummy_physics(states, actions):
            return states + 0.01 * actions[..., :states.shape[-1]]
        
        # 虚拟参数
        policy_params = {'dummy': jnp.array(0.0)}
        gnn_params = {'dummy': jnp.array(0.0)}
        
        config = {
            'sensing_radius': 2.0,
            'safety_radius': 0.5,
            'dt': 0.02,
            'use_gradient_checkpointing': False
        }
        
        # 执行rollout
        rollout_data, metrics = rollout_with_checkpointing(
            initial_states,
            target_velocities,
            dummy_policy,
            dummy_gnn,
            dummy_physics,
            policy_params,
            gnn_params,
            config
        )
        
        print(f"✅ Rollout成功完成")
        print(f"  - 最终状态形状: {rollout_data['states'].shape}")
        print(f"  - 平均CBF值: {metrics['avg_cbf_value']:.4f}")
        print(f"  - 最小碰撞距离: {metrics['min_collision_distance']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

__all__ = [
    'ScanCarry',
    'ScanOutput',
    'create_scan_step_fn',
    'execute_bptt_rollout',
    'rollout_with_checkpointing',
    'compute_pairwise_distances',
    'test_loop_jit_compatibility'
]