"""
修复的安全层模块 - 确保qpax求解器正确处理CBF约束和梯度传播
参考GCBF+的QP安全过滤器实现
"""

import jax
import jax.numpy as jnp
import qpax
from typing import Dict, Tuple, Optional, NamedTuple
import chex
from functools import partial

class SafetyFilterOutput(NamedTuple):
    """安全过滤器输出"""
    safe_actions: jnp.ndarray      # 安全动作
    qp_solved: jnp.ndarray         # QP求解成功标志
    slack_variables: jnp.ndarray   # 松弛变量
    qp_objective: jnp.ndarray      # QP目标值
    constraint_violations: jnp.ndarray  # 约束违反数

class CBFConstraintData(NamedTuple):
    """CBF约束数据"""
    A_cbf: jnp.ndarray            # CBF约束矩阵 [n_constraints, action_dim]
    b_cbf: jnp.ndarray            # CBF约束向量 [n_constraints]
    constraint_active: jnp.ndarray # 约束激活标志

def construct_cbf_constraints(
    cbf_values: jnp.ndarray,           # [n_agents,] 
    cbf_derivatives: jnp.ndarray,      # [n_agents,]
    cbf_gradients: jnp.ndarray,        # [n_agents, state_dim]
    current_states: jnp.ndarray,       # [n_agents, state_dim]
    dynamics_matrices: Dict[str, jnp.ndarray],  # 系统动力学矩阵
    alpha: float = 1.0,
    safety_margin: float = 0.01
) -> CBFConstraintData:
    """
    构建CBF约束 - 参考GCBF+论文的约束构建方法
    
    CBF约束形式: dh/dt + alpha * h >= -safety_margin
    即: ∇h · (f(x) + g(x) * u) + alpha * h >= -safety_margin
    整理为: (∇h · g(x)) * u >= -alpha * h - ∇h · f(x) - safety_margin
    
    Args:
        cbf_values: 当前CBF值
        cbf_derivatives: CBF时间导数
        cbf_gradients: CBF关于状态的梯度
        current_states: 当前状态
        dynamics_matrices: 动力学矩阵 {'f': f(x), 'g': g(x)}
        alpha: CBF类K函数参数
        safety_margin: 安全边界
        
    Returns:
        CBFConstraintData: CBF约束数据
    """
    n_agents = cbf_values.shape[0]
    action_dim = dynamics_matrices['g'].shape[-1]
    
    # 提取动力学 f(x) 和 g(x)
    f_x = dynamics_matrices['f']  # [n_agents, state_dim]
    g_x = dynamics_matrices['g']  # [n_agents, state_dim, action_dim]
    
    # 计算 ∇h · g(x) - 这是约束矩阵
    # cbf_gradients: [n_agents, state_dim]
    # g_x: [n_agents, state_dim, action_dim]
    A_cbf = jnp.einsum('ns,nsa->na', cbf_gradients, g_x)  # [n_agents, action_dim]
    
    # 计算约束右侧: -alpha * h - ∇h · f(x) - safety_margin
    grad_f_dot = jnp.einsum('ns,ns->n', cbf_gradients, f_x)  # [n_agents,]
    b_cbf = -alpha * cbf_values - grad_f_dot - safety_margin  # [n_agents,]
    
    # 确定哪些约束是激活的（CBF值接近0或为负）
    constraint_active = cbf_values <= 0.1  # 激活阈值
    
    return CBFConstraintData(
        A_cbf=A_cbf,
        b_cbf=b_cbf, 
        constraint_active=constraint_active
    )

def compute_system_dynamics_matrices(
    states: jnp.ndarray,
    config: Dict[str, jnp.ndarray]
) -> Dict[str, jnp.ndarray]:
    """
    计算系统动力学矩阵
    
    对于点质量模型: ẋ = [v, u] 其中状态 x = [p, v]
    - f(x) = [v, 0]  
    - g(x) = [0, I]
    
    Args:
        states: 状态 [n_agents, state_dim]
        config: 配置参数
        
    Returns:
        dynamics: 包含f和g的字典
    """
    n_agents, state_dim = states.shape
    action_dim = config.get('action_dim', 3)
    
    if state_dim >= 6:
        # 完整状态模型 [position, velocity]
        positions = states[:, :3]
        velocities = states[:, 3:6]
        
        # f(x) = [v, 0]
        f_x = jnp.concatenate([
            velocities,                           # 位置导数 = 速度
            jnp.zeros((n_agents, action_dim))     # 速度导数的偏置项
        ], axis=1)
        
        # g(x) = [0, I]  
        g_x = jnp.concatenate([
            jnp.zeros((n_agents, 3, action_dim)),     # 位置对动作的导数 = 0
            jnp.eye(action_dim)[None, :, :].repeat(n_agents, axis=0)  # 速度对动作的导数 = I
        ], axis=1)
        
    else:
        # 简化状态模型（只有位置）
        # f(x) = 0
        f_x = jnp.zeros((n_agents, state_dim))
        
        # g(x) = I
        g_x = jnp.eye(state_dim)[None, :, :].repeat(n_agents, axis=0)
    
    return {
        'f': f_x,    # [n_agents, state_dim]
        'g': g_x     # [n_agents, state_dim, action_dim] 
    }

@partial(jax.jit, static_argnums=(5, 6))
def safety_filter_qp(
    nominal_actions: jnp.ndarray,      # [n_agents, action_dim] 
    cbf_values: jnp.ndarray,           # [n_agents,]
    cbf_derivatives: jnp.ndarray,      # [n_agents,]
    cbf_gradients: jnp.ndarray,        # [n_agents, state_dim]
    current_states: jnp.ndarray,       # [n_agents, state_dim]
    action_dim: int,
    config: Dict
) -> SafetyFilterOutput:
    """
    QP安全过滤器 - 使用qpax求解CBF-QP问题
    
    目标函数: min ||u - u_nom||^2 + beta * ||slack||^2
    约束: A_cbf * u >= b_cbf - slack
          u_min <= u <= u_max  
          slack >= 0
    
    Args:
        nominal_actions: 名义动作（来自策略网络）
        cbf_values: CBF值
        cbf_derivatives: CBF时间导数 
        cbf_gradients: CBF梯度
        current_states: 当前状态
        action_dim: 动作维度
        config: 配置参数
        
    Returns:
        SafetyFilterOutput: 安全过滤器输出
    """
    n_agents = nominal_actions.shape[0]
    
    # 计算系统动力学矩阵
    dynamics = compute_system_dynamics_matrices(current_states, config)
    
    # 构建CBF约束
    cbf_constraints = construct_cbf_constraints(
        cbf_values, cbf_derivatives, cbf_gradients, current_states, 
        dynamics, config.get('alpha', 1.0), config.get('safety_margin', 0.01)
    )
    
    # QP参数
    max_action = config.get('max_action', 1.0)
    slack_penalty = config.get('slack_penalty', 1000.0)
    
    # 为每个智能体求解QP（批量处理）
    def solve_single_agent_qp(agent_idx):
        """为单个智能体求解QP"""
        
        u_nom = nominal_actions[agent_idx]  # [action_dim,]
        
        # 检查约束是否激活
        if not cbf_constraints.constraint_active[agent_idx]:
            # 约束不激活，直接返回名义动作
            return {
                'solution': u_nom,
                'solved': True,
                'slack': 0.0,
                'objective': 0.0,
                'violation': 0.0
            }
        
        # QP变量: [u, slack] 其中 u: [action_dim], slack: [1]
        n_vars = action_dim + 1
        
        # 目标函数: ||u - u_nom||^2 + slack_penalty * slack^2
        # Q = diag([I, slack_penalty])
        Q = jnp.eye(n_vars)
        Q = Q.at[-1, -1].set(slack_penalty)
        
        # q = [-2 * u_nom, 0]
        q = jnp.concatenate([-2 * u_nom, jnp.array([0.0])])
        
        # 不等式约束: A_ub * x <= b_ub
        # 1. CBF约束: A_cbf * u >= b_cbf - slack  =>  [-A_cbf, 1] * [u, slack] <= -b_cbf
        # 2. 动作界限: -max_action <= u <= max_action
        # 3. 松弛变量非负: slack >= 0  =>  -slack <= 0
        
        A_cbf_agent = cbf_constraints.A_cbf[agent_idx]  # [action_dim,]
        b_cbf_agent = cbf_constraints.b_cbf[agent_idx]  # scalar
        
        # CBF约束行
        A_cbf_row = jnp.concatenate([-A_cbf_agent, jnp.array([1.0])])  # [action_dim + 1,]
        b_cbf_row = -b_cbf_agent
        
        # 动作上界约束: u <= max_action
        A_action_upper = jnp.concatenate([jnp.eye(action_dim), jnp.zeros((action_dim, 1))], axis=1)
        b_action_upper = jnp.full(action_dim, max_action)
        
        # 动作下界约束: -u <= max_action (即 u >= -max_action)
        A_action_lower = jnp.concatenate([-jnp.eye(action_dim), jnp.zeros((action_dim, 1))], axis=1)
        b_action_lower = jnp.full(action_dim, max_action)
        
        # 松弛变量非负约束: -slack <= 0
        A_slack_nonneg = jnp.array([[0.0] * action_dim + [-1.0]])
        b_slack_nonneg = jnp.array([0.0])
        
        # 组合所有约束
        A_ub = jnp.vstack([
            A_cbf_row[None, :],     # [1, n_vars]
            A_action_upper,         # [action_dim, n_vars]
            A_action_lower,         # [action_dim, n_vars]  
            A_slack_nonneg          # [1, n_vars]
        ])
        b_ub = jnp.concatenate([
            jnp.array([b_cbf_row]),
            b_action_upper,
            b_action_lower,
            b_slack_nonneg
        ])
        
        try:
            # 使用qpax求解QP
            solution = qpax.solve_qp(
                P=Q,
                q=q,
                G=A_ub,
                h=b_ub,
                solver='osqp',
                check_primal_dual_infeasibility=False
            )
            
            if solution.primal is not None and jnp.isfinite(solution.primal).all():
                u_safe = solution.primal[:action_dim]
                slack = solution.primal[action_dim]
                
                # 计算约束违反
                constraint_violation = jnp.maximum(0.0, -b_cbf_agent - A_cbf_agent.dot(u_safe))
                
                return {
                    'solution': u_safe,
                    'solved': True,
                    'slack': slack,
                    'objective': solution.primal.T @ Q @ solution.primal + q.T @ solution.primal,
                    'violation': constraint_violation
                }
            else:
                # QP求解失败，使用备用策略
                return fallback_safety_action(u_nom, agent_idx, config)
                
        except Exception:
            # 求解器异常，使用备用策略
            return fallback_safety_action(u_nom, agent_idx, config)
    
    # 为所有智能体求解QP
    agent_results = jax.vmap(solve_single_agent_qp)(jnp.arange(n_agents))
    
    # 提取结果
    safe_actions = jnp.stack([result['solution'] for result in agent_results])
    qp_solved = jnp.array([result['solved'] for result in agent_results])
    slack_variables = jnp.array([result['slack'] for result in agent_results])
    qp_objectives = jnp.array([result['objective'] for result in agent_results])
    constraint_violations = jnp.array([result['violation'] for result in agent_results])
    
    return SafetyFilterOutput(
        safe_actions=safe_actions,
        qp_solved=qp_solved,
        slack_variables=slack_variables,
        qp_objective=qp_objectives,
        constraint_violations=constraint_violations
    )

def fallback_safety_action(
    nominal_action: jnp.ndarray,
    agent_idx: int, 
    config: Dict
) -> Dict:
    """
    备用安全策略 - 当QP求解失败时使用
    
    简单策略：将动作缩放到安全范围内并添加制动
    """
    max_action = config.get('max_action', 1.0)
    emergency_brake_factor = config.get('emergency_brake_factor', 0.5)
    
    # 限制动作范围并施加紧急制动
    safe_action = jnp.clip(nominal_action * emergency_brake_factor, -max_action, max_action)
    
    return {
        'solution': safe_action,
        'solved': False,  # 标记为未通过QP求解
        'slack': 1.0,     # 大松弛变量表示约束违反
        'objective': jnp.inf,
        'violation': 1.0
    }

@jax.jit
def apply_safety_filter(
    policy_actions: jnp.ndarray,       # [batch_size, n_agents, action_dim]
    cbf_values: jnp.ndarray,           # [batch_size, n_agents] 
    cbf_derivatives: jnp.ndarray,      # [batch_size, n_agents]
    cbf_gradients: jnp.ndarray,        # [batch_size, n_agents, state_dim]
    current_states: jnp.ndarray,       # [batch_size, n_agents, state_dim]
    config: Dict
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    批量应用安全过滤器
    
    Args:
        policy_actions: 策略网络输出的动作
        cbf_values: CBF值
        cbf_derivatives: CBF导数
        cbf_gradients: CBF梯度
        current_states: 当前状态
        config: 配置参数
        
    Returns:
        safe_actions: 安全动作
        filter_info: 过滤器信息
    """
    batch_size, n_agents, action_dim = policy_actions.shape
    
    # 为每个batch处理
    def process_batch(batch_idx):
        return safety_filter_qp(
            policy_actions[batch_idx],
            cbf_values[batch_idx],
            cbf_derivatives[batch_idx], 
            cbf_gradients[batch_idx],
            current_states[batch_idx],
            action_dim,
            config
        )
    
    # 批量处理
    batch_results = jax.vmap(process_batch)(jnp.arange(batch_size))
    
    # 整理结果
    safe_actions = batch_results.safe_actions  # [batch_size, n_agents, action_dim]
    
    filter_info = {
        'qp_solved': batch_results.qp_solved,                    # [batch_size, n_agents]
        'slack_variables': batch_results.slack_variables,         # [batch_size, n_agents]  
        'qp_objectives': batch_results.qp_objective,             # [batch_size, n_agents]
        'constraint_violations': batch_results.constraint_violations, # [batch_size, n_agents]
        'qp_success_rate': jnp.mean(batch_results.qp_solved),
        'avg_slack': jnp.mean(batch_results.slack_variables),
        'max_violation': jnp.max(batch_results.constraint_violations)
    }
    
    return safe_actions, filter_info

def validate_safety_constraints(
    actions: jnp.ndarray,              # [batch_size, n_agents, action_dim]
    cbf_values: jnp.ndarray,           # [batch_size, n_agents]
    cbf_derivatives: jnp.ndarray,      # [batch_size, n_agents]
    alpha: float = 1.0
) -> Dict[str, jnp.ndarray]:
    """
    验证安全约束是否满足
    
    检查: dh/dt + alpha * h >= 0
    
    Returns:
        validation_info: 验证信息
    """
    # 计算CBF条件
    cbf_condition = cbf_derivatives + alpha * cbf_values  # [batch_size, n_agents]
    
    # 约束违反标志
    constraint_violations = cbf_condition < 0  # [batch_size, n_agents]
    
    validation_info = {
        'cbf_condition_values': cbf_condition,
        'constraint_violations': constraint_violations,
        'violation_count': jnp.sum(constraint_violations),
        'violation_rate': jnp.mean(constraint_violations.astype(jnp.float32)),
        'min_cbf_condition': jnp.min(cbf_condition),
        'avg_cbf_condition': jnp.mean(cbf_condition)
    }
    
    return validation_info

# 创建安全过滤器的便捷函数
def create_safety_filter(config: Dict):
    """创建安全过滤器函数"""
    
    @jax.jit
    def safety_filter_fn(policy_actions, cbf_values, cbf_derivatives, cbf_gradients, states):
        return apply_safety_filter(
            policy_actions, cbf_values, cbf_derivatives, cbf_gradients, states, config
        )
    
    return safety_filter_fn

# 主要导出函数
__all__ = [
    'SafetyFilterOutput',
    'CBFConstraintData',
    'safety_filter_qp',
    'apply_safety_filter', 
    'validate_safety_constraints',
    'create_safety_filter'
]