"""
安全模块：基于qpax的可微分安全层实现
实现GCBF+的QP安全约束
"""

import jax
import jax.numpy as jnp
import qpax
from typing import Tuple, NamedTuple, Optional
import chex


class SafetyParams(NamedTuple):
    """安全参数配置"""
    alpha: float = 1.0  # CBF class-K函数参数
    safety_margin: float = 0.1  # 安全余量
    max_control_norm: float = 15.0  # 最大控制输入范数
    qp_solver_max_iter: int = 100
    qp_tolerance: float = 1e-6


def construct_cbf_qp_matrices(u_nom: chex.Array,
                             h: float,
                             grad_h: chex.Array,
                             drone_velocity: chex.Array,
                             safety_params: SafetyParams) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
    """
    构建CBF-QP的矩阵形式
    
    QP形式: min_u 0.5 * u^T Q u + q^T u
           s.t.  G u <= h_constraint
                 A u = b (equality constraints, 暂时为空)
    
    Args:
        u_nom: 名义控制输入 [3]
        h: CBF值 (标量)
        grad_h: CBF梯度 [3] 
        drone_velocity: 无人机速度 [3]
        safety_params: 安全参数
        
    Returns:
        (Q, q, G, h_constraint): QP矩阵参数
    """
    
    # 目标函数：最小化与名义控制的偏差
    # min ||u - u_nom||^2 = min u^T u - 2 u_nom^T u + const
    Q = jnp.eye(3)  # [3, 3]
    q = -u_nom  # [3]
    
    # CBF约束：grad_h^T * u >= -alpha * h
    # 转换为不等式形式：-grad_h^T * u <= alpha * h
    
    # 为了数值稳定性，我们添加速度项
    # CBF导数 = grad_h^T * velocity + grad_h^T * u >= -alpha * h
    h_dot_current = jnp.dot(grad_h, drone_velocity)
    
    # 约束：grad_h^T * u >= -alpha * h - h_dot_current
    # 转换：-grad_h^T * u <= alpha * h + h_dot_current
    G = -grad_h[None, :]  # [1, 3]
    h_constraint = jnp.array([safety_params.alpha * h + h_dot_current + safety_params.safety_margin])  # [1]
    
    # 添加控制输入界限约束
    # |u_i| <= max_control_norm 等价于 -max_control_norm <= u_i <= max_control_norm
    # 转换为 u_i <= max_control_norm 和 -u_i <= max_control_norm
    
    # 上界约束：u <= max_control_norm
    G_upper = jnp.eye(3)  # [3, 3]
    h_upper = jnp.full(3, safety_params.max_control_norm)  # [3]
    
    # 下界约束：-u <= max_control_norm (即 u >= -max_control_norm)
    G_lower = -jnp.eye(3)  # [3, 3]
    h_lower = jnp.full(3, safety_params.max_control_norm)  # [3]
    
    # 组合所有约束
    G_combined = jnp.concatenate([G, G_upper, G_lower], axis=0)  # [7, 3]
    h_combined = jnp.concatenate([h_constraint, h_upper, h_lower])  # [7]
    
    return Q, q, G_combined, h_combined


def safety_filter(u_nom: chex.Array,
                 h: float,
                 grad_h: chex.Array,
                 drone_velocity: chex.Array,
                 safety_params: SafetyParams = None) -> chex.Array:
    """
    安全滤波器：解决CBF-QP优化问题
    
    Args:
        u_nom: 名义控制输入 [3]
        h: CBF值
        grad_h: CBF梯度 [3]
        drone_velocity: 当前速度 [3]
        safety_params: 安全参数
        
    Returns:
        u_safe: 安全的控制输入 [3]
    """
    
    if safety_params is None:
        safety_params = SafetyParams()
    
    # 构建QP矩阵
    Q, q, G, h_constraint = construct_cbf_qp_matrices(
        u_nom, h, grad_h, drone_velocity, safety_params
    )
    
    try:
        # 使用qpax求解QP
        solution = qpax.solve_qp(
            Q=Q,
            c=q,
            A_ineq=G,
            b_ineq=h_constraint,
            A_eq=None,  # 无等式约束
            b_eq=None,
            max_iter=safety_params.qp_solver_max_iter,
            tol=safety_params.qp_tolerance
        )
        
        u_safe = solution.x
        
        # 检查求解状态
        if not solution.converged:
            # 如果QP求解失败，使用回退策略
            print("警告：QP求解未收敛，使用回退策略")
            u_safe = apply_fallback_strategy(u_nom, h, safety_params)
        
    except Exception as e:
        print(f"QP求解器错误：{e}，使用回退策略")
        u_safe = apply_fallback_strategy(u_nom, h, safety_params)
    
    return u_safe


def apply_fallback_strategy(u_nom: chex.Array,
                          h: float,
                          safety_params: SafetyParams) -> chex.Array:
    """
    回退策略：当QP求解失败时使用
    
    Args:
        u_nom: 名义控制
        h: CBF值
        safety_params: 安全参数
        
    Returns:
        u_fallback: 回退控制输入
    """
    
    # 策略1：如果CBF值为负（不安全），采用保守控制
    if h < 0:
        # 紧急制动：减小推力到安全水平
        u_fallback = u_nom * 0.1
    else:
        # 如果CBF值为正（安全），使用限幅的名义控制
        u_fallback = jnp.clip(u_nom, -safety_params.max_control_norm, safety_params.max_control_norm)
    
    return u_fallback


def safety_filter_with_relaxation(u_nom: chex.Array,
                                 h: float,
                                 grad_h: chex.Array,
                                 drone_velocity: chex.Array,
                                 relaxation_penalty: float = 1000.0,
                                 safety_params: SafetyParams = None) -> Tuple[chex.Array, float]:
    """
    带松弛变量的安全滤波器
    用于处理不可行的QP问题
    
    Returns:
        (u_safe, relaxation): 安全控制和松弛变量值
    """
    
    if safety_params is None:
        safety_params = SafetyParams()
    
    # 构建带松弛变量的QP
    # 变量：[u(3维), δ(1维松弛)]
    
    # 目标函数：||u - u_nom||^2 + penalty * δ^2
    Q_extended = jnp.block([
        [jnp.eye(3), jnp.zeros((3, 1))],
        [jnp.zeros((1, 3)), jnp.array([[relaxation_penalty]])]
    ])  # [4, 4]
    
    q_extended = jnp.concatenate([-u_nom, jnp.array([0.0])])  # [4]
    
    # 约束：grad_h^T * u + δ >= -alpha * h - h_dot_current
    # 转换：-grad_h^T * u - δ <= alpha * h + h_dot_current
    h_dot_current = jnp.dot(grad_h, drone_velocity)
    
    # CBF约束矩阵
    G_cbf = jnp.concatenate([-grad_h, jnp.array([-1.0])])[None, :]  # [1, 4]
    h_cbf = jnp.array([safety_params.alpha * h + h_dot_current + safety_params.safety_margin])
    
    # 控制界限约束（只对u，不对δ）
    G_bounds = jnp.block([
        [jnp.eye(3), jnp.zeros((3, 1))],      # u <= max
        [-jnp.eye(3), jnp.zeros((3, 1))]     # -u <= max
    ])  # [6, 4]
    
    h_bounds = jnp.full(6, safety_params.max_control_norm)
    
    # 松弛变量非负约束：δ >= 0, 即 -δ <= 0
    G_relax = jnp.array([[0.0, 0.0, 0.0, -1.0]])  # [1, 4]
    h_relax = jnp.array([0.0])
    
    # 组合约束
    G_all = jnp.concatenate([G_cbf, G_bounds, G_relax], axis=0)
    h_all = jnp.concatenate([h_cbf, h_bounds, h_relax])
    
    try:
        solution = qpax.solve_qp(
            Q=Q_extended,
            c=q_extended,
            A_ineq=G_all,
            b_ineq=h_all,
            A_eq=None,
            b_eq=None,
            max_iter=safety_params.qp_solver_max_iter,
            tol=safety_params.qp_tolerance
        )
        
        if solution.converged:
            u_safe = solution.x[:3]
            relaxation = solution.x[3]
        else:
            # 回退策略
            u_safe = apply_fallback_strategy(u_nom, h, safety_params)
            relaxation = jnp.maximum(0.0, -h)  # 估计松弛量
            
    except Exception:
        u_safe = apply_fallback_strategy(u_nom, h, safety_params)
        relaxation = jnp.maximum(0.0, -h)
    
    return u_safe, relaxation


# JIT编译版本
safety_filter_jit = jax.jit(safety_filter)
safety_filter_with_relaxation_jit = jax.jit(safety_filter_with_relaxation)


def test_safety_filter():
    """测试安全滤波器功能"""
    
    print("测试安全滤波器...")
    
    # 测试场景1：安全情况（h > 0）
    u_nom = jnp.array([2.0, 1.0, 8.0])
    h_safe = 1.5  # 安全的CBF值
    grad_h = jnp.array([0.1, 0.2, -0.5])  # CBF梯度
    velocity = jnp.array([1.0, 0.5, 0.0])
    
    u_safe_1 = safety_filter(u_nom, h_safe, grad_h, velocity)
    print(f"安全场景 - 名义控制: {u_nom}")
    print(f"安全场景 - 安全控制: {u_safe_1}")
    print(f"安全场景 - 控制变化: {jnp.linalg.norm(u_safe_1 - u_nom):.4f}")
    
    # 测试场景2：不安全情况（h < 0）
    h_unsafe = -0.5  # 不安全的CBF值
    u_safe_2 = safety_filter(u_nom, h_unsafe, grad_h, velocity)
    print(f"\n不安全场景 - 名义控制: {u_nom}")
    print(f"不安全场景 - 安全控制: {u_safe_2}")
    print(f"不安全场景 - 控制变化: {jnp.linalg.norm(u_safe_2 - u_nom):.4f}")
    
    # 测试梯度计算
    def test_grad_computation():
        def loss_fn(u_nom_test):
            return jnp.sum(safety_filter(u_nom_test, h_safe, grad_h, velocity)**2)
        
        grad_fn = jax.grad(loss_fn)
        grad_result = grad_fn(u_nom)
        
        print(f"\n梯度计算测试:")
        print(f"输入: {u_nom}")
        print(f"梯度: {grad_result}")
        print(f"梯度范数: {jnp.linalg.norm(grad_result):.6f}")
        
        # 验证梯度不是NaN或零
        assert not jnp.any(jnp.isnan(grad_result)), "梯度包含NaN"
        assert jnp.linalg.norm(grad_result) > 1e-8, "梯度为零"
        
        print("✅ 梯度计算正常")
    
    test_grad_computation()
    
    # 测试带松弛的版本
    print(f"\n测试带松弛变量的安全滤波器:")
    u_safe_relax, relaxation = safety_filter_with_relaxation(u_nom, h_unsafe, grad_h, velocity)
    print(f"安全控制: {u_safe_relax}")
    print(f"松弛变量: {relaxation:.4f}")
    
    print("\n✅ 安全滤波器测试完成")
    
    return u_safe_1, u_safe_2


if __name__ == "__main__":
    test_safety_filter()