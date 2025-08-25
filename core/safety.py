"""
安全模块：基于qpax的可微分安全层实现 - 完全修复JAX兼容性
实现GCBF+的QP安全约束
"""

import jax
import jax.numpy as jnp
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
    构建CBF-QP的矩阵形式 - JAX兼容版本
    
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


def apply_fallback_strategy(u_nom: chex.Array,
                          h: float,
                          safety_params: SafetyParams) -> chex.Array:
    """
    回退策略：当QP求解失败时使用 - 完全JAX兼容版本
    
    Args:
        u_nom: 名义控制
        h: CBF值
        safety_params: 安全参数
        
    Returns:
        u_fallback: 回退控制输入
    """
    
    # 使用jnp.where替代if语句（JAX兼容）
    # 如果CBF值为负（不安全），采用保守控制；否则使用限幅的名义控制
    conservative_control = u_nom * 0.1  # 紧急制动，保留方向但大幅减小幅度
    clipped_control = jnp.clip(u_nom, -safety_params.max_control_norm, safety_params.max_control_norm)
    
    u_fallback = jnp.where(
        h < 0,
        conservative_control,
        clipped_control
    )
    
    return u_fallback


def safety_filter(u_nom: chex.Array,
                 h: float,
                 grad_h: chex.Array,
                 drone_velocity: chex.Array,
                 safety_params: SafetyParams = None) -> chex.Array:
    """
    安全滤波器：解决CBF-QP优化问题 - 完全JAX兼容版本
    
    Args:
        u_nom: 名义控制输入 [3]
        h: CBF值
        grad_h: CBF梯度 [3]
        drone_velocity: 当前速度 [3]
        safety_params: 安全参数
        
    Returns:
        u_safe: 安全的控制输入 [3]
    """
    
    # 处理safety_params为None的情况
    if safety_params is None:
        safety_params = SafetyParams()
    
    # 构建QP矩阵
    Q, q, G, h_constraint = construct_cbf_qp_matrices(
        u_nom, h, grad_h, drone_velocity, safety_params
    )
    
    # 简化版本：不使用qpax，直接使用解析解或近似解
    # 对于简单的CBF-QP，我们可以用投影方法
    u_safe = safety_filter_projection_method(
        u_nom, h, grad_h, drone_velocity, safety_params
    )
    
    return u_safe


def safety_filter_projection_method(u_nom: chex.Array,
                                   h: float,
                                   grad_h: chex.Array,
                                   drone_velocity: chex.Array,
                                   safety_params: SafetyParams) -> chex.Array:
    """
    基于投影的安全滤波器 - 避免使用qpax的JAX兼容版本
    
    Args:
        u_nom: 名义控制输入 [3]
        h: CBF值
        grad_h: CBF梯度 [3]
        drone_velocity: 当前速度 [3]
        safety_params: 安全参数
        
    Returns:
        u_safe: 安全的控制输入 [3]
    """
    
    # 首先限制控制输入在界限内
    u_clipped = jnp.clip(u_nom, -safety_params.max_control_norm, safety_params.max_control_norm)
    
    # 检查CBF约束是否违反
    h_dot_current = jnp.dot(grad_h, drone_velocity)
    h_dot_with_u_nom = h_dot_current + jnp.dot(grad_h, u_clipped)
    cbf_constraint = h_dot_with_u_nom + safety_params.alpha * h
    
    # 如果约束满足，直接返回；否则投影到约束边界
    constraint_satisfied = cbf_constraint >= -safety_params.safety_margin
    
    # 投影方法：如果违反约束，修正控制输入
    # 计算需要的最小修正
    grad_norm_sq = jnp.dot(grad_h, grad_h)
    
    # 避免除零
    grad_norm_sq_safe = jnp.maximum(grad_norm_sq, 1e-8)
    
    # 计算修正量
    correction_magnitude = jnp.maximum(
        0.0,
        (-cbf_constraint - safety_params.safety_margin) / grad_norm_sq_safe
    )
    
    # 修正方向：沿梯度方向
    u_correction = correction_magnitude * grad_h
    u_corrected = u_clipped + u_correction
    
    # 再次限制在控制界限内
    u_final = jnp.clip(u_corrected, -safety_params.max_control_norm, safety_params.max_control_norm)
    
    # 使用jnp.where选择最终控制
    u_safe = jnp.where(
        constraint_satisfied,
        u_clipped,  # 约束满足，使用原始控制
        u_final     # 约束违反，使用修正控制
    )
    
    return u_safe


def safety_filter_with_relaxation(u_nom: chex.Array,
                                 h: float,
                                 grad_h: chex.Array,
                                 drone_velocity: chex.Array,
                                 relaxation_penalty: float,
                                 safety_params: SafetyParams) -> Tuple[chex.Array, float]:
    """
    带松弛变量的安全滤波器 - JAX兼容版本
    用于处理不可行的QP问题
    
    Returns:
        (u_safe, relaxation): 安全控制和松弛变量值
    """
    
    # 先尝试普通安全滤波
    u_safe = safety_filter(u_nom, h, grad_h, drone_velocity, safety_params)
    
    # 检查是否需要松弛
    h_dot_current = jnp.dot(grad_h, drone_velocity)
    h_dot_with_u_safe = h_dot_current + jnp.dot(grad_h, u_safe)
    cbf_constraint = h_dot_with_u_safe + safety_params.alpha * h
    
    # 估计松弛量
    relaxation = jnp.maximum(0.0, -cbf_constraint - safety_params.safety_margin)
    
    # 如果需要松弛，应用更保守的控制
    conservative_u_safe = jnp.where(
        relaxation > 1e-6,
        apply_fallback_strategy(u_nom, h, safety_params),
        u_safe
    )
    
    return conservative_u_safe, relaxation


# JIT编译版本
def create_safety_filter_jit(safety_params: SafetyParams):
    """创建JIT编译的安全滤波器"""
    def safety_filter_with_params(u_nom: chex.Array,
                                  h: float,
                                  grad_h: chex.Array,
                                  drone_velocity: chex.Array) -> chex.Array:
        return safety_filter(u_nom, h, grad_h, drone_velocity, safety_params)
    
    return jax.jit(safety_filter_with_params)


def create_safety_filter_with_relaxation_jit(safety_params: SafetyParams, relaxation_penalty: float):
    """创建JIT编译的带松弛变量的安全滤波器"""
    def safety_filter_relax_with_params(u_nom: chex.Array,
                                        h: float,
                                        grad_h: chex.Array,
                                        drone_velocity: chex.Array) -> Tuple[chex.Array, float]:
        return safety_filter_with_relaxation(u_nom, h, grad_h, drone_velocity, relaxation_penalty, safety_params)
    
    return jax.jit(safety_filter_relax_with_params)


def test_safety_filter():
    """测试安全滤波器功能"""
    
    print("测试安全滤波器...")
    
    # 创建安全参数
    safety_params = SafetyParams()
    
    # 测试场景1：安全情况（h > 0）
    u_nom = jnp.array([2.0, 1.0, 8.0])
    h_safe = 1.5  # 安全的CBF值
    grad_h = jnp.array([0.1, 0.2, -0.5])  # CBF梯度
    velocity = jnp.array([1.0, 0.5, 0.0])
    
    u_safe_1 = safety_filter(u_nom, h_safe, grad_h, velocity, safety_params)
    print(f"安全场景 - 名义控制: {u_nom}")
    print(f"安全场景 - 安全控制: {u_safe_1}")
    print(f"安全场景 - 控制变化: {jnp.linalg.norm(u_safe_1 - u_nom):.4f}")
    
    # 测试场景2：不安全情况（h < 0）
    h_unsafe = -0.5  # 不安全的CBF值
    u_safe_2 = safety_filter(u_nom, h_unsafe, grad_h, velocity, safety_params)
    print(f"\n不安全场景 - 名义控制: {u_nom}")
    print(f"不安全场景 - 安全控制: {u_safe_2}")
    print(f"不安全场景 - 控制变化: {jnp.linalg.norm(u_safe_2 - u_nom):.4f}")
    
    # 测试梯度计算
    def test_grad_computation():
        def loss_fn(u_nom_test):
            return jnp.sum(safety_filter(u_nom_test, h_safe, grad_h, velocity, safety_params)**2)
        
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
    u_safe_relax, relaxation = safety_filter_with_relaxation(
        u_nom, h_unsafe, grad_h, velocity, 10.0, safety_params
    )
    print(f"安全控制: {u_safe_relax}")
    print(f"松弛变量: {relaxation:.4f}")
    
    print("\n✅ 安全滤波器测试完成")
    
    return u_safe_1, u_safe_2


if __name__ == "__main__":
    test_safety_filter()