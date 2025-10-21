"""
core.safety

以可微分二次规划实现的 CBF 安全层。整体结构沿用 GCBF+ 中的双积分器 CBF，决策变量包括加速度指令 ``u`` 以及用于保证可行性的非负松弛量 ``δ``。借助``qpax`` 求解可保持该层的端到端可微。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp
from core.flax_compat import struct

from .physics import DroneState


def _config_tuple(config: SafetyConfig) -> Tuple[float, float, float, float, float]:
    return (
        float(config.alpha0),
        float(config.alpha1),
        float(config.max_acceleration),
        float(config.relaxation_penalty),
        float(config.max_relaxation),
    )


def _config_from_tuple(cfg: Tuple[float, float, float, float, float], tolerance: float) -> SafetyConfig:
    alpha0, alpha1, max_acc, relax_penalty, max_relax = cfg
    return SafetyConfig(
        alpha0=alpha0,
        alpha1=alpha1,
        max_acceleration=max_acc,
        relaxation_penalty=relax_penalty,
        max_relaxation=max_relax,
        tolerance=tolerance,
    )


def _qp_forward(
    u_nom: jnp.ndarray,
    cbf_inputs: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, DroneState, Tuple[float, float, float, float, float], float],
) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, DroneState, Tuple[float, float, float, float, float], float]]]:
    h_value, grad_p, hess_p, state, config_tuple, tolerance = cbf_inputs
    config = _config_from_tuple(config_tuple, tolerance)
    Q, q, G, h_vec = _assemble_qp(
        u_nom,
        grad_p,
        hess_p,
        h_value,
        state,
        config,
    )
    primal_raw = _solve_qp(Q, q, G, h_vec)
    finite = jnp.all(jnp.isfinite(primal_raw))
    primal_clean = jnp.nan_to_num(
        primal_raw,
        nan=0.0,
        posinf=config.max_acceleration,
        neginf=-config.max_acceleration,
    )

    def _fallback(_):
        return jnp.concatenate([u_nom, jnp.array([0.0])])

    primal = jax.lax.cond(
        finite,
        lambda _: primal_clean,
        _fallback,
        operand=None,
    )
    u_safe_raw = primal[:3]
    relaxation = primal[3]

    # 为策略保留梯度，同时阻断来自 QP 的梯度
    u_safe = u_nom + jax.lax.stop_gradient(u_safe_raw - u_nom)
    relaxation = jax.lax.stop_gradient(relaxation)
    residual = jnp.maximum(jnp.max(G @ primal - h_vec), 0.0)
    diagnostics = SafetyDiagnostics(
        constraint_violation=jax.lax.stop_gradient(residual),
        relaxation=relaxation,
        used_relaxation=jax.lax.stop_gradient(jnp.array(relaxation > 1e-6, dtype=jnp.float32)),
    )
    return (u_safe, diagnostics), (u_nom, cbf_inputs)


def _qp_backward(
    cbf_inputs: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, DroneState, SafetyConfig],
    res: Tuple[jnp.ndarray, Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, DroneState, SafetyConfig]]],
    grad_output: Tuple[jnp.ndarray, SafetyDiagnostics],
):
    u_safe, _ = res
    grad_u, grad_diag = grad_output
    del grad_diag  # 诊断信息已通过 stop_gradient 阻断

    grad_u_nom = grad_u
    # 对 CBF 输入不回传梯度
    zeros_grad = jax.tree_util.tree_map(jnp.zeros_like, cbf_inputs)
    return grad_u_nom, zeros_grad


def _qp_raw(
    u_nom: jnp.ndarray,
    h_value: jnp.ndarray,
    grad_position: jnp.ndarray,
    hessian_position: jnp.ndarray,
    state: DroneState,
    config_tuple: Tuple[float, float, float, float, float],
    tolerance: float,
):
    config = _config_from_tuple(config_tuple, tolerance)
    Q, q, G, h_vec = _assemble_qp(
        u_nom,
        grad_position,
        hessian_position,
        h_value,
        state,
        config,
    )
    primal_raw = _solve_qp(Q, q, G, h_vec)
    finite = jnp.all(jnp.isfinite(primal_raw))
    primal_clean = jnp.nan_to_num(
        primal_raw,
        nan=0.0,
        posinf=config.max_acceleration,
        neginf=-config.max_acceleration,
    )

    def _fallback(_):
        return jnp.concatenate([u_nom, jnp.array([0.0])])

    primal = jax.lax.cond(
        finite,
        lambda _: primal_clean,
        _fallback,
        operand=None,
    )
    u_safe_raw = primal[:3]
    relaxation = primal[3]

    # 为策略保留梯度，同时阻断来自 QP 的梯度
    u_safe = u_nom + jax.lax.stop_gradient(u_safe_raw - u_nom)
    relaxation = jax.lax.stop_gradient(relaxation)
    residual = jnp.maximum(jnp.max(G @ primal - h_vec), 0.0)
    diagnostics = SafetyDiagnostics(
        constraint_violation=jax.lax.stop_gradient(residual),
        relaxation=relaxation,
        used_relaxation=jax.lax.stop_gradient(jnp.array(relaxation > 1e-6, dtype=jnp.float32)),
    )
    return u_safe, diagnostics


@jax.custom_vjp
def _qp_custom_vjp(
    u_nom: jnp.ndarray,
    h_value: jnp.ndarray,
    grad_position: jnp.ndarray,
    hessian_position: jnp.ndarray,
    state: DroneState,
    config_tuple: Tuple[float, float, float, float, float],
    tolerance: float,
):
    return _qp_raw(u_nom, h_value, grad_position, hessian_position, state, config_tuple, tolerance)


def _qp_custom_vjp_fwd(
    u_nom,
    h_value,
    grad_position,
    hessian_position,
    state,
    config_tuple,
    tolerance,
):
    outputs = _qp_raw(u_nom, h_value, grad_position, hessian_position, state, config_tuple, tolerance)
    saved = (
        u_nom,
        h_value,
        grad_position,
        hessian_position,
        state,
        config_tuple,
        tolerance,
    )
    return outputs, saved


def _qp_custom_vjp_bwd(saved, grad_outputs):
    (
        u_nom,
        h_value,
        grad_position,
        hessian_position,
        state,
        config_tuple,
        tolerance,
    ) = saved
    grad_u, _ = grad_outputs
    zeros_h = jnp.zeros_like(h_value)
    zeros_grad = jnp.zeros_like(grad_position)
    zeros_hess = jnp.zeros_like(hessian_position)
    zeros_state = jax.tree_util.tree_map(jnp.zeros_like, state)
    zero_scalar = jnp.array(0.0, dtype=jnp.asarray(tolerance).dtype)
    zeros_cfg = tuple(jnp.array(0.0, dtype=jnp.asarray(tolerance).dtype) for _ in config_tuple)
    zeros_tol = zero_scalar
    return (
        grad_u,
        zeros_h,
        zeros_grad,
        zeros_hess,
        zeros_state,
        zeros_cfg,
        zeros_tol,
    )


_qp_custom_vjp.defvjp(_qp_custom_vjp_fwd, _qp_custom_vjp_bwd)

try:  # pragma: no cover - 可选依赖在运行时加载
    import qpax  # type: ignore

    HAS_QPAX = True
except ImportError:  # pragma: no cover
    qpax = None
    HAS_QPAX = False


# ---------------------------------------------------------------------------
# 配置与诊断结构体
# ---------------------------------------------------------------------------


@struct.dataclass
class SafetyConfig:
    """
    控制 CBF-QP 安全滤波器的参数。

    属性说明
    ----------
    alpha0, alpha1:
        二阶（相对阶数为 2）CBF 条件的 Class-K 增益。
    max_acceleration:
        加速度指令的盒约束上界。
    relaxation_penalty:
        作用于松弛变量 δ 的二次惩罚权重。
    max_relaxation:
        δ 的上界，用于避免解发散。
    tolerance:
        可接受的最大约束违背量（仅用于诊断）。
    """

    alpha0: float = 1.0
    alpha1: float = 2.0
    max_acceleration: float = 5.0
    relaxation_penalty: float = 100.0
    max_relaxation: float = 2.0
    tolerance: float = 1e-5


@struct.dataclass
class SafetyDiagnostics:
    constraint_violation: jnp.ndarray
    relaxation: jnp.ndarray
    used_relaxation: jnp.ndarray


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _assemble_qp(
    u_nom: jnp.ndarray,
    grad_p: jnp.ndarray,
    hess_p: jnp.ndarray,
    h_value: jnp.ndarray,
    state: DroneState,
    config: SafetyConfig,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    为优化变量 ``[u, δ]`` 构造 (Q, q, G, h)。
    """
    velocity = state.velocity

    # 双积分器模型对应的 CBF 项
    lf_h = jnp.dot(grad_p, velocity)
    psi1 = lf_h + config.alpha0 * h_value

    # Hessian 项：∂ψ1/∂p = H_h * v + α0 * grad_p
    partial_psi1_pos = hess_p @ velocity + config.alpha0 * grad_p
    lf_psi1 = jnp.dot(partial_psi1_pos, velocity)
    lg_psi1 = grad_p  # 对速度求导后可直接映射为控制输入

    rhs = lf_psi1 + config.alpha1 * psi1

    # 代价项：0.5 (u - u_nom)^T I (u - u_nom) + 0.5 ρ δ^2
    accel_weight = jnp.ones(3)
    H_diag = jnp.concatenate([accel_weight, jnp.array([config.relaxation_penalty])])
    Q = jnp.diag(H_diag)
    q = jnp.concatenate([-accel_weight * u_nom, jnp.array([0.0])])

    # CBF 约束：-Lg ψ1 · u - δ <= rhs
    cbf_row = jnp.concatenate([-lg_psi1, jnp.array([-1.0])])
    cbf_rhs = rhs

    # 对 u 施加盒约束
    eye = jnp.eye(3)
    upper = jnp.concatenate([eye, jnp.zeros((3, 1))], axis=1)
    lower = jnp.concatenate([-eye, jnp.zeros((3, 1))], axis=1)
    accel_bound = jnp.full((3,), config.max_acceleration)

    # 松弛变量范围：δ >= 0，δ <= max_relaxation
    slack_lb = jnp.array([[0.0, 0.0, 0.0, -1.0]])
    slack_lb_rhs = jnp.array([0.0])
    slack_ub = jnp.array([[0.0, 0.0, 0.0, 1.0]])
    slack_ub_rhs = jnp.array([config.max_relaxation])

    G = jnp.concatenate(
        [cbf_row[None, :], upper, lower, slack_lb, slack_ub],
        axis=0,
    )
    h_vec = jnp.concatenate(
        [
            jnp.array([cbf_rhs]),
            accel_bound,
            accel_bound,
            slack_lb_rhs,
            slack_ub_rhs,
        ],
        axis=0,
    )

    return Q, q, G, h_vec


def _solve_qp(
    Q: jnp.ndarray,
    q: jnp.ndarray,
    G: jnp.ndarray,
    h_vec: jnp.ndarray,
) -> jnp.ndarray:
    """求解该 QP 并返回原始变量 ``[u, δ]``。"""
    if not HAS_QPAX:  # pragma: no cover - 生产部署默认会提供 qpax
        raise RuntimeError("可微安全层需要安装 qpax。")

    sol = qpax.solve_qp_primal(
        Q,
        q,
        jnp.zeros((0, Q.shape[0])),  # 无等式约束
        jnp.zeros((0,)),
        G,
        h_vec,
        solver_tol=1e-6,
        target_kappa=1e-3,
    )
    return sol


# ---------------------------------------------------------------------------
# 公共接口
# ---------------------------------------------------------------------------


def safety_filter(
    u_nom: jnp.ndarray,
    h_value: jnp.ndarray,
    grad_position: jnp.ndarray,
    hessian_position: jnp.ndarray,
    state: DroneState,
    config: SafetyConfig,
) -> Tuple[jnp.ndarray, SafetyDiagnostics]:
    """
    应用可微的 CBF-QP 安全滤波器。

    参数
    ----------
    u_nom:
        策略输出的名义加速度指令。
    h_value, grad_position, hessian_position:
        CBF 值及其一阶、二阶空间导数。
    state:
        当前无人机状态。

    返回值
    -------
    u_safe:
        经过滤波后的加速度指令。
    diagnostics:
        约束违背幅度与松弛使用情况（仅用于日志记录）。
    """
    u_safe, diagnostics = _qp_custom_vjp(
        u_nom,
        h_value,
        grad_position,
        hessian_position,
        state,
        _config_tuple(config),
        float(config.tolerance),
    )
    return u_safe, diagnostics


__all__ = [
    "SafetyConfig",
    "SafetyDiagnostics",
    "safety_filter",
]
