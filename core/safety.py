"""
core.safety

跨平台可微 CBF 安全层实现：优先使用解析主动集求解，必要时回退到 OSQP，
若依旧不可行则触发紧急制动。所有分支统一封装自定义 VJP，确保梯度只回传给
名义控制输入。
"""

from __future__ import annotations

import dataclasses
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
try:
    import osqp  # type: ignore
    import scipy.sparse as sp  # type: ignore
    _OSQP_IMPORT_ERROR = None
except ImportError as exc:
    osqp = None  # type: ignore
    sp = None  # type: ignore
    _OSQP_IMPORT_ERROR = exc

from core.flax_compat import struct

from .physics import DroneState


# ---------------------------------------------------------------------------
# 配置与诊断数据结构
# ---------------------------------------------------------------------------


@struct.dataclass
class SafetyConfig:
    """
    控制 CBF-QP 安全滤波器的参数。
    """

    alpha0: float = 1.0
    alpha1: float = 2.0
    max_acceleration: float = 5.0
    relaxation_penalty: float = 100.0
    max_relaxation: float = 2.0
    tolerance: float = 1e-5
    relaxation_alert: float = 0.5


@struct.dataclass
class SafetyDiagnostics:
    constraint_violation: jnp.ndarray
    relaxation: jnp.ndarray
    used_relaxation: jnp.ndarray
    qp_failed: jnp.ndarray
    nan_detected: jnp.ndarray
    relaxation_limit_exceeded: jnp.ndarray
    qp_iterations: jnp.ndarray
    qp_status: jnp.ndarray
    solver_mode: jnp.ndarray
    solver_status: jnp.ndarray


# ---------------------------------------------------------------------------
# 求解器辅助函数
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
    为变量 ``[u, δ]`` 构造 (Q, q, G, h)。
    """
    velocity = state.velocity

    lf_h = jnp.dot(grad_p, velocity)
    psi1 = lf_h + config.alpha0 * h_value
    partial_psi1_pos = hess_p @ velocity + config.alpha0 * grad_p
    lf_psi1 = jnp.dot(partial_psi1_pos, velocity)
    lg_psi1 = grad_p
    rhs = lf_psi1 + config.alpha1 * psi1

    accel_weight = jnp.ones(3)
    H_diag = jnp.concatenate([accel_weight, jnp.array([config.relaxation_penalty])])
    Q = jnp.diag(H_diag)
    q = jnp.concatenate([-accel_weight * u_nom, jnp.array([0.0])])

    cbf_row = jnp.concatenate([-lg_psi1, jnp.array([-1.0])])
    cbf_rhs = rhs

    eye = jnp.eye(3)
    upper = jnp.concatenate([eye, jnp.zeros((3, 1))], axis=1)
    lower = jnp.concatenate([-eye, jnp.zeros((3, 1))], axis=1)
    accel_bound = jnp.full((3,), config.max_acceleration)

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


def _numpy(array: jnp.ndarray) -> np.ndarray:
    return np.asarray(jax.device_get(array), dtype=np.float64)


def _analytic_attempt(
    u_nom: np.ndarray,
    G: np.ndarray,
    h_vec: np.ndarray,
    config: SafetyConfig,
) -> Tuple[bool, np.ndarray]:
    max_acc = config.max_acceleration
    u_clipped = np.clip(u_nom, -max_acc, max_acc)
    cbf_row = G[0]
    lg = -cbf_row[:3]
    rhs = h_vec[0]
    delta_req = max(0.0, -np.dot(lg, u_clipped) - rhs)
    if delta_req <= config.max_relaxation + 1e-9:
        delta = min(delta_req, config.max_relaxation)
        return True, np.concatenate([u_clipped, np.array([delta])])

    # 再次尝试：沿 lg 方向投影，使约束在 δ = max_relaxation 内可行
    target_rhs = rhs + config.max_relaxation
    lg_norm_sq = float(np.dot(lg, lg)) + 1e-9
    shift = (-np.dot(lg, u_clipped) - target_rhs) / lg_norm_sq
    u_proj = u_clipped + shift * lg
    u_proj = np.clip(u_proj, -max_acc, max_acc)
    delta_proj = max(0.0, -np.dot(lg, u_proj) - rhs)
    if delta_proj <= config.max_relaxation + 1e-9:
        delta = min(delta_proj, config.max_relaxation)
        return True, np.concatenate([u_proj, np.array([delta])])

    return False, np.concatenate([u_clipped, np.array([config.max_relaxation])])


def _osqp_solve(
    u_nom: np.ndarray,
    Q: np.ndarray,
    q: np.ndarray,
    G: np.ndarray,
    h_vec: np.ndarray,
    config: SafetyConfig,
) -> Tuple[bool, np.ndarray, int, int]:
    if osqp is None or sp is None:  # pragma: no cover - 仅在未安装 OSQP 时触发
        return False, np.concatenate([np.clip(u_nom, -config.max_acceleration, config.max_acceleration), np.array([config.max_relaxation], dtype=np.float64)]), 0, -99

    rho = config.relaxation_penalty
    P = sp.diags([1.0, 1.0, 1.0, rho], format="csc")
    q_vec = np.concatenate([-u_nom, np.array([0.0])])

    rows = []
    l = []
    u = []

    # CBF 约束
    rows.append(G[0])
    l.append(-np.inf)
    u.append(h_vec[0])

    # 控制盒约束
    for i in range(3):
        row = np.zeros(4, dtype=np.float64)
        row[i] = 1.0
        rows.append(row)
        l.append(-config.max_acceleration)
        u.append(config.max_acceleration)

    # 松弛变量约束
    row_delta = np.zeros(4, dtype=np.float64)
    row_delta[3] = 1.0
    rows.append(row_delta)
    l.append(0.0)
    u.append(config.max_relaxation)

    A = sp.csc_matrix(np.vstack(rows))
    l_vec = np.asarray(l, dtype=np.float64)
    u_vec = np.asarray(u, dtype=np.float64)

    solver = osqp.OSQP()
    solver.setup(
        P=P,
        q=q_vec,
        A=A,
        l=l_vec,
        u=u_vec,
        eps_abs=1e-7,
        eps_rel=1e-7,
        warm_start=False,
        verbose=False,
    )
    res = solver.solve()
    status_val = int(res.info.status_val)
    iterations = int(res.info.iter)
    success = status_val in (1, 2) and res.x is not None

    if not success:
        return False, np.concatenate([np.clip(u_nom, -config.max_acceleration, config.max_acceleration), np.array([config.max_relaxation], dtype=np.float64)]), iterations, status_val

    primal = np.asarray(res.x, dtype=np.float64)
    primal[:3] = np.clip(primal[:3], -config.max_acceleration, config.max_acceleration)
    primal[3] = np.clip(primal[3], 0.0, config.max_relaxation)
    if not np.all(np.isfinite(primal)):
        return False, np.concatenate([np.clip(u_nom, -config.max_acceleration, config.max_acceleration), np.array([config.max_relaxation], dtype=np.float64)]), iterations, status_val

    return True, primal, iterations, status_val


def _emergency_control(state: DroneState, config: SafetyConfig) -> np.ndarray:
    velocity = _numpy(state.velocity)
    u_emergency = np.zeros(3, dtype=np.float64)
    if np.linalg.norm(velocity[:2]) > 1e-6:
        u_emergency[:2] = -velocity[:2] / (np.linalg.norm(velocity[:2]) + 1e-9) * config.max_acceleration
    u_emergency[2] = config.max_acceleration
    return np.concatenate([np.clip(u_emergency, -config.max_acceleration, config.max_acceleration), np.array([config.max_relaxation], dtype=np.float64)])


def _solve_safe_qp(
    u_nom: jnp.ndarray,
    h_value: jnp.ndarray,
    grad_position: jnp.ndarray,
    hessian_position: jnp.ndarray,
    state: DroneState,
    config: SafetyConfig,
) -> Tuple[jnp.ndarray, SafetyDiagnostics]:
    Q, q, G, h_vec = _assemble_qp(u_nom, grad_position, hessian_position, h_value, state, config)
    u_nom_np = _numpy(u_nom)
    G_np = _numpy(G)
    h_np = _numpy(h_vec)

    analytic_success, primal = _analytic_attempt(u_nom_np, G_np, h_np, config)
    solver_mode = 0 if analytic_success else 1
    iterations = 0
    status_val = 0

    if not analytic_success:
        osqp_success, primal, iterations, status_val = _osqp_solve(u_nom_np, _numpy(Q), _numpy(q), G_np, h_np, config)
        solver_mode = 1 if osqp_success else 2

        if not osqp_success:
            primal = _emergency_control(state, config)

    u_safe_np = primal[:3]
    relaxation_np = primal[3]

    G_primal = G_np @ primal
    residual = float(np.maximum(np.max(G_primal - h_np), 0.0))
    nan_inputs = (
        bool(np.isnan(_numpy(h_value)))
        or bool(np.any(np.isnan(_numpy(grad_position))))
        or bool(np.any(np.isnan(_numpy(hessian_position))))
    )

    qp_failed = 1.0 if solver_mode == 2 else 0.0
    nan_detected = 1.0 if nan_inputs else 0.0
    relax_limit = 1.0 if relaxation_np > config.relaxation_alert else 0.0
    used_relax = 1.0 if relaxation_np > 1e-6 else 0.0

    u_safe = u_nom + jax.lax.stop_gradient(jnp.asarray(u_safe_np, dtype=jnp.float32) - u_nom)
    relaxation = jax.lax.stop_gradient(jnp.asarray(relaxation_np, dtype=jnp.float32))

    diagnostics = SafetyDiagnostics(
        constraint_violation=jnp.asarray(residual, dtype=jnp.float32),
        relaxation=relaxation,
        used_relaxation=jnp.asarray(used_relax, dtype=jnp.float32),
        qp_failed=jnp.asarray(qp_failed, dtype=jnp.float32),
        nan_detected=jnp.asarray(nan_detected, dtype=jnp.float32),
        relaxation_limit_exceeded=jnp.asarray(relax_limit, dtype=jnp.float32),
        qp_iterations=jnp.asarray(iterations, dtype=jnp.float32),
        qp_status=jnp.asarray(status_val, dtype=jnp.float32),
        solver_mode=jnp.asarray(solver_mode, dtype=jnp.float32),
        solver_status=jnp.asarray(status_val, dtype=jnp.float32),
    )
    return u_safe, diagnostics


# ---------------------------------------------------------------------------
# 自定义 VJP 封装
# ---------------------------------------------------------------------------


def _config_tuple(config: SafetyConfig) -> Tuple[float, float, float, float, float, float]:
    return (
        float(config.alpha0),
        float(config.alpha1),
        float(config.max_acceleration),
        float(config.relaxation_penalty),
        float(config.max_relaxation),
        float(config.relaxation_alert),
    )


def _config_from_tuple(cfg: Tuple[float, float, float, float, float, float], tolerance: float) -> SafetyConfig:
    alpha0, alpha1, max_acc, relax_penalty, max_relax, relax_alert = cfg
    return SafetyConfig(
        alpha0=alpha0,
        alpha1=alpha1,
        max_acceleration=max_acc,
        relaxation_penalty=relax_penalty,
        max_relaxation=max_relax,
        tolerance=tolerance,
        relaxation_alert=relax_alert,
    )


@jax.custom_vjp
def _solve_safe_control(
    u_nom: jnp.ndarray,
    h_value: jnp.ndarray,
    grad_position: jnp.ndarray,
    hessian_position: jnp.ndarray,
    state: DroneState,
    config_tuple: Tuple[float, float, float, float, float, float],
    tolerance: float,
) -> Tuple[jnp.ndarray, SafetyDiagnostics]:
    config = _config_from_tuple(config_tuple, tolerance)
    return _solve_safe_qp(u_nom, h_value, grad_position, hessian_position, state, config)


def _solve_safe_control_fwd(
    u_nom,
    h_value,
    grad_position,
    hessian_position,
    state,
    config_tuple,
    tolerance,
):
    outputs = _solve_safe_control(u_nom, h_value, grad_position, hessian_position, state, config_tuple, tolerance)
    saved = (u_nom, h_value, grad_position, hessian_position, state, config_tuple, tolerance)
    return outputs, saved


def _solve_safe_control_bwd(saved, grad_outputs):
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
    zeros_cfg = tuple(jnp.array(0.0, dtype=jnp.asarray(tolerance).dtype) for _ in config_tuple)
    zeros_tol = jnp.array(0.0, dtype=jnp.asarray(tolerance).dtype)
    return (
        grad_u,
        zeros_h,
        zeros_grad,
        zeros_hess,
        zeros_state,
        zeros_cfg,
        zeros_tol,
    )


_solve_safe_control.defvjp(_solve_safe_control_fwd, _solve_safe_control_bwd)


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
    应用可微 CBF 安全滤波器。
    """
    return _solve_safe_control(
        u_nom,
        h_value,
        grad_position,
        hessian_position,
        state,
        _config_tuple(config),
        float(config.tolerance),
    )


__all__ = [
    "SafetyConfig",
    "SafetyDiagnostics",
    "safety_filter",
]
