"""
core.safety
============

Control-barrier-function (CBF) safety layer implemented as a differentiable
quadratic programme.  The formulation mirrors the double-integrator CBF used in
GCBF+: the decision variables are the acceleration command ``u`` and a
non-negative relaxation slack ``δ`` that guarantees feasibility.  Solving the
QP with ``qpax`` keeps the layer fully differentiable.
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

    # Preserve gradients for the policy while stopping gradients from the QP
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
    del grad_diag  # diagnostics are stop_gradient'ed

    grad_u_nom = grad_u
    # No gradients w.r.t. CBF inputs
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

    # Preserve gradients for the policy while stopping gradients from the QP
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

try:  # pragma: no cover - optional dependency resolved at runtime
    import qpax  # type: ignore

    HAS_QPAX = True
except ImportError:  # pragma: no cover
    qpax = None
    HAS_QPAX = False


# ---------------------------------------------------------------------------
# Configuration and diagnostics
# ---------------------------------------------------------------------------


@struct.dataclass
class SafetyConfig:
    """
    Parameters governing the CBF-QP safety filter.

    Attributes
    ----------
    alpha0, alpha1:
        Class-K gains for the second-order (relative-degree-two) CBF condition.
    max_acceleration:
        Box constraint for the acceleration command.
    relaxation_penalty:
        Quadratic weight applied to the slack variable δ.
    max_relaxation:
        Upper bound on δ to avoid runaway solutions.
    tolerance:
        Acceptable maximum constraint violation (diagnostics only).
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
# Helper functions
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
    Construct (Q, q, G, h) for the optimisation variable ``[u, δ]``.
    """
    velocity = state.velocity

    # CBF terms for a double-integrator model
    lf_h = jnp.dot(grad_p, velocity)
    psi1 = lf_h + config.alpha0 * h_value

    # Hessian-based term: ∂ψ1/∂p = H_h * v + α0 * grad_p
    partial_psi1_pos = hess_p @ velocity + config.alpha0 * grad_p
    lf_psi1 = jnp.dot(partial_psi1_pos, velocity)
    lg_psi1 = grad_p  # derivative wrt velocity maps directly to control input

    rhs = lf_psi1 + config.alpha1 * psi1

    # Cost: 0.5 (u - u_nom)^T (I) (u - u_nom) + 0.5 ρ δ^2
    accel_weight = jnp.ones(3)
    H_diag = jnp.concatenate([accel_weight, jnp.array([config.relaxation_penalty])])
    Q = jnp.diag(H_diag)
    q = jnp.concatenate([-accel_weight * u_nom, jnp.array([0.0])])

    # CBF constraint: -Lg ψ1 · u - δ <= rhs
    cbf_row = jnp.concatenate([-lg_psi1, jnp.array([-1.0])])
    cbf_rhs = rhs

    # Box constraints on u
    eye = jnp.eye(3)
    upper = jnp.concatenate([eye, jnp.zeros((3, 1))], axis=1)
    lower = jnp.concatenate([-eye, jnp.zeros((3, 1))], axis=1)
    accel_bound = jnp.full((3,), config.max_acceleration)

    # Slack bounds: δ >= 0, δ <= max_relaxation
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
    """
    Solve the QP and return the primal vector ``[u, δ]``.
    """
    if not HAS_QPAX:  # pragma: no cover - qpax is expected in production
        raise RuntimeError("qpax is required for the differentiable safety layer.")

    sol = qpax.solve_qp_primal(
        Q,
        q,
        jnp.zeros((0, Q.shape[0])),  # no equality constraints
        jnp.zeros((0,)),
        G,
        h_vec,
        solver_tol=1e-6,
        target_kappa=1e-3,
    )
    return sol


# ---------------------------------------------------------------------------
# Public API
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
    Apply the differentiable CBF-QP filter.

    Parameters
    ----------
    u_nom:
        Nominal acceleration command from the policy.
    h_value, grad_position, hessian_position:
        CBF value and its first/second spatial derivatives.
    state:
        Current drone state.

    Returns
    -------
    u_safe:
        Filtered acceleration command.
    diagnostics:
        Constraint violation magnitude and slack usage (for logging only).
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
