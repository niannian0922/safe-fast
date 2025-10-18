"""
core.physics
================

Minimal differentiable quadrotor-style point-mass dynamics that stays faithful
to the methodological commitments in *GCBF+* and *DiffPhysDrone*:

- single-agent, fully differentiable, JAX-first implementation;
- controls act as accelerations in world coordinates and are box-constrained;
- discrete-time update matches the linearised model used for CBF-QP;
- optional temporal gradient decay to stabilise long-horizon BPTT.

The implementation intentionally omits multi-agent conveniences from the legacy
code base so that every function is a small, pure PyTree transformation that can
be dropped directly inside a `jax.lax.scan`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from core.flax_compat import struct

# ---------------------------------------------------------------------------
# State and parameter containers
# ---------------------------------------------------------------------------


@struct.dataclass
class DroneState:
    """Single-drone point-mass state compatible with JAX transformations."""

    position: jnp.ndarray  # (3,)
    velocity: jnp.ndarray  # (3,)
    acceleration: jnp.ndarray  # (3,) -- most recent commanded acceleration
    time: jnp.ndarray  # scalar simulation time

    # Orientation is optional but kept to ease later perception integration.
    orientation: jnp.ndarray = struct.field(
        default_factory=lambda: jnp.eye(3)
    )  # (3,3)


@dataclass(frozen=True)
class PhysicsParams:
    """Physical constants shared with the safety layer."""

    dt: float = 1.0 / 15.0  # control frequency (s)
    gravity: float = 9.80665  # m/s^2, only used for reference
    max_acceleration: float = 5.0  # |u|_inf bound (m/s^2)
    max_velocity: float = 10.0  # soft speed cap (m/s)
    workspace_radius: float = 30.0  # used for soft position clamp
    gradient_decay: float = 0.4  # DiffPhys-style temporal gradient decay


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _clip_vector_norm(vector: jnp.ndarray, limit: float) -> jnp.ndarray:
    """Clips a vector to the given Euclidean norm in a smooth fashion."""
    vector = jnp.asarray(vector)
    norm = jnp.linalg.norm(vector) + 1e-8
    scale = jnp.minimum(1.0, limit / norm)
    return vector * scale


def _project_box(u: jnp.ndarray, limit: float) -> jnp.ndarray:
    return jnp.clip(u, -limit, limit)


# ---------------------------------------------------------------------------
# Core dynamics
# ---------------------------------------------------------------------------


def dynamics_step(
    state: DroneState,
    control_input: jnp.ndarray,
    params: PhysicsParams,
) -> DroneState:
    """
    Forward Euler step for a point-mass model.

    `control_input` is interpreted as world-frame acceleration (m/s²) and is
    box-clipped to `±params.max_acceleration`.  This matches the control channel
    assumed when constructing the CBF-QP constraints.
    """
    u = _project_box(control_input, params.max_acceleration)
    dt = params.dt

    new_velocity = _clip_vector_norm(state.velocity + u * dt, params.max_velocity)
    new_position = state.position + state.velocity * dt + 0.5 * u * dt**2
    new_position = _clip_vector_norm(new_position, params.workspace_radius)

    return DroneState(
        position=new_position,
        velocity=new_velocity,
        acceleration=u,
        time=state.time + dt,
        orientation=state.orientation,
    )


# ---------------------------------------------------------------------------
# Temporal gradient decay (DiffPhysDrone)
# ---------------------------------------------------------------------------


def temporal_gradient_decay(x: jnp.ndarray, alpha: float) -> jnp.ndarray:
    """
    Implements the DiffPhys `g_decay` operator in JAX form.
    """
    return alpha * x + (1.0 - alpha) * jax.lax.stop_gradient(x)


def apply_temporal_gradient_decay_to_state(
    state: DroneState,
    decay_alpha: float,
) -> DroneState:
    """
    Applies temporal gradient decay to position, velocity, and acceleration.
    """
    return DroneState(
        position=temporal_gradient_decay(state.position, decay_alpha),
        velocity=temporal_gradient_decay(state.velocity, decay_alpha),
        acceleration=temporal_gradient_decay(state.acceleration, decay_alpha),
        time=state.time,
        orientation=state.orientation,
    )


# ---------------------------------------------------------------------------
# Convenience initialisers
# ---------------------------------------------------------------------------


def create_initial_state(
    position: jnp.ndarray | None = None,
    velocity: jnp.ndarray | None = None,
) -> DroneState:
    """Returns a zero-velocity hover state at the specified position."""
    pos = jnp.zeros(3) if position is None else position
    vel = jnp.zeros(3) if velocity is None else velocity
    return DroneState(
        position=pos,
        velocity=vel,
        acceleration=jnp.zeros(3),
        time=jnp.array(0.0, dtype=jnp.float32),
    )


# ---------------------------------------------------------------------------
# Batch utilities (optional)
# ---------------------------------------------------------------------------


def dynamics_step_batch(
    states: DroneState,
    controls: jnp.ndarray,
    params: PhysicsParams,
) -> Tuple[jnp.ndarray, Callable]:
    """
    vmap-friendly wrapper operating on stacked DroneState components.
    """
    def _single_step(position, velocity, acceleration, time, orientation, control):
        next_state = dynamics_step(
            DroneState(position, velocity, acceleration, time, orientation),
            control,
            params,
        )
        return (
            next_state.position,
            next_state.velocity,
            next_state.acceleration,
            next_state.time,
            next_state.orientation,
        )

    return jax.vmap(_single_step)(
        states.position,
        states.velocity,
        states.acceleration,
        states.time,
        states.orientation,
        controls,
    )


__all__ = [
    "DroneState",
    "PhysicsParams",
    "dynamics_step",
    "temporal_gradient_decay",
    "apply_temporal_gradient_decay_to_state",
    "create_initial_state",
    "dynamics_step_batch",
]
