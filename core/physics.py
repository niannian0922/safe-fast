"""
core.physics
================

轻量的可微分四旋翼点质量动力学实现，保持全流程可微与 JAX 优先；控制量视为世界坐标系下的加速度，并受盒约束限制；离散时间更新与 CBF-QP 使用的线性化模型一致；可选的时间梯度衰减用于稳定长视角的 BPTT 训练。

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from core.flax_compat import struct

# ---------------------------------------------------------------------------
# 状态与参数容器
# ---------------------------------------------------------------------------


@struct.dataclass
class DroneState:
    """与 JAX 变换兼容的单机点质量状态表示。"""

    position: jnp.ndarray  # (3,) 位置
    velocity: jnp.ndarray  # (3,) 速度
    acceleration: jnp.ndarray  # (3,) 最近一次下发的加速度指令
    time: jnp.ndarray  # 标量仿真时间

    # 姿态信息可选，保留它方便后续做感知模块的融合。
    orientation: jnp.ndarray = struct.field(
        default_factory=lambda: jnp.eye(3)
    )  # (3,3)


@dataclass(frozen=True)
class PhysicsParams:
    """与安全层共享的物理常量。"""

    dt: float = 1.0 / 15.0  # 控制频率（秒）
    gravity: float = 9.80665  # 重力加速度，仅作为参考
    max_acceleration: float = 5.0  # 加速度无穷范数上限（m/s^2）
    max_velocity: float = 10.0  # 速度软上限（m/s）
    workspace_radius: float = 30.0  # 软位置裁剪使用的工作半径
    gradient_decay: float = 0.4  # DiffPhys 风格的时间梯度衰减系数


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------


def _clip_vector_norm(vector: jnp.ndarray, limit: float) -> jnp.ndarray:
    """在保持平滑的前提下，将向量范数裁剪到给定欧氏长度。"""
    vector = jnp.asarray(vector)
    norm = jnp.linalg.norm(vector) + 1e-8
    scale = jnp.minimum(1.0, limit / norm)
    return vector * scale


def _project_box(u: jnp.ndarray, limit: float) -> jnp.ndarray:
    return jnp.clip(u, -limit, limit)


# ---------------------------------------------------------------------------
# 核心动力学
# ---------------------------------------------------------------------------


def dynamics_step(
    state: DroneState,
    control_input: jnp.ndarray,
    params: PhysicsParams,
) -> DroneState:
    """
    点质量模型的显式欧拉更新。

    `control_input` 被视为世界坐标系下的加速度（m/s²），并会被裁剪到
    `±params.max_acceleration`，与构造 CBF-QP 约束时假设的控制通道保持一致。
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
# 时间梯度衰减（DiffPhysDrone）
# ---------------------------------------------------------------------------


def temporal_gradient_decay(x: jnp.ndarray, alpha: float) -> jnp.ndarray:
    """以 JAX 形式实现 DiffPhys 中的 `g_decay` 运算。"""
    return alpha * x + (1.0 - alpha) * jax.lax.stop_gradient(x)


def apply_temporal_gradient_decay_to_state(
    state: DroneState,
    decay_alpha: float,
) -> DroneState:
    """将时间梯度衰减应用到位置、速度与加速度。"""
    return DroneState(
        position=temporal_gradient_decay(state.position, decay_alpha),
        velocity=temporal_gradient_decay(state.velocity, decay_alpha),
        acceleration=temporal_gradient_decay(state.acceleration, decay_alpha),
        time=state.time,
        orientation=state.orientation,
    )


# ---------------------------------------------------------------------------
# 便捷初始化函数
# ---------------------------------------------------------------------------


def create_initial_state(
    position: jnp.ndarray | None = None,
    velocity: jnp.ndarray | None = None,
) -> DroneState:
    """在给定位置返回零速度悬停状态。"""
    pos = jnp.zeros(3) if position is None else position
    vel = jnp.zeros(3) if velocity is None else velocity
    return DroneState(
        position=pos,
        velocity=vel,
        acceleration=jnp.zeros(3),
        time=jnp.array(0.0, dtype=jnp.float32),
    )


# ---------------------------------------------------------------------------
# 批量操作工具（可选）
# ---------------------------------------------------------------------------


def dynamics_step_batch(
    states: DroneState,
    controls: jnp.ndarray,
    params: PhysicsParams,
) -> Tuple[jnp.ndarray, Callable]:
    """对堆叠后的 DroneState 组件执行、适用于 vmap 的包装函数。"""
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
