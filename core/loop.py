"""
core.loop
=========

基于 JAX 的 rollout 工具，将感知、策略、安全过滤与物理仿真耦合成可微的
`lax.scan` 循环。这里的辅助结构会被训练脚本复用，使主文件保持精简，也方便
后续在评估代码中再次调用。
"""

from __future__ import annotations

from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from core.flax_compat import struct

from .physics import (
    DroneState,
    PhysicsParams,
    apply_temporal_gradient_decay_to_state,
    dynamics_step,
)
from .policy import PolicyNetwork, PolicyState
from .perception import compute_cbf_statistics, GraphConfig, _analytic_cbf_statistics
from .safety import SafetyConfig, SafetyDiagnostics, safety_filter


@struct.dataclass
class RolloutCarry:
    drone: DroneState
    policy: PolicyState


@struct.dataclass
class RolloutStepOutput:
    position: jnp.ndarray
    velocity: jnp.ndarray
    u_nominal: jnp.ndarray
    u_safe: jnp.ndarray
    cbf_value: jnp.ndarray
    soft_violation: jnp.ndarray
    relaxation: jnp.ndarray
    constraint_violation: jnp.ndarray
    relaxation_active: jnp.ndarray
    qp_failed: jnp.ndarray  # 记录安全求解器是否退化到备用策略，方便训练时统计
    qp_nan_detected: jnp.ndarray
    relaxation_limit_exceeded: jnp.ndarray
    qp_iterations: jnp.ndarray
    qp_status: jnp.ndarray


def rollout_episode(
    params: Dict[str, Dict],
    policy_net: PolicyNetwork,
    policy_state: PolicyState,
    initial_state: DroneState,
    physics_params: PhysicsParams,
    point_cloud: jnp.ndarray,
    graph_config: GraphConfig,
    safety_config: SafetyConfig | None,
    target_position: jnp.ndarray,
    horizon: int,
    gradient_decay: float,
    rng: jax.Array,
    noise_scale: float = 0.0,
    cbf_blend_alpha: float = 1.0,
) -> Tuple[RolloutCarry, RolloutStepOutput]:
    """
    执行一次可微分的单回合 rollout。参数设计与训练脚本保持一致，保持公共接口简洁；循环内部不再额外消耗 RNG，以避免 JIT 编译时出现非确定性。
    """

    carry = RolloutCarry(drone=initial_state, policy=policy_state)
    noise_keys = jax.random.split(rng, horizon)
    cbf_params = params.get("cbf") if isinstance(params, dict) else None
    use_safety = safety_config is not None and cbf_params is not None

    def scan_step(carry: RolloutCarry, key):
        state = carry.drone
        p_state = carry.policy

        cloud_key, pos_key, vel_key, tgt_key = jax.random.split(key, 4)
        point_noise = noise_scale * jax.random.normal(cloud_key, point_cloud.shape)
        noisy_cloud = point_cloud + point_noise

        if use_safety:
            cbf_value, grad_pos, hess_pos = compute_cbf_statistics(
                cbf_params, state, noisy_cloud, graph_config
            )
            analytic_value, analytic_grad, analytic_hess = _analytic_cbf_statistics(
                state, noisy_cloud
            )
            blend = jnp.asarray(cbf_blend_alpha, dtype=cbf_value.dtype)
            cbf_value = blend * cbf_value + (1.0 - blend) * analytic_value
            grad_pos = blend * grad_pos + (1.0 - blend) * analytic_grad
            hess_pos = blend * hess_pos + (1.0 - blend) * analytic_hess
            grad_pos = jax.lax.stop_gradient(grad_pos)
            hess_pos = jax.lax.stop_gradient(hess_pos)
        else:
            cbf_value = jnp.array(0.0, dtype=jnp.float32)
            grad_pos = jnp.zeros((3,), dtype=jnp.float32)
            hess_pos = jnp.zeros((3, 3), dtype=jnp.float32)

        pos_obs = state.position + noise_scale * jax.random.normal(pos_key, (3,))
        vel_obs = state.velocity + noise_scale * jax.random.normal(vel_key, (3,))
        tgt_obs = target_position + noise_scale * jax.random.normal(tgt_key, (3,))

        observation = jnp.concatenate(
            [
                pos_obs,
                vel_obs,
                tgt_obs - pos_obs,
                jnp.array([cbf_value]),
            ]
        )
        action, new_policy_state = policy_net.apply(
            params["policy"], observation[None, :], p_state, training=True
        )
        action = action.squeeze(0)

        if use_safety:
            u_safe, safety_diag = safety_filter(
                action, cbf_value, grad_pos, hess_pos, state, safety_config
            )
        else:
            u_safe = action
            # 当未启用安全层时，构造一个占位的诊断对象，所有数值置零
            safety_diag = SafetyDiagnostics(
                constraint_violation=jnp.array(0.0, dtype=jnp.float32),
                relaxation=jnp.array(0.0, dtype=jnp.float32),
                used_relaxation=jnp.array(0.0, dtype=jnp.float32),
                qp_failed=jnp.array(0.0, dtype=jnp.float32),
                nan_detected=jnp.array(0.0, dtype=jnp.float32),
                relaxation_limit_exceeded=jnp.array(0.0, dtype=jnp.float32),
                qp_iterations=jnp.array(0.0, dtype=jnp.float32),
                qp_status=jnp.array(0.0, dtype=jnp.float32),
                solver_mode=jnp.array(0.0, dtype=jnp.float32),
                solver_status=jnp.array(0.0, dtype=jnp.float32),
            )

        next_state = dynamics_step(state, u_safe, physics_params)
        next_state = apply_temporal_gradient_decay_to_state(
            next_state, gradient_decay
        )

        soft_violation = jnp.maximum(0.0, -cbf_value)

        outputs = RolloutStepOutput(
            position=next_state.position,
            velocity=next_state.velocity,
            u_nominal=action,
            u_safe=u_safe,
            cbf_value=cbf_value,
            soft_violation=soft_violation,
            relaxation=safety_diag.relaxation,
            constraint_violation=safety_diag.constraint_violation,
            relaxation_active=safety_diag.used_relaxation,
            qp_failed=safety_diag.qp_failed,
            qp_nan_detected=safety_diag.nan_detected,
            relaxation_limit_exceeded=safety_diag.relaxation_limit_exceeded,
            qp_iterations=safety_diag.qp_iterations,
            qp_status=safety_diag.qp_status,
        )
        new_carry = RolloutCarry(drone=next_state, policy=new_policy_state)
        return new_carry, outputs

    final_carry, outputs = jax.lax.scan(
        scan_step,
        carry,
        xs=noise_keys,
        length=horizon,
    )
    return final_carry, outputs


__all__ = ["RolloutCarry", "RolloutStepOutput", "rollout_episode"]
