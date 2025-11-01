"""
重构后的安全敏捷飞行训练入口。

本脚本串联了以下核心模块：

- `core.physics`：可微分的点质量动力学；
- `core.policy`：控制策略（MLP/GRU）；
- `core.perception`：GNN CBF 估计器；
- `core.safety`：可微 CBF-QP 安全层；
- `core.simple_training`：面向效率的损失塑形。

目标是提供一个紧凑但完整的基线，方便在研究实验中持续迭代。
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Tuple
import functools
import argparse
import pickle

import jax
import jax.numpy as jnp
import jax.nn as jnn
from core.flax_compat import struct
from jax import tree_util

from core.physics import (
    PhysicsParams,
    DroneState,
    create_initial_state,
)
from core.policy import (
    PolicyConfig,
    PolicyNetwork,
    PolicyState,
    initialise_policy_params,
)
from core.perception import (
    GraphConfig,
    build_graph_from_point_cloud,
    initialise_cbf_params,
    _analytic_cbf_statistics,
)
from core.safety import SafetyConfig
from core.simple_training import (
    DEFAULT_EFFICIENCY_CONFIG,
    compute_efficiency_loss,
)
from core.loop import rollout_episode as execute_rollout
from configs import default_config, stage1_analytic, stage1_efficiency
import optax


# ---------------------------------------------------------------------------
# 配置定义
# ---------------------------------------------------------------------------


@struct.dataclass
class TrainingConfig:
    horizon: int = 40
    policy_lr: float = 3e-4
    cbf_lr: float = 3e-4
    safety_weight: float = 2.0
    solver_violation_weight: float = 5.0
    relaxation_weight: float = 1.0
    relaxation_usage_weight: float = 0.0
    cbf_safe_weight: float = 0.5
    cbf_unsafe_weight: float = 0.5
    cbf_hdot_weight: float = 0.2
    cbf_margin: float = 0.05
    cbf_value_weight: float = 1.0
    cbf_weight_scale: float = 1.0
    gradient_decay: float = 0.4
    grad_clip: float = 1.0
    seed: int = 0
    episodes: int = 200
    curriculum_enabled: bool = True
    stage_steps: Tuple[int, int, int] = (300, 400, 300)
    noise_levels: Tuple[float, float, float] = (0.0, 0.02, 0.05)
    use_safety: bool = True
    target_position: jnp.ndarray = field(
        default_factory=lambda: jnp.array([4.0, 0.0, 1.5], dtype=jnp.float32)
    )
    policy_freeze_steps: int = 0
    policy_l2_weight: float = 0.0
    cbf_blend_alpha: float = 1.0
    augment_point_cloud: bool = True
    success_eval_frequency: int = 10
    success_eval_trials: int = 8
    success_eval_noise: float = 0.0
    success_eval_random_pc: bool = False
    success_threshold: float = 0.95
    success_tolerance: float = 0.1
    success_patience: int = 3
    success_eval_schedule: Tuple[float, ...] | None = None
    success_eval_noise_schedule: Tuple[float, ...] | None = None
    success_eval_random_pc_schedule: Tuple[int, ...] | None = None
    policy_distill_weight: float = 0.0
    cbf_blend_levels: Tuple[float, ...] = (1.0,)
    augment_levels: Tuple[int, ...] = (0,)
    violation_threshold: float = 20.0
    violation_threshold_schedule: Tuple[float, ...] | None = None
    robust_eval_frequency: int = 50
    robust_eval_trials: int = 16
    robust_eval_noise_levels: Tuple[float, ...] = (0.0, 0.03, 0.06)
    robust_eval_random_pc: bool = True
    relax_penalty_boost: float = 1.5
    relax_penalty_max: float = 8.0
    solver_penalty_boost: float = 1.2
    solver_penalty_max: float = 5.0
    blend_backoff: float = 0.2
    blend_min: float = 0.2
    relaxation_alert: float = 0.1
    hard_nan_rate: float = 0.05
    hard_relaxation_exceed_rate: float = 0.5
    hard_qp_fail_rate: float = 0.3
    point_cloud_modes: Tuple[str, ...] = ("ring",)
    hard_abort_warmup_episodes: int = 40
    hard_nan_schedule: Tuple[float, ...] | None = None
    hard_relax_schedule: Tuple[float, ...] | None = None
    hard_qp_fail_schedule: Tuple[float, ...] | None = None
    relaxation_scale_schedule: Tuple[float, ...] | None = None
    solver_scale_schedule: Tuple[float, ...] | None = None
    target_distance_schedule: Tuple[float, ...] | None = None
    teacher_gain_p: float = 0.0
    teacher_gain_d: float = 0.0
    teacher_weight: float = 0.0
    velocity_alignment_weight: float = 0.0
    desired_speed: float = 1.0
    final_velocity_weight: float = 0.0
    distance_bonus_weight: float = 0.0
    distance_bonus_threshold: float = 0.0
    initial_xy_range: float = 0.5
    initial_z_range: Tuple[float, float] = (0.8, 1.2)
    trajectory_projection_weight: float = 0.0
    distance_tracking_weight: float = 0.0
    teacher_force_schedule: Tuple[float, ...] | None = None
    final_distance_weight: float = 0.0


def _ensure_param_dict(obj):
    if isinstance(obj, dict):
        if "params" in obj:
            if set(obj.keys()) == {"params"}:
                return obj
            return {"params": obj["params"]}
        if "params_policy" in obj:
            val = obj["params_policy"]
            if isinstance(val, dict) and set(val.keys()) == {"params"}:
                return val
            return {"params": val}
    return obj


def make_ring_point_cloud(
    num_points: int,
    radius: float,
    height: float,
) -> jnp.ndarray:
    angles = jnp.linspace(0.0, 2.0 * jnp.pi, num_points, endpoint=False)
    x = radius * jnp.cos(angles)
    y = radius * jnp.sin(angles)
    z = jnp.full_like(angles, height)
    return jnp.stack([x, y, z], axis=1)


def _clamp_point_cloud(cloud: jnp.ndarray, graph_config: GraphConfig) -> jnp.ndarray:
    max_dist = jnp.asarray(graph_config.max_distance, dtype=cloud.dtype)
    cloud = jnp.clip(cloud, -max_dist, max_dist)
    z = jnp.clip(cloud[:, 2], 0.05, max_dist)
    return cloud.at[:, 2].set(z)


def make_cylinder_point_cloud(
    num_points: int,
    graph_config: GraphConfig,
    rng: jax.Array,
) -> Tuple[jnp.ndarray, jax.Array]:
    rng_angle, rng_radius, rng_height_scale, rng_height_offset, rng_noise, rng_next = jax.random.split(rng, 6)
    angles = jax.random.uniform(rng_angle, (num_points,), minval=0.0, maxval=2.0 * jnp.pi)
    base_radius = 0.4 * graph_config.max_distance
    radii = base_radius * (0.6 + 0.4 * jax.random.uniform(rng_radius, (num_points,)))
    x = radii * jnp.cos(angles)
    y = radii * jnp.sin(angles)
    heights = (0.6 + 0.4 * jax.random.uniform(rng_height_scale, (num_points,))) * 1.5
    z = jax.random.uniform(rng_height_offset, (num_points,), minval=-0.5, maxval=0.5) * heights + 1.0
    cloud = jnp.stack([x, y, z], axis=1)
    cloud += 0.1 * jax.random.normal(rng_noise, cloud.shape)
    return _clamp_point_cloud(cloud, graph_config), rng_next


def make_box_point_cloud(
    num_points: int,
    graph_config: GraphConfig,
    rng: jax.Array,
) -> Tuple[jnp.ndarray, jax.Array]:
    rng_center, rng_extent, rng_points, rng_next = jax.random.split(rng, 4)
    center = jax.random.uniform(
        rng_center,
        (3,),
        minval=jnp.array([-1.0, -1.0, 0.5]),
        maxval=jnp.array([1.0, 1.0, 1.5]),
    )
    extent = jax.random.uniform(rng_extent, (3,), minval=0.5, maxval=1.5)
    points = jax.random.uniform(rng_points, (num_points, 3), minval=-extent, maxval=extent)
    cloud = center + points
    return _clamp_point_cloud(cloud, graph_config), rng_next


def make_noise_point_cloud(
    num_points: int,
    graph_config: GraphConfig,
    rng: jax.Array,
) -> Tuple[jnp.ndarray, jax.Array]:
    rng_points, rng_radius, rng_next = jax.random.split(rng, 3)
    offsets = jax.random.normal(rng_points, (num_points, 3))
    radii = jax.random.uniform(rng_radius, (num_points, 1), minval=0.2, maxval=1.0)
    cloud = offsets * radii
    cloud = cloud.at[:, 2].add(1.0)
    return _clamp_point_cloud(cloud, graph_config), rng_next


def build_point_cloud_for_mode(
    mode: str,
    base_cloud: jnp.ndarray,
    graph_config: GraphConfig,
    rng: jax.Array,
) -> Tuple[jnp.ndarray, jax.Array]:
    mode_lower = mode.lower()
    num_points = graph_config.max_points
    if mode_lower == "ring":
        return base_cloud, rng
    if mode_lower == "cylinder":
        return make_cylinder_point_cloud(num_points, graph_config, rng)
    if mode_lower == "box":
        return make_box_point_cloud(num_points, graph_config, rng)
    if mode_lower == "noise":
        return make_noise_point_cloud(num_points, graph_config, rng)
    if mode_lower == "mixed":
        rng_cyl, rng_box, rng_noise, rng_next = jax.random.split(rng, 4)
        cyl_cloud, _ = make_cylinder_point_cloud(num_points // 3, graph_config, rng_cyl)
        box_cloud, _ = make_box_point_cloud(num_points // 3, graph_config, rng_box)
        noise_cloud, _ = make_noise_point_cloud(num_points - 2 * (num_points // 3), graph_config, rng_noise)
        cloud = jnp.concatenate([cyl_cloud, box_cloud, noise_cloud], axis=0)
        return _clamp_point_cloud(cloud, graph_config), rng_next
    return base_cloud, rng


def sample_augmented_point_cloud(
    base_cloud: jnp.ndarray,
    graph_config: GraphConfig,
    rng: jax.Array,
    jitter_scale: float = 0.3,
    replace_prob: float = 0.3,
) -> Tuple[jnp.ndarray, jax.Array]:
    rng_jitter, rng_replace, rng_new, rng_next = jax.random.split(rng, 4)
    noise = jitter_scale * jax.random.normal(rng_jitter, base_cloud.shape)
    jittered = base_cloud + noise
    max_dist = jnp.asarray(graph_config.max_distance, dtype=jittered.dtype)
    jittered = jnp.clip(jittered, -max_dist, max_dist)
    z = jittered[:, 2]
    z = jnp.clip(z, 0.05, max_dist)
    jittered = jittered.at[:, 2].set(z)

    replace_mask = jax.random.bernoulli(rng_replace, replace_prob, (base_cloud.shape[0], 1))
    new_samples = jax.random.uniform(
        rng_new,
        base_cloud.shape,
        minval=-max_dist,
        maxval=max_dist,
    )
    new_z = jnp.abs(new_samples[:, 2]) + 0.05
    new_samples = new_samples.at[:, 2].set(jnp.clip(new_z, 0.05, max_dist))
    augmented = jnp.where(replace_mask, new_samples, jittered)
    return augmented, rng_next


# ---------------------------------------------------------------------------
# Rollout 相关的数据结构
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 环境辅助函数
# ---------------------------------------------------------------------------


def sample_initial_state(
    rng: jax.Array,
    xy_range: float = 0.5,
    z_min: float = 0.8,
    z_max: float = 1.2,
) -> DroneState:
    rng_xy, rng_z = jax.random.split(rng)
    xy = jax.random.uniform(rng_xy, (2,), minval=-xy_range, maxval=xy_range)
    z = jax.random.uniform(rng_z, (), minval=z_min, maxval=z_max)
    pos = jnp.concatenate([xy, jnp.array([z])])
    return create_initial_state(position=pos)


# ---------------------------------------------------------------------------
# Rollout 逻辑
# ---------------------------------------------------------------------------


def compute_total_loss(
    params: Dict[str, Dict],
    rng: jax.Array,
    config: TrainingConfig,
    physics_params: PhysicsParams,
    policy_net: PolicyNetwork,
    policy_state: PolicyState,
    base_point_cloud: jnp.ndarray,
    graph_config: GraphConfig,
    safety_config: SafetyConfig,
    noise_scale: float,
    baseline_policy_params: Dict | None,
    blend_alpha: float,
    point_cloud_mode: str,
    augment_flag: bool,
    target_position: jnp.ndarray,
    relaxation_weight_scale: float = 1.0,
    solver_weight_scale: float = 1.0,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    key_init, rollout_rng = jax.random.split(rng)
    point_cloud, rollout_rng = build_point_cloud_for_mode(
        point_cloud_mode, base_point_cloud, graph_config, rollout_rng
    )
    if augment_flag:
        point_cloud, rollout_rng = sample_augmented_point_cloud(
            point_cloud, graph_config, rollout_rng
        )
    init_state = sample_initial_state(
        key_init,
        xy_range=config.initial_xy_range,
        z_min=float(config.initial_z_range[0]),
        z_max=float(config.initial_z_range[1]),
    )

    _, rollout = execute_rollout(
        params=params,
        policy_net=policy_net,
        policy_state=policy_state,
        initial_state=init_state,
        physics_params=physics_params,
        point_cloud=point_cloud,
        graph_config=graph_config,
        safety_config=safety_config,
        target_position=target_position,
        horizon=config.horizon,
        gradient_decay=config.gradient_decay,
        rng=rollout_rng,
        noise_scale=noise_scale,
        cbf_blend_alpha=blend_alpha,
    )
    positions = jnp.nan_to_num(rollout.position)
    controls = jnp.nan_to_num(rollout.u_safe)
    velocities = jnp.nan_to_num(rollout.velocity)
    cbf_values = jnp.nan_to_num(rollout.cbf_value)

    trajectory = {
        "positions": positions,
        "controls": controls,
        "velocities": velocities,
    }
    efficiency_loss, eff_metrics = compute_efficiency_loss(
        trajectory, target_position, DEFAULT_EFFICIENCY_CONFIG
    )
    velocity_dirs = target_position - positions
    velocity_dirs_norm = jnp.linalg.norm(velocity_dirs, axis=1, keepdims=True) + 1e-6
    velocity_dirs_unit = velocity_dirs / velocity_dirs_norm
    velocity_projection = jnp.sum(velocities * velocity_dirs_unit, axis=1)
    desired_speed = jnp.asarray(config.desired_speed, dtype=velocity_projection.dtype)
    velocity_alignment_loss = jnp.mean(jnn.relu(desired_speed - velocity_projection))
    projection_reward = jnp.mean(velocity_projection)
    projection_deficit = velocity_alignment_loss
    final_velocity = velocities[-1]
    final_velocity_penalty = jnp.linalg.norm(final_velocity) ** 2
    final_distance = jnp.linalg.norm(target_position - positions[-1])
    distance_bonus = -jnn.relu(config.distance_bonus_threshold - final_distance)
    trajectory_projection_penalty = projection_deficit
    final_distance_penalty = final_distance**2
    teacher_loss = jnp.array(0.0, dtype=jnp.float32)
    if config.teacher_weight > 0.0 and (config.teacher_gain_p > 0.0 or config.teacher_gain_d > 0.0):
        teacher_actions = (
            config.teacher_gain_p * (target_position - positions)
            - config.teacher_gain_d * velocities
        )
        teacher_actions = jnp.clip(
            teacher_actions,
            -physics_params.max_acceleration,
            physics_params.max_acceleration,
        )
        teacher_loss = jnp.mean((rollout.u_nominal - teacher_actions) ** 2)
    safety_penalty = jnp.mean(jnp.nan_to_num(rollout.soft_violation))
    violation_penalty = jnp.mean(jnp.nan_to_num(rollout.constraint_violation))
    relaxation_cost = jnp.mean(jnp.nan_to_num(rollout.relaxation))
    relaxation_rate = jnp.mean(jnp.nan_to_num(rollout.relaxation_active))
    # 安全求解器可能退回到备用控制，因此单独统计失败频次，后续可以在日志中对齐定位。
    qp_fail_rate = jnp.mean(jnp.nan_to_num(rollout.qp_failed))
    qp_nan_rate = jnp.mean(jnp.nan_to_num(rollout.qp_nan_detected))
    relaxation_exceeded_rate = jnp.mean(jnp.nan_to_num(rollout.relaxation_limit_exceeded))
    qp_iteration_mean = jnp.mean(jnp.nan_to_num(rollout.qp_iterations))
    qp_iteration_max = jnp.max(jnp.nan_to_num(rollout.qp_iterations))
    qp_status_latest = jnp.nan_to_num(rollout.qp_status[-1])
    cbf_mean = jnp.mean(cbf_values)
    cbf_min = jnp.min(cbf_values)
    distill_loss = jnp.array(0.0, dtype=jnp.float32)
    if baseline_policy_params is not None and config.policy_distill_weight > 0.0:
        target_offsets = target_position - positions
        observations = jnp.concatenate(
            [positions, velocities, target_offsets, cbf_values[:, None]], axis=1
        )
        def fwd(params_net, obs_seq):
            def step(carry, obs):
                action, carry = policy_net.apply(params_net, obs[None, :], carry, training=False)
                return carry, action.squeeze(0)
            _, actions = jax.lax.scan(step, policy_state, obs_seq)
            return actions

        baseline_actions = fwd(baseline_policy_params, observations)
        student_actions = fwd(params["policy"], observations)
        distill_loss = jnp.mean((student_actions - baseline_actions) ** 2)

    total_loss = (
        efficiency_loss
        + config.safety_weight * safety_penalty
        + (config.solver_violation_weight * solver_weight_scale) * violation_penalty
        + (config.relaxation_weight * relaxation_weight_scale) * relaxation_cost
        + (config.relaxation_usage_weight * relaxation_weight_scale) * relaxation_rate
        + config.policy_distill_weight * distill_loss
        + config.velocity_alignment_weight * velocity_alignment_loss
        + config.teacher_weight * teacher_loss
        + config.final_velocity_weight * final_velocity_penalty
        + config.distance_bonus_weight * distance_bonus
        - config.trajectory_projection_weight * projection_reward
        + config.distance_tracking_weight * trajectory_projection_penalty
        + config.final_distance_weight * final_distance_penalty
    )

    cbf_safe_loss = jnp.array(0.0, dtype=jnp.float32)
    cbf_unsafe_loss = jnp.array(0.0, dtype=jnp.float32)
    cbf_hdot_loss = jnp.array(0.0, dtype=jnp.float32)
    cbf_safe_fraction = jnp.array(0.0, dtype=jnp.float32)
    cbf_unsafe_fraction = jnp.array(0.0, dtype=jnp.float32)
    cbf_analytic_mean = cbf_mean
    cbf_analytic_min = cbf_min

    cbf_value_loss = jnp.array(0.0, dtype=jnp.float32)

    if safety_config is not None and "cbf" in params:
        margin = jnp.asarray(config.cbf_margin, dtype=cbf_values.dtype)
        dt = jnp.asarray(physics_params.dt, dtype=cbf_values.dtype)

        def analytic_value_fn(position: jnp.ndarray) -> jnp.ndarray:
            state = DroneState(
                position=position,
                velocity=jnp.zeros_like(position),
                acceleration=jnp.zeros_like(position),
                time=jnp.array(0.0, dtype=position.dtype),
            )
            value, _, _ = _analytic_cbf_statistics(state, point_cloud)
            return value

        analytic_values = jax.vmap(analytic_value_fn)(positions)
        analytic_values = jnp.nan_to_num(analytic_values)

        safe_mask = analytic_values >= margin
        unsafe_mask = analytic_values <= -margin
        safe_mask_f = safe_mask.astype(cbf_values.dtype)
        unsafe_mask_f = unsafe_mask.astype(cbf_values.dtype)

        safe_penalty = jnp.where(safe_mask, jnn.relu(margin - cbf_values), 0.0)
        unsafe_penalty = jnp.where(unsafe_mask, jnn.relu(cbf_values + margin), 0.0)

        safe_count = jnp.sum(safe_mask_f)
        unsafe_count = jnp.sum(unsafe_mask_f)
        horizon_float = jnp.array(float(cbf_values.shape[0]), dtype=cbf_values.dtype)

        cbf_safe_loss = jnp.sum(safe_penalty) / (safe_count + 1e-6)
        cbf_unsafe_loss = jnp.sum(unsafe_penalty) / (unsafe_count + 1e-6)
        cbf_safe_fraction = safe_count / (horizon_float + 1e-6)
        cbf_unsafe_fraction = unsafe_count / (horizon_float + 1e-6)

        if cbf_values.shape[0] > 1:
            h_curr = cbf_values[:-1]
            h_next = cbf_values[1:]
            h_dot = (h_next - h_curr) / dt
            constraint = h_dot + safety_config.alpha0 * h_curr
            hdot_penalty = jnn.relu(-constraint + margin * 0.5)
            cbf_hdot_loss = jnp.mean(hdot_penalty)
        else:
            cbf_hdot_loss = jnp.array(0.0, dtype=jnp.float32)

        cbf_value_loss = jnp.mean((cbf_values - analytic_values) ** 2)

        cbf_weight_scale = jnp.asarray(config.cbf_weight_scale, dtype=cbf_values.dtype)
        total_loss = total_loss + cbf_weight_scale * (
            config.cbf_safe_weight * cbf_safe_loss
            + config.cbf_unsafe_weight * cbf_unsafe_loss
            + config.cbf_hdot_weight * cbf_hdot_loss
            + config.cbf_value_weight * cbf_value_loss
        )

        cbf_analytic_mean = jnp.mean(analytic_values)
        cbf_analytic_min = jnp.min(analytic_values)

    target_xy = jnp.linalg.norm(target_position[:2])
    metrics = {
        "loss/total": total_loss,
        "loss/efficiency": efficiency_loss,
        "loss/safety_soft": safety_penalty,
        "loss/constraint_violation": violation_penalty,
        "loss/distill": distill_loss,
        "safety/relaxation_mean": relaxation_cost,
        "safety/relaxation_rate": relaxation_rate,
        "safety/qp_fail_rate": qp_fail_rate,
        "safety/qp_nan_rate": qp_nan_rate,
        "safety/relaxation_exceeded_rate": relaxation_exceeded_rate,
        "safety/qp_iterations_mean": qp_iteration_mean,
        "safety/qp_iterations_max": qp_iteration_max,
        "safety/qp_status_last": qp_status_latest,
        "safety/cbf_mean": cbf_mean,
        "safety/cbf_min": cbf_min,
        "loss/cbf_safe": cbf_safe_loss,
        "loss/cbf_unsafe": cbf_unsafe_loss,
        "loss/cbf_hdot": cbf_hdot_loss,
        "loss/cbf_value": cbf_value_loss,
        "loss/velocity_alignment": velocity_alignment_loss,
        "loss/teacher": teacher_loss,
        "loss/final_velocity": final_velocity_penalty,
        "loss/distance_bonus": distance_bonus,
        "loss/distance_tracking": trajectory_projection_penalty,
        "loss/final_distance": final_distance_penalty,
        "reward/projection": projection_reward,
        "cbf/safe_fraction": cbf_safe_fraction,
        "cbf/unsafe_fraction": cbf_unsafe_fraction,
        "cbf/analytic_mean": cbf_analytic_mean,
        "cbf/analytic_min": cbf_analytic_min,
        "curriculum/noise_scale": jnp.array(noise_scale, dtype=jnp.float32),
        "curriculum/target_distance": jnp.array(target_xy, dtype=jnp.float32),
    }
    metrics.update({f"eff/{k}": v for k, v in eff_metrics.items()})
    metrics = {k: jnp.nan_to_num(v) for k, v in metrics.items()}
    total_loss = jnp.nan_to_num(total_loss)
    return total_loss, metrics


def make_train_step(
    config: TrainingConfig,
    physics_params: PhysicsParams,
    policy_net: PolicyNetwork,
    policy_state: PolicyState,
    base_point_cloud: jnp.ndarray,
    graph_config: GraphConfig,
    safety_config: SafetyConfig,
    optimizer_policy: optax.GradientTransformation,
    optimizer_cbf: optax.GradientTransformation | None,
    baseline_policy_params: Dict | None,
):

    @functools.partial(jax.jit, static_argnames=("point_cloud_mode", "augment_flag"))
    def train_step(
        params: Dict[str, Dict],
        opt_state_policy: optax.OptState,
        opt_state_cbf: optax.OptState | None,
        rng: jax.Array,
        noise_scale: float,
        blend_alpha: float,
        point_cloud_mode: str,
        augment_flag: bool,
        target_position: jnp.ndarray,
        step_index: int,
        relaxation_scale: float,
        solver_scale: float,
        teacher_blend: float = 0.0,
    ):
        def loss_fn(p):
            return compute_total_loss(
                p,
                rng,
                config,
                physics_params,
                policy_net,
                policy_state,
                base_point_cloud,
                graph_config,
                safety_config,
                noise_scale,
                baseline_policy_params,
                blend_alpha,
                point_cloud_mode,
                augment_flag,
                target_position,
                relaxation_scale,
                solver_scale,
            )

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        grads = tree_util.tree_map(lambda g: jnp.nan_to_num(g), grads)

        freeze_policy = step_index < config.policy_freeze_steps

        policy_grads = grads["policy"]
        policy_grad_norm = optax.global_norm(policy_grads)
        policy_updates, new_opt_state_policy = optimizer_policy.update(
            policy_grads, opt_state_policy, params["policy"]
        )
        new_policy_params = optax.apply_updates(params["policy"], policy_updates)

        new_policy_params, new_opt_state_policy = jax.lax.cond(
            freeze_policy,
            lambda _: (params["policy"], opt_state_policy),
            lambda _: (new_policy_params, new_opt_state_policy),
            operand=None,
        )
        policy_grad_norm = jax.lax.cond(
            freeze_policy,
            lambda _: jnp.array(0.0, dtype=jnp.float32),
            lambda _: policy_grad_norm,
            operand=None,
        )

        new_params: Dict[str, Dict] = dict(params)
        new_params["policy"] = new_policy_params

        if optimizer_cbf is not None and "cbf" in params:
            cbf_grads = grads["cbf"]
            cbf_grad_norm = optax.global_norm(cbf_grads)
            cbf_updates, new_opt_state_cbf = optimizer_cbf.update(
                cbf_grads, opt_state_cbf, params["cbf"]
            )
            new_params["cbf"] = optax.apply_updates(params["cbf"], cbf_updates)
        else:
            cbf_grad_norm = jnp.array(0.0, dtype=jnp.float32)
            new_opt_state_cbf = opt_state_cbf
            if "cbf" in params and "cbf" not in new_params:
                new_params["cbf"] = params["cbf"]

        metrics = {
            **metrics,
            "loss/grad_norm_policy": jnp.nan_to_num(policy_grad_norm),
            "loss/grad_norm_cbf": jnp.nan_to_num(cbf_grad_norm),
            "loss/grad_norm": jnp.nan_to_num(policy_grad_norm),
        }
        return new_params, new_opt_state_policy, new_opt_state_cbf, metrics

    return train_step


# ---------------------------------------------------------------------------
# 主训练流程
# ---------------------------------------------------------------------------


CONFIG_REGISTRY = {
    "default": default_config.get_config,
    "stage1": stage1_analytic.get_config,
    "stage1_analytic": stage1_analytic.get_config,
    "stage1_efficiency": stage1_efficiency.get_config,
}


def load_base_config(name: str):
    """根据用户指定的配置名称返回基础配置，确保 macOS 与 Windows 都能走相同流程。"""
    key = name.lower()
    if key not in CONFIG_REGISTRY:
        raise ValueError(f"未识别的配置名称: {name}")
    return CONFIG_REGISTRY[key]()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 CBF-QP 训练安全策略。")
    parser.add_argument("--episodes", type=int, default=None, help="训练的 episode 数量")
    parser.add_argument("--horizon", type=int, default=None, help="单次 rollout 的步长")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument(
        "--grad-decay",
        type=float,
        default=None,
        help="时间梯度衰减系数",
    )
    parser.add_argument(
        "--safety-weight",
        type=float,
        default=None,
        help="软 CBF 惩罚项的权重",
    )
    parser.add_argument(
        "--solver-weight",
        type=float,
        default=None,
        help="约束违约惩罚的权重",
    )
    parser.add_argument(
        "--relaxation-weight",
        type=float,
        default=None,
        help="安全层松弛项的权重",
    )
    parser.add_argument(
        "--relax-usage-weight",
        type=float,
        default=None,
        help="松弛项平均激活率的权重",
    )
    parser.add_argument(
        "--policy-lr",
        type=float,
        default=None,
        help="策略参数的学习率",
    )
    parser.add_argument(
        "--target-position",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="目标位置（米）",
    )
    parser.add_argument(
        "--use-rnn",
        action="store_true",
        help="启用基于 GRU 的策略（默认使用纯 MLP）",
    )
    parser.add_argument(
        "--stage-steps",
        type=str,
        default=None,
        help="逗号分隔的课程阶段长度（例如 300,400,300）",
    )
    parser.add_argument(
        "--noise-levels",
        type=str,
        default=None,
        help="逗号分隔的各阶段噪声标准差（如 0.0,0.02,0.05）",
    )
    parser.add_argument(
        "--disable-curriculum",
        action="store_true",
        help="禁用分阶段课程（仅使用最后阶段的噪声）",
    )
    parser.add_argument(
        "--disable-safety",
        action="store_true",
        help="跳过 CBF/QP 安全层（直接使用策略输出）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="可选：用于保存最终参数和训练日志的目录",
    )
    parser.add_argument(
        "--cbf-params",
        type=str,
        default=None,
        help="可选：预训练 CBF 参数的 pickle 文件",
    )
    parser.add_argument(
        "--policy-params",
        type=str,
        default=None,
        help="可选：预训练策略参数的 pickle 文件",
    )
    parser.add_argument(
        "--policy-freeze-steps",
        type=int,
        default=None,
        help="策略冻结的初始 episode 数",
    )
    parser.add_argument(
        "--cbf-blend-alpha",
        type=float,
        default=None,
        help="神经/解析 CBF 的混合权重（1.0 代表仅使用神经 CBF）",
    )
    parser.add_argument(
        "--disable-pointcloud-augment",
        action="store_true",
        help="训练期间禁用点云增强",
    )
    parser.add_argument(
        "--success-threshold",
        type=float,
        default=None,
        help="触发 checkpoint/回滚的成功率阈值",
    )
    parser.add_argument(
        "--success-eval-schedule",
        type=str,
        default=None,
        help="逗号分隔的成功率评估混合因子（如 0.0,0.5,1.0）",
    )
    parser.add_argument(
        "--success-eval-noise",
        type=str,
        default=None,
        help="逗号分隔的成功率评估噪声",
    )
    parser.add_argument(
        "--success-eval-random-pc",
        type=str,
        default=None,
        help="逗号分隔的 0/1 标记，控制各阶段成功率评估是否启用随机点云",
    )
    parser.add_argument(
        "--violation-threshold",
        type=float,
        default=None,
        help="评估阶段允许的最大违约值，超出则触发回滚",
    )
    parser.add_argument(
        "--violation-schedule",
        type=str,
        default=None,
        help="逗号分隔的各阶段违约阈值",
    )
    parser.add_argument(
        "--robust-eval-frequency",
        type=int,
        default=None,
        help="执行鲁棒性评估的频率（以 episode 计）",
    )
    parser.add_argument(
        "--robust-eval-trials",
        type=int,
        default=None,
        help="鲁棒性评估中每个噪声水平的 rollout 次数",
    )
    parser.add_argument(
        "--robust-eval-noise",
        type=str,
        default=None,
        help="逗号分隔的鲁棒性噪声水平",
    )
    parser.add_argument(
        "--robust-eval-no-pc",
        action="store_true",
        help="鲁棒性评估时禁用点云增强",
    )
    parser.add_argument(
        "--blend-backoff",
        type=float,
        default=None,
        help="违约超过阈值时应用的混合权重衰减",
    )
    parser.add_argument(
        "--blend-min",
        type=float,
        default=None,
        help="自动退回后的混合权重下限",
    )
    parser.add_argument(
        "--relax-boost",
        type=float,
        default=None,
        help="违约超过阈值时松弛惩罚的放大倍数",
    )
    parser.add_argument(
        "--relax-max",
        type=float,
        default=None,
        help="松弛惩罚的放大上限",
    )
    parser.add_argument(
        "--solver-boost",
        type=float,
        default=None,
        help="违约超过阈值时求解器惩罚的放大倍数",
    )
    parser.add_argument(
        "--solver-max",
        type=float,
        default=None,
        help="求解器惩罚的放大上限",
    )
    parser.add_argument(
        "--relax-alert",
        type=float,
        default=None,
        help="即便不评估也会触发自适应惩罚的松弛均值阈值",
    )
    parser.add_argument(
        "--distill-policy",
        type=str,
        default=None,
        help="可选：用于蒸馏正则的教师策略参数",
    )
    parser.add_argument(
        "--distill-weight",
        type=float,
        default=None,
        help="策略蒸馏损失的权重",
    )
    parser.add_argument(
        "--blend-levels",
        type=str,
        default=None,
        help="逗号分隔的混合权重设置，与阶段步数一一对应",
    )
    parser.add_argument(
        "--augment-levels",
        type=str,
        default=None,
        help="逗号分隔的 0/1 标记，控制各阶段是否启用点云增强",
    )
    parser.add_argument(
        "--point-cloud-modes",
        type=str,
        default=None,
        help="逗号分隔的点云模式（ring/cylinder/box/noise/mixed），与阶段对应",
    )
    parser.add_argument(
        "--hard-nan-rate",
        type=float,
        default=None,
        help="安全诊断硬阈值：允许的 qp NaN 率上限（<0 表示禁用）",
    )
    parser.add_argument(
        "--hard-relax-rate",
        type=float,
        default=None,
        help="安全诊断硬阈值：允许的松弛超限率上限（<0 表示禁用）",
    )
    parser.add_argument(
        "--hard-qp-fail-rate",
        type=float,
        default=None,
        help="安全诊断硬阈值：允许的 QP 失败率上限（<0 表示禁用）",
    )
    parser.add_argument(
        "--hard-warmup-episodes",
        type=int,
        default=None,
        help="硬阈值启用前的热身 episode 数（0 表示立即启用）",
    )
    parser.add_argument(
        "--hard-nan-schedule",
        type=str,
        default=None,
        help="逗号分隔的 NaN 率阈值 schedule，与阶段对应",
    )
    parser.add_argument(
        "--hard-relax-schedule",
        type=str,
        default=None,
        help="逗号分隔的松弛超限阈值 schedule，与阶段对应",
    )
    parser.add_argument(
        "--hard-qp-fail-schedule",
        type=str,
        default=None,
        help="逗号分隔的 QP 失败率阈值 schedule，与阶段对应",
    )
    parser.add_argument(
        "--relax-scale-schedule",
        type=str,
        default=None,
        help="逗号分隔的初始松弛惩罚缩放 schedule",
    )
    parser.add_argument(
        "--solver-scale-schedule",
        type=str,
        default=None,
        help="逗号分隔的初始求解器惩罚缩放 schedule",
    )
    parser.add_argument(
        "--target-distance-schedule",
        type=str,
        default=None,
        help="逗号分隔的目标距离课程（米），与阶段对应",
    )
    parser.add_argument(
        "--teacher-gain-p",
        type=float,
        default=None,
        help="教师控制比例增益，如果>0 则启用教师蒸馏",
    )
    parser.add_argument(
        "--teacher-gain-d",
        type=float,
        default=None,
        help="教师控制速度增益",
    )
    parser.add_argument(
        "--teacher-weight",
        type=float,
        default=None,
        help="教师蒸馏损失权重",
    )
    parser.add_argument(
        "--velocity-alignment-weight",
        type=float,
        default=None,
        help="速度对齐损失权重",
    )
    parser.add_argument(
        "--desired-speed",
        type=float,
        default=None,
        help="速度对齐的期望速度",
    )
    parser.add_argument(
        "--final-velocity-weight",
        type=float,
        default=None,
        help="末端速度惩罚权重",
    )
    parser.add_argument(
        "--final-distance-weight",
        type=float,
        default=None,
        help="终点距离平方惩罚权重",
    )
    parser.add_argument(
        "--distance-bonus-weight",
        type=float,
        default=None,
        help="终点距离奖励的权重",
    )
    parser.add_argument(
        "--distance-bonus-threshold",
        type=float,
        default=None,
        help="终点距离奖励的阈值（米）",
    )
    parser.add_argument(
        "--trajectory-projection-weight",
        type=float,
        default=None,
        help="轨迹沿目标方向积分奖励权重",
    )
    parser.add_argument(
        "--distance-tracking-weight",
        type=float,
        default=None,
        help="轨迹距离平方损失权重",
    )
    parser.add_argument(
        "--teacher-force-schedule",
        type=str,
        default=None,
        help="逗号分隔的教师混合系数 schedule (0-1)",
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default="default",
        help="基础配置名称，例如 default 或 stage1_analytic，用于不同阶段之间对齐参数",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace, base_cfg) -> TrainingConfig:
    """结合基础配置与命令行参数，生成训练阶段使用的结构化配置。"""
    cfg = base_cfg

    horizon = args.horizon if args.horizon is not None else int(cfg.physics.max_steps)
    policy_lr = args.policy_lr if args.policy_lr is not None else float(cfg.training.learning_rate_policy)
    cbf_lr = float(cfg.training.learning_rate_gcbf)
    safety_weight = args.safety_weight if args.safety_weight is not None else float(cfg.training.loss_safety_coef)
    solver_weight = args.solver_weight if args.solver_weight is not None else float(cfg.training.loss_collision_coef)
    relaxation_weight = args.relaxation_weight if args.relaxation_weight is not None else float(cfg.training.loss_control_coef)
    relaxation_usage_weight = args.relax_usage_weight if args.relax_usage_weight is not None else 0.0
    grad_decay = args.grad_decay if args.grad_decay is not None else float(cfg.physics.gradient_decay.alpha)
    grad_clip = float(cfg.training.gradient_clip_norm)
    seed = args.seed if args.seed is not None else int(cfg.system.seed)
    episodes = args.episodes if args.episodes is not None else int(cfg.training.max_steps)

    target_tuple = args.target_position
    if target_tuple is None:
        target_tuple = (4.0, 0.0, 1.5)

    def parse_int_tuple(raw: str) -> Tuple[int, ...]:
        try:
            values = tuple(int(s.strip()) for s in raw.split(",") if s.strip())
        except ValueError as exc:  # pragma: no cover
            raise ValueError(f"stage_steps 格式不合法: {raw}") from exc
        if not values:
            raise ValueError("stage_steps 至少需要包含一个整数")
        return values

    def parse_float_tuple(raw: str) -> Tuple[float, ...]:
        try:
            values = tuple(float(s.strip()) for s in raw.split(",") if s.strip())
        except ValueError as exc:  # pragma: no cover
            raise ValueError(f"noise_levels 格式不合法: {raw}") from exc
        if not values:
            raise ValueError("noise_levels 至少需要包含一个浮点数")
        return values

    def parse_str_tuple(raw: str) -> Tuple[str, ...]:
        values = tuple(s.strip() for s in raw.split(",") if s.strip())
        if not values:
            raise ValueError("point_cloud_modes 至少需要包含一个模式名称")
        return values

    curriculum_enabled = bool(cfg.training.curriculum.enable) and not args.disable_curriculum

    if args.stage_steps:
        stage_steps = parse_int_tuple(args.stage_steps)
    else:
        stage_steps = (
            int(cfg.training.curriculum.stage1_steps),
            int(cfg.training.curriculum.stage2_steps),
            int(cfg.training.curriculum.stage3_steps),
        )

    if args.noise_levels:
        noise_levels = parse_float_tuple(args.noise_levels)
    elif "stage_noise_level" in cfg.training.curriculum:
        noise_levels = tuple(cfg.training.curriculum.stage_noise_level)
    else:
        noise_levels = (0.0, 0.02, 0.05)

    def parse_float_optional(raw: str) -> Tuple[float, ...]:
        try:
            values = tuple(float(s.strip()) for s in raw.split(",") if s.strip())
        except ValueError as exc:
            raise ValueError(f"blend_levels 格式不合法: {raw}") from exc
        if not values:
            raise ValueError("blend_levels 至少需要包含一个数值")
        return values

    def parse_int_optional(raw: str) -> Tuple[int, ...]:
        try:
            values = tuple(int(s.strip()) for s in raw.split(",") if s.strip())
        except ValueError as exc:
            raise ValueError(f"augment_levels 格式不合法: {raw}") from exc
        if not values:
            raise ValueError("augment_levels 至少需要包含一个数值")
        return values

    if args.blend_levels:
        blend_levels = parse_float_optional(args.blend_levels)
    else:
        blend_levels = (1.0,) * len(stage_steps)

    if args.augment_levels:
        augment_levels = parse_int_optional(args.augment_levels)
    else:
        augment_levels = (1 if not args.disable_pointcloud_augment else 0,) * len(stage_steps)

    stage_count = len(stage_steps) if curriculum_enabled else 1

    if len(blend_levels) != len(stage_steps):
        raise ValueError("blend_levels 的长度必须与 stage_steps 一致")
    if len(augment_levels) != len(stage_steps):
        raise ValueError("augment_levels 的长度必须与 stage_steps 一致")

    policy_freeze_steps = (
        args.policy_freeze_steps
        if args.policy_freeze_steps is not None
        else int(cfg.training.curriculum.stage1_steps)
    )
    cbf_blend_alpha = (
        args.cbf_blend_alpha if args.cbf_blend_alpha is not None else 1.0
    )
    augment_point_cloud = not args.disable_pointcloud_augment
    base_success_threshold = float(getattr(cfg.evaluation, "success_threshold", 0.95))
    success_threshold = (
        args.success_threshold if args.success_threshold is not None else base_success_threshold
    )
    success_tolerance = float(getattr(cfg.env, "goal_tolerance", 0.1))
    policy_distill_weight = (
        args.distill_weight if args.distill_weight is not None else 0.0
    )
    violation_threshold = (
        args.violation_threshold if args.violation_threshold is not None else 20.0
    )
    if args.violation_schedule:
        violation_schedule = tuple(
            float(s.strip()) for s in args.violation_schedule.split(",") if s.strip()
        )
    else:
        if stage_count > 1:
            start = max(violation_threshold * 2.5, 40.0)
            end = max(violation_threshold, 15.0)
            violation_schedule = tuple(
                float(
                    start
                    + (end - start) * (idx / (stage_count - 1))
                )
                for idx in range(stage_count)
            )
        else:
            violation_schedule = None
    robust_eval_frequency = (
        args.robust_eval_frequency if args.robust_eval_frequency is not None else 50
    )
    robust_eval_trials = (
        args.robust_eval_trials if args.robust_eval_trials is not None else 16
    )
    if args.robust_eval_noise:
        robust_eval_noise_levels = parse_float_tuple(args.robust_eval_noise)
    else:
        if stage_count > 0:
            robust_eval_noise_levels = tuple(
                min(0.06, 0.015 * (idx + 1)) for idx in range(stage_count)
            )
        else:
            robust_eval_noise_levels = (0.03, 0.06)
    robust_eval_random_pc = not args.robust_eval_no_pc
    blend_backoff = args.blend_backoff if args.blend_backoff is not None else 0.2
    blend_min = args.blend_min if args.blend_min is not None else 0.2
    relax_boost = args.relax_boost if args.relax_boost is not None else 1.5
    relax_max = args.relax_max if args.relax_max is not None else 8.0
    solver_boost = args.solver_boost if args.solver_boost is not None else 1.2
    solver_max = args.solver_max if args.solver_max is not None else 5.0
    relax_alert = args.relax_alert if args.relax_alert is not None else 0.1

    if args.success_eval_schedule:
        success_eval_schedule = tuple(
            float(s.strip()) for s in args.success_eval_schedule.split(",") if s.strip()
        )
    else:
        if stage_count > 1:
            success_eval_schedule = tuple(
                max(0.2, 1.0 - 0.4 * i) for i in range(stage_count)
            )
        else:
            success_eval_schedule = None
    if args.success_eval_noise:
        success_eval_noise_schedule = tuple(
            float(s.strip()) for s in args.success_eval_noise.split(",") if s.strip()
        )
    else:
        if stage_count > 0:
            success_eval_noise_schedule = tuple(
                float(min(0.05, 0.02 * i)) for i in range(stage_count)
            )
        else:
            success_eval_noise_schedule = None
    if args.success_eval_random_pc:
        success_eval_random_pc_schedule = tuple(
            int(s.strip()) for s in args.success_eval_random_pc.split(",") if s.strip()
        )
    else:
        if stage_count > 0:
            success_eval_random_pc_schedule = tuple(
                1 if i == stage_count - 1 else 0 for i in range(stage_count)
            )
        else:
            success_eval_random_pc_schedule = None

    if args.point_cloud_modes:
        point_cloud_modes = parse_str_tuple(args.point_cloud_modes)
    else:
        point_cloud_modes = tuple(getattr(cfg.training, "point_cloud_modes", ("ring",)))
    if len(point_cloud_modes) < stage_count and stage_count > 0:
        point_cloud_modes = point_cloud_modes + (point_cloud_modes[-1],) * (stage_count - len(point_cloud_modes))

    hard_nan_rate = (
        args.hard_nan_rate
        if args.hard_nan_rate is not None
        else float(getattr(cfg.training, "hard_nan_rate", 0.05))
    )
    hard_relax_rate = (
        args.hard_relax_rate
        if args.hard_relax_rate is not None
        else float(getattr(cfg.training, "hard_relaxation_exceed_rate", 0.5))
    )
    hard_qp_fail_rate = (
        args.hard_qp_fail_rate
        if args.hard_qp_fail_rate is not None
        else float(getattr(cfg.training, "hard_qp_fail_rate", 0.3))
    )
    hard_warmup = (
        args.hard_warmup_episodes
        if args.hard_warmup_episodes is not None
        else int(getattr(cfg.training, "hard_abort_warmup_episodes", 40))
    )

    if args.hard_nan_schedule:
        hard_nan_schedule = parse_float_tuple(args.hard_nan_schedule)
    else:
        hard_nan_schedule = tuple(getattr(cfg.training, "hard_nan_schedule", ()))
    if args.hard_relax_schedule:
        hard_relax_schedule = parse_float_tuple(args.hard_relax_schedule)
    else:
        hard_relax_schedule = tuple(getattr(cfg.training, "hard_relax_schedule", ()))
    if args.hard_qp_fail_schedule:
        hard_qp_schedule = parse_float_tuple(args.hard_qp_fail_schedule)
    else:
        hard_qp_schedule = tuple(getattr(cfg.training, "hard_qp_fail_schedule", ()))

    if args.relax_scale_schedule:
        relax_scale_schedule = parse_float_tuple(args.relax_scale_schedule)
    else:
        relax_scale_schedule = tuple(getattr(cfg.training, "relaxation_scale_schedule", ()))

    if args.solver_scale_schedule:
        solver_scale_schedule = parse_float_tuple(args.solver_scale_schedule)
    else:
        solver_scale_schedule = tuple(getattr(cfg.training, "solver_scale_schedule", ()))
    if args.target_distance_schedule:
        target_distance_schedule = parse_float_tuple(args.target_distance_schedule)
    else:
        target_distance_schedule = tuple(getattr(cfg.training, "target_distance_schedule", ()))
    teacher_gain_p = (
        args.teacher_gain_p if args.teacher_gain_p is not None else float(getattr(cfg.training, "teacher_gain_p", 0.0))
    )
    teacher_gain_d = (
        args.teacher_gain_d if args.teacher_gain_d is not None else float(getattr(cfg.training, "teacher_gain_d", 0.0))
    )
    teacher_weight = (
        args.teacher_weight if args.teacher_weight is not None else float(getattr(cfg.training, "teacher_weight", 0.0))
    )
    velocity_alignment_weight = (
        args.velocity_alignment_weight if args.velocity_alignment_weight is not None else float(getattr(cfg.training, "velocity_alignment_weight", 0.0))
    )
    desired_speed = (
        args.desired_speed if args.desired_speed is not None else float(getattr(cfg.training, "desired_speed", 1.0))
    )
    final_velocity_weight = (
        args.final_velocity_weight if args.final_velocity_weight is not None else float(getattr(cfg.training, "final_velocity_weight", 0.0))
    )
    final_distance_weight = (
        args.final_distance_weight if args.final_distance_weight is not None else float(getattr(cfg.training, "final_distance_weight", 0.0))
    )
    distance_bonus_weight = (
        args.distance_bonus_weight if args.distance_bonus_weight is not None else float(getattr(cfg.training, "distance_bonus_weight", 0.0))
    )
    distance_bonus_threshold = (
        args.distance_bonus_threshold if args.distance_bonus_threshold is not None else float(getattr(cfg.training, "distance_bonus_threshold", 0.0))
    )
    initial_xy_range = float(getattr(cfg.training, "initial_xy_range", 0.5))
    initial_z_range = tuple(getattr(cfg.training, "initial_z_range", (0.8, 1.2)))
    trajectory_projection_weight = (
        args.trajectory_projection_weight if args.trajectory_projection_weight is not None else float(getattr(cfg.training, "trajectory_projection_weight", 0.0))
    )
    distance_tracking_weight = (
        args.distance_tracking_weight if args.distance_tracking_weight is not None else float(getattr(cfg.training, "distance_tracking_weight", 0.0))
    )
    if args.teacher_force_schedule:
        teacher_force_schedule = parse_float_tuple(args.teacher_force_schedule)
    else:
        teacher_force_schedule = tuple(getattr(cfg.training, "teacher_force_schedule", ()))

    def _normalize_schedule(schedule: Tuple[float, ...]) -> Tuple[float, ...] | None:
        if stage_count > 0 and schedule:
            if len(schedule) < stage_count:
                schedule = schedule + (schedule[-1],) * (stage_count - len(schedule))
            return schedule
        return None

    def _normalize_schedule_float(schedule: Tuple[float, ...]) -> Tuple[float, ...] | None:
        if stage_count > 0 and schedule:
            if len(schedule) < stage_count:
                schedule = schedule + (schedule[-1],) * (stage_count - len(schedule))
            return schedule
        return None

    hard_nan_schedule = _normalize_schedule_float(hard_nan_schedule)
    hard_relax_schedule = _normalize_schedule_float(hard_relax_schedule)
    hard_qp_schedule = _normalize_schedule_float(hard_qp_schedule)
    relax_scale_schedule = _normalize_schedule_float(relax_scale_schedule)
    solver_scale_schedule = _normalize_schedule_float(solver_scale_schedule)
    teacher_force_schedule = _normalize_schedule_float(teacher_force_schedule)
    target_distance_schedule = _normalize_schedule_float(target_distance_schedule)

    config = TrainingConfig(
        horizon=horizon,
        policy_lr=policy_lr,
        cbf_lr=cbf_lr,
        safety_weight=safety_weight,
        solver_violation_weight=solver_weight,
        relaxation_weight=relaxation_weight,
        relaxation_usage_weight=relaxation_usage_weight,
        cbf_safe_weight=float(cfg.training.loss_safety_coef),
        cbf_unsafe_weight=float(cfg.training.loss_cbf_coef),
        cbf_hdot_weight=float(cfg.training.loss_collision_coef),
        cbf_margin=float(cfg.gcbf.gamma),
        cbf_value_weight=float(cfg.training.loss_cbf_coef),
        cbf_weight_scale=5.0,
        gradient_decay=grad_decay,
        grad_clip=grad_clip,
        seed=seed,
        episodes=episodes,
        curriculum_enabled=curriculum_enabled,
        stage_steps=stage_steps,
        noise_levels=noise_levels,
        use_safety=not args.disable_safety,
        target_position=jnp.array(target_tuple, dtype=jnp.float32),
        policy_freeze_steps=policy_freeze_steps,
        policy_l2_weight=1e-4,
        cbf_blend_alpha=cbf_blend_alpha,
        augment_point_cloud=augment_point_cloud,
        success_threshold=success_threshold,
        success_tolerance=success_tolerance,
        policy_distill_weight=policy_distill_weight,
        cbf_blend_levels=blend_levels,
        augment_levels=augment_levels,
        success_eval_schedule=success_eval_schedule,
        success_eval_noise_schedule=success_eval_noise_schedule,
        success_eval_random_pc_schedule=success_eval_random_pc_schedule,
        violation_threshold=violation_threshold,
        violation_threshold_schedule=violation_schedule,
        robust_eval_frequency=robust_eval_frequency,
        robust_eval_trials=robust_eval_trials,
        robust_eval_noise_levels=robust_eval_noise_levels,
        robust_eval_random_pc=robust_eval_random_pc,
        relax_penalty_boost=relax_boost,
        relax_penalty_max=relax_max,
        solver_penalty_boost=solver_boost,
        solver_penalty_max=solver_max,
        blend_backoff=blend_backoff,
        blend_min=blend_min,
        relaxation_alert=relax_alert,
        hard_nan_rate=hard_nan_rate,
        hard_relaxation_exceed_rate=hard_relax_rate,
        hard_qp_fail_rate=hard_qp_fail_rate,
        point_cloud_modes=point_cloud_modes,
        hard_abort_warmup_episodes=hard_warmup,
        hard_nan_schedule=hard_nan_schedule,
        hard_relax_schedule=hard_relax_schedule,
        hard_qp_fail_schedule=hard_qp_schedule,
        relaxation_scale_schedule=relax_scale_schedule,
        solver_scale_schedule=solver_scale_schedule,
        target_distance_schedule=target_distance_schedule,
        teacher_gain_p=teacher_gain_p,
        teacher_gain_d=teacher_gain_d,
        teacher_weight=teacher_weight,
        velocity_alignment_weight=velocity_alignment_weight,
        desired_speed=desired_speed,
        final_velocity_weight=final_velocity_weight,
        distance_bonus_weight=distance_bonus_weight,
        distance_bonus_threshold=distance_bonus_threshold,
        initial_xy_range=initial_xy_range,
        initial_z_range=initial_z_range,
        trajectory_projection_weight=trajectory_projection_weight,
        distance_tracking_weight=distance_tracking_weight,
        teacher_force_schedule=teacher_force_schedule,
        final_distance_weight=final_distance_weight,
    )

    return config


def main(
    config: TrainingConfig,
    base_cfg,
    use_rnn: bool,
    output_dir: str | None = None,
    cbf_params_path: str | None = None,
    policy_params_path: str | None = None,
    distill_policy_path: str | None = None,
):
    cfg = base_cfg

    physics_params = PhysicsParams(
        dt=float(cfg.physics.dt),
        max_acceleration=float(cfg.physics.control.max_thrust),
        gradient_decay=config.gradient_decay,
    )
    policy_hidden = tuple(cfg.policy.hidden_dims)
    policy_config = PolicyConfig(
        hidden_dims=policy_hidden,
        activation=cfg.policy.activation,
        output_dim=int(cfg.policy.output_dim),
        use_rnn=use_rnn or bool(cfg.policy.use_rnn),
        rnn_hidden_size=int(cfg.policy.rnn_hidden_size),
        action_limit=float(cfg.policy.action_limit),
    )
    policy_net = PolicyNetwork(policy_config)
    policy_params, policy_state = initialise_policy_params(
        jax.random.PRNGKey(config.seed),
        policy_config,
        observation_dim=10,
    )
    if policy_params_path:
        with Path(policy_params_path).open("rb") as fh:
            loaded_policy = pickle.load(fh)
        policy_params = _ensure_param_dict(loaded_policy)
    distill_policy_params = None
    if distill_policy_path:
        with Path(distill_policy_path).open("rb") as fh:
            distill_policy_params = pickle.load(fh)
        distill_policy_params = _ensure_param_dict(distill_policy_params)

    max_points = int(cfg.gcbf.max_neighbors)
    graph_config = GraphConfig(
        max_points=max_points,
        max_distance=float(cfg.gcbf.sensing_radius * 2.0),
        k_neighbors=int(cfg.gcbf.k_neighbors),
    )
    base_point_cloud = make_ring_point_cloud(
        graph_config.max_points,
        radius=float(cfg.env.area_size / 2.0),
        height=float(cfg.physics.drone.radius + 1.0),
    )
    dummy_state = create_initial_state()
    dummy_graph, _ = build_graph_from_point_cloud(
        dummy_state, base_point_cloud, graph_config
    )

    if config.use_safety:
        # 以解析 CBF 起步，可选加载预训练的神经网络参数。
        if cbf_params_path:
            with Path(cbf_params_path).open("rb") as fh:
                cbf_payload = pickle.load(fh)
            if isinstance(cbf_payload, dict):
                cbf_params = _ensure_param_dict(cbf_payload)
                loaded_graph_cfg = cbf_payload.get("graph_config")
            else:
                cbf_params = cbf_payload
                loaded_graph_cfg = None
            if loaded_graph_cfg is not None:
                graph_config = loaded_graph_cfg
                base_point_cloud = make_ring_point_cloud(
                    graph_config.max_points,
                    radius=float(cfg.env.area_size / 2.0),
                    height=float(cfg.physics.drone.radius + 1.0),
                )
                dummy_graph, _ = build_graph_from_point_cloud(
                    dummy_state, base_point_cloud, graph_config
                )
            if not jax.tree_util.tree_leaves(cbf_params):
                cbf_params = initialise_cbf_params(
                    jax.random.PRNGKey(config.seed + 1), dummy_graph
                )
        else:
            cbf_params = initialise_cbf_params(
                jax.random.PRNGKey(config.seed + 1), dummy_graph
            )
        safety_config = SafetyConfig(
            alpha0=float(cfg.safety.alpha0),
            alpha1=float(cfg.safety.alpha1),
            max_acceleration=float(cfg.safety.max_acceleration),
            relaxation_penalty=float(cfg.safety.relaxation_penalty),
            max_relaxation=float(cfg.safety.max_relaxation),
            tolerance=float(cfg.safety.violation_tolerance),
            relaxation_alert=float(getattr(cfg.safety, "relaxation_alert", config.relaxation_alert)),
        )
        params = {"policy": policy_params, "cbf": cbf_params}
    else:
        safety_config = None
        params = {"policy": policy_params}
    optimizer_policy = optax.chain(
        optax.clip_by_global_norm(config.grad_clip),
        optax.add_decayed_weights(config.policy_l2_weight),
        optax.adam(config.policy_lr),
    )
    opt_state_policy = optimizer_policy.init(policy_params)
    if config.use_safety:
        optimizer_cbf = optax.chain(
            optax.clip_by_global_norm(config.grad_clip),
            optax.adam(config.cbf_lr),
        )
        opt_state_cbf = optimizer_cbf.init(params["cbf"])
    else:
        optimizer_cbf = None
        opt_state_cbf = None
    train_step = make_train_step(
        config,
        physics_params,
        policy_net,
        policy_state,
        base_point_cloud,
        graph_config,
        safety_config,
        optimizer_policy,
        optimizer_cbf,
        distill_policy_params,
    )

    stage_count_runtime = len(config.stage_steps) if config.curriculum_enabled and len(config.stage_steps) > 0 else 1
    stage_count_runtime = max(stage_count_runtime, 1)
    blend_overrides = list(config.cbf_blend_levels if config.cbf_blend_levels else (config.cbf_blend_alpha,))
    if len(blend_overrides) < stage_count_runtime:
        blend_overrides.extend([blend_overrides[-1]] * (stage_count_runtime - len(blend_overrides)))
    if config.relaxation_scale_schedule is not None:
        relaxation_scales = [float(config.relaxation_scale_schedule[min(i, len(config.relaxation_scale_schedule) - 1)]) for i in range(stage_count_runtime)]
    else:
        relaxation_scales = [1.0] * stage_count_runtime
    if config.solver_scale_schedule is not None:
        solver_scales = [float(config.solver_scale_schedule[min(i, len(config.solver_scale_schedule) - 1)]) for i in range(stage_count_runtime)]
    else:
        solver_scales = [1.0] * stage_count_runtime
    point_cloud_modes_runtime = list(config.point_cloud_modes if config.point_cloud_modes else ("ring",))
    if len(point_cloud_modes_runtime) < stage_count_runtime:
        point_cloud_modes_runtime.extend([point_cloud_modes_runtime[-1]] * (stage_count_runtime - len(point_cloud_modes_runtime)))
    point_mode_indices = {mode: idx for idx, mode in enumerate(sorted(set(point_cloud_modes_runtime)))}
    base_target = jnp.asarray(config.target_position)
    base_xy = base_target[:2]
    base_xy_norm = jnp.linalg.norm(base_xy) + 1e-6
    base_xy_dir = base_xy / base_xy_norm
    base_z = float(base_target[2])
    default_distance = float(base_xy_norm)
    if config.target_distance_schedule is not None and len(config.target_distance_schedule) > 0:
        target_distances_runtime = [float(config.target_distance_schedule[min(i, len(config.target_distance_schedule) - 1)]) for i in range(stage_count_runtime)]
    else:
        target_distances_runtime = [default_distance] * stage_count_runtime
    final_target_distance = target_distances_runtime[-1]
    if config.teacher_force_schedule is not None and len(config.teacher_force_schedule) > 0:
        teacher_force_runtime = [float(config.teacher_force_schedule[min(i, len(config.teacher_force_schedule) - 1)]) for i in range(stage_count_runtime)]
    else:
        teacher_force_runtime = [0.0] * stage_count_runtime
    final_teacher_force = teacher_force_runtime[-1]

    final_blend = config.cbf_blend_levels[-1] if config.cbf_blend_levels else config.cbf_blend_alpha
    final_augment = bool(config.augment_levels[-1]) if config.augment_levels else config.augment_point_cloud
    final_point_mode = point_cloud_modes_runtime[-1]

    @functools.partial(jax.jit, static_argnums=(2, 3))
    def evaluate_success(
        params,
        rng_key,
        stage_idx: int,
        point_cloud_mode: str,
        target_position: jnp.ndarray,
    ):
        keys = jax.random.split(rng_key, config.success_eval_trials)
        if config.success_eval_schedule:
            idx = min(len(config.success_eval_schedule) - 1, stage_idx)
            eval_blend = config.success_eval_schedule[idx]
        else:
            eval_blend = final_blend
        if config.success_eval_noise_schedule:
            idx_noise = min(len(config.success_eval_noise_schedule) - 1, stage_idx)
            eval_noise = config.success_eval_noise_schedule[idx_noise]
        else:
            eval_noise = config.success_eval_noise
        if config.success_eval_random_pc_schedule:
            idx_pc = min(len(config.success_eval_random_pc_schedule) - 1, stage_idx)
            eval_random_pc = bool(config.success_eval_random_pc_schedule[idx_pc])
        else:
            eval_random_pc = config.success_eval_random_pc

        def rollout(key):
            key_init, key_cloud, key_rollout = jax.random.split(key, 3)
            init_state = sample_initial_state(
                key_init,
                xy_range=config.initial_xy_range,
                z_min=float(config.initial_z_range[0]),
                z_max=float(config.initial_z_range[1]),
            )
            point_cloud, key_cloud = build_point_cloud_for_mode(
                point_cloud_mode, base_point_cloud, graph_config, key_cloud
            )
            if eval_random_pc or final_augment:
                point_cloud, key_cloud = sample_augmented_point_cloud(
                    point_cloud, graph_config, key_cloud
                )
            _, outputs = execute_rollout(
                params=params,
                policy_net=policy_net,
                policy_state=policy_state,
                initial_state=init_state,
                physics_params=physics_params,
                point_cloud=point_cloud,
                graph_config=graph_config,
                safety_config=safety_config,
                target_position=target_position,
                horizon=config.horizon,
                gradient_decay=config.gradient_decay,
                rng=key_rollout,
                noise_scale=eval_noise,
                cbf_blend_alpha=eval_blend,
            )
            final_pos = outputs.position[-1]
            dist = jnp.linalg.norm(final_pos - target_position)
            # 这里使用可配置的距离阈值，便于在不同平台上同步评估标准。
            success = dist < config.success_tolerance
            relax = jnp.mean(outputs.relaxation)
            violation = jnp.max(outputs.constraint_violation)
            return success.astype(jnp.float32), relax, violation

        successes, relaxes, violations = jax.vmap(rollout)(keys)
        return jnp.mean(successes), jnp.mean(relaxes), jnp.max(violations)

    best_params = jax.tree_util.tree_map(lambda x: x, params)
    best_success = -1.0
    bad_eval_count = 0
    robust_noise_levels = tuple(float(x) for x in config.robust_eval_noise_levels)

    @functools.partial(jax.jit, static_argnums=(2, 4))
    def evaluate_robustness(
        params,
        rng_key,
        noise_levels: Tuple[float, ...],
        blend_alpha: float,
        point_cloud_mode: str,
        target_position: jnp.ndarray,
    ):
        noise_array = jnp.array(noise_levels, dtype=jnp.float32)
        keys = jax.random.split(rng_key, noise_array.shape[0])

        def single(noise, key):
            rollout_keys = jax.random.split(key, config.robust_eval_trials)

            def rollout_eval(inner_key):
                key_init, key_cloud, key_rollout = jax.random.split(inner_key, 3)
                init_state = sample_initial_state(
                    key_init,
                    xy_range=config.initial_xy_range,
                    z_min=float(config.initial_z_range[0]),
                    z_max=float(config.initial_z_range[1]),
                )
                point_cloud, key_cloud = build_point_cloud_for_mode(
                    point_cloud_mode, base_point_cloud, graph_config, key_cloud
                )
                if config.robust_eval_random_pc or final_augment:
                    point_cloud, key_cloud = sample_augmented_point_cloud(
                        point_cloud, graph_config, key_cloud
                    )
                _, outputs = execute_rollout(
                    params=params,
                    policy_net=policy_net,
                    policy_state=policy_state,
                    initial_state=init_state,
                    physics_params=physics_params,
                    point_cloud=point_cloud,
                    graph_config=graph_config,
                    safety_config=safety_config,
                    target_position=target_position,
                    horizon=config.horizon,
                    gradient_decay=config.gradient_decay,
                    rng=key_rollout,
                    noise_scale=jnp.asarray(noise, dtype=jnp.float32),
                    cbf_blend_alpha=blend_alpha,
                )
                success = jnp.linalg.norm(outputs.position[-1] - target_position) < config.success_tolerance
                relax = jnp.mean(outputs.relaxation)
                violation = jnp.max(outputs.constraint_violation)
                cbf_min = jnp.min(outputs.cbf_value)
                return success.astype(jnp.float32), relax, violation, cbf_min

            successes, relaxes, violations, cbf_mins = jax.vmap(rollout_eval)(rollout_keys)
            return (
                jnp.mean(successes),
                jnp.mean(relaxes),
                jnp.max(violations),
                jnp.min(cbf_mins),
            )

        return jax.vmap(single)(noise_array, keys)

    def stage_values(step: int) -> Tuple[float, float, bool, int, str, float, float]:
        if not config.curriculum_enabled or len(config.stage_steps) == 0:
            noise = config.noise_levels[-1] if len(config.noise_levels) else 0.0
            return noise, config.cbf_blend_alpha, config.augment_point_cloud, 0, point_cloud_modes_runtime[0], target_distances_runtime[0], teacher_force_runtime[0]
        cumulative = 0
        for idx, span in enumerate(config.stage_steps):
            cumulative += span
            if step < cumulative:
                noise_idx = min(idx, len(config.noise_levels) - 1)
                blend_idx = min(idx, len(blend_overrides) - 1)
                augment_idx = min(idx, len(config.augment_levels) - 1)
                mode_idx = min(idx, len(point_cloud_modes_runtime) - 1)
                target_idx = min(idx, len(target_distances_runtime) - 1)
                teacher_idx = min(idx, len(teacher_force_runtime) - 1)
                return (
                    config.noise_levels[noise_idx],
                    blend_overrides[blend_idx],
                    bool(config.augment_levels[augment_idx]),
                    idx,
                    point_cloud_modes_runtime[mode_idx],
                    target_distances_runtime[target_idx],
                    teacher_force_runtime[teacher_idx],
                )
        last_idx = len(config.stage_steps) - 1
        return (
            config.noise_levels[-1],
            blend_overrides[-1],
            bool(config.augment_levels[-1]),
            last_idx,
            point_cloud_modes_runtime[-1],
            target_distances_runtime[-1],
            teacher_force_runtime[-1],
        )

    rng = jax.random.PRNGKey(config.seed + 2)
    history = []
    for step in range(config.episodes):
        rng, step_key = jax.random.split(rng)
        noise_scale, current_blend, current_augment, current_stage_idx, current_point_mode, current_target_distance, current_teacher_force = stage_values(step)
        current_stage_idx = int(current_stage_idx)
        current_stage_idx = max(0, min(current_stage_idx, stage_count_runtime - 1))
        current_relax_scale = relaxation_scales[current_stage_idx]
        current_solver_scale = solver_scales[current_stage_idx]
        current_target_xy = base_xy_dir * current_target_distance
        current_target = jnp.array([current_target_xy[0], current_target_xy[1], base_z], dtype=jnp.float32)
        if config.violation_threshold_schedule:
            idx_v = min(len(config.violation_threshold_schedule) - 1, current_stage_idx)
            stage_violation_threshold = config.violation_threshold_schedule[idx_v]
        else:
            stage_violation_threshold = config.violation_threshold

        params, opt_state_policy, opt_state_cbf, metrics = train_step(
            params,
            opt_state_policy,
            opt_state_cbf,
            step_key,
            noise_scale,
            current_blend,
            point_cloud_mode=current_point_mode,
            augment_flag=current_augment,
            target_position=current_target,
            step_index=step,
            relaxation_scale=current_relax_scale,
            solver_scale=current_solver_scale,
            teacher_blend=current_teacher_force,
        )

        metrics["curriculum/blend_alpha"] = current_blend
        metrics["curriculum/augment"] = float(current_augment)
        metrics["curriculum/point_mode_id"] = jnp.array(point_mode_indices.get(current_point_mode, 0), dtype=jnp.float32)
        metrics["adaptive/relax_scale"] = jnp.array(current_relax_scale, dtype=jnp.float32)
        metrics["adaptive/solver_scale"] = jnp.array(current_solver_scale, dtype=jnp.float32)
        metrics["adaptive/blend_alpha_effective"] = jnp.array(current_blend, dtype=jnp.float32)
        metrics["adaptive/blend_backoff"] = jnp.array(0.0, dtype=jnp.float32)
        metrics["adaptive/penalty_boost"] = jnp.array(0.0, dtype=jnp.float32)
        metrics["adaptive/relax_alert"] = jnp.array(0.0, dtype=jnp.float32)
        metrics["adaptive/relax_scale_next"] = jnp.array(current_relax_scale, dtype=jnp.float32)
        metrics["adaptive/solver_scale_next"] = jnp.array(current_solver_scale, dtype=jnp.float32)
        metrics["adaptive/blend_alpha_next"] = jnp.array(blend_overrides[current_stage_idx], dtype=jnp.float32)
        metrics["adaptive/hard_abort"] = jnp.array(0.0, dtype=jnp.float32)
        metrics["eval/violation_threshold"] = jnp.array(stage_violation_threshold, dtype=jnp.float32)
        metrics["eval/violation_exceeded"] = 0.0
        metrics["eval/rollback"] = 0.0

        relax_alert_triggered = float(metrics["safety/relaxation_mean"]) > config.relaxation_alert
        if relax_alert_triggered:
            metrics["adaptive/relax_alert"] = jnp.array(1.0, dtype=jnp.float32)

        adaptive_triggered = relax_alert_triggered
        robust_triggered = False
        robust_peak_violation = 0.0
        eval_success = None
        violation_exceeded = False

        if (step + 1) % config.success_eval_frequency == 0:
            rng, eval_key = jax.random.split(rng)
            eval_success, eval_relax, eval_violation = map(
                float,
                evaluate_success(
                    params,
                    eval_key,
                    current_stage_idx,
                    current_point_mode,
                    current_target,
                ),
            )
            metrics["eval/success_rate"] = eval_success
            metrics["eval/relax_mean"] = eval_relax
            metrics["eval/max_violation"] = eval_violation
            # 这里额外打印一次评估结果，方便在 macOS、Windows 或 Kaggle 上快速对比训练状态。
            print(
                f"[eval] step={step + 1} stage={current_stage_idx} "
                f"success={eval_success:.3f} max_violation={eval_violation:.3f} "
                f"relax_mean={eval_relax:.3e}"
            )
            violation_exceeded = eval_violation > stage_violation_threshold
            if violation_exceeded:
                eval_success = 0.0
                metrics["eval/violation_exceeded"] = 1.0
                # 当约束违约超标时立刻提醒，便于我们针对性调整课程或参数。
                print(
                    f"[warn] 约束跨越阈值，当前阈值={stage_violation_threshold:.2f} "
                    f"实际最大违约={eval_violation:.3f}"
                )
            else:
                metrics["eval/violation_exceeded"] = 0.0
            if eval_relax > config.relaxation_alert:
                eval_success = 0.0
                metrics["eval/relaxation_alert"] = 1.0
                print(
                    f"[warn] 松弛均值超出阈值 {config.relaxation_alert:.3f}, 当前 {eval_relax:.3f}"
                )
            else:
                metrics["eval/relaxation_alert"] = 0.0
            adaptive_triggered = adaptive_triggered or violation_exceeded
            if eval_success >= config.success_threshold:
                best_params = jax.tree_util.tree_map(lambda x: x, params)
                best_success = eval_success
                bad_eval_count = 0
            else:
                bad_eval_count += 1
                if best_success >= 0.0 and bad_eval_count >= config.success_patience:
                    params = jax.tree_util.tree_map(lambda x: x, best_params)
                    metrics["eval/rollback"] = 1.0
                    bad_eval_count = 0
                else:
                    metrics["eval/rollback"] = 0.0

        if (
            config.use_safety
            and config.robust_eval_frequency > 0
            and (step + 1) % config.robust_eval_frequency == 0
        ):
            rng, robust_key = jax.random.split(rng)
            blend_results = evaluate_robustness(
                params,
                robust_key,
                robust_noise_levels,
                current_blend,
                current_point_mode,
                current_target,
            )
            rng, robust_key_neural = jax.random.split(rng)
            neural_results = evaluate_robustness(
                params,
                robust_key_neural,
                robust_noise_levels,
                1.0,
                current_point_mode,
                current_target,
            )
            blend_results = jnp.asarray(blend_results)
            neural_results = jnp.asarray(neural_results)

            for idx_noise in range(blend_results.shape[0]):
                metrics[f"robust/blend_success_{idx_noise}"] = float(blend_results[idx_noise, 0])
                metrics[f"robust/blend_relax_{idx_noise}"] = float(blend_results[idx_noise, 1])
                metrics[f"robust/blend_violation_{idx_noise}"] = float(blend_results[idx_noise, 2])
                metrics[f"robust/neural_success_{idx_noise}"] = float(neural_results[idx_noise, 0])
                metrics[f"robust/neural_violation_{idx_noise}"] = float(neural_results[idx_noise, 2])

            robust_peak_violation = float(jnp.max(neural_results[:, 2]))
            metrics["robust/neural_max_violation"] = robust_peak_violation
            metrics["robust/blend_max_violation"] = float(jnp.max(blend_results[:, 2]))
            robust_triggered = robust_peak_violation > stage_violation_threshold
            metrics["robust/triggered"] = 1.0 if robust_triggered else 0.0
        else:
            metrics["robust/triggered"] = 0.0
            metrics["robust/neural_max_violation"] = 0.0
            metrics["robust/blend_max_violation"] = 0.0

        if adaptive_triggered or robust_triggered:
            new_relax = min(
                relaxation_scales[current_stage_idx] * config.relax_penalty_boost,
                config.relax_penalty_max,
            )
            new_solver = min(
                solver_scales[current_stage_idx] * config.solver_penalty_boost,
                config.solver_penalty_max,
            )
            relaxation_scales[current_stage_idx] = new_relax
            solver_scales[current_stage_idx] = new_solver
            metrics["adaptive/penalty_boost"] = jnp.array(1.0, dtype=jnp.float32)
            new_blend = max(
                config.blend_min,
                blend_overrides[current_stage_idx] - config.blend_backoff,
            )
            if new_blend < blend_overrides[current_stage_idx]:
                blend_overrides[current_stage_idx] = new_blend
                metrics["adaptive/blend_backoff"] = jnp.array(1.0, dtype=jnp.float32)
            metrics["adaptive/relax_scale_next"] = jnp.array(new_relax, dtype=jnp.float32)
            metrics["adaptive/solver_scale_next"] = jnp.array(new_solver, dtype=jnp.float32)
            metrics["adaptive/blend_alpha_next"] = jnp.array(blend_overrides[current_stage_idx], dtype=jnp.float32)

        metrics["adaptive/robust_violation_peak"] = jnp.array(robust_peak_violation, dtype=jnp.float32)

        hard_abort = False
        nan_rate = 0.0
        relax_exceed_rate = 0.0
        qp_fail_rate = float(metrics.get("safety/qp_fail_rate", 0.0))
        base_nan_threshold = (
            config.hard_nan_schedule[current_stage_idx]
            if (config.hard_nan_schedule is not None and len(config.hard_nan_schedule) > current_stage_idx)
            else config.hard_nan_rate
        )
        base_relax_threshold = (
            config.hard_relax_schedule[current_stage_idx]
            if (config.hard_relax_schedule is not None and len(config.hard_relax_schedule) > current_stage_idx)
            else config.hard_relaxation_exceed_rate
        )
        base_qp_threshold = (
            config.hard_qp_fail_schedule[current_stage_idx]
            if (config.hard_qp_fail_schedule is not None and len(config.hard_qp_fail_schedule) > current_stage_idx)
            else config.hard_qp_fail_rate
        )
        warmup_active = (
            config.hard_abort_warmup_episodes > 0
            and step < config.hard_abort_warmup_episodes
        )

        def _effective_threshold(base: float) -> float:
            if base < 0.0:
                return float("inf")
            if warmup_active:
                return 1.0
            return base

        nan_threshold = _effective_threshold(base_nan_threshold)
        relax_threshold = _effective_threshold(base_relax_threshold)
        qp_fail_threshold = _effective_threshold(base_qp_threshold)

        if config.use_safety:
            nan_rate = float(metrics.get("safety/qp_nan_rate", 0.0))
            relax_exceed_rate = float(metrics.get("safety/relaxation_exceeded_rate", 0.0))
            hard_abort = (
                (nan_rate > nan_threshold)
                or (relax_exceed_rate > relax_threshold)
                or (qp_fail_rate > qp_fail_threshold)
            )

        metrics["adaptive/hard_threshold_nan"] = jnp.array(nan_threshold, dtype=jnp.float32)
        metrics["adaptive/hard_threshold_relax"] = jnp.array(relax_threshold, dtype=jnp.float32)
        metrics["adaptive/hard_threshold_qp_fail"] = jnp.array(qp_fail_threshold, dtype=jnp.float32)
        metrics["adaptive/hard_threshold_base_nan"] = jnp.array(base_nan_threshold, dtype=jnp.float32)
        metrics["adaptive/hard_threshold_base_relax"] = jnp.array(base_relax_threshold, dtype=jnp.float32)
        metrics["adaptive/hard_threshold_base_qp_fail"] = jnp.array(base_qp_threshold, dtype=jnp.float32)
        metrics["adaptive/hard_warmup"] = jnp.array(1.0 if warmup_active else 0.0, dtype=jnp.float32)

        if hard_abort:
            metrics["adaptive/hard_abort"] = jnp.array(1.0, dtype=jnp.float32)
            print(
                "[abort] 检测到安全诊断指标超限: "
                f"nan_rate={nan_rate:.3f}, relax_exceed={relax_exceed_rate:.3f}, qp_fail={qp_fail_rate:.3f}. "
                "回退至最佳参数并终止训练。"
            )
            params = jax.tree_util.tree_map(lambda x: x, best_params)

        history.append(
            {
                k: float(v) if jnp.ndim(v) == 0 else [float(x) for x in jnp.atleast_1d(v)]
                for k, v in metrics.items()
            }
        )
        if hard_abort:
            break
        if step % 10 == 0 or step == config.episodes - 1:
            print(
                f"[{step:04d}] total={metrics['loss/total']:.4f} "
                f"eff={metrics['loss/efficiency']:.4f} "
                f"soft={metrics['loss/safety_soft']:.4f} "
                f"violation={metrics['loss/constraint_violation']:.4f} "
                f"relax_mean={metrics['safety/relaxation_mean']:.4f} "
                f"cbf_min={metrics['safety/cbf_min']:.4f} "
                f"cbf_safe={metrics['loss/cbf_safe']:.4f} "
                f"cbf_hdot={metrics['loss/cbf_hdot']:.4f} "
                f"noise={metrics['curriculum/noise_scale']:.4f} "
                f"grad_pi={metrics['loss/grad_norm_policy']:.4f} "
                f"grad_cbf={metrics['loss/grad_norm_cbf']:.4f}"
            )

    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": asdict(config),
            "params_policy": params["policy"],
            "history": history,
        }
        if config.use_safety and "cbf" in params:
            payload["params_cbf"] = params["cbf"]
        with open(out_path / "training_results.pkl", "wb") as fh:
            pickle.dump(payload, fh)
        print(f"Saved training artifacts to {out_path}")


if __name__ == "__main__":
    args = parse_args()
    base_cfg = load_base_config(args.config_name)
    cfg = build_config(args, base_cfg)
    main(
        cfg,
        base_cfg,
        use_rnn=args.use_rnn,
        output_dir=args.output_dir,
        cbf_params_path=args.cbf_params,
        policy_params_path=args.policy_params,
    )
