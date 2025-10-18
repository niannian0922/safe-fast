"""
Entry-point training script for the refactored safe agile flight pipeline.

This script wires together the core modules:

- `core.physics` for differentiable point-mass dynamics;
- `core.policy` for the control policy (MLP/GRU);
- `core.perception` for the GNN-based CBF estimator;
- `core.safety` for the differentiable CBF-QP safety filter;
- `core.simple_training` for efficiency-oriented loss shaping.

The goal is to provide a compact yet complete baseline that can be iterated on
for research experimentation.
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
from configs import default_config
import optax


# ---------------------------------------------------------------------------
# Configurations
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


def _ensure_param_dict(obj):
    if isinstance(obj, dict) and "params" in obj:
        if set(obj.keys()) == {"params"}:
            return obj
        return {"params": obj["params"]}
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


def sample_augmented_point_cloud(
    base_cloud: jnp.ndarray,
    graph_config: GraphConfig,
    rng: jax.Array,
    jitter_scale: float = 0.3,
    replace_prob: float = 0.3,
) -> jnp.ndarray:
    rng_jitter, rng_replace, rng_new = jax.random.split(rng, 3)
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
    return augmented


# ---------------------------------------------------------------------------
# Rollout data structures
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------


def sample_initial_state(rng: jax.Array) -> DroneState:
    pos = jax.random.uniform(rng, (3,), minval=-0.5, maxval=0.5)
    pos = pos.at[2].set(jax.random.uniform(rng, (), minval=0.8, maxval=1.2))
    return create_initial_state(position=pos)


# ---------------------------------------------------------------------------
# Rollout logic
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
    augment_flag: bool,
    relaxation_weight_scale: float = 1.0,
    solver_weight_scale: float = 1.0,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    key_init, rollout_rng = jax.random.split(rng)

    def do_augment(key):
        cloud_rng, next_rng = jax.random.split(key)
        augmented = sample_augmented_point_cloud(
            base_point_cloud, graph_config, cloud_rng
        )
        return augmented, next_rng

    def no_augment(key):
        return base_point_cloud, key

    point_cloud, rollout_rng = jax.lax.cond(
        augment_flag,
        do_augment,
        no_augment,
        rollout_rng,
    )
    init_state = sample_initial_state(key_init)

    _, rollout = execute_rollout(
        params=params,
        policy_net=policy_net,
        policy_state=policy_state,
        initial_state=init_state,
        physics_params=physics_params,
        point_cloud=point_cloud,
        graph_config=graph_config,
        safety_config=safety_config,
        target_position=config.target_position,
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
        trajectory, config.target_position, DEFAULT_EFFICIENCY_CONFIG
    )
    safety_penalty = jnp.mean(jnp.nan_to_num(rollout.soft_violation))
    violation_penalty = jnp.mean(jnp.nan_to_num(rollout.constraint_violation))
    relaxation_cost = jnp.mean(jnp.nan_to_num(rollout.relaxation))
    relaxation_rate = jnp.mean(jnp.nan_to_num(rollout.relaxation_active))
    cbf_mean = jnp.mean(cbf_values)
    cbf_min = jnp.min(cbf_values)
    distill_loss = jnp.array(0.0, dtype=jnp.float32)
    if baseline_policy_params is not None and config.policy_distill_weight > 0.0:
        target_offsets = config.target_position - positions
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

    metrics = {
        "loss/total": total_loss,
        "loss/efficiency": efficiency_loss,
        "loss/safety_soft": safety_penalty,
        "loss/constraint_violation": violation_penalty,
        "loss/distill": distill_loss,
        "safety/relaxation_mean": relaxation_cost,
        "safety/relaxation_rate": relaxation_rate,
        "safety/cbf_mean": cbf_mean,
        "safety/cbf_min": cbf_min,
        "loss/cbf_safe": cbf_safe_loss,
        "loss/cbf_unsafe": cbf_unsafe_loss,
        "loss/cbf_hdot": cbf_hdot_loss,
        "loss/cbf_value": cbf_value_loss,
        "cbf/safe_fraction": cbf_safe_fraction,
        "cbf/unsafe_fraction": cbf_unsafe_fraction,
        "cbf/analytic_mean": cbf_analytic_mean,
        "cbf/analytic_min": cbf_analytic_min,
        "curriculum/noise_scale": jnp.array(noise_scale, dtype=jnp.float32),
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

    @jax.jit
    def train_step(
        params: Dict[str, Dict],
        opt_state_policy: optax.OptState,
        opt_state_cbf: optax.OptState | None,
        rng: jax.Array,
        noise_scale: float,
        blend_alpha: float,
        augment_flag: bool,
        step_index: int,
        relaxation_scale: float,
        solver_scale: float,
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
                augment_flag,
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
# Main routine
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train safe policy with CBF-QP.")
    parser.add_argument("--episodes", type=int, default=None, help="Training episodes.")
    parser.add_argument("--horizon", type=int, default=None, help="Rollout horizon.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument(
        "--grad-decay",
        type=float,
        default=None,
        help="Temporal gradient decay factor.",
    )
    parser.add_argument(
        "--safety-weight",
        type=float,
        default=None,
        help="Weight for soft CBF penalty.",
    )
    parser.add_argument(
        "--solver-weight",
        type=float,
        default=None,
        help="Weight for constraint violation penalty.",
    )
    parser.add_argument(
        "--relaxation-weight",
        type=float,
        default=None,
        help="Weight for slack usage in the safety layer.",
    )
    parser.add_argument(
        "--relax-usage-weight",
        type=float,
        default=None,
        help="Weight for the average activation of the relaxation slack.",
    )
    parser.add_argument(
        "--policy-lr",
        type=float,
        default=None,
        help="Learning rate for the policy parameters.",
    )
    parser.add_argument(
        "--target-position",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Goal position in metres.",
    )
    parser.add_argument(
        "--use-rnn",
        action="store_true",
        help="Enable GRU-based policy instead of pure MLP.",
    )
    parser.add_argument(
        "--stage-steps",
        type=str,
        default=None,
        help="Comma-separated curriculum stage lengths (e.g. 300,400,300).",
    )
    parser.add_argument(
        "--noise-levels",
        type=str,
        default=None,
        help="Comma-separated noise std for each stage (e.g. 0.0,0.02,0.05).",
    )
    parser.add_argument(
        "--disable-curriculum",
        action="store_true",
        help="Disable staged curriculum (use last noise level only).",
    )
    parser.add_argument(
        "--disable-safety",
        action="store_true",
        help="Bypass CBF/QP safety layer (policy output is used directly).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional directory to store final parameters and training history.",
    )
    parser.add_argument(
        "--cbf-params",
        type=str,
        default=None,
        help="Optional pickle file containing pretrained CBF parameters.",
    )
    parser.add_argument(
        "--policy-params",
        type=str,
        default=None,
        help="Optional pickle file containing pretrained policy parameters.",
    )
    parser.add_argument(
        "--policy-freeze-steps",
        type=int,
        default=None,
        help="Number of initial episodes to freeze policy updates.",
    )
    parser.add_argument(
        "--cbf-blend-alpha",
        type=float,
        default=None,
        help="Blend factor between neural and analytic CBF (1.0 = neural).",
    )
    parser.add_argument(
        "--disable-pointcloud-augment",
        action="store_true",
        help="Disable random point cloud augmentation during training.",
    )
    parser.add_argument(
        "--success-threshold",
        type=float,
        default=None,
        help="Success rate threshold for checkpointing/rollback.",
    )
    parser.add_argument(
        "--success-eval-schedule",
        type=str,
        default=None,
        help="Comma-separated blend factors for success evaluation (e.g. 0.0,0.5,1.0).",
    )
    parser.add_argument(
        "--success-eval-noise",
        type=str,
        default=None,
        help="Comma-separated noise values used during success eval per stage.",
    )
    parser.add_argument(
        "--success-eval-random-pc",
        type=str,
        default=None,
        help="Comma-separated 0/1 flags for random point cloud during success eval per stage.",
    )
    parser.add_argument(
        "--violation-threshold",
        type=float,
        default=None,
        help="Maximum allowed violation during eval before triggering rollback.",
    )
    parser.add_argument(
        "--violation-schedule",
        type=str,
        default=None,
        help="Comma-separated violation thresholds per stage.",
    )
    parser.add_argument(
        "--robust-eval-frequency",
        type=int,
        default=None,
        help="How often (episodes) to run robustness evaluation.",
    )
    parser.add_argument(
        "--robust-eval-trials",
        type=int,
        default=None,
        help="Number of rollouts per noise level in robustness evaluation.",
    )
    parser.add_argument(
        "--robust-eval-noise",
        type=str,
        default=None,
        help="Comma-separated noise levels for robustness stress testing.",
    )
    parser.add_argument(
        "--robust-eval-no-pc",
        action="store_true",
        help="Disable random point cloud augmentation during robustness evaluation.",
    )
    parser.add_argument(
        "--blend-backoff",
        type=float,
        default=None,
        help="Blend reduction applied when violations exceed thresholds.",
    )
    parser.add_argument(
        "--blend-min",
        type=float,
        default=None,
        help="Minimum allowed blend after automatic backoff.",
    )
    parser.add_argument(
        "--relax-boost",
        type=float,
        default=None,
        help="Multiplier applied to relaxation penalties when violations exceed thresholds.",
    )
    parser.add_argument(
        "--relax-max",
        type=float,
        default=None,
        help="Maximum relaxation penalty multiplier.",
    )
    parser.add_argument(
        "--solver-boost",
        type=float,
        default=None,
        help="Multiplier applied to solver violation penalties when thresholds are exceeded.",
    )
    parser.add_argument(
        "--solver-max",
        type=float,
        default=None,
        help="Maximum solver penalty multiplier.",
    )
    parser.add_argument(
        "--relax-alert",
        type=float,
        default=None,
        help="Relaxation mean threshold that triggers adaptive penalties even without eval.",
    )
    parser.add_argument(
        "--distill-policy",
        type=str,
        default=None,
        help="Optional policy parameters used for distillation regularization.",
    )
    parser.add_argument(
        "--distill-weight",
        type=float,
        default=None,
        help="Weight for policy distillation loss.",
    )
    parser.add_argument(
        "--blend-levels",
        type=str,
        default=None,
        help="Comma-separated blend factors matching stage steps.",
    )
    parser.add_argument(
        "--augment-levels",
        type=str,
        default=None,
        help="Comma-separated 0/1 flags for point cloud augmentation per stage.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> TrainingConfig:
    cfg = default_config.get_config()

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
            raise ValueError(f"Invalid stage_steps spec: {raw}") from exc
        if not values:
            raise ValueError("stage_steps must contain at least one integer")
        return values

    def parse_float_tuple(raw: str) -> Tuple[float, ...]:
        try:
            values = tuple(float(s.strip()) for s in raw.split(",") if s.strip())
        except ValueError as exc:  # pragma: no cover
            raise ValueError(f"Invalid noise_levels spec: {raw}") from exc
        if not values:
            raise ValueError("noise_levels must contain at least one float")
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
            raise ValueError(f"Invalid blend_levels spec: {raw}") from exc
        if not values:
            raise ValueError("blend_levels must contain at least one value")
        return values

    def parse_int_optional(raw: str) -> Tuple[int, ...]:
        try:
            values = tuple(int(s.strip()) for s in raw.split(",") if s.strip())
        except ValueError as exc:
            raise ValueError(f"Invalid augment_levels spec: {raw}") from exc
        if not values:
            raise ValueError("augment_levels must contain at least one value")
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
        raise ValueError("blend_levels must match stage_steps length")
    if len(augment_levels) != len(stage_steps):
        raise ValueError("augment_levels must match stage_steps length")

    policy_freeze_steps = (
        args.policy_freeze_steps
        if args.policy_freeze_steps is not None
        else int(cfg.training.curriculum.stage1_steps)
    )
    cbf_blend_alpha = (
        args.cbf_blend_alpha if args.cbf_blend_alpha is not None else 1.0
    )
    augment_point_cloud = not args.disable_pointcloud_augment
    success_threshold = (
        args.success_threshold if args.success_threshold is not None else 0.95
    )
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
    )

    return config


def main(
    config: TrainingConfig,
    use_rnn: bool,
    output_dir: str | None = None,
    cbf_params_path: str | None = None,
    policy_params_path: str | None = None,
    distill_policy_path: str | None = None,
):
    cfg = default_config.get_config()

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
        # Start with analytic CBF, optionally load pretrained neural parameters.
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
    relaxation_scales = [1.0] * stage_count_runtime
    solver_scales = [1.0] * stage_count_runtime

    final_blend = config.cbf_blend_levels[-1] if config.cbf_blend_levels else config.cbf_blend_alpha
    final_augment = bool(config.augment_levels[-1]) if config.augment_levels else config.augment_point_cloud

    @functools.partial(jax.jit, static_argnums=(2,))
    def evaluate_success(params, rng_key, stage_idx: int):
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
            key_init, key_cloud = jax.random.split(key)
            init_state = sample_initial_state(key_init)
            point_cloud = base_point_cloud
            if eval_random_pc or final_augment:
                point_cloud = sample_augmented_point_cloud(
                    base_point_cloud, graph_config, key_cloud
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
                target_position=config.target_position,
                horizon=config.horizon,
                gradient_decay=config.gradient_decay,
                rng=key,
                noise_scale=eval_noise,
                cbf_blend_alpha=eval_blend,
            )
            final_pos = outputs.position[-1]
            dist = jnp.linalg.norm(final_pos - config.target_position)
            success = dist < 0.1
            relax = jnp.mean(outputs.relaxation)
            violation = jnp.max(outputs.constraint_violation)
            return success.astype(jnp.float32), relax, violation

        successes, relaxes, violations = jax.vmap(rollout)(keys)
        return jnp.mean(successes), jnp.mean(relaxes), jnp.max(violations)

    best_params = jax.tree_util.tree_map(lambda x: x, params)
    best_success = -1.0
    bad_eval_count = 0
    robust_noise_levels = tuple(float(x) for x in config.robust_eval_noise_levels)

    @functools.partial(jax.jit, static_argnums=(2,))
    def evaluate_robustness(params, rng_key, noise_levels: Tuple[float, ...], blend_alpha: float):
        noise_array = jnp.array(noise_levels, dtype=jnp.float32)
        keys = jax.random.split(rng_key, noise_array.shape[0])

        def single(noise, key):
            rollout_keys = jax.random.split(key, config.robust_eval_trials)

            def rollout_eval(inner_key):
                key_init, key_cloud = jax.random.split(inner_key)
                init_state = sample_initial_state(key_init)
                point_cloud = base_point_cloud
                if config.robust_eval_random_pc or final_augment:
                    point_cloud = sample_augmented_point_cloud(
                        base_point_cloud, graph_config, key_cloud
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
                    target_position=config.target_position,
                    horizon=config.horizon,
                    gradient_decay=config.gradient_decay,
                    rng=inner_key,
                    noise_scale=noise,
                    cbf_blend_alpha=blend_alpha,
                )
                success = jnp.linalg.norm(outputs.position[-1] - config.target_position) < 0.1
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

    def stage_values(step: int) -> Tuple[float, float, bool, int]:
        if not config.curriculum_enabled or len(config.stage_steps) == 0:
            noise = config.noise_levels[-1] if len(config.noise_levels) else 0.0
            return noise, config.cbf_blend_alpha, config.augment_point_cloud, 0
        cumulative = 0
        for idx, span in enumerate(config.stage_steps):
            cumulative += span
            if step < cumulative:
                noise_idx = min(idx, len(config.noise_levels) - 1)
                blend_idx = min(idx, len(blend_overrides) - 1)
                augment_idx = min(idx, len(config.augment_levels) - 1)
                return (
                    config.noise_levels[noise_idx],
                    blend_overrides[blend_idx],
                    bool(config.augment_levels[augment_idx]),
                    idx,
                )
        last_idx = len(config.stage_steps) - 1
        return (
            config.noise_levels[-1],
            blend_overrides[-1],
            bool(config.augment_levels[-1]),
            last_idx,
        )

    rng = jax.random.PRNGKey(config.seed + 2)
    history = []
    for step in range(config.episodes):
        rng, step_key = jax.random.split(rng)
        noise_scale, current_blend, current_augment, current_stage_idx = stage_values(step)
        current_stage_idx = int(current_stage_idx)
        current_stage_idx = max(0, min(current_stage_idx, stage_count_runtime - 1))
        current_relax_scale = relaxation_scales[current_stage_idx]
        current_solver_scale = solver_scales[current_stage_idx]
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
            current_augment,
            step,
            current_relax_scale,
            current_solver_scale,
        )

        metrics["curriculum/blend_alpha"] = current_blend
        metrics["curriculum/augment"] = float(current_augment)
        metrics["adaptive/relax_scale"] = jnp.array(current_relax_scale, dtype=jnp.float32)
        metrics["adaptive/solver_scale"] = jnp.array(current_solver_scale, dtype=jnp.float32)
        metrics["adaptive/blend_alpha_effective"] = jnp.array(current_blend, dtype=jnp.float32)
        metrics["adaptive/blend_backoff"] = jnp.array(0.0, dtype=jnp.float32)
        metrics["adaptive/penalty_boost"] = jnp.array(0.0, dtype=jnp.float32)
        metrics["adaptive/relax_alert"] = jnp.array(0.0, dtype=jnp.float32)
        metrics["adaptive/relax_scale_next"] = jnp.array(current_relax_scale, dtype=jnp.float32)
        metrics["adaptive/solver_scale_next"] = jnp.array(current_solver_scale, dtype=jnp.float32)
        metrics["adaptive/blend_alpha_next"] = jnp.array(blend_overrides[current_stage_idx], dtype=jnp.float32)
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
                float, evaluate_success(params, eval_key, current_stage_idx)
            )
            metrics["eval/success_rate"] = eval_success
            metrics["eval/relax_mean"] = eval_relax
            metrics["eval/max_violation"] = eval_violation
            violation_exceeded = eval_violation > stage_violation_threshold
            if violation_exceeded:
                eval_success = 0.0
                metrics["eval/violation_exceeded"] = 1.0
            else:
                metrics["eval/violation_exceeded"] = 0.0
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
            blend_results = evaluate_robustness(params, robust_key, robust_noise_levels, current_blend)
            rng, robust_key_neural = jax.random.split(rng)
            neural_results = evaluate_robustness(params, robust_key_neural, robust_noise_levels, 1.0)
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

        history.append(
            {
                k: float(v) if jnp.ndim(v) == 0 else [float(x) for x in jnp.atleast_1d(v)]
                for k, v in metrics.items()
            }
        )
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
    cfg = build_config(args)
    main(
        cfg,
        use_rnn=args.use_rnn,
        output_dir=args.output_dir,
        cbf_params_path=args.cbf_params,
        policy_params_path=args.policy_params,
    )
