#!/usr/bin/env python3
"""Evaluate a saved training artifact over multiple rollouts."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
import sys

import jax
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from configs import default_config
from core.loop import rollout_episode
from core.perception import GraphConfig, build_graph_from_point_cloud
from core.physics import PhysicsParams
from core.policy import PolicyConfig, PolicyNetwork
from core.safety import SafetyConfig
from core.flax_compat import struct
from train_safe_policy import (
    TrainingConfig,
    make_ring_point_cloud,
    sample_augmented_point_cloud,
    sample_initial_state,
)


@struct.dataclass
class EvalResult:
    success_rate: float
    distance_mean: float
    distance_p95: float
    relax_mean: float
    violation_max: float
    cbf_min_mean: float


def load_training_config(payload: dict) -> TrainingConfig:
    cfg = default_config.get_config()
    base = TrainingConfig()
    data = payload.get("config", {})
    kwargs = {
        "horizon": data.get("horizon", base.horizon),
        "gradient_decay": data.get("gradient_decay", base.gradient_decay),
        "target_position": jnp.array(data.get("target_position", base.target_position)),
        "policy_freeze_steps": data.get("policy_freeze_steps", base.policy_freeze_steps),
        "cbf_blend_alpha": data.get("cbf_blend_alpha", base.cbf_blend_alpha),
        "augment_point_cloud": data.get("augment_point_cloud", base.augment_point_cloud),
    }
    return TrainingConfig(**{**base.__dict__, **kwargs})


def build_evaluation_components(cfg: TrainingConfig):
    defaults = default_config.get_config()

    physics_params = PhysicsParams(
        dt=float(defaults.physics.dt),
        max_acceleration=float(defaults.physics.control.max_thrust),
        gradient_decay=cfg.gradient_decay,
    )

    policy_config = PolicyConfig(
        hidden_dims=tuple(defaults.policy.hidden_dims),
        activation=defaults.policy.activation,
        output_dim=int(defaults.policy.output_dim),
        use_rnn=bool(defaults.policy.use_rnn),
        rnn_hidden_size=int(defaults.policy.rnn_hidden_size),
        action_limit=float(defaults.policy.action_limit),
    )
    policy_net = PolicyNetwork(policy_config)
    policy_state = policy_net.init_state(batch_size=1)

    graph_config = GraphConfig(
        max_points=int(defaults.gcbf.max_neighbors),
        max_distance=float(defaults.gcbf.sensing_radius * 2.0),
        k_neighbors=int(defaults.gcbf.k_neighbors),
    )
    base_point_cloud = make_ring_point_cloud(
        graph_config.max_points,
        radius=float(defaults.env.area_size / 2.0),
        height=float(defaults.physics.drone.radius + 1.0),
    )
    _ = build_graph_from_point_cloud(
        sample_initial_state(jax.random.PRNGKey(0)), base_point_cloud, graph_config
    )

    safety_config = SafetyConfig(
        alpha0=float(defaults.safety.alpha0),
        alpha1=float(defaults.safety.alpha1),
        max_acceleration=float(defaults.safety.max_acceleration),
        relaxation_penalty=float(defaults.safety.relaxation_penalty),
        max_relaxation=float(defaults.safety.max_relaxation),
        tolerance=float(defaults.safety.violation_tolerance),
    )

    return physics_params, policy_net, policy_state, graph_config, base_point_cloud, safety_config


def evaluate(
    artifact: Path,
    trials: int,
    seed: int,
    noise: float,
    random_point_cloud: bool,
) -> EvalResult:
    payload = pickle.load(artifact.open("rb"))
    params_policy = payload["params_policy"]
    params_cbf = payload.get("params_cbf", {})
    cfg = load_training_config(payload)

    (
        physics_params,
        policy_net,
        policy_state,
        graph_config,
        base_point_cloud,
        safety_config,
    ) = build_evaluation_components(cfg)

    params = {"policy": params_policy}
    if params_cbf:
        params["cbf"] = params_cbf

    def single_rollout(key):
        key_init, key_cloud = jax.random.split(key)
        init_state = sample_initial_state(key_init)
        if random_point_cloud:
            point_cloud = sample_augmented_point_cloud(base_point_cloud, graph_config, key_cloud)
        else:
            point_cloud = base_point_cloud
        _, outputs = rollout_episode(
            params,
            policy_net,
            policy_state,
            init_state,
            physics_params,
            point_cloud,
            graph_config,
            safety_config,
            cfg.target_position,
            cfg.horizon,
            cfg.gradient_decay,
            key,
            noise_scale=noise,
        )
        final_pos = outputs.position[-1]
        dist = jnp.linalg.norm(final_pos - cfg.target_position)
        success = (dist < 0.1).astype(jnp.float32)
        relax_mean = jnp.mean(outputs.relaxation)
        violation_max = jnp.max(outputs.constraint_violation)
        cbf_min = jnp.min(outputs.cbf_value)
        return dist, success, relax_mean, violation_max, cbf_min

    vmapped = jax.jit(jax.vmap(single_rollout))
    keys = jax.random.split(jax.random.PRNGKey(seed), trials)
    dists, success, relax, violation, cbf_min = vmapped(keys)

    return EvalResult(
        success_rate=float(jnp.mean(success)),
        distance_mean=float(jnp.mean(dists)),
        distance_p95=float(jnp.percentile(dists, 95)),
        relax_mean=float(jnp.mean(relax)),
        violation_max=float(jnp.max(violation)),
        cbf_min_mean=float(jnp.mean(cbf_min)),
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate a training artifact.")
    parser.add_argument("artifact", type=Path, help="Path to training_results.pkl")
    parser.add_argument("--trials", type=int, default=64, help="Number of rollouts")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed")
    parser.add_argument("--noise", type=float, default=0.0, help="Observation noise scale")
    parser.add_argument(
        "--random-point-cloud",
        action="store_true",
        help="Use augmented point clouds during evaluation",
    )
    parser.add_argument("--json", type=Path, default=None, help="Optional JSON output path")
    args = parser.parse_args()

    result = evaluate(
        args.artifact,
        args.trials,
        args.seed,
        args.noise,
        args.random_point_cloud,
    )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with args.json.open("w") as fh:
            json.dump(result.__dict__, fh, indent=2)
    print(
        "success_rate={:.3f} distance_mean={:.3f} distance_p95={:.3f} relax_mean={:.3e} max_violation={:.3f} cbf_min_mean={:.3f}".format(
            result.success_rate,
            result.distance_mean,
            result.distance_p95,
            result.relax_mean,
            result.violation_max,
            result.cbf_min_mean,
        )
    )


if __name__ == "__main__":
    main()
