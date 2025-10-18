"""
调试脚本：逐步验证核心模块
============================

该脚本以最小化设置串联感知 → 策略 → 安全过滤 → 物理引擎的完整循环，
方便在重构后快速检查端到端梯度与数值稳定性。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from core.physics import (
    PhysicsParams,
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
)
from core.safety import SafetyConfig
from core.loop import rollout_episode
from core.simple_training import DEFAULT_EFFICIENCY_CONFIG, compute_efficiency_loss


@dataclass
class DebugConfig:
    horizon: int = 20
    dt: float = 1.0 / 15.0
    max_accel: float = 5.0
    policy_hidden: tuple[int, ...] = (128, 128)
    num_points: int = 32
    lidar_radius: float = 2.5
    lidar_height: float = 1.5
    seed: int = 0
    noise_scale: float = 0.0


def make_ring_point_cloud(num_points: int, radius: float, height: float) -> jnp.ndarray:
    angles = jnp.linspace(0.0, 2 * jnp.pi, num_points, endpoint=False)
    x = radius * jnp.cos(angles)
    y = radius * jnp.sin(angles)
    z = jnp.full_like(angles, height)
    return jnp.stack([x, y, z], axis=1)


def initialise_components(cfg: DebugConfig) -> tuple[
    dict,
    PolicyNetwork,
    PolicyState,
    PhysicsParams,
    jnp.ndarray,
    GraphConfig,
    SafetyConfig,
]:
    rng = jax.random.PRNGKey(cfg.seed)
    rng_policy, rng_cbf = jax.random.split(rng)

    physics_params = PhysicsParams(dt=cfg.dt, max_acceleration=cfg.max_accel)

    policy_config = PolicyConfig(
        hidden_dims=cfg.policy_hidden,
        activation="relu",
        output_dim=3,
        use_rnn=False,
        action_limit=cfg.max_accel,
    )
    policy_net = PolicyNetwork(policy_config)
    policy_params, policy_state = initialise_policy_params(
        rng_policy, policy_config, observation_dim=10
    )

    graph_config = GraphConfig(
        max_points=cfg.num_points,
        max_distance=6.0,
        k_neighbors=6,
    )
    point_cloud = make_ring_point_cloud(cfg.num_points, cfg.lidar_radius, cfg.lidar_height)

    dummy_state = create_initial_state()
    dummy_graph, _ = build_graph_from_point_cloud(dummy_state, point_cloud, graph_config)
    cbf_params = initialise_cbf_params(rng_cbf, dummy_graph)

    safety_config = SafetyConfig(
        alpha0=1.0,
        alpha1=2.0,
        max_acceleration=cfg.max_accel,
        relaxation_penalty=150.0,
        max_relaxation=2.0,
        tolerance=1e-5,
    )

    params = {"policy": policy_params, "cbf": cbf_params}
    return (
        params,
        policy_net,
        policy_state,
        physics_params,
        point_cloud,
        graph_config,
        safety_config,
    )


def run_debug_episode(cfg: DebugConfig):
    (
        params,
        policy_net,
        policy_state,
        physics_params,
        point_cloud,
        graph_config,
        safety_config,
    ) = initialise_components(cfg)

    init_state = create_initial_state(position=jnp.array([0.0, 0.0, 1.0]))
    target = jnp.array([4.0, 0.0, 1.5])

    _, rollout = rollout_episode(
        params=params,
        policy_net=policy_net,
        policy_state=policy_state,
        initial_state=init_state,
        physics_params=physics_params,
        point_cloud=point_cloud,
        graph_config=graph_config,
        safety_config=safety_config,
        target_position=target,
        horizon=cfg.horizon,
        gradient_decay=0.4,
        rng=jax.random.PRNGKey(cfg.seed + 1),
        noise_scale=cfg.noise_scale,
    )

    trajectory = {
        "positions": rollout.position,
        "controls": rollout.u_safe,
        "velocities": rollout.velocity,
    }
    loss, metrics = compute_efficiency_loss(
        trajectory, target, DEFAULT_EFFICIENCY_CONFIG
    )

    summary = {
        "total_loss": float(loss),
        "mean_relaxation": float(jnp.mean(rollout.relaxation)),
        "max_constraint_violation": float(jnp.max(rollout.constraint_violation)),
        "cbf_mean": float(jnp.mean(rollout.cbf_value)),
        "cbf_min": float(jnp.min(rollout.cbf_value)),
        "eff_final_distance": float(metrics["goal_final_distance_to_goal"]),
    }
    return rollout, summary


def main():
    parser = argparse.ArgumentParser(description="Run a single debug rollout.")
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise-scale", type=float, default=0.0, help="Gaussian noise std applied to perception/observation")
    args = parser.parse_args()

    cfg = DebugConfig(horizon=args.horizon, seed=args.seed, noise_scale=args.noise_scale)
    rollout, summary = run_debug_episode(cfg)

    print("=== Debug summary ===")
    for key, value in summary.items():
        print(f"{key:>28}: {value: .6f}")

    print("\nLast step diagnostics:")
    print(f"  u_nominal: {rollout.u_nominal[-1]}")
    print(f"  u_safe   : {rollout.u_safe[-1]}")
    print(f"  cbf_value: {rollout.cbf_value[-1]: .6f}")
    print(f"  relaxation: {rollout.relaxation[-1]: .6f}")


if __name__ == "__main__":
    main()
