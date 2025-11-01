#!/usr/bin/env python3
"""
策略观测与QP诊断辅助脚本

这个脚本用于快速查看某次 rollout 中的观测向量、策略输出、QP 回退情况等细节，
方便定位“加入安全层后策略失效”的根因。默认采用 stage1_analytic 配置，可选加载
效率阶段的教师策略或神经 CBF 参数。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

import jax
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from core.loop import RolloutCarry  # noqa: E402
from core.perception import (  # noqa: E402
    GraphConfig,
    _analytic_cbf_statistics,
    build_graph_from_point_cloud,
    compute_cbf_statistics,
    initialise_cbf_params,
)
from core.physics import (  # noqa: E402
    PhysicsParams,
    apply_temporal_gradient_decay_to_state,
    create_initial_state,
    dynamics_step,
)
from core.policy import (  # noqa: E402
    PolicyConfig,
    PolicyNetwork,
    initialise_policy_params,
)
from core.safety import SafetyConfig, safety_filter  # noqa: E402
from core.simple_training import DEFAULT_EFFICIENCY_CONFIG, compute_efficiency_loss  # noqa: E402
from train_safe_policy import (  # noqa: E402
    TrainingConfig,
    load_base_config,
    make_ring_point_cloud,
    sample_initial_state,
)


def load_policy(path: Optional[Path], policy_net: PolicyNetwork, observation_dim: int, seed: int):
    """为了兼容没有教师策略的情况，这里提供一个简洁的加载逻辑。"""
    params, state = initialise_policy_params(jax.random.PRNGKey(seed), policy_config, observation_dim)
    if path is not None:
        payload = Path(path).read_bytes()
        import pickle

        params = pickle.loads(payload)
        if isinstance(params, dict) and "params" in params:
            params = {"params": params["params"]}
    return params, state


def pretty_vec(name: str, vec: jnp.ndarray) -> str:
    """把三维向量转成易读的中文字符串。"""
    values = ", ".join(f"{float(x): .4f}" for x in vec)
    return f"{name}: [{values}]"


def main():
    parser = argparse.ArgumentParser(description="查看策略输入与安全层行为的详细信息")
    parser.add_argument("--config-name", type=str, default="stage1_analytic", help="基础配置名称，默认取 stage1_analytic")
    parser.add_argument("--policy-params", type=Path, default=None, help="学生策略参数路径，例如 outputs/efficiency_baseline_policy.pkl")
    parser.add_argument("--teacher-params", type=Path, default=None, help="可选：教师策略参数，用于对比蒸馏效果")
    parser.add_argument("--cbf-params", type=Path, default=None, help="可选：神经 CBF 参数；若不提供就使用解析 CBF")
    parser.add_argument("--use-safety", action="store_true", help="开启可微安全层以复现实验环境")
    parser.add_argument("--steps", type=int, default=20, help="打印的时间步数")
    parser.add_argument("--seed", type=int, default=2025, help="初始状态与点云扰动所用的随机种子")
    parser.add_argument("--target", type=float, nargs=3, default=(4.0, 0.0, 1.5), help="目标位置，用于复现不同课程阶段")
    args = parser.parse_args()

    base_cfg = load_base_config(args.config_name)

    physics_params = PhysicsParams(
        dt=float(base_cfg.physics.dt),
        max_acceleration=float(base_cfg.physics.control.max_thrust),
        gradient_decay=float(base_cfg.physics.gradient_decay.alpha),
    )
    global policy_config
    policy_config = PolicyConfig(
        hidden_dims=tuple(base_cfg.policy.hidden_dims),
        activation=base_cfg.policy.activation,
        output_dim=int(base_cfg.policy.output_dim),
        use_rnn=bool(base_cfg.policy.use_rnn),
        rnn_hidden_size=int(base_cfg.policy.rnn_hidden_size),
        action_limit=float(base_cfg.policy.action_limit),
    )
    policy_net = PolicyNetwork(policy_config)

    policy_params, policy_state = load_policy(args.policy_params, policy_net, observation_dim=10, seed=args.seed)
    teacher_params = None
    if args.teacher_params is not None:
        teacher_params, _ = load_policy(args.teacher_params, policy_net, observation_dim=10, seed=args.seed + 999)

    graph_config = GraphConfig(
        max_points=int(base_cfg.gcbf.max_neighbors),
        max_distance=float(base_cfg.gcbf.sensing_radius * 2.0),
        k_neighbors=int(base_cfg.gcbf.k_neighbors),
    )
    base_point_cloud = make_ring_point_cloud(
        graph_config.max_points,
        radius=float(base_cfg.env.area_size / 2.0),
        height=float(base_cfg.physics.drone.radius + 1.0),
    )

    cbf_params = None
    blend_alpha = 0.0
    if args.cbf_params is not None:
        import pickle

        payload = pickle.load(Path(args.cbf_params).open("rb"))
        cbf_params = {"params": payload["params"]} if isinstance(payload, dict) and "params" in payload else payload
        blend_alpha = 1.0

    safety_config = SafetyConfig(
        alpha0=float(base_cfg.safety.alpha0),
        alpha1=float(base_cfg.safety.alpha1),
        max_acceleration=float(base_cfg.safety.max_acceleration),
        relaxation_penalty=float(base_cfg.safety.relaxation_penalty),
        max_relaxation=float(base_cfg.safety.max_relaxation),
        tolerance=float(base_cfg.safety.violation_tolerance),
    )

    rng = jax.random.PRNGKey(args.seed)
    rng_state, _ = jax.random.split(rng)

    carry = RolloutCarry(
        drone=sample_initial_state(rng_state),
        policy=policy_net.init_state(batch_size=1),
    )

    target_position = jnp.asarray(args.target, dtype=jnp.float32)
    point_cloud = base_point_cloud

    metrics: Dict[str, float] = {}
    positions = []
    velocities = []
    controls = []

    for step in range(args.steps):
        state = carry.drone
        policy_hidden = carry.policy

        if cbf_params is not None:
            cbf_value, grad_pos, hess_pos = compute_cbf_statistics(
                cbf_params, state, point_cloud, graph_config
            )
        else:
            cbf_value, grad_pos, hess_pos = _analytic_cbf_statistics(state, point_cloud)

        pos_obs = state.position
        vel_obs = state.velocity
        target_obs = target_position

        observation = jnp.concatenate(
            [
                pos_obs,
                vel_obs,
                target_obs - pos_obs,
                jnp.array([cbf_value]),
            ]
        )

        action, new_policy_state = policy_net.apply(
            policy_params, observation[None, :], policy_hidden, training=False
        )
        action = action.squeeze(0)

        teacher_action = None
        if teacher_params is not None:
            teacher_action, _ = policy_net.apply(
                teacher_params, observation[None, :], policy_hidden, training=False
            )
            teacher_action = teacher_action.squeeze(0)

        if args.use_safety:
            u_safe, diag = safety_filter(
                action,
                cbf_value,
                grad_pos,
                hess_pos,
                state,
                safety_config,
            )
        else:
            u_safe = action
            diag = None

        next_state = dynamics_step(state, u_safe, physics_params)
        next_state = apply_temporal_gradient_decay_to_state(next_state, physics_params.gradient_decay)

        positions.append(next_state.position)
        velocities.append(next_state.velocity)
        controls.append(u_safe)

        print(f"—— 第 {step:02d} 步 ——")
        print(pretty_vec("位置", state.position))
        print(pretty_vec("速度", state.velocity))
        print(pretty_vec("目标偏差", target_obs - state.position))
        print(f"CBF 值: {float(cbf_value): .4f}")
        print(pretty_vec("策略动作", action))
        if teacher_action is not None:
            diff = jnp.linalg.norm(action - teacher_action)
            print(pretty_vec("教师动作", teacher_action))
            print(f"策略与教师差异范数: {float(diff): .4f}")
        if diag is not None:
            print(pretty_vec("安全动作", u_safe))
            print(
                f"QP回退: failed={float(diag.qp_failed):.0f} "
                f"relax={float(diag.relaxation): .4f} "
                f"constraint_residual={float(diag.constraint_violation): .4f}"
            )
        else:
            print(pretty_vec("最终动作", u_safe))
        print()

        carry = RolloutCarry(drone=next_state, policy=new_policy_state)

    trajectory = {
        "positions": jnp.stack(positions),
        "velocities": jnp.stack(velocities),
        "controls": jnp.stack(controls),
    }
    eff_loss, eff_metrics = compute_efficiency_loss(trajectory, target_position, DEFAULT_EFFICIENCY_CONFIG)
    print("—— 轨迹指标 ——")
    print(f"效率损失: {float(eff_loss): .4f}")
    mean_dist = eff_metrics.get("mean_distance_to_goal", eff_metrics.get("mean_anisotropic_distance", 0.0))
    final_dist = eff_metrics.get("final_distance_to_goal", eff_metrics.get("final_anisotropic_distance", 0.0))
    print(f"平均目标距离: {float(mean_dist): .4f}")
    print(f"最终距离: {float(final_dist): .4f}")


if __name__ == "__main__":
    main()
