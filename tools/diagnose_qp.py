#!/usr/bin/env python3
"""
单步 QP 诊断工具

用途：
 1. 在 macOS、Windows、Kaggle 任意环境下复现解析 CBF + qpax 的输入；
 2. 打印 CBF 值、导数、QP 约束右端项等关键量，帮助判断约束不可行的根源；
 3. 支持自定义随机种子与名义控制，便于快速尝试不同场景。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import sys

import jax
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from core.physics import PhysicsParams
from core.perception import GraphConfig, _analytic_cbf_statistics
from core.safety import SafetyConfig, _assemble_qp
from train_safe_policy import (
    load_base_config,
    sample_initial_state,
    make_ring_point_cloud,
)


def format_vec(name: str, vec: jnp.ndarray) -> str:
    """用中文标签打印向量内容，方便人工阅读。"""
    values = ", ".join(f"{float(x): .4f}" for x in vec)
    return f"{name}: [{values}]"


def main():
    parser = argparse.ArgumentParser(description="诊断解析 CBF + QP 的可行性。")
    parser.add_argument("--config-name", type=str, default="stage1_analytic", help="基础配置名称，例如 default 或 stage1_analytic")
    parser.add_argument("--seed", type=int, default=2025, help="采样点云与初始状态使用的随机种子")
    parser.add_argument("--nom-u", type=float, nargs=3, default=(0.0, 0.0, 0.0), help="人为指定的名义加速度，用于分析不同控制输入下的约束情况")
    parser.add_argument("--position", type=float, nargs=3, default=None, help="可选的无人机位置覆盖，格式为 x y z，方便直接观察贴边情况下的 CBF 表现")
    args = parser.parse_args()

    base_cfg = load_base_config(args.config_name)

    rng = jax.random.PRNGKey(args.seed)
    rng_state, rng_cloud = jax.random.split(rng)

    physics_params = PhysicsParams(
        dt=float(base_cfg.physics.dt),
        max_acceleration=float(base_cfg.physics.control.max_thrust),
        gradient_decay=float(base_cfg.physics.gradient_decay.alpha) if hasattr(base_cfg.physics.gradient_decay, "alpha") else 0.92,
    )

    graph_config = GraphConfig(
        max_points=int(base_cfg.gcbf.max_neighbors),
        max_distance=float(base_cfg.gcbf.sensing_radius * 2.0),
        k_neighbors=int(base_cfg.gcbf.k_neighbors),
    )
    base_cloud = make_ring_point_cloud(
        graph_config.max_points,
        radius=float(base_cfg.env.area_size / 2.0),
        height=float(base_cfg.physics.drone.radius + 1.0),
    )

    sampled_state = sample_initial_state(rng_state)
    if args.position is not None:
        pos = jnp.asarray(args.position, dtype=jnp.float32)
        sampled_state = sampled_state.replace(position=pos)
    point_cloud = base_cloud + 0.0

    cbf_value, grad_pos, hess_pos = _analytic_cbf_statistics(sampled_state, point_cloud)

    safety_config = SafetyConfig(
        alpha0=float(base_cfg.safety.alpha0),
        alpha1=float(base_cfg.safety.alpha1),
        max_acceleration=float(base_cfg.safety.max_acceleration),
        relaxation_penalty=float(base_cfg.safety.relaxation_penalty),
        max_relaxation=float(base_cfg.safety.max_relaxation),
        tolerance=float(base_cfg.safety.violation_tolerance),
    )

    u_nom = jnp.asarray(args.nom_u, dtype=jnp.float32)
    Q, q, G, h_vec = _assemble_qp(
        u_nom,
        grad_pos,
        hess_pos,
        cbf_value,
        sampled_state,
        safety_config,
    )

    lf_h = float(jnp.dot(grad_pos, sampled_state.velocity))
    psi1 = lf_h + safety_config.alpha0 * float(cbf_value)
    partial_psi1_pos = hess_pos @ sampled_state.velocity + safety_config.alpha0 * grad_pos
    lf_psi1 = float(jnp.dot(partial_psi1_pos, sampled_state.velocity))
    rhs = lf_psi1 + safety_config.alpha1 * psi1

    print("======== 单步 QP 诊断 ========")
    print("解析 CBF 数值概览：")
    print(f"  h(x) = {float(cbf_value): .4f}")
    print(format_vec("  ∂h/∂p", grad_pos))
    print("  Hessian(∂²h/∂p²) 行列式: {det: .4f}".format(det=float(jnp.linalg.det(hess_pos + jnp.eye(3) * 1e-6))))
    print()
    print("动力学项：")
    print(f"  当前速度: {format_vec('', sampled_state.velocity)}")
    print(f"  L_f h = {lf_h: .4f}")
    print(f"  ψ₁ = {psi1: .4f}")
    print(f"  L_f ψ₁ = {lf_psi1: .4f}")
    print(f"  约束右端项 rhs = {rhs: .4f}")
    print()
    print("QP 系数矩阵：")
    print(f"  Q =\n{Q}")
    print(f"  q = {q}")
    print(f"  第一行 CBF 约束 = {G[0]}")
    print(f"  h_vec[0] (CBF 约束) = {h_vec[0]: .4f}")
    print()
    print("名义控制对 CBF 的影响：")
    cbf_gradient_control = float(jnp.dot(grad_pos, -u_nom))
    print(f"  -Lg ψ₁ · u_nom = {cbf_gradient_control: .4f}")
    print("  若该项无法压低 rhs，说明需要更强的松弛或重新设计解析 CBF。")


if __name__ == "__main__":
    main()
