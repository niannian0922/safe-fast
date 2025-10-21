#!/usr/bin/env python3
"""使用解析 soft-min 屏障对 CBF 网络进行预训练。"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Tuple

import sys
from pathlib import Path as _Path

ROOT = _Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import jax
import jax.numpy as jnp
import optax
import jax.tree_util as jtu

from core.physics import DroneState
from core.perception import (
    GraphConfig,
    build_graph_from_point_cloud,
    create_cbf_model,
    initialise_cbf_params,
    _analytic_cbf_statistics,
)


def sample_state(rng: jax.Array, max_range: float, max_velocity: float) -> Tuple[DroneState, jax.Array]:
    rng, rng_pos = jax.random.split(rng)
    position = jax.random.uniform(rng_pos, (3,), minval=-max_range, maxval=max_range)
    rng, rng_vel = jax.random.split(rng)
    velocity = jax.random.uniform(rng_vel, (3,), minval=-max_velocity, maxval=max_velocity)
    acceleration = jnp.zeros(3)
    state = DroneState(
        position=position,
        velocity=velocity,
        acceleration=acceleration,
        time=jnp.array(0.0, dtype=jnp.float32),
        orientation=jnp.eye(3),
    )
    return state, rng


def sample_point_cloud(rng: jax.Array, num_points: int, max_range: float) -> Tuple[jnp.ndarray, jax.Array]:
    rng, key = jax.random.split(rng)
    cloud = jax.random.uniform(key, (num_points, 3), minval=-max_range, maxval=max_range)
    return cloud, rng


def generate_dataset(
    rng: jax.Array,
    dataset_size: int,
    graph_config: GraphConfig,
    max_range: float,
    max_velocity: float,
):
    @jax.jit
    def single(key):
        key_state, key_cloud = jax.random.split(key)
        state, _ = sample_state(key_state, max_range, max_velocity)
        cloud, _ = sample_point_cloud(key_cloud, graph_config.max_points, max_range)
        graph, _ = build_graph_from_point_cloud(state, cloud, graph_config)
        value, _, _ = _analytic_cbf_statistics(state, cloud)
        return graph, value

    keys = jax.random.split(rng, dataset_size)
    graphs, values = jax.vmap(single)(keys)
    return graphs, values


def main() -> None:
    parser = argparse.ArgumentParser(description="使用解析监督对 CBF 网络进行预训练")
    parser.add_argument("--dataset-size", type=int, default=512, help="数据集大小")
    parser.add_argument("--batch-size", type=int, default=32, help="批大小")
    parser.add_argument("--steps", type=int, default=2000, help="训练步数")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--max-range", type=float, default=3.0, help="采样范围上限")
    parser.add_argument("--max-velocity", type=float, default=2.0, help="速度采样上限")
    parser.add_argument("--log-every", type=int, default=100, help="日志输出频率")
    parser.add_argument("--output", type=str, default="outputs/cbf_pretrained.pkl", help="输出文件路径")
    parser.add_argument("--init-params", type=str, default=None, help="初始参数文件")
    parser.add_argument("--no-jit", action="store_true", help="禁用 JIT 更新（用于调试）")
    args = parser.parse_args()

    rng = jax.random.PRNGKey(args.seed)
    graph_config = GraphConfig()
    params = None
    if args.init_params:
        ckpt = pickle.load(Path(args.init_params).open("rb"))
        params = ckpt.get("params")
        graph_config = ckpt.get("graph_config", graph_config)

    graphs, targets = generate_dataset(
        rng,
        dataset_size=args.dataset_size,
        graph_config=graph_config,
        max_range=args.max_range,
        max_velocity=args.max_velocity,
    )

    if params is None:
        dummy_graph = jax.tree_util.tree_map(lambda x: x[0], graphs)
        params = initialise_cbf_params(jax.random.PRNGKey(args.seed + 1), dummy_graph)

    model = create_cbf_model()
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)
    dataset_indices = jnp.arange(args.dataset_size)
    use_jit = not args.no_jit

    def gather_batch(tree, idx):
        return jtu.tree_map(lambda x: jnp.take(x, idx, axis=0), tree)

    def compute_loss(p, idx):
        batch_graphs = gather_batch(graphs, idx)
        batch_targets = targets[idx]
        preds = jax.vmap(model.apply, in_axes=(None, 0))(p, batch_graphs)
        return jnp.mean((preds - batch_targets) ** 2)

    if use_jit:

        @jax.jit
        def train_step(p, state, idx):
            loss_value, grads = jax.value_and_grad(compute_loss)(p, idx)
            updates, state = optimizer.update(grads, state, p)
            new_params = optax.apply_updates(p, updates)
            return new_params, state, loss_value

    else:

        def train_step(p, state, idx):
            loss_value, grads = jax.value_and_grad(compute_loss)(p, idx)
            updates, state = optimizer.update(grads, state, p)
            new_params = optax.apply_updates(p, updates)
            return new_params, state, loss_value

    for step in range(args.steps):
        rng, rng_idx = jax.random.split(rng)
        batch_indices = jax.random.choice(
            rng_idx,
            dataset_indices,
            (min(args.batch_size, args.dataset_size),),
            replace=False,
        )
        params, opt_state, loss = train_step(params, opt_state, batch_indices)
        if (step + 1) % args.log_every == 0:
            print(f"[{step+1:05d}/{args.steps}] loss={float(loss):.6f}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as fh:
        pickle.dump({"params": params, "graph_config": graph_config}, fh)
    print(f"Saved pretrained CBF parameters to {output_path}")


if __name__ == "__main__":
    main()
