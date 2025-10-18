import pickle
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jax  # noqa: E402  保证在 sys.path 更新后导入
jax.config.update("jax_disable_jit", True)
import jax.numpy as jnp

from core.perception import (
    GraphConfig,
    build_graph_from_point_cloud,
    evaluate_cbf,
    initialise_cbf_params,
    _analytic_cbf_statistics,
)
from core.physics import create_initial_state
from core.safety import SafetyConfig, safety_filter
from tools.stage_summary import summarize_pickle
from utils.batch_pytree import generate_training_batch, unbatch_pytree_objects


def test_generate_training_batch_shapes():
    batch = generate_training_batch(batch_size=3, rng_key=jax.random.PRNGKey(0))
    state = batch["initial_states"]
    targets = batch["target_positions"]

    assert state.position.shape == (3, 3)
    assert state.velocity.shape == (3, 3)
    assert state.orientation.shape == (3, 3, 3)
    assert targets.shape == (3, 3)

    unbatched = unbatch_pytree_objects(state)
    assert len(unbatched) == 3
    assert all(s.position.shape == (3,) for s in unbatched)


def test_cbf_and_safety_smoke():
    state = create_initial_state(position=jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32))
    angles = jnp.linspace(0.0, 2.0 * jnp.pi, 8, endpoint=False)
    point_cloud = jnp.stack(
        [
            2.0 * jnp.cos(angles),
            2.0 * jnp.sin(angles),
            jnp.ones_like(angles) * 1.0,
        ],
        axis=1,
    ).astype(jnp.float32)

    graph_config = GraphConfig(max_points=8, max_distance=4.0, k_neighbors=4)
    graph, _ = build_graph_from_point_cloud(state, point_cloud, graph_config)
    params = initialise_cbf_params(jax.random.PRNGKey(1), graph)
    cbf_value = evaluate_cbf(params, graph)
    assert jnp.isfinite(cbf_value)

    h_val, grad, hess = _analytic_cbf_statistics(state, point_cloud)

    cfg = SafetyConfig(max_acceleration=5.0)
    u_nom = jnp.array([6.0, -4.0, 2.0], dtype=jnp.float32)
    u_safe, diagnostics = safety_filter(u_nom, h_val, grad, hess, state, cfg)

    assert jnp.all(jnp.abs(u_safe) <= cfg.max_acceleration + 1e-5)
    assert diagnostics.relaxation >= 0.0


def test_stage_summary_handles_basic_history(tmp_path: Path):
    payload = {
        "history": [
            {"loss/total": 2.0, "loss/efficiency": 3.0},
            {"loss/total": 1.0, "loss/efficiency": 1.5},
        ],
        "config": {"horizon": 10},
    }
    pkl_path = tmp_path / "training_results.pkl"
    with pkl_path.open("wb") as fh:
        pickle.dump(payload, fh)

    summary = summarize_pickle(pkl_path)
    assert Path(summary["file"]).name == "training_results.pkl"
    assert summary["loss/total"] == 1.0
    assert summary["loss/efficiency"] == 1.5
