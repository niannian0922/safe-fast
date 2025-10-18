"""
core.perception
================

This module adapts the GCBF+ perception stack to the single-drone setting used
throughout the refactored code base.  It turns a LiDAR-style point cloud into a
`jraph.GraphsTuple`, applies a lightweight attention GNN, and produces a scalar
control barrier function (CBF) value together with its gradient w.r.t. the drone
position via automatic differentiation.

All routines are pure functions and therefore safe to use inside JAX
transformations such as `jit` and `lax.scan`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import jraph
from flax import linen as nn

from .physics import DroneState


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GraphConfig:
    """
    Configuration for graph construction.

    The incoming point cloud is expected to have at least `max_points` samples
    (padding with zeros is acceptable).  Any additional points are ignored.
    """

    max_points: int = 64
    max_distance: float = 6.0
    k_neighbors: int = 8

    ego_feature_dim: int = 9  # position, velocity, acceleration
    obstacle_feature_dim: int = 9  # padded to match ego feature dim
    edge_feature_dim: int = 4  # relative xyz + distance


def _relative_point_cloud(
    state: DroneState, point_cloud_world: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Returns relative vectors and distances."""
    rel = point_cloud_world - state.position
    dist = jnp.linalg.norm(rel, axis=-1)
    return rel, dist


def build_graph_from_point_cloud(
    state: DroneState,
    point_cloud_world: jnp.ndarray,
    config: GraphConfig,
) -> Tuple[jraph.GraphsTuple, jnp.ndarray]:
    """
    Converts a point cloud into a `GraphsTuple`.  Returns the graph and a node
    mask that marks real obstacle nodes (True) versus padded placeholders.
    """
    # Use only the first `max_points` samples and treat the remainder as padding.
    cloud = point_cloud_world[: config.max_points]
    rel, dist = _relative_point_cloud(state, cloud)
    non_zero = jnp.any(jnp.abs(rel) > 1e-6, axis=-1)
    obstacle_mask = (dist <= config.max_distance) & non_zero
    rel = jnp.where(obstacle_mask[:, None], rel, 0.0)

    # Assemble node features.
    ego_features = jnp.concatenate([state.position, state.velocity, state.acceleration])
    ego_features = jnp.pad(
        ego_features[: config.ego_feature_dim],
        (0, config.ego_feature_dim - ego_features.shape[0]),
    )
    obstacle_features = jnp.pad(
        rel, ((0, 0), (0, config.obstacle_feature_dim - rel.shape[1]))
    )

    nodes = jnp.concatenate(
        [ego_features[None, :], obstacle_features],
        axis=0,
    )
    node_mask = jnp.concatenate(
        [jnp.array([True]), obstacle_mask],
        axis=0,
    )

    obstacle_positions = jnp.where(
        obstacle_mask[:, None],
        state.position[None, :] + rel,
        state.position[None, :],
    )
    all_positions = jnp.concatenate(
        [state.position[None, :], obstacle_positions],
        axis=0,
    )
    n_nodes = int(nodes.shape[0])
    distances = jnp.linalg.norm(
        all_positions[:, None, :] - all_positions[None, :, :], axis=-1
    )
    # Prevent invalid nodes from being selected as neighbours.
    node_mask = jnp.concatenate([jnp.array([True]), obstacle_mask], axis=0)
    valid_matrix = node_mask[:, None] & node_mask[None, :]
    large = jnp.max(distances) + 1e3
    distances = jnp.where(valid_matrix, distances, large)
    distances = distances + jnp.eye(n_nodes) * 1e6  # exclude self

    k = max(0, min(config.k_neighbors, n_nodes - 1))
    if k == 0:
        senders = jnp.zeros((0,), dtype=jnp.int32)
        receivers = jnp.zeros((0,), dtype=jnp.int32)
        edges = jnp.zeros((0, config.edge_feature_dim), dtype=jnp.float32)
    else:
        _, neighbors = jax.lax.top_k(-distances, k)  # take k smallest distances
        senders = jnp.repeat(jnp.arange(n_nodes), k)
        receivers = neighbors.reshape(-1)
        edge_vectors = all_positions[receivers] - all_positions[senders]
        edge_dist = jnp.linalg.norm(edge_vectors, axis=-1, keepdims=True)
        edges = jnp.concatenate([edge_vectors, edge_dist], axis=-1)

        # Mask edges touching padded nodes by zeroing their contribution.
        valid_edges = node_mask[senders] & node_mask[receivers]
        senders = jnp.where(valid_edges, senders, 0)
        receivers = jnp.where(valid_edges, receivers, 0)
        edges = jnp.where(valid_edges[:, None], edges, 0.0)

    graph = jraph.GraphsTuple(
        nodes=nodes,
        edges=edges,
        senders=senders,
        receivers=receivers,
        n_node=jnp.array([n_nodes]),
        n_edge=jnp.array([edges.shape[0]]),
        globals=None,
    )
    return graph, node_mask


# ---------------------------------------------------------------------------
# GNN + CBF head (adapted from GCBF+)
# ---------------------------------------------------------------------------


def _safe_get(array: jnp.ndarray, indices: jnp.ndarray) -> jnp.ndarray:
    valid = indices >= 0
    safe_idx = jnp.where(valid, indices, 0)
    gathered = array[safe_idx]
    return jnp.where(valid[:, None], gathered, 0.0)


class MLP(nn.Module):
    widths: Tuple[int, ...]
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i, width in enumerate(self.widths):
            x = nn.Dense(width, kernel_init=nn.initializers.xavier_uniform())(x)
            if i < len(self.widths) - 1:
                x = self.activation(x)
        return x


class GNNLayer(nn.Module):
    hidden_dim: int
    msg_mlp: Tuple[int, ...] = (128, 128)
    attn_mlp: Tuple[int, ...] = (64,)
    update_mlp: Tuple[int, ...] = (128, 128)

    @nn.compact
    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        nodes = graph.nodes
        send = _safe_get(nodes, graph.senders)
        recv = _safe_get(nodes, graph.receivers)

        msg_input = jnp.concatenate([graph.edges, send, recv], axis=-1)
        msgs = MLP(self.msg_mlp)(msg_input)
        msgs = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.xavier_uniform())(
            msgs
        )

        gate = MLP(self.attn_mlp)(msgs)
        gate = nn.Dense(1, kernel_init=nn.initializers.xavier_uniform())(gate).squeeze(-1)
        attn = jraph.segment_softmax(gate, graph.receivers, num_segments=nodes.shape[0])
        aggregated = jraph.segment_sum(
            attn[:, None] * msgs,
            graph.receivers,
            num_segments=nodes.shape[0],
        )

        update_input = jnp.concatenate([nodes, aggregated], axis=-1)
        new_nodes = MLP(self.update_mlp)(update_input)
        new_nodes = nn.Dense(
            self.hidden_dim, kernel_init=nn.initializers.xavier_uniform()
        )(new_nodes)
        return graph._replace(nodes=new_nodes)


class CBFBackbone(nn.Module):
    hidden_dim: int = 128
    num_layers: int = 3

    @nn.compact
    def __call__(self, graph: jraph.GraphsTuple) -> jnp.ndarray:
        g = graph
        for i in range(self.num_layers):
            dim = self.hidden_dim
            g = GNNLayer(dim, name=f"layer_{i}")(g)
        return g.nodes


class CBFNetwork(nn.Module):
    backbone: CBFBackbone
    head_widths: Tuple[int, ...] = (128, 64)

    @nn.compact
    def __call__(self, graph: jraph.GraphsTuple) -> jnp.ndarray:
        embeddings = self.backbone(graph)
        ego_embedding = embeddings[0]  # first node is always the ego drone
        x = MLP(self.head_widths)(ego_embedding)
        x = nn.Dense(1, kernel_init=nn.initializers.xavier_uniform())(x)
        return jnp.squeeze(nn.tanh(x), axis=0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def create_cbf_model() -> CBFNetwork:
    return CBFNetwork(backbone=CBFBackbone())


def initialise_cbf_params(rng: jax.Array, graph: jraph.GraphsTuple) -> dict:
    model = create_cbf_model()
    params = model.init(rng, graph)
    return params


def evaluate_cbf(params: dict, graph: jraph.GraphsTuple) -> jnp.ndarray:
    model = create_cbf_model()
    return model.apply(params, graph)


def _analytic_cbf_statistics(
    state: DroneState,
    point_cloud_world: jnp.ndarray,
    safety_radius: float = 0.3,
    temperature: float = 10.0,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Analytic soft-min distance based barrier used as a fallback when the neural
    CBF is unavailable or numerically unstable.
    """
    points = point_cloud_world
    if points.shape[0] == 0:
        # No obstacles -> large positive value
        value = jnp.array(1.0, dtype=jnp.float32)
        grad = jnp.zeros(3, dtype=jnp.float32)
        hess = jnp.zeros((3, 3), dtype=jnp.float32)
        return value, grad, hess

    def softmin_distance(position: jnp.ndarray) -> jnp.ndarray:
        diffs = points - position[None, :]
        dists = jnp.linalg.norm(diffs, axis=-1)
        weights = jax.nn.softmax(-temperature * dists)
        return jnp.sum(weights * dists)

    value_fn = lambda pos: softmin_distance(pos) - safety_radius
    h_value, grad = jax.value_and_grad(value_fn)(state.position)
    hess = jax.jacfwd(jax.grad(value_fn))(state.position)
    return h_value, grad, hess


def compute_cbf_statistics(
    params: dict | None,
    state: DroneState,
    point_cloud_world: jnp.ndarray,
    config: GraphConfig,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Evaluate the CBF and return the value, gradient, and Hessian w.r.t. position.
    Falls back to an analytic soft-min distance barrier when the learned CBF is
    unavailable or numerically unstable.
    """
    graph, _ = build_graph_from_point_cloud(state, point_cloud_world, config)
    use_network = params is not None and len(jtu.tree_leaves(params)) > 0

    def network_eval():
        model = create_cbf_model()

        def cbf_at_position(position: jnp.ndarray) -> jnp.ndarray:
            shifted_state = DroneState(
                position=position,
                velocity=state.velocity,
                acceleration=state.acceleration,
                time=state.time,
                orientation=state.orientation,
            )
            graph_shifted, _ = build_graph_from_point_cloud(
                shifted_state, point_cloud_world, config
            )
            return model.apply(params, graph_shifted)

        value_fn = lambda pos: cbf_at_position(pos)
        h_value, grad_h = jax.value_and_grad(value_fn)(state.position)
        hessian_h = jax.jacfwd(jax.grad(value_fn))(state.position)
        return h_value, grad_h, hessian_h

    def fallback_eval():
        return _analytic_cbf_statistics(state, point_cloud_world)

    if not use_network:
        return fallback_eval()

    h_value, grad_h, hessian_h = network_eval()

    def replace_if_nan(value, replacement):
        return jax.lax.cond(
            jnp.any(jnp.isnan(value)),
            lambda _: replacement,
            lambda _: value,
            operand=None,
        )

    if use_network:
        analytic_value, analytic_grad, analytic_hess = fallback_eval()
        h_value = replace_if_nan(h_value, analytic_value)
        grad_h = replace_if_nan(grad_h, analytic_grad)
        hessian_h = replace_if_nan(hessian_h, analytic_hess)

    return h_value, grad_h, hessian_h


__all__ = [
    "GraphConfig",
    "build_graph_from_point_cloud",
    "create_cbf_model",
    "initialise_cbf_params",
    "evaluate_cbf",
    "compute_cbf_statistics",
    "_analytic_cbf_statistics",
]
