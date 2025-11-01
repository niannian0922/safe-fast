"""
core.perception

本模块将 GCBF+ 的感知栈改造为单机形态：把 LiDAR 风格的点云转换为`jraph.GraphsTuple`，再通过轻量级注意力 GNN 生成标量 CBF 值，并配合自动微分求出相对于无人机位置的梯度。所有函数均为纯函数，可放心用于 `jit`、`lax.scan` 等 JAX 变换。
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
# 图构建
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GraphConfig:
    """
    图构建相关配置。

    输入点云需要至少包含 `max_points` 个样本（允许用 0 填充），多余的点
    会被忽略。
    """

    max_points: int = 64
    max_distance: float = 6.0
    k_neighbors: int = 8

    ego_feature_dim: int = 9  # 位置、速度、加速度
    obstacle_feature_dim: int = 9  # 填充至与自机特征维度相同
    edge_feature_dim: int = 4  # 相对 xyz 与距离


def _relative_point_cloud(
    state: DroneState, point_cloud_world: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """返回相对向量与对应距离。"""
    rel = point_cloud_world - state.position
    dist = jnp.linalg.norm(rel, axis=-1)
    return rel, dist


def build_graph_from_point_cloud(
    state: DroneState,
    point_cloud_world: jnp.ndarray,
    config: GraphConfig,
) -> Tuple[jraph.GraphsTuple, jnp.ndarray]:
    """
    将点云转换为 `GraphsTuple`，并返回图以及节点掩码；掩码中 True 表示真实障碍
    节点，False 表示填充节点。
    """
    # 只保留前 `max_points` 个样本，其余作为填充处理。
    cloud = point_cloud_world[: config.max_points]
    rel, dist = _relative_point_cloud(state, cloud)
    non_zero = jnp.any(jnp.abs(rel) > 1e-6, axis=-1)
    obstacle_mask = (dist <= config.max_distance) & non_zero
    rel = jnp.where(obstacle_mask[:, None], rel, 0.0)

    # 组装节点特征。
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
    # 防止无效节点被选为邻居。
    node_mask = jnp.concatenate([jnp.array([True]), obstacle_mask], axis=0)
    valid_matrix = node_mask[:, None] & node_mask[None, :]
    large = jnp.max(distances) + 1e3
    distances = jnp.where(valid_matrix, distances, large)
    distances = distances + jnp.eye(n_nodes) * 1e6  # 排除自身

    k = max(0, min(config.k_neighbors, n_nodes - 1))
    if k == 0:
        senders = jnp.zeros((0,), dtype=jnp.int32)
        receivers = jnp.zeros((0,), dtype=jnp.int32)
        edges = jnp.zeros((0, config.edge_feature_dim), dtype=jnp.float32)
    else:
        _, neighbors = jax.lax.top_k(-distances, k)  # 选取距离最小的 k 个
        senders = jnp.repeat(jnp.arange(n_nodes), k)
        receivers = neighbors.reshape(-1)
        edge_vectors = all_positions[receivers] - all_positions[senders]
        edge_dist = jnp.linalg.norm(edge_vectors, axis=-1, keepdims=True)
        edges = jnp.concatenate([edge_vectors, edge_dist], axis=-1)

        # 触碰填充节点的边会被掩蔽，其贡献清零。
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
# GNN + CBF 头部（改写自 GCBF+）
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
        ego_embedding = embeddings[0]  # 第一个节点始终代表自机
        x = MLP(self.head_widths)(ego_embedding)
        x = nn.Dense(1, kernel_init=nn.initializers.xavier_uniform())(x)
        return jnp.squeeze(nn.tanh(x), axis=0)


# ---------------------------------------------------------------------------
# 公共接口
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
    基于 soft-min 距离的解析屏障，当神经网络 CBF 不可用或数值发散时作为回退方案。
    """
    points = point_cloud_world
    if points.shape[0] == 0:
        value = jnp.array(5.0, dtype=jnp.float32)
        grad = jnp.zeros(3, dtype=jnp.float32)
        hess = jnp.zeros((3, 3), dtype=jnp.float32)
        return value, grad, hess

    diffs = state.position[None, :] - points
    dists = jnp.linalg.norm(diffs, axis=-1)
    weights = jax.nn.softmax(-temperature * dists)

    smooth_dist = jnp.sum(weights * dists)
    value = smooth_dist - safety_radius

    grad_each = diffs / (dists[:, None] + 1e-6)
    grad = jnp.sum(weights[:, None] * grad_each, axis=0)

    hess = jnp.eye(3) * 2.0
    return value, grad, hess


def compute_cbf_statistics(
    params: dict | None,
    state: DroneState,
    point_cloud_world: jnp.ndarray,
    config: GraphConfig,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    计算 CBF 的值、梯度与相对于位置的 Hessian；若神经 CBF 不可用或数值不稳，则
    回退到 soft-min 解析屏障。
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
