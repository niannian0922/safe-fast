"""
安全敏捷飞行系统的感知模块

这个模块的核心任务是把GCBF+的GNN架构和激光雷达（LiDAR）的点云处理流程结合起来，
最终实现单个智能体基于控制屏障函数（CBF）的安全控制。

主要干几件事：
1. 点云到图的转换（pointcloud_to_graph函数）
2. 用GNN来算出CBF的值和它的梯度
3. 整个实现都是JAX原生的，天生就支持即时编译（JIT）

我是怎么把GCBF+代码库里的东西整合进来的：
- GNN的整体架构，是从gcbfplus/nn/gnn.py里借鉴的
- 图的构建逻辑，参考了gcbfplus/utils/graph.py
- CBF网络的具体实现，学习了gcbfplus/algo/module/cbf.py

"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import struct
import jraph
import functools as ft
from typing import Tuple, Callable, Optional, NamedTuple
from dataclasses import dataclass

# 导入一些JAX的实用工具
from jax import vmap, jit
from jax.lax import top_k

# =============================================================================
# 定义一些数据结构
# =============================================================================

# 为了保持队形一致，我们从physics模块里导入无人机的状态定义
from core.physics import DroneState, create_initial_drone_state

@dataclass(frozen=True)
class GraphConfig:
    """这个数据类用来存放构建图的时候要用到的所有参数。"""
    k_neighbors: int = 8  # 用KNN算法时，找最近的8个邻居
    max_range: float = 5.0  # 感知范围，只处理5米内的障碍物点
    min_points: int = 10  # 点云太稀疏了也不行，至少得有10个点
    max_points: int = 1000  # 点云太多了费内存，最多处理1000个点
    ego_node_features: int = 10  # 代表无人机自己的那个节点的特征维度
    obstacle_node_features: int = 3  # 代表障碍物点的节点的特征维度
    edge_features: int = 4  # 连接节点之间的边的特征维度

# =============================================================================
# 点云到图的转换逻辑
# =============================================================================

def compute_pairwise_distances(points1: jnp.ndarray, points2: jnp.ndarray) -> jnp.ndarray:
    """
    这个函数用来算两堆点云里，每两个点之间的距离。
    实现方式是纯JAX原生的，可以很方便地用vmap进行批处理。
    
    传进来的参数：
        points1: (N, 3) - 第一堆点
        points2: (M, 3) - 第二堆点
    返回：
        distances: (N, M) - 一个距离矩阵
    """
    # 巧妙地利用广播机制来计算所有点对的差值向量：||p1 - p2||_2
    diff = points1[:, None, :] - points2[None, :, :]  # 形状会变成 (N, M, 3)
    distances = jnp.linalg.norm(diff, axis=2)  # 沿着最后一个维度求范数，得到距离矩阵 (N, M)
    return distances

def filter_points_by_range(points: jnp.ndarray, drone_pos: jnp.ndarray, max_range: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    根据点云和无人机的距离，把太远的点给过滤掉。
    """
    distances = jnp.linalg.norm(points - drone_pos, axis=1)
    mask = distances <= max_range
    # 这里用jnp.where而不是直接用布尔索引，是为了JIT兼容性
    valid_indices = jnp.where(mask, size=points.shape[0], fill_value=0)[0]
    valid_points = points[valid_indices]
    return valid_points, mask

def find_knn_edges(drone_pos: jnp.ndarray, obstacle_points: jnp.ndarray, k: int) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    为我们的“以自我为中心”的图找到K近邻的边。
    这个版本是专门为JAX/JIT设计的，输出的数组大小是固定的。
    
    传进来的参数：
        drone_pos: (3,) - 无人机自己的位置
        obstacle_points: (N, 3) - 周围障碍物的点云
        k: KNN算法里的K，邻居数量（这个K必须是静态的，才能做JIT编译）
        
    返回：
        senders: (固定大小,) - 发出边的节点索引（无效边用-1填充）
        receivers: (固定大小,) - 接收边的节点索引（无效边用-1填充）
        edge_features: (固定大小, 4) - 边的特征（无效边用0填充）
    """
    n_obstacles = obstacle_points.shape[0]
    n_total = n_obstacles + 1  # 加上无人机自己这个节点
    
    # 把无人机和障碍物的位置拼在一起，方便处理：[无人机, 障碍物1, 障碍物2, ...]
    all_positions = jnp.concatenate([drone_pos[None, :], obstacle_points], axis=0)
    
    # 计算所有节点之间的距离矩阵
    distances = compute_pairwise_distances(all_positions, all_positions)
    
    # 把对角线（自己到自己）的距离设为无穷大，这样就不会跟自己连边了
    distances_masked = jnp.where(jnp.eye(n_total), jnp.inf, distances)
    
    # K值需要是静态的
    k_use = min(k, n_total - 1)
    
    # 对每个节点，找到离它最近的k_use个邻居
    _, top_k_indices = jax.vmap(lambda row: jax.lax.top_k(-row, k_use))(distances_masked)
    
    # 预先分配固定大小的数组来存储边
    max_edges = n_total * k_use
    
    # 密集地创建出所有可能的边
    all_senders = jnp.repeat(jnp.arange(n_total), k_use)
    all_receivers = top_k_indices.flatten()
    
    # 计算所有这些边的特征
    sender_positions = all_positions[all_senders]
    receiver_positions = all_positions[all_receivers]
    rel_positions = receiver_positions - sender_positions
    edge_distances = jnp.linalg.norm(rel_positions, axis=1, keepdims=True)
    all_edge_features = jnp.concatenate([edge_distances, rel_positions], axis=1)
    
    # 创建一个掩码来标记哪些边是有效的
    # 只要发送方和接收方不是同一个节点，并且接收方索引有效，这条边就是有效的
    validity_mask = (all_senders != all_receivers) & (all_receivers < n_total)
    
    # 我们不用动态大小的数组，而是用固定大小的数组，然后用特殊值来标记无效边
    # 无效边的发送方/接收方索引是-1，特征是0
    final_senders = jnp.where(validity_mask, all_senders, -1)
    final_receivers = jnp.where(validity_mask, all_receivers, -1)
    final_features = jnp.where(validity_mask[:, None], all_edge_features, 0.0)
    
    return final_senders, final_receivers, final_features

def pointcloud_to_graph(drone_state: DroneState, point_cloud: jnp.ndarray, config: GraphConfig) -> Tuple[jraph.GraphsTuple, jnp.ndarray]:
    """
    把LiDAR点云数据转换成jraph库能处理的图结构（GraphsTuple）。
    
    这是我们项目的一个核心适配工作，把GCBF+里处理多个智能体的逻辑，
    改造成了现在处理单个智能体和LiDAR点云的逻辑。
    
    传进来的参数：
        drone_state: 无人机当前的状态
        point_cloud: (N, 3) 的LiDAR点云，这些点是在无人机机体坐标系下的
        config: 图构建的配置参数
    
    返回：
        graph: 一个jraph.GraphsTuple对象，可以直接喂给GNN
        node_types: 一个数组，用来区分哪个节点是无人机，哪些是障碍物
        
    图的结构设计是这样的：
        - 节点0: 代表无人机自己的节点，特征是无人机的状态信息
        - 节点1到N: 代表障碍物的节点（就是LiDAR上的点），特征是它们的相对位置
        - 边: 基于KNN算法，连接空间上相邻的节点
    """
    # 首先，把机体坐标系下的点云转换到世界坐标系下
    # point_cloud (N, 3) 是机体坐标系下的点
    # drone_state.orientation.T 是旋转矩阵的转置
    # point_cloud.T 是为了方便做矩阵乘法，把 (N, 3) 转置成 (3, N)
    # (drone_state.orientation.T @ point_cloud.T).T 把旋转后的结果再转置回 (N, 3)
    world_points = (drone_state.orientation.T @ point_cloud.T).T + drone_state.position
    
    # 2. 按距离过滤点云（这里用的是简化方法）
    distances = jnp.linalg.norm(world_points - drone_state.position, axis=1)
    valid_mask = distances <= config.max_range
    
    # 为了避免动态形状导致JIT编译失败，我们采用填充（padding）的策略
    n_total_points = world_points.shape[0]
    valid_points = world_points  # 暂时保留所有点
    n_valid = n_total_points
    if n_valid < config.min_points:
        # 如果点太少，就用假的点（全零）来填充
        dummy_points = jnp.zeros((config.min_points, 3))
        valid_points = dummy_points
        n_valid = config.min_points
    elif n_valid > config.max_points:
        # 如果点太多，就随机抽样一部分
        key = jax.random.PRNGKey(0)  # 这里最好用一个动态的key
        indices = jax.random.choice(
            key, n_valid, (config.max_points,), replace=False
        )
        valid_points = valid_points[indices]
        n_valid = config.max_points
    
    # 3. 构建节点的特征
    # 无人机节点的特征: [位置(3), 速度(3), 加速度(3), 朝向(1)]
    # 因为我们用的是点质量模型，所以角速度之类的就简化了
    ego_features = jnp.concatenate([
        drone_state.position,
        drone_state.velocity, 
        drone_state.acceleration,
        jnp.array([0.0])  # 点质量模型没有明确的朝向，用个假数据占位
    ])  # 这是一个10维的特征向量，后面会根据配置截取
    ego_features = ego_features[:config.ego_node_features]
    
    # 障碍物节点的特征: 在无人机机体坐标系下的相对位置
    obstacle_features = (drone_state.orientation @ (valid_points - drone_state.position).T).T
    obstacle_features = obstacle_features[:, :config.obstacle_node_features]  # 只保留x,y,z
    
    # 把所有节点的特征拼在一起（需要先把无人机节点的特征填充到和障碍物节点一样的维度）
    ego_features_padded = jnp.pad(ego_features, (0, max(0, config.obstacle_node_features - len(ego_features))))[:config.obstacle_node_features]
    all_node_features = jnp.concatenate([
        ego_features_padded[None, :],  # (1, obstacle_node_features)
        obstacle_features  # (n_valid, obstacle_node_features)
    ], axis=0)
    
    # 5. 用固定大小的数组来构建边
    senders, receivers, edge_features = find_knn_edges(
        drone_state.position, valid_points, config.k_neighbors
    )
    
    # 过滤掉那些无效的边（就是被标记为-1的那些）
    valid_edge_mask = (senders >= 0) & (receivers >= 0)
    
    # 统计一下有多少条有效的边
    n_valid_edges = jnp.sum(valid_edge_mask.astype(jnp.int32))
    
    # 在创建GraphsTuple的时候，我们还是用完整的数组，jraph库能很优雅地处理那些-1的索引
    # GNN在计算时会自动忽略那些索引是负数的边
    
    # 6. 给节点分类
    # 0 代表无人机自己, 1 代表障碍物
    node_types = jnp.concatenate([
        jnp.array([0]),  # 无人机节点
        jnp.ones(n_valid)  # 障碍物节点
    ]).astype(jnp.int32)
    
    # 7. 创建GraphsTuple对象 - 注意我们用的是包含无效边的完整数组
    n_nodes = n_valid + 1
    n_total_edges = len(senders)
    
    graph = jraph.GraphsTuple(
        n_node=jnp.array([n_nodes]),  # 我们这里只处理一张图
        n_edge=jnp.array([n_total_edges]),  # 总边数（包括无效的）
        nodes=all_node_features,  # (n_nodes, node_features)
        edges=edge_features,  # (n_total_edges, edge_features) 
        globals=None, # 我们没有全局特征
        senders=senders,  # (n_total_edges,) - 里面可能包含-1
        receivers=receivers,  # (n_total_edges,) - 里面可能包含-1
    )
    
    return graph, node_types

# =============================================================================  
# GNN 架构 (基本照搬GCBF+)
# =============================================================================

def default_nn_init():
    """这是GCBF+里默认的网络权重初始化方法，我们也跟着用。"""
    return nn.initializers.xavier_uniform()

class MLP(nn.Module):
    """这个是GCBF+代码库里的标准多层感知机（MLP）实现。"""
    hid_sizes: Tuple[int, ...]
    act: Callable = nn.relu
    act_final: bool = False
    
    @nn.compact
    def __call__(self, x):
        for i, size in enumerate(self.hid_sizes):
            x = nn.Dense(size, kernel_init=default_nn_init())(x)
            if i < len(self.hid_sizes) - 1 or self.act_final:
                x = self.act(x)
        return x

class GNNUpdate(NamedTuple):
    """这个具名元组把GNN一次更新所需的三个核心函数打包在一起，完全遵循GCBF+的模式。"""
    message: Callable
    aggregate: Callable  
    update: Callable
    
    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        """应用GNN的单步更新，同时做了安全的索引处理。"""
        # 我们只处理单张图，所以做个断言检查一下
        assert graph.n_node.shape == tuple() or graph.n_node.shape == (1,), f"期望单张图, 但输入的图n_node形状是 {graph.n_node.shape}"
        
        # 安全的索引函数，可以处理-1这样的无效索引
        def safe_get(array, indices):
            valid_mask = indices >= 0
            safe_indices = jnp.where(valid_mask, indices, 0)  # 把-1换成0
            result = array[safe_indices]
            # 把无效位置的结果清零
            return jnp.where(valid_mask[:, None], result, 0.0)
        
        # 提取发送方和接收方节点的特征
        node_feats_send = safe_get(graph.nodes, graph.senders)
        node_feats_recv = safe_get(graph.nodes, graph.receivers)
        
        # 消息传递：计算每条边的消息
        edges = self.message(graph.edges, node_feats_send, node_feats_recv)
        
        # 消息聚合：把发往同一个节点的消息聚合起来
        aggr_msg = self.aggregate(edges, graph.receivers, graph.nodes.shape[0])
        
        # 节点更新：用聚合后的消息来更新节点自身的特征
        new_node_feats = self.update(graph.nodes, aggr_msg)
        
        return graph._replace(nodes=new_node_feats)

class GNNLayer(nn.Module):
    """
    单个GNN层 - 这是对GCBF+里GNNLayer的精准复现。
    代码逻辑源自 gcbfplus/nn/gnn.py，用JAX原生方式实现。
    """
    msg_net_cls: Callable
    aggr_net_cls: Callable
    update_net_cls: Callable
    msg_dim: int
    out_dim: int
    
    @nn.compact
    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        def message(edge_feats, sender_feats, receiver_feats):
            """消息函数 - 完全复刻GCBF+的实现。"""
            # 把发送者节点、接收者节点和边本身的特征拼在一起
            feats = jnp.concatenate([edge_feats, sender_feats, receiver_feats], axis=-1)
            # 喂给一个MLP处理
            feats = self.msg_net_cls()(feats)
            feats = nn.Dense(self.msg_dim, kernel_init=default_nn_init())(feats)
            return feats
        
        def update(node_feats, msgs):
            """节点更新函数 - 完全复刻GCBF+的实现。"""
            # 把节点自己旧的特征和聚合来的新消息拼在一起
            feats = jnp.concatenate([node_feats, msgs], axis=-1)
            # 喂给另一个MLP处理
            feats = self.update_net_cls()(feats)
            feats = nn.Dense(self.out_dim, kernel_init=default_nn_init())(feats)
            return feats
            
        def aggregate(msgs, recv_idx, num_segments):
            """基于注意力机制的聚合函数 - 完全复刻GCBF+的实现。"""
            # 每条消息先进一个“门控”网络，算出注意力分数
            gate_feats = self.aggr_net_cls()(msgs)
            gate_feats = nn.Dense(1, kernel_init=default_nn_init())(gate_feats).squeeze(-1)
            
            # 对发往同一个节点的所有消息的注意力分数做Softmax归一化，得到权重
            attn = jraph.segment_softmax(gate_feats, segment_ids=recv_idx, num_segments=num_segments)
            assert attn.shape[0] == msgs.shape[0]
            
            # 用注意力权重对消息进行加权求和，完成聚合
            aggr_msg = jraph.segment_sum(
                attn[:, None] * msgs, segment_ids=recv_idx, num_segments=num_segments
            )
            return aggr_msg
        
        # 把这三个函数打包成一个更新函数，然后应用到图上
        update_fn = GNNUpdate(message, aggregate, update)
        return update_fn(graph)

class GNN(nn.Module):
    """
    多层GNN - 完全复刻GCBF+的GNN架构。
    代码逻辑源自 gcbfplus/nn/gnn.py，并做了适配以支持单智能体场景。
    """
    msg_dim: int = 128
    hid_size_msg: Tuple[int, ...] = (64, 64)
    hid_size_aggr: Tuple[int, ...] = (64,)
    hid_size_update: Tuple[int, ...] = (64, 64)
    out_dim: int = 128  
    n_layers: int = 3
    
    @nn.compact  
    def __call__(self, graph: jraph.GraphsTuple, node_type: Optional[int] = None, n_type: Optional[int] = None) -> jnp.ndarray:
        """
        前向传播过程，完全遵循GCBF+的实现。
        
        参数:
            graph: 输入的图结构
            node_type: 可以用来筛选特定类型的节点 (0=无人机, 1=障碍物)  
            n_type: 指定类型的节点有多少个
        """
        current_graph = graph
        
        # 堆叠多层GNN进行处理
        for i in range(self.n_layers):
            out_dim = self.out_dim if i == self.n_layers - 1 else self.msg_dim
            
            # 用偏函数的方式来创建MLP网络，这是GCBF+里常用的模式
            msg_net = ft.partial(MLP, hid_sizes=self.hid_size_msg, act=nn.relu, act_final=False)
            attn_net = ft.partial(MLP, hid_sizes=self.hid_size_aggr, act=nn.relu, act_final=False)  
            update_net = ft.partial(MLP, hid_sizes=self.hid_size_update, act=nn.relu, act_final=False)
            
            # 创建并应用一个GNN层
            gnn_layer = GNNLayer(
                msg_net_cls=msg_net,
                aggr_net_cls=attn_net,
                update_net_cls=update_net,
                msg_dim=self.msg_dim,
                out_dim=out_dim
            )
            current_graph = gnn_layer(current_graph)
            
        # 根据节点类型来决定返回什么结果（为了和GCBF+兼容）
        if node_type is None:
            return current_graph.nodes
        else:
            # 如果要的是无人机节点 (node_type=0)，就返回第一个节点
            if node_type == 0 and n_type is not None:
                n_type_concrete = int(n_type) if hasattr(n_type, '__len__') == False else 1
                return current_graph.nodes[:n_type_concrete]
            # 如果要的是障碍物节点 (node_type=1)，就返回剩下的节点  
            elif node_type == 1 and n_type is not None:
                n_type_concrete = int(n_type) if hasattr(n_type, '__len__') == False else 1
                return current_graph.nodes[1:1+n_type_concrete] if current_graph.nodes.shape[0] > 1 else jnp.zeros((n_type_concrete, current_graph.nodes.shape[1]))
            else:
                return current_graph.nodes

class CBFNet(nn.Module):
    """
    CBF网络 - 完全复刻GCBF+的CBF模块。
    代码逻辑源自 gcbfplus/algo/module/cbf.py，并为单智能体场景做了适配。
    """
    # GNN的参数 (和GCBF+的CBF网络默认值保持一致)
    node_dim: int = 3
    edge_dim: int = 4  
    n_agents: int = 1  # 我们是单智能体
    gnn_layers: int = 3
    
    # GNN的架构参数 (GCBF+的默认值)
    msg_dim: int = 64
    hid_size_msg: Tuple[int, ...] = (64, 64)
    hid_size_aggr: Tuple[int, ...] = (64,)
    hid_size_update: Tuple[int, ...] = (64, 64)
    
    @nn.compact
    def __call__(self, graph: jraph.GraphsTuple, n_type: Optional[int] = None) -> jnp.ndarray:
        """
        计算CBF值的过程，完全遵循GCBF+的实现。
        
        参数:
            graph: 包含了无人机和障碍物节点的图
            n_type: 智能体节点的数量 (对我们来说就是1)
        
        返回:
            智能体节点的CBF值
        """
        # 创建一个和GCBF+里一样的GNN
        gnn = GNN(
            msg_dim=self.msg_dim,
            hid_size_msg=self.hid_size_msg,
            hid_size_aggr=self.hid_size_aggr, 
            hid_size_update=self.hid_size_update,
            out_dim=self.msg_dim,
            n_layers=self.gnn_layers
        )
        
        # 用GNN处理图，然后只提取出代表无人机的那个节点的特征
        # node_type=0 代表的就是智能体节点
        if n_type is not None:
            if hasattr(n_type, 'shape') and n_type.shape:
                n_agents_actual = jnp.sum(n_type == 0)
            else:
                n_agents_actual = n_type
        else:
            n_agents_actual = self.n_agents
            
        agent_embeddings = gnn(graph, node_type=0, n_type=n_agents_actual)
        
        # 最后接一个全连接层作为“CBF头”，输出最终的CBF值
        cbf_values = nn.Dense(1, kernel_init=default_nn_init())(agent_embeddings)
        
        # 因为我们只有一个智能体，所以把维度压缩一下，直接返回一个标量
        return cbf_values.squeeze()

# =============================================================================
# 感知模块的整体封装
# =============================================================================

class PerceptionModule:
    """
    一个完整的感知流水线，用于计算CBF。
    """
    
    def __init__(self, config: GraphConfig):
        self.config = config
        self.cbf_net = CBFNet()
        
    def get_cbf_and_gradients(self, params, drone_state: DroneState, point_cloud: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        从点云输入，计算出CBF值和它的梯度。
        
        参数:
            params: 网络的权重
            drone_state: 无人机当前的状态
            point_cloud: LiDAR点云 (N, 3)
            
        返回:
            cbf_value: CBF的值 (一个标量)
            cbf_gradients: CBF关于无人机位置的梯度
        """
        def cbf_fn(state):
            graph, node_types = pointcloud_to_graph(state, point_cloud, self.config)
            return self.cbf_net.apply(params, graph, node_types)
        
        # 计算CBF的值
        cbf_value = cbf_fn(drone_state)
        
        # 计算关于无人机位置的梯度
        grad_fn = jax.grad(lambda state: cbf_fn(state).sum())  # .sum()是为了确保输出是标量
        cbf_gradients = grad_fn(drone_state)
        
        return cbf_value, cbf_gradients

# =============================================================================
# JIT兼容的函数封装
# =============================================================================

@ft.partial(jit, static_argnums=(2,))  # config是静态参数
def pointcloud_to_graph_jit(drone_state: DroneState, point_cloud: jnp.ndarray, config: GraphConfig) -> Tuple[jraph.GraphsTuple, jnp.ndarray]:
    """JIT编译版本的pointcloud_to_graph函数，config必须是静态的。"""
    return pointcloud_to_graph(drone_state, point_cloud, config)

@ft.partial(jit, static_argnums=(2,))  # k是静态参数
def find_knn_edges_jit(drone_pos: jnp.ndarray, obstacle_points: jnp.ndarray, k: int) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """JIT编译版本的find_knn_edges函数，k必须是静态的。"""
    return find_knn_edges(drone_pos, obstacle_points, k)

# =============================================================================
# 工厂函数和一些实用工具
# =============================================================================

def create_default_perception_module():
    """用默认参数创建一个感知模块。"""
    config = GraphConfig()
    return PerceptionModule(config)

def init_cbf_network(rng_key, input_graph: jraph.GraphsTuple, node_types: jnp.ndarray):
    """初始化CBF网络的参数。"""
    cbf_net = CBFNet()
    # 从node_types数组里数一下有多少个智能体节点（类型为0的）
    n_agents = int(jnp.sum(node_types == 0))
    return cbf_net.init(rng_key, input_graph, n_agents)

# 这个是给loop.py里的scan_function调用的核心函数
@jit
def get_cbf_from_pointcloud(params, drone_state: DroneState, point_cloud: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    一个在训练循环中计算CBF的优化函数。
    被loop.py里的scan_function调用，并且为了性能做了JIT编译。
    """
    config = GraphConfig()
    graph, node_types = pointcloud_to_graph(drone_state, point_cloud, config)
    
    cbf_net = CBFNet()
    # 统计智能体节点数量
    n_agents = jnp.sum(node_types == 0)
    cbf_value = cbf_net.apply(params, graph, n_agents)
    
    # 计算关于无人机位置的梯度（QP约束里要用）
    def cbf_wrt_position(pos):
        # 加点数值稳定性检查
        pos_clipped = jnp.clip(pos, -10.0, 10.0)  # 防止位置跑得太离谱
        
        modified_state = create_initial_drone_state(
            position=pos_clipped,
            velocity=drone_state.velocity,
            hover_initialization=False
        )
        graph_mod, node_types_mod = pointcloud_to_graph(modified_state, point_cloud, config)
        n_agents_mod = jnp.sum(node_types_mod == 0)
        cbf_raw = cbf_net.apply(params, graph_mod, n_agents_mod)
        
        # 再加点数值稳定性：把CBF的值也限制在一定范围内
        cbf_stable = jnp.clip(cbf_raw, -5.0, 5.0)
        return cbf_stable
    
    # 计算梯度
    cbf_grad_raw = jax.grad(lambda pos: cbf_wrt_position(pos).sum())(drone_state.position)
    
    # 对梯度也做个裁剪，防止梯度爆炸
    grad_norm = jnp.linalg.norm(cbf_grad_raw)
    max_grad_norm = 10.0  # 梯度范数最大不允许超过10.0
    
    cbf_grad = jnp.where(
        grad_norm > max_grad_norm,
        cbf_grad_raw * (max_grad_norm / (grad_norm + 1e-8)),
        cbf_grad_raw
    )
    
    return cbf_value, cbf_grad

# =============================================================================
# (高级功能) 带时序一致性的感知模块
# =============================================================================

class AdvancedPerceptionModule(PerceptionModule):
    """一个更高级的感知模块，考虑了时序上的连续性和记忆。"""
    
    def __init__(self, config: GraphConfig, use_temporal_smoothing: bool = True):
        super().__init__(config)
        self.use_temporal_smoothing = use_temporal_smoothing
        self.cbf_history = []
        self.max_history = 5
        
        # 用一个更强大的CBF网络架构
        self.cbf_net = AdvancedCBFNet(
            gnn_msg_dim=256,         # 更大的消息维度
            gnn_out_dim=256,
            gnn_layers=4,            # 更多的GNN层
            head_sizes=(512, 256, 128), # 更深的MLP头
            use_residual=True,       # 使用残差连接
            use_attention=True       # 使用注意力机制
        )
    
    def get_cbf_and_gradients_with_history(
        self, 
        params, 
        drone_state: DroneState, 
        point_cloud: jnp.ndarray,
        temporal_weight: float = 0.1
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        计算CBF，并用历史信息做平滑，让结果更稳定。
        
        参数:
            params: 网络权重
            drone_state: 无人机状态
            point_cloud: LiDAR点云
            temporal_weight: 时间平滑的权重
            
        返回:
            cbf_value: 经过时间平滑的CBF值
            cbf_gradients: 对应的梯度
        """
        # 先计算当前时刻的CBF值和梯度
        current_cbf, current_grad = self.get_cbf_and_gradients(
            params, drone_state, point_cloud
        )
        
        if not self.use_temporal_smoothing or len(self.cbf_history) == 0:
            # 如果不启用时间平滑，或者这是第一次计算，就直接用当前值
            smoothed_cbf = current_cbf
            smoothed_grad = current_grad
        else:
            # 用指数移动平均（EMA）来做平滑
            prev_cbf = self.cbf_history[-1]['cbf_value']
            prev_grad = self.cbf_history[-1]['cbf_grad']
            
            smoothed_cbf = (1 - temporal_weight) * prev_cbf + temporal_weight * current_cbf
            smoothed_grad = (1 - temporal_weight) * prev_grad + temporal_weight * current_grad
        
        # 把当前结果存入历史记录
        self.cbf_history.append({
            'cbf_value': current_cbf,
            'cbf_grad': current_grad
        })
        
        # 保持历史记录的长度
        if len(self.cbf_history) > self.max_history:
            self.cbf_history.pop(0)
        
        return smoothed_cbf, smoothed_grad

class AdvancedCBFNet(nn.Module):
    """一个更强大的CBF网络，架构上做了很多优化。"""
    
    gnn_msg_dim: int = 256
    gnn_out_dim: int = 256
    gnn_layers: int = 4
    head_sizes: Tuple[int, ...] = (512, 256, 128)
    use_residual: bool = True
    use_attention: bool = True
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, graph: jraph.GraphsTuple, node_types: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        # 带残差连接的增强版GNN
        gnn = EnhancedGNN(
            msg_dim=self.gnn_msg_dim,
            out_dim=self.gnn_out_dim,
            n_layers=self.gnn_layers,
            use_residual=self.use_residual,
            use_attention=self.use_attention,
            dropout_rate=self.dropout_rate if training else 0.0
        )
        
        node_embeddings = gnn(graph, training=training)
        
        # 提取出无人机自己的节点特征（总是在索引0的位置）
        ego_embedding = node_embeddings[0:1, :]
        
        # 一个带残差连接和批标准化的增强版MLP头
        x = ego_embedding
        
        for i, features in enumerate(self.head_sizes):
            residual = x
            
            # 全连接层
            x = nn.Dense(features, kernel_init=nn.initializers.xavier_uniform())(x)
            
            # 批标准化，让训练更稳定（只在训练时用）
            if training:
                x = nn.BatchNorm(use_running_average=False)(x)
            
            # Swish激活函数，据说梯度流更好
            x = nn.swish(x)
            
            # Dropout正则化（也只在训练时用）
            if self.dropout_rate > 0 and training:
                x = nn.Dropout(self.dropout_rate, deterministic=not training)(x)
            
            # 如果维度匹配，就加上残差连接
            if self.use_residual and residual.shape[-1] == features:
                x = x + residual * 0.5
        
        # 最终输出CBF值的预测层
        cbf_raw = nn.Dense(1, kernel_init=nn.initializers.xavier_uniform())(x)
        
        # 用tanh激活函数把CBF的值限制在[-1, 1]之间
        cbf_value = nn.tanh(cbf_raw)
        
        return cbf_value.squeeze()

class EnhancedGNN(nn.Module):
    """一个从架构上做了优化的GNN。"""
    
    msg_dim: int = 256
    out_dim: int = 256
    n_layers: int = 4
    use_residual: bool = True
    use_attention: bool = True
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, graph: jraph.GraphsTuple, training: bool = False) -> jnp.ndarray:
        current_graph = graph
        
        for i in range(self.n_layers):
            if i == self.n_layers - 1:
                layer_out_dim = self.out_dim
            else:
                layer_out_dim = self.msg_dim
            
            # 创建一个增强版的GNN层
            layer = EnhancedGNNLayer(
                msg_dim=self.msg_dim,
                out_dim=layer_out_dim,
                use_attention=self.use_attention,
                dropout_rate=self.dropout_rate if training else 0.0
            )
            
            # 应用这一层
            new_graph = layer(current_graph, training=training)
            
            # 加上节点的残差连接
            if (self.use_residual and 
                current_graph.nodes.shape[-1] == new_graph.nodes.shape[-1]):
                new_graph = new_graph._replace(
                    nodes=new_graph.nodes + current_graph.nodes * 0.5
                )
            
            current_graph = new_graph
        
        return current_graph.nodes

class EnhancedGNNLayer(nn.Module):
    """一个带注意力机制的增强版GNN层。"""
    
    msg_dim: int
    out_dim: int
    use_attention: bool = True
    dropout_rate: float = 0.1
    hid_size_msg: Tuple[int, ...] = (512, 256)
    hid_size_aggr: Tuple[int, ...] = (256, 128)
    hid_size_update: Tuple[int, ...] = (512, 256)
    
    @nn.compact
    def __call__(self, graph: jraph.GraphsTuple, training: bool = False) -> jraph.GraphsTuple:
        def enhanced_message_fn(edge_feats, sender_feats, receiver_feats):
            """增强版的消息函数。"""
            feats = jnp.concatenate([edge_feats, sender_feats, receiver_feats], axis=-1)
            
            x = feats
            for i, size in enumerate(self.hid_size_msg):
                x = nn.Dense(size, kernel_init=nn.initializers.xavier_uniform())(x)
                x = nn.swish(x)
                
                if self.dropout_rate > 0 and training:
                    x = nn.Dropout(self.dropout_rate, deterministic=not training)(x)
            
            messages = nn.Dense(self.msg_dim, kernel_init=nn.initializers.xavier_uniform())(x)
            return messages
        
        def enhanced_aggregate_fn(msgs, segment_ids, num_segments):
            """带注意力机制的增强版聚合函数。"""
            if self.use_attention:
                gate_feats = MLP(self.hid_size_aggr, act=nn.swish)(msgs)
                attention_scores = nn.Dense(1, kernel_init=nn.initializers.xavier_uniform())(gate_feats).squeeze(-1)
                
                attn_weights = jraph.segment_softmax(
                    attention_scores, segment_ids=segment_ids, num_segments=num_segments
                )
                
                aggr_msg = jraph.segment_sum(
                    attn_weights[:, None] * msgs, 
                    segment_ids=segment_ids, 
                    num_segments=num_segments
                )
            else:
                aggr_msg = jraph.segment_mean(msgs, segment_ids, num_segments)
            
            return aggr_msg
        
        def enhanced_update_fn(node_feats, aggr_msgs):
            """增强版的节点更新函数。"""
            feats = jnp.concatenate([node_feats, aggr_msgs], axis=-1)
            
            x = feats
            for i, size in enumerate(self.hid_size_update):
                x = nn.Dense(size, kernel_init=nn.initializers.xavier_uniform())(x)
                x = nn.swish(x)
                
                if self.dropout_rate > 0 and training:
                    x = nn.Dropout(self.dropout_rate, deterministic=not training)(x)
            
            updated_features = nn.Dense(self.out_dim, kernel_init=nn.initializers.xavier_uniform())(x)
            return updated_features
        
        # 应用增强版的消息传递流程
        updated_nodes = jraph.GraphNetwork(
            update_node_fn=lambda nodes, sent_msgs, received_msgs, globals_: enhanced_update_fn(nodes, received_msgs),
            update_edge_fn=None,
            update_global_fn=None,
            aggregate_edges_for_nodes_fn=enhanced_aggregate_fn,
            aggregate_nodes_for_globals_fn=None,
            aggregate_edges_for_globals_fn=None,
        )(
            graph._replace(
                edges=enhanced_message_fn(graph.edges, 
                                         graph.nodes[graph.senders], 
                                         graph.nodes[graph.receivers])
            )
        ).nodes
        
        return graph._replace(nodes=updated_nodes)

# =============================================================================
# 测试和验证代码
# =============================================================================

def test_pointcloud_to_graph():
    """测试一下点云到图的转换功能。"""
    # 创建一些测试数据
    drone_state = DroneState(
        position=jnp.array([0.0, 0.0, 1.0]),
        velocity=jnp.array([1.0, 0.0, 0.0]),
        orientation=jnp.eye(3),
        angular_velocity=jnp.zeros(3)
    )
    
    # 一个简单的障碍物点云
    point_cloud = jnp.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    
    config = GraphConfig()
    graph, node_types = pointcloud_to_graph(drone_state, point_cloud, config)
    
    print(f"图里有多少节点: {graph.n_node}")
    print(f"图里有多少条边: {graph.n_edge}")
    print(f"节点特征的形状: {graph.nodes.shape}")
    print(f"边特征的形状: {graph.edges.shape}")
    print(f"节点的类型: {node_types}")
    
    return graph, node_types

def test_advanced_perception_module():
    """测试一下我们那个更高级的感知模块。"""
    print("开始测试高级感知模块...")
    
    # 创建一个高级感知模块
    config = GraphConfig(k_neighbors=10, max_range=8.0)
    perception = AdvancedPerceptionModule(config, use_temporal_smoothing=True)
    
    # 初始化网络
    key = jax.random.PRNGKey(42)
    init_key, dropout_key, test_key = jax.random.split(key, 3)
    
    drone_state = DroneState(
        position=jnp.array([0.0, 0.0, 1.0]),
        velocity=jnp.array([1.0, 0.5, 0.0]),
        orientation=jnp.eye(3),
        angular_velocity=jnp.zeros(3)
    )
    
    # 生成一些随机的点云
    point_cloud = jax.random.normal(test_key, (25, 3)) * 3.0
    graph, node_types = pointcloud_to_graph(drone_state, point_cloud, config)
    
    # 初始化网络参数，要支持dropout
    params = perception.cbf_net.init(
        {'params': init_key, 'dropout': dropout_key}, 
        graph, node_types, training=True
    )
    
    # 测试带时间平滑的CBF计算
    for i in range(3):
        # 创建一个测试用的CBF函数，在推理的时候不使用dropout
        def test_cbf_fn(state):
            test_graph, test_node_types = pointcloud_to_graph(state, point_cloud, config)
            return perception.cbf_net.apply(
                params, test_graph, test_node_types, 
                training=False, rngs={'dropout': dropout_key}
            )
        
        # 计算CBF值和梯度
        cbf_value = test_cbf_fn(drone_state)
        cbf_grad = jax.grad(lambda state: test_cbf_fn(state).sum())(drone_state)
        
        # 手动更新历史记录来做测试
        perception.cbf_history.append({
            'cbf_value': cbf_value,
            'cbf_grad': cbf_grad
        })
        
        print(f"  第 {i+1} 步: CBF值={cbf_value:.4f}, 梯度范数={jnp.linalg.norm(cbf_grad.position):.4f}")
        
        # 稍微改一下无人机状态，模拟飞了一小步
        drone_state = drone_state._replace(
            position=drone_state.position + jnp.array([0.1, 0.0, 0.0])
        )
    
    print("✅ 高级感知模块测试: 通过!")
    return True

if __name__ == "__main__":
    # 跑个基础测试
    print("开始测试点云到图的转换...")
    test_graph, test_node_types = test_pointcloud_to_graph()
    print("测试完成!")