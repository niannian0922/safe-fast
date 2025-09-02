"""
安全敏捷飞行的BPTT（随时间反向传播）核心循环实现。

本模块实现了整个项目的核心训练循环，深度融合了以下两个前沿思想：
1. GCBF+ (MIT-REALM): 基于图神经网络的安全约束与多智能体协调机制。
2. DiffPhysDrone (SJTU): 利用时间梯度衰减技术的可微分物理仿真。

为了追求极致的性能，整个循环都构建在JAX的`lax.scan`之上，这不仅能让代码被高效编译，
还天然支持了以下关键特性：
- 梯度检查点技术，用计算换内存，有效降低长序列训练时的显存占用。
- 时间梯度衰减，稳定长时序BPTT的训练过程，防止梯度爆炸或消失。
- 通过图结构进行灵活的多智能体信息聚合与协调。
- 端到端贯穿物理引擎的可微分能力，实现高效的梯度传播。
"""

import jax
import jax.numpy as jnp
from jax import lax, random, grad, jit
from typing import NamedTuple, Tuple, Optional, Dict, Any, Callable
import chex
from flax import struct
import functools

# 从我自己的代码库中导入相关模块
from .physics import (
    DroneState, MultiAgentState, PhysicsParams,
    dynamics_step, multi_agent_dynamics_step,
    apply_temporal_gradient_decay, create_temporal_decay_schedule
)
from .policy import (
    PolicyParams, PolicyState, PolicyNetworkMLP, PolicyNetworkRNN,
    evaluate_policy_mlp, evaluate_policy_rnn, apply_control_constraints
)

@struct.dataclass
class ScanCarry:
    """
    为 `jax.lax.scan` 设计的状态携带结构体 (carry)。
    这个结构体专门为了和 main.py 的训练流程兼容，并且原生支持批处理 (batching)。
    它包含了所有需要在一个时间步传递到下一个时间步的状态信息。
    """
    drone_state: Any  # 无人机的完整物理状态。为了灵活性，这里可以是单个DroneState，也可以是批处理后的PyTree
    rnn_hidden_state: chex.Array  # 循环神经网络(RNN)的隐藏状态，维度为 [批大小, 隐藏层维度] 或单个 [隐藏层维度]
    step_count: chex.Array  # 当前的步数计数器，维度为 [批大小] 或单个标量
    cumulative_reward: chex.Array  # 累积奖励，维度为 [批大小] 或单个标量


@struct.dataclass
class ScanOutput:
    """
    `jax.lax.scan` 在每个时间步需要记录和输出的数据。
    这些输出最终会被堆叠成一个完整的轨迹，用于后续的损失计算。
    """
    # 基础的轨迹数据
    positions: chex.Array         # [3] - 无人机的位置
    velocities: chex.Array        # [3] - 无人机的速度
    control_commands: chex.Array  # [3] - 经过安全层处理后，最终施加的控制指令
    nominal_commands: chex.Array  # [3] - 策略网络输出的原始、名义上的控制指令
    step_loss: float              # 当前这一步的损失值
    safety_violation: float       # 安全违规的量化指标，例如CBF的负值部分

    # 为了兼容性和扩展性，预留了一些可以动态添加的字段
    drone_states: Optional[chex.Array] = None      # 完整的无人机状态向量
    cbf_values: Optional[chex.Array] = None        # CBF的值
    cbf_gradients: Optional[chex.Array] = None     # CBF的梯度
    safe_controls: Optional[chex.Array] = None     # 安全控制指令 (同 control_commands)
    obstacle_distances: Optional[chex.Array] = None# 与最近障碍物的距离
    trajectory_lengths: Optional[chex.Array] = None# 轨迹长度

# =============================================================================
# 为了与 main.py 兼容的接口层
# =============================================================================

def create_complete_bptt_scan_function(
    cbf_net_params, policy_params, safety_config, physics_params
) -> Callable:
    """
    创建一个集成了所有核心组件的、完整的BPTT扫描函数。

    这个函数是我整个方法论的核心实现，它把从输入到输出的整个流水线串联起来：
    输入 -> GNN感知 -> 策略网络 -> 可微分安全层 -> 物理引擎 -> BPTT梯度回传

    这里的实现严格遵循了最初设计的科研架构。
    """

    @jax.checkpoint  # 核心技术点：应用梯度检查点，用计算换取内存，支持更长的BPTT序列
    def scan_function_body(carry: ScanCarry, external_input):
        """
        这个函数是 `lax.scan` 循环体内的具体逻辑，完整实现了我设计的流水线：

        1.  感知模块 (GCBF+ GNN): GNN处理点云，输出CBF值和梯度。
        2.  策略模块: 策略网络根据当前状态，输出名义上的控制指令。
        3.  可微分安全层**: 使用qpax求解QP问题，将名义控制修正为安全控制。
        4.  物理引擎: JAX原生的物理引擎根据安全控制，计算出下一个状态。
        5.  梯度衰减 (DiffPhysDrone): 应用时间梯度衰减，稳定训练过程。
        """
        # 从 carry 中提取出当前的状态信息
        drone_state = carry.drone_state
        rnn_hidden = carry.rnn_hidden_state
        step = carry.step_count

        # 1. 感知模块 (GCBF+ GNN) 
        # 在实际应用中，点云数据应该来自传感器。为了演示，先手动创建一些合成的障碍物点。
        relative_positions = jnp.array([
            [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0], [0.5, 0.5, 1.0], [-0.5, -0.5, -1.0]
        ])  # (6, 3) - 合成的障碍物点

        # 为了避免循环导入问题，在函数内部局部导入感知模块的必要组件
        from .perception import pointcloud_to_graph, CBFNet, GraphConfig

        config = GraphConfig()
        # 将点云数据动态构建成图结构
        graph, node_types = pointcloud_to_graph(drone_state, relative_positions, config)

        # 使用GNN计算CBF值
        cbf_net = CBFNet()
        cbf_value = cbf_net.apply(cbf_net_params, graph, n_type=1) # n_type=1 表示只关心智能体节点

        # 为了构造QP约束，需要计算CBF关于无人机位置的梯度
        def cbf_wrt_position(pos):
            modified_state = drone_state.replace(position=pos)
            graph_mod, _ = pointcloud_to_graph(modified_state, relative_positions, config)
            return cbf_net.apply(cbf_net_params, graph_mod, n_type=1)

        cbf_gradients = jax.grad(cbf_wrt_position)(drone_state.position)

        #  2. 策略模块 =
        # 构建策略网络的输入观测向量
        observation = jnp.concatenate([
            drone_state.position,                                 # 当前位置
            drone_state.velocity,                                 # 当前速度
            external_input.get('target_velocity', jnp.zeros(3)),  # 目标速度 (从外部输入获取)
            jnp.array([cbf_value])                                # 将CBF值也作为输入，让策略感知安全状态
        ])

        # 策略网络前向传播（使用RNN来保持时序记忆）
        from .policy import PolicyNetworkRNN, PolicyParams

        # 使用默认参数创建策略网络
        policy_config = PolicyParams(
            hidden_dims=(32, 32),
            use_rnn=True,
            rnn_hidden_size=16
        )
        policy_net = PolicyNetworkRNN(params=policy_config)
        # RNN需要传入上一时刻的隐藏状态
        u_nominal, new_rnn_hidden = policy_net.apply(
            policy_params, observation[None, :], rnn_hidden  # 增加批处理维度
        )

        # 3. 可微分安全层 (qpax QP) 
        from .safety import SafetyLayer
        safety_layer = SafetyLayer(safety_config)
        # 将名义控制和CBF信息传入安全层，得到安全控制
        u_safe, qp_info = safety_layer.safety_filter(
            u_nominal, cbf_value, cbf_gradients, drone_state
        )

        #  4. 物理仿真 
        from .physics import dynamics_step, apply_temporal_gradient_decay_to_state

        # 将安全控制指令输入物理引擎，得到下一个状态
        next_drone_state = dynamics_step(drone_state, u_safe, physics_params)

        # 5. 时间梯度衰减 (DiffPhysDrone 核心思想) 
        if physics_params.enable_gradient_decay:
            # 在反向传播时，对流经这个状态的梯度进行衰减
            next_drone_state = apply_temporal_gradient_decay_to_state(
                next_drone_state, physics_params.gradient_decay_alpha
            )

        #  更新需要跨步传递的状态 (carry) 
        new_carry = ScanCarry(
            drone_state=next_drone_state,
            rnn_hidden_state=new_rnn_hidden,
            step_count=step + 1,
            cumulative_reward=carry.cumulative_reward # 累积奖励在这里只是传递，具体计算在损失函数中
        )

        # 准备需要记录的输出 
        scan_output = ScanOutput(
            # 基础轨迹信息
            positions=next_drone_state.position,
            velocities=next_drone_state.velocity,
            control_commands=u_safe,
            nominal_commands=u_nominal,
            step_loss=0.0,  # 这一步的损失将在 training.py 中统一计算
            safety_violation=jnp.maximum(-cbf_value, 0.0),  # CBF值为负则表示违反安全

            # 用于损失计算和调试的扩展数据
            drone_states=jnp.concatenate([
                next_drone_state.position,
                next_drone_state.velocity,
                jnp.zeros(6)  # 填充以兼容需要12维状态的函数
            ])[None, :],
            cbf_values=jnp.array([cbf_value])[None, :],
            cbf_gradients=cbf_gradients[None, :],
            safe_controls=u_safe[None, :],
            obstacle_distances=jnp.array([1.0])[None, :],  # 占位符：到障碍物的最小距离
            trajectory_lengths=jnp.array([jnp.linalg.norm(u_safe)])
        )

        return new_carry, scan_output

    return scan_function_body


def create_scan_function(
    gnn_perception, policy_network, safety_layer, physics_params
) -> Callable:
    """这是一个为了兼容旧版 main.py 接口而保留的包装函数"""
    # 为了兼容性，使用一些默认参数
    from .perception import CBFNet
    from .safety import SafetyConfig

    # 创建一些虚拟参数 (在实际训练中，这些参数应该来自训练状态)
    dummy_cbf_params = {}
    dummy_policy_params = {}
    safety_config = SafetyConfig()

    return create_complete_bptt_scan_function(
        dummy_cbf_params, dummy_policy_params, safety_config, physics_params
    )


def run_complete_trajectory_scan(
    scan_function,
    initial_carry,
    scan_inputs,
    params,
    physics_params,
    sequence_length
):
    """运行一个完整的轨迹扫描，同样是为了兼容 main.py"""
    # 将输入转换为BPTT循环需要的格式
    bptt_inputs = BPTTInputs(
        target_velocity=jnp.zeros(3),
        external_forces=jnp.zeros(3)
    )

    # 为序列中的每一步创建虚拟输入
    inputs_sequence = [bptt_inputs] * sequence_length

    # 使用 lax.scan 执行整个序列
    final_carry, outputs = lax.scan(
        lambda carry, inp: scan_function(carry, inp, params, physics_params),
        initial_carry,
        inputs_sequence,
        length=sequence_length
    )

    return final_carry, outputs


# =============================================================================
# BPTT 状态表示结构体
# =============================================================================

@struct.dataclass
class BPTTCarry:
    """
    这是JAX `lax.scan` BPTT循环专用的状态携带结构体 (carry)。
    它严格遵守JAX的纯函数编程范式，包含了所有需要跨时间步传递的状态信息。
    """
    # 物理状态
    drone_state: DroneState                         # 当前无人机的完整物理状态
    multi_agent_state: Optional[MultiAgentState]    # 多智能体状态 (如果适用)

    # 策略网络的状态
    policy_state: PolicyState                       # 主要包含RNN的隐藏状态和记忆

    # 控制指令的状态
    last_control: chex.Array                        # 上一步的控制指令，用于计算平滑度损失
    control_history: chex.Array                     # 控制指令的历史记录，用于平滑处理

    # 训练过程的状态
    step: int                                       # 当前是第几步
    accumulated_loss: float                         # 在当前序列中累积的总损失


@struct.dataclass
class BPTTInputs:
    """
    BPTT `scan` 函数在每个时间步的外部输入。
    这部分数据是在循环开始前就已知的、随时间变化的外部信息。
    """
    target_velocity: chex.Array                 # [3] - 当前时间步的目标速度
    external_forces: chex.Array                 # [3] - 外部扰动力 (可选，用于鲁棒性训练)
    obstacle_info: Optional[chex.Array] = None  # 动态障碍物的信息
    goal_position: Optional[chex.Array] = None  # 动态更新的目标点


@struct.dataclass
class BPTTOutputs:
    """
    BPTT `scan` 在每个时间步需要收集并输出的数据。
    这些数据会沿时间维度被堆叠起来，形成完整的轨迹，用于最终的损失计算。
    """
    # 状态轨迹
    positions: chex.Array                   # [3] - 无人机位置
    velocities: chex.Array                  # [3] - 无人机速度

    # 控制指令轨迹
    control_commands: chex.Array            # [3] - 最终施加的控制指令
    nominal_commands: chex.Array            # [3] - 安全层处理前的名义控制指令

    # 用于计算损失的中间量
    step_loss: float                        # 当前步的损失值
    safety_violation: float                 # 安全违规度量

    # 用于调试和分析的信息
    cbf_value: Optional[float] = None       # CBF的值 (如果计算了)
    constraint_active: Optional[bool] = None# 安全约束是否被激活


# =============================================================================
# 核心BPTT扫描函数的创建
# =============================================================================

def create_bptt_scan_function(
    policy_network: Any,              # 策略网络模型 (MLP 或 RNN)
    policy_params: chex.Array,        # 策略网络的参数
    physics_params: PhysicsParams,    # 物理仿真的参数
    policy_config: PolicyParams,      # 策略网络的配置
    loss_config: Dict[str, float],    # 损失函数各项的权重
    use_rnn: bool = True,             # 是否使用RNN策略
    enable_safety_layer: bool = False,# 是否启用安全层
    enable_gradient_decay: bool = True# 是否启用时间梯度衰减
) -> Callable:
    """
    创建一个包含了所有必要闭包的BPTT扫描函数。

    这个函数遵循了DiffPhysDrone的设计思想，创建一个可以被JIT编译、
    并支持时间梯度衰减的扫描函数。

    参数:
        policy_network: 神经网络策略模型。
        policy_params: 网络的权重参数。
        physics_params: 物理引擎的参数。
        policy_config: 策略网络的配置，如控制约束等。
        loss_config: 一个字典，包含各项损失的权重，例如: {'velocity': 1.0, 'safety': 2.0, ...}。
        use_rnn: 是否使用带记忆的RNN策略。
        enable_safety_layer: 是否启用基于CBF的安全过滤层。
        enable_gradient_decay: 是否启用时间梯度衰减。

    返回:
        一个配置好的、可用于BPTT的扫描函数。
    """

    def scan_step(carry: BPTTCarry, inputs: BPTTInputs) -> Tuple[BPTTCarry, BPTTOutputs]:
        """
        BPTT `scan` 循环的单步执行函数。

        为了能被JAX的变换(jit, grad)正确处理，这个函数必须是纯函数，不能有任何副作用。
        """
        # 从 carry 中分解出所有需要跨步传递的状态
        drone_state = carry.drone_state
        policy_state = carry.policy_state
        last_control = carry.last_control
        control_history = carry.control_history
        step = carry.step
        accumulated_loss = carry.accumulated_loss

        # === 构造观测向量 ===
        # 从无人机状态和外部输入中构建策略网络的观测向量
        # 这是一个简化的观测，在完整的实现中，这里可能包含深度图、LiDAR数据等
        observation = jnp.concatenate([
            drone_state.position,      # [3] 当前位置
            drone_state.velocity,      # [3] 当前速度
            inputs.target_velocity,    # [3] 目标速度
            last_control,              # [3] 上一步的控制指令，让策略知道自己的历史行为
        ])  # 总共: [12] 维的观测向量

        # === 策略网络评估 ===
        if use_rnn:
            # 使用带记忆的RNN策略
            raw_control, new_rnn_state = evaluate_policy_rnn(
                policy_network,
                policy_params,
                observation[None, :],          # 增加批处理维度
                policy_state.rnn_state,
                carry.control_history[None, :],# 增加批处理维度
                training=True
            )
            raw_control = raw_control[0]      # 移除批处理维度

            # 更新策略网络的状态
            new_policy_state = policy_state.replace(
                rnn_state=new_rnn_state[0],  # 移除批处理维度
                step_count=step + 1
            )
        else:
            # 使用无状态的MLP策略
            raw_control = evaluate_policy_mlp(
                policy_network,
                policy_params,
                observation[None, :],          # 增加批处理维度
                training=True
            )[0]                               # 移除批处理维度

            new_policy_state = policy_state.replace(step_count=step + 1)

        # === 控制指令处理 ===
        # 应用控制约束和时间平滑
        nominal_control = apply_control_constraints(
            raw_control,
            policy_config,
            last_control
        )

        # 集成安全层 (在MVP阶段2中简化)
        if enable_safety_layer:
            # 在完整实现中，这里会调用CBF-QP求解器
            # 在阶段2，我们直接使用名义控制作为安全控制
            safe_control = nominal_control
            cbf_value = 0.0  # 占位符
            constraint_active = False
        else:
            safe_control = nominal_control
            cbf_value = None
            constraint_active = None

        # === 物理仿真 ===
        # 应用动力学，计算下一步的状态
        new_drone_state = dynamics_step(
            drone_state,
            safe_control,
            physics_params
        )

        # === 计算损失 ===
        # 速度跟踪损失 (DiffPhysDrone中的主要目标)
        velocity_error = new_drone_state.velocity - inputs.target_velocity
        velocity_loss = jnp.sum(velocity_error ** 2)

        # 安全损失 (在阶段2中，简化为一个基本的高度约束)
        min_altitude = 0.5  # 最小安全高度
        safety_loss = jnp.maximum(0.0, min_altitude - new_drone_state.position[2]) ** 2
        safety_violation = float(new_drone_state.position[2] < min_altitude)

        # 控制能耗损失
        control_loss = jnp.sum(safe_control ** 2)

        # 控制平滑度损失
        control_change = safe_control - last_control
        smoothness_loss = jnp.sum(control_change ** 2)

        # 组合成当前步的总损失
        step_loss = (
            loss_config.get('velocity', 1.0) * velocity_loss +
            loss_config.get('safety', 2.0) * safety_loss +
            loss_config.get('control', 0.01) * control_loss +
            loss_config.get('smoothness', 0.001) * smoothness_loss
        )

        # 应用时间梯度衰减 (DiffPhysDrone的核心创新)
        if enable_gradient_decay:
            step_loss = apply_temporal_gradient_decay(
                step_loss,
                step,
                physics_params.gradient_decay_alpha,
                physics_params.dt
            )

        # === 更新状态 ===
        # 更新控制历史记录
        new_control_history = jnp.roll(control_history, shift=1, axis=0)
        new_control_history = new_control_history.at[0].set(safe_control)

        # 创建新的carry状态，传递给下一步
        new_carry = BPTTCarry(
            drone_state=new_drone_state,
            multi_agent_state=carry.multi_agent_state,  # 阶段2未使用
            policy_state=new_policy_state,
            last_control=safe_control,
            control_history=new_control_history,
            step=step + 1,
            accumulated_loss=accumulated_loss + step_loss
        )

        # 创建当前步的输出，用于记录
        outputs = BPTTOutputs(
            positions=new_drone_state.position,
            velocities=new_drone_state.velocity,
            control_commands=safe_control,
            nominal_commands=nominal_control,
            step_loss=step_loss,
            safety_violation=safety_violation,
            cbf_value=cbf_value,
            constraint_active=constraint_active
        )

        return new_carry, outputs

    return scan_step



# BPTT 执行函数


def execute_bptt_sequence(
    scan_fn: Callable,
    initial_carry: BPTTCarry,
    input_sequence: BPTTInputs,  # 维度: [序列长度, ...]
    sequence_length: int
) -> Tuple[BPTTCarry, BPTTOutputs]:
    """
    使用 JAX `lax.scan` 来执行BPTT序列。

    这是实际执行BPTT计算的核心函数，它能够完全地在物理仿真中进行微分。

    参数:
        scan_fn: 已经配置好的扫描函数。
        initial_carry: 初始的carry状态。
        input_sequence: 每个时间步的输入序列。
        sequence_length: 需要仿真的序列长度。

    返回:
        (final_carry, stacked_outputs) - 最终的carry状态和堆叠起来的轨迹输出。
    """
    # 执行扫描循环
    final_carry, outputs_sequence = lax.scan(
        scan_fn,
        initial_carry,
        input_sequence,
        length=sequence_length
    )

    return final_carry, outputs_sequence#outputs_sequence是lax.scan的产出，一个包含了从 t=1 到 t=N 所有时间步输出的完整轨迹


@functools.partial(jax.jit, static_argnames=['sequence_length', 'use_rnn'])
def jit_bptt_sequence(
    policy_network: Any,
    policy_params: chex.Array,
    physics_params: PhysicsParams,
    policy_config: PolicyParams,
    loss_config: Dict[str, float],
    initial_carry: BPTTCarry,
    input_sequence: BPTTInputs,
    sequence_length: int,
    use_rnn: bool = True
) -> Tuple[BPTTCarry, BPTTOutputs]:
    """
    JIT编译版本的BPTT序列执行函数。

    为了在训练中达到最高的性能，这个函数被JIT编译。
    """
    # 创建扫描函数,创建单步模拟函数
    scan_fn = create_bptt_scan_function(
        policy_network=policy_network,
        policy_params=policy_params,
        physics_params=physics_params,
        policy_config=policy_config,
        loss_config=loss_config,
        use_rnn=use_rnn,
        enable_safety_layer=False,  
        enable_gradient_decay=True
    )

    # BPTT,执行整个序列模拟
    return execute_bptt_sequence(
        scan_fn,
        initial_carry,
        input_sequence,
        sequence_length
    )


# 梯度检查点支持


@functools.partial(jax.checkpoint, prevent_cse=False)
def checkpointed_scan_step(scan_fn, carry, inputs):
    """
    梯度检查点版本的扫描步骤。

    通过不在前向传播中存储中间激活值，它可以在长序列训练中节省大量内存，
    这完全遵循了我最初的设计文档。
    """
    return scan_fn(carry, inputs)


def create_checkpointed_bptt_scan(
    *args,
    checkpoint_every: int = 5,
    **kwargs
) -> Callable:
    """
    创建一个带梯度检查点的BPTT扫描函数。

    参数:
        checkpoint_every: 每 N 步应用一次检查点。
        *args, **kwargs: 传递给 `create_bptt_scan_function` 的参数。

    返回:
        一个支持检查点的扫描函数。
    """
    base_scan_fn = create_bptt_scan_function(*args, **kwargs)

    def checkpointed_scan_fn(carry, inputs):
        # 通过条件判断，选择性地应用检查点
        if carry.step % checkpoint_every == 0:
            return checkpointed_scan_step(base_scan_fn, carry, inputs)
        else:
            return base_scan_fn(carry, inputs)

    return checkpointed_scan_fn


# =============================================================================
# 损失计算与分析
# =============================================================================

def compute_sequence_loss(
    outputs_sequence: BPTTOutputs,
    loss_config: Dict[str, float],
    sequence_length: int
) -> Tuple[float, Dict[str, float]]:
    """
    计算整个BPTT序列的总损失。

    参数:
        outputs_sequence: 从BPTT扫描中堆叠起来的输出。
        loss_config: 损失各项的权重。
        sequence_length: 序列的长度。

    返回:
        (total_loss, loss_breakdown) - 总损失和一个包含各项指标的字典。
    """
    # 从轨迹输出中提取损失相关的部分
    step_losses = outputs_sequence.step_loss              # 维度: [序列长度]
    safety_violations = outputs_sequence.safety_violation  # 维度: [序列长度]

    # 计算序列级别的指标
    total_loss = jnp.mean(step_losses)  # 在序列上取平均损失
    safety_violation_rate = jnp.mean(safety_violations)

    # 用于分析的额外指标
    # 例如，计算最终位置与目标位置的误差
    final_position_error = jnp.linalg.norm(
        outputs_sequence.positions[-1] - jnp.array([0.0, 0.0, 2.0])  # 假设目标是 [0,0,2]
    )

    # 平均控制指令的大小
    average_control_magnitude = jnp.mean(
        jnp.linalg.norm(outputs_sequence.control_commands, axis=-1)
    )

    # 控制平滑度 (指令的变化率)
    control_smoothness = jnp.mean(
        jnp.linalg.norm(
            jnp.diff(outputs_sequence.control_commands, axis=0), axis=-1
        )
    )

    loss_breakdown = {
        'total_loss': float(total_loss),
        'safety_violation_rate': float(safety_violation_rate),
        'final_position_error': float(final_position_error),
        'average_control_magnitude': float(average_control_magnitude),
        'control_smoothness': float(control_smoothness)
    }

    return total_loss, loss_breakdown


# =============================================================================
# 工具函数与辅助函数
# =============================================================================

def create_test_input_sequence(
    sequence_length: int,
    key: chex.PRNGKey,
    target_position: chex.Array = jnp.array([1.0, 1.0, 2.0])
) -> BPTTInputs:
    """创建一个用于验证和测试的输入序列。"""
    keys = random.split(key, sequence_length)

    # 创建一个简单的目标速度序列 (用于点对点导航)
    target_velocities = []
    for i in range(sequence_length):
        # 逐步接近目标点
        progress = i / sequence_length
        current_target_pos = progress * target_position
        # 计算一个简单的指向目标的期望速度
        target_vel = jnp.clip(
            (target_position - current_target_pos) * 0.5,
            -2.0, 2.0
        )
        target_velocities.append(target_vel)

    # 堆叠成一个序列
    target_velocities = jnp.stack(target_velocities, axis=0)

    # 创建其他输入 (暂时用零填充)
    external_forces = jnp.zeros((sequence_length, 3))

    return BPTTInputs(
        target_velocity=target_velocities,
        external_forces=external_forces
    )


def validate_bptt_implementation():
    """
    对BPTT实现进行全面的验证。

    这个函数验证了MVP阶段2所需的核心功能。
    """
    print("🧪 正在验证BPTT循环的实现...")

    # 导入测试所需的模块
    from core.physics import PhysicsParams, create_initial_drone_state
    from core.policy import PolicyParams, PolicyNetworkMLP, init_policy_state

    # 创建测试用的参数
    physics_params = PhysicsParams()
    policy_params = PolicyParams(
        hidden_dims=(64, 32),  # 测试时用小一点的网络
        rnn_hidden_size=32,
        use_rnn=False         # 为了简单，先从MLP开始
    )

    loss_config = {
        'velocity': 1.0,
        'safety': 2.0,
        'control': 0.01,
        'smoothness': 0.001
    }

    # 创建策略网络
    key = random.PRNGKey(42)
    policy_network = PolicyNetworkMLP(params=policy_params, output_dim=3)

    # 初始化网络参数
    dummy_obs = jnp.ones(12)  # 观测向量的维度
    network_params = policy_network.init(key, dummy_obs[None, :])

    # 创建初始状态
    initial_drone_state = create_initial_drone_state(jnp.array([0.0, 0.0, 1.0]))
    initial_policy_state = init_policy_state(policy_params, key)

    initial_carry = BPTTCarry(
        drone_state=initial_drone_state,
        multi_agent_state=None,
        policy_state=initial_policy_state,
        last_control=jnp.zeros(3),
        control_history=jnp.zeros((3, 3)),
        step=0,
        accumulated_loss=0.0
    )

    # 创建测试输入序列
    sequence_length = 10
    input_sequence = create_test_input_sequence(sequence_length, key)

    # 测试扫描函数的创建
    scan_fn = create_bptt_scan_function(
        policy_network=policy_network,
        policy_params=network_params,
        physics_params=physics_params,
        policy_config=policy_params,
        loss_config=loss_config,
        use_rnn=False
    )

    print("✅ 扫描函数创建成功")

    # 测试单步执行
    single_inputs = BPTTInputs(
        target_velocity=jnp.array([0.5, 0.0, 0.1]),
        external_forces=jnp.zeros(3)
    )

    new_carry, step_outputs = scan_fn(initial_carry, single_inputs)

    print(f"✅ 单步执行: {initial_carry.drone_state.position} -> {new_carry.drone_state.position}")
    print(f"   单步损失: {step_outputs.step_loss:.4f}")

    # 测试完整的BPTT序列
    final_carry, outputs_sequence = execute_bptt_sequence(
        scan_fn,
        initial_carry,
        input_sequence,
        sequence_length
    )

    print(f"✅ 完整BPTT序列执行完毕")
    print(f"   初始位置: {initial_carry.drone_state.position}")
    print(f"   最终位置: {final_carry.drone_state.position}")
    print(f"   总累积损失: {final_carry.accumulated_loss:.4f}")

    # 测试JIT编译
    jit_final_carry, jit_outputs_sequence = jit_bptt_sequence(
        policy_network,
        network_params,
        physics_params,
        policy_params,
        loss_config,
        initial_carry,
        input_sequence,
        sequence_length,
        use_rnn=False
    )

    print("✅ JIT编译成功")

    # 验证JIT编译结果的一致性
    position_diff = jnp.linalg.norm(
        final_carry.drone_state.position - jit_final_carry.drone_state.position
    )
    print(f"   JIT编译后位置差异: {position_diff:.10f}")
    assert position_diff < 1e-8, "JIT编译结果应该与非编译版本完全一致"

    # 测试损失计算
    total_loss, loss_breakdown = compute_sequence_loss(
        outputs_sequence,
        loss_config,
        sequence_length
    )

    print(f"✅ 损失计算成功")
    print(f"   总损失: {total_loss:.4f}")
    print(f"   安全违规率: {loss_breakdown['safety_violation_rate']:.2%}")
    print(f"   最终位置误差: {loss_breakdown['final_position_error']:.4f}")

    # 测试梯度计算 (这是阶段2最关键的测试!)
    def bptt_loss_fn(network_params):
        final_carry, outputs = jit_bptt_sequence(
            policy_network,
            network_params,
            physics_params,
            policy_params,
            loss_config,
            initial_carry,
            input_sequence,
            sequence_length,
            use_rnn=False
        )
        loss, _ = compute_sequence_loss(outputs, loss_config, sequence_length)
        return loss

    # 计算梯度
    loss_value = bptt_loss_fn(network_params)
    gradients = grad(bptt_loss_fn)(network_params)

    print(f"✅ **梯度成功流过BPTT！**")
    print(f"   损失值: {loss_value:.4f}")

    # 检查梯度属性
    def check_gradients(params, name=""):
        if isinstance(params, dict):
            for key, value in params.items():
                check_gradients(value, f"{name}/{key}")
        else:
            grad_norm = jnp.linalg.norm(params)
            grad_mean = jnp.mean(jnp.abs(params))
            grad_max = jnp.max(jnp.abs(params))

            print(f"   {name}: 范数={grad_norm:.6f}, 均值={grad_mean:.6f}, 最大值={grad_max:.6f}")

            assert jnp.all(jnp.isfinite(params)), f"梯度在 {name} 中包含NaN或Inf"
            assert grad_norm > 1e-8, f"梯度在 {name} 中过小 (梯度消失)"
            assert grad_norm < 1e3, f"梯度在 {name} 中过大 (梯度爆炸)"

    check_gradients(gradients, "policy_network")

    print("\n🎉 **MVP阶段2 BPTT循环验证：所有测试通过！**")
    print("✅ 扫描函数编译")
    print("✅ 单步执行")
    print("✅ 完整BPTT序列执行")
    print("✅ JIT编译与一致性")
    print("✅ 损失计算")
    print("✅ **端到端梯度成功流过物理引擎和策略网络**")
    print("\n🚀 **准备进入MVP阶段3：集成安全层！**")


if __name__ == "__main__":
    validate_bptt_implementation()