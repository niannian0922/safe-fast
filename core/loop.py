"""
安全敏捷飞行的BPTT（时间反向传播）循环实现。

本模块实现核心训练循环，结合：
1. GCBF+ (MIT-REALM): 基于图的安全约束和多智能体协调
2. DiffPhysDrone (SJTU): 时间梯度衰减和可微分物理学

循环使用JAX的lax.scan进行高效编译并支持：
- 用于内存效率的梯度检查点
- 用于训练稳定性的时间梯度衰减
- 通过图结构的多智能体协调
- 端到端可微分物理仿真
"""

import jax
import jax.numpy as jnp
from jax import lax, random, grad, jit
from typing import NamedTuple, Tuple, Optional, Dict, Any, Callable
import chex
from flax import struct
import functools

# 导入我们的实现
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
    """和main.py接口兼容的扫描携带状态，支持批处理"""
    drone_state: Any  # DroneState或批处理DroneState（灵活设计）
    rnn_hidden_state: chex.Array  # [batch_size, hidden_dim]或单个[hidden_dim]
    step_count: chex.Array  # [batch_size]或单个标量
    cumulative_reward: chex.Array  # [batch_size]或单个标量


@struct.dataclass 
class ScanOutput:
    """和main.py接口兼容的扫描输出"""
    # 基本轨迹数据
    positions: chex.Array  # [3] 位置
    velocities: chex.Array  # [3] 速度
    control_commands: chex.Array  # [3] 控制命令
    nominal_commands: chex.Array  # [3] 名义命令
    step_loss: float  # Step loss
    safety_violation: float  # Safety violations
    
    # 扩展兼容性字段（动态添加）
    drone_states: Optional[chex.Array] = None  # 完整状态向量
    cbf_values: Optional[chex.Array] = None  # CBF值
    cbf_gradients: Optional[chex.Array] = None  # CBF梯度
    safe_controls: Optional[chex.Array] = None  # 安全控制
    obstacle_distances: Optional[chex.Array] = None  # 障碍物距离
    trajectory_lengths: Optional[chex.Array] = None  # 轨迹长度

# =============================================================================
# MAIN.PY 兼容层
# =============================================================================

def create_complete_bptt_scan_function(
    cbf_net_params, policy_params, safety_config, physics_params
) -> Callable:
    """
    创建整合所有组件的完整BPTT扫描函数
    
    这是实现完整方法论的核心函数：
    输入 -> GNN感知 -> 策略 -> 安全层 -> 物理 -> BPTT
    
    严格遵循你方法论中描述的架构。
    """
    
    @jax.checkpoint  # 按照你的方法论应用梯度检查点
    def scan_function_body(carry: ScanCarry, external_input):
        """
        实现完整流水线的完整扫描函数：
        
        1. GCBF+ GNN感知进行CBF计算
        2. 策略网络进行名义控制  
        3. 使用qpax QP求解的安全层
        4. JAX原生物理仿真
        5. DiffPhysDrone时间梯度衰减
        """
        # 提取当前状态
        drone_state = carry.drone_state
        rnn_hidden = carry.rnn_hidden_state
        step = carry.step_count
        
        # === 1. 感知模块 (GCBF+ GNN) ===
        # 模拟演示用的点云（实际使用中来自传感器）
        # 暂时在无人机周围创建合成障碍物
        relative_positions = jnp.array([
            [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], 
            [0.0, -1.0, 0.0], [0.5, 0.5, 1.0], [-0.5, -0.5, -1.0]
        ])  # (6, 3) 合成障碍物
        
        # 局部导入感知函数避免循环导入
        from .perception import pointcloud_to_graph, CBFNet, GraphConfig
        
        config = GraphConfig()
        graph, node_types = pointcloud_to_graph(drone_state, relative_positions, config)
        
        # 使用GNN计算CBF值和梯度
        cbf_net = CBFNet()
        cbf_value = cbf_net.apply(cbf_net_params, graph, n_type=1)
        
        # 计算相对于无人机位置的CBF梯度
        def cbf_wrt_position(pos):
            modified_state = drone_state.replace(position=pos)
            graph_mod, _ = pointcloud_to_graph(modified_state, relative_positions, config)
            return cbf_net.apply(cbf_net_params, graph_mod, n_type=1)
        
        cbf_gradients = jax.grad(cbf_wrt_position)(drone_state.position)
        
        # === 2. 策略模块 ===
        # 创建观测向量
        observation = jnp.concatenate([
            drone_state.position,     # 当前位置
            drone_state.velocity,     # 当前速度  
            external_input.get('target_velocity', jnp.zeros(3)),  # 目标速度
            jnp.array([cbf_value])    # CBF值作为额外输入
        ])
        
        # 策略网络前向传播（使用RNN保持时间一致性）
        from .policy import PolicyNetworkRNN, PolicyParams
        
        # 使用默认参数创建策略网络
        policy_config = PolicyParams(
            hidden_dims=(32, 32),  # 匹配测试配置
            use_rnn=True,
            rnn_hidden_size=16
        )
        policy_net = PolicyNetworkRNN(params=policy_config)
        u_nominal, new_rnn_hidden = policy_net.apply(
            policy_params, observation[None, :], rnn_hidden  # 增加批次维度
        )
        
        # === 3. 安全层 (qpax QP) ===
        from .safety import SafetyLayer
        safety_layer = SafetyLayer(safety_config)
        u_safe, qp_info = safety_layer.safety_filter(
            u_nominal, cbf_value, cbf_gradients, drone_state
        )
        
        # === 4. 物理仿真 ===
        from .physics import dynamics_step, apply_temporal_gradient_decay_to_state
        
        # 应用控制并获取下一状态
        next_drone_state = dynamics_step(drone_state, u_safe, physics_params)
        
        # === 5. DIFFPHYSDRONE 时间梯度衰减 ===
        if physics_params.enable_gradient_decay:
            next_drone_state = apply_temporal_gradient_decay_to_state(
                next_drone_state, physics_params.gradient_decay_alpha
            )
        
        # === 更新携带状态 ===
        new_carry = ScanCarry(
            drone_state=next_drone_state,
            rnn_hidden_state=new_rnn_hidden,
            step_count=step + 1,
            cumulative_reward=carry.cumulative_reward
        )
        
        # === 创建输出记录 ===
        scan_output = ScanOutput(
            # 基本轨迹数据
            positions=next_drone_state.position,
            velocities=next_drone_state.velocity, 
            control_commands=u_safe,
            nominal_commands=u_nominal,
            step_loss=0.0,  # 将在training.py中计算
            safety_violation=jnp.maximum(-cbf_value, 0.0),  # CBF违反
            
            # 用于损失计算的扩展数据
            drone_states=jnp.concatenate([
                next_drone_state.position,
                next_drone_state.velocity,
                jnp.zeros(6)  # 为12维兼容性填充
            ])[None, :],
            cbf_values=jnp.array([cbf_value])[None, :],
            cbf_gradients=cbf_gradients[None, :],
            safe_controls=u_safe[None, :],
            obstacle_distances=jnp.array([1.0])[None, :],  # 到障碍物的最小距离
            trajectory_lengths=jnp.array([jnp.linalg.norm(u_safe)])
        )
        
        return new_carry, scan_output
    
    return scan_function_body


def create_scan_function(
    gnn_perception, policy_network, safety_layer, physics_params
) -> Callable:
    """为main.py的传统兼容性包装器"""
    # 为兼容性使用默认参数
    from .perception import CBFNet
    from .safety import SafetyConfig
    
    # 创建虚拟参数（实际使用中这些来自训练状态）
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
    """运行与main.py兼容的完整轨迹扫描"""
    # 转换为BPTTInputs格式
    bptt_inputs = BPTTInputs(
        target_velocity=jnp.zeros(3),
        external_forces=jnp.zeros(3)
    )
    
    # 为每个时间步创建虚拟输入
    inputs_sequence = [bptt_inputs] * sequence_length
    
    # 使用scan执行序列
    final_carry, outputs = lax.scan(
        lambda carry, inp: scan_function(carry, inp, params, physics_params),
        initial_carry,
        inputs_sequence,
        length=sequence_length
    )
    
    return final_carry, outputs


# =============================================================================
# BPTT 状态表示
# =============================================================================

@struct.dataclass
class BPTTCarry:
    """
    JAX lax.scan BPTT循环的携带状态。
    
    包含所有需要在时间步之间传递的状态，
    遵微jax的函数式编程要求。
    """
    # 物理状态
    drone_state: DroneState  # 当前无人机状态
    multi_agent_state: Optional[MultiAgentState]  # 如果适用的多代理状态
    
    # 策略状态  
    policy_state: PolicyState  # RNN隐藏状态和内存
    
    # 控制状态
    last_control: chex.Array  # 上一次控制命令
    control_history: chex.Array  # 用于平滑性的控制命令历史
    
    # 训练状态
    step: int  # 当前时间步
    accumulated_loss: float  # 序列上的累积损失


@struct.dataclass
class BPTTInputs:
    """
    BPTT扫描函数的每时间步输入。
    
    这表示每个时间步变化的外部输入。
    """
    target_velocity: chex.Array  # [3] 这个时间步的目标速度
    external_forces: chex.Array  # [3] 外部干扰（可选）
    obstacle_info: Optional[chex.Array] = None  # 动态障碍物信息
    goal_position: Optional[chex.Array] = None  # 动态目标更新


@struct.dataclass
class BPTTOutputs:
    """
    从BPTT扫描的每个时间步收集的输出。
    
    这些在时间维度上堆叠用于损失计算。
    """
    # 状态轨迹
    positions: chex.Array  # [3] 无人机位置
    velocities: chex.Array  # [3] 无人机速度
    
    # 控制轨迹
    control_commands: chex.Array  # [3] 应用的控制命令
    nominal_commands: chex.Array  # [3] 安全过滤器前的名义控制
    
    # 损失组件
    step_loss: float  # 这个时间步的损失
    safety_violation: float  # 安全违反指标
    
    # 调试信息
    cbf_value: Optional[float] = None  # 如果计算则为CBF值
    constraint_active: Optional[bool] = None  # 安全约束是否激活


# =============================================================================
# 核心BPTT扫描函数
# =============================================================================

def create_bptt_scan_function(
    policy_network: Any,  # Policy network (MLP or RNN)
    policy_params: chex.Array,  # Policy network parameters
    physics_params: PhysicsParams,  # Physics simulation parameters
    policy_config: PolicyParams,  # Policy configuration
    loss_config: Dict[str, float],  # Loss function weights
    use_rnn: bool = True,  # Whether using RNN policy
    enable_safety_layer: bool = False,  # Whether to enable safety filtering
    enable_gradient_decay: bool = True  # Whether to use temporal gradient decay
) -> Callable:
    """
    Create the BPTT scan function with all necessary closures.
    
    This follows the DiffPhysDrone methodology of creating a scan function
    that can be JIT compiled and used with temporal gradient decay.
    
    Args:
        policy_network: Neural network policy
        policy_params: Network parameters
        physics_params: Physics simulation parameters  
        policy_config: Policy configuration
        loss_config: Loss weights dict with keys: 'velocity', 'safety', 'control', 'smoothness'
        use_rnn: Whether using recurrent policy
        enable_safety_layer: Enable CBF-based safety filtering
        enable_gradient_decay: Enable temporal gradient decay
        
    Returns:
        Compiled scan function for BPTT
    """
    
    def scan_step(carry: BPTTCarry, inputs: BPTTInputs) -> Tuple[BPTTCarry, BPTTOutputs]:
        """
        Single step of the BPTT scan loop.
        
        This function must be pure (no side effects) to work with JAX transformations.
        """
        # Extract carry components
        drone_state = carry.drone_state
        policy_state = carry.policy_state
        last_control = carry.last_control
        control_history = carry.control_history
        step = carry.step
        accumulated_loss = carry.accumulated_loss
        
        # === OBSERVATION CONSTRUCTION ===
        # Create observation vector from drone state
        # This is a simplified observation - in full implementation would include
        # depth images, LiDAR data, etc.
        observation = jnp.concatenate([
            drone_state.position,      # [3] current position
            drone_state.velocity,      # [3] current velocity  
            inputs.target_velocity,    # [3] target velocity
            last_control,             # [3] previous control command
        ])  # Total: [12] observation vector
        
        # === POLICY EVALUATION ===
        if use_rnn:
            # RNN policy with memory
            raw_control, new_rnn_state = evaluate_policy_rnn(
                policy_network,
                policy_params,
                observation[None, :],  # Add batch dimension
                policy_state.rnn_state,
                carry.control_history[None, :],  # Add batch dimension
                training=True
            )
            raw_control = raw_control[0]  # Remove batch dimension
            
            # Update policy state
            new_policy_state = policy_state.replace(
                rnn_state=new_rnn_state[0],  # Remove batch dimension
                step_count=step + 1
            )
        else:
            # MLP policy (stateless)
            raw_control = evaluate_policy_mlp(
                policy_network,
                policy_params, 
                observation[None, :],  # Add batch dimension
                training=True
            )[0]  # Remove batch dimension
            
            new_policy_state = policy_state.replace(step_count=step + 1)
        
        # === CONTROL PROCESSING ===
        # Apply control constraints and smoothing
        nominal_control = apply_control_constraints(
            raw_control, 
            policy_config,
            last_control
        )
        
        # Safety layer integration (simplified for Stage 2)
        if enable_safety_layer:
            # In full implementation, this would call the CBF-QP solver
            # For Stage 2, we use the nominal control directly
            safe_control = nominal_control
            cbf_value = 0.0  # Placeholder
            constraint_active = False
        else:
            safe_control = nominal_control
            cbf_value = None
            constraint_active = None
        
        # === PHYSICS SIMULATION ===
        # Apply dynamics step
        new_drone_state = dynamics_step(
            drone_state,
            safe_control,
            physics_params
        )
        
        # === LOSS COMPUTATION ===
        # Velocity tracking loss (main objective from DiffPhysDrone)
        velocity_error = new_drone_state.velocity - inputs.target_velocity
        velocity_loss = jnp.sum(velocity_error ** 2)
        
        # Safety loss (basic altitude constraint for Stage 2)
        min_altitude = 0.5  # Minimum safe altitude
        safety_loss = jnp.maximum(0.0, min_altitude - new_drone_state.position[2]) ** 2
        safety_violation = float(new_drone_state.position[2] < min_altitude)
        
        # Control effort loss
        control_loss = jnp.sum(safe_control ** 2)
        
        # Control smoothness loss
        control_change = safe_control - last_control
        smoothness_loss = jnp.sum(control_change ** 2)
        
        # Combined step loss
        step_loss = (
            loss_config.get('velocity', 1.0) * velocity_loss +
            loss_config.get('safety', 2.0) * safety_loss +
            loss_config.get('control', 0.01) * control_loss +
            loss_config.get('smoothness', 0.001) * smoothness_loss
        )
        
        # Apply temporal gradient decay (DiffPhysDrone innovation)
        if enable_gradient_decay:
            step_loss = apply_temporal_gradient_decay(
                step_loss, 
                step, 
                physics_params.gradient_decay_alpha,
                physics_params.dt
            )
        
        # === STATE UPDATES ===
        # Update control history
        new_control_history = jnp.roll(control_history, shift=1, axis=0)
        new_control_history = new_control_history.at[0].set(safe_control)
        
        # Create new carry state
        new_carry = BPTTCarry(
            drone_state=new_drone_state,
            multi_agent_state=carry.multi_agent_state,  # Not used in Stage 2
            policy_state=new_policy_state,
            last_control=safe_control,
            control_history=new_control_history,
            step=step + 1,
            accumulated_loss=accumulated_loss + step_loss
        )
        
        # Create outputs for this timestep
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


# =============================================================================
# BPTT EXECUTION FUNCTIONS
# =============================================================================

def execute_bptt_sequence(
    scan_fn: Callable,
    initial_carry: BPTTCarry,
    input_sequence: BPTTInputs,  # [sequence_length, ...]
    sequence_length: int
) -> Tuple[BPTTCarry, BPTTOutputs]:
    """
    Execute BPTT sequence using JAX lax.scan.
    
    This is the core function that performs the actual BPTT computation
    with full differentiability through the physics simulation.
    
    Args:
        scan_fn: Compiled scan function
        initial_carry: Initial carry state
        input_sequence: Sequence of inputs for each timestep
        sequence_length: Length of sequence to simulate
        
    Returns:
        (final_carry, stacked_outputs)
    """
    # Execute scan loop
    final_carry, outputs_sequence = lax.scan(
        scan_fn,
        initial_carry,
        input_sequence,
        length=sequence_length
    )
    
    return final_carry, outputs_sequence


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
    JIT-compiled BPTT sequence execution.
    
    This function is JIT compiled for maximum performance during training.
    """
    # Create scan function
    scan_fn = create_bptt_scan_function(
        policy_network=policy_network,
        policy_params=policy_params,
        physics_params=physics_params,
        policy_config=policy_config,
        loss_config=loss_config,
        use_rnn=use_rnn,
        enable_safety_layer=False,  # Disabled for Stage 2
        enable_gradient_decay=True
    )
    
    # Execute BPTT
    return execute_bptt_sequence(
        scan_fn,
        initial_carry,
        input_sequence,
        sequence_length
    )


# =============================================================================
# GRADIENT CHECKPOINTING SUPPORT
# =============================================================================

@functools.partial(jax.checkpoint, prevent_cse=False)
def checkpointed_scan_step(scan_fn, carry, inputs):
    """
    Gradient checkpointed version of scan step.
    
    This saves memory during long sequences by not storing intermediate
    activations, following the methodology from your design document.
    """
    return scan_fn(carry, inputs)


def create_checkpointed_bptt_scan(
    *args, 
    checkpoint_every: int = 5,
    **kwargs
) -> Callable:
    """
    Create BPTT scan function with gradient checkpointing.
    
    Args:
        checkpoint_every: Apply checkpointing every N steps
        *args, **kwargs: Arguments for create_bptt_scan_function
        
    Returns:
        Checkpointed scan function
    """
    base_scan_fn = create_bptt_scan_function(*args, **kwargs)
    
    def checkpointed_scan_fn(carry, inputs):
        if carry.step % checkpoint_every == 0:
            return checkpointed_scan_step(base_scan_fn, carry, inputs)
        else:
            return base_scan_fn(carry, inputs)
    
    return checkpointed_scan_fn


# =============================================================================
# LOSS COMPUTATION AND ANALYSIS
# =============================================================================

def compute_sequence_loss(
    outputs_sequence: BPTTOutputs,
    loss_config: Dict[str, float],
    sequence_length: int
) -> Tuple[float, Dict[str, float]]:
    """
    Compute total loss over the BPTT sequence.
    
    Args:
        outputs_sequence: Stacked outputs from BPTT scan
        loss_config: Loss component weights
        sequence_length: Length of the sequence
        
    Returns:
        (total_loss, loss_breakdown)
    """
    # Extract loss components
    step_losses = outputs_sequence.step_loss  # [sequence_length]
    safety_violations = outputs_sequence.safety_violation  # [sequence_length]
    
    # Compute sequence-level metrics
    total_loss = jnp.mean(step_losses)  # Average loss over sequence
    safety_violation_rate = jnp.mean(safety_violations)
    
    # Additional metrics for analysis
    final_position_error = jnp.linalg.norm(
        outputs_sequence.positions[-1] - jnp.array([0.0, 0.0, 2.0])  # Target position
    )
    
    average_control_magnitude = jnp.mean(
        jnp.linalg.norm(outputs_sequence.control_commands, axis=-1)
    )
    
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
# UTILITIES AND HELPERS
# =============================================================================

def create_test_input_sequence(
    sequence_length: int,
    key: chex.PRNGKey,
    target_position: chex.Array = jnp.array([1.0, 1.0, 2.0])
) -> BPTTInputs:
    """Create a test input sequence for validation."""
    keys = random.split(key, sequence_length)
    
    # Create target velocity sequence (simple point-to-point navigation)
    target_velocities = []
    for i in range(sequence_length):
        # Gradually approach target position
        progress = i / sequence_length
        current_target_pos = progress * target_position
        target_vel = jnp.clip(
            (target_position - current_target_pos) * 0.5,
            -2.0, 2.0
        )
        target_velocities.append(target_vel)
    
    # Stack into sequence
    target_velocities = jnp.stack(target_velocities, axis=0)
    
    # Create other inputs (zeros for now)
    external_forces = jnp.zeros((sequence_length, 3))
    
    return BPTTInputs(
        target_velocity=target_velocities,
        external_forces=external_forces
    )


def validate_bptt_implementation():
    """
    Comprehensive validation of BPTT implementation.
    
    This validates the core functionality needed for Stage 2.
    """
    print("🧪 Validating BPTT Loop Implementation...")
    
    # Import required modules for testing
    from core.physics import PhysicsParams, create_initial_drone_state
    from core.policy import PolicyParams, PolicyNetworkMLP, init_policy_state
    
    # Create test parameters
    physics_params = PhysicsParams()
    policy_params = PolicyParams(
        hidden_dims=(64, 32),  # Smaller for testing
        rnn_hidden_size=32,
        use_rnn=False  # Start with MLP for simplicity
    )
    
    loss_config = {
        'velocity': 1.0,
        'safety': 2.0, 
        'control': 0.01,
        'smoothness': 0.001
    }
    
    # Create policy network
    key = random.PRNGKey(42)
    policy_network = PolicyNetworkMLP(params=policy_params, output_dim=3)
    
    # Initialize network parameters
    dummy_obs = jnp.ones(12)  # Observation dimension
    network_params = policy_network.init(key, dummy_obs[None, :])
    
    # Create initial states
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
    
    # Create test input sequence
    sequence_length = 10
    input_sequence = create_test_input_sequence(sequence_length, key)
    
    # Test scan function creation
    scan_fn = create_bptt_scan_function(
        policy_network=policy_network,
        policy_params=network_params,
        physics_params=physics_params,
        policy_config=policy_params,
        loss_config=loss_config,
        use_rnn=False
    )
    
    print("✅ Scan function created successfully")
    
    # Test single step execution
    single_inputs = BPTTInputs(
        target_velocity=jnp.array([0.5, 0.0, 0.1]),
        external_forces=jnp.zeros(3)
    )
    
    new_carry, step_outputs = scan_fn(initial_carry, single_inputs)
    
    print(f"✅ Single step: {initial_carry.drone_state.position} -> {new_carry.drone_state.position}")
    print(f"   Step loss: {step_outputs.step_loss:.4f}")
    
    # Test full BPTT sequence
    final_carry, outputs_sequence = execute_bptt_sequence(
        scan_fn,
        initial_carry,
        input_sequence,
        sequence_length
    )
    
    print(f"✅ Full BPTT sequence executed")
    print(f"   Initial position: {initial_carry.drone_state.position}")
    print(f"   Final position: {final_carry.drone_state.position}")
    print(f"   Total accumulated loss: {final_carry.accumulated_loss:.4f}")
    
    # Test JIT compilation
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
    
    print("✅ JIT compilation successful")
    
    # Verify JIT results match
    position_diff = jnp.linalg.norm(
        final_carry.drone_state.position - jit_final_carry.drone_state.position
    )
    print(f"   JIT position difference: {position_diff:.10f}")
    assert position_diff < 1e-8, "JIT results should match exactly"
    
    # Test loss computation
    total_loss, loss_breakdown = compute_sequence_loss(
        outputs_sequence, 
        loss_config, 
        sequence_length
    )
    
    print(f"✅ Loss computation successful")
    print(f"   Total loss: {total_loss:.4f}")
    print(f"   Safety violation rate: {loss_breakdown['safety_violation_rate']:.2%}")
    print(f"   Final position error: {loss_breakdown['final_position_error']:.4f}")
    
    # Test gradient computation (the critical test for Stage 2!)
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
    
    # Compute gradients
    loss_value = bptt_loss_fn(network_params)
    gradients = grad(bptt_loss_fn)(network_params)
    
    print(f"✅ **GRADIENT FLOW THROUGH BPTT SUCCESSFUL!**")
    print(f"   Loss value: {loss_value:.4f}")
    
    # Check gradient properties
    def check_gradients(params, name=""):
        if isinstance(params, dict):
            for key, value in params.items():
                check_gradients(value, f"{name}/{key}")
        else:
            grad_norm = jnp.linalg.norm(params)
            grad_mean = jnp.mean(jnp.abs(params))
            grad_max = jnp.max(jnp.abs(params))
            
            print(f"   {name}: norm={grad_norm:.6f}, mean={grad_mean:.6f}, max={grad_max:.6f}")
            
            assert jnp.all(jnp.isfinite(params)), f"Gradients contain NaN/Inf in {name}"
            assert grad_norm > 1e-8, f"Gradients too small in {name} (vanishing gradient)"
            assert grad_norm < 1e3, f"Gradients too large in {name} (exploding gradient)"
    
    check_gradients(gradients, "policy_network")
    
    print("\n🎉 **STAGE 2 BPTT LOOP VALIDATION: ALL TESTS PASSED!**")
    print("✅ Scan function compilation")
    print("✅ Single step execution") 
    print("✅ Full BPTT sequence execution")
    print("✅ JIT compilation and consistency")
    print("✅ Loss computation")
    print("✅ **END-TO-END GRADIENT FLOW THROUGH PHYSICS AND POLICY**")
    print("\n🚀 **Ready for Stage 3: Safety Layer Integration!**")


if __name__ == "__main__":
    validate_bptt_implementation()