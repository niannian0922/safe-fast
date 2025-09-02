"""
安全敏捷飞行系统的完整训练框架。

本模块实现了我设计的综合训练方法，它深度融合了以下几个关键思想：
1. GCBF+ 的CBF损失函数形式，用于保证安全性。
2. DiffPhysDrone 的物理驱动损失，用于提升飞行效率和性能。
3. 结合了时间和空间信息的时间梯度衰减机制，用于稳定长时序训练。
4. 完全基于JAX原生实现，以达到最高的计算性能。

核心组件：
- 一个多目标的损失函数，它同时权衡了效率、安全性和控制平滑度。
- 对CBF约束的违反及其导数条件的惩罚项。
- 源自DiffPhysDrone思想的、物理驱动的损失项 (如速度跟踪)。
- 用于智能地平衡不同优化目标的多梯度下降算法（MGDA）。
"""

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import optax
from typing import Dict, Tuple, NamedTuple, Optional
import chex
from dataclasses import dataclass

# 从我自己的代码库中导入相关模块
from .physics import DroneState, PhysicsParams, dynamics_step
from .perception import PerceptionModule, DroneState as PerceptionDroneState
from .policy import PolicyNetworkMLP
from .safety import SafetyLayer, SafetyConfig, QSolutionInfo, compute_safety_loss
from .loop import ScanCarry, ScanOutput


# =============================================================================
# 损失函数的配置与数据结构
# =============================================================================

@dataclass
class LossConfig:
    """这是一个数据类，用来集中管理损失函数中所有组件的权重系数。"""
    # GCBF+ 相关的损失系数
    cbf_violation_coef: float = 5.0       # 对违反CBF安全约束的惩罚权重
    cbf_derivative_coef: float = 3.0      # 对不满足CBF导数条件的惩罚权重
    cbf_boundary_coef: float = 2.0        # 对CBF在安全边界附近平滑性的惩罚权重

    # DiffPhysDrone 物理驱动相关的损失系数
    velocity_tracking_coef: float = 1.0   # 对速度跟踪误差的惩罚权重
    collision_avoidance_coef: float = 4.0 # 对碰撞的惩罚权重
    control_smoothness_coef: float = 0.1  # 对控制指令变化的正则化，鼓励平滑控制
    control_jerk_coef: float = 0.05       # 对控制指令变化率的惩罚 (急动)，鼓励更平滑的控制

    # 效率相关的损失系数
    goal_reaching_coef: float = 2.0       # 对与目标点距离的惩罚权重
    time_efficiency_coef: float = 0.1     # 对到达目标时间的惩罚权重

    # 安全系统相关的损失系数
    safety_layer_coef: float = 1.0        # 对安全层QP求解失败的惩罚
    emergency_coef: float = 100.0         # 对触发紧急制动的巨大惩罚

    # 梯度衰减相关的参数
    temporal_decay_alpha: float = 0.95    # 基础的时间梯度衰减因子
    spatial_decay_enable: bool = True     # 是否启用空间自适应衰减
    spatial_decay_range: float = 2.0      # 空间衰减的距离范围


class LossMetrics(NamedTuple):
    """一个具名元组，用于在训练过程中记录和监控所有详细的损失指标。"""
    # 总的损失分类
    total_loss: chex.Array
    efficiency_loss: chex.Array
    safety_loss: chex.Array
    control_loss: chex.Array

    # GCBF+ 相关的具体指标
    cbf_violation: chex.Array
    cbf_derivative: chex.Array
    cbf_boundary: chex.Array

    # DiffPhysDrone 相关的具体指标
    velocity_tracking: chex.Array
    collision_penalty: chex.Array
    control_smoothness: chex.Array
    control_jerk: chex.Array

    # 效率指标
    goal_distance: chex.Array
    time_penalty: chex.Array

    # 安全指标
    safety_violations: chex.Array
    emergency_activations: chex.Array
    qp_success_rate: chex.Array

    # 训练动态指标
    gradient_norm: chex.Array
    temporal_decay_factor: chex.Array


# =============================================================================
# 带有课程学习和多目标优化的先进训练框架
# =============================================================================

class AdvancedTrainingFramework:
    """一个先进的训练框架，集成了课程学习和多目标优化策略。"""

    def __init__(self, loss_config: LossConfig, use_curriculum: bool = True):
        self.loss_config = loss_config
        self.use_curriculum = use_curriculum
        self.training_stage = 0  # 0: 效率优先, 1: 安全感知, 2: 联合优化
        self.loss_history = {'total': [], 'safety': [], 'efficiency': []}
        # 定义从一个阶段进入下一个阶段的性能阈值
        self.curriculum_thresholds = {
            'stage_1_to_2': {'min_efficiency': 0.7, 'max_safety_violations': 5},
            'stage_2_to_3': {'min_efficiency': 0.85, 'max_safety_violations': 2}
        }

    def compute_comprehensive_loss_with_curriculum(
        self,
        scan_outputs: ScanOutput,
        target_positions: chex.Array,
        target_velocities: chex.Array,
        physics_params: PhysicsParams,
        training_step: int = 0
    ) -> Tuple[chex.Array, LossMetrics, dict]:
        """
        计算带有课程学习自适应调整的综合损失。

        三阶段课程学习：
        阶段0: 专注于基本的控制和目标到达，此时安全约束非常宽松。
        阶段1: 逐步引入和加强安全约束，让网络学会感知安全。
        阶段2: 进行完整的安全约束和效率的联合优化。
        """
        # 根据训练步数和历史性能，确定当前处于哪个课程阶段
        current_stage = self._get_current_curriculum_stage(training_step)

        # 根据当前阶段，动态调整各项损失的权重
        adapted_config = self._adapt_loss_config(current_stage)

        # 使用调整后的权重计算基础的综合损失
        total_loss, metrics = compute_comprehensive_loss(
            scan_outputs, target_positions, target_velocities,
            adapted_config, physics_params
        )

        # 记录课程学习相关的信息，用于监控
        curriculum_info = {
            'current_stage': current_stage,
            'stage_progress': self._get_stage_progress(training_step, current_stage),
            'adapted_weights': self._get_weight_summary(adapted_config)
        }

        # 更新训练历史，用于后续的阶段晋升判断
        self._update_training_history(total_loss, metrics)

        # 检查是否满足进入下一阶段的条件
        stage_advanced = self._check_stage_advancement(metrics, current_stage)
        if stage_advanced:
            print(f"🎓 课程学习已从阶段 {current_stage} 晋升到 {current_stage + 1}")
            self.training_stage = current_stage + 1
            curriculum_info['stage_advanced'] = True

        return total_loss, metrics, curriculum_info

    def _get_current_curriculum_stage(self, training_step: int) -> int:
        """根据训练步数和历史性能确定当前课程阶段。"""
        if not self.use_curriculum:
            return 2  # 如果不使用课程学习，直接进入最终阶段

        # 手动覆盖自动晋升的机制
        if hasattr(self, 'manual_stage_override'):
            return self.manual_stage_override

        # 基于训练步数的自动阶段划分
        stage_duration = 3000  # 每个阶段的持续步数
        automatic_stage = min(2, training_step // stage_duration)

        # 采用手动和自动阶段中更高级的那个
        return max(self.training_stage, automatic_stage)

    def _adapt_loss_config(self, stage: int) -> LossConfig:
        """根据当前课程阶段，动态调整损失配置。"""
        base_config = self.loss_config

        if stage == 0:  # 阶段0：效率优先
            return LossConfig(
                cbf_violation_coef=base_config.cbf_violation_coef * 0.2,      # 安全约束非常宽松
                cbf_derivative_coef=base_config.cbf_derivative_coef * 0.1,
                cbf_boundary_coef=base_config.cbf_boundary_coef * 0.1,
                velocity_tracking_coef=base_config.velocity_tracking_coef * 1.5, # 专注于控制
                collision_avoidance_coef=base_config.collision_avoidance_coef * 0.3,
                control_smoothness_coef=base_config.control_smoothness_coef * 2.0, # 鼓励平滑控制
                goal_reaching_coef=base_config.goal_reaching_coef * 2.0,      # 专注于到达目标
                safety_layer_coef=base_config.safety_layer_coef * 0.1,
                emergency_coef=base_config.emergency_coef * 0.5,
                temporal_decay_alpha=base_config.temporal_decay_alpha,
                spatial_decay_enable=base_config.spatial_decay_enable,
                spatial_decay_range=base_config.spatial_decay_range
            )
        elif stage == 1:  # 阶段1：安全感知
            return LossConfig(
                cbf_violation_coef=base_config.cbf_violation_coef * 0.7,      # 适度的安全权重
                cbf_derivative_coef=base_config.cbf_derivative_coef * 0.6,
                cbf_boundary_coef=base_config.cbf_boundary_coef * 0.6,
                velocity_tracking_coef=base_config.velocity_tracking_coef * 1.2,
                collision_avoidance_coef=base_config.collision_avoidance_coef * 0.8,
                control_smoothness_coef=base_config.control_smoothness_coef * 1.2,
                goal_reaching_coef=base_config.goal_reaching_coef * 1.5,
                safety_layer_coef=base_config.safety_layer_coef * 0.7,
                emergency_coef=base_config.emergency_coef * 0.8,
                temporal_decay_alpha=base_config.temporal_decay_alpha,
                spatial_decay_enable=base_config.spatial_decay_enable,
                spatial_decay_range=base_config.spatial_decay_range
            )
        else:  # 阶段2：联合优化
            return base_config

    def _get_stage_progress(self, training_step: int, current_stage: int) -> float:
        """获取在当前课程阶段内的训练进度。"""
        stage_duration = 3000
        stage_start = current_stage * stage_duration
        progress = min(1.0, (training_step - stage_start) / stage_duration)
        return progress

    def _get_weight_summary(self, config: LossConfig) -> dict:
        """获取当前损失权重的摘要，用于日志记录。"""
        return {
            'safety_weight': config.cbf_violation_coef,
            'efficiency_weight': config.goal_reaching_coef,
            'control_weight': config.control_smoothness_coef
        }

    def _update_training_history(self, total_loss: chex.Array, metrics: LossMetrics):
        """更新训练历史，用于后续的课程决策。"""
        self.loss_history['total'].append(float(total_loss))
        self.loss_history['safety'].append(float(metrics.safety_loss))
        self.loss_history['efficiency'].append(float(metrics.efficiency_loss))

        # 保持历史记录的长度是可控的
        max_history = 1000
        for key in self.loss_history:
            if len(self.loss_history[key]) > max_history:
                self.loss_history[key] = self.loss_history[key][-max_history//2:]

    def _check_stage_advancement(self, metrics: LossMetrics, current_stage: int) -> bool:
        """检查课程是否应该晋升到下一个阶段。"""
        if current_stage >= 2:  # 已经是最终阶段
            return False

        # 需要足够的训练历史来进行判断
        if len(self.loss_history['total']) < 100:
            return False

        # 计算最近一段时间的性能指标
        recent_window = 50
        recent_safety_violations = jnp.mean(jnp.array(self.loss_history['safety'][-recent_window:]))
        # 效率的度量是1/(1+loss)，loss越小效率越高
        recent_efficiency = 1.0 / (1.0 + jnp.mean(jnp.array(self.loss_history['efficiency'][-recent_window:])))

        # 检查是否满足晋升标准
        if current_stage == 0:  # 从阶段 0 -> 1
            criteria = self.curriculum_thresholds['stage_1_to_2']
            return (recent_efficiency >= criteria['min_efficiency'] * 0.8 and  # 为阶段1放宽标准
                   recent_safety_violations <= criteria['max_safety_violations'] * 2.0)
        elif current_stage == 1:  # 从阶段 1 -> 2
            criteria = self.curriculum_thresholds['stage_2_to_3']
            return (recent_efficiency >= criteria['min_efficiency'] and
                   recent_safety_violations <= criteria['max_safety_violations'])

        return False

class MultiObjectiveOptimizer:
    """一个使用梯度平衡技术的多目标优化器。"""

    def __init__(self, balance_method: str = 'adaptive_weights'):
        self.balance_method = balance_method
        self.objective_history = {'safety': [], 'efficiency': [], 'control': []}
        self.weight_adaptation_rate = 0.01
        self.current_weights = {'safety': 1.0, 'efficiency': 1.0, 'control': 1.0}

    def compute_balanced_loss(
        self,
        safety_loss: chex.Array,
        efficiency_loss: chex.Array,
        control_loss: chex.Array,
        training_step: int = 0
    ) -> Tuple[chex.Array, dict]:
        """计算一个经过平衡的多目标损失。"""

        if self.balance_method == 'adaptive_weights':
            return self._adaptive_weight_balancing(safety_loss, efficiency_loss, control_loss)
        elif self.balance_method == 'gradient_cosine':
            # 这是一个简化的实现，完整的MGDA需要计算梯度
            return self._gradient_cosine_balancing(safety_loss, efficiency_loss, control_loss)
        else:
            # 默认使用简单的加权求和
            weights = self.current_weights
            total_loss = (weights['safety'] * safety_loss +
                         weights['efficiency'] * efficiency_loss +
                         weights['control'] * control_loss)
            balance_info = {'method': 'fixed_weights', 'weights': weights}
            return total_loss, balance_info

    def _adaptive_weight_balancing(
        self,
        safety_loss: chex.Array,
        efficiency_loss: chex.Array,
        control_loss: chex.Array
    ) -> Tuple[chex.Array, dict]:
        """基于损失大小的自适应权重平衡。"""

        # 更新历史记录
        self.objective_history['safety'].append(float(safety_loss))
        self.objective_history['efficiency'].append(float(efficiency_loss))
        self.objective_history['control'].append(float(control_loss))

        # 基于最近的损失大小计算自适应权重
        window_size = min(50, len(self.objective_history['safety']))
        if window_size > 10:
            recent_safety = jnp.mean(jnp.array(self.objective_history['safety'][-window_size:]))
            recent_efficiency = jnp.mean(jnp.array(self.objective_history['efficiency'][-window_size:]))
            recent_control = jnp.mean(jnp.array(self.objective_history['control'][-window_size:]))

            # 反向加权：给较小的损失更大的权重，以平衡各个目标
            total_magnitude = recent_safety + recent_efficiency + recent_control + 1e-6
            target_weights = {
                'safety': (recent_efficiency + recent_control) / (2 * total_magnitude) * 3,
                'efficiency': (recent_safety + recent_control) / (2 * total_magnitude) * 3,
                'control': (recent_safety + recent_efficiency) / (2 * total_magnitude) * 3
            }

            # 平滑地调整权重
            for key in self.current_weights:
                self.current_weights[key] = (
                    (1 - self.weight_adaptation_rate) * self.current_weights[key] +
                    self.weight_adaptation_rate * target_weights[key]
                )

        # 计算平衡后的损失
        weights = self.current_weights
        total_loss = (weights['safety'] * safety_loss +
                     weights['efficiency'] * efficiency_loss +
                     weights['control'] * control_loss)

        balance_info = {
            'method': 'adaptive_weights',
            'weights': weights,
            'weight_adaptation_rate': self.weight_adaptation_rate
        }

        return total_loss, balance_info

    def _gradient_cosine_balancing(
        self,
        safety_loss: chex.Array,
        efficiency_loss: chex.Array,
        control_loss: chex.Array
    ) -> Tuple[chex.Array, dict]:
        """基于梯度余弦相似度的平衡方法 (简化实现)。"""
        # 完整的实现需要计算每个损失对参数的梯度，这里用损失大小来近似
        loss_magnitudes = jnp.array([safety_loss, efficiency_loss, control_loss])

        # 归一化到单位尺度
        normalized_losses = loss_magnitudes / (jnp.linalg.norm(loss_magnitudes) + 1e-8)

        # 以均等权重为基准
        equal_weights = jnp.ones(3) / 3.0

        # 根据与均等贡献的偏差来调整权重
        weights = equal_weights + 0.1 * (equal_weights - normalized_losses)
        weights = jnp.maximum(weights, 0.1)  # 保证最小权重
        weights = weights / jnp.sum(weights)  # 归一化

        total_loss = jnp.sum(weights * loss_magnitudes)

        balance_info = {
            'method': 'gradient_cosine',
            'weights': {'safety': weights[0], 'efficiency': weights[1], 'control': weights[2]},
            'normalized_losses': normalized_losses
        }

        return total_loss, balance_info

def compute_cbf_violation_loss(
    h_values: chex.Array,
    h_dots: chex.Array,
    alpha: float = 1.0
) -> Tuple[chex.Array, Dict]:
    """
    计算CBF约束违反的损失 (源自GCBF+的方法)。

    CBF约束条件为: h_dot(x) + alpha * h(x) >= 0

    参数:
        h_values: 每个时间步的CBF值 (T, B)
        h_dots: CBF的时间导数 (T, B)
        alpha: CBF的class-K函数参数

    返回:
        loss: CBF违反损失
        metrics: 详细的违反指标
    """
    # CBF约束: h_dot + alpha * h >= 0
    cbf_constraint = h_dots + alpha * h_values

    # 当约束为负时，即为违反
    violations = jnp.maximum(0.0, -cbf_constraint)
    violation_loss = jnp.mean(violations ** 2)

    # 安全区域分类 (GCBF+的方法)
    safe_region = h_values > 0.0      # CBF为正，定义为安全
    unsafe_region = h_values <= 0.0   # CBF为负或零，定义为不安全

    # 惩罚在不安全区域预测出正的CBF值
    false_safe_penalty = jnp.mean(
        jnp.where(unsafe_region, jnp.maximum(0.0, h_values) ** 2, 0.0)
    )

    # 惩罚在安全区域预测出负的CBF值
    false_unsafe_penalty = jnp.mean(
        jnp.where(safe_region, jnp.maximum(0.0, -h_values) ** 2, 0.0)
    )

    total_loss = violation_loss + false_safe_penalty + false_unsafe_penalty

    metrics = {
        "cbf_violations": jnp.sum(violations > 0.0),
        "violation_magnitude": jnp.mean(violations),
        "false_safe_rate": jnp.mean(unsafe_region & (h_values > 0)),
        "false_unsafe_rate": jnp.mean(safe_region & (h_values < 0)),
        "constraint_satisfaction": jnp.mean(cbf_constraint >= 0)
    }

    return total_loss, metrics


def compute_cbf_derivative_loss(
    h_values: chex.Array,
    h_grads: chex.Array,
    drone_states: chex.Array,
    control_inputs: chex.Array,
    physics_params: PhysicsParams
) -> Tuple[chex.Array, Dict]:
    """
    计算CBF导数条件的一致性损失。

    确保CBF的时间导数计算正确: h_dot = grad_h^T * (f(x) + g(x)u)

    参数:
        h_values: CBF的值 (T, B)
        h_grads: CBF关于位置的梯度 (T, B, 3)
        drone_states: 完整的无人机状态 (T, B, state_dim)
        control_inputs: 控制输入 (T, B, 3)
        physics_params: 物理参数

    返回:
        loss: 导数一致性损失
        metrics: 导数准确性指标
    """
    # 从无人机状态中提取位置和速度
    positions = drone_states[:, :, :3]   # (T, B, 3)
    velocities = drone_states[:, :, 3:6] # (T, B, 3)

    # 简化的动力学模型: f(x) = [v, -g], g(x) = [0, I]
    f_dynamics = jnp.concatenate([
        velocities,
        jnp.tile(jnp.array([0.0, 0.0, -9.81]), (h_values.shape[0], h_values.shape[1], 1))
    ], axis=-1)  # (T, B, 6)

    g_matrix = jnp.concatenate([
        jnp.zeros((h_values.shape[0], h_values.shape[1], 3, 3)),  # 位置部分
        jnp.tile(jnp.eye(3), (h_values.shape[0], h_values.shape[1], 1, 1))  # 速度部分
    ], axis=-2)  # (T, B, 6, 3)

    # 将h_grads扩展到整个状态空间 (假设对速度的梯度为零)
    h_grads_full = jnp.concatenate([
        h_grads,                # 位置梯度
        jnp.zeros_like(h_grads) # 速度梯度 (简化)
    ], axis=-1)  # (T, B, 6)

    # 预测的导数: grad_h^T * (f + gu)
    predicted_h_dot = jnp.sum(h_grads_full * f_dynamics, axis=-1) + jnp.sum(
        h_grads_full[:, :, None, :] @ g_matrix * control_inputs[:, :, None, :], axis=(-2, -1)
    )

    # 通过有限差分计算实际的导数
    dt = physics_params.dt
    actual_h_dot = (h_values[1:] - h_values[:-1]) / dt
    predicted_h_dot_aligned = predicted_h_dot[:-1]  # 对齐形状

    # 导数一致性损失
    derivative_error = predicted_h_dot_aligned - actual_h_dot
    derivative_loss = jnp.mean(derivative_error ** 2)

    metrics = {
        "derivative_mse": derivative_loss,
        "derivative_mae": jnp.mean(jnp.abs(derivative_error)),
        "prediction_accuracy": 1.0 - jnp.mean(jnp.abs(derivative_error) / (jnp.abs(actual_h_dot) + 1e-8))
    }

    return derivative_loss, metrics


# =============================================================================
# 物理驱动的损失 (集成DiffPhysDrone思想)
# =============================================================================

def compute_velocity_tracking_loss(
    actual_velocities: chex.Array,
    target_velocities: chex.Array,
    time_weights: Optional[chex.Array] = None
) -> Tuple[chex.Array, Dict]:
    """
    计算速度跟踪损失 (源自DiffPhysDrone的方法)。

    参数:
        actual_velocities: 模拟出的实际速度 (T, B, 3)
        target_velocities: 期望的目标速度 (T, B, 3)
        time_weights: 可选的时间权重 (T, B)

    返回:
        loss: 速度跟踪损失
        metrics: 跟踪性能指标
    """
    velocity_errors = actual_velocities - target_velocities

    # 使用平滑L1损失 (比MSE更鲁棒)
    smooth_l1_loss = jnp.mean(
        jnp.where(
            jnp.abs(velocity_errors) < 1.0,
            0.5 * velocity_errors ** 2,
            jnp.abs(velocity_errors) - 0.5
        )
    )

    # 如果提供了时间权重，则应用
    if time_weights is not None:
        smooth_l1_loss = jnp.mean(time_weights * smooth_l1_loss)

    # 分项的跟踪指标
    velocity_norms = jnp.linalg.norm(velocity_errors, axis=-1)

    metrics = {
        "velocity_mse": jnp.mean(velocity_errors ** 2),
        "velocity_mae": jnp.mean(jnp.abs(velocity_errors)),
        "tracking_accuracy": jnp.mean(velocity_norms < 0.5),  # 误差在0.5 m/s以内
        "max_error": jnp.max(velocity_norms)
    }

    return smooth_l1_loss, metrics


def compute_collision_avoidance_loss(
    distances_to_obstacles: chex.Array,
    safety_margins: chex.Array,
    velocity_magnitudes: chex.Array
) -> Tuple[chex.Array, Dict]:
    """
    计算一个与速度相关的避障损失。

    参数:
        distances_to_obstacles: 到最近障碍物的距离 (T, B)
        safety_margins: 所需的安全边际 (T, B)
        velocity_magnitudes: 当前的速度大小 (T, B)

    返回:
        loss: 避障损失
        metrics: 安全指标
    """
    # 类似屏障函数的惩罚 (越靠近障碍物惩罚越大)
    clearance = distances_to_obstacles - safety_margins

    # 速度加权的惩罚 (速度越快，惩罚越大)
    velocity_weights = 1.0 + velocity_magnitudes

    # 软屏障函数
    collision_penalty = jnp.where(
        clearance < 0.5,  # 在距离安全边界0.5m内激活
        velocity_weights * jnp.exp(-clearance * 4.0),  # 指数屏障
        0.0
    )

    collision_loss = jnp.mean(collision_penalty)

    # 额外的二次间隙损失 (来自DiffPhysDrone)
    quadratic_clearance = jnp.where(
        clearance < 1.0,  # 在1m内激活
        jnp.maximum(0.0, 1.0 - clearance) ** 2,
        0.0
    )

    total_loss = collision_loss + 0.5 * jnp.mean(quadratic_clearance)

    metrics = {
        "collision_risk": jnp.mean(clearance < 0.1),
        "safety_violations": jnp.sum(clearance < 0.0),
        "average_clearance": jnp.mean(clearance),
        "min_clearance": jnp.min(clearance)
    }

    return total_loss, metrics


def compute_control_regularization_loss(
    control_sequence: chex.Array,
    dt: float
) -> Tuple[chex.Array, Dict]:
    """
    计算控制能耗和平滑度的正则化损失。

    参数:
        control_sequence: 控制输入序列 (T, B, 3)
        dt: 时间步长

    返回:
        loss: 组合的控制正则化损失
        metrics: 控制能耗指标
    """
    # 控制大小惩罚
    control_magnitude_loss = jnp.mean(jnp.sum(control_sequence ** 2, axis=-1))

    # 控制平滑度 (加速度惩罚)
    control_diff = jnp.diff(control_sequence, axis=0) / dt
    control_smoothness_loss = jnp.mean(jnp.sum(control_diff ** 2, axis=-1))

    # 控制急动惩罚 (二阶导数)
    control_jerk = jnp.diff(control_diff, axis=0) / dt
    control_jerk_loss = jnp.mean(jnp.sum(control_jerk ** 2, axis=-1))

    total_loss = control_magnitude_loss + control_smoothness_loss + 0.1 * control_jerk_loss

    metrics = {
        "control_magnitude": jnp.mean(jnp.linalg.norm(control_sequence, axis=-1)),
        "control_smoothness": jnp.mean(jnp.linalg.norm(control_diff, axis=-1)),
        "control_jerk": jnp.mean(jnp.linalg.norm(control_jerk, axis=-1)),
        "max_control": jnp.max(jnp.linalg.norm(control_sequence, axis=-1))
    }

    return total_loss, metrics


# =============================================================================
# 效率与目标导向的损失
# =============================================================================

def compute_goal_reaching_loss(
    final_positions: chex.Array,
    target_positions: chex.Array,
    trajectory_positions: chex.Array
) -> Tuple[chex.Array, Dict]:
    """
    计算到达目标的效率损失。

    参数:
        final_positions: 最终位置 (B, 3)
        target_positions: 目标位置 (B, 3)
        trajectory_positions: 完整的轨迹位置 (T, B, 3)

    返回:
        loss: 到达目标损失
        metrics: 到达目标指标
    """
    # 主要目标: 到达目标位置
    final_distance_error = jnp.linalg.norm(final_positions - target_positions, axis=-1)
    goal_reaching_loss = jnp.mean(final_distance_error ** 2)

    # 次要目标: 高效的路径 (最小化轨迹长度)
    trajectory_lengths = jnp.sum(
        jnp.linalg.norm(jnp.diff(trajectory_positions, axis=0), axis=-1), axis=0
    )
    direct_distances = jnp.linalg.norm(
        trajectory_positions[-1] - trajectory_positions[0], axis=-1
    )
    path_efficiency = direct_distances / (trajectory_lengths + 1e-8)
    efficiency_loss = jnp.mean((1.0 - path_efficiency) ** 2)

    total_loss = goal_reaching_loss + 0.1 * efficiency_loss

    metrics = {
        "final_distance_error": jnp.mean(final_distance_error),
        "goal_success_rate": jnp.mean(final_distance_error < 0.5), # 距离目标0.5m内算成功
        "path_efficiency": jnp.mean(path_efficiency),
        "trajectory_length": jnp.mean(trajectory_lengths)
    }

    return total_loss, metrics


# =============================================================================
# 时间梯度衰减 (DiffPhysDrone)
# =============================================================================

def apply_spatial_temporal_gradient_decay(
    gradients: chex.Array,
    timestep: int,
    distance_to_obstacles: chex.Array,
    config: LossConfig
) -> chex.Array:
    """
    应用一个结合了时间和空间信息的梯度衰减机制。

    参数:
        gradients: 需要衰减的梯度 (任意形状)
        timestep: 当前在轨迹中的时间步
        distance_to_obstacles: 到最近障碍物的距离
        config: 包含衰减参数的损失配置

    返回:
        decayed_gradients: 经过衰减的梯度
    """
    # 基础的时间衰减 (指数衰减)
    temporal_decay = config.temporal_decay_alpha ** timestep

    if config.spatial_decay_enable:
        # 空间自适应：当离障碍物远时，应用更强的衰减
        min_distance = jnp.min(distance_to_obstacles)
        spatial_factor = jnp.minimum(
            1.0,
            jnp.maximum(0.3, min_distance / config.spatial_decay_range)
        )
        decay_factor = temporal_decay * spatial_factor
    else:
        decay_factor = temporal_decay

    return gradients * decay_factor


# =============================================================================
# 多目标损失函数的集成
# =============================================================================

def compute_comprehensive_loss(
    scan_outputs: ScanOutput,
    target_positions: chex.Array,
    target_velocities: chex.Array,
    config: LossConfig,
    physics_params: PhysicsParams
) -> Tuple[chex.Array, LossMetrics]:
    """
    计算一个综合的多目标损失函数。

    这个函数集成了所有来自GCBF+和DiffPhysDrone方法的损失组件。

    参数:
        scan_outputs: 来自扫描循环的完整轨迹输出。
        target_positions: 目标位置 (B, 3)。
        target_velocities: 目标速度 (T, B, 3)。
        config: 损失配置。
        physics_params: 物理参数。

    返回:
        total_loss: 组合的加权损失。
        metrics: 综合的损失指标。
    """
    T, B = scan_outputs.drone_states.shape[:2]

    # 从扫描输出中提取轨迹组件
    positions = scan_outputs.drone_states[:, :, :3]   # (T, B, 3)
    velocities = scan_outputs.drone_states[:, :, 3:6] # (T, B, 3)

    # 处理可选的CBF输出 (在简化版本中可能不存在)
    h_values = getattr(scan_outputs, 'cbf_values', jnp.zeros((T, B)))
    h_grads = getattr(scan_outputs, 'cbf_gradients', jnp.zeros((T, B, 3)))
    control_inputs = getattr(scan_outputs, 'safe_controls', getattr(scan_outputs, 'controls', jnp.zeros((T, B, 3))))
    distances = getattr(scan_outputs, 'obstacle_distances', jnp.ones((T, B)) * 10.0)  # 安全的默认值
    safety_info = getattr(scan_outputs, 'safety_info', None)

    # 初始化损失累加器
    total_loss = 0.0
    loss_components = {}

    # 1. GCBF+ CBF 相关的损失
    # ---------------------------------------------------------

    # CBF违反损失
    h_dots = jnp.gradient(h_values, axis=0) / physics_params.dt
    cbf_violation_loss, cbf_metrics = compute_cbf_violation_loss(
        h_values, h_dots, physics_params.cbf_alpha if hasattr(physics_params, 'cbf_alpha') else 1.0
    )
    total_loss += config.cbf_violation_coef * cbf_violation_loss
    loss_components.update(cbf_metrics)

    # CBF导数一致性损失
    cbf_derivative_loss, derivative_metrics = compute_cbf_derivative_loss(
        h_values, h_grads, scan_outputs.drone_states, control_inputs, physics_params
    )
    total_loss += config.cbf_derivative_coef * cbf_derivative_loss
    loss_components.update(derivative_metrics)

    # 2. DiffPhysDrone 物理驱动的损失
    # ---------------------------------------------------------

    # 速度跟踪损失
    velocity_loss, velocity_metrics = compute_velocity_tracking_loss(
        velocities, target_velocities
    )
    total_loss += config.velocity_tracking_coef * velocity_loss
    loss_components.update(velocity_metrics)

    # 避障损失
    velocity_magnitudes = jnp.linalg.norm(velocities, axis=-1)
    safety_margins = jnp.full_like(distances, 0.2)  # 20cm的安全边际
    collision_loss, collision_metrics = compute_collision_avoidance_loss(
        distances, safety_margins, velocity_magnitudes
    )
    total_loss += config.collision_avoidance_coef * collision_loss
    loss_components.update(collision_metrics)

    # 控制正则化
    control_loss, control_metrics = compute_control_regularization_loss(
        control_inputs, physics_params.dt
    )
    total_loss += config.control_smoothness_coef * control_loss
    loss_components.update(control_metrics)

    # 3. 效率与目标导向的损失
    # ---------------------------------------------------------

    # 到达目标损失
    goal_loss, goal_metrics = compute_goal_reaching_loss(
        positions[-1], target_positions, positions
    )
    total_loss += config.goal_reaching_coef * goal_loss
    loss_components.update(goal_metrics)

    # 4. 安全系统损失
    # ---------------------------------------------------------

    # 从QP求解器信息中提取与安全相关的损失
    safety_losses = []
    emergency_count = 0
    qp_success_count = 0

    for t in range(T):
        for b in range(B):
            # 这里需要根据实际的safety_info结构来正确实现
            # 目前使用占位符逻辑
            safety_status = 0  # 占位符
            if safety_status == 3:  # 紧急模式
                emergency_count += 1
                safety_losses.append(config.emergency_coef)
            elif safety_status > 0:  # QP求解失败
                safety_losses.append(config.safety_layer_coef)
            else:  # 成功
                qp_success_count += 1
                safety_losses.append(0.0)

    safety_loss = jnp.mean(jnp.array(safety_losses)) if safety_losses else 0.0
    total_loss += safety_loss

    # 5. 编译成综合的指标
    # ---------------------------------------------------------

    metrics = LossMetrics(
        total_loss=total_loss,
        efficiency_loss=goal_loss,
        safety_loss=cbf_violation_loss + collision_loss,
        control_loss=control_loss,

        # GCBF+ 指标
        cbf_violation=cbf_violation_loss,
        cbf_derivative=cbf_derivative_loss,
        cbf_boundary=0.0,  # 占位符

        # DiffPhysDrone 指标
        velocity_tracking=velocity_loss,
        collision_penalty=collision_loss,
        control_smoothness=control_loss,
        control_jerk=loss_components.get('control_jerk', 0.0),

        # 效率指标
        goal_distance=jnp.mean(jnp.linalg.norm(positions[-1] - target_positions, axis=-1)),
        time_penalty=0.0,  # 占位符

        # 安全指标
        safety_violations=loss_components.get('safety_violations', 0.0),
        emergency_activations=float(emergency_count) / (T * B),
        qp_success_rate=float(qp_success_count) / (T * B),

        # 训练动态
        gradient_norm=0.0,  # 将在训练循环中填充
        temporal_decay_factor=config.temporal_decay_alpha
    )

    return total_loss, metrics


# =============================================================================
# MGDA (多梯度下降算法)
# =============================================================================

def mgda_gradient_balancing(
    gradients_dict: Dict[str, chex.Array],
    loss_weights: Dict[str, float]
) -> Tuple[chex.Array, Dict[str, float]]:
    """
    使用多梯度下降算法来平衡多目标优化。

    参数:
        gradients_dict: 包含每个目标梯度的字典。
        loss_weights: 当前的损失权重。

    返回:
        balanced_gradients: 平衡后的组合梯度。
        updated_weights: 更新后的损失权重。
    """
    # 将梯度展平以便进行MGDA计算
    flat_gradients = {}
    original_shapes = {}

    for name, grad in gradients_dict.items():
        original_shapes[name] = grad.shape
        flat_gradients[name] = grad.flatten()

    # 将梯度堆叠成矩阵 (目标数量, 参数数量)
    gradient_matrix = jnp.stack([flat_gradients[name] for name in gradients_dict.keys()])

    # 计算格拉姆矩阵 G_ij = <g_i, g_j>
    gram_matrix = gradient_matrix @ gradient_matrix.T

    # 求解最优权重 (简化的Frank-Wolfe算法)
    n_objectives = len(gradients_dict)
    current_weights = jnp.array([loss_weights[name] for name in gradients_dict.keys()])

    # 将权重投影到单纯形上 (确保和为1)
    current_weights = current_weights / jnp.sum(current_weights)

    # 基于梯度冲突更新权重 (简化版)
    gradient_conflicts = jnp.diag(gram_matrix) - jnp.sum(
        gram_matrix * current_weights[None, :], axis=1
    )

    # 调整权重以减少冲突
    weight_adjustment = 0.01 * gradient_conflicts
    new_weights = current_weights - weight_adjustment
    new_weights = jnp.maximum(0.1, new_weights)  # 保证最小权重
    new_weights = new_weights / jnp.sum(new_weights)  # 归一化

    # 计算平衡后的梯度
    balanced_flat_gradients = jnp.sum(new_weights[:, None] * gradient_matrix, axis=0)

    # 恢复回原始形状 (为简单起见，假设所有梯度形状相同)
    first_shape = next(iter(original_shapes.values()))
    balanced_gradients = balanced_flat_gradients.reshape(first_shape)

    # 更新权重字典
    updated_weights = dict(zip(gradients_dict.keys(), new_weights))

    return balanced_gradients, updated_weights


# =============================================================================
# 使用的简单加权损失
# =============================================================================

def compute_simple_weighted_loss(
    scan_outputs: ScanOutput,
    target_positions: chex.Array,
    target_velocities: chex.Array,
    physics_params: PhysicsParams,
    alpha_efficiency: float = 1.0,
    beta_safety: float = 2.0
) -> Tuple[chex.Array, Dict[str, chex.Array]]:
    """
    计算一个简单的加权损失函数: L_total = α * L_efficiency + β * L_safety

    这是的核心损失函数，用简单的加权和替代了复杂的MGDA。

    参数:
        scan_outputs: BPTT扫描的输出。
        target_positions: 目标位置 (B, 3)。
        target_velocities: 目标速度 (T, B, 3)。
        physics_params: 物理参数。
        alpha_efficiency: 效率损失的权重。
        beta_safety: 安全损失的权重。

    返回:
        total_loss: 总损失。
        loss_breakdown: 各项损失的分解。
    """
    T, B = scan_outputs.drone_states.shape[:2]

    # 提取轨迹组件
    positions = scan_outputs.drone_states[:, :, :3]   # (T, B, 3)
    velocities = scan_outputs.drone_states[:, :, 3:6] # (T, B, 3)

    # CBF值和控制输入
    h_values = getattr(scan_outputs, 'cbf_values', jnp.zeros((T, B)))
    control_inputs = getattr(scan_outputs, 'safe_controls',
                           getattr(scan_outputs, 'controls', jnp.zeros((T, B, 3))))

    # 效率损失 L_efficiency 

    # 1. 目标到达损失 (最重要)
    final_positions = positions[-1]  # (B, 3)
    goal_distance_error = jnp.linalg.norm(final_positions - target_positions, axis=-1)#这个函数直接计算了欧几里得距离
    goal_reaching_loss = jnp.mean(goal_distance_error ** 2)#批处理训练,计算了批次中所有样本的平均损失

    # 2. 速度跟踪损失
    velocity_error = velocities - target_velocities
    velocity_tracking_loss = jnp.mean(jnp.sum(velocity_error ** 2, axis=-1))

    # 3. 路径效率损失
    trajectory_length = jnp.sum(
        jnp.linalg.norm(jnp.diff(positions, axis=0), axis=-1), axis=0
    )  # (B,)
    direct_distance = jnp.linalg.norm(positions[-1] - positions[0], axis=-1)  # (B,)
    path_efficiency = direct_distance / (trajectory_length + 1e-8)
    path_efficiency_loss = jnp.mean((1.0 - path_efficiency) ** 2)

    # 总效率损失
    L_efficiency = (
        2.0 * goal_reaching_loss +        # 最重要: 到达目标
        1.0 * velocity_tracking_loss +    # 重要: 速度跟踪
        0.2 * path_efficiency_loss        # 次要: 路径效率
    )

    # 安全损失 L_safety 

    # 1. CBF约束违反损失 (核心安全)
    h_dots = jnp.gradient(h_values, axis=0) / physics_params.dt
    cbf_alpha = getattr(physics_params, 'cbf_alpha', 1.0)
    cbf_constraint = h_dots + cbf_alpha * h_values
    cbf_violation = jnp.mean(jnp.maximum(0.0, -cbf_constraint) ** 2)

    # 2. 基本避障损失 (简化版)
    min_altitude = 0.3  # 最小安全高度
    altitude_violation = jnp.mean(jnp.maximum(0.0, min_altitude - positions[:, :, 2]) ** 2)

    # 3. 控制约束违反
    max_control_magnitude = 1.0  # 最大控制大小
    control_violation = jnp.mean(
        jnp.maximum(0.0, jnp.linalg.norm(control_inputs, axis=-1) - max_control_magnitude) ** 2
    )

    # 总安全损失
    L_safety = (
        3.0 * cbf_violation +          # 最重要: CBF约束
        2.0 * altitude_violation +     # 重要: 基本避障
        1.0 * control_violation        # 次要: 控制约束
    )

    #总损失 
    L_total = alpha_efficiency * L_efficiency + beta_safety * L_safety

    # 损失分解 (用于监控)
    loss_breakdown = {
        'total_loss': L_total,
        'efficiency_loss': L_efficiency,
        'safety_loss': L_safety,
        'goal_reaching_loss': goal_reaching_loss,
        'velocity_tracking_loss': velocity_tracking_loss,
        'path_efficiency_loss': path_efficiency_loss,
        'cbf_violation_loss': cbf_violation,
        'altitude_violation_loss': altitude_violation,
        'control_violation_loss': control_violation,
        'final_goal_distance': jnp.mean(goal_distance_error),
        'average_cbf_value': jnp.mean(h_values),
        'control_magnitude': jnp.mean(jnp.linalg.norm(control_inputs, axis=-1))
    }

    return L_total, loss_breakdown


def simple_training_step(
    params_dict: Dict,
    optimizer_state: optax.OptState,
    batch_data: Dict,
    physics_params: PhysicsParams,
    optimizer: optax.GradientTransformation,
    alpha_efficiency: float = 1.0,
    beta_safety: float = 2.0
) -> Tuple[Dict, optax.OptState, Dict[str, chex.Array]]:
    """
    MVP阶段4的简化训练步骤，使用简单的加权损失函数。

    参数:
        params_dict: 包含所有模型参数的字典。
        optimizer_state: 优化器的状态。
        batch_data: 一个批次的训练数据。
        physics_params: 物理引擎的参数。
        optimizer: 优化器。
        alpha_efficiency: 效率损失的权重。
        beta_safety: 安全损失的权重。

    返回:
        updated_params: 更新后的参数。
        updated_opt_state: 更新后的优化器状态。
        loss_breakdown: 损失分解。
    """
    def loss_fn(params):
        """定义一个计算损失的函数，用于后续的梯度计算。"""
        scan_outputs = batch_data['scan_outputs']
        target_positions = batch_data['target_positions']
        target_velocities = batch_data['target_velocities']

        loss, loss_breakdown = compute_simple_weighted_loss(
            scan_outputs, target_positions, target_velocities,
            physics_params, alpha_efficiency, beta_safety
        )
        return loss, loss_breakdown

    # 计算损失和梯度
    (loss, loss_breakdown), gradients = jax.value_and_grad(loss_fn, has_aux=True)(params_dict)

    # 应用梯度更新
    updates, new_optimizer_state = optimizer.update(gradients, optimizer_state, params_dict)
    updated_params = optax.apply_updates(params_dict, updates)

    # 将梯度信息添加到损失分解中，用于监控
    gradient_norm = jnp.sqrt(sum(
        jnp.sum(g ** 2) for g in jax.tree_util.tree_leaves(gradients)
    ))
    loss_breakdown['gradient_norm'] = gradient_norm

    return updated_params, new_optimizer_state, loss_breakdown


# =============================================================================
# 训练步骤函数
# =============================================================================

def training_step(
    params_dict: Dict,
    optimizer_state: optax.OptState,
    batch_data: Dict,
    config: LossConfig,
    physics_params: PhysicsParams,
    optimizer: optax.GradientTransformation
) -> Tuple[Dict, optax.OptState, LossMetrics]:
    """
    一个完整的训练步骤，包含完整的损失计算和梯度更新。

    参数:
        params_dict: 模型参数 (GNN, Policy, Safety)。
        optimizer_state: 优化器状态。
        batch_data: 训练批次数据。
        config: 损失配置。
        physics_params: 物理参数。

    返回:
        updated_params: 更新后的模型参数。
        updated_opt_state: 更新后的优化器状态。
        metrics: 训练指标。
    """
    def loss_fn(params):
        # 这里会与完整的扫描循环集成
        # 目前是占位符实现
        scan_outputs = batch_data['scan_outputs']  # 占位符
        target_positions = batch_data['target_positions']
        target_velocities = batch_data['target_velocities']

        loss, metrics = compute_comprehensive_loss(
            scan_outputs, target_positions, target_velocities, config, physics_params
        )
        return loss, metrics

    # 计算损失和梯度
    (loss, metrics), gradients = jax.value_and_grad(loss_fn, has_aux=True)(params_dict)

    # 使用优化器应用梯度更新
    updates, new_optimizer_state = optimizer.update(gradients, optimizer_state, params_dict)
    updated_params = optax.apply_updates(params_dict, updates)

    # 更新指标，加入梯度信息
    gradient_norm = jnp.sqrt(sum(
        jnp.sum(g ** 2) for g in jax.tree_util.tree_leaves(gradients)
    ))

    updated_metrics = metrics._replace(gradient_norm=gradient_norm)

    return updated_params, new_optimizer_state, updated_metrics


# =============================================================================
# 工厂函数和工具函数
# =============================================================================

def create_default_loss_config() -> LossConfig:
    """创建一个默认的损失配置。"""
    return LossConfig()


def create_optimizer(learning_rate: float = 1e-3) -> optax.GradientTransformation:
    """创建一个Adam优化器，并带有梯度裁剪功能。"""
    return optax.chain(
        optax.clip_by_global_norm(1.0),  # 梯度裁剪，防止梯度爆炸
        optax.adam(learning_rate)
    )


def log_training_metrics(metrics: LossMetrics, step: int):
    """记录详细的训练指标。"""
    print(f"训练步数 {step}:")
    print(f"  总损失: {metrics.total_loss:.6f}")
    print(f"  效率损失: {metrics.efficiency_loss:.6f}")
    print(f"  安全损失: {metrics.safety_loss:.6f}")
    print(f"  CBF违反: {metrics.cbf_violation:.6f}")
    print(f"  碰撞风险: {metrics.collision_penalty:.6f}")
    print(f"  目标距离: {metrics.goal_distance:.3f}m")
    print(f"  QP成功率: {metrics.qp_success_rate:.3f}")
    print(f"  梯度范数: {metrics.gradient_norm:.6f}")