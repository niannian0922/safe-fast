"""
效率优先训练模块

这个模块定义了纯粹追求效率的损失函数计算逻辑，完全不考虑安全约束。
核心职责：通过数学函数定义什么样的轨迹是"好"的，指导优化器训练策略网络。

主要功能：
1. EfficiencyLossConfig: 损失项权重配置类
2. compute_efficiency_loss: 核心效率损失计算函数
"""

import jax
import jax.nn as jnn
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Dict, Any, Tuple


@dataclass
class EfficiencyLossConfig:
    """
    效率损失配置类
    
    将所有损失项的权重参数化，实现超参数与算法逻辑的解耦。
    这样调参时只需修改配置值，无需触碰核心算法代码。
    
    新增物理感知特性：
    - Z轴权重：针对无人机各向异性动力学的特殊处理
    - 悬停损失：鼓励到达目标时减速悬停
    """
    # 目标到达损失权重 - 最重要的损失项
    goal_weight: float = 10.0
    goal_hard_weight: float = 0.0
    goal_hard_threshold: float = 0.0
    
    # Z轴特殊权重 - 针对无人机在垂直方向的控制难度
    # 该权重会乘以Z轴误差，让网络明白维持高度比水平移动更重要
    z_axis_weight_multiplier: float = 10.0
    
    # 控制能耗损失权重 - 正则化项，鼓励节能控制
    control_weight: float = 0.1
    
    # 控制平滑度损失权重 - 防止控制指令剧烈变化
    smoothness_weight: float = 0.5
    
    # 终点额外惩罚权重 - 确保最终精确到达目标
    final_goal_weight: float = 50.0
    
    # 悬停损失权重 - 惩罚轨迹终点的速度，鼓励减速悬停
    hover_weight: float = 5.0
    
    # 时间衰减因子 - 用于时间加权的指数衰减
    time_decay_factor: float = 0.95


def compute_goal_loss(positions: jnp.ndarray, 
                     target_position: jnp.ndarray, 
                     config: EfficiencyLossConfig) -> Tuple[float, Dict[str, float]]:
    """
    计算目标到达损失（各向异性版本）
    
    新的物理感知方法：
    1. 分解误差：将XY平面和Z轴误差分开计算
    2. Z轴加权：针对无人机垂直控制的困难性给予更高权重
    3. 时间加权：早期偏离目标的惩罚更大
    4. 终点加重：最后一个点的额外惩罚
    
    这种方法解决了各向同性损失函数 vs 各向异性无人机动力学的矛盾。
    
    参数
    ----------
    positions:
        `[T, 3]` 的轨迹位置序列。
    target_position:
        `[3]` 目标位置向量。
    config:
        损失配置对象。

    返回
    ----------
    goal_loss:
        标量损失值。
    metrics:
        详细指标字典。
    """
    # 计算位置误差向量
    position_errors = positions - target_position  # [T, 3]
    
    # 分解为XY平面误差和Z轴误差
    xy_errors = position_errors[:, :2]  # [T, 2] - XY平面误差
    z_errors = position_errors[:, 2]    # [T] - Z轴误差
    
    # 计算各向异性距离：XY用欧氏距离，Z轴单独计算并加权
    xy_distances = jnp.linalg.norm(xy_errors, axis=1)  # [T] - XY平面距离
    z_distances = jnp.abs(z_errors)  # [T] - Z轴距离（绝对值）
    
    # 各向异性总距离：Z轴误差被特别加权
    anisotropic_distances = xy_distances + config.z_axis_weight_multiplier * z_distances  # [T]
    
    # 时间加权：早期步骤的权重更大
    T = positions.shape[0]
    time_weights = config.time_decay_factor ** jnp.arange(T)  # [T]
    
    # 全程加权距离损失
    weighted_distance_loss = jnp.sum(time_weights * anisotropic_distances)
    
    # 终点额外惩罚（也使用各向异性距离）
    final_xy_distance = xy_distances[-1]
    final_z_distance = z_distances[-1]
    final_anisotropic_distance = final_xy_distance + config.z_axis_weight_multiplier * final_z_distance
    final_penalty = config.final_goal_weight * final_anisotropic_distance
    
    # 总目标损失
    hard_penalty = config.goal_hard_weight * jnn.relu(final_anisotropic_distance - config.goal_hard_threshold)
    total_goal_loss = config.goal_weight * (weighted_distance_loss + final_penalty) + hard_penalty
    
    # 收集详细指标，包括XY和Z的分别统计
    metrics = {
        'mean_xy_distance': jnp.mean(xy_distances),
        'mean_z_distance': jnp.mean(z_distances),
        'mean_anisotropic_distance': jnp.mean(anisotropic_distances),
        'final_xy_distance': final_xy_distance,
        'final_z_distance': final_z_distance,
        'final_anisotropic_distance': final_anisotropic_distance,
        'weighted_distance_loss': weighted_distance_loss,
        'final_penalty': final_penalty,
        # 为了向后兼容，保留原有的总距离指标
        'mean_distance_to_goal': jnp.mean(anisotropic_distances),
        'final_distance_to_goal': final_anisotropic_distance
    }
    
    return total_goal_loss, metrics


def compute_control_loss(controls: jnp.ndarray, 
                        config: EfficiencyLossConfig) -> Tuple[float, Dict[str, float]]:
    """
    计算控制能耗损失
    
    通过L2正则化惩罚过大的控制指令，鼓励节能飞行。
    
    参数
    ----------
    controls:
        `[T, control_dim]` 控制指令序列。
    config:
        损失配置对象。

    返回
    ----------
    control_loss:
        标量损失值。
    metrics:
        详细指标字典。
    """
    # L2范数的平方和
    control_magnitudes = jnp.linalg.norm(controls, axis=1)  # [T]
    control_energy = jnp.sum(control_magnitudes ** 2)
    
    # 应用权重
    total_control_loss = config.control_weight * control_energy
    
    # 收集指标
    metrics = {
        'mean_control_magnitude': jnp.mean(control_magnitudes),
        'max_control_magnitude': jnp.max(control_magnitudes),
        'total_control_energy': control_energy
    }
    
    return total_control_loss, metrics


def compute_smoothness_loss(controls: jnp.ndarray, 
                           config: EfficiencyLossConfig) -> Tuple[float, Dict[str, float]]:
    """
    计算控制平滑度损失
    
    惩罚相邻时间步控制指令的剧烈变化，近似于"急动度"(Jerk)。
    这确保飞行轨迹平滑稳定，避免高频抖动。
    
    参数
    ----------
    controls:
        `[T, control_dim]` 控制指令序列。
    config:
        损失配置对象。

    返回
    ----------
    smoothness_loss:
        标量损失值。
    metrics:
        详细指标字典。
    """
    if controls.shape[0] <= 1:
        # 只有一个时间步，没有平滑度概念
        return 0.0, {'control_variation': 0.0, 'max_control_change': 0.0}
    
    # 计算相邻时间步的控制变化
    control_diffs = controls[1:] - controls[:-1]  # [T-1, control_dim]
    control_variations = jnp.linalg.norm(control_diffs, axis=1)  # [T-1]
    
    # 平滑度损失：变化幅度的平方和
    smoothness_penalty = jnp.sum(control_variations ** 2)
    
    # 应用权重
    total_smoothness_loss = config.smoothness_weight * smoothness_penalty
    
    # 收集指标
    metrics = {
        'control_variation': jnp.mean(control_variations),
        'max_control_change': jnp.max(control_variations)
    }
    
    return total_smoothness_loss, metrics


def compute_hover_loss(trajectory_outputs: Dict[str, jnp.ndarray], 
                      config: EfficiencyLossConfig) -> Tuple[float, Dict[str, float]]:
    """
    计算悬停损失
    
    惩罚轨迹终点的速度，鼓励无人机在到达目标时减速至悬停状态，
    而不是直接冲过去。这解决了网络只学会"冲向目标"而不会"停下来"的问题。
    
    参数
    ----------
    trajectory_outputs:
        轨迹数据字典，需要包含 `'velocities'`。
    config:
        损失配置对象。

    返回
    ----------
    hover_loss:
        标量损失值。
    metrics:
        详细指标字典。
    """
    # 检查是否包含速度信息
    if 'velocities' not in trajectory_outputs:
        # 如果没有速度信息，返回零损失
        return 0.0, {
            'final_speed': 0.0,
            'final_xy_speed': 0.0,
            'final_z_speed': 0.0,
        }
    
    velocities = trajectory_outputs['velocities']  # [T, 3]
    
    # 获取终点速度
    final_velocity = velocities[-1]  # [3]
    
    # 分解为XY平面和Z轴速度
    final_xy_velocity = final_velocity[:2]  # [2]
    final_z_velocity = final_velocity[2]    # 标量
    
    # 计算速度大小
    final_xy_speed = jnp.linalg.norm(final_xy_velocity)
    final_z_speed = jnp.abs(final_z_velocity)
    final_total_speed = jnp.linalg.norm(final_velocity)
    
    # 悬停损失：惩罚终点速度，鼓励悬停
    # 也可以考虑Z轴速度的特殊权重，但这里使用总速度
    hover_penalty = final_total_speed ** 2
    
    # 应用权重
    total_hover_loss = config.hover_weight * hover_penalty
    
    # 收集指标
    metrics = {
        'final_speed': final_total_speed,
        'final_xy_speed': final_xy_speed,
        'final_z_speed': final_z_speed,
        'hover_penalty': hover_penalty
    }
    
    return total_hover_loss, metrics


def compute_efficiency_loss(trajectory_outputs: Dict[str, jnp.ndarray],
                          target_position: jnp.ndarray,
                          config: EfficiencyLossConfig) -> Tuple[float, Dict[str, float]]:
    """
    计算完整的效率损失函数（物理感知版本）
    
    这是模块的核心函数，整合所有损失项并返回总损失和详细指标。
    新增了各向异性目标损失和悬停损失，专门针对无人机的物理特性设计。
    
    参数
    ----------
    trajectory_outputs:
        轨迹数据字典，需包含：
            - `'positions'`: `[T, 3]` 位置序列；
            - `'controls'`: `[T, control_dim]` 控制指令序列；
            - `'velocities'`: `[T, 3]` 速度序列（可选，用于悬停损失）。
    target_position:
        `[3]` 目标位置。
    config:
        损失配置对象。

    返回
    ----------
    total_loss:
        总损失（标量），用于反向传播。
    metrics_dict:
        详细指标字典，用于监控训练过程。
    """
    # 提取轨迹数据
    positions = trajectory_outputs['positions']
    controls = trajectory_outputs['controls']
    
    # 计算各项损失
    goal_loss, goal_metrics = compute_goal_loss(positions, target_position, config)
    control_loss, control_metrics = compute_control_loss(controls, config)
    smoothness_loss, smoothness_metrics = compute_smoothness_loss(controls, config)
    hover_loss, hover_metrics = compute_hover_loss(trajectory_outputs, config)
    
    # 总损失（新增悬停损失项）
    total_loss = goal_loss + control_loss + smoothness_loss + hover_loss
    
    # 整合所有指标
    metrics_dict = {
        'total_loss': total_loss,
        'goal_loss': goal_loss,
        'control_loss': control_loss,
        'smoothness_loss': smoothness_loss,
        'hover_loss': hover_loss,
        **{f'goal_{k}': v for k, v in goal_metrics.items()},
        **{f'control_{k}': v for k, v in control_metrics.items()},
        **{f'smoothness_{k}': v for k, v in smoothness_metrics.items()},
        **{f'hover_{k}': v for k, v in hover_metrics.items()}
    }
    
    return total_loss, metrics_dict


def create_efficiency_loss_fn(target_position: jnp.ndarray, 
                            config: EfficiencyLossConfig):
    """
    创建效率损失函数的工厂函数
    
    返回一个部分应用的损失函数，只需要trajectory_outputs作为输入。
    这种设计方便在训练循环中使用。
    
    参数
    ----------
    target_position:
        `[3]` 目标位置。
    config:
        损失配置对象。

    返回
    ----------
    loss_fn:
        接受 `trajectory_outputs` 并返回 `(loss, metrics)` 的损失函数。
    """
    def loss_fn(trajectory_outputs: Dict[str, jnp.ndarray]) -> Tuple[float, Dict[str, float]]:
        return compute_efficiency_loss(trajectory_outputs, target_position, config)
    
    return loss_fn


# 用于验证的默认配置（物理感知版本）
DEFAULT_EFFICIENCY_CONFIG = EfficiencyLossConfig(
    goal_weight=3.0,
    z_axis_weight_multiplier=2.0,  # 垂直方向仍保留更高权重，但不过度放大
    control_weight=0.08,
    smoothness_weight=0.45,
    final_goal_weight=8.0,
    hover_weight=8.0,  # 鼓励悬停行为
    time_decay_factor=0.95
)


def validate_trajectory_outputs(trajectory_outputs: Dict[str, jnp.ndarray]) -> None:
    """
    验证轨迹输出数据的格式和完整性
    
    参数
    ----------
    trajectory_outputs:
        待验证的轨迹数据。

    异常
    ----------
    ValueError:
        当数据格式不正确时抛出。
    """
    required_keys = ['positions', 'controls']
    
    for key in required_keys:
        if key not in trajectory_outputs:
            raise ValueError(f"Missing required key '{key}' in trajectory_outputs")
    
    positions = trajectory_outputs['positions']
    controls = trajectory_outputs['controls']
    
    # 检查形状
    if len(positions.shape) != 2 or positions.shape[1] != 3:
        raise ValueError(f"positions should be [T, 3], got {positions.shape}")
    
    if len(controls.shape) != 2:
        raise ValueError(f"controls should be [T, control_dim], got {controls.shape}")
    
    # 检查时间步一致性
    if positions.shape[0] != controls.shape[0]:
        raise ValueError(f"Time steps mismatch: positions {positions.shape[0]} vs controls {controls.shape[0]}")
    
    print(f"✓ 轨迹数据验证通过: {positions.shape[0]} 时间步, 控制维度 {controls.shape[1]}")


if __name__ == "__main__":
    """
    简单的模块测试
    """
    import numpy as np
    
    print("🧪 测试 simple_training 模块...")
    
    # 创建测试数据
    T, control_dim = 50, 4
    positions = jnp.array(np.random.randn(T, 3))
    controls = jnp.array(np.random.randn(T, control_dim))
    target_position = jnp.array([5.0, 5.0, 5.0])
    
    trajectory_outputs = {
        'positions': positions,
        'controls': controls
    }
    
    # 验证数据格式
    validate_trajectory_outputs(trajectory_outputs)
    
    # 测试损失计算
    config = DEFAULT_EFFICIENCY_CONFIG
    total_loss, metrics = compute_efficiency_loss(trajectory_outputs, target_position, config)
    
    print(f"✓ 总损失: {total_loss:.4f}")
    print(f"✓ 目标损失: {metrics['goal_loss']:.4f}")
    print(f"✓ 控制损失: {metrics['control_loss']:.4f}")
    print(f"✓ 平滑度损失: {metrics['smoothness_loss']:.4f}")
    print(f"✓ 最终距离: {metrics['goal_final_distance_to_goal']:.4f}")
    
    # 测试工厂函数
    loss_fn = create_efficiency_loss_fn(target_position, config)
    loss_2, metrics_2 = loss_fn(trajectory_outputs)
    
    assert abs(loss_2 - total_loss) < 1e-6, "工厂函数结果不一致"
    print("✓ 工厂函数测试通过")
    
    # 测试梯度
    def test_grad_fn(positions):
        test_outputs = {'positions': positions, 'controls': controls}
        loss, _ = compute_efficiency_loss(test_outputs, target_position, config)
        return loss
    
    grad_fn = jax.grad(test_grad_fn)
    grads = grad_fn(positions)
    
    print(f"✓ 梯度计算成功，梯度范数: {jnp.linalg.norm(grads):.6f}")
    
    print("🎉 所有测试通过！simple_training 模块功能正常。")
