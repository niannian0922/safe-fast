"""
可视化工具模块

提供无人机轨迹和训练过程的可视化功能，用于验证训练效果和调试。

主要功能：
1. plot_trajectory: 绘制无人机飞行轨迹
2. plot_training_metrics: 绘制训练指标变化
3. create_animation: 创建轨迹动画 (可选)
4. 各种辅助可视化函数
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, List, Dict, Any
import jax.numpy as jnp


def setup_matplotlib():
    """
    设置matplotlib的中文显示和样式
    """
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.style.use('default')


def plot_3d_trajectory(positions: np.ndarray,
                      initial_position: np.ndarray,
                      target_position: np.ndarray,
                      velocities: Optional[np.ndarray] = None,
                      title: str = "无人机3D轨迹",
                      save_path: Optional[str] = None,
                      show: bool = False,
                      figsize: tuple = (12, 8)) -> None:
    """
    绘制3D无人机飞行轨迹
    
    Args:
        positions: [T, 3] 位置序列
        initial_position: [3] 起始位置
        target_position: [3] 目标位置
        velocities: [T, 3] 可选的速度序列
        title: 图标题
        save_path: 保存路径
        show: 是否显示
        figsize: 图片尺寸
    """
    setup_matplotlib()
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制主轨迹
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
            'b-', linewidth=2.5, alpha=0.8, label='飞行轨迹')
    
    # 标记关键点
    ax.scatter(*initial_position, color='green', s=150, marker='o', 
              label='起始点', edgecolor='black', linewidth=1)
    ax.scatter(*target_position, color='red', s=200, marker='*', 
              label='目标点', edgecolor='black', linewidth=1)
    
    # 添加轨迹方向箭头
    if len(positions) > 5:
        n_arrows = min(8, len(positions) // 10)
        arrow_indices = np.linspace(5, len(positions)-5, n_arrows, dtype=int)
        
        for idx in arrow_indices:
            if idx < len(positions) - 1:
                # 计算方向向量
                direction = positions[idx+1] - positions[idx]
                direction = direction / (np.linalg.norm(direction) + 1e-8)  # 归一化
                
                ax.quiver(positions[idx, 0], positions[idx, 1], positions[idx, 2],
                         direction[0], direction[1], direction[2], 
                         length=0.5, color='orange', alpha=0.7, arrow_length_ratio=0.3)
    
    # 设置坐标轴
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12) 
    ax.set_zlabel('Z (m)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 设置等比例坐标轴
    max_range = np.array([positions[:, 0].max()-positions[:, 0].min(),
                         positions[:, 1].max()-positions[:, 1].min(),
                         positions[:, 2].max()-positions[:, 2].min()]).max() / 2.0
    mid_x = (positions[:, 0].max()+positions[:, 0].min()) * 0.5
    mid_y = (positions[:, 1].max()+positions[:, 1].min()) * 0.5
    mid_z = (positions[:, 2].max()+positions[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 3D轨迹图保存至: {save_path}")
    
    if show:
        plt.show()
    
    plt.close()


def plot_comprehensive_trajectory_analysis(positions: np.ndarray,
                                         controls: np.ndarray,
                                         initial_position: np.ndarray,
                                         target_position: np.ndarray,
                                         dt: float = 0.02,
                                         title: str = "轨迹综合分析",
                                         save_path: Optional[str] = None,
                                         show: bool = False) -> None:
    """
    绘制轨迹的综合分析图，包含多个子图
    
    Args:
        positions: [T, 3] 位置序列
        controls: [T, control_dim] 控制序列
        initial_position: [3] 起始位置
        target_position: [3] 目标位置
        dt: 时间步长
        title: 总标题
        save_path: 保存路径
        show: 是否显示
    """
    setup_matplotlib()
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    T = len(positions)
    time_steps = np.arange(T) * dt
    
    # 1. 3D轨迹图
    ax = fig.add_subplot(2, 3, 1, projection='3d')
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2, alpha=0.8)
    ax.scatter(*initial_position, color='green', s=100, marker='o', label='起始')
    ax.scatter(*target_position, color='red', s=100, marker='*', label='目标')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D轨迹')
    ax.legend()
    ax.grid(True)
    
    # 2. XY平面投影
    axes[0, 1].plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2, alpha=0.8)
    axes[0, 1].scatter(initial_position[0], initial_position[1], color='green', s=100, marker='o')
    axes[0, 1].scatter(target_position[0], target_position[1], color='red', s=100, marker='*')
    axes[0, 1].set_xlabel('X (m)')
    axes[0, 1].set_ylabel('Y (m)')
    axes[0, 1].set_title('XY平面投影')
    axes[0, 1].grid(True)
    axes[0, 1].axis('equal')
    
    # 3. 到目标的距离
    distances = np.linalg.norm(positions - target_position, axis=1)
    axes[0, 2].plot(time_steps, distances, 'r-', linewidth=2)
    axes[0, 2].axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='成功阈值')
    axes[0, 2].set_xlabel('时间 (s)')
    axes[0, 2].set_ylabel('距离 (m)')
    axes[0, 2].set_title('到目标距离随时间变化')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    # 4. 速度分析
    if T > 1:
        velocities = np.diff(positions, axis=0) / dt
        velocity_magnitudes = np.linalg.norm(velocities, axis=1)
        axes[1, 0].plot(time_steps[1:], velocity_magnitudes, 'g-', linewidth=2, label='速度大小')
        axes[1, 0].set_xlabel('时间 (s)')
        axes[1, 0].set_ylabel('速度 (m/s)')
        axes[1, 0].set_title('飞行速度')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # 5. 控制输入分析
    control_magnitudes = np.linalg.norm(controls, axis=1)
    axes[1, 1].plot(time_steps, control_magnitudes, 'purple', linewidth=2, label='控制大小')
    axes[1, 1].set_xlabel('时间 (s)')
    axes[1, 1].set_ylabel('控制量')
    axes[1, 1].set_title('控制输入强度')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # 6. 控制平滑度分析
    if T > 1:
        control_diffs = np.diff(controls, axis=0)
        control_variations = np.linalg.norm(control_diffs, axis=1)
        axes[1, 2].plot(time_steps[1:], control_variations, 'orange', linewidth=2, label='控制变化')
        axes[1, 2].set_xlabel('时间 (s)')
        axes[1, 2].set_ylabel('控制变化量')
        axes[1, 2].set_title('控制平滑度')
        axes[1, 2].legend()
        axes[1, 2].grid(True)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 综合轨迹分析图保存至: {save_path}")
    
    if show:
        plt.show()
        
    plt.close()


def plot_training_progress(loss_history: List[float],
                          gradient_norm_history: List[float],
                          final_distance_history: List[float],
                          title: str = "训练进展",
                          save_path: Optional[str] = None,
                          show: bool = False,
                          log_scale: bool = True) -> None:
    """
    绘制训练进展图表
    
    Args:
        loss_history: 损失历史
        gradient_norm_history: 梯度范数历史
        final_distance_history: 最终距离历史
        title: 图标题
        save_path: 保存路径
        show: 是否显示
        log_scale: 是否使用对数坐标
    """
    setup_matplotlib()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    steps = range(len(loss_history))
    
    # 损失曲线
    axes[0, 0].plot(steps, loss_history, 'b-', alpha=0.8, linewidth=2)
    axes[0, 0].set_xlabel('训练步数')
    axes[0, 0].set_ylabel('总损失')
    axes[0, 0].set_title('损失变化曲线')
    axes[0, 0].grid(True, alpha=0.3)
    if log_scale:
        axes[0, 0].set_yscale('log')
    
    # 梯度范数
    axes[0, 1].plot(steps, gradient_norm_history, 'r-', alpha=0.8, linewidth=2)
    axes[0, 1].set_xlabel('训练步数')
    axes[0, 1].set_ylabel('梯度范数')
    axes[0, 1].set_title('梯度范数变化')
    axes[0, 1].grid(True, alpha=0.3)
    if log_scale:
        axes[0, 1].set_yscale('log')
    
    # 最终距离
    axes[1, 0].plot(steps, final_distance_history, 'g-', alpha=0.8, linewidth=2)
    axes[1, 0].axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='成功阈值')
    axes[1, 0].set_xlabel('训练步数')
    axes[1, 0].set_ylabel('最终距离 (m)')
    axes[1, 0].set_title('到目标最终距离')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 移动平均曲线 (如果数据足够多)
    if len(loss_history) > 50:
        window_size = min(100, len(loss_history) // 10)
        
        # 损失移动平均
        loss_ma = np.convolve(loss_history, np.ones(window_size)/window_size, mode='valid')
        distance_ma = np.convolve(final_distance_history, np.ones(window_size)/window_size, mode='valid')
        
        ma_steps = range(window_size-1, len(loss_history))
        
        ax_ma = axes[1, 1]
        line1 = ax_ma.plot(ma_steps, loss_ma, 'b-', linewidth=2, label=f'损失 MA({window_size})')
        ax_ma.set_xlabel('训练步数')
        ax_ma.set_ylabel('损失 (移动平均)', color='b')
        ax_ma.tick_params(axis='y', labelcolor='b')
        if log_scale:
            ax_ma.set_yscale('log')
        
        # 创建第二个y轴
        ax_ma2 = ax_ma.twinx()
        line2 = ax_ma2.plot(ma_steps, distance_ma, 'g-', linewidth=2, label=f'距离 MA({window_size})')
        ax_ma2.set_ylabel('距离 (m)', color='g')
        ax_ma2.tick_params(axis='y', labelcolor='g')
        
        # 添加图例
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax_ma.legend(lines, labels, loc='upper right')
        
        ax_ma.set_title(f'移动平均趋势 (窗口={window_size})')
        ax_ma.grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, '数据不足\n无法显示移动平均', 
                       ha='center', va='center', transform=axes[1, 1].transAxes,
                       fontsize=12, alpha=0.5)
        axes[1, 1].set_title('移动平均 (数据不足)')
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 训练进展图保存至: {save_path}")
    
    if show:
        plt.show()
    
    plt.close()


def plot_performance_comparison(metrics_dict: Dict[str, List[float]],
                               title: str = "性能对比",
                               save_path: Optional[str] = None,
                               show: bool = False) -> None:
    """
    绘制多个性能指标的对比图
    
    Args:
        metrics_dict: 指标字典，格式为 {metric_name: [values]}
        title: 图标题
        save_path: 保存路径
        show: 是否显示
    """
    setup_matplotlib()
    
    n_metrics = len(metrics_dict)
    if n_metrics == 0:
        print("⚠️ 没有指标数据可以绘制")
        return
    
    # 计算子图布局
    rows = int(np.ceil(np.sqrt(n_metrics)))
    cols = int(np.ceil(n_metrics / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
    if n_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_metrics))
    
    for i, (metric_name, values) in enumerate(metrics_dict.items()):
        ax = axes[i]
        steps = range(len(values))
        
        ax.plot(steps, values, color=colors[i], linewidth=2, alpha=0.8)
        ax.set_xlabel('步数')
        ax.set_ylabel(metric_name)
        ax.set_title(metric_name)
        ax.grid(True, alpha=0.3)
        
        # 添加统计信息
        mean_val = np.mean(values)
        std_val = np.std(values)
        ax.axhline(y=mean_val, color=colors[i], linestyle='--', alpha=0.5, 
                  label=f'均值: {mean_val:.3f}±{std_val:.3f}')
        ax.legend(fontsize=8)
    
    # 隐藏多余的子图
    for i in range(n_metrics, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 性能对比图保存至: {save_path}")
    
    if show:
        plt.show()
    
    plt.close()


def create_training_summary_report(training_data: Dict[str, Any],
                                  save_dir: str = "reports") -> str:
    """
    创建训练总结报告，包含多个图表
    
    Args:
        training_data: 训练数据字典
        save_dir: 保存目录
    
    Returns:
        报告路径
    """
    from pathlib import Path
    import time
    
    Path(save_dir).mkdir(exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # 1. 训练进展图
    if 'loss_history' in training_data:
        progress_path = f"{save_dir}/training_progress_{timestamp}.png"
        plot_training_progress(
            loss_history=training_data['loss_history'],
            gradient_norm_history=training_data.get('gradient_norm_history', []),
            final_distance_history=training_data.get('final_distance_history', []),
            title="训练进展总结",
            save_path=progress_path
        )
    
    # 2. 最终轨迹示例
    if 'sample_trajectory' in training_data:
        traj_data = training_data['sample_trajectory']
        trajectory_path = f"{save_dir}/final_trajectory_{timestamp}.png"
        plot_comprehensive_trajectory_analysis(
            positions=traj_data['positions'],
            controls=traj_data['controls'],
            initial_position=traj_data['initial_position'],
            target_position=traj_data['target_position'],
            title="最终训练策略轨迹示例",
            save_path=trajectory_path
        )
    
    # 3. 性能指标对比
    if 'performance_metrics' in training_data:
        metrics_path = f"{save_dir}/performance_metrics_{timestamp}.png"
        plot_performance_comparison(
            metrics_dict=training_data['performance_metrics'],
            title="性能指标总览",
            save_path=metrics_path
        )
    
    print(f"✅ 训练总结报告已生成，保存在目录: {save_dir}")
    return save_dir


# 快捷函数，简化常用操作
def quick_trajectory_plot(positions: np.ndarray, 
                         initial_pos: np.ndarray, 
                         target_pos: np.ndarray,
                         save_name: str = "trajectory.png") -> None:
    """快捷绘制轨迹的简化接口"""
    plot_3d_trajectory(positions, initial_pos, target_pos, 
                      title="无人机轨迹", save_path=save_name)


def quick_training_plot(loss_history: List[float], save_name: str = "training.png") -> None:
    """快捷绘制训练过程的简化接口"""
    plot_training_progress(loss_history, [], [], 
                          title="训练损失", save_path=save_name)


# 为了兼容性，添加create_trajectory_plot和create_training_curves函数
def create_trajectory_plot(positions: np.ndarray,
                          initial_position: np.ndarray,
                          target_position: np.ndarray,
                          obstacles: Optional[np.ndarray] = None,
                          title: str = "无人机轨迹",
                          save_path: Optional[str] = None,
                          show: bool = False) -> None:
    """
    创建轨迹图（兼容接口）
    """
    plot_3d_trajectory(positions, initial_position, target_position,
                      title=title, save_path=save_path, show=show)
    
    # 如果有障碍物，添加到图中
    if obstacles is not None and save_path:
        # 重新绘制包含障碍物的版本
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制轨迹
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                'b-', linewidth=2.5, alpha=0.8, label='飞行轨迹')
        
        # 标记关键点
        ax.scatter(*initial_position, color='green', s=150, marker='o', 
                  label='起始点', edgecolor='black', linewidth=1)
        ax.scatter(*target_position, color='red', s=200, marker='*', 
                  label='目标点', edgecolor='black', linewidth=1)
        
        # 绘制障碍物
        if len(obstacles) > 0:
            ax.scatter(obstacles[:, 0], obstacles[:, 1], obstacles[:, 2],
                      color='orange', s=100, marker='s', alpha=0.7,
                      label='障碍物', edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_zlabel('Z (m)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


def create_training_curves(training_history: Dict[str, List],
                          title: str = "训练曲线",
                          save_path: Optional[str] = None,
                          show: bool = False) -> None:
    """
    创建训练曲线图（兼容接口）
    """
    setup_matplotlib()
    
    # 确定需要绘制的指标
    metrics = {}
    if 'total_losses' in training_history:
        metrics['总损失'] = training_history['total_losses']
    if 'efficiency_losses' in training_history:
        metrics['效率损失'] = training_history['efficiency_losses']
    if 'safety_losses' in training_history:
        metrics['安全损失'] = training_history['safety_losses']
    if 'final_distances' in training_history:
        metrics['最终距离'] = training_history['final_distances']
    if 'qp_success_rates' in training_history:
        metrics['QP成功率'] = training_history['qp_success_rates']
    
    # 如果有MGDA权重数据
    if 'mgda_weights_efficiency' in training_history:
        metrics['效率权重(MGDA)'] = training_history['mgda_weights_efficiency']
    if 'mgda_weights_safety' in training_history:
        metrics['安全权重(MGDA)'] = training_history['mgda_weights_safety']
    
    # 调用现有的性能对比函数
    plot_performance_comparison(metrics, title=title, save_path=save_path, show=show)


if __name__ == "__main__":
    """
    可视化模块测试
    """
    print("🎨 测试可视化模块...")
    
    # 生成测试数据
    T = 100
    t = np.linspace(0, 10, T)
    
    # 螺旋轨迹测试
    positions = np.column_stack([
        2 * np.cos(t),
        2 * np.sin(t), 
        0.1 * t
    ])
    
    controls = np.random.randn(T, 3) * 0.5
    initial_pos = positions[0]
    target_pos = np.array([0, 0, 10])
    
    # 测试轨迹绘制
    plot_3d_trajectory(positions, initial_pos, target_pos, 
                      title="测试轨迹", save_path="test_trajectory.png")
    
    # 测试综合分析
    plot_comprehensive_trajectory_analysis(positions, controls, initial_pos, target_pos,
                                         title="测试综合分析", save_path="test_analysis.png")
    
    # 测试训练进展
    loss_hist = [100 * np.exp(-0.01*i) + np.random.randn()*5 for i in range(200)]
    grad_hist = [10 * np.exp(-0.005*i) + np.random.randn()*0.5 for i in range(200)]
    dist_hist = [5 * np.exp(-0.008*i) + np.random.randn()*0.2 for i in range(200)]
    
    plot_training_progress(loss_hist, grad_hist, dist_hist,
                          title="测试训练进展", save_path="test_training.png")
    
    print("✅ 可视化模块测试完成！")