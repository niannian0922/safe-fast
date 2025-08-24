#!/usr/bin/env python3
"""
主训练脚本：安全敏捷飞行端到端学习系统
结合GCBF+和DiffPhysDrone的方法论
"""

import jax
import jax.numpy as jnp
import optax
import time
from typing import Dict, List, Tuple
import argparse
import os
import pickle
from pathlib import Path

from core.physics import create_initial_state, create_default_params
from core.perception import create_dummy_pointcloud
from core.safety import SafetyParams
from core.training import (
    CompleteTrainingConfig,
    initialize_complete_training,
    create_complete_training_step,
    test_complete_gradient_flow
)


def create_training_batch(batch_size: int,
                         trajectory_length: int,
                         rng_key: jax.random.PRNGKey,
                         config: CompleteTrainingConfig) -> Dict:
    """
    创建训练批次数据
    
    Args:
        batch_size: 批次大小
        trajectory_length: 轨迹长度
        rng_key: 随机数种子
        config: 训练配置
        
    Returns:
        batch: 包含批次数据的字典
    """
    
    keys = jax.random.split(rng_key, batch_size + 3)
    
    # 生成初始状态批次
    initial_positions = jax.random.uniform(
        keys[0], (batch_size, 3), minval=-2.0, maxval=2.0
    )
    initial_velocities = jax.random.uniform(
        keys[1], (batch_size, 3), minval=-1.0, maxval=1.0
    )
    
    initial_states = []
    for i in range(batch_size):
        state = create_initial_state(
            position=initial_positions[i],
            velocity=initial_velocities[i]
        )
        initial_states.append(state)
    
    # 生成目标位置和速度
    target_positions = jax.random.uniform(
        keys[2], (batch_size, 3), minval=5.0, maxval=12.0
    )
    target_velocities = jax.random.uniform(
        keys[3], (batch_size, 3), minval=0.5, maxval=3.0
    )
    
    # 生成点云序列
    point_clouds = []
    for i in range(batch_size):
        cloud_seq = []
        for t in range(trajectory_length):
            key_idx = (i * trajectory_length + t) % len(keys)
            cloud = create_dummy_pointcloud(
                keys[key_idx], 
                num_points=config.num_obstacles,
                bounds=config.obstacle_bounds
            )
            cloud_seq.append(cloud)
        point_clouds.append(jnp.stack(cloud_seq))
    
    return {
        'initial_states': initial_states,
        'point_cloud_sequences': jnp.stack(point_clouds),
        'target_positions': target_positions,
        'target_velocities': target_velocities
    }


def evaluate_model(policy_model, gnn_model,
                  policy_params, gnn_params,
                  physics_params, safety_params,
                  config: CompleteTrainingConfig,
                  rng_key: jax.random.PRNGKey,
                  num_eval_episodes: int = 10) -> Dict[str, float]:
    """
    评估模型性能
    
    Returns:
        metrics: 评估指标字典
    """
    
    from core.loop import complete_rollout_trajectory
    
    eval_metrics = []
    
    for episode in range(num_eval_episodes):
        episode_key = jax.random.fold_in(rng_key, episode)
        
        # 创建评估场景
        batch = create_training_batch(1, config.trajectory_length, episode_key, config)
        
        initial_state = batch['initial_states'][0]
        point_cloud_seq = batch['point_cloud_sequences'][0]
        target_pos = batch['target_positions'][0]
        
        # 执行轨迹rollout
        final_carry, trajectory_outputs = complete_rollout_trajectory(
            initial_state=initial_state,
            point_cloud_sequence=point_cloud_seq,
            policy_params=policy_params,
            policy_model=policy_model,
            gnn_params=gnn_params,
            gnn_model=gnn_model,
            physics_params=physics_params,
            safety_params=safety_params,
            trajectory_length=config.trajectory_length,
            dt=config.dt,
            use_rnn=False
        )
        
        # 计算评估指标
        final_distance = jnp.linalg.norm(final_carry.drone_state.position - target_pos)
        safety_violations = jnp.sum(trajectory_outputs.h < 0)
        mean_cbf = jnp.mean(trajectory_outputs.h)
        success = (final_distance < 2.0) and (safety_violations == 0)
        
        episode_metrics = {
            'final_distance': float(final_distance),
            'safety_violations': float(safety_violations),
            'mean_cbf_value': float(mean_cbf),
            'success': float(success),
            'trajectory_length': config.trajectory_length
        }
        
        eval_metrics.append(episode_metrics)
    
    # 聚合指标
    aggregated_metrics = {}
    for key in eval_metrics[0].keys():
        if key != 'trajectory_length':
            values = [m[key] for m in eval_metrics]
            aggregated_metrics[f'eval/{key}_mean'] = float(jnp.mean(jnp.array(values)))
            aggregated_metrics[f'eval/{key}_std'] = float(jnp.std(jnp.array(values)))
    
    aggregated_metrics['eval/success_rate'] = aggregated_metrics['eval/success_mean']
    
    return aggregated_metrics


def save_checkpoint(policy_params, gnn_params,
                   policy_opt_state, gnn_opt_state,
                   step: int, save_dir: str):
    """保存检查点"""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'policy_params': policy_params,
        'gnn_params': gnn_params,
        'policy_optimizer_state': policy_opt_state,
        'gnn_optimizer_state': gnn_opt_state,
        'step': step
    }
    
    checkpoint_path = save_path / f'checkpoint_{step}.pkl'
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint, f)
    
    print(f"检查点已保存至: {checkpoint_path}")


def main(args):
    """主训练循环"""
    
    print("🚁 安全敏捷飞行端到端学习系统")
    print("=" * 50)
    
    # 设置JAX
    if args.gpu:
        print("使用GPU加速")
    else:
        jax.config.update("jax_platform_name", "cpu")
        print("使用CPU训练")
    
    # 配置
    config = CompleteTrainingConfig(
        learning_rate=args.learning_rate,
        trajectory_length=args.trajectory_length,
        batch_size=args.batch_size,
        gradient_clip_norm=args.grad_clip,
    )
    
    physics_params = create_default_params()
    safety_params = SafetyParams()
    
    print(f"轨迹长度: {config.trajectory_length}")
    print(f"批次大小: {config.batch_size}")
    print(f"学习率: {config.learning_rate}")
    
    # 运行梯度流测试
    print("\n步骤1: 验证系统完整性...")
    if not test_complete_gradient_flow():
        print("❌ 系统验证失败，退出训练")
        return
    
    # 初始化训练组件
    print("\n步骤2: 初始化训练组件...")
    rng_key = jax.random.PRNGKey(args.seed)
    
    (policy_model, gnn_model,
     policy_params, gnn_params,
     policy_optimizer, gnn_optimizer,
     policy_opt_state, gnn_opt_state) = initialize_complete_training(config, rng_key)
    
    train_step = create_complete_training_step(config, physics_params, safety_params)
    
    print("✅ 初始化完成")
    
    # 训练循环
    print(f"\n步骤3: 开始训练 ({args.num_steps} 步)...")
    
    training_metrics = []
    best_success_rate = 0.0
    
    for step in range(args.num_steps):
        step_start_time = time.time()
        
        # 生成训练批次
        batch_key = jax.random.fold_in(rng_key, step)
        batch = create_training_batch(
            config.batch_size, config.trajectory_length, batch_key, config
        )
        
        # 执行训练步骤（暂时使用批次中的第一个样本）
        train_key = jax.random.fold_in(rng_key, step + 10000)
        
        (policy_params, gnn_params,
         policy_opt_state, gnn_opt_state,
         train_info) = train_step(
            policy_params, policy_model,
            gnn_params, gnn_model,
            policy_opt_state, gnn_opt_state,
            policy_optimizer, gnn_optimizer,
            batch['initial_states'][0],  # 使用第一个样本
            batch['point_cloud_sequences'][0],
            batch['target_positions'][0],
            batch['target_velocities'][0],
            train_key
        )
        
        step_time = time.time() - step_start_time
        
        # 记录训练指标
        train_info['step'] = step
        train_info['step_time'] = step_time
        training_metrics.append(train_info)
        
        # 打印进度
        if step % args.log_interval == 0:
            print(f"步骤 {step:4d}: "
                  f"损失={train_info['total_loss']:.4f}, "
                  f"CBF损失={train_info.get('cbf_unsafe_penalty', 0):.4f}, "
                  f"安全违规={train_info.get('safety_violations', 0)}, "
                  f"时间={step_time:.2f}s")
        
        # 评估模型
        if step % args.eval_interval == 0 and step > 0:
            print(f"\n评估模型 (步骤 {step})...")
            eval_key = jax.random.fold_in(rng_key, step + 20000)
            eval_metrics = evaluate_model(
                policy_model, gnn_model,
                policy_params, gnn_params,
                physics_params, safety_params,
                config, eval_key, num_eval_episodes=5
            )
            
            success_rate = eval_metrics['eval/success_rate']
            print(f"成功率: {success_rate:.2%}")
            print(f"平均最终距离: {eval_metrics['eval/final_distance_mean']:.3f}")
            print(f"平均安全违规: {eval_metrics['eval/safety_violations_mean']:.1f}")
            
            # 保存最佳模型
            if success_rate > best_success_rate:
                best_success_rate = success_rate
                save_checkpoint(
                    policy_params, gnn_params,
                    policy_opt_state, gnn_opt_state,
                    step, args.save_dir
                )
                print(f"✅ 新的最佳模型 (成功率: {success_rate:.2%})")
        
        # 定期保存检查点
        if step % args.save_interval == 0 and step > 0:
            save_checkpoint(
                policy_params, gnn_params,
                policy_opt_state, gnn_opt_state,
                step, args.save_dir
            )
    
    print(f"\n🎉 训练完成!")
    print(f"最佳成功率: {best_success_rate:.2%}")
    
    # 保存最终模型
    save_checkpoint(
        policy_params, gnn_params,
        policy_opt_state, gnn_opt_state,
        args.num_steps, args.save_dir
    )
    
    # 保存训练历史
    history_path = Path(args.save_dir) / 'training_history.pkl'
    with open(history_path, 'wb') as f:
        pickle.dump(training_metrics, f)
    print(f"训练历史已保存至: {history_path}")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='安全敏捷飞行训练')
    
    # 训练参数
    parser.add_argument('--num_steps', type=int, default=1000,
                       help='训练步数')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='批次大小')
    parser.add_argument('--trajectory_length', type=int, default=30,
                       help='轨迹长度')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                       help='学习率')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                       help='梯度裁剪范数')
    
    # 系统参数
    parser.add_argument('--seed', type=int, default=42,
                       help='随机数种子')
    parser.add_argument('--gpu', action='store_true',
                       help='使用GPU')
    
    # 日志和保存
    parser.add_argument('--log_interval', type=int, default=10,
                       help='日志打印间隔')
    parser.add_argument('--eval_interval', type=int, default=100,
                       help='评估间隔')
    parser.add_argument('--save_interval', type=int, default=200,
                       help='保存间隔')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                       help='保存目录')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)