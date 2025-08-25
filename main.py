#!/usr/bin/env python3
"""
主训练脚本：安全敏捷飞行端到端学习系统
使用修复后的训练系统
"""

import jax
import jax.numpy as jnp
import time
from typing import Dict, List
import argparse
import os
import pickle
from pathlib import Path

from core.physics import create_initial_state
from core.training import (
    TrainingConfig, TrainingSystem,
    CompleteTrainingConfig, CompleteTrainingSystem
)


def train_basic_system(args):
    """训练基础系统"""
    print("🚁 基础系统训练")
    print("=" * 40)
    
    # 配置
    config = TrainingConfig(
        learning_rate=args.learning_rate,
        trajectory_length=args.trajectory_length,
        batch_size=args.batch_size,
        gradient_clip_norm=args.grad_clip,
    )
    
    print(f"轨迹长度: {config.trajectory_length}")
    print(f"学习率: {config.learning_rate}")
    
    # 设置阶段
    rng_key = jax.random.PRNGKey(args.seed)
    training_system = TrainingSystem(config, rng_key)
    
    print("✅ 系统初始化完成")
    
    # 训练循环
    training_state = training_system.get_initial_training_state()
    best_loss = float('inf')
    
    print(f"开始训练 ({args.num_steps} 步)...")
    
    for step in range(args.num_steps):
        step_start_time = time.time()
        
        # 生成训练数据
        step_key = jax.random.fold_in(rng_key, step)
        
        # 随机初始状态
        init_pos = jax.random.uniform(step_key, (3,), minval=-2.0, maxval=2.0)
        initial_state = create_initial_state(position=init_pos)
        
        # 随机目标位置
        target_pos = jax.random.uniform(
            jax.random.fold_in(step_key, 1), (3,), minval=3.0, maxval=8.0
        )
        
        # 执行训练步骤
        training_state, train_info = training_system.train_step(
            training_state, initial_state, target_pos
        )
        
        step_time = time.time() - step_start_time
        current_loss = train_info['total_loss']
        
        # 更新最佳损失
        if current_loss < best_loss:
            best_loss = current_loss
        
        # 打印进度
        if step % args.log_interval == 0:
            print(f"步骤 {step:4d}: "
                  f"损失={current_loss:.4f}, "
                  f"最佳={best_loss:.4f}, "
                  f"梯度范数={train_info['grad_norm']:.6f}, "
                  f"最终距离={train_info['final_distance']:.3f}, "
                  f"时间={step_time:.3f}s")
    
    print(f"\n训练完成! 最佳损失: {best_loss:.4f}")


def main(args):
    """主函数"""
    
    print("🚁 安全敏捷飞行端到端学习系统")
    print("=" * 50)
    
    # 设置JAX
    if args.gpu:
        print("使用GPU加速")
    else:
        jax.config.update("jax_platform_name", "cpu")
        print("使用CPU训练")
    
    # 根据模式选择训练系统
    if args.mode == "basic":
        train_basic_system(args)
    elif args.mode == "complete":
        print("完整系统训练功能开发中...")
    else:
        print("运行系统验证...")
        from core.training import test_gradient_flow, test_complete_gradient_flow
        
        print("\n=== 基础梯度流测试 ===")
        basic_success = test_gradient_flow()
        
        print("\n=== 完整系统梯度流测试 ===")
        complete_success = test_complete_gradient_flow()
        
        if basic_success and complete_success:
            print("\n🎉 系统验证通过!")
        else:
            print("\n❌ 系统验证失败")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='安全敏捷飞行训练')
    
    # 训练模式
    parser.add_argument('--mode', type=str, default='test',
                       choices=['basic', 'complete', 'test'],
                       help='训练模式')
    
    # 训练参数
    parser.add_argument('--num_steps', type=int, default=500,
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
    
    # 日志参数
    parser.add_argument('--log_interval', type=int, default=50,
                       help='日志打印间隔')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)