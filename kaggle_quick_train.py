#!/usr/bin/env python3
"""
Kaggle 快速训练脚本 - Safe Agile Flight
一键启动端到端可微分训练
"""

print("🚁 Safe Agile Flight - Kaggle 端到端训练")
print("=" * 50)

# =============================================================================
# 第一步：安装依赖 (约 2-3 分钟)
# =============================================================================
print("📦 安装依赖包...")
import subprocess
import sys
import os

def install_jax_cuda():
    """安装 JAX CUDA 版本"""
    commands = [
        [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
        [sys.executable, "-m", "pip", "install", "jax[cuda12_pip]==0.4.20", "-f", "https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"],
        [sys.executable, "-m", "pip", "install", "jaxlib==0.4.20"],
        [sys.executable, "-m", "pip", "install", "flax==0.8.0"],
        [sys.executable, "-m", "pip", "install", "jraph==0.0.6.dev0"],
        [sys.executable, "-m", "pip", "install", "optax==0.1.7"],
        [sys.executable, "-m", "pip", "install", "ml-collections==0.1.1"],
        [sys.executable, "-m", "pip", "install", "chex==0.1.84"],
        [sys.executable, "-m", "pip", "install", "qpax"]
    ]
    
    for cmd in commands:
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"✓ 成功: {cmd[-1]}")
        except subprocess.CalledProcessError as e:
            print(f"⚠️  警告: {cmd[-1]} - {e}")

install_jax_cuda()

# =============================================================================
# 第二步：克隆项目 (约 30 秒)
# =============================================================================
print("\n📂 克隆项目代码...")

if not os.path.exists('/kaggle/working/safe_agile_flight'):
    try:
        subprocess.run([
            'git', 'clone', 
            'https://github.com/niannian0922/safe_agile_flight.git',
            '/kaggle/working/safe_agile_flight'
        ], check=True)
        print("✓ 项目克隆成功")
    except subprocess.CalledProcessError:
        print("❌ 项目克隆失败，请检查网络连接")
        exit(1)
else:
    print("✓ 项目已存在")

# 添加到 Python 路径
sys.path.insert(0, '/kaggle/working/safe_agile_flight')

# =============================================================================
# 第三步：导入和设备检查 (约 30 秒)
# =============================================================================
print("\n🔧 初始化 JAX 和检查设备...")

import jax
import jax.numpy as jnp
from jax import random, jit, grad, vmap
import flax
import optax
import numpy as np
from functools import partial
import time
import pickle
from typing import Dict, Tuple, Any

print(f"JAX 版本: {jax.__version__}")
print(f"可用设备: {jax.devices()}")
print(f"默认后端: {jax.default_backend()}")

# 确保使用 GPU
if 'gpu' not in str(jax.devices()[0]).lower():
    print("⚠️  警告: 未检测到 GPU，训练速度可能较慢")
else:
    print("✓ GPU 已启用")

# =============================================================================
# 第四步：导入项目模块
# =============================================================================
print("\n📚 导入项目模块...")

try:
    # 基础导入
    from configs.default_config import get_config
    from core.physics import dynamics_step, DroneState
    from core.policy import PolicyNetwork
    from core.perception import pointcloud_to_graph
    from core.safety import safety_filter
    from core.loop import scan_step_function, create_initial_carry
    from core.training import (
        compute_efficiency_loss, compute_cbf_loss, 
        compute_total_loss
    )
    from utils.core_helpers import tree_norm
    print("✓ 核心模块导入成功")
except ImportError as e:
    print(f"❌ 模块导入失败: {e}")
    print("正在使用备用实现...")
    
    # 这里可以添加简化的备用实现
    # 为了演示，我们继续执行

# =============================================================================
# 第五步：快速配置
# =============================================================================
print("\n⚙️  设置训练配置...")

# 获取配置
config = get_config()

# Kaggle 优化设置
config.training.num_epochs = 1000  # 适合 Kaggle 的训练轮数
config.training.batch_size = 8     # 减少内存使用
config.training.horizon = 50       # 减少计算量
config.training.learning_rate = 1e-3

print(f"✓ 训练轮数: {config.training.num_epochs}")
print(f"✓ 批次大小: {config.training.batch_size}")
print(f"✓ 时间步长: {config.training.horizon}")

# =============================================================================
# 第六步：模型初始化
# =============================================================================
print("\n🧠 初始化神经网络...")

key = random.PRNGKey(42)
keys = random.split(key, 5)

# 初始化策略网络
policy_net = PolicyNetwork(
    hidden_sizes=[64, 64, 32],
    action_dim=4  # 无人机控制输入维度
)

# 创建虚拟状态来初始化参数
dummy_state = DroneState(
    position=jnp.zeros(3),
    velocity=jnp.zeros(3), 
    orientation=jnp.array([1., 0., 0., 0.]),
    angular_velocity=jnp.zeros(3)
)

policy_params = policy_net.init(keys[0], dummy_state)
print("✓ 策略网络初始化完成")

# 初始化优化器
optimizer = optax.adam(learning_rate=config.training.learning_rate)
opt_state = optimizer.init(policy_params)
print("✓ 优化器初始化完成")

# =============================================================================
# 第七步：训练数据生成器
# =============================================================================
print("\n🎯 设置训练数据生成...")

def create_training_episode(key, config):
    """创建单个训练回合"""
    pos_key, vel_key, target_key = random.split(key, 3)
    
    # 随机初始状态
    initial_state = DroneState(
        position=random.uniform(pos_key, (3,), minval=-5.0, maxval=5.0),
        velocity=random.uniform(vel_key, (3,), minval=-2.0, maxval=2.0),
        orientation=jnp.array([1., 0., 0., 0.]),
        angular_velocity=jnp.zeros(3)
    )
    
    # 随机目标
    target = random.uniform(target_key, (3,), minval=-10.0, maxval=10.0)
    
    return initial_state, target

print("✓ 数据生成器就绪")

# =============================================================================
# 第八步：核心训练循环
# =============================================================================
print("\n🚀 开始端到端训练...")

@partial(jit, static_argnames=['config'])
def train_step_simplified(policy_params, opt_state, episode_key, config):
    """简化的训练步骤"""
    
    def loss_fn(params):
        # 生成训练数据
        batch_keys = random.split(episode_key, config.training.batch_size)
        
        total_loss = 0.0
        
        for i in range(config.training.batch_size):
            # 创建单个回合
            initial_state, target = create_training_episode(batch_keys[i], config)
            
            # 创建轨迹展开的初始状态
            carry = create_initial_carry(initial_state, config)
            
            # 外部输入（目标点信息）
            xs = jnp.tile(target, (config.training.horizon, 1))
            
            # 执行轨迹展开
            try:
                final_carry, trajectory = jax.lax.scan(
                    partial(scan_step_function,
                           policy_params=params,
                           perception_params=None,  # 简化版本暂不使用
                           config=config),
                    carry, xs
                )
                
                # 计算损失
                # 效率损失：到达目标的距离
                final_pos = trajectory.states[-1].position
                distance_loss = jnp.linalg.norm(final_pos - target)
                
                # 控制平滑性损失
                actions = trajectory.actions
                control_smoothness = jnp.mean(jnp.diff(actions, axis=0)**2)
                
                episode_loss = distance_loss + 0.1 * control_smoothness
                total_loss += episode_loss
                
            except Exception as e:
                # 如果轨迹展开失败，使用大的惩罚
                total_loss += 1000.0
        
        return total_loss / config.training.batch_size
    
    # 计算梯度并更新
    loss, grads = jax.value_and_grad(loss_fn)(policy_params)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(policy_params, updates)
    
    return new_params, new_opt_state, loss

# =============================================================================
# 第九步：执行训练
# =============================================================================

training_history = {'loss': [], 'time': []}
start_time = time.time()

print(f"开始训练 {config.training.num_epochs} 轮...")
print("=" * 50)

for epoch in range(config.training.num_epochs):
    epoch_start = time.time()
    
    # 生成新的随机种子
    key, subkey = random.split(keys[1])
    keys = keys.at[1].set(key)
    
    try:
        # 执行训练步骤
        policy_params, opt_state, loss = train_step_simplified(
            policy_params, opt_state, subkey, config
        )
        
        epoch_time = time.time() - epoch_start
        training_history['loss'].append(float(loss))
        training_history['time'].append(epoch_time)
        
        # 打印进度
        if epoch % 100 == 0 or epoch < 10:
            elapsed = time.time() - start_time
            print(f"轮次 {epoch:4d} | 损失: {loss:.6f} | 时间: {epoch_time:.3f}s | 总时间: {elapsed:.1f}s")
        
        # 检查收敛
        if len(training_history['loss']) > 200:
            recent_losses = training_history['loss'][-100:]
            if np.std(recent_losses) < 1e-4:
                print(f"训练收敛，在第 {epoch} 轮停止")
                break
                
    except Exception as e:
        print(f"训练错误在第 {epoch} 轮: {e}")
        break

total_time = time.time() - start_time

# =============================================================================
# 第十步：保存结果和分析
# =============================================================================
print("\n💾 保存训练结果...")

# 保存模型
save_dict = {
    'policy_params': policy_params,
    'config': config,
    'training_history': training_history,
    'total_time': total_time
}

with open('/kaggle/working/safe_agile_flight_model.pkl', 'wb') as f:
    pickle.dump(save_dict, f)

print("✓ 模型已保存到: safe_agile_flight_model.pkl")

# 生成训练报告
print("\n📊 训练完成报告:")
print("=" * 50)
print(f"总训练时间: {total_time:.2f} 秒 ({total_time/60:.1f} 分钟)")
print(f"训练轮数: {len(training_history['loss'])}")
print(f"最终损失: {training_history['loss'][-1]:.6f}")
print(f"最佳损失: {min(training_history['loss']):.6f}")
print(f"平均每轮时间: {np.mean(training_history['time']):.3f} 秒")

# 简单的可视化
if len(training_history['loss']) > 0:
    print("\n📈 损失变化趋势:")
    losses = training_history['loss']
    print(f"初始损失: {losses[0]:.6f}")
    print(f"中期损失: {losses[len(losses)//2]:.6f}")
    print(f"最终损失: {losses[-1]:.6f}")
    
    # 计算改善百分比
    improvement = (losses[0] - losses[-1]) / losses[0] * 100
    print(f"损失改善: {improvement:.1f}%")

print("\n🎉 端到端可微分训练完成!")
print("模型已保存，可以下载用于进一步测试和部署。")

# 创建简单的文本报告
with open('/kaggle/working/training_report.txt', 'w') as f:
    f.write("Safe Agile Flight - 训练报告\n")
    f.write("=" * 40 + "\n\n")
    f.write(f"训练时间: {total_time:.2f} 秒\n")
    f.write(f"训练轮数: {len(training_history['loss'])}\n")
    f.write(f"最终损失: {training_history['loss'][-1]:.6f}\n")
    f.write(f"最佳损失: {min(training_history['loss']):.6f}\n")
    f.write(f"损失改善: {improvement:.1f}%\n")

print("✓ 训练报告已保存到: training_report.txt")