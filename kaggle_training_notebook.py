# =============================================================================
# 完整的 Kaggle 端到端可微分训练脚本
# Safe Agile Flight - JAX 实现
# =============================================================================

# Cell 1: 环境设置和依赖安装
print("=== 开始安装依赖项 ===")

# Kaggle 环境的依赖安装命令
import subprocess
import sys

def install_package(package):
    """安装包的辅助函数"""
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# 核心依赖
packages = [
    "jax[cuda12_pip]==0.4.20",
    "jaxlib==0.4.20",
    "flax==0.8.0", 
    "jraph==0.0.6.dev0",
    "optax==0.1.7",
    "ml-collections==0.1.1",
    "qpax",
    "chex==0.1.84"
]

for package in packages:
    try:
        print(f"正在安装 {package}...")
        install_package(package)
        print(f"✓ {package} 安装成功")
    except Exception as e:
        print(f"✗ {package} 安装失败: {e}")

print("=== 依赖安装完成 ===")

# =============================================================================
# Cell 2: 克隆项目和导入
print("=== 克隆项目代码 ===")

import os
import subprocess

# 克隆项目
if not os.path.exists('/kaggle/working/safe_agile_flight'):
    subprocess.run([
        'git', 'clone', 
        'https://github.com/niannian0922/safe_agile_flight.git',
        '/kaggle/working/safe_agile_flight'
    ], check=True)
    print("✓ 项目克隆成功")
else:
    print("✓ 项目已存在")

# 设置 Python 路径
sys.path.append('/kaggle/working/safe_agile_flight')

# =============================================================================
# Cell 3: 核心导入
print("=== 导入核心模块 ===")

import jax
import jax.numpy as jnp
from jax import random, jit, grad, vmap
import flax
import optax
import numpy as np
from functools import partial
import time
from typing import Dict, Tuple, Any, NamedTuple
import chex

# 导入项目模块
try:
    from configs.default_config import get_config
    from core.physics import dynamics_step, DroneState
    from core.policy import PolicyNetwork, PolicyParams
    from core.perception import PerceptionNetwork, pointcloud_to_graph
    from core.safety import safety_filter, build_qp_matrices
    from core.loop import scan_step_function, create_initial_carry
    from core.training import (
        compute_efficiency_loss, compute_cbf_loss, 
        compute_total_loss, create_train_step
    )
    from utils.core_helpers import tree_norm, safe_mean
    print("✓ 所有模块导入成功")
except ImportError as e:
    print(f"✗ 模块导入失败: {e}")
    # 这里可以添加备用的模块定义
    raise

# =============================================================================
# Cell 4: 配置和设备检查
print("=== 配置环境 ===")

# 检查 JAX 设备
print(f"JAX 版本: {jax.__version__}")
print(f"可用设备: {jax.devices()}")
print(f"默认后端: {jax.default_backend()}")

# 获取配置
config = get_config()
print(f"✓ 配置加载成功")
print(f"训练时间步: {config.training.horizon}")
print(f"批次大小: {config.training.batch_size}")
print(f"学习率: {config.training.learning_rate}")

# 设置随机种子
key = random.PRNGKey(config.training.seed)
print(f"✓ 随机种子设置: {config.training.seed}")

# =============================================================================
# Cell 5: 模型初始化
print("=== 初始化模型 ===")

def initialize_models(key, config):
    """初始化所有网络模型"""
    keys = random.split(key, 3)
    
    # 初始化策略网络
    policy_net = PolicyNetwork(
        hidden_sizes=config.policy.hidden_sizes,
        action_dim=config.drone.action_dim
    )
    
    # 创建虚拟输入来初始化参数
    dummy_state = DroneState(
        position=jnp.zeros(3),
        velocity=jnp.zeros(3),
        orientation=jnp.array([1., 0., 0., 0.]),  # 四元数
        angular_velocity=jnp.zeros(3)
    )
    
    policy_params = policy_net.init(keys[0], dummy_state)
    
    # 初始化感知网络 (如果需要)
    if config.perception.use_perception:
        perception_net = PerceptionNetwork(
            hidden_sizes=config.perception.hidden_sizes,
            output_dim=1  # CBF 输出
        )
        # 这里需要根据实际的感知输入来初始化
        perception_params = None  # 暂时设为 None
    else:
        perception_net = None
        perception_params = None
    
    return {
        'policy': policy_net,
        'policy_params': policy_params,
        'perception': perception_net,
        'perception_params': perception_params
    }

models = initialize_models(keys[0], config)
print("✓ 模型初始化完成")

# =============================================================================
# Cell 6: 优化器设置
print("=== 设置优化器 ===")

# 创建优化器
optimizer = optax.adam(learning_rate=config.training.learning_rate)

# 初始化优化器状态
opt_state = optimizer.init(models['policy_params'])
print("✓ 优化器初始化完成")

# =============================================================================
# Cell 7: 训练环境设置
print("=== 设置训练环境 ===")

def create_training_batch(key, config):
    """创建训练批次数据"""
    batch_keys = random.split(key, config.training.batch_size)
    
    # 创建初始状态批次
    initial_states = []
    for k in batch_keys:
        # 随机初始位置和速度
        pos_key, vel_key = random.split(k)
        position = random.uniform(pos_key, (3,), minval=-5.0, maxval=5.0)
        velocity = random.uniform(vel_key, (3,), minval=-2.0, maxval=2.0)
        
        state = DroneState(
            position=position,
            velocity=velocity,
            orientation=jnp.array([1., 0., 0., 0.]),
            angular_velocity=jnp.zeros(3)
        )
        initial_states.append(state)
    
    # 创建目标位置
    target_keys = random.split(key, config.training.batch_size)
    targets = []
    for tk in target_keys:
        target = random.uniform(tk, (3,), minval=-10.0, maxval=10.0)
        targets.append(target)
    
    return initial_states, targets

# 测试训练批次创建
test_states, test_targets = create_training_batch(keys[1], config)
print(f"✓ 训练批次创建成功 - 批次大小: {len(test_states)}")

# =============================================================================
# Cell 8: 端到端训练循环
print("=== 创建训练步骤函数 ===")

@partial(jit, static_argnames=['config'])
def train_step(policy_params, opt_state, batch_key, config):
    """单步训练函数"""
    
    # 创建训练数据
    initial_states, targets = create_training_batch(batch_key, config)
    
    # 定义损失函数
    def loss_fn(params):
        # 为每个样本运行轨迹展开
        def rollout_single(state, target):
            # 创建 scan 的初始 carry
            carry = create_initial_carry(state, config)
            
            # 外部输入 (目标信息)
            xs = jnp.tile(target, (config.training.horizon, 1))
            
            # 执行轨迹展开
            final_carry, trajectory = jax.lax.scan(
                partial(scan_step_function, 
                       policy_params=params,
                       perception_params=models['perception_params'],
                       config=config),
                carry, xs
            )
            
            return trajectory
        
        # 对批次中的所有样本进行 vmap
        trajectories = vmap(rollout_single)(initial_states, targets)
        
        # 计算损失
        efficiency_loss = compute_efficiency_loss(trajectories, targets, config)
        cbf_loss = compute_cbf_loss(trajectories, config) if config.safety.use_cbf else 0.0
        
        total_loss = efficiency_loss + config.training.cbf_weight * cbf_loss
        
        return total_loss, {
            'total_loss': total_loss,
            'efficiency_loss': efficiency_loss, 
            'cbf_loss': cbf_loss
        }
    
    # 计算梯度
    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(policy_params)
    
    # 更新参数
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(policy_params, updates)
    
    return new_params, new_opt_state, metrics

print("✓ 训练步骤函数创建完成")

# =============================================================================
# Cell 9: 完整训练循环
print("=== 开始端到端训练 ===")

def run_training(policy_params, opt_state, config, num_epochs=1000):
    """运行完整的训练循环"""
    
    key = random.PRNGKey(config.training.seed)
    
    # 训练指标记录
    training_history = {
        'total_loss': [],
        'efficiency_loss': [],
        'cbf_loss': [],
        'grad_norm': []
    }
    
    # 训练循环
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # 生成新的随机种子
        key, subkey = random.split(key)
        
        # 执行训练步骤
        policy_params, opt_state, metrics = train_step(
            policy_params, opt_state, subkey, config
        )
        
        # 记录指标
        training_history['total_loss'].append(float(metrics['total_loss']))
        training_history['efficiency_loss'].append(float(metrics['efficiency_loss']))
        training_history['cbf_loss'].append(float(metrics['cbf_loss']))
        
        # 计算梯度范数
        grad_norm = tree_norm(policy_params)
        training_history['grad_norm'].append(float(grad_norm))
        
        epoch_time = time.time() - epoch_start
        
        # 打印进度
        if epoch % 50 == 0 or epoch < 10:
            print(f"Epoch {epoch:4d} | "
                  f"Loss: {metrics['total_loss']:.6f} | "
                  f"Efficiency: {metrics['efficiency_loss']:.6f} | "
                  f"CBF: {metrics['cbf_loss']:.6f} | "
                  f"Grad Norm: {grad_norm:.6f} | "
                  f"Time: {epoch_time:.3f}s")
        
        # 早停检查
        if len(training_history['total_loss']) > 100:
            recent_losses = training_history['total_loss'][-50:]
            if np.std(recent_losses) < 1e-6:
                print(f"训练收敛，在第 {epoch} 轮停止")
                break
    
    return policy_params, opt_state, training_history

# 开始训练
print("开始端到端可微分训练...")
print(f"预计训练时间步数: {config.training.num_epochs}")

trained_params, final_opt_state, history = run_training(
    models['policy_params'], 
    opt_state, 
    config,
    num_epochs=config.training.num_epochs
)

print("✓ 训练完成!")

# =============================================================================
# Cell 10: 训练结果分析和可视化
print("=== 分析训练结果 ===")

import matplotlib.pyplot as plt

# 绘制训练曲线
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 总损失
axes[0,0].plot(history['total_loss'])
axes[0,0].set_title('总损失')
axes[0,0].set_xlabel('训练轮次')
axes[0,0].set_ylabel('损失值')
axes[0,0].grid(True)

# 效率损失
axes[0,1].plot(history['efficiency_loss'])
axes[0,1].set_title('效率损失')
axes[0,1].set_xlabel('训练轮次')
axes[0,1].set_ylabel('损失值')
axes[0,1].grid(True)

# CBF 损失
axes[1,0].plot(history['cbf_loss'])
axes[1,0].set_title('安全损失 (CBF)')
axes[1,0].set_xlabel('训练轮次')
axes[1,0].set_ylabel('损失值')
axes[1,0].grid(True)

# 梯度范数
axes[1,1].plot(history['grad_norm'])
axes[1,1].set_title('梯度范数')
axes[1,1].set_xlabel('训练轮次')
axes[1,1].set_ylabel('范数')
axes[1,1].grid(True)

plt.tight_layout()
plt.savefig('/kaggle/working/training_curves.png', dpi=300, bbox_inches='tight')
plt.show()

# 打印最终统计
print("\n=== 训练统计 ===")
print(f"最终总损失: {history['total_loss'][-1]:.6f}")
print(f"最终效率损失: {history['efficiency_loss'][-1]:.6f}")
print(f"最终安全损失: {history['cbf_loss'][-1]:.6f}")
print(f"最终梯度范数: {history['grad_norm'][-1]:.6f}")

# =============================================================================
# Cell 11: 模型保存
print("=== 保存训练好的模型 ===")

import pickle

# 保存模型参数
model_save_path = '/kaggle/working/trained_model.pkl'
with open(model_save_path, 'wb') as f:
    pickle.dump({
        'policy_params': trained_params,
        'config': config,
        'training_history': history
    }, f)

print(f"✓ 模型已保存到: {model_save_path}")

# =============================================================================
# Cell 12: 模型评估
print("=== 评估训练好的模型 ===")

def evaluate_model(policy_params, config, num_test_episodes=100):
    """评估模型性能"""
    
    key = random.PRNGKey(42)  # 固定种子用于评估
    
    success_count = 0
    total_distances = []
    total_times = []
    
    for episode in range(num_test_episodes):
        key, subkey = random.split(key)
        
        # 创建测试场景
        test_states, test_targets = create_training_batch(subkey, config)
        
        # 选择一个测试样本
        test_state = test_states[0] 
        test_target = test_targets[0]
        
        # 运行轨迹
        carry = create_initial_carry(test_state, config)
        xs = jnp.tile(test_target, (config.training.horizon, 1))
        
        final_carry, trajectory = jax.lax.scan(
            partial(scan_step_function,
                   policy_params=policy_params,
                   perception_params=models['perception_params'],
                   config=config),
            carry, xs
        )
        
        # 计算最终距离
        final_position = trajectory.states[-1].position
        final_distance = jnp.linalg.norm(final_position - test_target)
        
        # 判断是否成功 (距离小于阈值)
        if final_distance < config.evaluation.success_threshold:
            success_count += 1
        
        total_distances.append(float(final_distance))
        total_times.append(config.training.horizon * config.simulation.dt)
    
    success_rate = success_count / num_test_episodes
    avg_distance = np.mean(total_distances)
    avg_time = np.mean(total_times)
    
    print(f"\n=== 评估结果 ===")
    print(f"成功率: {success_rate:.2%}")
    print(f"平均最终距离: {avg_distance:.4f}")
    print(f"平均任务时间: {avg_time:.2f}s")
    
    return {
        'success_rate': success_rate,
        'avg_distance': avg_distance,
        'avg_time': avg_time
    }

# 运行评估
evaluation_results = evaluate_model(trained_params, config)

print("\n=== 端到端可微分训练完成! ===")
print("训练结果已保存到 Kaggle 输出目录")