"""
安全敏捷飞行系统的内存优化工具。

此模块提供以下工具：
1. 在训练过程中检测和管理内存使用
2. 基于可用内存自动调整序列长度
3. 提供内存安全的配置默认值
4. 在训练过程中监控内存使用

目标是在长序列训练期间防止内存溢出，
同时保持训练有效性。
"""

import jax
import jax.numpy as jnp
import psutil
import gc
from typing import Dict, Tuple, Optional
import warnings


def get_memory_info() -> Dict[str, float]:
    """获取当前内存使用信息"""
    try:
        # 获取系统内存信息
        memory = psutil.virtual_memory()
        
        # 获取JAX设备内存信息（如果可用）
        devices = jax.devices()
        device_memory = {}
        
        for i, device in enumerate(devices):
            try:
                if hasattr(device, 'memory_stats'):
                    stats = device.memory_stats()
                    device_memory[f'device_{i}'] = {
                        'used': stats.get('bytes_in_use', 0) / 1e9,  # GB 已使用
                        'total': stats.get('peak_bytes_in_use', 0) / 1e9  # GB 总计
                    }
            except:
                pass  # 设备不支持内存统计
        
        return {
            'system_total_gb': memory.total / 1e9,
            'system_available_gb': memory.available / 1e9,
            'system_used_percent': memory.percent,
            'device_memory': device_memory
        }
    except Exception as e:
        warnings.warn(f"Could not get memory info: {e}")
        return {'system_total_gb': 8.0, 'system_available_gb': 4.0, 'system_used_percent': 50.0}


def estimate_memory_usage(batch_size: int, sequence_length: int, model_size: str = "medium") -> float:
    """
    估计给定配置的内存使用量
    
    参数:
        batch_size: 训练批处理大小
        sequence_length: BPTT序列长度 
        model_size: "small"、"medium"或"large"
        
    返回值:
        估计的内存使用量（GB）
    """
    
    # 基础内存估计（粗略近似）
    base_memory = {
        "small": 1.0,   # GB
        "medium": 2.5,  # GB  
        "large": 5.0    # GB
    }
    
    # 内存缩放因子
    batch_factor = batch_size / 16.0  # 参考批处理大小
    sequence_factor = sequence_length / 20.0  # 参考序列长度
    
    # 估计总内存
    estimated_memory = base_memory[model_size] * batch_factor * sequence_factor
    
    # 为JAX编译和中间值添加缓冲区
    estimated_memory *= 1.5
    
    return estimated_memory


def get_memory_safe_config(base_config, target_memory_gb: float = 4.0):
    """
    调整配置以确保内存安全
    
    参数:
        base_config: 要调整的基础配置
        target_memory_gb: 目标最大内存使用量（GB）
        
    返回值:
        内存安全的配置
    """
    config = base_config
    
    # 获取当前内存信息
    memory_info = get_memory_info()
    available_memory = min(target_memory_gb, memory_info['system_available_gb'] * 0.8)
    
    print(f"🧠 Memory optimization target: {available_memory:.1f}GB")
    
    # 从当前配置开始
    current_batch_size = config.training.batch_size
    current_seq_length = config.training.sequence_length
    
    # 估计当前内存使用量
    current_memory = estimate_memory_usage(current_batch_size, current_seq_length, "medium")
    
    if current_memory <= available_memory:
        print(f"✅ Current config fits in memory: {current_memory:.1f}GB")
        return config
    
    print(f"⚠️ Current config may exceed memory: {current_memory:.1f}GB > {available_memory:.1f}GB")
    print("🔧 Adjusting configuration for memory safety...")
    
    # 调整参数以适应内存
    # 优先级：首先减少序列长度，然后减少批处理大小
    
    # 尝试减少序列长度
    safe_seq_length = current_seq_length
    while safe_seq_length > 5:
        test_memory = estimate_memory_usage(current_batch_size, safe_seq_length, "medium")
        if test_memory <= available_memory:
            break
        safe_seq_length = max(5, int(safe_seq_length * 0.8))
    
    # 如果仍然太大，减少批处理大小
    safe_batch_size = current_batch_size
    while safe_batch_size > 1:
        test_memory = estimate_memory_usage(safe_batch_size, safe_seq_length, "medium")
        if test_memory <= available_memory:
            break
        safe_batch_size = max(1, int(safe_batch_size * 0.8))
    
    # 更新配置
    if safe_seq_length != current_seq_length:
        config.training.sequence_length = safe_seq_length
        print(f"   Reduced sequence length: {current_seq_length} → {safe_seq_length}")
    
    if safe_batch_size != current_batch_size:
        config.training.batch_size = safe_batch_size
        print(f"   Reduced batch size: {current_batch_size} → {safe_batch_size}")
    
    # 同时调整其他内存敏感参数
    if current_memory > available_memory * 1.5:
        # 为严重受限的内存减少模型大小
        config.policy.hidden_dims = [min(128, d) for d in config.policy.hidden_dims]
        config.gcbf.gnn.hidden_dims = [min(128, d) for d in config.gcbf.gnn.hidden_dims]
        print("   Reduced model sizes for memory constraints")
    
    final_memory = estimate_memory_usage(
        config.training.batch_size, 
        config.training.sequence_length, 
        "medium"
    )
    
    print(f"✅ Memory-optimized config: {final_memory:.1f}GB (target: {available_memory:.1f}GB)")
    
    return config


def clear_jax_cache():
    """清除JAX编译缓存并运行垃圾回收"""
    try:
        # 清除JAX缓存（如果可用）
        if hasattr(jax, 'clear_caches'):
            jax.clear_caches()
        
        # 强制垃圾回收
        gc.collect()
        
        print("🧹 Cleared JAX cache and ran garbage collection")
    except Exception as e:
        warnings.warn(f"Could not clear cache: {e}")


def monitor_training_memory(step: int, clear_every: int = 50):
    """在训练过程中监控内存使用量，必要时清除缓存"""
    if step % clear_every == 0 and step > 0:
        memory_info = get_memory_info()
        
        if memory_info['system_used_percent'] > 85:
            print(f"⚠️ High memory usage at step {step}: {memory_info['system_used_percent']:.1f}%")
            clear_jax_cache()
            
            # 清理后再次检查
            new_memory_info = get_memory_info()
            print(f"   Memory after cleanup: {new_memory_info['system_used_percent']:.1f}%")


def get_debug_config(base_config):
    """获取具有最小内存使用量的调试配置"""
    config = base_config
    
    # 调试的最小设置
    config.training.batch_size = 2
    config.training.sequence_length = 5
    config.training.num_epochs = 2
    config.training.batches_per_epoch = 3
    config.training.validation_batch_size = 2
    
    # 降低模型复杂度
    config.policy.hidden_dims = [32, 32]
    config.gcbf.gnn.hidden_dims = [64, 64, 32]
    config.gcbf.k_neighbors = 3
    config.gcbf.max_neighbors = 4
    
    # 禁用昂贵的功能
    config.optimization.use_checkpoint = False
    config.optimization.nested_checkpoint = False
    config.logging.video_logging = False
    config.training.curriculum.enable = False
    
    print("🐛 Using debug configuration with minimal memory usage")
    return config


def validate_memory_config(config) -> bool:
    """验证配置对于可用内存是合理的"""
    memory_info = get_memory_info()
    estimated_usage = estimate_memory_usage(
        config.training.batch_size,
        config.training.sequence_length,
        "medium"
    )
    
    available_memory = memory_info['system_available_gb']
    
    if estimated_usage > available_memory * 0.9:
        print(f"❌ Configuration may exceed available memory:")
        print(f"   Estimated usage: {estimated_usage:.1f}GB")
        print(f"   Available memory: {available_memory:.1f}GB")
        return False
    
    print(f"✅ Memory configuration validated:")
    print(f"   Estimated usage: {estimated_usage:.1f}GB")
    print(f"   Available memory: {available_memory:.1f}GB")
    return True


if __name__ == "__main__":
    # 测试内存工具
    print("Testing memory optimization utilities...")
    
    memory_info = get_memory_info()
    print(f"System memory: {memory_info}")
    
    # 测试内存估计
    for batch_size in [2, 8, 16]:
        for seq_len in [5, 20, 50]:
            usage = estimate_memory_usage(batch_size, seq_len, "medium")
            print(f"Batch {batch_size}, Seq {seq_len}: ~{usage:.1f}GB")