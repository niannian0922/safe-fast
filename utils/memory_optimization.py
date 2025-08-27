"""
Memory optimization utilities for Safe Agile Flight system.

This module provides utilities to:
1. Detect and manage memory usage during training
2. Automatically adjust sequence lengths based on available memory
3. Provide memory-safe configuration defaults
4. Monitor memory usage during training

The goal is to prevent memory overflow during long sequence training
while maintaining training effectiveness.
"""

import jax
import jax.numpy as jnp
import psutil
import gc
from typing import Dict, Tuple, Optional
import warnings


def get_memory_info() -> Dict[str, float]:
    """Get current memory usage information"""
    try:
        # Get system memory info
        memory = psutil.virtual_memory()
        
        # Get JAX device memory info if available
        devices = jax.devices()
        device_memory = {}
        
        for i, device in enumerate(devices):
            try:
                if hasattr(device, 'memory_stats'):
                    stats = device.memory_stats()
                    device_memory[f'device_{i}'] = {
                        'used': stats.get('bytes_in_use', 0) / 1e9,  # GB
                        'total': stats.get('peak_bytes_in_use', 0) / 1e9  # GB
                    }
            except:
                pass  # Device doesn't support memory stats
        
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
    Estimate memory usage for a given configuration
    
    Args:
        batch_size: Training batch size
        sequence_length: BPTT sequence length 
        model_size: "small", "medium", or "large"
        
    Returns:
        Estimated memory usage in GB
    """
    
    # Base memory estimates (rough approximations)
    base_memory = {
        "small": 1.0,   # GB
        "medium": 2.5,  # GB  
        "large": 5.0    # GB
    }
    
    # Memory scaling factors
    batch_factor = batch_size / 16.0  # Reference batch size
    sequence_factor = sequence_length / 20.0  # Reference sequence length
    
    # Estimate total memory
    estimated_memory = base_memory[model_size] * batch_factor * sequence_factor
    
    # Add buffer for JAX compilation and intermediate values
    estimated_memory *= 1.5
    
    return estimated_memory


def get_memory_safe_config(base_config, target_memory_gb: float = 4.0):
    """
    Adjust configuration to be memory safe
    
    Args:
        base_config: Base configuration to adjust
        target_memory_gb: Target maximum memory usage in GB
        
    Returns:
        Memory-safe configuration
    """
    config = base_config
    
    # Get current memory info
    memory_info = get_memory_info()
    available_memory = min(target_memory_gb, memory_info['system_available_gb'] * 0.8)
    
    print(f"🧠 Memory optimization target: {available_memory:.1f}GB")
    
    # Start with current configuration
    current_batch_size = config.training.batch_size
    current_seq_length = config.training.sequence_length
    
    # Estimate current memory usage
    current_memory = estimate_memory_usage(current_batch_size, current_seq_length, "medium")
    
    if current_memory <= available_memory:
        print(f"✅ Current config fits in memory: {current_memory:.1f}GB")
        return config
    
    print(f"⚠️ Current config may exceed memory: {current_memory:.1f}GB > {available_memory:.1f}GB")
    print("🔧 Adjusting configuration for memory safety...")
    
    # Adjust parameters to fit memory
    # Priority: reduce sequence length first, then batch size
    
    # Try reducing sequence length
    safe_seq_length = current_seq_length
    while safe_seq_length > 5:
        test_memory = estimate_memory_usage(current_batch_size, safe_seq_length, "medium")
        if test_memory <= available_memory:
            break
        safe_seq_length = max(5, int(safe_seq_length * 0.8))
    
    # If still too large, reduce batch size
    safe_batch_size = current_batch_size
    while safe_batch_size > 1:
        test_memory = estimate_memory_usage(safe_batch_size, safe_seq_length, "medium")
        if test_memory <= available_memory:
            break
        safe_batch_size = max(1, int(safe_batch_size * 0.8))
    
    # Update configuration
    if safe_seq_length != current_seq_length:
        config.training.sequence_length = safe_seq_length
        print(f"   Reduced sequence length: {current_seq_length} → {safe_seq_length}")
    
    if safe_batch_size != current_batch_size:
        config.training.batch_size = safe_batch_size
        print(f"   Reduced batch size: {current_batch_size} → {safe_batch_size}")
    
    # Also adjust other memory-sensitive parameters
    if current_memory > available_memory * 1.5:
        # Reduce model sizes for very constrained memory
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
    """Clear JAX compilation cache and run garbage collection"""
    try:
        # Clear JAX cache if available
        if hasattr(jax, 'clear_caches'):
            jax.clear_caches()
        
        # Force garbage collection
        gc.collect()
        
        print("🧹 Cleared JAX cache and ran garbage collection")
    except Exception as e:
        warnings.warn(f"Could not clear cache: {e}")


def monitor_training_memory(step: int, clear_every: int = 50):
    """Monitor memory usage during training and clear cache if needed"""
    if step % clear_every == 0 and step > 0:
        memory_info = get_memory_info()
        
        if memory_info['system_used_percent'] > 85:
            print(f"⚠️ High memory usage at step {step}: {memory_info['system_used_percent']:.1f}%")
            clear_jax_cache()
            
            # Check again after cleanup
            new_memory_info = get_memory_info()
            print(f"   Memory after cleanup: {new_memory_info['system_used_percent']:.1f}%")


def get_debug_config(base_config):
    """Get a debug configuration with minimal memory usage"""
    config = base_config
    
    # Minimal settings for debugging
    config.training.batch_size = 2
    config.training.sequence_length = 5
    config.training.num_epochs = 2
    config.training.batches_per_epoch = 3
    config.training.validation_batch_size = 2
    
    # Reduce model complexity
    config.policy.hidden_dims = [32, 32]
    config.gcbf.gnn.hidden_dims = [64, 64, 32]
    config.gcbf.k_neighbors = 3
    config.gcbf.max_neighbors = 4
    
    # Disable expensive features
    config.optimization.use_checkpoint = False
    config.optimization.nested_checkpoint = False
    config.logging.video_logging = False
    config.training.curriculum.enable = False
    
    print("🐛 Using debug configuration with minimal memory usage")
    return config


def validate_memory_config(config) -> bool:
    """Validate that the configuration is reasonable for available memory"""
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
    # Test memory utilities
    print("Testing memory optimization utilities...")
    
    memory_info = get_memory_info()
    print(f"System memory: {memory_info}")
    
    # Test memory estimation
    for batch_size in [2, 8, 16]:
        for seq_len in [5, 20, 50]:
            usage = estimate_memory_usage(batch_size, seq_len, "medium")
            print(f"Batch {batch_size}, Seq {seq_len}: ~{usage:.1f}GB")