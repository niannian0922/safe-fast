"""
JAX PyTree批处理工具

此模块提供工具来处理JAX中基本的"结构数组 vs 数组结构"
问题。它实现了个别结构化对象（如DroneState）与批处理兼容的JAX数组
之间的无缝转换。

关键函数：
- batch_pytree_objects: 将 [struct1, struct2, ...] -> struct_of_arrays
- unbatch_pytree_objects: 将 struct_of_arrays -> [struct1, struct2, ...]
- stack_pytree_list: PyTree对象的安全堆叠
- validate_batch_structure: 验证批处理一致性

这解决了JAX操作（如jnp.stack）无法直接处理
Flax @struct.dataclass对象的核心问题。
"""

import jax
import jax.numpy as jnp
from jax import tree_util
from typing import List, Any, Union, Dict, TypeVar, Generic
import chex
from flax import struct

T = TypeVar('T')

def batch_pytree_objects(objects_list: List[T]) -> T:
    """
    将PyTree对象列表转换为批处理PyTree。
    
    将 [struct1, struct2, ...] -> struct_of_arrays 转换
    这是解决JAX"结构数组"限制的核心方案。
    
    参数:
        objects_list: PyTree对象列表（如DroneState实例）
        
    返回值:
        批处理PyTree，其中每个叶子沿批处理维度堆叠
        
    示例:
        states = [DroneState(...), DroneState(...), DroneState(...)]
        batched_states = batch_pytree_objects(states)
        # 结果: DroneState with position.shape = [3, 3] instead of [3]
    """
    if not objects_list:
        raise ValueError("Cannot batch empty list")
        
    # 验证所有对象具有相同结构
    first_obj = objects_list[0]
    for i, obj in enumerate(objects_list[1:], 1):
        if not tree_util.tree_structure(obj) == tree_util.tree_structure(first_obj):
            raise ValueError(f"Object {i} has different structure than object 0")
    
    # 使用tree_map和堆叠来创建批处理维度
    return tree_util.tree_map(
        lambda *args: jnp.stack(args, axis=0), 
        *objects_list
    )

def unbatch_pytree_objects(batched_struct: T) -> List[T]:
    """
    将批处理PyTree转换回个别PyTree的列表。
    
    将 struct_of_arrays -> [struct1, struct2, ...] 转换
    
    参数:
        batched_struct: 以批处理数组作为叶子的PyTree
        
    返回值:
        个别PyTree对象的列表
        
    示例:
        batched_states = DroneState(position=jnp.array([[1,2,3], [4,5,6]]), ...)
        states = unbatch_pytree_objects(batched_states)
        # 结果: [DroneState(position=[1,2,3]), DroneState(position=[4,5,6])]
    """
    # 从第一个叶子获取批处理大小
    first_leaf = tree_util.tree_leaves(batched_struct)[0]
    if first_leaf.ndim == 0:
        raise ValueError("Cannot unbatch scalar (0-dimensional) arrays")
        
    batch_size = first_leaf.shape[0]
    
    # 提取每个批处理元素
    individual_objects = []
    for i in range(batch_size):
        individual_obj = tree_util.tree_map(
            lambda x: x[i], 
            batched_struct
        )
        individual_objects.append(individual_obj)
    
    return individual_objects

def stack_pytree_list(objects_list: List[T], axis: int = 0) -> T:
    """
    使用自定义轴的PyTree对象安全堆叠。
    
    参数:
        objects_list: 要堆叠的PyTree对象列表
        axis: 堆叠的轴（默认值：0表示批处理维度）
        
    返回值:
        堆叠后的PyTree
    """
    if not objects_list:
        raise ValueError("Cannot stack empty list")
        
    return tree_util.tree_map(
        lambda *args: jnp.stack(args, axis=axis),
        *objects_list
    )

def validate_batch_structure(batched_struct: Any, expected_batch_size: int) -> bool:
    """
    验证批处理PyTree具有一致的结构。
    
    参数:
        batched_struct: 要验证的批处理PyTree
        expected_batch_size: 批处理维度的期望大小
        
    返回值:
        如果结构有效则为True，否则为False
    """
    try:
        leaves = tree_util.tree_leaves(batched_struct)
        
        if not leaves:
            return False
            
        # 检查所有叶子具有正确的批处理维度
        for leaf in leaves:
            if not hasattr(leaf, 'shape'):
                return False
            if leaf.ndim == 0:
                return False
            if leaf.shape[0] != expected_batch_size:
                return False
                
        return True
        
    except Exception:
        return False

def tree_batch_dimension_size(batched_struct: Any) -> int:
    """
    Get the batch dimension size from a batched PyTree.
    
    Args:
        batched_struct: Batched PyTree
        
    Returns:
        Batch dimension size
        
    Raises:
        ValueError: If structure is invalid or inconsistent
    """
    leaves = tree_util.tree_leaves(batched_struct)
    
    if not leaves:
        raise ValueError("Empty PyTree has no batch dimension")
        
    first_leaf = leaves[0]
    if first_leaf.ndim == 0:
        raise ValueError("Scalar leaf has no batch dimension")
        
    batch_size = first_leaf.shape[0]
    
    # Validate consistency
    for leaf in leaves[1:]:
        if leaf.shape[0] != batch_size:
            raise ValueError(f"Inconsistent batch sizes: {batch_size} vs {leaf.shape[0]}")
            
    return batch_size

def safe_pytree_stack(objects_list: List[T]) -> T:
    """
    具有全面错误处理的超安全PyTree堆叠。
    
    此函数提供了将个别结构化对象转换为批处理格式
    以供JAX操作的最稳健方式。
    
    参数:
        objects_list: PyTree对象列表
        
    返回值:
        安全批处理的PyTree
        
    异常:
        ValueError: 带有详细错误信息用于调试
    """
    if not objects_list:
        raise ValueError("Cannot stack empty list of objects")
        
    if len(objects_list) == 1:
        # 单个对象 - 添加批处理维度
        return tree_util.tree_map(
            lambda x: jnp.expand_dims(x, axis=0),
            objects_list[0]
        )
    
    # 多个对象 - 验证并堆叠
    first_obj = objects_list[0]
    first_structure = tree_util.tree_structure(first_obj)
    
    for i, obj in enumerate(objects_list[1:], 1):
        obj_structure = tree_util.tree_structure(obj)
        if obj_structure != first_structure:
            raise ValueError(
                f"Object {i} has incompatible PyTree structure. "
                f"Expected: {first_structure}, Got: {obj_structure}"
            )
    
    # 执行堆叠并提供详细的错误报告
    try:
        return tree_util.tree_map(
            lambda *args: jnp.stack(args, axis=0),
            *objects_list
        )
    except Exception as e:
        # 提供详细的调试信息
        first_leaves = tree_util.tree_leaves(first_obj)
        error_info = []
        
        for i, leaf in enumerate(first_leaves):
            error_info.append(f"Leaf {i}: shape={leaf.shape}, dtype={leaf.dtype}")
            
        raise ValueError(
            f"Failed to stack PyTree objects. Original error: {e}\n"
            f"First object leaf info: {error_info}"
        )

# 常用用例的专用函数

def batch_drone_states(states: List['DroneState']) -> 'DroneState':
    """用于批处理DroneState对象的专用函数。"""
    return batch_pytree_objects(states)

def unbatch_drone_states(batched_states: 'DroneState') -> List['DroneState']:
    """用于解批处理DroneState对象的专用函数。"""
    return unbatch_pytree_objects(batched_states)

# 使所有函数可供导入
__all__ = [
    'batch_pytree_objects',
    'unbatch_pytree_objects', 
    'stack_pytree_list',
    'validate_batch_structure',
    'tree_batch_dimension_size',
    'safe_pytree_stack',
    'batch_drone_states',
    'unbatch_drone_states'
]