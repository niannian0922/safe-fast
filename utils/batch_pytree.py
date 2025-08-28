"""
JAX PyTree Batch Processing Utilities

This module provides utilities to handle the fundamental "Array of Structs vs Struct of Arrays"
problem in JAX. It enables seamless conversion between individual structured objects 
(like DroneState) and batch-compatible JAX arrays.

Key Functions:
- batch_pytree_objects: Convert [struct1, struct2, ...] -> struct_of_arrays
- unbatch_pytree_objects: Convert struct_of_arrays -> [struct1, struct2, ...]
- stack_pytree_list: Safe stacking of PyTree objects
- validate_batch_structure: Validate batch consistency

This solves the core issue where JAX operations like jnp.stack cannot handle
Flax @struct.dataclass objects directly.
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
    Convert a list of PyTree objects to a batched PyTree.
    
    Transforms [struct1, struct2, ...] -> struct_of_arrays
    This is the core solution to JAX's "Array of Structs" limitation.
    
    Args:
        objects_list: List of PyTree objects (e.g., DroneState instances)
        
    Returns:
        Batched PyTree where each leaf is stacked along batch dimension
        
    Example:
        states = [DroneState(...), DroneState(...), DroneState(...)]
        batched_states = batch_pytree_objects(states)
        # Result: DroneState with position.shape = [3, 3] instead of [3]
    """
    if not objects_list:
        raise ValueError("Cannot batch empty list")
        
    # Validate all objects have the same structure
    first_obj = objects_list[0]
    for i, obj in enumerate(objects_list[1:], 1):
        if not tree_util.tree_structure(obj) == tree_util.tree_structure(first_obj):
            raise ValueError(f"Object {i} has different structure than object 0")
    
    # Use tree_map with stacking to create batch dimension
    return tree_util.tree_map(
        lambda *args: jnp.stack(args, axis=0), 
        *objects_list
    )

def unbatch_pytree_objects(batched_struct: T) -> List[T]:
    """
    Convert a batched PyTree back to a list of individual PyTrees.
    
    Transforms struct_of_arrays -> [struct1, struct2, ...]
    
    Args:
        batched_struct: PyTree with batched arrays as leaves
        
    Returns:
        List of individual PyTree objects
        
    Example:
        batched_states = DroneState(position=jnp.array([[1,2,3], [4,5,6]]), ...)
        states = unbatch_pytree_objects(batched_states)
        # Result: [DroneState(position=[1,2,3]), DroneState(position=[4,5,6])]
    """
    # Get batch size from first leaf
    first_leaf = tree_util.tree_leaves(batched_struct)[0]
    if first_leaf.ndim == 0:
        raise ValueError("Cannot unbatch scalar (0-dimensional) arrays")
        
    batch_size = first_leaf.shape[0]
    
    # Extract each batch element
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
    Safe stacking of PyTree objects with custom axis.
    
    Args:
        objects_list: List of PyTree objects to stack
        axis: Axis along which to stack (default: 0 for batch dimension)
        
    Returns:
        Stacked PyTree
    """
    if not objects_list:
        raise ValueError("Cannot stack empty list")
        
    return tree_util.tree_map(
        lambda *args: jnp.stack(args, axis=axis),
        *objects_list
    )

def validate_batch_structure(batched_struct: Any, expected_batch_size: int) -> bool:
    """
    Validate that a batched PyTree has consistent structure.
    
    Args:
        batched_struct: Batched PyTree to validate
        expected_batch_size: Expected size of batch dimension
        
    Returns:
        True if structure is valid, False otherwise
    """
    try:
        leaves = tree_util.tree_leaves(batched_struct)
        
        if not leaves:
            return False
            
        # Check all leaves have correct batch dimension
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
    Ultra-safe PyTree stacking with comprehensive error handling.
    
    This function provides the most robust way to convert individual 
    structured objects to batch format for JAX operations.
    
    Args:
        objects_list: List of PyTree objects
        
    Returns:
        Safely batched PyTree
        
    Raises:
        ValueError: With detailed error messages for debugging
    """
    if not objects_list:
        raise ValueError("Cannot stack empty list of objects")
        
    if len(objects_list) == 1:
        # Single object - add batch dimension
        return tree_util.tree_map(
            lambda x: jnp.expand_dims(x, axis=0),
            objects_list[0]
        )
    
    # Multiple objects - validate and stack
    first_obj = objects_list[0]
    first_structure = tree_util.tree_structure(first_obj)
    
    for i, obj in enumerate(objects_list[1:], 1):
        obj_structure = tree_util.tree_structure(obj)
        if obj_structure != first_structure:
            raise ValueError(
                f"Object {i} has incompatible PyTree structure. "
                f"Expected: {first_structure}, Got: {obj_structure}"
            )
    
    # Perform stacking with detailed error reporting
    try:
        return tree_util.tree_map(
            lambda *args: jnp.stack(args, axis=0),
            *objects_list
        )
    except Exception as e:
        # Provide detailed debugging information
        first_leaves = tree_util.tree_leaves(first_obj)
        error_info = []
        
        for i, leaf in enumerate(first_leaves):
            error_info.append(f"Leaf {i}: shape={leaf.shape}, dtype={leaf.dtype}")
            
        raise ValueError(
            f"Failed to stack PyTree objects. Original error: {e}\n"
            f"First object leaf info: {error_info}"
        )

# Specialized functions for common use cases

def batch_drone_states(states: List['DroneState']) -> 'DroneState':
    """Specialized function for batching DroneState objects."""
    return batch_pytree_objects(states)

def unbatch_drone_states(batched_states: 'DroneState') -> List['DroneState']:
    """Specialized function for unbatching DroneState objects."""
    return unbatch_pytree_objects(batched_states)

# Make all functions available for import
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