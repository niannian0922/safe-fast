"""
JAX-native differentiable physics engine for safe agile flight.

This module implements the core differentiable physics simulation combining insights from:
1. GCBF+ (MIT-REALM): Graph-based safety mechanisms and multi-agent coordination
2. DiffPhysDrone (SJTU): Differentiable physics with temporal gradient decay

The physics engine is designed to be:
- Pure functional (no side effects)  
- JIT-compilable with JAX
- End-to-end differentiable
- Compatible with BPTT (Backpropagation Through Time)
- Supports temporal gradient decay for training stability
"""

import jax
import jax.numpy as jnp
from jax import random
from typing import NamedTuple, Optional, Tuple, Any, Dict
import chex
from flax import struct


# =============================================================================
# PHYSICS STATE REPRESENTATIONS
# =============================================================================

@struct.dataclass 
class DroneState:
    """State representation for a single drone using point-mass dynamics.
    
    Based on DiffPhysDrone's simplified dynamics model, this represents
    each drone as a point mass with position, velocity, and control history.
    """
    position: chex.Array  # [3] - x, y, z coordinates in world frame
    velocity: chex.Array  # [3] - velocity components in world frame  
    thrust_history: chex.Array  # [n_history, 3] - control input history for smoothing
    mass: float  # Drone mass in kg
    time: float  # Current simulation time


@struct.dataclass
class MultiAgentState:
    """State representation for multi-agent scenarios.
    
    Combines individual drone states with global environment information
    needed for GCBF+ graph construction and safety verification.
    """
    drone_states: chex.Array  # [n_agents, state_dim] - individual drone states
    adjacency_matrix: chex.Array  # [n_agents, n_agents] - graph connectivity
    obstacle_positions: chex.Array  # [n_obstacles, 3] - static obstacle locations
    obstacle_radii: chex.Array  # [n_obstacles] - obstacle radii
    global_time: float  # Global simulation time


@struct.dataclass  
class PhysicsParams:
    """Physics simulation parameters.
    
    Encapsulates all physical constants and simulation parameters to ensure
    pure functional design and enable parameter optimization.
    """
    # Time integration
    dt: float = 1.0/15.0  # Simulation timestep
    gravity: chex.Array = jnp.array([0.0, 0.0, -9.81])  # Gravitational acceleration
    
    # Drone physical properties
    mass: float = 0.027  # kg (Crazyflie mass)
    drag_coefficient: float = 0.01  # Linear air drag coefficient
    max_thrust: float = 0.8  # Maximum normalized thrust
    thrust_to_weight: float = 3.0  # Thrust-to-weight ratio
    
    # Control dynamics (from DiffPhysDrone calibration)
    control_delay: float = 1.0/15.0  # Control latency tau
    control_smoothing: float = 12.0  # Exponential smoothing lambda
    
    # Safety parameters (GCBF+ integration)
    safety_radius: float = 0.05  # Collision avoidance radius
    sensing_radius: float = 0.5   # Neighbor detection radius
    
    # Temporal gradient decay (DiffPhysDrone innovation)
    gradient_decay_alpha: float = 0.92  # Gradient decay rate
    enable_gradient_decay: bool = True


# =============================================================================
# CORE PHYSICS FUNCTIONS
# =============================================================================

def dynamics_step(
    state: DroneState, 
    control_input: chex.Array,
    params: PhysicsParams,
    dt: Optional[float] = None
) -> DroneState:
    """
    Single timestep of differentiable drone dynamics.
    
    Implements point-mass dynamics with control delay and air drag based on
    DiffPhysDrone methodology. This function is pure and JIT-compilable.
    
    Args:
        state: Current drone state
        control_input: [3] thrust command in body frame (normalized)
        params: Physics parameters
        dt: Optional timestep override
        
    Returns:
        Updated drone state after one simulation step
    """
    if dt is None:
        dt = params.dt
        
    # Extract state components
    pos = state.position
    vel = state.velocity
    thrust_hist = state.thrust_history
    
    # === CONTROL PROCESSING (DiffPhysDrone approach) ===
    # Apply exponential moving average for control smoothing
    # This models the flight controller response with delay
    alpha_smooth = jnp.exp(-params.control_smoothing * dt)
    
    # Update thrust history with new command
    new_thrust_hist = jnp.roll(thrust_hist, shift=1, axis=0)
    new_thrust_hist = new_thrust_hist.at[0].set(control_input)
    
    # Compute filtered thrust using exponential smoothing
    weights = jnp.power(alpha_smooth, jnp.arange(thrust_hist.shape[0]))
    weights = weights / jnp.sum(weights)  # Normalize
    filtered_thrust = jnp.sum(new_thrust_hist * weights[:, None], axis=0)
    
    # === FORCE COMPUTATION ===
    # Convert normalized thrust to force (body frame to world frame)
    max_force = params.mass * params.thrust_to_weight * 9.81
    thrust_force = filtered_thrust * max_force
    
    # Air drag force (opposing velocity)
    drag_force = -params.drag_coefficient * jnp.linalg.norm(vel) * vel
    
    # Total external force
    total_force = thrust_force + drag_force + params.mass * params.gravity
    
    # === NUMERICAL INTEGRATION ===
    # Using Euler integration (can be upgraded to RK4 for better accuracy)
    acceleration = total_force / params.mass
    
    # Update velocity and position
    new_vel = vel + acceleration * dt
    new_pos = pos + vel * dt + 0.5 * acceleration * dt**2  # Semi-implicit Euler
    
    # === STATE UPDATE ===
    new_state = DroneState(
        position=new_pos,
        velocity=new_vel, 
        thrust_history=new_thrust_hist,
        mass=state.mass,
        time=state.time + dt
    )
    
    return new_state


def multi_agent_dynamics_step(
    state: MultiAgentState,
    control_inputs: chex.Array,  # [n_agents, 3]
    params: PhysicsParams,
    dt: Optional[float] = None
) -> MultiAgentState:
    """
    Multi-agent dynamics step with graph structure updates.
    
    Processes all agents simultaneously while maintaining graph connectivity
    information needed for GCBF+ safety verification.
    
    Args:
        state: Multi-agent system state
        control_inputs: Control commands for all agents [n_agents, 3]
        params: Physics parameters
        dt: Optional timestep override
        
    Returns:
        Updated multi-agent state
    """
    if dt is None:
        dt = params.dt
        
    n_agents = state.drone_states.shape[0]
    
    # === INDIVIDUAL DYNAMICS ===
    # Process each agent's dynamics (vectorized over agents)
    def single_agent_update(i: int) -> chex.Array:
        # Extract individual drone state
        drone_state = DroneState(
            position=state.drone_states[i, :3],
            velocity=state.drone_states[i, 3:6], 
            thrust_history=state.drone_states[i, 6:15].reshape(3, 3),
            mass=params.mass,
            time=state.global_time
        )
        
        # Apply dynamics
        new_drone_state = dynamics_step(
            drone_state, 
            control_inputs[i], 
            params, 
            dt
        )
        
        # Pack back into array format
        new_state_array = jnp.concatenate([
            new_drone_state.position,
            new_drone_state.velocity,
            new_drone_state.thrust_history.flatten()
        ])
        
        return new_state_array
    
    # Vectorized application across all agents
    new_drone_states = jax.vmap(single_agent_update)(jnp.arange(n_agents))
    
    # === GRAPH CONNECTIVITY UPDATE (GCBF+ integration) ===
    # Recompute adjacency matrix based on sensing radius
    positions = new_drone_states[:, :3]  # Extract positions
    
    # Compute pairwise distances
    diff = positions[:, None, :] - positions[None, :, :]  # [n, n, 3]
    distances = jnp.linalg.norm(diff, axis=-1)  # [n, n]
    
    # Create adjacency matrix (within sensing radius)
    new_adjacency = (distances < params.sensing_radius) & (distances > 0)
    
    # === COLLISION DETECTION ===
    # Check for inter-agent collisions
    collision_matrix = distances < (2 * params.safety_radius)
    collision_detected = jnp.any(collision_matrix & ~jnp.eye(n_agents, dtype=bool))
    
    # Check for obstacle collisions
    obstacle_collisions = check_obstacle_collisions(
        positions, 
        state.obstacle_positions, 
        state.obstacle_radii, 
        params.safety_radius
    )
    
    # === STATE ASSEMBLY ===
    new_state = MultiAgentState(
        drone_states=new_drone_states,
        adjacency_matrix=new_adjacency,
        obstacle_positions=state.obstacle_positions,
        obstacle_radii=state.obstacle_radii,
        global_time=state.global_time + dt
    )
    
    return new_state


def check_obstacle_collisions(
    positions: chex.Array,      # [n_agents, 3]
    obstacle_positions: chex.Array,  # [n_obstacles, 3] 
    obstacle_radii: chex.Array,      # [n_obstacles]
    safety_radius: float
) -> chex.Array:
    """
    Check for collisions between agents and static obstacles.
    
    Args:
        positions: Agent positions
        obstacle_positions: Obstacle center positions
        obstacle_radii: Obstacle radii
        safety_radius: Agent safety radius
        
    Returns:
        [n_agents] boolean array indicating collision status
    """
    # Compute distances from each agent to each obstacle
    agent_obs_diff = positions[:, None, :] - obstacle_positions[None, :, :]
    agent_obs_distances = jnp.linalg.norm(agent_obs_diff, axis=-1)
    
    # Check collision condition (agent safety radius + obstacle radius)
    collision_distances = safety_radius + obstacle_radii[None, :]
    collisions = agent_obs_distances < collision_distances
    
    # Any collision per agent
    agent_collisions = jnp.any(collisions, axis=1)
    
    return agent_collisions


# =============================================================================
# TEMPORAL GRADIENT DECAY (DiffPhysDrone Innovation)
# =============================================================================

def apply_temporal_gradient_decay(
    gradient: chex.Array,
    time_step: int,
    alpha: float = 0.92,
    dt: float = 1.0/15.0
) -> chex.Array:
    """
    Apply temporal gradient decay to mitigate gradient explosion in BPTT.
    
    This implements the key innovation from DiffPhysDrone for stable training
    of long-horizon differentiable physics simulations. The decay mechanism
    ensures that gradients from distant future states don't overwhelm
    near-term supervision signals.
    
    Args:
        gradient: Gradient tensor to be decayed
        time_step: Current timestep in the BPTT sequence  
        alpha: Decay rate parameter (0 < alpha < 1)
        dt: Simulation timestep
        
    Returns:
        Decayed gradient with exponential time weighting
    """
    decay_factor = jnp.power(alpha, time_step * dt)
    return gradient * decay_factor


def create_temporal_decay_schedule(
    sequence_length: int,
    alpha: float = 0.92,
    dt: float = 1.0/15.0
) -> chex.Array:
    """
    Create a temporal decay schedule for an entire BPTT sequence.
    
    Args:
        sequence_length: Length of the BPTT sequence
        alpha: Decay rate parameter
        dt: Simulation timestep
        
    Returns:
        [sequence_length] array of decay factors
    """
    time_steps = jnp.arange(sequence_length)
    decay_factors = jnp.power(alpha, time_steps * dt)
    return decay_factors


# =============================================================================
# PHYSICS ENGINE INITIALIZATION AND UTILITIES  
# =============================================================================

def create_initial_drone_state(
    position: chex.Array,
    velocity: Optional[chex.Array] = None,
    mass: float = 0.027,
    thrust_history_length: int = 3
) -> DroneState:
    """
    Create initial state for a single drone.
    
    Args:
        position: Initial position [3]
        velocity: Initial velocity [3] (default: zero)
        mass: Drone mass in kg
        thrust_history_length: Length of thrust history buffer
        
    Returns:
        Initialized drone state
    """
    if velocity is None:
        velocity = jnp.zeros(3)
        
    # Initialize thrust history with zeros
    thrust_history = jnp.zeros((thrust_history_length, 3))
    
    return DroneState(
        position=position,
        velocity=velocity,
        thrust_history=thrust_history,
        mass=mass,
        time=0.0
    )


def create_initial_multi_agent_state(
    positions: chex.Array,      # [n_agents, 3]
    velocities: Optional[chex.Array] = None,  # [n_agents, 3]
    obstacle_positions: Optional[chex.Array] = None,  # [n_obstacles, 3]
    obstacle_radii: Optional[chex.Array] = None,      # [n_obstacles]
    params: PhysicsParams = PhysicsParams()
) -> MultiAgentState:
    """
    Create initial state for multi-agent system.
    
    Args:
        positions: Initial agent positions [n_agents, 3]
        velocities: Initial agent velocities [n_agents, 3]
        obstacle_positions: Static obstacle positions [n_obstacles, 3]
        obstacle_radii: Static obstacle radii [n_obstacles]
        params: Physics parameters
        
    Returns:
        Initialized multi-agent state
    """
    n_agents = positions.shape[0]
    
    if velocities is None:
        velocities = jnp.zeros((n_agents, 3))
        
    if obstacle_positions is None:
        obstacle_positions = jnp.zeros((0, 3))
        
    if obstacle_radii is None:
        obstacle_radii = jnp.zeros(0)
        
    # Create drone state array [n_agents, state_dim]
    # State format: [pos(3), vel(3), thrust_hist(9)] = 15 dimensions
    thrust_histories = jnp.zeros((n_agents, 3, 3))  # 3 history steps, 3D thrust
    
    drone_states = jnp.concatenate([
        positions,                            # [n_agents, 3]
        velocities,                          # [n_agents, 3] 
        thrust_histories.reshape(n_agents, 9) # [n_agents, 9]
    ], axis=1)
    
    # Initialize adjacency matrix
    distances = jnp.linalg.norm(
        positions[:, None, :] - positions[None, :, :], 
        axis=-1
    )
    adjacency_matrix = (distances < params.sensing_radius) & (distances > 0)
    
    return MultiAgentState(
        drone_states=drone_states,
        adjacency_matrix=adjacency_matrix,
        obstacle_positions=obstacle_positions,
        obstacle_radii=obstacle_radii,
        global_time=0.0
    )


def validate_physics_state(state: DroneState) -> bool:
    """
    Validate drone state for numerical stability and physical constraints.
    
    Args:
        state: Drone state to validate
        
    Returns:
        True if state is valid, False otherwise
    """
    # Check for NaN or infinite values
    if not jnp.all(jnp.isfinite(state.position)):
        return False
    if not jnp.all(jnp.isfinite(state.velocity)):
        return False
    if not jnp.all(jnp.isfinite(state.thrust_history)):
        return False
        
    # Check reasonable physical bounds
    max_position = 100.0  # meters
    max_velocity = 50.0   # m/s
    
    if jnp.any(jnp.abs(state.position) > max_position):
        return False
    if jnp.any(jnp.abs(state.velocity) > max_velocity):
        return False
        
    return True


# =============================================================================
# JIT-COMPILED VERSIONS FOR PERFORMANCE
# =============================================================================

# JIT compile the core dynamics functions for maximum performance
dynamics_step_jit = jax.jit(dynamics_step)
multi_agent_dynamics_step_jit = jax.jit(multi_agent_dynamics_step)
check_obstacle_collisions_jit = jax.jit(check_obstacle_collisions)

# Make functions available for import
__all__ = [
    'DroneState',
    'MultiAgentState', 
    'PhysicsParams',
    'dynamics_step',
    'multi_agent_dynamics_step',
    'dynamics_step_jit',
    'multi_agent_dynamics_step_jit',
    'apply_temporal_gradient_decay',
    'create_temporal_decay_schedule', 
    'create_initial_drone_state',
    'create_initial_multi_agent_state',
    'validate_physics_state',
    'check_obstacle_collisions',
    'check_obstacle_collisions_jit'
]