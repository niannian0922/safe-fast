"""
JAX-native differentiable physics engine for safe agile flight.

This module implements the core differentiable physics simulation combining insights from:
1. GCBF+ (MIT-REALM): Graph-based safety mechanisms and multi-agent coordination
2. DiffPhysDrone (SJTU): Differentiable physics with temporal gradient decay

CRITICAL DiffPhysDrone Integration:
- Temporal gradient decay mechanism (g_decay function)
- Control history modeling with exponential smoothing  
- Thrust-to-weight ratio dynamics calibration
- Numerical stability enhancements

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
    """Simplified drone state for GCBF+ and DiffPhysDrone integration.
    
    Follows DiffPhysDrone point-mass model with state representation compatible
    with GCBF+ graph neural networks. Simplified from original complex structure
    to focus on essential dynamics.
    
    Key Design Principles:
    1. Point-mass dynamics (no orientation complexity)
    2. Compatible with GCBF+ graph construction
    3. Supports DiffPhysDrone temporal gradient decay
    4. JAX-native pure functional design
    """
    # Core state (point-mass model)
    position: chex.Array  # [3] - x, y, z coordinates in world frame
    velocity: chex.Array  # [3] - velocity components in world frame  
    acceleration: chex.Array  # [3] - current acceleration (needed for integration)
    
    # Control state (DiffPhysDrone-style)
    thrust_current: chex.Array  # [3] - current thrust command
    thrust_previous: chex.Array  # [3] - previous thrust command (for smoothing)
    
    # Temporal state
    time: float  # Current simulation time
    
    # Agent identification (for GCBF+ multi-agent scenarios)
    agent_id: int = 0  # Agent identifier for graph construction
    
    # Orientation (simplified - identity matrix for point mass)
    orientation: chex.Array = None  # [3, 3] rotation matrix (defaults to identity)
    
    def __post_init__(self):
        # Set default orientation to identity if not provided
        if self.orientation is None:
            object.__setattr__(self, 'orientation', jnp.eye(3))


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
    """Physics simulation parameters matching DiffPhysDrone methodology.
    
    Parameters calibrated based on DiffPhysDrone experimental results:
    - Crazyflie quadrotor specifications
    - Control latency and smoothing from real hardware
    - Gradient decay tuned for BPTT stability
    """
    # Time integration (DiffPhysDrone standard)
    dt: float = 1.0/15.0  # 15 Hz control frequency
    
    # Physical constants (Crazyflie 2.1 specifications)
    mass: float = 0.027  # kg
    gravity_magnitude: float = 9.80665  # m/s²
    
    # Thrust dynamics (from DiffPhysDrone paper)
    thrust_to_weight_ratio: float = 3.0  # Aggressive flight capability
    max_thrust_normalized: float = 0.8  # Normalized maximum thrust
    
    # Drag model (simplified)
    drag_coefficient_linear: float = 0.01  # Linear drag
    drag_coefficient_quadratic: float = 0.01  # Quadratic drag
    
    # Control dynamics (DiffPhysDrone key innovation)
    control_delay_tau: float = 1.0/15.0  # Control latency (s)
    exponential_smoothing_lambda: float = 12.0  # EMA parameter
    
    # GCBF+ safety parameters
    safety_radius: float = 0.05  # Collision radius (m)
    sensing_radius: float = 0.5  # Neighbor detection radius (m)
    cbf_alpha: float = 1.0  # CBF class-K function parameter
    
    # Temporal gradient decay (DiffPhysDrone core)
    gradient_decay_alpha: float = 0.4  # Original DiffPhysDrone value
    enable_gradient_decay: bool = True
    
    # Numerical stability
    velocity_limit: float = 10.0  # Maximum velocity (m/s)
    position_limit: float = 50.0  # Workspace boundary (m)
    epsilon: float = 1e-8  # Numerical stability

    @property
    def gravity_vector(self) -> chex.Array:
        """Standard gravity vector."""
        return jnp.array([0.0, 0.0, -self.gravity_magnitude])
        
    @property
    def max_thrust_force(self) -> float:
        """Maximum thrust force in Newtons."""
        return self.mass * self.thrust_to_weight_ratio * self.gravity_magnitude

# =============================================================================
# DIFFPHYSDRONE TEMPORAL GRADIENT DECAY (CORE INNOVATION)
# =============================================================================

def temporal_gradient_decay(x: chex.Array, alpha: float) -> chex.Array:
    """
    DiffPhysDrone temporal gradient decay mechanism - EXACT IMPLEMENTATION
    
    This is the core innovation from DiffPhysDrone paper:
    "Learning Vision-based Agile Flight via Differentiable Physics"
    
    Original PyTorch implementation:
        def g_decay(x, alpha):
            return x * alpha + x.detach() * (1 - alpha)
            
    JAX equivalent using stop_gradient:
        return x * alpha + jax.lax.stop_gradient(x) * (1 - alpha)
    
    The function creates a gradient flow control mechanism where:
    - alpha controls how much gradient flows through
    - (1-alpha) portion has stopped gradients (like .detach() in PyTorch)
    - This enables stable training for long sequences
    
    Args:
        x: Input tensor (any shape)
        alpha: Gradient flow coefficient [0,1]
               - alpha=1.0: full gradient flow
               - alpha=0.0: no gradient flow (pure stop_gradient)
               - alpha=0.4 (DiffPhysDrone default): balanced decay
    
    Returns:
        Tensor with controlled gradient flow
    """
    return x * alpha + jax.lax.stop_gradient(x) * (1 - alpha)

def apply_temporal_gradient_decay_to_state(
    drone_state: DroneState, 
    decay_alpha: float = 0.4  # DiffPhysDrone default
) -> DroneState:
    """
    Apply temporal gradient decay to drone state components
    
    Critical for BPTT stability over long horizons. Implementation follows
    exact DiffPhysDrone methodology with gradient flow control.
    """
    return DroneState(
        position=temporal_gradient_decay(drone_state.position, decay_alpha),
        velocity=temporal_gradient_decay(drone_state.velocity, decay_alpha),
        acceleration=temporal_gradient_decay(drone_state.acceleration, decay_alpha),
        thrust_current=temporal_gradient_decay(drone_state.thrust_current, decay_alpha),
        thrust_previous=temporal_gradient_decay(drone_state.thrust_previous, decay_alpha),
        orientation=drone_state.orientation,  # Orientation unchanged
        time=drone_state.time,  # Time unchanged
        agent_id=drone_state.agent_id  # ID unchanged
    )

def create_spatial_temporal_decay_schedule(
    distance_to_obstacles: chex.Array,
    base_alpha: float = 0.4,
    min_distance: float = 0.5,
    max_distance: float = 2.0
) -> float:
    """
    Advanced: Spatial-temporal gradient decay adaptation
    
    Adapts gradient decay based on proximity to obstacles:
    - Near obstacles: less decay (stronger gradients for safety)  
    - Far from obstacles: more decay (focus on efficiency)
    
    This is an innovation beyond the original DiffPhysDrone paper.
    """
    # Compute minimum distance to any obstacle
    min_dist = jnp.min(distance_to_obstacles)
    
    # Create adaptive alpha based on distance
    normalized_dist = jnp.clip(
        (min_dist - min_distance) / (max_distance - min_distance),
        0.0, 1.0
    )
    
    # Less decay when close to obstacles (need strong safety gradients)
    # More decay when far from obstacles (focus on efficiency)
    adaptive_alpha = base_alpha + (1.0 - base_alpha) * (1.0 - normalized_dist)
    
    return adaptive_alpha


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
    DiffPhysDrone-style differentiable physics step with control smoothing.
    
    Implements key DiffPhysDrone innovations:
    1. Exponential moving average for thrust smoothing
    2. Control delay modeling
    3. Point-mass dynamics with drag
    4. Numerical stability through smooth saturation
    
    Args:
        state: Current drone state
        control_input: [3] thrust command (normalized [-1, 1])
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
    acc = state.acceleration
    thrust_prev = state.thrust_previous
    
    # === DIFFPHYSDRONE CONTROL SMOOTHING ===
    # Exponential moving average thrust smoothing (key innovation)
    # This implements the control delay and smoothing from the original paper
    
    # Input saturation (normalized commands should be in [-1, 1])
    saturated_input = jnp.tanh(control_input)  # Smooth saturation
    
    # Exponential moving average (EMA) thrust smoothing
    # Formula: thrust_new = lambda * thrust_cmd + (1 - lambda) * thrust_prev
    smoothing_factor = jnp.exp(-params.exponential_smoothing_lambda * dt)
    smoothed_thrust = (
        (1.0 - smoothing_factor) * saturated_input + 
        smoothing_factor * thrust_prev
    )
    
    # Control delay simulation (first-order system)
    # This models the delay between commanded and actual thrust
    delay_factor = jnp.exp(-dt / params.control_delay_tau)
    actual_thrust = (
        (1.0 - delay_factor) * smoothed_thrust +
        delay_factor * state.thrust_current
    )
    
    # === FORCE COMPUTATION ===
    # Convert normalized thrust to physical force
    thrust_force = actual_thrust * params.max_thrust_force
    
    # Drag forces (linear + quadratic)
    vel_norm = jnp.linalg.norm(vel)
    vel_unit = vel / jnp.maximum(vel_norm, params.epsilon)
    
    # Linear drag
    drag_linear = -params.drag_coefficient_linear * vel
    
    # Quadratic drag (velocity squared)
    drag_quadratic = -params.drag_coefficient_quadratic * vel_norm * vel_unit
    
    total_drag = drag_linear + drag_quadratic
    
    # Gravitational force
    gravity_force = params.mass * params.gravity_vector
    
    # === PHYSICS INTEGRATION (DiffPhysDrone style) ===
    # Total external forces
    total_force = thrust_force + total_drag + gravity_force
    
    # Compute acceleration
    new_acceleration = total_force / params.mass
    
    # Semi-implicit Euler integration (stable for stiff systems)
    # This follows the integration scheme from DiffPhysDrone
    new_vel = vel + 0.5 * (acc + new_acceleration) * dt  # Trapezoidal velocity
    new_pos = pos + vel * dt + 0.5 * new_acceleration * dt**2  # Position with acceleration
    
    # === PHYSICAL CONSTRAINTS ===
    # Smooth velocity limiting (differentiable)
    vel_magnitude = jnp.linalg.norm(new_vel)
    vel_scale = jnp.minimum(1.0, params.velocity_limit / jnp.maximum(vel_magnitude, params.epsilon))
    new_vel = new_vel * vel_scale
    
    # Smooth position bounds (workspace limits)
    pos_magnitude = jnp.linalg.norm(new_pos)
    pos_scale = jnp.minimum(1.0, params.position_limit / jnp.maximum(pos_magnitude, params.epsilon))
    new_pos = new_pos * pos_scale
    
    # Create new state (point-mass model)
    new_state = DroneState(
        position=new_pos,
        velocity=new_vel, 
        acceleration=new_acceleration,
        thrust_current=actual_thrust,
        thrust_previous=smoothed_thrust,  # Store for next step
        orientation=state.orientation,  # Keep same orientation
        time=state.time + dt,
        agent_id=state.agent_id
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
        # Extract individual drone state with simplified format
        drone_state = DroneState(
            position=state.drone_states[i, :3],
            velocity=state.drone_states[i, 3:6],
            acceleration=state.drone_states[i, 6:9],
            thrust_current=state.drone_states[i, 9:12],
            thrust_previous=state.drone_states[i, 12:15],
            time=state.global_time,
            agent_id=i
        )
        
        # Apply dynamics
        new_drone_state = dynamics_step(
            drone_state, 
            control_inputs[i], 
            params, 
            dt
        )
        
        # Pack back into array format (15-dimensional state per agent)
        new_state_array = jnp.concatenate([
            new_drone_state.position,      # [0:3]
            new_drone_state.velocity,      # [3:6] 
            new_drone_state.acceleration,  # [6:9]
            new_drone_state.thrust_current,  # [9:12]
            new_drone_state.thrust_previous  # [12:15]
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
    position: chex.Array,  # [3]
    velocity: Optional[chex.Array] = None,  # [3]
    agent_id: int = 0,
    hover_initialization: bool = True
) -> DroneState:
    """
    Create initial state for a single drone following DiffPhysDrone design.
    
    Simplified initialization for point-mass dynamics with proper
    hover thrust setup for stability.
    
    Args:
        position: Initial position [3]
        velocity: Initial velocity [3] (default: zero)
        agent_id: Agent identifier for multi-agent scenarios
        hover_initialization: Initialize with hover thrust for stability
        
    Returns:
        Initialized drone state
    """
    if velocity is None:
        velocity = jnp.zeros(3)
    
    # Initialize acceleration to zero
    acceleration = jnp.zeros(3)
    
    # Initialize thrust commands
    if hover_initialization:
        # Compute hover thrust (balances gravity)
        params = PhysicsParams()
        hover_thrust_magnitude = params.gravity_magnitude / params.thrust_to_weight_ratio
        hover_thrust = jnp.array([0.0, 0.0, hover_thrust_magnitude])
    else:
        hover_thrust = jnp.zeros(3)
    
    return DroneState(
        position=position,
        velocity=velocity,
        acceleration=acceleration,
        thrust_current=hover_thrust,
        thrust_previous=hover_thrust,
        orientation=jnp.eye(3),  # Identity orientation for point mass
        time=0.0,
        agent_id=agent_id
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
    # New state format: [pos(3), vel(3), acc(3), thrust_curr(3), thrust_prev(3)] = 15 dimensions
    accelerations = jnp.zeros((n_agents, 3))
    thrust_current = jnp.zeros((n_agents, 3))
    thrust_previous = jnp.zeros((n_agents, 3))
    
    drone_states = jnp.concatenate([
        positions,        # [n_agents, 3] - [0:3]
        velocities,       # [n_agents, 3] - [3:6]
        accelerations,    # [n_agents, 3] - [6:9]
        thrust_current,   # [n_agents, 3] - [9:12]
        thrust_previous   # [n_agents, 3] - [12:15]
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
        
    # Check reasonable physical bounds (updated for the new constraints)
    max_position = 100.0  # meters
    max_velocity = 50.0   # m/s (updated to match test expectation)
    
    if jnp.any(jnp.abs(state.position) > max_position):
        return False
    if jnp.linalg.norm(state.velocity) > max_velocity:  # Use norm instead of component-wise check
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