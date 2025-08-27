"""
Complete Safety Layer Implementation for Safe Agile Flight System

This module implements the differentiable safety layer combining CBF constraints
with quadratic programming (QP) for safe control synthesis.

Key Components:
1. QP-based safety filter using qpax for JAX-native differentiability
2. Complete three-layer safety fallback mechanism for robustness
3. CBF constraint construction and gradient integration
4. Temporal gradient decay from DiffPhysDrone integration

Safety Hierarchy:
1. Nominal QP solving (preferred)
2. Relaxed QP with slack variables (fallback)  
3. Emergency brake control (last resort)
"""

import jax
import jax.numpy as jnp
from jax import jit, grad
import qpax
from typing import Tuple, Optional, NamedTuple
import chex
from dataclasses import dataclass

from .perception import DroneState

# =============================================================================
# SAFETY CONFIGURATION AND STRUCTURES  
# =============================================================================

@dataclass
class SafetyConfig:
    """Configuration parameters for safety layer"""
    # QP solver settings
    max_iterations: int = 100
    tolerance: float = 1e-6
    regularization: float = 1e-8
    
    # Control constraints
    max_thrust: float = 0.8  # Maximum thrust magnitude
    max_torque: float = 0.5  # Maximum angular control
    
    # CBF parameters
    cbf_alpha: float = 1.0  # CBF class-K function parameter
    safety_margin: float = 0.1  # Additional safety buffer
    relaxation_penalty: float = 1000.0  # Slack variable penalty (beta)
    
    # Emergency fallback
    emergency_brake_force: float = -0.6  # Emergency deceleration
    failure_penalty: float = 10000.0  # Loss penalty for failures

@dataclass
class QSolutionInfo:
    """QP solution information and diagnostics"""
    u_safe: chex.Array  # Computed safe control
    is_feasible: bool  # Whether QP was feasible
    solver_status: int  # Solver status code (0=success, 1=infeasible, 2=failed)
    slack_violation: float  # Magnitude of constraint relaxation
    num_iterations: int  # Solver iterations used
    
class SafetyLayer:
    """
    Differentiable safety layer with QP-based control synthesis
    
    Implements the complete three-layer safety architecture:
    1. Standard CBF-QP (nominal operation)
    2. Relaxed CBF-QP with slack variables (safety fallback)
    3. Emergency braking (last resort)
    """
    
    def __init__(self, config: SafetyConfig):
        self.config = config

    def _construct_qp_matrices(
        self, 
        u_nom: chex.Array,
        h: chex.Array, 
        grad_h: chex.Array,
        drone_state: DroneState
    ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
        """
        Construct QP matrices for CBF constraint
        
        Standard CBF-QP formulation:
        minimize: ||u - u_nom||^2
        subject to: grad_h^T * (f + g*u) >= -alpha * h
        """
        # Cost function: minimize ||u - u_nom||^2
        Q = 2.0 * jnp.eye(3)  # Factor of 2 for standard QP form
        q = -2.0 * u_nom
        
        # Simplified dynamics for point mass model:
        # f = [v_x, v_y, v_z - g]  (current velocity + gravity)
        # g = I (direct thrust control)
        f_dynamics = jnp.concatenate([
            drone_state.velocity,
            jnp.array([0.0, 0.0, -9.81])  # Gravity effect
        ])
        
        # Control input matrix (identity for direct thrust control)
        g_matrix = jnp.eye(3)
        
        # Constraint matrix: -grad_h^T * g
        G_cbf = -(grad_h @ g_matrix)[None, :]  # (1, 3)
        
        # Constraint vector: grad_h^T * f + alpha * h
        h_cbf = grad_h @ f_dynamics + self.config.cbf_alpha * h
        
        # Add control magnitude constraints: -max_thrust <= u_i <= max_thrust
        G_control = jnp.vstack([
            jnp.eye(3),      # u_i <= max_thrust
            -jnp.eye(3)      # -u_i <= max_thrust (i.e., u_i >= -max_thrust)
        ])
        h_control = jnp.full(6, self.config.max_thrust)
        
        # Combine constraints
        G = jnp.vstack([G_cbf, G_control])  # (7, 3)
        h_constraint = jnp.concatenate([h_cbf[None], h_control])  # (7,)
        
        return Q, q, G, h_constraint

    def _construct_relaxed_qp_matrices(
        self,
        u_nom: chex.Array,
        h: chex.Array,
        grad_h: chex.Array, 
        drone_state: DroneState
    ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
        """
        Construct relaxed QP with slack variables
        
        Relaxed formulation:
        minimize: ||u - u_nom||^2 + beta * ||delta||^2
        subject to: grad_h^T * (f + g*u) >= -alpha * h - delta
                   delta >= 0
        
        Variables: [u (3), delta (1)] = (4,)
        """
        beta = self.config.relaxation_penalty
        
        # Extended cost matrix for [u, delta]
        Q = jnp.block([
            [2.0 * jnp.eye(3), jnp.zeros((3, 1))],
            [jnp.zeros((1, 3)), 2.0 * beta * jnp.ones((1, 1))]
        ])  # (4, 4)
        
        q = jnp.concatenate([-2.0 * u_nom, jnp.zeros(1)])  # (4,)
        
        # Get original CBF constraint components
        _, _, G_orig, h_orig = self._construct_qp_matrices(u_nom, h, grad_h, drone_state)
        
        # Modify CBF constraint to include slack: G*[u, delta] <= h
        # CBF constraint: -grad_h^T * u - delta <= grad_h^T * f + alpha * h
        G_cbf_relaxed = jnp.concatenate([G_orig[0], jnp.array([-1.0])])[None, :]  # (1, 4)
        h_cbf_relaxed = h_orig[0:1]
        
        # Control constraints (only affect u, not delta)
        G_control_relaxed = jnp.hstack([G_orig[1:], jnp.zeros((6, 1))])  # (6, 4)
        h_control_relaxed = h_orig[1:]
        
        # Non-negativity constraint on slack: -delta <= 0
        G_slack = jnp.array([[0., 0., 0., -1.]])  # (1, 4)
        h_slack = jnp.array([0.])
        
        # Combine all constraints
        G = jnp.vstack([G_cbf_relaxed, G_control_relaxed, G_slack])  # (8, 4)
        h_constraint = jnp.concatenate([h_cbf_relaxed, h_control_relaxed, h_slack])  # (8,)
        
        return Q, q, G, h_constraint

    def _solve_standard_qp(
        self,
        u_nom: chex.Array,
        h: chex.Array,
        grad_h: chex.Array,
        drone_state: DroneState
    ) -> QSolutionInfo:
        """Solve standard CBF-QP (Layer 1)"""
        try:
            Q, q, G, h_constraint = self._construct_qp_matrices(u_nom, h, grad_h, drone_state)
            
            # Solve QP using qpax
            solution, info = qpax.solve_qp(
                Q=Q, q=q, 
                G=G, h=h_constraint,
                A=None, b=None,  # No equality constraints
                maxiter=self.config.max_iterations,
                tol=self.config.tolerance
            )
            
            # Check convergence and feasibility
            is_feasible = info.status == 0
            u_safe = solution if is_feasible else u_nom
            
            return QSolutionInfo(
                u_safe=u_safe,
                is_feasible=is_feasible,
                solver_status=info.status,
                slack_violation=0.0,
                num_iterations=info.niter
            )
            
        except Exception as e:
            # Solver failed - return nominal control
            return QSolutionInfo(
                u_safe=u_nom,
                is_feasible=False,
                solver_status=2,  # Failed
                slack_violation=0.0,
                num_iterations=0
            )

    def _solve_relaxed_qp(
        self,
        u_nom: chex.Array,
        h: chex.Array,
        grad_h: chex.Array,
        drone_state: DroneState
    ) -> QSolutionInfo:
        """Solve relaxed CBF-QP with slack variables (Layer 2)"""
        try:
            Q, q, G, h_constraint = self._construct_relaxed_qp_matrices(
                u_nom, h, grad_h, drone_state
            )
            
            # Solve relaxed QP
            solution, info = qpax.solve_qp(
                Q=Q, q=q,
                G=G, h=h_constraint,
                A=None, b=None,
                maxiter=self.config.max_iterations,
                tol=self.config.tolerance
            )
            
            is_feasible = info.status == 0
            
            if is_feasible:
                u_safe = solution[:3]  # Extract control part
                slack_violation = solution[3]  # Slack variable value
            else:
                u_safe = u_nom
                slack_violation = 0.0
            
            return QSolutionInfo(
                u_safe=u_safe,
                is_feasible=is_feasible,
                solver_status=info.status,
                slack_violation=slack_violation,
                num_iterations=info.niter
            )
            
        except Exception as e:
            # Relaxed solver failed - return nominal
            return QSolutionInfo(
                u_safe=u_nom,
                is_feasible=False,
                solver_status=2,
                slack_violation=0.0,
                num_iterations=0
            )

    def _emergency_brake(self, drone_state: DroneState) -> chex.Array:
        """Emergency braking control (Layer 3)"""
        velocity_magnitude = jnp.linalg.norm(drone_state.velocity)
        
        # Use jnp.where for JAX compatibility instead of if/else
        brake_direction = jnp.where(
            velocity_magnitude > 1e-6,
            -drone_state.velocity / (velocity_magnitude + 1e-8),  # Add small epsilon for stability
            jnp.array([0.0, 0.0, -1.0])  # Default downward direction
        )
        
        emergency_control = jnp.where(
            velocity_magnitude > 1e-6,
            self.config.emergency_brake_force * brake_direction,
            jnp.array([0.0, 0.0, -0.1])  # Gentle downward for hovering
        )
        
        emergency_control = jnp.clip(
            emergency_control, -self.config.max_thrust, self.config.max_thrust
        )
        
        return emergency_control

    def safety_filter(
        self,
        u_nom: chex.Array,
        h: chex.Array,
        grad_h: chex.Array,
        drone_state: DroneState
    ) -> Tuple[chex.Array, QSolutionInfo]:
        """
        Main safety filter with complete three-layer architecture
        
        Args:
            u_nom: Nominal control input (3,)
            h: CBF value (scalar)
            grad_h: CBF gradient w.r.t. position (3,)
            drone_state: Current drone state
            
        Returns:
            u_safe: Safe control output (3,)
            solution_info: Solution diagnostics
        """
        # Layer 1: Try standard CBF-QP
        solution_info = self._solve_standard_qp(u_nom, h, grad_h, drone_state)
        
        if solution_info.is_feasible:
            return solution_info.u_safe, solution_info
        
        # Layer 2: Try relaxed QP with slack variables
        solution_info = self._solve_relaxed_qp(u_nom, h, grad_h, drone_state)
        
        if solution_info.is_feasible:
            return solution_info.u_safe, solution_info
        
        # Layer 3: Emergency brake
        emergency_control = self._emergency_brake(drone_state)
        
        emergency_info = QSolutionInfo(
            u_safe=emergency_control,
            is_feasible=False,
            solver_status=3,  # Emergency mode
            slack_violation=float('inf'),  # Large penalty
            num_iterations=0
        )
        
        return emergency_control, emergency_info

# =============================================================================
# LOSS FUNCTIONS FOR TRAINING
# =============================================================================

def compute_safety_loss(
    solution_info: QSolutionInfo,
    config: SafetyConfig
) -> Tuple[chex.Array, dict]:
    """
    Compute safety-related loss components for training
    
    Loss components:
    1. Relaxation penalty: penalize slack variable usage
    2. Failure penalty: heavily penalize solver failures
    3. Control magnitude: regularize control effort
    
    Returns:
        total_loss: Combined safety loss
        loss_dict: Individual loss components
    """
    # Relaxation penalty (encourage feasible solutions)
    relaxation_loss = config.relaxation_penalty * jnp.maximum(0.0, solution_info.slack_violation)
    
    # Failure penalty (strongly discourage solver failures)
    failure_penalty = jnp.where(
        solution_info.solver_status >= 2,
        config.failure_penalty,
        0.0
    )
    
    # Control effort regularization
    control_magnitude_loss = 0.01 * jnp.sum(solution_info.u_safe ** 2)
    
    total_loss = relaxation_loss + failure_penalty + control_magnitude_loss
    
    loss_dict = {
        "relaxation_loss": relaxation_loss,
        "failure_penalty": failure_penalty,
        "control_magnitude_loss": control_magnitude_loss,
        "solver_status": solution_info.solver_status,
        "slack_violation": solution_info.slack_violation,
        "total_safety_loss": total_loss
    }
    
    return total_loss, loss_dict

# =============================================================================
# DIFFERENTIABLE SAFETY LAYER INTERFACE
# =============================================================================

def differentiable_safety_filter(
    params_dict: dict,
    u_nom: chex.Array,
    h: chex.Array,
    grad_h: chex.Array,
    drone_state: DroneState
) -> Tuple[chex.Array, dict]:
    """
    JIT-compiled differentiable safety filter
    
    This is the main interface used by the training loop.
    
    Args:
        params_dict: Configuration parameters (converted to static)
        u_nom: Nominal control input
        h: CBF value
        grad_h: CBF gradient
        drone_state: Current drone state
        
    Returns:
        u_safe: Safe control output
        info_dict: Solution information for loss computation
    """
    config = SafetyConfig(**params_dict)
    safety_layer = SafetyLayer(config)
    
    u_safe, solution_info = safety_layer.safety_filter(u_nom, h, grad_h, drone_state)
    
    # Convert solution info to dictionary for JAX compatibility
    info_dict = {
        "u_safe": u_safe,
        "is_feasible": solution_info.is_feasible,
        "solver_status": solution_info.solver_status,
        "slack_violation": solution_info.slack_violation,
        "num_iterations": solution_info.num_iterations
    }
    
    return u_safe, info_dict

# =============================================================================
# TEMPORAL GRADIENT DECAY (DiffPhysDrone Integration)
# =============================================================================

def apply_temporal_gradient_decay(
    gradients: chex.Array,
    time_step: int,
    decay_factor: float = 0.95,
    distance_to_obstacles: Optional[chex.Array] = None
) -> chex.Array:
    """
    Apply temporal gradient decay for BPTT stability (from DiffPhysDrone)
    
    Implements the temporal gradient decay mechanism with additional 
    spatial adaptation based on obstacle proximity.
    
    Args:
        gradients: Gradients from safety layer
        time_step: Current time step in trajectory
        decay_factor: Base decay factor
        distance_to_obstacles: Distance to nearest obstacles (optional)
        
    Returns:
        decayed_gradients: Scaled gradients
    """
    # Base temporal decay (exponential)
    temporal_decay = decay_factor ** time_step
    
    # Spatial adaptation (stronger decay when far from obstacles)
    if distance_to_obstacles is not None:
        # Increase decay factor when far from obstacles (less safety critical)
        min_distance = jnp.min(distance_to_obstacles)
        spatial_factor = jnp.maximum(0.5, jnp.minimum(1.0, min_distance / 2.0))
        temporal_decay *= spatial_factor
    
    return gradients * temporal_decay

# =============================================================================
# FACTORY FUNCTIONS AND UTILITIES
# =============================================================================

def create_default_safety_layer() -> SafetyLayer:
    """Create safety layer with default configuration"""
    config = SafetyConfig()
    return SafetyLayer(config)

def validate_safety_constraints(
    u_safe: chex.Array,
    h: chex.Array,
    grad_h: chex.Array,
    drone_state: DroneState,
    config: SafetyConfig
) -> bool:
    """
    Validate that computed control satisfies CBF constraints
    
    Used for testing and verification.
    """
    # Check control magnitude constraints
    if jnp.any(jnp.abs(u_safe) > config.max_thrust + 1e-6):
        return False
    
    # Check CBF constraint satisfaction
    # grad_h^T * (f + g*u) >= -alpha * h
    f_dynamics = jnp.concatenate([
        drone_state.velocity,
        jnp.array([0.0, 0.0, -9.81])
    ])
    
    cbf_constraint_value = grad_h @ (f_dynamics + u_safe)
    cbf_threshold = -config.cbf_alpha * h
    
    return cbf_constraint_value >= cbf_threshold - 1e-6

# =============================================================================
# TESTING AND VALIDATION
# =============================================================================

def test_complete_safety_layer():
    """Test complete safety layer implementation with all three layers"""
    print("Testing Complete Safety Layer Implementation...")
    
    # Create test configuration
    config = SafetyConfig()
    safety_layer = SafetyLayer(config)
    
    # Create test drone state
    drone_state = DroneState(
        position=jnp.array([0.0, 0.0, 1.0]),
        velocity=jnp.array([1.0, 0.0, 0.0]),
        orientation=jnp.eye(3),
        angular_velocity=jnp.zeros(3)
    )
    
    # Test scenario 1: Safe nominal control (Layer 1 should succeed)
    u_nom = jnp.array([0.1, 0.0, 0.0])
    h = jnp.array(0.5)  # Positive CBF (safe)
    grad_h = jnp.array([1.0, 0.0, 0.0])
    
    u_safe, solution_info = safety_layer.safety_filter(u_nom, h, grad_h, drone_state)
    print(f" Layer 1 Test: u_safe={u_safe}, feasible={solution_info.is_feasible}, status={solution_info.solver_status}")
    assert solution_info.is_feasible and solution_info.solver_status == 0, "Safe scenario should be feasible in Layer 1"
    
    # Test scenario 2: Infeasible QP (should trigger Layer 2)
    u_nom_infeasible = jnp.array([2.0, 0.0, 0.0])  # Exceeds max thrust
    h_unsafe = jnp.array(-0.1)  # Negative CBF (unsafe)
    
    u_safe_2, solution_info_2 = safety_layer.safety_filter(
        u_nom_infeasible, h_unsafe, grad_h, drone_state
    )
    print(f" Layer 2 Test: u_safe={u_safe_2}, status={solution_info_2.solver_status}, slack={solution_info_2.slack_violation}")
    assert jnp.all(jnp.abs(u_safe_2) <= config.max_thrust), "Control should respect bounds"
    
    # Test scenario 3: Emergency brake (high speed, very dangerous)
    drone_state_emergency = DroneState(
        position=jnp.array([0.0, 0.0, 0.5]),  # Low altitude
        velocity=jnp.array([5.0, 0.0, -2.0]),  # High speed toward ground
        orientation=jnp.eye(3),
        angular_velocity=jnp.zeros(3)
    )
    h_emergency = jnp.array(-2.0)  # Very unsafe
    
    u_safe_3, solution_info_3 = safety_layer.safety_filter(
        u_nom_infeasible, h_emergency, grad_h, drone_state_emergency
    )
    print(f" Layer 3 Test: u_safe={u_safe_3}, status={solution_info_3.solver_status}")
    assert solution_info_3.solver_status == 3, "Should trigger emergency mode"
    
    # Test JIT compilation
    params_dict = {
        "max_thrust": 0.8,
        "cbf_alpha": 1.0,
        "relaxation_penalty": 1000.0,
        "max_iterations": 100,
        "tolerance": 1e-6,
        "regularization": 1e-8,
        "max_torque": 0.5,
        "safety_margin": 0.1,
        "emergency_brake_force": -0.6,
        "failure_penalty": 10000.0
    }
    
    jit_safety_filter = jax.jit(differentiable_safety_filter)
    u_safe_jit, info_jit = jit_safety_filter(params_dict, u_nom, h, grad_h, drone_state)
    print(f" JIT Test: Successful")
    assert jnp.allclose(u_safe, u_safe_jit), "JIT result should match"
    
    # Test gradient computation
    def safety_loss_fn(u_nom):
        u_safe, info = differentiable_safety_filter(params_dict, u_nom, h, grad_h, drone_state)
        return jnp.sum(u_safe ** 2)
    
    grad_fn = jax.grad(safety_loss_fn)
    gradients = grad_fn(u_nom)
    print(f" Gradient Test: gradients={gradients}")
    assert not jnp.any(jnp.isnan(gradients)), "Gradients should be valid"
    
    print("Complete Safety Layer Validation: ALL TESTS PASSED!")

if __name__ == "__main__":
    test_complete_safety_layer()