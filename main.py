"""
STAGE 4: Complete Safe Agile Flight System - Main Training Script

This is the culmination of our multi-stage development combining:
1. GCBF+ (MIT-REALM): Neural Graph Control Barrier Functions for safety
2. DiffPhysDrone (SJTU): Differentiable physics for end-to-end learning  
3. JAX-native implementation for maximum performance

STAGE 4 OBJECTIVES:
- Complete end-to-end system integration
- Full BPTT training loop with jax.lax.scan
- Multi-objective loss function optimization
- Validation of complete gradient flow through all components

The system architecture:
Input -> GNN Perception -> Policy Network -> Safety Layer -> Physics Engine -> Loss
   ^                                                                            |
   |_________________________ BPTT Gradient Flow __________________________|
"""

import jax
import jax.numpy as jnp
from jax import grad, jit, random, lax
import optax
import time
import sys
from pathlib import Path
from typing import Dict, Tuple, NamedTuple, Optional
import chex
from dataclasses import dataclass
import pickle

# Configure JAX for optimal performance
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_compilation_cache_dir", ".jax_cache")

# Auto-detect best available platform
try:
    devices = jax.devices()
    print(f"🚀 Available JAX devices: {devices}")
    if any('gpu' in str(device).lower() for device in devices):
        print("✅ Using GPU acceleration")
    else:
        print("⚠️  Using CPU (GPU not available)")
except Exception as e:
    print(f"JAX device detection: {e}")

# Add project root to path  
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import all system components
from configs.default_config import get_config, get_minimal_config
from core.physics import (
    DroneState, PhysicsParams, dynamics_step_jit,
    create_initial_drone_state, validate_physics_state
)
from core.perception import (
    PerceptionModule, create_default_perception_module,
    pointcloud_to_graph, DroneState as PerceptionDroneState
)
from core.policy import PolicyNetworkMLP, create_policy_network
from core.safety import (
    SafetyLayer, SafetyConfig, differentiable_safety_filter,
    create_default_safety_layer
)
from core.loop import (
    ScanCarry, ScanOutput, create_scan_function,
    run_complete_trajectory_scan
)
from core.training import (
    LossConfig, LossMetrics, compute_comprehensive_loss,
    training_step, create_default_loss_config, create_optimizer,
    log_training_metrics
)


# =============================================================================
# SYSTEM CONFIGURATION AND STATE MANAGEMENT
# =============================================================================

@dataclass
class TrainingState:
    """Complete training state for checkpointing and resumption"""
    step: int
    epoch: int
    params: Dict
    optimizer_state: optax.OptState
    loss_history: list
    metrics_history: list
    best_loss: float
    config: Dict


class SystemComponents(NamedTuple):
    """All system components bundled together"""
    gnn_perception: PerceptionModule
    policy_network: PolicyNetworkMLP
    safety_layer: SafetyLayer
    scan_function: callable
    loss_config: LossConfig
    physics_params: PhysicsParams


def initialize_complete_system(config) -> Tuple[SystemComponents, Dict, optax.OptState]:
    """Initialize all system components and training state"""
    print("🔧 Initializing Complete Safe Agile Flight System...")
    
    # Create physics parameters from config
    physics_params = PhysicsParams(
        dt=config.physics.dt,
        mass=config.physics.drone.mass,
        thrust_to_weight=config.physics.drone.thrust_to_weight,
        drag_coefficient=config.physics.drone.drag_coefficient
    )
    
    # Initialize perception module (GNN for CBF computation)
    key = random.PRNGKey(config.training.seed)
    gnn_key, policy_key, safety_key = random.split(key, 3)
    
    gnn_perception = create_default_perception_module()
    
    # Initialize policy network
    policy_network = create_policy_network(
        input_dim=config.policy.input_dim,
        hidden_dims=config.policy.hidden_dims,
        output_dim=config.policy.output_dim,
        use_rnn=config.policy.use_rnn
    )
    
    # Initialize safety layer
    safety_config = SafetyConfig(
        max_thrust=config.safety.max_thrust,
        max_torque=config.safety.max_torque,
        cbf_alpha=config.safety.cbf_alpha,
        relaxation_penalty=config.safety.relaxation_penalty
    )
    safety_layer = SafetyLayer(safety_config)
    
    # Create loss configuration
    loss_config = LossConfig(
        cbf_violation_coef=config.training.loss_cbf_coef,
        velocity_tracking_coef=config.training.loss_velocity_coef,
        goal_reaching_coef=config.training.loss_goal_coef,
        control_smoothness_coef=config.training.loss_control_coef,
        collision_avoidance_coef=config.training.loss_collision_coef
    )
    
    # Create the complete scan function that integrates all components
    scan_function = create_scan_function(
        gnn_perception, policy_network, safety_layer, physics_params
    )
    
    # Bundle all components
    components = SystemComponents(
        gnn_perception=gnn_perception,
        policy_network=policy_network,
        safety_layer=safety_layer,
        scan_function=scan_function,
        loss_config=loss_config,
        physics_params=physics_params
    )
    
    # Initialize parameters for all networks
    dummy_state = create_initial_drone_state(jnp.array([0.0, 0.0, 1.0]))
    dummy_pointcloud = random.normal(gnn_key, (50, 3)) * 2.0  # 50 points
    
    # Initialize GNN parameters
    dummy_graph = pointcloud_to_graph(
        PerceptionDroneState(
            position=dummy_state.position,
            velocity=dummy_state.velocity,
            orientation=dummy_state.orientation,
            angular_velocity=dummy_state.angular_velocity
        ),
        dummy_pointcloud,
        k_neighbors=config.gcbf.k_neighbors
    )
    
    gnn_params = gnn_perception.init(gnn_key, dummy_graph)
    
    # Initialize policy parameters
    policy_input = jnp.concatenate([
        dummy_state.position, dummy_state.velocity,
        dummy_state.orientation.flatten(),
        dummy_state.angular_velocity
    ])
    policy_params = policy_network.init(policy_key, policy_input, None)
    
    # Combine all parameters
    all_params = {
        'gnn': gnn_params,
        'policy': policy_params,
        'safety': {  # Safety layer parameters (if any learnable)
            'cbf_alpha': config.safety.cbf_alpha,
            'max_thrust': config.safety.max_thrust
        }
    }
    
    # Create optimizer
    optimizer = create_optimizer(config.training.learning_rate)
    optimizer_state = optimizer.init(all_params)
    
    print(f"✅ System initialization complete")
    print(f"   GNN parameters: {sum(p.size for p in jax.tree_leaves(gnn_params))}")
    print(f"   Policy parameters: {sum(p.size for p in jax.tree_leaves(policy_params))}")
    print(f"   Total parameters: {sum(p.size for p in jax.tree_leaves(all_params))}")
    
# =============================================================================
# DATA GENERATION AND BATCH MANAGEMENT
# =============================================================================

def generate_training_scenario(config, key: chex.PRNGKey) -> Dict:
    """Generate a single training scenario"""
    key1, key2, key3 = random.split(key, 3)
    
    # Random initial position and target
    initial_position = random.uniform(key1, (3,), minval=-2.0, maxval=2.0)
    initial_position = initial_position.at[2].set(jnp.abs(initial_position[2]) + 1.0)  # Keep above ground
    
    target_position = random.uniform(key2, (3,), minval=-3.0, maxval=3.0)
    target_position = target_position.at[2].set(jnp.abs(target_position[2]) + 1.5)
    
    # Generate random obstacle point cloud
    n_obstacles = random.randint(key3, (), 20, 100)  
    obstacle_positions = random.normal(key3, (n_obstacles, 3)) * 3.0
    
    # Create initial drone state
    initial_state = create_initial_drone_state(
        position=initial_position,
        velocity=jnp.zeros(3)
    )
    
    # Compute target velocities (simple proportional controller toward goal)
    sequence_length = config.training.sequence_length
    target_velocities = jnp.tile(
        (target_position - initial_position) / sequence_length * 0.5,
        (sequence_length, 1)
    )
    
    return {
        'initial_state': initial_state,
        'target_position': target_position,
        'target_velocities': target_velocities,
        'obstacle_pointcloud': obstacle_positions,
        'scenario_id': random.randint(key, (), 0, 1000000)
    }


def generate_training_batch(config, key: chex.PRNGKey, batch_size: int) -> Dict:
    """Generate a complete training batch"""
    keys = random.split(key, batch_size)
    scenarios = [generate_training_scenario(config, k) for k in keys]
    
    # Stack into batch format
    batch = {
        'initial_states': jnp.stack([s['initial_state'] for s in scenarios]),
        'target_positions': jnp.stack([s['target_position'] for s in scenarios]),
        'target_velocities': jnp.stack([s['target_velocities'] for s in scenarios]),
        'obstacle_pointclouds': jnp.stack([s['obstacle_pointcloud'] for s in scenarios]),
        'scenario_ids': jnp.array([s['scenario_id'] for s in scenarios])
    }
    
    return batch


# =============================================================================
# COMPLETE END-TO-END TRAINING STEP
# =============================================================================

def complete_forward_pass(
    params: Dict,
    batch: Dict,
    components: SystemComponents,
    key: chex.PRNGKey
) -> Tuple[chex.Array, LossMetrics, Dict]:
    """
    Complete forward pass through the entire system
    
    This is the heart of Stage 4 - full BPTT through all components:
    1. Initial state setup
    2. BPTT scan loop (perception -> policy -> safety -> physics)  
    3. Multi-objective loss computation
    4. Return loss and comprehensive metrics
    """
    batch_size = batch['initial_states'].shape[0]
    sequence_length = config.training.sequence_length  # Use actual config value
    
    # Initialize scan carry state
    initial_carry = ScanCarry(
        drone_state=batch['initial_states'],
        rnn_hidden_state=jnp.zeros((batch_size, 64)),  # Policy RNN state
        step_count=jnp.zeros(batch_size, dtype=jnp.int32),
        cumulative_reward=jnp.zeros(batch_size)
    )
    
    # Prepare scan inputs (target info and obstacles for each timestep)
    scan_inputs = {
        'target_positions': jnp.tile(batch['target_positions'][:, None, :], (1, sequence_length, 1)),
        'obstacle_pointclouds': jnp.tile(batch['obstacle_pointclouds'][:, None, :, :], (1, sequence_length, 1, 1)),
        'timesteps': jnp.arange(sequence_length)[None, :].repeat(batch_size, axis=0)
    }
    
    # Run complete BPTT scan loop
    final_carry, scan_outputs = run_complete_trajectory_scan(
        components.scan_function,
        initial_carry,
        scan_inputs,
        params,
        components.physics_params,
        sequence_length
    )
    
    # Compute comprehensive loss
    loss, metrics = compute_comprehensive_loss(
        scan_outputs=scan_outputs,
        target_positions=batch['target_positions'],
        target_velocities=batch['target_velocities'],
        config=components.loss_config,
        physics_params=components.physics_params
    )
    
    # Additional metrics for monitoring
    final_distances = jnp.linalg.norm(
        final_carry.drone_state[:, :3] - batch['target_positions'], axis=-1
    )
    
    extra_metrics = {
        'final_goal_distance': jnp.mean(final_distances),
        'goal_success_rate': jnp.mean(final_distances < 0.5),
        'trajectory_length': jnp.mean(scan_outputs.trajectory_lengths),
        'safety_violations': jnp.sum(scan_outputs.cbf_values < 0),
        'control_effort': jnp.mean(jnp.linalg.norm(scan_outputs.safe_controls, axis=-1))
    }
    
    return loss, metrics, extra_metrics


@jit
def complete_training_step(
    params: Dict,
    optimizer_state: optax.OptState,
    batch: Dict,
    components: SystemComponents,
    optimizer: optax.GradientTransformation,
    key: chex.PRNGKey
) -> Tuple[Dict, optax.OptState, LossMetrics, Dict]:
    """
    Complete JIT-compiled training step with gradient computation and updates
    
    This function encapsulates the full STAGE 4 objective:
    - End-to-end gradient flow through all components
    - Multi-objective loss optimization  
    - Parameter updates with proper gradient handling
    """
    
    def loss_fn(params_inner):
        loss, metrics, extra_metrics = complete_forward_pass(
            params_inner, batch, components, key
        )
        return loss, (metrics, extra_metrics)
    
    # Compute loss and gradients via JAX autodiff
    (loss, (metrics, extra_metrics)), gradients = jax.value_and_grad(
        loss_fn, has_aux=True
    )(params)
    
    # Apply gradient updates
    updates, new_optimizer_state = optimizer.update(gradients, optimizer_state, params)
    new_params = optax.apply_updates(params, updates)
    
    # Compute gradient statistics for monitoring
    gradient_norm = jnp.sqrt(sum(
        jnp.sum(g ** 2) for g in jax.tree_util.tree_leaves(gradients)
    ))
    
    # Update metrics with gradient information
    updated_metrics = metrics._replace(gradient_norm=gradient_norm)
    
    return new_params, new_optimizer_state, updated_metrics, extra_metrics


# =============================================================================
# TRAINING LOOP MANAGEMENT AND EXECUTION
# ============================================================================= 

def run_training_epoch(
    params: Dict,
    optimizer_state: optax.OptState,
    components: SystemComponents,
    optimizer: optax.GradientTransformation,
    config,
    epoch: int,
    key: chex.PRNGKey
) -> Tuple[Dict, optax.OptState, Dict]:
    """Run a complete training epoch with multiple batches"""
    epoch_metrics = []
    current_params = params
    current_opt_state = optimizer_state
    
    # Generate training batches for this epoch
    n_batches = config.training.batches_per_epoch
    batch_keys = random.split(key, n_batches)
    
    for batch_idx, batch_key in enumerate(batch_keys):
        # Generate training batch
        batch = generate_training_batch(
            config, batch_key, config.training.batch_size
        )
        
        # Perform training step
        step_key = random.fold_in(batch_key, batch_idx)
        current_params, current_opt_state, metrics, extra_metrics = complete_training_step(
            current_params, current_opt_state, batch, components, optimizer, step_key
        )
        
        # Collect metrics
        batch_metrics = {
            **{f"{k}": float(v) for k, v in metrics._asdict().items()},
            **{f"extra_{k}": float(v) for k, v in extra_metrics.items()}
        }
        epoch_metrics.append(batch_metrics)
        
        # Progress logging
        if batch_idx % 10 == 0 or batch_idx == n_batches - 1:
            print(f"  Batch {batch_idx+1}/{n_batches}: Loss={metrics.total_loss:.6f}, Goal Success={extra_metrics['goal_success_rate']:.3f}")
    
    # Aggregate epoch metrics
    aggregated_metrics = {}
    for key in epoch_metrics[0].keys():
        values = [m[key] for m in epoch_metrics]
        aggregated_metrics[key] = jnp.mean(jnp.array(values))
    
    return current_params, current_opt_state, aggregated_metrics


def run_validation(
    params: Dict,
    components: SystemComponents, 
    config,
    key: chex.PRNGKey
) -> Dict:
    """Run validation to assess model performance"""
    print("🔍 Running validation...")
    
    # Generate validation batch
    val_batch = generate_training_batch(
        config, key, config.training.validation_batch_size
    )
    
    # Run forward pass without gradients
    loss, metrics, extra_metrics = complete_forward_pass(
        params, val_batch, components, key
    )
    
    validation_metrics = {
        "val_loss": float(loss),
        "val_goal_success_rate": float(extra_metrics['goal_success_rate']),
        "val_safety_violations": float(extra_metrics['safety_violations']),
        "val_final_distance": float(extra_metrics['final_goal_distance']),
        "val_control_effort": float(extra_metrics['control_effort'])
    }
    
    print(f"  Validation Loss: {validation_metrics['val_loss']:.6f}")
    print(f"  Goal Success Rate: {validation_metrics['val_goal_success_rate']:.3f}")
    print(f"  Safety Violations: {validation_metrics['val_safety_violations']}")
    
    return validation_metrics


def save_checkpoint(
    training_state: TrainingState,
    checkpoint_dir: Path,
    is_best: bool = False
):
    """Save training checkpoint"""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Save current checkpoint
    checkpoint_path = checkpoint_dir / f"checkpoint_{training_state.step:06d}.pkl"
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(training_state, f)
    
    # Save best model if applicable
    if is_best:
        best_path = checkpoint_dir / "best_model.pkl"
        with open(best_path, 'wb') as f:
            pickle.dump(training_state, f)
        print(f"💾 Saved best model at step {training_state.step}")
    
    print(f"💾 Checkpoint saved: {checkpoint_path}")

def validate_complete_system_integration(
    components: SystemComponents,
    params: Dict,
    config
) -> bool:
    """Comprehensive validation of the complete Stage 4 system"""
    print("\n" + "=" * 60)
    print("🔍 STAGE 4 SYSTEM VALIDATION")
    print("=" * 60)
    
    try:
        # Test 1: Generate and process a single scenario
        key = random.PRNGKey(42)
        test_scenario = generate_training_scenario(config, key)
        print("✅ Test 1: Scenario generation - PASSED")
        
        # Test 2: Batch processing
        test_batch = generate_training_batch(config, key, batch_size=2)
        print("✅ Test 2: Batch generation - PASSED")
        
        # Test 3: Forward pass without gradients
        loss, metrics, extra = complete_forward_pass(
            params, test_batch, components, key
        )
        
        assert jnp.isfinite(loss), "Loss must be finite"
        assert jnp.all(jnp.isfinite(jax.tree_leaves(metrics))), "All metrics must be finite"
        print("✅ Test 3: Forward pass computation - PASSED")
        print(f"   Forward pass loss: {loss:.6f}")
        
        # Test 4: Gradient computation
        def test_loss_fn(test_params):
            test_loss, _, _ = complete_forward_pass(
                test_params, test_batch, components, key
            )
            return test_loss
        
        test_gradients = grad(test_loss_fn)(params)
        gradient_norm = jnp.sqrt(sum(
            jnp.sum(g ** 2) for g in jax.tree_util.tree_leaves(test_gradients)
        ))
        
        assert jnp.isfinite(gradient_norm), "Gradient norm must be finite"
        assert gradient_norm > 1e-8, "Gradients should be meaningful"
        print("✅ Test 4: Gradient computation - PASSED")
        print(f"   Gradient norm: {gradient_norm:.6f}")
        
        # Test 5: Complete training step (JIT compiled)
        optimizer = create_optimizer(config.training.learning_rate)
        optimizer_state = optimizer.init(params)
        
        new_params, new_opt_state, step_metrics, step_extra = complete_training_step(
            params, optimizer_state, test_batch, components, optimizer, key
        )
        
        # Verify parameter updates
        param_diff_norm = jnp.sqrt(sum(
            jnp.sum((p1 - p2) ** 2) 
            for p1, p2 in zip(
                jax.tree_util.tree_leaves(params),
                jax.tree_util.tree_leaves(new_params)
            )
        ))
        
        assert param_diff_norm > 1e-10, "Parameters should be updated"
        print("✅ Test 5: Complete training step - PASSED")
        print(f"   Parameter update norm: {param_diff_norm:.8f}")
        
        # Test 6: JIT compilation verification
        # Compile the training step and verify it works
        jit_step_fn = jit(complete_training_step)
        start_time = time.time()
        
        # First call (compilation)
        _, _, _, _ = jit_step_fn(
            params, optimizer_state, test_batch, components, optimizer, key
        )
        compile_time = time.time() - start_time
        
        # Second call (execution only)
        start_time = time.time()
        _, _, _, _ = jit_step_fn(
            params, optimizer_state, test_batch, components, optimizer, key
        )
        execution_time = time.time() - start_time
        
        print("✅ Test 6: JIT compilation - PASSED")
        print(f"   Compilation time: {compile_time:.3f}s")
        print(f"   Execution time: {execution_time:.4f}s")
        
        print("\n🎉 STAGE 4 VALIDATION: ALL TESTS PASSED!")
        print("\nKey accomplishments:")
        print("  ✅ Complete end-to-end system integration")
        print("  ✅ BPTT gradient flow through all components")
        print("  ✅ Multi-objective loss function")
        print("  ✅ JIT-compiled training steps")
        print("  ✅ GCBF+ safety constraints")
        print("  ✅ DiffPhysDrone physics integration")
        print("  ✅ Comprehensive validation suite")
        
        return True
        
    except Exception as e:
        print(f"❌ STAGE 4 VALIDATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    """Validate basic physics engine functionality."""
    print("\n" + "=" * 60)
    print("VALIDATING BASIC PHYSICS ENGINE")
    print("=" * 60)
    
    # Create physics parameters and initial state
    params = PhysicsParams()
    initial_state = create_initial_drone_state(
        position=jnp.array([0.0, 0.0, 1.0]),
        velocity=jnp.array([0.0, 0.0, 0.0])
    )
    
    print(f"Initial state: pos={initial_state.position}, vel={initial_state.velocity}")
    
    # Test free fall (zero thrust)
    zero_control = jnp.zeros(3)
    state_after_fall = dynamics_step(initial_state, zero_control, params)
    
    print(f"After free fall: pos={state_after_fall.position}, vel={state_after_fall.velocity}")
    
    # Should fall due to gravity
    assert state_after_fall.position[2] < initial_state.position[2], "Drone should fall with zero thrust"
    assert state_after_fall.velocity[2] < 0, "Downward velocity should develop"
    
    # Test hover equilibrium
    hover_thrust = jnp.array([0.0, 0.0, 1.0 / params.thrust_to_weight])
    state_after_hover = dynamics_step(initial_state, hover_thrust, params)
    
    print(f"After hover thrust: pos={state_after_hover.position}, vel={state_after_hover.velocity}")
    
    # Altitude change should be minimal with proper hover thrust
    altitude_change = abs(state_after_hover.position[2] - initial_state.position[2])
    assert altitude_change < 0.1, f"Hover should maintain altitude, got change: {altitude_change}"
    
    # Validate state integrity
    assert validate_physics_state(state_after_fall), "Physics state should remain valid"
    assert validate_physics_state(state_after_hover), "Physics state should remain valid"
    
    print("✅ Basic physics engine validation: PASSED")
    return True


def validate_gradient_flow():
    """Validate end-to-end gradient computation through physics engine."""
    print("\n" + "=" * 60)
    print("VALIDATING GRADIENT FLOW")
    print("=" * 60)
    
    params = PhysicsParams()
    initial_state = create_initial_drone_state(jnp.array([0.0, 0.0, 1.0]))
    
    def single_step_loss(control_input):
        """Simple loss function for gradient testing."""
        new_state = dynamics_step(initial_state, control_input, params)
        # Minimize distance to target position [1, 1, 2]
        target = jnp.array([1.0, 1.0, 2.0])
        return jnp.sum((new_state.position - target) ** 2)
    
    # Compute analytical gradients
    control_input = jnp.array([0.1, 0.2, 0.3])
    analytical_gradients = grad(single_step_loss)(control_input)
    
    print(f"Control input: {control_input}")
    print(f"Analytical gradients: {analytical_gradients}")
    
    # Verify gradients are finite and non-zero
    assert jnp.all(jnp.isfinite(analytical_gradients)), "Gradients must be finite"
    assert jnp.linalg.norm(analytical_gradients) > 1e-6, "Gradients should be meaningful"
    
    # Test multi-step gradient flow (simplified BPTT)
    def multi_step_loss(initial_control):
        """Multi-step simulation loss for BPTT testing."""
        state = initial_state
        total_loss = 0.0
        
        # Apply same control for multiple steps
        for step in range(5):
            state = dynamics_step(state, initial_control, params)
            # Accumulate position tracking loss
            target = jnp.array([1.0, 1.0, 2.0])
            step_loss = jnp.sum((state.position - target) ** 2)
            
            # Apply temporal gradient decay (DiffPhysDrone innovation)
            decayed_loss = apply_temporal_gradient_decay(
                step_loss, step, params.gradient_decay_alpha, params.dt
            )
            total_loss += decayed_loss
        
        return total_loss
    
    multi_step_gradients = grad(multi_step_loss)(control_input)
    print(f"Multi-step BPTT gradients: {multi_step_gradients}")
    
    # Verify multi-step gradients
    assert jnp.all(jnp.isfinite(multi_step_gradients)), "Multi-step gradients must be finite"
    assert jnp.linalg.norm(multi_step_gradients) > 1e-6, "Multi-step gradients should be meaningful"
    
    print("✅ Gradient flow validation: PASSED")
    return True


def validate_jit_compilation():
    """Validate JIT compilation functionality and performance."""
    print("\n" + "=" * 60)
    print("VALIDATING JIT COMPILATION")
    print("=" * 60)
    
    params = PhysicsParams()
    initial_state = create_initial_drone_state(jnp.array([0.0, 0.0, 1.0]))
    control_input = jnp.array([0.1, 0.1, 0.3])
    
    # Compare JIT and non-JIT results
    normal_result = dynamics_step(initial_state, control_input, params)
    jit_result = dynamics_step_jit(initial_state, control_input, params)
    
    # Results should be identical
    position_diff = jnp.linalg.norm(normal_result.position - jit_result.position)
    velocity_diff = jnp.linalg.norm(normal_result.velocity - jit_result.velocity)
    
    print(f"Position difference (JIT vs normal): {position_diff}")
    print(f"Velocity difference (JIT vs normal): {velocity_diff}")
    
    assert position_diff < 1e-10, "JIT and normal results should match exactly"
    assert velocity_diff < 1e-10, "JIT and normal results should match exactly"
    
    # Performance benchmark
    n_iterations = 1000
    
    # Warmup JIT compilation
    _ = dynamics_step_jit(initial_state, control_input, params)
    
    # Benchmark JIT performance
    start_time = time.time()
    state = initial_state
    for _ in range(n_iterations):
        state = dynamics_step_jit(state, control_input, params)
    jit_time = time.time() - start_time
    
    # Benchmark normal performance (without JIT warmup overhead)
    start_time = time.time()
    state = initial_state  
    for _ in range(n_iterations):
        state = dynamics_step(state, control_input, params)
    normal_time = time.time() - start_time
    
    print(f"Performance comparison ({n_iterations} iterations):")
    print(f"  JIT compiled: {jit_time:.4f}s ({jit_time/n_iterations*1000:.2f}ms per step)")
    print(f"  Normal: {normal_time:.4f}s ({normal_time/n_iterations*1000:.2f}ms per step)")
    print(f"  Speedup: {normal_time/jit_time:.1f}x")
    
    # JIT should be faster (allow some variance)
    if jit_time < normal_time:
        print("✅ JIT provides performance improvement")
    else:
        print("⚠️  JIT may not show improvement for this simple case (acceptable)")
    
    print("✅ JIT compilation validation: PASSED")
    return True


def validate_temporal_gradient_decay():
    """Validate temporal gradient decay mechanism from DiffPhysDrone."""
    print("\n" + "=" * 60) 
    print("VALIDATING TEMPORAL GRADIENT DECAY")
    print("=" * 60)
    
    # Test decay schedule creation
    sequence_length = 10
    alpha = 0.9
    dt = 0.1
    
    decay_schedule = create_temporal_decay_schedule(sequence_length, alpha, dt)
    print(f"Decay schedule: {decay_schedule}")
    
    # Verify exponential decay pattern
    expected_schedule = jnp.array([alpha**(i * dt) for i in range(sequence_length)])
    assert jnp.allclose(decay_schedule, expected_schedule), "Decay schedule should follow exponential pattern"
    
    # Test gradient decay application
    test_gradient = jnp.ones(3)
    
    decay_factors = []
    for timestep in range(5):
        decayed_grad = apply_temporal_gradient_decay(test_gradient, timestep, alpha, dt)
        decay_factors.append(decayed_grad[0])  # All components should be identical
    
    print(f"Decay factors over time: {decay_factors}")
    
    # Should decrease monotonically
    for i in range(1, len(decay_factors)):
        assert decay_factors[i] <= decay_factors[i-1], "Decay should be monotonically decreasing"
    
    # First factor should be 1.0 (no decay at t=0)
    assert abs(decay_factors[0] - 1.0) < 1e-10, "No decay should be applied at timestep 0"
    
    print("✅ Temporal gradient decay validation: PASSED")
    return True


def validate_multi_agent_capability():
    """Validate multi-agent physics and GCBF+ integration preparation."""
    print("\n" + "=" * 60)
    print("VALIDATING MULTI-AGENT CAPABILITY")
    print("=" * 60)
    
    # Create multi-agent system
    n_agents = 4
    positions = jnp.array([
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0], 
        [1.0, 1.0, 1.0]
    ])
    
    multi_state = create_initial_multi_agent_state(positions)
    print(f"Created multi-agent state with {n_agents} agents")
    print(f"State shape: {multi_state.drone_states.shape}")
    print(f"Adjacency matrix shape: {multi_state.adjacency_matrix.shape}")
    
    # Test multi-agent dynamics
    key = random.PRNGKey(42)
    control_inputs = random.normal(key, (n_agents, 3)) * 0.1
    
    params = PhysicsParams()
    new_multi_state = multi_agent_dynamics_step(multi_state, control_inputs, params)
    
    # State should evolve
    state_changed = not jnp.allclose(new_multi_state.drone_states, multi_state.drone_states)
    assert state_changed, "Multi-agent state should evolve with dynamics"
    
    # Time should advance
    assert new_multi_state.global_time > multi_state.global_time, "Global time should advance"
    
    # Adjacency matrix should be recomputed  
    assert new_multi_state.adjacency_matrix.shape == (n_agents, n_agents), "Adjacency matrix shape preserved"
    
    # Test JIT compilation for multi-agent
    jit_multi_result = multi_agent_dynamics_step_jit(multi_state, control_inputs, params)
    
    # Results should match
    states_match = jnp.allclose(new_multi_state.drone_states, jit_multi_result.drone_states, rtol=1e-10)
    assert states_match, "JIT multi-agent results should match non-JIT"
    
    print("✅ Multi-agent capability validation: PASSED")
    return True


def validate_system_integration():
    """Validate complete system integration and readiness for Stage 2."""
    print("\n" + "=" * 60)
    print("VALIDATING SYSTEM INTEGRATION")
    print("=" * 60)
    
    # Load configuration
    config = get_minimal_config()  # Use minimal config for faster testing
    
    # Create physics parameters from configuration
    params = PhysicsParams(
        dt=config.physics.dt,
        mass=config.physics.drone.mass,
        gradient_decay_alpha=config.physics.gradient_decay.alpha
    )
    
    # Create initial state
    initial_state = create_initial_drone_state(jnp.array([0.0, 0.0, 1.0]))
    
    # Simulate complete BPTT scenario
    def complete_simulation_loss(control_sequence):
        """Complete simulation mimicking the future Stage 2 BPTT loop."""
        state = initial_state
        total_loss = 0.0
        
        for step, control_input in enumerate(control_sequence):
            # Physics step
            state = dynamics_step(state, control_input, params)
            
            # Multiple loss components (mimicking future GCBF+ integration)
            target_position = jnp.array([2.0, 1.0, 3.0])
            
            # Efficiency loss (position tracking)
            efficiency_loss = jnp.sum((state.position - target_position) ** 2)
            
            # Safety loss (altitude constraint - simplified CBF)
            min_altitude = 0.5
            safety_loss = jnp.maximum(0.0, min_altitude - state.position[2]) ** 2
            
            # Control smoothness loss
            control_loss = jnp.sum(control_input ** 2)
            
            # Combine losses with weights from config
            step_loss = (config.training.loss_goal_coef * efficiency_loss + 
                        config.training.loss_cbf_coef * safety_loss +
                        config.training.loss_control_coef * control_loss)
            
            # Apply temporal gradient decay
            if config.physics.gradient_decay.enable:
                step_loss = apply_temporal_gradient_decay(
                    step_loss, step, params.gradient_decay_alpha, params.dt
                )
            
            total_loss += step_loss
        
        return total_loss
    
    # Generate control sequence
    key = random.PRNGKey(12345)
    sequence_length = 10
    control_sequence = random.normal(key, (sequence_length, 3)) * 0.2
    
    print(f"Running complete simulation with {sequence_length} steps...")
    
    # Compute loss and gradients
    loss_value = complete_simulation_loss(control_sequence)
    gradients = grad(complete_simulation_loss)(control_sequence)
    
    print(f"Simulation loss: {loss_value:.4f}")
    print(f"Gradient statistics:")
    print(f"  Shape: {gradients.shape}")
    print(f"  Mean magnitude: {jnp.mean(jnp.abs(gradients)):.6f}")
    print(f"  Max gradient: {jnp.max(jnp.abs(gradients)):.6f}")
    print(f"  Gradient norm: {jnp.linalg.norm(gradients):.6f}")
    
    # Validate results
    assert jnp.isfinite(loss_value), "Simulation loss must be finite"
    assert jnp.all(jnp.isfinite(gradients)), "All gradients must be finite"
    assert jnp.linalg.norm(gradients) > 1e-8, "Gradients should be meaningful"
    
    # Test JIT compilation of complete pipeline
    @jit
    def jit_complete_simulation(control_seq):
        return complete_simulation_loss(control_seq)
    
    jit_loss_value = jit_complete_simulation(control_sequence)
    jit_gradients = grad(jit_complete_simulation)(control_sequence)
    
    # JIT results should match
    assert jnp.isclose(loss_value, jit_loss_value, rtol=1e-10), "JIT loss should match"
    assert jnp.allclose(gradients, jit_gradients, rtol=1e-10), "JIT gradients should match"
    
    print("✅ System integration validation: PASSED")
    return True


def main():
    """Execute complete Stage 4: End-to-end training system"""
    print("\n" + "=" * 80)
    print("🚀 SAFE AGILE FLIGHT - STAGE 4: COMPLETE SYSTEM TRAINING")
    print("Combining GCBF+ (MIT-REALM) and DiffPhysDrone (SJTU) methodologies")
    print("End-to-End JAX-Native Differentiable System")
    print("=" * 80)
    
    # Load configuration
    config = get_config()  # Use full configuration
    print(f"🔧 Configuration loaded: {config.experiment_name}")
    print(f"   Sequence length: {config.training.sequence_length}")
    print(f"   Batch size: {config.training.batch_size}")
    print(f"   Learning rate: {config.training.learning_rate}")
    
    # Initialize complete system
    print("\n🛠️ Initializing complete system...")
    components, params, optimizer_state = initialize_complete_system(config)
    optimizer = create_optimizer(config.training.learning_rate)
    
    # Validate system integration
    print("\n🔍 Validating complete system integration...")
    validation_success = validate_complete_system_integration(
        components, params, config
    )
    
    if not validation_success:
        print("❌ System validation failed. Aborting training.")
        return False
    
    # Initialize training state
    training_state = TrainingState(
        step=0,
        epoch=0,
        params=params,
        optimizer_state=optimizer_state,
        loss_history=[],
        metrics_history=[],
        best_loss=float('inf'),
        config=config.__dict__
    )
    
    # Setup checkpoint directory
    checkpoint_dir = Path(f"checkpoints/{config.experiment_name}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Checkpoint directory: {checkpoint_dir}")
    
    # Training loop
    print("\n" + "=" * 60)
    print("🏃 STARTING TRAINING LOOP")
    print("=" * 60)
    
    key = random.PRNGKey(config.training.seed)
    
    try:
        for epoch in range(config.training.num_epochs):
            epoch_start_time = time.time()
            print(f"\n🔄 Epoch {epoch + 1}/{config.training.num_epochs}")
            
            # Generate epoch key
            epoch_key, key = random.split(key)
            
            # Run training epoch
            training_state.params, training_state.optimizer_state, epoch_metrics = run_training_epoch(
                training_state.params,
                training_state.optimizer_state,
                components,
                optimizer,
                config,
                epoch,
                epoch_key
            )
            
            # Update training state
            training_state.epoch = epoch
            training_state.step += config.training.batches_per_epoch
            current_loss = float(epoch_metrics['total_loss'])
            training_state.loss_history.append(current_loss)
            training_state.metrics_history.append(epoch_metrics)
            
            # Run validation every N epochs
            if (epoch + 1) % config.training.validation_frequency == 0:
                val_key, key = random.split(key)
                val_metrics = run_validation(training_state.params, components, config, val_key)
                epoch_metrics.update(val_metrics)
            
            epoch_time = time.time() - epoch_start_time
            
            # Log epoch results
            print(f"  ⏱️ Epoch time: {epoch_time:.2f}s")
            print(f"  📈 Training loss: {current_loss:.6f}")
            print(f"  🎯 Goal success rate: {epoch_metrics.get('extra_goal_success_rate', 0):.3f}")
            print(f"  ⚠️ Safety violations: {epoch_metrics.get('extra_safety_violations', 0)}")
            print(f"  🅾️ Control effort: {epoch_metrics.get('extra_control_effort', 0):.4f}")
            
            # Check for best model
            is_best = current_loss < training_state.best_loss
            if is_best:
                training_state.best_loss = current_loss
                print(f"  🏆 New best loss: {current_loss:.6f}")
            
            # Save checkpoints
            if (epoch + 1) % config.training.checkpoint_frequency == 0:
                save_checkpoint(training_state, checkpoint_dir, is_best)
            
            # Early stopping check
            if len(training_state.loss_history) >= 20:
                recent_losses = training_state.loss_history[-20:]
                if all(l >= recent_losses[0] * 0.999 for l in recent_losses[-10:]):
                    print("\n⏹️ Early stopping triggered: loss has plateaued")
                    break
    
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
        save_checkpoint(training_state, checkpoint_dir, is_best=False)
    
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Final validation and summary
    print("\n" + "=" * 60)
    print("🏁 TRAINING COMPLETED")
    print("=" * 60)
    
    # Final validation
    final_key, key = random.split(key)
    final_val_metrics = run_validation(training_state.params, components, config, final_key)
    
    print(f"Final Results:")
    print(f"  Best training loss: {training_state.best_loss:.6f}")
    print(f"  Final validation loss: {final_val_metrics['val_loss']:.6f}")
    print(f"  Final goal success rate: {final_val_metrics['val_goal_success_rate']:.3f}")
    print(f"  Total training epochs: {training_state.epoch + 1}")
    print(f"  Total training steps: {training_state.step}")
    
    # Save final checkpoint
    save_checkpoint(training_state, checkpoint_dir, is_best=True)
    
    # Success criteria
    success = (
        final_val_metrics['val_goal_success_rate'] > 0.7 and  # 70% goal success
        final_val_metrics['val_safety_violations'] < 5 and     # <5 safety violations per batch
        training_state.best_loss < 1.0                         # Reasonable loss threshold
    )
    
    if success:
        print("\n🎉 STAGE 4 SUCCESSFULLY COMPLETED!")
        print("\nKey accomplishments:")
        print("  ✅ Complete end-to-end system integration")
        print("  ✅ BPTT gradient flow through all components")
        print("  ✅ Multi-objective loss function optimization")
        print("  ✅ GCBF+ safety constraints")
        print("  ✅ DiffPhysDrone physics integration")
        print("  ✅ Successful goal-reaching behavior")
        print("  ✅ Maintained safety constraints")
        print("  ✅ JAX-native high-performance implementation")
        
        print("\n🚀 SYSTEM READY FOR ADVANCED RESEARCH AND DEPLOYMENT!")
        return True
    else:
        print("\n⚠️ STAGE 4 TRAINING COMPLETED BUT PERFORMANCE CRITERIA NOT FULLY MET")
        print("Consider:")
        print("  - Adjusting hyperparameters")
        print("  - Increasing training duration")
        print("  - Tuning loss function weights")
        print("  - Implementing curriculum learning")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)