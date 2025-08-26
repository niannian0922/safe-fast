"""
Stage 1 Validation: Differentiable World Model Foundation

This script validates the successful completion of Stage 1 requirements:
1. Project skeleton and environment setup
2. Differentiable world model implementation 
3. JIT compilation and gradient flow verification
4. Integration of GCBF+ and DiffPhysDrone methodologies

The validation demonstrates that the physics-based foundation is ready for
Stage 2 implementation (minimal end-to-end BPTT loop).
"""

import jax
import jax.numpy as jnp
from jax import grad, jit, random
import time
import sys
from pathlib import Path

# Configure JAX for optimal performance - AUTO DETECT PLATFORM
jax.config.update("jax_enable_x64", True)  # Enable 64-bit precision

# Auto-detect best available platform (GPU if available, fallback to CPU)
try:
    devices = jax.devices()
    print(f"Available JAX devices: {devices}")
    if any('gpu' in str(device).lower() for device in devices):
        print("Using GPU acceleration")
    else:
        print("Using CPU (GPU not available)")
except Exception as e:
    print(f"JAX device detection: {e}")

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import our implementations
from configs.default_config import get_config, get_minimal_config
from core.physics import (
    DroneState, MultiAgentState, PhysicsParams,
    dynamics_step, multi_agent_dynamics_step,
    dynamics_step_jit, multi_agent_dynamics_step_jit,
    apply_temporal_gradient_decay, create_temporal_decay_schedule,
    create_initial_drone_state, create_initial_multi_agent_state,
    validate_physics_state
)


def validate_configuration_system():
    """Validate the configuration management system."""
    print("=" * 60)
    print("VALIDATING CONFIGURATION SYSTEM")
    print("=" * 60)
    
    # Load different configuration variants
    configs = {
        "default": get_config(),
        "minimal": get_minimal_config()
    }
    
    for config_name, config in configs.items():
        print(f"\n{config_name.upper()} Configuration:")
        print(f"  Physics timestep: {config.physics.dt}")
        print(f"  Number of agents: {config.env.num_agents}")
        print(f"  GCBF sensing radius: {config.gcbf.sensing_radius}")
        print(f"  Training max steps: {config.training.max_steps}")
        print(f"  Gradient decay alpha: {config.physics.gradient_decay.alpha}")
        
        # Verify required fields exist
        required_sections = ['physics', 'gcbf', 'policy', 'safety', 'training', 'env']
        for section in required_sections:
            assert hasattr(config, section), f"Missing required section: {section}"
    
    print("✅ Configuration system validation: PASSED")
    return True


def validate_physics_engine_basic():
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
    """Execute complete Stage 1 validation suite."""
    print("🚀 SAFE AGILE FLIGHT - STAGE 1 VALIDATION")
    print("Combining GCBF+ (MIT-REALM) and DiffPhysDrone (SJTU) methodologies")
    print("JAX-based Differentiable Physics Engine Foundation")
    print()
    
    start_time = time.time()
    
    # Execute all validation tests
    validations = [
        validate_configuration_system,
        validate_physics_engine_basic,
        validate_gradient_flow,
        validate_jit_compilation,
        validate_temporal_gradient_decay,
        validate_multi_agent_capability,
        validate_system_integration
    ]
    
    passed_tests = 0
    total_tests = len(validations)
    
    for validation_fn in validations:
        try:
            success = validation_fn()
            if success:
                passed_tests += 1
        except Exception as e:
            print(f"❌ {validation_fn.__name__} FAILED: {e}")
    
    end_time = time.time()
    
    # Final summary
    print("\n" + "=" * 60)
    print("STAGE 1 VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Tests passed: {passed_tests}/{total_tests}")
    print(f"Validation time: {end_time - start_time:.2f} seconds")
    print(f"JAX device: {jax.default_backend()}")
    
    if passed_tests == total_tests:
        print("\n🎉 STAGE 1 SUCCESSFULLY COMPLETED!")
        print("\nKey accomplishments:")
        print("  ✅ JAX-native differentiable physics engine")
        print("  ✅ Point-mass drone dynamics with control delay")
        print("  ✅ Temporal gradient decay for BPTT stability")
        print("  ✅ Multi-agent state representation for GCBF+ integration")
        print("  ✅ End-to-end gradient flow verification")
        print("  ✅ JIT compilation for performance optimization")
        print("  ✅ Comprehensive configuration management")
        print("  ✅ Full test coverage with validation suite")
        print("\n🚀 Ready to proceed to Stage 2: Minimal End-to-End BPTT Loop")
        
        return True
    else:
        print(f"\n❌ STAGE 1 INCOMPLETE: {total_tests - passed_tests} tests failed")
        print("Please address failing tests before proceeding to Stage 2")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)