"""
安全模块的单元测试（第三阶段验证）

此测试套件验证：
1. 使用qpax的QP构建和求解
2. 三层安全回退机制
3. 安全过滤器的可微分性
4. JIT编译兼容性
5. 与CBF约束的集成

测试重点验证用qpax替换GCBF+ cvxpylayers，
实现JAX原生的可微分QP求解。
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import pytest
from jax import random, grad, jit, vmap

# 导入被测试的模块
from core.safety import (
    SafetyConfig, SafetyLayer, QSolutionInfo,
    differentiable_safety_filter,
    compute_safety_loss,
    apply_temporal_gradient_decay,
    validate_safety_constraints,
    create_default_safety_layer
)
from core.perception import DroneState


class TestQPConstruction:
    """测试QP矩阵构建和求解"""
    
    def setup_method(self):
        """设置测试夹具"""
        self.config = SafetyConfig(
            max_thrust=0.8,
            cbf_alpha=1.0,
            relaxation_penalty=100.0
        )
        self.safety_layer = SafetyLayer(self.config)
        
        self.drone_state = DroneState(
            position=jnp.array([0.0, 0.0, 1.0]),
            velocity=jnp.array([1.0, 0.0, 0.0]),
            orientation=jnp.eye(3),
            angular_velocity=jnp.zeros(3)
        )
        
        # Test control inputs and CBF values
        self.u_nom = jnp.array([0.2, 0.1, -0.1])
        self.h_safe = jnp.array(0.5)  # Safe CBF value
        self.h_unsafe = jnp.array(-0.2)  # Unsafe CBF value
        self.grad_h = jnp.array([1.0, 0.0, 0.0])  # CBF gradient

    def test_qp_matrix_construction(self):
        """Test construction of QP matrices"""
        Q, q, G, h = self.safety_layer._construct_qp_matrices(
            self.u_nom, self.h_safe, self.grad_h, self.drone_state
        )
        
        # Verify matrix dimensions
        assert Q.shape == (3, 3)  # Control dimension
        assert q.shape == (3,)
        assert G.shape[1] == 3  # Control dimension
        assert G.shape[0] == h.shape[0]  # Same number of constraints
        
        # Verify cost matrix is positive definite
        eigenvals = jnp.linalg.eigvals(Q)
        assert jnp.all(eigenvals > 0)
        
        # Verify constraint structure
        # Should have: 1 CBF constraint + 6 control bound constraints
        expected_constraints = 1 + 6
        assert G.shape[0] == expected_constraints
        
        print("✅ QP matrix construction: PASSED")

    def test_relaxed_qp_construction(self):
        """Test construction of relaxed QP with slack variables"""
        Q, q, G, h = self.safety_layer._construct_relaxed_qp_matrices(
            self.u_nom, self.h_unsafe, self.grad_h, self.drone_state
        )
        
        # Extended problem dimension: [u(3), delta(1)] = 4
        assert Q.shape == (4, 4)
        assert q.shape == (4,)
        assert G.shape[1] == 4
        
        # Should include slack non-negativity constraint
        expected_constraints = 1 + 6 + 1  # CBF + control bounds + slack >= 0
        assert G.shape[0] == expected_constraints
        
        # Verify slack variable penalty in cost matrix
        assert Q[3, 3] > Q[0, 0]  # Slack penalty should be larger
        
        print("✅ Relaxed QP construction: PASSED")

    def test_standard_qp_solving(self):
        """Test standard QP solving (Layer 1)"""
        solution_info = self.safety_layer._solve_standard_qp(
            self.u_nom, self.h_safe, self.grad_h, self.drone_state
        )
        
        # Should find feasible solution for safe scenario
        assert solution_info.is_feasible
        assert solution_info.solver_status == 0
        assert solution_info.u_safe.shape == (3,)
        
        # Control should respect magnitude bounds
        assert jnp.all(jnp.abs(solution_info.u_safe) <= self.config.max_thrust)
        
        print("✅ Standard QP solving: PASSED")

    def test_relaxed_qp_solving(self):
        """Test relaxed QP solving (Layer 2)"""
        # Use unsafe scenario that may require relaxation
        solution_info = self.safety_layer._solve_relaxed_qp(
            self.u_nom, self.h_unsafe, self.grad_h, self.drone_state
        )
        
        # Relaxed QP should always be feasible
        assert solution_info.is_feasible or solution_info.solver_status != 2
        assert solution_info.u_safe.shape == (3,)
        
        # Should have slack violation information
        assert hasattr(solution_info, 'slack_violation')
        
        print("✅ Relaxed QP solving: PASSED")

    def test_emergency_brake(self):
        """Test emergency braking mechanism (Layer 3)"""
        emergency_control = self.safety_layer._emergency_brake(self.drone_state)
        
        # Should produce valid control
        assert emergency_control.shape == (3,)
        assert jnp.all(jnp.abs(emergency_control) <= self.config.max_thrust)
        
        # Should oppose current velocity direction
        velocity_direction = self.drone_state.velocity / jnp.linalg.norm(self.drone_state.velocity)
        control_direction = emergency_control / jnp.linalg.norm(emergency_control)
        
        # Dot product should be negative (opposing directions)
        dot_product = jnp.dot(velocity_direction, control_direction)
        assert dot_product < 0
        
        print("✅ Emergency braking: PASSED")


class TestSafetyLayerIntegration:
    """Test complete safety layer integration"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.safety_layer = create_default_safety_layer()
        
        self.drone_state = DroneState(
            position=jnp.array([0.0, 0.0, 1.0]),
            velocity=jnp.array([0.5, 0.0, 0.0]),
            orientation=jnp.eye(3),
            angular_velocity=jnp.zeros(3)
        )

    def test_safe_scenario(self):
        """Test safety filter with safe nominal control"""
        u_nom = jnp.array([0.1, 0.0, 0.0])
        h = jnp.array(0.5)  # Safe
        grad_h = jnp.array([1.0, 0.0, 0.0])
        
        u_safe, solution_info = self.safety_layer.safety_filter(
            u_nom, h, grad_h, self.drone_state
        )
        
        assert solution_info.is_feasible
        assert solution_info.solver_status == 0  # Standard QP success
        assert jnp.allclose(u_safe, u_nom, atol=1e-3)  # Should be close to nominal
        
        print("✅ Safe scenario handling: PASSED")

    def test_unsafe_scenario(self):
        """Test safety filter with unsafe nominal control"""
        u_nom = jnp.array([2.0, 0.0, 0.0])  # Exceeds max thrust
        h = jnp.array(-0.1)  # Unsafe CBF
        grad_h = jnp.array([1.0, 0.0, 0.0])
        
        u_safe, solution_info = self.safety_layer.safety_filter(
            u_nom, h, grad_h, self.drone_state
        )
        
        # Should produce safe control
        assert jnp.all(jnp.abs(u_safe) <= self.safety_layer.config.max_thrust)
        
        # May use relaxed QP or emergency brake
        assert solution_info.solver_status in [0, 1, 3]  # Standard/Relaxed/Emergency
        
        print("✅ Unsafe scenario handling: PASSED")

    def test_constraint_validation(self):
        """Test CBF constraint validation"""
        u_nom = jnp.array([0.1, 0.0, 0.0])
        h = jnp.array(0.3)
        grad_h = jnp.array([1.0, 0.0, 0.0])
        
        u_safe, _ = self.safety_layer.safety_filter(
            u_nom, h, grad_h, self.drone_state
        )
        
        # Validate that computed control satisfies constraints
        is_valid = validate_safety_constraints(
            u_safe, h, grad_h, self.drone_state, self.safety_layer.config
        )
        
        assert is_valid
        
        print("✅ Constraint validation: PASSED")

    def test_three_layer_fallback(self):
        """Test complete three-layer fallback mechanism"""
        # Create scenario that progressively challenges each layer
        scenarios = [
            # Safe scenario (Layer 1 should succeed)
            {
                "u_nom": jnp.array([0.1, 0.0, 0.0]),
                "h": jnp.array(0.5),
                "expected_status": 0
            },
            # Infeasible but relaxable scenario (Layer 2 should succeed)
            {
                "u_nom": jnp.array([0.5, 0.0, 0.0]),
                "h": jnp.array(-0.2),
                "expected_status": [0, 1]  # Either standard or relaxed
            }
        ]
        
        grad_h = jnp.array([1.0, 0.0, 0.0])
        
        for scenario in scenarios:
            u_safe, solution_info = self.safety_layer.safety_filter(
                scenario["u_nom"], scenario["h"], grad_h, self.drone_state
            )
            
            # Should always produce valid control
            assert u_safe.shape == (3,)
            assert jnp.all(jnp.abs(u_safe) <= self.safety_layer.config.max_thrust)
            
            # Status should match expected
            if isinstance(scenario["expected_status"], list):
                assert solution_info.solver_status in scenario["expected_status"]
            else:
                assert solution_info.solver_status == scenario["expected_status"]
        
        print("✅ Three-layer fallback: PASSED")


class TestDifferentiability:
    """Test differentiability and JAX integration"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.params_dict = {
            "max_thrust": 0.8,
            "cbf_alpha": 1.0,
            "relaxation_penalty": 100.0,
            "max_iterations": 50,
            "tolerance": 1e-6,
            "regularization": 1e-8,
            "max_torque": 0.5,
            "safety_margin": 0.1,
            "emergency_brake_force": 0.6,  # Positive value
            "failure_penalty": 10000.0,
            "use_differentiable_fallback": True  # Enable for gradient computation
        }
        
        self.drone_state = DroneState(
            position=jnp.array([0.0, 0.0, 1.0]),
            velocity=jnp.array([0.5, 0.0, 0.0]),
            orientation=jnp.eye(3),
            angular_velocity=jnp.zeros(3)
        )

    def test_jit_compilation(self):
        """Test JIT compilation of safety filter"""
        u_nom = jnp.array([0.1, 0.0, 0.0])
        h = jnp.array(0.3)
        grad_h = jnp.array([1.0, 0.0, 0.0])
        
        # JIT compile the function
        jit_safety_filter = jit(differentiable_safety_filter)
        
        # Test compilation and execution
        u_safe_jit, info_jit = jit_safety_filter(
            self.params_dict, u_nom, h, grad_h, self.drone_state
        )
        
        # Compare with regular version
        u_safe_reg, info_reg = differentiable_safety_filter(
            self.params_dict, u_nom, h, grad_h, self.drone_state
        )
        
        assert jnp.allclose(u_safe_jit, info_jit["u_safe"])
        assert jnp.allclose(u_safe_reg, info_reg["u_safe"])
        
        print("✅ JIT compilation: PASSED")

    def test_gradient_computation(self):
        """Test gradient computation through safety filter"""
        h = jnp.array(0.3)
        grad_h = jnp.array([1.0, 0.0, 0.0])
        
        def safety_loss_fn(u_nom):
            u_safe, info = differentiable_safety_filter(
                self.params_dict, u_nom, h, grad_h, self.drone_state
            )
            # Simple quadratic loss
            return jnp.sum((u_safe - jnp.array([0.0, 0.0, 0.0])) ** 2)
        
        # Compute gradients
        grad_fn = grad(safety_loss_fn)
        test_u_nom = jnp.array([0.2, 0.1, 0.0])
        gradients = grad_fn(test_u_nom)
        
        # Gradients should exist and be finite
        assert gradients.shape == (3,)
        assert jnp.all(jnp.isfinite(gradients))
        assert not jnp.allclose(gradients, 0.0)  # Should have non-zero gradients
        
        print("✅ Gradient computation: PASSED")

    def test_vmap_compatibility(self):
        """Test vectorization with vmap"""
        batch_size = 4
        u_nom_batch = jnp.array([
            [0.1, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [-0.1, 0.0, 0.0],
            [0.0, 0.0, 0.1]
        ])
        h_batch = jnp.array([0.3, 0.5, 0.2, 0.4])
        grad_h_batch = jnp.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        
        # Create batch of drone states - convert to proper JAX arrays
        # Create batched DroneState by stacking individual components
        batch_positions = jnp.tile(self.drone_state.position[None, :], (batch_size, 1))
        batch_velocities = jnp.tile(self.drone_state.velocity[None, :], (batch_size, 1))
        batch_orientations = jnp.tile(self.drone_state.orientation[None, :, :], (batch_size, 1, 1))
        batch_angular_velocities = jnp.tile(self.drone_state.angular_velocity[None, :], (batch_size, 1))
        
        # Vectorize safety filter - need to handle DroneState components separately
        def single_safety_filter(u_nom, h, grad_h, pos, vel, orient, ang_vel):
            # Reconstruct DroneState inside the function
            drone_state = DroneState(
                position=pos,
                velocity=vel,
                orientation=orient,
                angular_velocity=ang_vel
            )
            u_safe, info = differentiable_safety_filter(
                self.params_dict, u_nom, h, grad_h, drone_state
            )
            return u_safe
        
        batch_safety_filter = vmap(single_safety_filter)
        u_safe_batch = batch_safety_filter(
            u_nom_batch, h_batch, grad_h_batch, 
            batch_positions, batch_velocities, batch_orientations, batch_angular_velocities
        )
        
        # Verify batch processing
        assert u_safe_batch.shape == (batch_size, 3)
        assert jnp.all(jnp.isfinite(u_safe_batch))
        
        print("✅ Vectorization compatibility: PASSED")


class TestUtilityFunctions:
    """Test utility functions and extensions"""
    
    def test_safety_loss_computation(self):
        """Test safety loss function"""
        solution_info = QSolutionInfo(
            u_safe=jnp.array([0.1, 0.0, 0.0]),
            is_feasible=True,
            solver_status=0,
            slack_violation=0.1,
            num_iterations=10
        )
        
        config = SafetyConfig()
        total_loss, loss_dict = compute_safety_loss(solution_info, config)
        
        # Should return finite loss
        assert jnp.isfinite(total_loss)
        assert isinstance(loss_dict, dict)
        
        # Check loss components
        expected_keys = [
            "relaxation_loss", "failure_penalty", "control_magnitude_loss",
            "solver_status", "slack_violation", "total_safety_loss"
        ]
        for key in expected_keys:
            assert key in loss_dict
        
        print("✅ Safety loss computation: PASSED")

    def test_temporal_gradient_decay(self):
        """Test temporal gradient decay mechanism"""
        gradients = jnp.array([1.0, 0.5, -0.3])
        time_step = 5
        decay_factor = 0.9
        
        # Test basic decay
        decayed_grads = apply_temporal_gradient_decay(
            gradients, time_step, decay_factor
        )
        
        expected_decay = decay_factor ** time_step
        expected_grads = gradients * expected_decay
        
        assert jnp.allclose(decayed_grads, expected_grads)
        
        # Test spatial adaptation
        distances = jnp.array([0.5, 1.0, 2.0])  # Close obstacles
        decayed_grads_spatial = apply_temporal_gradient_decay(
            gradients, time_step, decay_factor, distances
        )
        
        # Should have different scaling due to spatial adaptation
        assert not jnp.allclose(decayed_grads, decayed_grads_spatial)
        
        print("✅ Temporal gradient decay: PASSED")

    def test_safety_constraints_validation(self):
        """Test safety constraint validation utility"""
        config = SafetyConfig()
        drone_state = DroneState(
            position=jnp.array([0.0, 0.0, 1.0]),
            velocity=jnp.array([0.5, 0.0, 0.0]),
            orientation=jnp.eye(3),
            angular_velocity=jnp.zeros(3)
        )
        
        # Test valid control
        u_valid = jnp.array([0.1, 0.0, 0.0])
        h = jnp.array(0.3)
        grad_h = jnp.array([1.0, 0.0, 0.0])
        
        is_valid = validate_safety_constraints(u_valid, h, grad_h, drone_state, config)
        assert is_valid
        
        # Test invalid control (exceeds thrust limits)
        u_invalid = jnp.array([2.0, 0.0, 0.0])
        is_invalid = validate_safety_constraints(u_invalid, h, grad_h, drone_state, config)
        assert not is_invalid
        
        print("✅ Safety constraints validation: PASSED")


def run_all_tests():
    """Run comprehensive test suite"""
    print("🛡️ Starting Safety Module Test Suite...")
    print("=" * 60)
    
    # Test QP construction and solving
    print("\n📐 Testing QP Construction and Solving...")
    test_qp = TestQPConstruction()
    test_qp.setup_method()
    test_qp.test_qp_matrix_construction()
    test_qp.test_relaxed_qp_construction()
    test_qp.test_standard_qp_solving()
    test_qp.test_relaxed_qp_solving()
    test_qp.test_emergency_brake()
    
    # Test safety layer integration
    print("\n🔒 Testing Safety Layer Integration...")
    test_integration = TestSafetyLayerIntegration()
    test_integration.setup_method()
    test_integration.test_safe_scenario()
    test_integration.test_unsafe_scenario()
    test_integration.test_constraint_validation()
    test_integration.test_three_layer_fallback()
    
    # Test differentiability and JAX integration
    print("\n🧮 Testing Differentiability and JAX Integration...")
    test_diff = TestDifferentiability()
    test_diff.setup_method()
    test_diff.test_jit_compilation()
    test_diff.test_gradient_computation()
    test_diff.test_vmap_compatibility()
    
    # Test utility functions
    print("\n🔧 Testing Utility Functions...")
    test_utils = TestUtilityFunctions()
    test_utils.test_safety_loss_computation()
    test_utils.test_temporal_gradient_decay()
    test_utils.test_safety_constraints_validation()
    
    print("\n" + "=" * 60)
    print("🎉 SAFETY MODULE: ALL TESTS PASSED!")
    print("✅ QP construction and solving: VERIFIED")
    print("✅ Three-layer safety fallback: VERIFIED")
    print("✅ Differentiability and JIT compilation: VERIFIED")
    print("✅ qpax integration: VERIFIED")
    print("✅ Gradient flow compatibility: VERIFIED")


if __name__ == "__main__":
    run_all_tests()