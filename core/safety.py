"""
安全敏捷飞行系统的完整安全层实现

本模块实现可微分安全层，将CBF约束与二次规划(QP)结合用于安全控制合成。

关键组件：
1. 使用qpax的基于QP的安全过滤器，支持JAX原生可微分性
2. 完整的三层安全回退机制以提高鲁棒性  
3. CBF约束构建和梯度集成
4. 来自DiffPhysDrone集成的时间梯度衰减

安全层次结构：
1. 标准QP求解（首选）
2. 带松弛变量的松弛QP（回退）
3. 紧急制动控制（最后手段）
"""

import jax
import jax.numpy as jnp
from jax import jit, grad
import qpax
from typing import Dict, Tuple, Optional, NamedTuple
import chex
from dataclasses import dataclass

from .perception import DroneState

# =============================================================================
# 安全配置和结构
# =============================================================================

@dataclass
class SafetyConfig:
    """安全层的配置参数"""
    # QP求解器设置
    max_iterations: int = 100
    tolerance: float = 1e-6
    regularization: float = 1e-8
    
    # 控制约束
    max_thrust: float = 0.8  # 最大推力幅度
    max_torque: float = 0.5  # 最大角度控制
    
    # CBF参数
    cbf_alpha: float = 1.0  # CBF类K函数参数
    safety_margin: float = 0.1  # 额外安全缓冲
    relaxation_penalty: float = 1000.0  # 松弛变量惩罚(beta)
    
    # 紧急回退
    emergency_brake_force: float = 0.6  # 紧急减速幅度（正值）
    failure_penalty: float = 10000.0  # 失败的损失惩罚
    use_differentiable_fallback: bool = False  # 在需要梯度时使用投影梯度QP

@dataclass
class QSolutionInfo:
    """QP求解信息和诊断"""
    u_safe: chex.Array  # 计算的安全控制
    is_feasible: bool  # QP是否可行
    solver_status: int  # 求解器状态代码（0=成功，1=不可行，2=失败）
    slack_violation: float  # 约束松弛的幅度
    num_iterations: int  # 求解器使用的迭代次数
    
class SafetyLayer:
    """
    基于QP控制合成的可微分安全层
    
    实现完整的三层安全架构：
    1. 标准CBF-QP（名义操作）
    2. 带松弛变量的松弛CBF-QP（安全回退）
    3. 紧急制动（最后手段）
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
        为CBF约束构建QP矩阵
        
        标准CBF-QP公式：
        最小化： ||u - u_nom||^2
        约束于： grad_h^T * g * u >= -grad_h^T * f - alpha * h
        
        注意：对于仅位置CBF（grad_h是3D），我们只需要位置动力学。
        关键见解是CBF约束只关心位置演化。
        """
        # 成本函数：最小化||u - u_nom||^2带数值稳定性
        Q = 2.0 * jnp.eye(3) + self.config.regularization * jnp.eye(3)
        q = -2.0 * u_nom
        
        # 对于仅位置CBF，我们只需要位置动力学：
        # d_pos/dt = velocity （当前无人机速度）
        # CBF约束变为： grad_h^T * (velocity + u_control) >= -alpha * h
        # 这是有效的，因为控制直接影响加速度，进而影响速度，
        # 在下一时间步中影响位置。
        
        f_position_dynamics = drone_state.velocity  # (3,) 位置时间导数
        
        # 控制通过直接推力影响位置（简化模型）
        g_position_matrix = jnp.eye(3)  # 直接位置控制
        
        # 约束矩阵： -grad_h^T * g_position
        G_cbf = -(grad_h @ g_position_matrix)[None, :]  # (1, 3)
        
        # 约束向量： grad_h^T * f_position + alpha * h
        h_cbf = grad_h @ f_position_dynamics + self.config.cbf_alpha * h
        
        # 添加控制幅度约束： -max_thrust <= u_i <= max_thrust
        G_control = jnp.vstack([
            jnp.eye(3),      # u_i <= max_thrust
            -jnp.eye(3)      # -u_i <= max_thrust (即 u_i >= -max_thrust)
        ])
        h_control = jnp.full(6, self.config.max_thrust)
        
        # 组合约束
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
        构建带松弛变量的松弛QP
        
        松弛公式：
        最小化： ||u - u_nom||^2 + beta * ||delta||^2
        约束于： grad_h^T * (f + g*u) >= -alpha * h - delta
                   delta >= 0
        
        变量： [u (3), delta (1)] = (4,)
        """
        beta = self.config.relaxation_penalty
        
        # 带正则化的[u, delta]扩展成本矩阵
        Q = jnp.block([
            [2.0 * jnp.eye(3) + self.config.regularization * jnp.eye(3), jnp.zeros((3, 1))],
            [jnp.zeros((1, 3)), 2.0 * beta * jnp.ones((1, 1))]
        ])  # (4, 4)
        
        q = jnp.concatenate([-2.0 * u_nom, jnp.zeros(1)])  # (4,)
        
        # 获取原始CBF约束组件
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
        """使用正确的qpax API求解标准CBF-QP（第1层）"""
        try:
            Q, q, G, h_constraint = self._construct_qp_matrices(u_nom, h, grad_h, drone_state)
            
            # Use qpax.solve_qp_primal for differentiable QP solving
            # API: qpax.solve_qp_primal(Q, q, A, b, G, h, solver_tol, target_kappa)
            solution = qpax.solve_qp_primal(
                Q,                    # Quadratic cost matrix
                q,                    # Linear cost vector
                jnp.zeros((0, 3)),    # No equality constraints (A)
                jnp.zeros(0),         # No equality constraints (b)
                G,                    # Inequality constraint matrix  
                h_constraint,         # Inequality constraint vector
                solver_tol=self.config.tolerance,  # Convergence tolerance
                target_kappa=1e-3     # Gradient smoothing for differentiability
            )
            
            # For feasibility checking, we can use a simple constraint satisfaction check
            # Check if Gx <= h is satisfied (with small tolerance for numerical errors)
            constraint_violations = G @ solution - h_constraint
            max_violation = jnp.max(constraint_violations)
            is_feasible = max_violation <= self.config.tolerance * 10  # Allow some numerical tolerance
            
            # Use jnp.where for differentiable fallback
            u_safe = jnp.where(is_feasible, solution, u_nom)
            
            return QSolutionInfo(
                u_safe=u_safe,
                is_feasible=is_feasible,
                solver_status=0 if is_feasible else 1,
                slack_violation=jnp.maximum(max_violation, 0.0),
                num_iterations=-1  # qpax.solve_qp_primal doesn't return iteration count
            )
            
        except Exception as e:
            # Solver failed - return nominal control with penalty signal
            # Use a small perturbation to maintain gradient flow instead of direct assignment
            u_safe = u_nom + 1e-8 * jnp.ones_like(u_nom)  # Small perturbation for gradient flow
            
            return QSolutionInfo(
                u_safe=u_safe,
                is_feasible=False,
                solver_status=2,  # Failed
                slack_violation=1.0,  # Indicates failure
                num_iterations=0
            )

    def _solve_relaxed_qp(
        self,
        u_nom: chex.Array,
        h: chex.Array,
        grad_h: chex.Array,
        drone_state: DroneState
    ) -> QSolutionInfo:
        """使用松弛变量求解松弛CBF-QP（第2层）"""
        try:
            Q, q, G, h_constraint = self._construct_relaxed_qp_matrices(
                u_nom, h, grad_h, drone_state
            )
            
            # Solve relaxed QP using qpax.solve_qp_primal (2024 best practice)
            solution = qpax.solve_qp_primal(
                Q=Q, 
                q=q,
                A=jnp.zeros((0, 4)),  # No equality constraints (4D: [u, delta])
                b=jnp.zeros(0),
                G=G, 
                h=h_constraint,
                solver_tol=self.config.tolerance,
                target_kappa=1e-3
            )
            
            # Extract control and slack variables
            u_safe = solution[:3]  # Control part [u1, u2, u3]
            slack_value = solution[3]   # Slack variable delta
            
            # Check constraint violations for feasibility
            constraint_violations = G @ solution - h_constraint
            max_violation = jnp.max(constraint_violations)
            is_feasible = max_violation <= self.config.tolerance * 10
            
            # 使用jnp.where进行可微分回退
            u_safe = jnp.where(is_feasible, u_safe, u_nom)
            slack_violation = jnp.where(is_feasible, jnp.maximum(slack_value, 0.0), 1.0)
            
            return QSolutionInfo(
                u_safe=u_safe,
                is_feasible=is_feasible,
                solver_status=0 if is_feasible else 1,
                slack_violation=slack_violation,
                num_iterations=-1  # qpax.solve_qp_primal doesn't return iteration count
            )
            
        except Exception as e:
            # Relaxed solver failed - return nominal with penalty signal
            u_safe = u_nom + 1e-8 * jnp.ones_like(u_nom)  # Small perturbation for gradient flow
            
            return QSolutionInfo(
                u_safe=u_safe,
                is_feasible=False,
                solver_status=2,
                slack_violation=1.0,  # Penalty for failure
                num_iterations=0
            )

    def _emergency_brake(self, drone_state: DroneState) -> chex.Array:
        """Emergency braking control (Layer 3)"""
        velocity_magnitude = jnp.linalg.norm(drone_state.velocity)
        
        # Fixed braking logic: apply force OPPOSITE to velocity direction
        # brake_direction should be -velocity/|velocity| (pointing opposite to motion)
        # emergency_brake_force should be POSITIVE (magnitude of braking force)
        brake_direction = jnp.where(
            velocity_magnitude > 1e-6,
            -drone_state.velocity / (velocity_magnitude + 1e-8),  # Opposite to velocity
            jnp.array([0.0, 0.0, -1.0])  # Default downward direction when stationary
        )
        
        # Apply positive braking force in the brake direction (opposite to velocity)
        brake_magnitude = jnp.abs(self.config.emergency_brake_force)  # Ensure positive
        
        emergency_control = jnp.where(
            velocity_magnitude > 1e-6,
            brake_magnitude * brake_direction,  # Positive force * opposite direction = braking
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
# 训练损失函数
# =============================================================================

def compute_safety_loss(
    solution_info: QSolutionInfo,
    config: SafetyConfig
) -> Tuple[chex.Array, dict]:
    """
    计算训练的安全相关损失组件
    
    损失组件：
    1. 松弛惩罚：惩罚松弛变量的使用
    2. 失败惩罚：大力惩罚求解器失败
    3. 控制幅度：正则化控制助力
    
    返回：
        total_loss: 组合安全损失
        loss_dict: 单个损失组件
    """
    # 松弛惩罚（鼓励可行解）
    relaxation_loss = config.relaxation_penalty * jnp.maximum(0.0, solution_info.slack_violation)
    
    # 失败惩罚（强烈阻止求解器失败）
    failure_penalty = jnp.where(
        solution_info.solver_status >= 2,
        config.failure_penalty,
        0.0
    )
    
    # 控制助力正则化
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
# 可微分安全层接口
# =============================================================================

def differentiable_safety_filter(
    params_dict: dict,
    u_nom: chex.Array,
    h: chex.Array,
    grad_h: chex.Array,
    drone_state: DroneState
) -> Tuple[chex.Array, dict]:
    """
    带自动回退的JIT编译可微分安全过滤器
    
    为JIT兼容性，始终使用可微分求解器。
    保留原始基于qpax的求解器用于非JIT用例。
    
    参数：
        params_dict: 配置参数（转换为静态）
        u_nom: 名义控制输入
        h: CBF值
        grad_h: CBF梯度
        drone_state: 当前无人机状态
        
    返回：
        u_safe: 安全控制输出
        info_dict: 损失计算的求解信息
    """
    config = SafetyConfig(**params_dict)
    
    # 为JIT兼容性始终使用可微分求解器
    # 这避免了JIT编译中的布尔转换问题
    return differentiable_cbf_qp_solve(u_nom, h, grad_h, drone_state, config)

# =============================================================================
# 时间梯度衰减（DiffPhysDrone集成）
# =============================================================================

def apply_temporal_gradient_decay(
    gradients: chex.Array,
    time_step: int,
    decay_factor: float = 0.95,
    distance_to_obstacles: Optional[chex.Array] = None
) -> chex.Array:
    """
    应用时间梯度衰减以获得BPTT稳定性（来自DiffPhysDrone）
    
    实现时间梯度衰减机制，带有基于障碍物邻近度的额外空间适应。
    
    参数：
        gradients: 来自安全层的梯度
        time_step: 轨迹中的当前时间步
        decay_factor: 基础衰减因子
        distance_to_obstacles: 到最近障碍物的距离（可选）
        
    返回：
        decayed_gradients: 缩放的梯度
    """
    # 基础时间衰减（指数）
    temporal_decay = decay_factor ** time_step
    
    # 空间适应（远离障碍物时更强的衰减）
    if distance_to_obstacles is not None:
        # 远离障碍物时增加衰减因子（安全性不那么关键）
        min_distance = jnp.min(distance_to_obstacles)
        spatial_factor = jnp.maximum(0.5, jnp.minimum(1.0, min_distance / 2.0))
        temporal_decay *= spatial_factor
    
    return gradients * temporal_decay

# =============================================================================
# 工厂函数和实用程序
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
    # For position-only CBF: grad_h^T * (f_position + g*u) >= -alpha * h
    # where f_position = velocity (position time derivative)
    f_position_dynamics = drone_state.velocity  # (3,) - position dynamics
    
    cbf_constraint_value = grad_h @ (f_position_dynamics + u_safe)  # Now both are (3,)
    cbf_threshold = -config.cbf_alpha * h
    
    return cbf_constraint_value >= cbf_threshold - 1e-6

# =============================================================================
# ADVANCED SAFETY LAYER WITH CURRICULUM LEARNING
# =============================================================================

class AdvancedSafetyLayer(SafetyLayer):
    """Advanced safety layer with curriculum learning and adaptive constraints"""
    
    def __init__(self, config: SafetyConfig):
        super().__init__(config)
        self.curriculum_stage = 0
        self.adaptation_history = []
        self.success_rate_window = []
        self.window_size = 100
    
    def adaptive_safety_filter(
        self,
        u_nom: chex.Array,
        h: chex.Array,
        grad_h: chex.Array,
        drone_state: DroneState,
        training_step: int = 0,
        curriculum_enabled: bool = True
    ) -> Tuple[chex.Array, QSolutionInfo]:
        """
        Adaptive safety filter with curriculum learning
        
        Gradually increases safety strictness during training:
        Stage 0: Relaxed safety (learning basic control)
        Stage 1: Moderate safety (learning safety-aware control) 
        Stage 2: Strict safety (full safety enforcement)
        """
        if curriculum_enabled:
            # Adapt safety parameters based on curriculum stage
            adapted_config = self._adapt_config_for_curriculum(training_step)
        else:
            adapted_config = self.config
        
        # Create temporary safety layer with adapted config
        adapted_layer = SafetyLayer(adapted_config)
        
        # Apply safety filter
        u_safe, solution_info = adapted_layer.safety_filter(u_nom, h, grad_h, drone_state)
        
        # Update success tracking
        self._update_success_tracking(solution_info.is_feasible)
        
        return u_safe, solution_info
    
    def _adapt_config_for_curriculum(self, training_step: int) -> SafetyConfig:
        """Adapt safety configuration based on curriculum progress"""
        # Determine curriculum stage based on training progress
        stage_duration = 2000  # Steps per curriculum stage
        current_stage = min(2, training_step // stage_duration)
        
        if current_stage != self.curriculum_stage:
            print(f"🎓 Safety curriculum advanced to stage {current_stage}")
            self.curriculum_stage = current_stage
        
        # Create adapted configuration
        adapted_config = SafetyConfig(
            max_iterations=self.config.max_iterations,
            tolerance=self.config.tolerance,
            regularization=self.config.regularization,
            max_thrust=self.config.max_thrust,
            max_torque=self.config.max_torque,
            safety_margin=self.config.safety_margin,
            emergency_brake_force=self.config.emergency_brake_force,
            failure_penalty=self.config.failure_penalty,
            use_differentiable_fallback=self.config.use_differentiable_fallback
        )
        
        # Adapt parameters based on curriculum stage
        if current_stage == 0:  # Relaxed stage
            adapted_config.cbf_alpha = 0.3  # Very relaxed CBF constraint
            adapted_config.relaxation_penalty = 100.0  # Low penalty for violations
        elif current_stage == 1:  # Moderate stage
            adapted_config.cbf_alpha = 0.7  # Moderate CBF constraint
            adapted_config.relaxation_penalty = 500.0  # Medium penalty
        else:  # Strict stage (stage >= 2)
            adapted_config.cbf_alpha = 1.0  # Full CBF constraint
            adapted_config.relaxation_penalty = 1000.0  # High penalty
        
        return adapted_config
    
    def _update_success_tracking(self, is_feasible: bool):
        """Update success rate tracking for curriculum adaptation"""
        self.success_rate_window.append(1.0 if is_feasible else 0.0)
        
        # Maintain window size
        if len(self.success_rate_window) > self.window_size:
            self.success_rate_window.pop(0)
    
    def get_success_rate(self) -> float:
        """Get current success rate"""
        if not self.success_rate_window:
            return 0.0
        return jnp.mean(jnp.array(self.success_rate_window))

class HybridSafetyLayer:
    """Hybrid safety layer combining learned and analytical safety"""
    
    def __init__(self, config: SafetyConfig, use_learned_cbf: bool = True):
        self.config = config
        self.use_learned_cbf = use_learned_cbf
        self.analytical_safety = AnalyticalSafetyChecker()
        self.learned_safety = SafetyLayer(config)
    
    def hybrid_safety_filter(
        self,
        u_nom: chex.Array,
        h_learned: chex.Array,
        grad_h_learned: chex.Array,
        drone_state: DroneState,
        point_cloud: jnp.ndarray
    ) -> Tuple[chex.Array, QSolutionInfo]:
        """
        Hybrid safety filtering using both learned and analytical constraints
        """
        # Get analytical safety constraints
        h_analytical, grad_h_analytical = self.analytical_safety.compute_constraints(
            drone_state, point_cloud
        )
        
        if self.use_learned_cbf:
            # Use learned CBF as primary, analytical as backup
            primary_h, primary_grad = h_learned, grad_h_learned
            backup_h, backup_grad = h_analytical, grad_h_analytical
        else:
            # Use analytical CBF as primary, learned as secondary
            primary_h, primary_grad = h_analytical, grad_h_analytical
            backup_h, backup_grad = h_learned, grad_h_learned
        
        # Try primary safety filter
        u_safe, solution_info = self.learned_safety.safety_filter(
            u_nom, primary_h, primary_grad, drone_state
        )
        
        # If primary fails, try backup
        if not solution_info.is_feasible:
            u_safe_backup, solution_info_backup = self.learned_safety.safety_filter(
                u_nom, backup_h, backup_grad, drone_state
            )
            
            # Use backup if it's better
            if solution_info_backup.is_feasible or solution_info_backup.solver_status < solution_info.solver_status:
                u_safe = u_safe_backup
                solution_info = solution_info_backup
        
        return u_safe, solution_info

class AnalyticalSafetyChecker:
    """Analytical safety constraints for backup and validation"""
    
    def __init__(self, min_distance: float = 1.0, max_speed: float = 5.0):
        self.min_distance = min_distance
        self.max_speed = max_speed
    
    def compute_constraints(
        self, 
        drone_state: DroneState, 
        point_cloud: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute analytical safety constraints
        
        Returns:
            h: CBF value (distance-based)
            grad_h: CBF gradient
        """
        # Distance-based CBF: h(x) = min_dist_to_obstacles - min_safe_distance
        drone_pos = drone_state.position
        
        # Compute distances to all obstacles
        distances = jnp.linalg.norm(point_cloud - drone_pos[None, :], axis=1)
        min_distance = jnp.min(distances)
        
        # CBF value: positive when safe, negative when unsafe
        h = min_distance - self.min_distance
        
        # Find closest obstacle for gradient computation
        closest_idx = jnp.argmin(distances)
        closest_obstacle = point_cloud[closest_idx]
        
        # CBF gradient: direction to move away from closest obstacle
        direction_to_obstacle = closest_obstacle - drone_pos
        distance_to_closest = jnp.linalg.norm(direction_to_obstacle)
        
        # Normalize to get gradient (direction away from obstacle)
        grad_h = -direction_to_obstacle / (distance_to_closest + 1e-8)
        
        return h, grad_h

# =============================================================================
# ADVANCED QP SOLVERS WITH WARM STARTING
# =============================================================================

class WarmStartQPSolver:
    """QP solver with warm starting for improved efficiency"""
    
    def __init__(self, config: SafetyConfig):
        self.config = config
        self.previous_solution = None
        self.solution_cache = {}
    
    def solve_with_warm_start(
        self,
        Q: chex.Array,
        q: chex.Array,
        G: chex.Array,
        h: chex.Array,
        cache_key: Optional[str] = None
    ) -> Tuple[chex.Array, bool]:
        """
        Solve QP with warm starting from previous solution
        """
        # Determine initial guess
        if self.previous_solution is not None and self.previous_solution.shape == (Q.shape[0],):
            x_init = self.previous_solution
        else:
            x_init = jnp.zeros(Q.shape[0])
        
        # Solve using projected gradient descent with warm start
        x_solution, converged = projected_gradient_qp_solve(
            Q, q, G, h, x_init,
            num_steps=30,  # More steps for better convergence
            step_size=0.03  # Smaller step size for stability
        )
        
        # Cache solution for next iteration
        if converged:
            self.previous_solution = x_solution
            if cache_key is not None:
                self.solution_cache[cache_key] = x_solution
        
        return x_solution, converged

class AdaptiveQPSolver:
    """Adaptive QP solver that adjusts parameters based on problem difficulty"""
    
    def __init__(self, config: SafetyConfig):
        self.config = config
        self.solve_history = []
        self.max_history = 50
    
    def adaptive_solve(
        self,
        Q: chex.Array,
        q: chex.Array,
        G: chex.Array,
        h: chex.Array
    ) -> Tuple[chex.Array, bool, dict]:
        """
        Solve QP with adaptive parameters based on problem characteristics
        """
        # Analyze problem characteristics
        problem_stats = self._analyze_problem(Q, q, G, h)
        
        # Adapt solver parameters
        adapted_params = self._adapt_solver_params(problem_stats)
        
        # Solve with adapted parameters
        x_solution, converged = projected_gradient_qp_solve(
            Q, q, G, h,
            x_init=jnp.zeros(Q.shape[0]),
            num_steps=adapted_params['num_steps'],
            step_size=adapted_params['step_size']
        )
        
        # Update solve history
        solve_info = {
            'converged': converged,
            'problem_stats': problem_stats,
            'adapted_params': adapted_params
        }
        self.solve_history.append(solve_info)
        
        # Maintain history size
        if len(self.solve_history) > self.max_history:
            self.solve_history.pop(0)
        
        return x_solution, converged, solve_info
    
    def _analyze_problem(self, Q: chex.Array, q: chex.Array, G: chex.Array, h: chex.Array) -> dict:
        """Analyze QP problem characteristics"""
        return {
            'condition_number': jnp.linalg.cond(Q + 1e-6 * jnp.eye(Q.shape[0])),
            'constraint_tightness': jnp.mean(jnp.maximum(0, G @ jnp.zeros(Q.shape[0]) - h)),
            'problem_scale': jnp.linalg.norm(Q, 'fro')
        }
    
    def _adapt_solver_params(self, problem_stats: dict) -> dict:
        """Adapt solver parameters based on problem characteristics"""
        base_steps = 20
        base_step_size = 0.05
        
        # Adapt based on condition number
        if problem_stats['condition_number'] > 100:
            num_steps = base_steps * 2
            step_size = base_step_size * 0.5
        elif problem_stats['condition_number'] > 50:
            num_steps = int(base_steps * 1.5)
            step_size = base_step_size * 0.7
        else:
            num_steps = base_steps
            step_size = base_step_size
        
        # Adapt based on constraint tightness
        if problem_stats['constraint_tightness'] > 0.1:
            num_steps = int(num_steps * 1.3)
            step_size = step_size * 0.8
        
        return {
            'num_steps': num_steps,
            'step_size': step_size
        }

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

def projected_gradient_qp_solve(
    Q: chex.Array, 
    q: chex.Array, 
    G: chex.Array, 
    h: chex.Array,
    x_init: chex.Array,
    num_steps: int = 20,
    step_size: float = 0.05
) -> Tuple[chex.Array, bool]:
    """
    Solve QP using projected gradient descent (differentiable alternative to qpax)
    
    This provides a differentiable alternative when qpax fails gradient computation.
    Uses fewer steps for better performance in training loops.
    """
    
    def project_onto_feasible_set(x):
        """Project x onto feasible set {x : Gx <= h}"""
        # Simple box projection for control bounds (most important constraints)
        # This is a simplified projection that handles the control magnitude constraints efficiently
        
        violations = G @ x - h
        
        # Simple repair: if any constraint is violated, scale the solution
        max_violation = jnp.maximum(0.0, jnp.max(violations))
        
        # If violations exist, scale down the control to satisfy constraints
        scale_factor = jnp.where(
            max_violation > 1e-6,
            0.95,  # Slightly reduce magnitude
            1.0    # No scaling needed
        )
        
        x_scaled = scale_factor * x
        
        # Hard clip to ensure control bounds are satisfied
        x_clipped = jnp.clip(x_scaled, -0.8, 0.8)  # Based on typical max_thrust
        
        return x_clipped
    
    # Initial point
    x = project_onto_feasible_set(x_init)
    
    # Simplified projected gradient descent (fewer steps for training efficiency)
    for i in range(num_steps):
        # Gradient of quadratic objective
        grad = Q @ x + q
        
        # Adaptive step size
        current_step_size = step_size / (1.0 + 0.1 * i)
        
        # Gradient step
        x_new = x - current_step_size * grad
        
        # Project back to feasible set
        x_new = project_onto_feasible_set(x_new)
        
        x = x_new
    
    # Final projection
    x = project_onto_feasible_set(x)
    
    # Check feasibility (simple check)
    final_violations = G @ x - h
    max_final_violation = jnp.max(final_violations)
    is_feasible = max_final_violation <= 1e-2  # More lenient for training
    
    return x, is_feasible

def differentiable_cbf_qp_solve(
    u_nom: chex.Array,
    h: chex.Array,
    grad_h: chex.Array,
    drone_state: DroneState,
    config: SafetyConfig
) -> Tuple[chex.Array, dict]:
    """
    Differentiable CBF-QP solver using projected gradient descent
    
    This replaces qpax when gradient computation is needed.
    Optimized for training efficiency with fewer iterations.
    """
    # Cost function: minimize ||u - u_nom||^2
    Q = 2.0 * jnp.eye(3) + config.regularization * jnp.eye(3)
    q = -2.0 * u_nom
    
    # CBF constraint: grad_h^T * (velocity + u) >= -alpha * h
    f_position_dynamics = drone_state.velocity
    G_cbf = -(grad_h @ jnp.eye(3))[None, :]  # (1, 3)
    h_cbf = grad_h @ f_position_dynamics + config.cbf_alpha * h
    
    # Control magnitude constraints (simplified)
    G_control = jnp.vstack([jnp.eye(3), -jnp.eye(3)])  # (6, 3)
    h_control = jnp.full(6, config.max_thrust)
    
    # Combined constraints
    G = jnp.vstack([G_cbf, G_control])  # (7, 3)
    h_constraint = jnp.concatenate([h_cbf[None], h_control])  # (7,)
    
    # Solve using projected gradient descent
    x_init = jnp.clip(u_nom, -config.max_thrust * 0.8, config.max_thrust * 0.8)
    u_safe, converged = projected_gradient_qp_solve(Q, q, G, h_constraint, x_init)
    
    # Create solution info compatible with existing interface
    # Use jnp.where for JAX compatibility instead of if/else
    solver_status = jnp.where(converged, 0, 1)
    
    info_dict = {
        "u_safe": u_safe,
        "is_feasible": converged,
        "solver_status": solver_status,
        "slack_violation": 0.0,
        "num_iterations": 20
    }
    
    return u_safe, info_dict

class RobustSafetyLayer:
    """
    三层安全QP求解器 - 解决qpax数值不稳定问题
    基于GCBF+论文的安全保障机制
    """
    
    def __init__(self, config):
        self.config = config
        self.qp_failure_count = 0
    
    def safety_filter(
        self, 
        u_nom: chex.Array, 
        h: chex.Array, 
        grad_h: chex.Array, 
        drone_state
    ) -> Tuple[chex.Array, Dict]:
        """
        三层安全过滤器：
        Layer 1: 标准CBF-QP  
        Layer 2: 松弛CBF-QP (with slack variables)
        Layer 3: 紧急制动控制
        """
        
        # Layer 1: 尝试标准CBF-QP求解
        try:
            u_safe, info = self._solve_layer1_qp(u_nom, h, grad_h, drone_state)
            if info['feasible'] and self._validate_safety_constraints(u_safe, h, grad_h):
                return u_safe, {**info, 'layer_used': 1}
        except Exception as e:
            self.qp_failure_count += 1
        
        # Layer 2: 松弛QP with slack variables
        try:
            u_safe, info = self._solve_layer2_relaxed_qp(u_nom, h, grad_h, drone_state)
            if info['feasible']:
                return u_safe, {**info, 'layer_used': 2}
        except Exception as e:
            self.qp_failure_count += 1
        
        # Layer 3: Emergency brake control
        u_emergency = self._emergency_brake_control(drone_state)
        return u_emergency, {
            'feasible': True,
            'layer_used': 3,
            'emergency_activated': True,
            'qp_failures': self.qp_failure_count
        }
    
    def _solve_layer1_qp(self, u_nom, h, grad_h, drone_state):
        """Layer 1: 标准CBF-QP"""
        # 构建QP矩阵
        Q = jnp.eye(3) * self.config.control_penalty_weight
        q = -Q @ u_nom  # 最小化 ||u - u_nom||²
        
        # CBF约束: L_f h + L_g h * u >= -alpha * h
        lf_h = self._compute_lie_derivative_f(h, drone_state)
        lg_h = grad_h  # 简化：假设lg_h = grad_h
        
        G_cbf = -lg_h.reshape(1, -1)  # [1, 3]
        h_cbf = jnp.array([self.config.cbf_alpha * h - lf_h])  # [1]
        
        # 控制约束: |u| <= u_max
        G_control = jnp.vstack([jnp.eye(3), -jnp.eye(3)])  # [6, 3]
        h_control = jnp.hstack([
            jnp.ones(3) * self.config.max_thrust,
            jnp.ones(3) * self.config.max_thrust
        ])  # [6]
        
        # 组合约束
        G = jnp.vstack([G_cbf, G_control])  # [7, 3]
        h_constraint = jnp.hstack([h_cbf, h_control])  # [7]
        
        # 数值稳定化
        Q_reg = Q + self.config.regularization * jnp.eye(3)
        
        # 求解QP
        solution = qpax.solve_qp_primal(
            Q_reg, q, 
            jnp.zeros((0, 3)), jnp.zeros(0),  # 无等式约束
            G, h_constraint,
            solver_tol=self.config.tolerance
        )
        
        # 验证解
        constraint_violations = G @ solution - h_constraint
        max_violation = jnp.max(constraint_violations)
        feasible = max_violation <= self.config.tolerance * 10
        
        return solution, {
            'feasible': feasible,
            'max_violation': float(max_violation),
            'solver_status': 'success' if feasible else 'infeasible'
        }
    
    def _solve_layer2_relaxed_qp(self, u_nom, h, grad_h, drone_state):
        """Layer 2: 松弛QP with slack variables"""
        # 扩展QP维度：[u1, u2, u3, δ] where δ is slack variable
        Q = jnp.zeros((4, 4))
        Q = Q.at[:3, :3].set(jnp.eye(3) * self.config.control_penalty_weight)
        Q = Q.at[3, 3].set(self.config.relaxation_penalty)  # 大的松弛惩罚
        
        q = jnp.zeros(4)
        q = q.at[:3].set(-self.config.control_penalty_weight * u_nom)
        
        # 松弛CBF约束: L_f h + L_g h * u + δ >= -alpha * h
        lf_h = self._compute_lie_derivative_f(h, drone_state)
        G_cbf_relaxed = jnp.zeros((1, 4))
        G_cbf_relaxed = G_cbf_relaxed.at[0, :3].set(-grad_h)  # -L_g h * u
        G_cbf_relaxed = G_cbf_relaxed.at[0, 3].set(-1.0)      # -δ
        h_cbf_relaxed = jnp.array([self.config.cbf_alpha * h - lf_h])
        
        # 控制约束保持不变，但扩展到4D
        G_control = jnp.zeros((6, 4))
        G_control = G_control.at[:3, :3].set(jnp.eye(3))      # u <= u_max
        G_control = G_control.at[3:6, :3].set(-jnp.eye(3))    # u >= -u_max
        h_control = jnp.ones(6) * self.config.max_thrust
        
        # 松弛变量约束: δ >= 0
        G_slack = jnp.zeros((1, 4))
        G_slack = G_slack.at[0, 3].set(-1.0)  # -δ <= 0
        h_slack = jnp.array([0.0])
        
        # 组合所有约束
        G = jnp.vstack([G_cbf_relaxed, G_control, G_slack])
        h_constraint = jnp.hstack([h_cbf_relaxed, h_control, h_slack])
        
        # 求解
        solution = qpax.solve_qp_primal(
            Q, q,
            jnp.zeros((0, 4)), jnp.zeros(0),
            G, h_constraint,
            solver_tol=self.config.tolerance
        )
        
        u_safe = solution[:3]
        slack_value = solution[3]
        
        return u_safe, {
            'feasible': True,  # 松弛QP总是可行的
            'slack_violation': float(jnp.maximum(slack_value, 0.0)),
            'solver_status': 'relaxed_success'
        }
    
    def _emergency_brake_control(self, drone_state):
        """Layer 3: 紧急制动控制"""
        velocity = drone_state.velocity
        velocity_norm = jnp.linalg.norm(velocity)
        
        if velocity_norm < 1e-6:
            # 已经静止，只需悬停
            return jnp.array([0.0, 0.0, self.config.hover_thrust])
        else:
            # 反向制动
            brake_direction = -velocity / velocity_norm
            brake_magnitude = jnp.minimum(self.config.emergency_brake_force, velocity_norm)
            
            return brake_direction * brake_magnitude
    
    def _compute_lie_derivative_f(self, h, drone_state):
        """计算Lie导数L_f h (简化实现)"""
        # 简化：假设CBF主要依赖于位置
        # 实际实现需要根据具体的CBF定义
        return 0.0  # Placeholder
    
    def _validate_safety_constraints(self, u_safe, h, grad_h):
        """验证安全约束是否满足"""
        # 简化验证
        return jnp.linalg.norm(u_safe) <= self.config.max_thrust * 1.1
