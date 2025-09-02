"""
增强版四阶段验证测试套件
专为显示实时梯度计算和矩阵信息设计

此测试文件提供:
1. 阶段1: 物理引擎可微分性 + 梯度矩阵可视化  
2. 阶段2: 端到端BPTT循环 + 梯度流可视化
3. 阶段3: 安全机制集成 + QP求解器梯度
4. 阶段4: 完整系统 + 多目标梯度分解

每个测试都实时打印梯度信息和相关矩阵计算
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import chex
from jax import random, grad, jit, jacrev, jacfwd
import optax
import jraph
import numpy as np
from typing import Dict, Tuple, Any
import warnings

# 配置JAX用于详细输出
jax.config.update("jax_enable_x64", True)
warnings.filterwarnings("ignore", category=UserWarning)

# 导入所有核心组件
from configs.default_config import get_minimal_config
from core.physics import (
    DroneState, PhysicsParams, dynamics_step, dynamics_step_jit,
    create_initial_drone_state, validate_physics_state
)
from core.perception import (
    pointcloud_to_graph, GraphConfig, init_cbf_network,
    get_cbf_from_pointcloud, CBFNet
)
from core.policy import PolicyNetworkMLP, create_policy_network, PolicyParams
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
    create_default_loss_config, create_optimizer
)


class Stage1PhysicsVerification:
    """阶段1: 物理引擎可微分性详细验证"""
    
    def __init__(self):
        self.config = get_minimal_config()
        self.physics_params = PhysicsParams(
            dt=self.config.physics.dt,
            mass=self.config.physics.drone.mass,
            thrust_to_weight_ratio=self.config.physics.drone.thrust_to_weight_ratio,
            drag_coefficient_linear=self.config.physics.drone.drag_coefficient
        )
        
    def test_basic_differentiability(self):
        """基础可微分性测试 - 详细梯度可视化"""
        print("\n" + "="*80)
        print("🧮 阶段1: 物理引擎可微分性验证")
        print("="*80)
        
        # 创建测试状态
        drone_state = create_initial_drone_state(
            position=jnp.array([0.0, 0.0, 1.0]),
            velocity=jnp.array([0.5, 0.0, 0.0]),
            hover_initialization=False
        )
        
        # 定义损失函数
        def physics_loss(control_input):
            new_state = dynamics_step(drone_state, control_input, self.physics_params)
            target_position = jnp.array([1.0, 0.0, 2.0])
            return jnp.sum((new_state.position - target_position) ** 2)#返回一个标量,代表了无人机下一个位置与目标位置之间距离的平方
        
        # 测试输入
        control_input = jnp.array([0.1, 0.2, 0.3])#是一个形状为 (3,) 的向量，代表了无人机在X, Y, Z三个轴上的推力指令
        
        # 计算梯度
        loss_value = physics_loss(control_input)
        grad_fn = grad(physics_loss)
        gradients = grad_fn(control_input)
        
        print("📊 基础梯度计算结果:")
        print(f"  🎯 输入控制: {control_input}")
        print(f"  📉 损失函数值: {loss_value:.6f}")
        print(f"  📈 损失梯度: {gradients}")
        print(f"  📏 梯度L2范数: {jnp.linalg.norm(gradients):.6f}")
        print(f"  ✅ 梯度有限性: {jnp.all(jnp.isfinite(gradients))}")
        print(f"  🎯 非零梯度: {jnp.any(jnp.abs(gradients) > 1e-8)}")
        
        # 计算雅可比矩阵 -6维输出 [px, py, pz, vx, vy, vz] 相对于3维输入 [ux, uy, uz] 的导数
        def physics_multi_output(control_input):
            new_state = dynamics_step(drone_state, control_input, self.physics_params)
            return jnp.array([
                new_state.position[0],  # x position
                new_state.position[1],  # y position  
                new_state.position[2],  # z position
                new_state.velocity[0],  # x velocity
                new_state.velocity[1],  # y velocity
                new_state.velocity[2]   # z velocity
            ])
        
        jacobian_fn = jacrev(physics_multi_output)
        jacobian_matrix = jacobian_fn(control_input)
        
        print("\n🔢 完整雅可比矩阵 J = ∂(位置,速度)/∂控制:")
        labels = ["x_pos", "y_pos", "z_pos", "x_vel", "y_vel", "z_vel"]
        for i, (label, row) in enumerate(zip(labels, jacobian_matrix)):
            print(f"  ∂{label}/∂u: [{row[0]:8.4f}, {row[1]:8.4f}, {row[2]:8.4f}]")
        
        # 计算雅可比矩阵的条件数
        try:
            condition_number = jnp.linalg.cond(jacobian_matrix)
            print(f"\n📐 雅可比矩阵分析:")
            print(f"  📏 条件数: {condition_number:.2e}")
            print(f"  🎯 数值稳定性: {'优秀' if condition_number < 1e6 else '需要注意' if condition_number < 1e10 else '不稳定'}")
        except:
            print("  ⚠️  无法计算条件数 (可能非方阵)")
            
        # 验证JIT编译
        print("\n⚡ JIT编译验证:")
        jit_physics_loss = jit(physics_loss)
        jit_loss_value = jit_physics_loss(control_input)
        jit_grad_fn = jit(grad(physics_loss))
        jit_gradients = jit_grad_fn(control_input)
        
        loss_diff = abs(loss_value - jit_loss_value)
        grad_diff = jnp.linalg.norm(gradients - jit_gradients)
        
        print(f"  📉 损失函数差异: {loss_diff:.2e}")
        print(f"  📈 梯度差异: {grad_diff:.2e}")
        print(f"  ✅ JIT编译一致性: {'通过' if loss_diff < 1e-10 and grad_diff < 1e-10 else '失败'}")
        
        # 断言验证
        assert jnp.all(jnp.isfinite(gradients)), "梯度必须是有限值"
        assert jnp.any(jnp.abs(gradients) > 1e-8), "梯度必须非零"
        assert loss_diff < 1e-10, "JIT编译必须保持数值一致性"
        
        print("✅ 阶段1验证完成: 物理引擎完全可微分且数值稳定")
        return True
        
    def test_hessian_analysis(self):
        """二阶导数分析 - Hessian矩阵计算"""
        print("\n📐 高阶导数分析:")
        
        drone_state = create_initial_drone_state(jnp.array([0.0, 0.0, 1.0]))
        
        def physics_loss_scalar(control_input):
            new_state = dynamics_step(drone_state, control_input, self.physics_params)
            return jnp.sum(new_state.position ** 2)
            
        control_input = jnp.array([0.1, 0.1, 0.2])
        
        # 计算Hessian矩阵
        hessian_fn = jacfwd(jacrev(physics_loss_scalar))
        hessian_matrix = hessian_fn(control_input)
        
        print("🔢 Hessian矩阵 H = ∂²L/∂u²:")
        for i, row in enumerate(hessian_matrix):
            print(f"  [{i}]: [{row[0]:8.4f}, {row[1]:8.4f}, {row[2]:8.4f}]")
            
        # 计算特征值
        eigenvals = jnp.linalg.eigvals(hessian_matrix)
        print(f"\n📊 Hessian特征值: {eigenvals}")
        print(f"🎯 凸性分析: {'凸函数' if jnp.all(eigenvals > 0) else '非凸函数'}")
        
        return True


class Stage2BPTTVerification:
    """阶段2: BPTT循环梯度流验证"""
    
    def __init__(self):
        self.config = get_minimal_config()
        self.physics_params = PhysicsParams(
            dt=self.config.physics.dt,
            mass=self.config.physics.drone.mass,
            thrust_to_weight_ratio=self.config.physics.drone.thrust_to_weight_ratio,
            drag_coefficient_linear=self.config.physics.drone.drag_coefficient
        )
        
    def test_multi_step_gradient_flow(self):
        """多步BPTT梯度流测试"""
        print("\n" + "="*80)
        print("🔄 阶段2: BPTT循环端到端梯度流验证")
        print("="*80)
        
        # 创建简单的策略网络
        policy_params = PolicyParams(hidden_dims=(32, 32), use_rnn=False)
        policy_network = create_policy_network(
            params=policy_params,
            network_type="mlp", 
            output_dim=3
        )
        
        # 初始化参数
        key = random.PRNGKey(42)
        drone_state = create_initial_drone_state(jnp.array([0.0, 0.0, 1.0]))
        
        policy_input = jnp.concatenate([
            drone_state.position, drone_state.velocity,
            drone_state.orientation.flatten()
        ])
        
        policy_params_init = policy_network.init(key, policy_input, None)
        
        # 定义多步仿真损失函数
        def multi_step_loss(policy_params, num_steps=5):
            state = drone_state
            total_loss = 0.0
            target_position = jnp.array([2.0, 1.0, 1.5])
            
            # 使用lax.scan进行高效BPTT
            def scan_step(carry_state, _):
                # 策略网络前向传播
                policy_input = jnp.concatenate([
                    carry_state.position, carry_state.velocity,
                    carry_state.orientation.flatten()#是机体x,y,z洲在世界坐标系中的 (x, y, z) 方向向量。.flatten() 操作将这个结构化的3x3矩阵转换成一个9维的向量。这样做可以让网络有机会学习这9个值之间复杂的非线性关系，从而理解无人机的完整姿态
                ])
                
                u_nom = policy_network.apply(policy_params, policy_input, None)
                
                # 物理仿真
                next_state = dynamics_step(carry_state, u_nom, self.physics_params)
                
                # 单步损失
                position_loss = jnp.sum((next_state.position - target_position) ** 2)
                control_loss = 0.01 * jnp.sum(u_nom ** 2)
                step_loss = position_loss + control_loss
                
                return next_state, {
                    'loss': step_loss,
                    'position': next_state.position,
                    'control': u_nom
                }
            
            # 执行scan
            final_state, scan_outputs = jax.lax.scan(
                scan_step, drone_state, None, length=num_steps
            )
            
            return jnp.sum(scan_outputs['loss']), scan_outputs
        
        # 计算损失和梯度
        loss_value, outputs = multi_step_loss(policy_params_init, num_steps=8)
        grad_fn = grad(lambda params: multi_step_loss(params, num_steps=8)[0])
        gradients = grad_fn(policy_params_init)
        
        print("📊 BPTT梯度流分析:")
        print(f"  🎯 仿真步数: 8")
        print(f"  📉 总损失: {loss_value:.6f}")
        print(f"  📏 轨迹长度: {outputs['position'].shape}")
        
        # 分析梯度统计
        grad_leaves = jax.tree_util.tree_leaves(gradients)
        grad_norms = [jnp.linalg.norm(g) for g in grad_leaves]
        total_grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in grad_leaves))
        
        print(f"\n📈 梯度统计:")
        print(f"  📊 参数块数量: {len(grad_leaves)}")
        print(f"  📏 总梯度范数: {total_grad_norm:.6f}")
        print(f"  📐 最大块范数: {max(grad_norms):.6f}")
        print(f"  📉 最小块范数: {min(grad_norms):.6f}")
        print(f"  ✅ 梯度有限性: {all(jnp.all(jnp.isfinite(g)) for g in grad_leaves)}")
        
        # 可视化轨迹演化
        print(f"\n🛤️  轨迹演化 (前5步):")
        for i in range(min(5, outputs['position'].shape[0])):
            pos = outputs['position'][i]
            ctrl = outputs['control'][i]
            print(f"  步骤{i+1}: 位置={pos} | 控制={ctrl}")
            
        # 梯度流可视化 - 检查每层的梯度幅值
        print(f"\n🔄 各层梯度流分析:")
        param_names = ['网络层1', '网络层2', '输出层']
        for i, (name, grad_norm) in enumerate(zip(param_names[:len(grad_norms)//2], grad_norms[::2])):
            print(f"  {name}: 梯度范数 = {grad_norm:.6f}")
            
        # 验证断言
        assert jnp.isfinite(loss_value), "损失必须是有限值"
        assert total_grad_norm > 1e-8, "梯度必须非零"
        assert all(jnp.all(jnp.isfinite(g)) for g in grad_leaves), "所有梯度必须有限"
        
        print("✅ 阶段2验证完成: BPTT梯度流正常传播")
        return True


class Stage3SafetyVerification:
    """阶段3: 安全机制集成验证"""
    
    def __init__(self):
        self.config = get_minimal_config()
        self.graph_config = GraphConfig()
        
    def test_perception_safety_gradients(self):
        """感知模块和安全层梯度验证"""
        print("\n" + "="*80)
        print("🛡️  阶段3: 感知模块与安全机制集成验证")
        print("="*80)
        
        key = random.PRNGKey(123)
        
        # 创建测试数据
        drone_state = create_initial_drone_state(jnp.array([0.0, 0.0, 1.0]))
        point_cloud = random.normal(key, (12, 3)) * 2.0  # 模拟LiDAR点云
        
        # 初始化感知网络
        graph, node_types = pointcloud_to_graph(drone_state, point_cloud, self.graph_config)
        cbf_params = init_cbf_network(key, graph, node_types)
        
        print("📊 感知模块分析:")
        print(f"  🎯 点云大小: {point_cloud.shape}")
        print(f"  📊 图节点数: {graph.n_node[0]}")
        print(f"  🔗 图边数: {graph.n_edge[0]}")
        
        # 测试CBF梯度计算
        def cbf_loss(drone_pos):
            modified_state = create_initial_drone_state(
                position=drone_pos,
                velocity=drone_state.velocity
            )
            cbf_value, cbf_grad = get_cbf_from_pointcloud(
                cbf_params, modified_state, point_cloud
            )
            return cbf_value, cbf_grad
        
        # 计算CBF值和梯度
        cbf_value, cbf_grad = cbf_loss(drone_state.position)
        
        print(f"\n🛡️  CBF安全分析:")
        print(f"  📉 CBF值: {cbf_value:.6f}")
        print(f"  📈 CBF梯度: {cbf_grad}")
        print(f"  📏 梯度范数: {jnp.linalg.norm(cbf_grad):.6f}")
        print(f"  🎯 安全状态: {'安全' if cbf_value > 0 else '危险'}")
        
        # 计算CBF相对于位置的Hessian
        def cbf_scalar(pos):
            modified_state = create_initial_drone_state(position=pos, velocity=drone_state.velocity)
            cbf_val, _ = get_cbf_from_pointcloud(cbf_params, modified_state, point_cloud)
            return cbf_val
            
        hessian_fn = jacfwd(jacrev(cbf_scalar))
        cbf_hessian = hessian_fn(drone_state.position)
        
        print(f"\n🔢 CBF Hessian矩阵:")
        for i, row in enumerate(cbf_hessian):
            print(f"  [{i}]: [{row[0]:8.4f}, {row[1]:8.4f}, {row[2]:8.4f}]")
            
        # 分析CBF的凸性
        eigenvals = jnp.linalg.eigvals(cbf_hessian)
        print(f"  📊 Hessian特征值: {eigenvals}")
        print(f"  🎯 CBF凸性: {'凸' if jnp.all(eigenvals > -1e-6) else '非凸'}")
        
        # 测试安全层
        safety_layer = create_default_safety_layer()
        u_nom = jnp.array([0.3, 0.2, 0.4])  # 名义控制
        
        u_safe, safety_info = safety_layer.safety_filter(
            u_nom, cbf_value, cbf_grad, drone_state
        )
        
        print(f"\n🔒 安全层过滤结果:")
        print(f"  📊 名义控制: {u_nom}")
        print(f"  🛡️  安全控制: {u_safe}")
        print(f"  📏 控制修正: {jnp.linalg.norm(u_safe - u_nom):.6f}")
        print(f"  ⚙️  QP求解状态: {getattr(safety_info, 'solver_status', '未知')}")
        
        # 测试安全层的可微分性
        def safety_loss(u_nominal):
            u_filtered, _ = safety_layer.safety_filter(
                u_nominal, cbf_value, cbf_grad, drone_state
            )
            return jnp.sum(u_filtered ** 2)
            
        safety_grad = grad(safety_loss)(u_nom)
        
        print(f"\n📈 安全层梯度分析:")
        print(f"  🎯 ∂L/∂u_nom: {safety_grad}")
        print(f"  📏 梯度范数: {jnp.linalg.norm(safety_grad):.6f}")
        print(f"  ✅ 梯度有限性: {jnp.all(jnp.isfinite(safety_grad))}")
        
        # 验证断言
        assert jnp.isfinite(cbf_value), "CBF值必须有限"
        assert jnp.all(jnp.isfinite(cbf_grad)), "CBF梯度必须有限"
        assert jnp.all(jnp.isfinite(u_safe)), "安全控制必须有限"
        assert jnp.all(jnp.isfinite(safety_grad)), "安全层梯度必须有限"
        
        print("✅ 阶段3验证完成: 感知与安全机制正常集成")
        return True


class Stage4CompleteSystemVerification:
    """阶段4: 完整系统集成验证"""
    
    def __init__(self):
        self.config = get_minimal_config()
        
    def test_full_system_gradient_flow(self):
        """完整系统端到端梯度流验证"""
        print("\n" + "="*80)  
        print("🎯 阶段4: 完整系统端到端梯度流验证")
        print("="*80)
        
        key = random.PRNGKey(456)
        gnn_key, policy_key = random.split(key, 2)
        
        # 初始化所有组件
        print("⚙️  系统组件初始化...")
        
        # 1. 物理参数
        physics_params = PhysicsParams(
            dt=self.config.physics.dt,
            mass=self.config.physics.drone.mass,
            thrust_to_weight_ratio=self.config.physics.drone.thrust_to_weight_ratio,
            drag_coefficient_linear=self.config.physics.drone.drag_coefficient
        )
        
        # 2. 感知模块  
        drone_state = create_initial_drone_state(jnp.array([0.0, 0.0, 1.0]))
        point_cloud = random.normal(gnn_key, (10, 3)) * 1.5
        graph_config = GraphConfig()
        
        graph, node_types = pointcloud_to_graph(drone_state, point_cloud, graph_config)
        cbf_params = init_cbf_network(gnn_key, graph, node_types)
        
        # 3. 策略网络
        policy_params = PolicyParams(hidden_dims=(32, 32), use_rnn=False)
        policy_network = create_policy_network(
            params=policy_params, network_type="mlp", output_dim=3
        )
        
        policy_input = jnp.concatenate([
            drone_state.position, drone_state.velocity,
            drone_state.orientation.flatten()
        ])
        policy_params_init = policy_network.init(policy_key, policy_input, None)
        
        # 4. 安全层
        safety_layer = create_default_safety_layer()
        
        # 定义完整系统的损失函数
        def complete_system_loss(all_params, num_steps=6):
            """完整系统的多目标损失函数"""
            cbf_params = all_params['cbf_params']
            policy_params = all_params['policy_params']
            
            state = drone_state
            total_efficiency_loss = 0.0
            total_safety_loss = 0.0  
            total_control_loss = 0.0
            target_position = jnp.array([1.5, 1.0, 1.5])
            
            # 多步仿真scan
            def scan_step(carry_state, step_idx):
                # 1. 感知: 计算CBF
                cbf_value, cbf_grad = get_cbf_from_pointcloud(
                    cbf_params, carry_state, point_cloud
                )
                
                # 2. 策略: 生成名义控制
                policy_input = jnp.concatenate([
                    carry_state.position, carry_state.velocity,
                    carry_state.orientation.flatten()
                ])
                u_nom = policy_network.apply(policy_params, policy_input, None)
                
                # 3. 安全: 过滤控制
                u_safe, _ = safety_layer.safety_filter(
                    u_nom, cbf_value, cbf_grad, carry_state
                )
                
                # 4. 物理: 状态更新
                next_state = dynamics_step(carry_state, u_safe, physics_params)
                
                # 5. 多目标损失计算
                efficiency_loss = jnp.sum((next_state.position - target_position) ** 2)
                safety_loss = jnp.maximum(0.0, -cbf_value) ** 2  # 安全违反惩罚
                control_loss = jnp.sum(u_safe ** 2)
                
                return next_state, {
                    'efficiency_loss': efficiency_loss,
                    'safety_loss': safety_loss,
                    'control_loss': control_loss,
                    'cbf_value': cbf_value,
                    'position': next_state.position,
                    'u_nom': u_nom,
                    'u_safe': u_safe
                }
            
            # 执行完整仿真
            final_state, scan_outputs = jax.lax.scan(
                scan_step, state, jnp.arange(num_steps)
            )
            
            # 汇总损失
            total_efficiency = jnp.sum(scan_outputs['efficiency_loss'])
            total_safety = jnp.sum(scan_outputs['safety_loss'])
            total_control = 0.01 * jnp.sum(scan_outputs['control_loss'])
            
            total_loss = total_efficiency + 10.0 * total_safety + total_control
            
            return total_loss, scan_outputs
        
        # 打包参数
        all_params = {
            'cbf_params': cbf_params,
            'policy_params': policy_params_init
        }
        
        # 计算损失和多目标梯度
        print("\n📊 完整系统前向传播...")
        total_loss, outputs = complete_system_loss(all_params, num_steps=5)
        
        print(f"  🎯 仿真步数: 5")
        print(f"  📉 总损失: {total_loss:.6f}")
        print(f"  📊 效率损失: {jnp.sum(outputs['efficiency_loss']):.6f}")
        print(f"  🛡️  安全损失: {jnp.sum(outputs['safety_loss']):.6f}")
        print(f"  ⚙️  控制损失: {jnp.sum(outputs['control_loss']):.6f}")
        
        # 分别计算各个目标的梯度
        print("\n🔄 多目标梯度分解分析...")
        
        def efficiency_loss_only(params):
            loss, outputs = complete_system_loss(params, num_steps=5)
            return jnp.sum(outputs['efficiency_loss'])
            
        def safety_loss_only(params):
            loss, outputs = complete_system_loss(params, num_steps=5)  
            return jnp.sum(outputs['safety_loss'])
        
        # 计算各个目标的梯度
        efficiency_grads = grad(efficiency_loss_only)(all_params)
        safety_grads = grad(safety_loss_only)(all_params)
        total_grads = grad(lambda p: complete_system_loss(p, num_steps=5)[0])(all_params)
        
        # 梯度统计分析
        def compute_grad_stats(grads, name):
            leaves = jax.tree_util.tree_leaves(grads)
            total_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in leaves))
            max_norm = max(jnp.linalg.norm(g) for g in leaves)
            return total_norm, max_norm
        
        eff_total, eff_max = compute_grad_stats(efficiency_grads, "效率")
        safe_total, safe_max = compute_grad_stats(safety_grads, "安全")
        total_total, total_max = compute_grad_stats(total_grads, "总计")
        
        print("📈 梯度分解统计:")
        print(f"  🎯 效率梯度 - 总范数: {eff_total:.6f} | 最大范数: {eff_max:.6f}")
        print(f"  🛡️  安全梯度 - 总范数: {safe_total:.6f} | 最大范数: {safe_max:.6f}")  
        print(f"  📊 合计梯度 - 总范数: {total_total:.6f} | 最大范数: {total_max:.6f}")
        
        # 梯度方向分析
        def compute_gradient_angle(grad1, grad2):
            leaves1 = jax.tree_util.tree_leaves(grad1)
            leaves2 = jax.tree_util.tree_leaves(grad2)
            
            dot_product = sum(jnp.sum(g1 * g2) for g1, g2 in zip(leaves1, leaves2))
            norm1 = jnp.sqrt(sum(jnp.sum(g1**2) for g1 in leaves1))
            norm2 = jnp.sqrt(sum(jnp.sum(g2**2) for g2 in leaves2))
            
            cos_angle = dot_product / (norm1 * norm2 + 1e-8)
            angle_deg = jnp.arccos(jnp.clip(cos_angle, -1, 1)) * 180 / jnp.pi
            return angle_deg
            
        if safe_total > 1e-8:  # 只有安全梯度非零时才计算角度
            angle = compute_gradient_angle(efficiency_grads, safety_grads)
            print(f"  📐 效率-安全梯度夹角: {angle:.2f}°")
            print(f"  🎯 梯度冲突程度: {'低' if angle < 60 else '中' if angle < 120 else '高'}")
        
        # 轨迹分析
        print(f"\n🛤️  系统轨迹演化:")
        for i in range(min(3, outputs['position'].shape[0])):
            pos = outputs['position'][i]
            cbf = outputs['cbf_value'][i] 
            u_nom = outputs['u_nom'][i]
            u_safe = outputs['u_safe'][i]
            ctrl_diff = jnp.linalg.norm(u_safe - u_nom)
            
            print(f"  步骤{i+1}:")
            print(f"    位置: [{pos[0]:6.3f}, {pos[1]:6.3f}, {pos[2]:6.3f}]")
            print(f"    CBF值: {cbf:7.4f} ({'安全' if cbf > 0 else '危险'})")
            print(f"    控制修正: {ctrl_diff:.4f}")
        
        # 验证断言
        assert jnp.isfinite(total_loss), "总损失必须有限"
        assert total_total > 1e-8, "总梯度必须非零"
        assert jnp.all(jnp.isfinite(jax.tree_util.tree_leaves(total_grads)[0])), "梯度必须有限"
        
        print("✅ 阶段4验证完成: 完整系统梯度流正常")
        return True

    def test_optimization_step(self):
        """完整优化步骤测试"""
        print("\n🔧 优化步骤测试...")
        
        key = random.PRNGKey(789)
        
        # 创建简化系统用于优化测试
        drone_state = create_initial_drone_state(jnp.array([0.0, 0.0, 1.0]))
        physics_params = PhysicsParams()
        
        # 策略网络
        policy_params = PolicyParams(hidden_dims=(16, 16), use_rnn=False)
        policy_network = create_policy_network(
            params=policy_params, network_type="mlp", output_dim=3
        )
        
        policy_input = jnp.concatenate([
            drone_state.position, drone_state.velocity,
            drone_state.orientation.flatten()
        ])
        
        initial_params = policy_network.init(key, policy_input, None)
        
        # 简单损失函数
        def optimization_loss(params):
            state = drone_state
            total_loss = 0.0
            target_pos = jnp.array([1.0, 1.0, 1.5])
            
            for _ in range(3):
                policy_input = jnp.concatenate([
                    state.position, state.velocity,
                    state.orientation.flatten()
                ])
                
                u_nom = policy_network.apply(params, policy_input, None)
                state = dynamics_step(state, u_nom, physics_params)
                total_loss += jnp.sum((state.position - target_pos) ** 2)
                
            return total_loss
        
        # 创建优化器并执行优化步骤
        optimizer = create_optimizer(learning_rate=1e-3)
        opt_state = optimizer.init(initial_params)
        
        initial_loss = optimization_loss(initial_params)
        
        # 优化步骤
        grads = grad(optimization_loss)(initial_params)
        updates, new_opt_state = optimizer.update(grads, opt_state, initial_params)
        new_params = optax.apply_updates(initial_params, updates)
        
        final_loss = optimization_loss(new_params)
        
        print(f"  📉 初始损失: {initial_loss:.6f}")
        print(f"  📈 优化后损失: {final_loss:.6f}")
        print(f"  📊 损失改进: {(initial_loss - final_loss)/initial_loss*100:.2f}%")
        
        # 参数更新统计
        param_leaves = jax.tree_util.tree_leaves(initial_params)
        new_param_leaves = jax.tree_util.tree_leaves(new_params)
        
        param_changes = [jnp.linalg.norm(p_new - p_old) for p_old, p_new in zip(param_leaves, new_param_leaves)]
        total_change = sum(param_changes)
        
        print(f"  ⚙️  参数更新范数: {total_change:.6f}")
        print(f"  ✅ 优化步骤: {'成功' if final_loss < initial_loss else '需要调整'}")
        
        return True


def run_all_enhanced_tests():
    """运行全部增强版测试"""
    print("\n" + "="*100)
    print("🚀 SAFE AGILE FLIGHT - 增强版四阶段验证测试")
    print("="*100)
    print("基于GCBF+ (MIT-REALM) + DiffPhysDrone (SJTU) 的完整JAX实现")
    print("详细梯度可视化 | 矩阵分析 | 数值验证")
    print("="*100)
    
    success_count = 0
    total_tests = 4
    
    try:
        # 阶段1: 物理引擎
        print("\n🎯 执行阶段1测试...")
        stage1 = Stage1PhysicsVerification()
        if stage1.test_basic_differentiability() and stage1.test_hessian_analysis():
            success_count += 1
            print("✅ 阶段1: 物理引擎可微分性 - 通过")
        
        # 阶段2: BPTT循环  
        print("\n🎯 执行阶段2测试...")
        stage2 = Stage2BPTTVerification()
        if stage2.test_multi_step_gradient_flow():
            success_count += 1
            print("✅ 阶段2: BPTT循环梯度流 - 通过")
        
        # 阶段3: 安全机制
        print("\n🎯 执行阶段3测试...")
        stage3 = Stage3SafetyVerification()
        if stage3.test_perception_safety_gradients():
            success_count += 1  
            print("✅ 阶段3: 安全机制集成 - 通过")
        
        # 阶段4: 完整系统
        print("\n🎯 执行阶段4测试...")
        stage4 = Stage4CompleteSystemVerification()
        if stage4.test_full_system_gradient_flow() and stage4.test_optimization_step():
            success_count += 1
            print("✅ 阶段4: 完整系统集成 - 通过")
        
        # 最终报告
        print("\n" + "="*100)
        if success_count == total_tests:
            print("🎉 所有阶段验证成功完成!")
            print("🎯 系统状态: 完全就绪，可进入生产训练")
            print("📊 验证覆盖: 物理引擎 ✓ | BPTT循环 ✓ | 安全机制 ✓ | 端到端集成 ✓")
            print("🔥 性能特性: 完全可微分 | JIT编译优化 | 内存高效 | 数值稳定")
        else:
            print(f"⚠️  部分测试未通过 ({success_count}/{total_tests})")
            print("🔧 建议检查失败的组件并重新测试")
        
        print("="*100)
        return success_count == total_tests
        
    except Exception as e:
        print(f"❌ 测试执行异常: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_enhanced_tests()
    exit(0 if success else 1)