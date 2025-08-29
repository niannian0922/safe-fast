"""
结合GCBF+和DiffPhysDrone方法的安全敏捷飞行系统的默认配置。

此配置文件定义了统一基于JAX框架的所有超参数、模型设置和环境参数，
该框架实现了带有可微分物理仿真的神经图控制屏障函数。
"""

import ml_collections


def get_config():
    """返回作为ConfigDict的默认配置。"""
    config = ml_collections.ConfigDict()
    
    # =============================================================================
    # 系统配置
    # =============================================================================
    config.system = ml_collections.ConfigDict()
    config.system.framework = "jax"  # 主要框架
    config.system.precision = "float32"  # JAX精度设置
    config.system.enable_jit = True  # 启用JIT编译
    config.system.enable_x64 = False  # 启用64位精度
    config.system.device = "auto"  # 设备选择："cpu", "gpu", "tpu", "auto"
    
    # 随机种子配置
    config.system.seed = 42
    config.system.deterministic = True
    
    # =============================================================================
    # 物理仿真参数（核心DiffPhysDrone集成）
    # =============================================================================
    config.physics = ml_collections.ConfigDict()
    
    # 时间积分设置
    config.physics.dt = 1.0/15.0  # 15Hz控制频率
    config.physics.integration_method = "euler"  # "euler", "rk4", "adaptive"
    config.physics.max_steps = 150  # 每集最大仿真步数
    
    # 无人机动力学参数（来自DiffPhysDrone的点质量模型）
    config.physics.drone = ml_collections.ConfigDict()
    config.physics.drone.mass = 0.027   # Crazyflie质量
    config.physics.drone.thrust_to_weight_ratio = 3.0  # 最大推重比
    config.physics.drone.max_acceleration = 29.4  # m/s^2 （激进飞行的3*g）
    config.physics.drone.drag_coefficient = 0.01  # 线性阻力系数
    config.physics.drone.radius = 0.05  # 碰撞检测的安全半径
    
    # 控制约束
    config.physics.control = ml_collections.ConfigDict()
    config.physics.control.max_thrust = 0.8  # 标准化最大推力
    config.physics.control.min_thrust = 0.0  # Normalized minimum thrust
    config.physics.control.thrust_delay = 1.0/15.0  # Control delay (tau parameter)
    config.physics.control.smoothing_factor = 12.0  # Exponential smoothing (lambda parameter)
    
    # Temporal gradient decay (Critical DiffPhysDrone innovation)
    config.physics.gradient_decay = ml_collections.ConfigDict()
    config.physics.gradient_decay.alpha = 0.92  # 时间梯度衰减
    config.physics.gradient_decay.enable = True  # Enable gradient decay mechanism
    
    # =============================================================================
    # GCBF+ CONFIGURATION (Neural Graph Control Barrier Functions)
    # =============================================================================
    config.gcbf = ml_collections.ConfigDict()
    
    # Graph construction parameters
    config.gcbf.sensing_radius = 0.5  # R parameter for neighbor detection
    config.gcbf.max_neighbors = 16  # M parameter for fixed-size neighborhoods
    config.gcbf.k_neighbors = 8  # KNN图构建
    config.gcbf.graph_construction_method = "knn"  # "knn" or "radius"
    
    # CBF parameters
    config.gcbf.alpha = 1.0  # CBF类K函数参数
    config.gcbf.gamma = 0.02  # Margin parameter for strict inequalities
    config.gcbf.look_ahead_horizon = 32  # T parameter for control invariant set computation
    
    # GNN architecture (from GCBF+ paper)
    config.gcbf.gnn = ml_collections.ConfigDict()
    config.gcbf.gnn.hidden_dims = [256, 256, 128]  # GNN架构
    config.gcbf.gnn.output_dim = 1  # CBF scalar output
    config.gcbf.gnn.activation = "relu"  # Activation function
    config.gcbf.gnn.use_attention = True  # Graph attention mechanism
    config.gcbf.gnn.attention_heads = 4  # Number of attention heads
    config.gcbf.gnn.dropout_rate = 0.1  # Dropout for regularization
    
    # =============================================================================
    # POLICY NETWORK CONFIGURATION 
    # =============================================================================
    config.policy = ml_collections.ConfigDict()
    
    # Input/Output dimensions
    config.policy.input_dim = 13  # 3(pos) + 3(vel) + 9(orientation) + 3(angular_vel) - 5 = 13
    config.policy.output_dim = 3  # 3D control input
    
    # Architecture
    config.policy.type = "mlp_rnn"  # "mlp", "rnn", "mlp_rnn"
    config.policy.hidden_dims = [256, 256]  # Hidden layer dimensions
    config.policy.rnn_hidden_size = 256  # RNN hidden state size
    config.policy.rnn_type = "gru"  # "gru", "lstm"
    config.policy.activation = "relu"  # Activation function
    config.policy.output_activation = "tanh"  # Output activation for control
    config.policy.use_rnn = False  # Enable RNN for memory
    
    # =============================================================================
    # SAFETY LAYER CONFIGURATION (QP-based Safety Filter)
    # =============================================================================
    config.safety = ml_collections.ConfigDict()
    
    # Control limits
    config.safety.max_thrust = 0.8  # Maximum thrust magnitude
    config.safety.max_torque = 0.5  # Maximum torque magnitude
    
    # CBF parameters
    config.safety.cbf_alpha = 1.0  # CBF alpha parameter
    
    #安全层配置 (qpax集成)
    config.safety.solver = "qpax"  # Differentiable QP solver
    config.safety.max_iterations = 100  # Maximum QP iterations
    config.safety.tolerance = 1e-6  # 松弛变量惩罚
    config.safety.regularization = 1e-8  # Regularization parameter
    
    # Three-layer safety mechanism parameters
    config.safety.relaxation_penalty = 1e6  # Beta parameter for slack variables
    config.safety.failsafe_mode = "emergency_brake"  # Failsafe strategy
    config.safety.enable_backoff = True  # Enable automatic backoff mechanism
    
    # Training configuration with performance optimization
    config.training = ml_collections.ConfigDict()
    
    # Optimized learning rates (from performance tuning research)
    config.training.optimizer = "adam"
    config.training.learning_rate = 1e-4  # Base learning rate (optimized)
    config.training.learning_rate_gcbf = 5e-5  # GNN稳定性
    config.training.learning_rate_policy = 2e-4  # 策略收敛
    config.training.batch_size = 32  # Optimized batch size for memory/performance
    config.training.sequence_length = 25  # BPTT效率平衡
    config.training.num_epochs = 100  # Extended for better convergence
    config.training.batches_per_epoch = 20  # More iterations per epoch
    config.training.validation_frequency = 5
    config.training.validation_batch_size = 16
    config.training.checkpoint_frequency = 10
    config.training.max_steps = 2000
    config.training.seed = 42
    
    # Optimized loss function coefficients (empirically tuned)
    config.training.loss_cbf_coef = 2.0  # Increased for safety emphasis
    config.training.loss_velocity_coef = 1.0  # Balanced velocity tracking
    config.training.loss_goal_coef = 3.0  # Strong goal-directed behavior  
    config.training.loss_control_coef = 0.05  # Reduced control penalty
    config.training.loss_collision_coef = 5.0  # High collision avoidance
    config.training.loss_safety_coef = 2.5  # Enhanced safety importance
    
    # Advanced gradient handling
    config.training.gradient_clip_norm = 1.0
    config.training.grad_decay_eta = 0.1  # Reduced for stability
    config.training.use_gradient_checkpointing = True  # Memory optimization
    
    # Performance optimization features
    config.training.performance_tuning = ml_collections.ConfigDict()
    config.training.performance_tuning.enable = True
    config.training.performance_tuning.adaptive_lr_schedule = "warmup_cosine"
    config.training.performance_tuning.adaptive_loss_weights = True
    config.training.performance_tuning.warmup_steps = 200
    config.training.performance_tuning.decay_steps = 1500
    config.training.performance_tuning.weight_update_frequency = 50
    
    # Curriculum learning (Three-stage approach)
    config.training.curriculum = ml_collections.ConfigDict()
    config.training.curriculum.enable = True
    config.training.curriculum.stage1_steps = 300  # Efficiency-first stage
    config.training.curriculum.stage2_steps = 400  # Safety-aware stage  
    config.training.curriculum.stage3_steps = 300  # Joint optimization
    config.training.curriculum.annealing_start = 1e-6  # Initial relaxation penalty
    config.training.curriculum.annealing_end = 1e6    # Final relaxation penalty
    
    # =============================================================================
    # ENVIRONMENT CONFIGURATION
    # =============================================================================
    config.env = ml_collections.ConfigDict()
    
    # Multi-agent settings
    config.env.num_agents = 8  # Number of agents (consistent with GCBF+ paper)
    config.env.area_size = 4.0  # Environment side length
    config.env.num_obstacles = 8  # Number of static obstacles
    config.env.obstacle_types = ["sphere", "box", "cylinder"]  # Obstacle primitives
    
    # LiDAR configuration (for real-world deployment)
    config.env.lidar = ml_collections.ConfigDict()
    config.env.lidar.num_rays = 32  # Number of LiDAR rays
    config.env.lidar.max_range = 2.0  # Maximum sensing range
    config.env.lidar.angular_resolution = 360.0 / 32  # Angular resolution in degrees
    
    # Goal configuration
    config.env.goal_tolerance = 0.1  # Goal reaching tolerance
    config.env.max_episode_length = 150  # Maximum episode length
    
    # =============================================================================
    # EVALUATION AND LOGGING
    # =============================================================================
    config.evaluation = ml_collections.ConfigDict()
    config.evaluation.eval_frequency = 100  # Evaluation frequency (training steps)
    config.evaluation.num_eval_episodes = 5  # Episodes per evaluation
    config.evaluation.success_threshold = 0.95  # Success rate threshold
    config.evaluation.safety_threshold = 1.0   # Required safety rate
    
    config.logging = ml_collections.ConfigDict()
    config.logging.wandb_project = "safe_agile_flight"  # Weights & Biases project
    config.logging.log_frequency = 10  # Logging frequency
    config.logging.save_frequency = 100  # Model checkpoint frequency
    config.logging.video_logging = True  # Enable trajectory video logging
    
    # =============================================================================
    # COMPUTATIONAL OPTIMIZATION
    # =============================================================================
    config.optimization = ml_collections.ConfigDict()
    
    # JAX-specific optimizations
    config.optimization.use_scan = True  # Use jax.lax.scan for BPTT
    config.optimization.use_checkpoint = True  # Enable gradient checkpointing
    config.optimization.checkpoint_strategy = "selective"  # "none", "all", "selective"
    config.optimization.nested_checkpoint = False  # Enable nested checkpointing
    
    # Memory management
    config.optimization.max_memory_usage = 0.8  # Maximum GPU memory fraction
    config.optimization.clear_cache_frequency = 50  # JAX cache clearing frequency
    
    # Experiment configuration
    config.experiment_name = "safe_agile_flight_stage4"
    config.experiment_description = "Complete system integration with GCBF+ and DiffPhysDrone"
    
    return config


def get_minimal_config():
    """Returns a minimal configuration for testing and development."""
    config = get_config()
    
    # Reduce computational requirements for testing
    config.physics.max_steps = 50
    config.training.max_steps = 100
    config.training.batch_size = 8
    config.training.sequence_length = 10  # Much shorter for testing memory
    config.env.num_agents = 2
    config.gcbf.max_neighbors = 4
    config.gcbf.k_neighbors = 3  # Smaller for testing
    
    # Disable expensive features for testing
    config.optimization.use_checkpoint = False
    config.logging.video_logging = False
    config.training.curriculum.enable = False
    
    return config


def get_single_agent_config():
    """Returns configuration for single-agent scenarios."""
    config = get_config()
    
    # Single agent modifications
    config.env.num_agents = 1
    config.gcbf.max_neighbors = 8  # Only obstacles as neighbors
    config.gcbf.k_neighbors = 6  # Reasonable for single agent
    config.env.num_obstacles = 16  # Increase obstacle density
    
    # Adjust training for single-agent case
    config.training.loss_cbf_coef = 0.5  # Reduced CBF importance
    config.training.batch_size = 32
    
    return config


def get_hardware_config():
    """Returns configuration optimized for hardware deployment."""
    config = get_config()
    
    # Hardware-specific optimizations
    config.system.precision = "float16"  # Reduce memory usage
    config.optimization.use_checkpoint = True  # Memory efficiency
    config.gcbf.gnn.hidden_dims = [128, 128, 64]  # Smaller network
    config.policy.hidden_dims = [128, 128]  # Smaller policy network
    
    # Real-time constraints
    config.physics.dt = 1.0/30.0  # Higher frequency for hardware
    config.physics.max_steps = 100  # Shorter horizons
    
    # Disable expensive logging for hardware
    config.logging.video_logging = False
    config.evaluation.num_eval_episodes = 1
    
    return config