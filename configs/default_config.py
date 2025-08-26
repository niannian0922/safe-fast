"""
Default configuration for the Safe Agile Flight system combining GCBF+ and DiffPhysDrone methodologies.

This configuration file defines all hyperparameters, model settings, and environment parameters
for the unified JAX-based framework implementing neural graph control barrier functions
with differentiable physics simulation.
"""

import ml_collections


def get_config():
    """Returns the default configuration as a ConfigDict."""
    config = ml_collections.ConfigDict()
    
    # =============================================================================
    # SYSTEM CONFIGURATION
    # =============================================================================
    config.system = ml_collections.ConfigDict()
    config.system.framework = "jax"  # Primary framework
    config.system.precision = "float32"  # JAX precision setting
    config.system.enable_jit = True  # Enable JIT compilation
    config.system.enable_x64 = False  # Enable 64-bit precision
    config.system.device = "auto"  # Device selection: "cpu", "gpu", "tpu", "auto"
    
    # Random seed configuration
    config.system.seed = 42
    config.system.deterministic = True
    
    # =============================================================================
    # PHYSICS SIMULATION PARAMETERS (Core DiffPhysDrone Integration)
    # =============================================================================
    config.physics = ml_collections.ConfigDict()
    
    # Time integration settings
    config.physics.dt = 1.0/15.0  # Simulation timestep (consistent with DiffPhysDrone)
    config.physics.integration_method = "euler"  # "euler", "rk4", "adaptive"
    config.physics.max_steps = 150  # Maximum simulation steps per episode
    
    # Drone dynamics parameters (Point-mass model from DiffPhysDrone)
    config.physics.drone = ml_collections.ConfigDict()
    config.physics.drone.mass = 0.027  # kg (Crazyflie mass)
    config.physics.drone.thrust_to_weight_ratio = 3.0  # Maximum thrust ratio
    config.physics.drone.max_acceleration = 29.4  # m/s^2 (3*g for aggressive flight)
    config.physics.drone.drag_coefficient = 0.01  # Linear drag coefficient
    config.physics.drone.radius = 0.05  # Safety radius for collision detection
    
    # Control constraints
    config.physics.control = ml_collections.ConfigDict()
    config.physics.control.max_thrust = 0.8  # Normalized maximum thrust
    config.physics.control.min_thrust = 0.0  # Normalized minimum thrust
    config.physics.control.thrust_delay = 1.0/15.0  # Control delay (tau parameter)
    config.physics.control.smoothing_factor = 12.0  # Exponential smoothing (lambda parameter)
    
    # Temporal gradient decay (Critical DiffPhysDrone innovation)
    config.physics.gradient_decay = ml_collections.ConfigDict()
    config.physics.gradient_decay.alpha = 0.92  # Decay rate for temporal gradients
    config.physics.gradient_decay.enable = True  # Enable gradient decay mechanism
    
    # =============================================================================
    # GCBF+ CONFIGURATION (Neural Graph Control Barrier Functions)
    # =============================================================================
    config.gcbf = ml_collections.ConfigDict()
    
    # Graph construction parameters
    config.gcbf.sensing_radius = 0.5  # R parameter for neighbor detection
    config.gcbf.max_neighbors = 16  # M parameter for fixed-size neighborhoods
    config.gcbf.graph_construction_method = "knn"  # "knn" or "radius"
    
    # CBF parameters
    config.gcbf.alpha = 1.0  # CBF alpha parameter (class-K function coefficient)
    config.gcbf.gamma = 0.02  # Margin parameter for strict inequalities
    config.gcbf.look_ahead_horizon = 32  # T parameter for control invariant set computation
    
    # GNN architecture (from GCBF+ paper)
    config.gcbf.gnn = ml_collections.ConfigDict()
    config.gcbf.gnn.hidden_dims = [256, 256, 128]  # Hidden layer dimensions
    config.gcbf.gnn.output_dim = 1  # CBF scalar output
    config.gcbf.gnn.activation = "relu"  # Activation function
    config.gcbf.gnn.use_attention = True  # Graph attention mechanism
    config.gcbf.gnn.attention_heads = 4  # Number of attention heads
    config.gcbf.gnn.dropout_rate = 0.1  # Dropout for regularization
    
    # =============================================================================
    # POLICY NETWORK CONFIGURATION 
    # =============================================================================
    config.policy = ml_collections.ConfigDict()
    
    # Architecture
    config.policy.type = "mlp_rnn"  # "mlp", "rnn", "mlp_rnn"
    config.policy.hidden_dims = [256, 256]  # Hidden layer dimensions
    config.policy.rnn_hidden_size = 256  # RNN hidden state size
    config.policy.rnn_type = "gru"  # "gru", "lstm"
    config.policy.activation = "relu"  # Activation function
    config.policy.output_activation = "tanh"  # Output activation for control
    
    # =============================================================================
    # SAFETY LAYER CONFIGURATION (QP-based Safety Filter)
    # =============================================================================
    config.safety = ml_collections.ConfigDict()
    
    # QP solver settings (qpax integration)
    config.safety.solver = "qpax"  # Differentiable QP solver
    config.safety.max_iterations = 100  # Maximum QP iterations
    config.safety.tolerance = 1e-6  # Convergence tolerance
    config.safety.regularization = 1e-8  # Regularization parameter
    
    # Three-layer safety mechanism parameters
    config.safety.relaxation_penalty = 1e6  # Beta parameter for slack variables
    config.safety.failsafe_mode = "emergency_brake"  # Failsafe strategy
    config.safety.enable_backoff = True  # Enable automatic backoff mechanism
    
    # =============================================================================
    # TRAINING CONFIGURATION
    # =============================================================================
    config.training = ml_collections.ConfigDict()
    
    # Optimization settings
    config.training.optimizer = "adam"  # Optimizer type
    config.training.learning_rate_gcbf = 1e-5  # Learning rate for GCBF
    config.training.learning_rate_policy = 1e-5  # Learning rate for policy
    config.training.batch_size = 64  # Training batch size
    config.training.max_steps = 1000  # Maximum training steps
    
    # Loss function coefficients
    config.training.loss_cbf_coef = 1.0  # CBF loss coefficient
    config.training.loss_safety_coef = 2.0  # Safety loss coefficient  
    config.training.loss_goal_coef = 1.0  # Goal reaching loss coefficient
    config.training.loss_control_coef = 1e-4  # Control smoothness coefficient
    
    # Gradient clipping and numerical stability
    config.training.gradient_clip_norm = 1.0  # Gradient clipping threshold
    config.training.grad_decay_eta = 0.2  # Derivative loss weight
    
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
    
    return config


def get_minimal_config():
    """Returns a minimal configuration for testing and development."""
    config = get_config()
    
    # Reduce computational requirements for testing
    config.physics.max_steps = 50
    config.training.max_steps = 100
    config.training.batch_size = 8
    config.env.num_agents = 2
    config.gcbf.max_neighbors = 4
    
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