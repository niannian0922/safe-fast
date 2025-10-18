"""
默认配置文件 —— 单机版安全敏捷飞行

配置围绕当前仓库实现的“单无人机 + JAX 原生管线”整理，吸收了
GCBF+（图神经 CBF / qpax 安全层）与 DiffPhysDrone（可微物理 / BPTT）
的核心思路，但仅保留对现有代码实际生效的字段，避免多智能体遗留选项
误导使用者。
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
    config.physics.control.max_thrust = 5.0  # 对应世界系加速度界
    config.physics.control.min_thrust = -5.0  # 允许对称加速度
    config.physics.control.thrust_delay = 1.0/15.0  # 控制延迟（tau参数）
    config.physics.control.smoothing_factor = 12.0  # 指数平滑（lambda参数）
    
    # 时间梯度衰减（DiffPhysDrone的关键创新）
    config.physics.gradient_decay = ml_collections.ConfigDict()
    config.physics.gradient_decay.alpha = 0.92  # 时间梯度衰减
    config.physics.gradient_decay.enable = True  # 启用梯度衰减机制
    
    # =============================================================================
    # GCBF+ 配置（神经图控制障碍函数）
    # =============================================================================
    config.gcbf = ml_collections.ConfigDict()
    
    # 图构建参数
    config.gcbf.sensing_radius = 3.0  # 与 GraphConfig.max_distance 对齐（约两倍半径）
    config.gcbf.max_neighbors = 64  # 单无人机点云上限
    config.gcbf.k_neighbors = 8  # 与 core.perception.GraphConfig 默认一致
    config.gcbf.graph_construction_method = "knn"  # 当前实现仅使用 KNN
    
    # CBF参数
    config.gcbf.alpha = 1.0  # CBF类K函数参数
    config.gcbf.gamma = 0.02  # 严格不等式的边界参数
    config.gcbf.look_ahead_horizon = 32  # 控制不变集计算的T参数
    
    # GNN架构（来自GCBF+论文）
    config.gcbf.gnn = ml_collections.ConfigDict()
    config.gcbf.gnn.hidden_dims = [128, 128, 128]  # 与 core.perception 中的轻量版一致
    config.gcbf.gnn.output_dim = 1  # CBF标量输出
    config.gcbf.gnn.activation = "relu"  # 激活函数
    config.gcbf.gnn.use_attention = True  # 图注意力机制
    config.gcbf.gnn.attention_heads = 4  # 注意力头数量
    config.gcbf.gnn.dropout_rate = 0.1  # 正则化的Dropout
    
    # =============================================================================
    # 策略网络配置 
    # =============================================================================
    config.policy = ml_collections.ConfigDict()
    
    # 输入/输出维度
    config.policy.input_dim = 10  # 3(pos) + 3(vel) + 3(target offset) + 1(cbf value)
    config.policy.output_dim = 3  # 3D control input
    
    # 架构
    config.policy.type = "mlp_rnn"  # "mlp", "rnn", "mlp_rnn"
    config.policy.hidden_dims = [256, 256]  # 隐藏层维度
    config.policy.rnn_hidden_size = 256  # RNN隐藏状态大小
    config.policy.rnn_type = "gru"  # "gru", "lstm"
    config.policy.activation = "relu"  # 激活函数
    config.policy.output_activation = "tanh"  # 控制输出激活函数
    config.policy.use_rnn = False  # 启用RNN内存
    config.policy.action_limit = 5.0  # 与物理加速度上限一致
    
    # =============================================================================
    # 安全层配置（基于QP的安全过滤器）
    # =============================================================================
    config.safety = ml_collections.ConfigDict()
    
    # CBF参数
    config.safety.alpha0 = 1.0
    config.safety.alpha1 = 2.0
    config.safety.max_acceleration = 5.0
    config.safety.relaxation_penalty = 150.0
    config.safety.max_relaxation = 3.0
    config.safety.violation_tolerance = 1e-5
    
    # 带性能优化的训练配置
    config.training = ml_collections.ConfigDict()
    
    # 优化的学习率（来自性能调优研究）
    config.training.optimizer = "adam"
    config.training.learning_rate = 1e-4  # 基础学习率（已优化）
    config.training.learning_rate_gcbf = 5e-5  # GNN稳定性
    config.training.learning_rate_policy = 2e-4  # 策略收敛
    config.training.batch_size = 32  # 优化的批次大小（内存/性能平衡）
    config.training.sequence_length = 25  # BPTT效率平衡
    config.training.num_epochs = 100  # 为更好收敛而扩展
    config.training.batches_per_epoch = 20  # 每个训练轮次更多迭代
    config.training.validation_frequency = 5
    config.training.validation_batch_size = 16
    config.training.checkpoint_frequency = 10
    config.training.max_steps = 2000
    config.training.seed = 42
    
    # 优化的损失函数系数（经验调优）
    config.training.loss_cbf_coef = 2.0  # 提高安全重要性
    config.training.loss_velocity_coef = 1.0  # 平衡的速度追踪
    config.training.loss_goal_coef = 3.0  # 强目标导向行为  
    config.training.loss_control_coef = 0.05  # 减少控制惩罚
    config.training.loss_collision_coef = 5.0  # 高碑撞规避
    config.training.loss_safety_coef = 2.5  # 增强安全重要性
    
    # 高级梯度处理
    config.training.gradient_clip_norm = 1.0
    config.training.grad_decay_eta = 0.1  # 为稳定性降低
    config.training.use_gradient_checkpointing = True  # 内存优化
    
    # 性能优化功能
    config.training.performance_tuning = ml_collections.ConfigDict()
    config.training.performance_tuning.enable = True
    config.training.performance_tuning.adaptive_lr_schedule = "warmup_cosine"
    config.training.performance_tuning.adaptive_loss_weights = True
    config.training.performance_tuning.warmup_steps = 200
    config.training.performance_tuning.decay_steps = 1500
    config.training.performance_tuning.weight_update_frequency = 50
    
    # 课程学习（三阶段方法）
    config.training.curriculum = ml_collections.ConfigDict()
    config.training.curriculum.enable = True
    config.training.curriculum.stage1_steps = 300  # 效率优先阶段
    config.training.curriculum.stage2_steps = 400  # 安全感知阶段  
    config.training.curriculum.stage3_steps = 300  # 联合优化
    config.training.curriculum.stage_noise_level = (0.0, 0.02, 0.05)
    config.training.curriculum.annealing_start = 1e-6  # 初始松弛惩罚
    config.training.curriculum.annealing_end = 1e6    # 最终松弛惩罚
    
    # =============================================================================
    # 环境配置
    # =============================================================================
    config.env = ml_collections.ConfigDict()
    
    # 多代理设置
    config.env.num_agents = 1  # 仅支持单无人机
    config.env.area_size = 8.0  # 目标区域半径配置
    config.env.num_obstacles = 16  # 点云采样的障碍点数估计
    config.env.obstacle_types = ["point_cloud"]  # 当前实现基于点云
    
    # LiDAR配置（用于现实世界部署）
    config.env.lidar = ml_collections.ConfigDict()
    config.env.lidar.num_rays = 64
    config.env.lidar.max_range = 6.0
    config.env.lidar.angular_resolution = 360.0 / 64
    
    # 目标配置
    config.env.goal_tolerance = 0.1  # 目标到达容差
    config.env.max_episode_length = 150  # 最大幕集长度
    
    # =============================================================================
    # 评估和日志
    # =============================================================================
    config.evaluation = ml_collections.ConfigDict()
    config.evaluation.eval_frequency = 100  # 评估频率（训练步数）
    config.evaluation.num_eval_episodes = 5  # 每次评估的幕集数
    config.evaluation.success_threshold = 0.95  # 成功率阈值
    config.evaluation.safety_threshold = 1.0   # 要求的安全率
    
    config.logging = ml_collections.ConfigDict()
    config.logging.wandb_project = "safe_agile_flight"  # Weights & Biases项目
    config.logging.log_frequency = 10  # 日志频率
    config.logging.save_frequency = 100  # 模型检查点频率
    config.logging.video_logging = True  # 启用轨迹视频日志
    
    # =============================================================================
    # 计算优化
    # =============================================================================
    config.optimization = ml_collections.ConfigDict()
    
    # JAX特定优化
    config.optimization.use_scan = True  # 为BPTT使用jax.lax.scan
    config.optimization.use_checkpoint = True  # 启用梯度检查点
    config.optimization.checkpoint_strategy = "selective"  # "none", "all", "selective"
    config.optimization.nested_checkpoint = False  # 启用嵌套检查点
    
    # 内存管理
    config.optimization.max_memory_usage = 0.8  # 最大GPU内存占用比
    config.optimization.clear_cache_frequency = 50  # JAX缓存清理频率
    
    # 实验配置
    config.experiment_name = "safe_agile_flight_stage4"
    config.experiment_description = "Complete system integration with GCBF+ and DiffPhysDrone"
    
    return config


def get_minimal_config():
    """返回用于测试和开发的最小配置。"""
    config = get_config()
    
    # 降低测试的计算需求
    config.physics.max_steps = 50
    config.training.max_steps = 100
    config.training.batch_size = 8
    config.training.sequence_length = 10  # 测试内存的较短序列
    config.env.num_agents = 1
    config.gcbf.max_neighbors = 16
    config.gcbf.k_neighbors = 3  # 测试用的较小值
    
    # 禁用测试中的昂贵功能
    config.optimization.use_checkpoint = False
    config.logging.video_logging = False
    config.training.curriculum.enable = False
    
    return config


def get_single_agent_config():
    """返回单代理场景的配置。"""
    config = get_config()
    
    # 单代理修改
    config.env.num_agents = 1
    config.gcbf.max_neighbors = 8  # 仅障碍物作为邻居
    config.gcbf.k_neighbors = 6  # 对单代理合理
    config.env.num_obstacles = 16  # 增加障碍物密度
    
    # 调整单代理情况的训练
    config.training.loss_cbf_coef = 0.5  # 降低CBF重要性
    config.training.batch_size = 32
    
    return config


def get_hardware_config():
    """返回为硬件部署优化的配置。"""
    config = get_config()
    
    # 硬件特定优化
    config.system.precision = "float16"  # 减少内存使用
    config.optimization.use_checkpoint = True  # 内存效率
    config.gcbf.gnn.hidden_dims = [128, 128, 64]  # 较小网络
    config.policy.hidden_dims = [128, 128]  # 较小策略网络
    
    # 实时约束
    config.physics.dt = 1.0/30.0  # 硬件的更高频率
    config.physics.max_steps = 100  # 更短的时间范围
    
    # 禁用硬件的昂贵日志
    config.logging.video_logging = False
    config.evaluation.num_eval_episodes = 1
    
    return config
