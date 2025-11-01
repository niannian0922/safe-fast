"""
效率阶段专用配置

该配置聚焦于让策略在关闭安全层的情况下快速学会从近距离启航，
并通过课程式的目标距离与扰动逐步提升难度。
"""

from __future__ import annotations

from ml_collections import ConfigDict

from configs import default_config


def get_config() -> ConfigDict:
    cfg = default_config.get_config()

    cfg.system.seed = 2025

    # 训练课程（距离由近到远，逐步引入噪声）
    cfg.training.curriculum.enable = True
    cfg.training.curriculum.stage1_steps = 300
    cfg.training.curriculum.stage2_steps = 300
    cfg.training.curriculum.stage3_steps = 300
    cfg.training.curriculum.stage_noise_level = (0.0, 0.003, 0.008)

    # 调整损失权重以强调快速到达
    cfg.training.loss_goal_coef = 6.0
    cfg.training.loss_control_coef = 0.015
    cfg.training.loss_collision_coef = 0.0
    cfg.training.loss_safety_coef = 0.0  # 效率阶段关闭安全损失

    cfg.training.policy_distill_weight = 0.0
    cfg.training.relaxation_alert = 0.2  # 虽未启用安全层，保持一致设置

    # 目标配置：基础目标用于计算方向，距离将由课程 schedule 覆盖
    cfg.env.goal_tolerance = 0.2
    cfg.evaluation.success_threshold = 0.95

    # 物理/策略约束稍作收紧，鼓励平稳加速
    cfg.physics.max_steps = 80
    cfg.physics.control.max_thrust = 1.2
    cfg.policy.action_limit = 1.0

    # 课程专用设置
    cfg.training.point_cloud_modes = ["ring", "ring", "cylinder"]
    cfg.training.target_distance_schedule = [1.2, 1.8, 2.6]
    cfg.training.relaxation_scale_schedule = [1.0, 1.0, 1.0]
    cfg.training.solver_scale_schedule = [1.0, 1.0, 1.0]
    cfg.training.initial_xy_range = 0.08
    cfg.training.initial_z_range = (0.95, 1.05)
    cfg.training.teacher_gain_p = 2.0
    cfg.training.teacher_gain_d = 0.9
    cfg.training.teacher_weight = 1.2
    cfg.training.velocity_alignment_weight = 0.0
    cfg.training.desired_speed = 0.6
    cfg.training.final_velocity_weight = 0.7
    cfg.training.distance_bonus_weight = 0.25
    cfg.training.distance_bonus_threshold = 0.2
    cfg.training.trajectory_projection_weight = 0.4
    cfg.training.distance_tracking_weight = 1.0
    cfg.training.final_distance_weight = 5.0
    cfg.training.learning_rate_policy = 5e-5

    return cfg
