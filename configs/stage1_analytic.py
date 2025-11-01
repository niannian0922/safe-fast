"""
阶段一专用配置

这个配置文件针对“解析 CBF + 零噪声”的安全基线训练做了精简，方便在 macOS、Windows 和 Kaggle 三端保持一致。
"""

from __future__ import annotations

from ml_collections import ConfigDict

from configs import default_config


def get_config() -> ConfigDict:
    """在默认配置基础上做有限度的改动，保持两端环境同步。"""
    cfg = default_config.get_config()

    cfg.system.seed = 2025

    cfg.physics.max_steps = 60
    cfg.physics.gradient_decay.alpha = 0.92
    cfg.physics.control.max_thrust = 1.0

    cfg.training.max_steps = 200
    cfg.training.curriculum.enable = True

    cfg.training.curriculum.stage1_steps = 200
    cfg.training.curriculum.stage2_steps = 150
    cfg.training.curriculum.stage3_steps = 150
    cfg.training.curriculum.stage_noise_level = (0.0, 0.01, 0.02)
    cfg.training.learning_rate_policy = 3e-4
    cfg.training.learning_rate_gcbf = 3e-4
    cfg.training.loss_safety_coef = 2.0
    cfg.training.loss_collision_coef = 3.0
    cfg.training.loss_control_coef = 0.2
    cfg.training.relaxation_alert = 0.2
    cfg.training.relax_penalty_boost = 1.2
    cfg.training.relax_penalty_max = 6.0
    cfg.training.policy_distill_weight = 0.2
    cfg.training.success_patience = 5
    cfg.training.hard_nan_rate = 0.1
    cfg.training.hard_relaxation_exceed_rate = 0.25
    cfg.training.hard_qp_fail_rate = 0.25
    cfg.training.point_cloud_modes = ["ring", "cylinder", "mixed"]
    cfg.training.relaxation_scale_schedule = [0.5, 1.0, 1.2]
    cfg.training.solver_scale_schedule = [0.8, 1.0, 1.2]
    cfg.training.hard_abort_warmup_episodes = 200
    cfg.training.hard_nan_schedule = [0.4, 0.2, 0.1]
    cfg.training.hard_relax_schedule = [0.3, 0.2, 0.1]
    cfg.training.hard_qp_fail_schedule = [0.4, 0.25, 0.15]

    cfg.env.goal_tolerance = 0.3

    cfg.gcbf.gamma = 0.05

    cfg.policy.use_rnn = False
    cfg.policy.action_limit = 0.6

    cfg.safety.alpha0 = 2.0
    cfg.safety.alpha1 = 8.0
    cfg.safety.max_acceleration = 1.2
    cfg.safety.max_relaxation = 5.0
    cfg.safety.relaxation_penalty = 200.0
    cfg.safety.relaxation_alert = 0.4

    cfg.evaluation.success_threshold = 0.8

    return cfg
