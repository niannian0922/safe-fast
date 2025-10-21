#!/usr/bin/env python3
"""
策略蒸馏 + 课程调度实验模板,该脚本批量生成策略蒸馏相关实验命令，可用于探索不同冻结步数、蒸馏权重、噪声课程的组合。默认仅打印命令，添加 `--execute` 真正运行。
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "train_safe_policy.py"


DISTILL_PRESETS = [
    {
        "name": "distill_light",
        "args": [
            "--stage-steps",
            "80,80,80",
            "--noise-levels",
            "0.0,0.02,0.04",
            "--policy-freeze-steps",
            "60",
            "--distill-weight",
            "0.2",
            "--cbf-blend-alpha",
            "0.6",
        ],
    },
    {
        "name": "distill_heavy",
        "args": [
            "--stage-steps",
            "120,120,120",
            "--noise-levels",
            "0.0,0.03,0.06",
            "--policy-freeze-steps",
            "120",
            "--distill-weight",
            "0.4",
            "--cbf-blend-alpha",
            "0.8",
            "--robust-eval-frequency",
            "40",
        ],
    },
    {
        "name": "distill_stop_grad",
        "args": [
            "--stage-steps",
            "100,80,60",
            "--noise-levels",
            "0.0,0.02,0.05",
            "--policy-freeze-steps",
            "80",
            "--distill-weight",
            "0.3",
            "--cbf-blend-alpha",
            "1.0",
            "--blend-backoff",
            "0.15",
        ],
    },
]


def build_command(preset: Dict, output_root: Path, episodes: int, horizon: int, distill_policy: str | None):
    name = preset["name"]
    out_dir = output_root / name
    cmd = [
        "python",
        str(TRAIN_SCRIPT),
        "--episodes",
        str(episodes),
        "--horizon",
        str(horizon),
        "--output-dir",
        str(out_dir),
    ]
    cmd.extend(preset["args"])
    if distill_policy:
        cmd.extend(["--distill-policy", distill_policy])
    return cmd


def main():
    parser = argparse.ArgumentParser(description="批量生成蒸馏实验命令")
    parser.add_argument("--episodes", type=int, default=240, help="每个实验的训练轮数")
    parser.add_argument("--horizon", type=int, default=60, help="单次 rollout 的步长")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "distill_sweep", help="输出根目录")
    parser.add_argument("--distill-policy", type=str, default=os.environ.get("DISTILL_POLICY_PATH"), help="教师策略参数路径")
    parser.add_argument("--execute", action="store_true", help="直接执行命令而非只打印")
    parser.add_argument("--fast", action="store_true", help="快速冒烟模式")
    args = parser.parse_args()

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    episodes = args.episodes
    horizon = args.horizon
    if args.fast:
        episodes = min(episodes, 40)
        horizon = min(horizon, 40)

    for preset in DISTILL_PRESETS:
        cmd = build_command(preset, output_root, episodes, horizon, args.distill_policy)
        print(" ".join(cmd))
        if args.execute:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
