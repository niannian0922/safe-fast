#!/usr/bin/env python3
"""
安全训练批处理脚本
该脚本用于执行一组预定义的安全训练实验，自动化完成以下流程：
构造训练命令（可自定义 episodes / 噪声 / 混合策略等参数）；
支持 `--dry-run` 打印命令而不真正执行，方便先审查；
每个实验结束后调用 `tools/collect_metrics.py`、`tools/stage_summary.py`以及生成快速可视化；
将所有指标写入单一 JSON/CSV，便于后续汇报。

比如：
    python tools/run_safety_suite.py --output-root outputs/safety_suite --fast
    python tools/run_safety_suite.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "train_safe_policy.py"
COLLECT_SCRIPT = ROOT / "tools" / "collect_metrics.py"
SUMMARY_SCRIPT = ROOT / "tools" / "stage_summary.py"


EXPERIMENT_PRESETS = {
    "analytic_baseline": {
        "name": "analytic_baseline",
        "args": [
            "--disable-curriculum",
            "--cbf-blend-alpha",
            "0.0",
        ],
    },
    "hybrid_moderate": {
        "name": "hybrid_moderate",
        "args": [
            "--noise-levels",
            "0.0,0.02,0.05",
            "--stage-steps",
            "100,100,100",
            "--robust-eval-frequency",
            "{robust_freq}",
            "--robust-eval-noise",
            "0.0,0.03,0.06",
        ],
    },
    "hybrid_aggressive": {
        "name": "hybrid_aggressive",
        "args": [
            "--noise-levels",
            "0.0,0.03,0.06",
            "--stage-steps",
            "120,120,120",
            "--cbf-blend-alpha",
            "1.0",
            "--robust-eval-frequency",
            "{robust_freq}",
            "--relax-boost",
            "2.0",
            "--solver-boost",
            "1.5",
        ],
    },
}


def build_command(
    preset: Dict[str, str], episodes: int, horizon: int, robust_freq: int, output_root: Path
) -> List[str]:
    name = preset["name"]
    out_dir = str(output_root / name)
    args = [
        "python",
        str(TRAIN_SCRIPT),
        "--episodes",
        str(episodes),
        "--horizon",
        str(horizon),
        "--output-dir",
        out_dir,
    ]

    substitutions = {
        "episodes": str(episodes),
        "horizon": str(horizon),
        "robust_freq": str(robust_freq),
        "out_dir": out_dir,
    }

    for token in preset["args"]:
        if token.startswith("{") and token.endswith("}"):
            key = token[1:-1]
            args.append(substitutions[key])
        else:
            args.append(token)
    return args


def run_command(cmd: List[str]) -> None:
    print(">>>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def collect_artifacts(exp_dir: Path, summary_rows: List[Dict]) -> None:
    if not exp_dir.exists():
        print(f"[警告] 未找到 {exp_dir}，已跳过指标收集。")
        return

    collect_cmd = [
        "python",
        str(COLLECT_SCRIPT),
        str(exp_dir),
        "--output",
        str(exp_dir / "metrics.json"),
        "--csv",
        str(exp_dir / "metrics.csv"),
    ]
    summary_cmd = [
        "python",
        str(SUMMARY_SCRIPT),
        str(exp_dir),
        "--json",
    ]
    run_command(collect_cmd)
    run_command(summary_cmd)

    metrics_json = exp_dir / "metrics.json"
    if metrics_json.exists():
        with metrics_json.open() as fh:
            records = json.load(fh)
        if records:
            summary_rows.append(records[0])


def main():
    parser = argparse.ArgumentParser(description="执行预定义的安全训练实验。")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs" / "safety_suite",
        help="用于存放实验输出的目录。",
    )
    parser.add_argument("--episodes", type=int, default=300, help="每个实验的训练轮数")
    parser.add_argument("--horizon", type=int, default=60, help="单次 rollout 的步长")
    parser.add_argument("--robust-frequency", type=int, default=50, help="鲁棒性评估的频率")
    parser.add_argument(
        "--only",
        type=str,
        nargs="*",
        default=None,
        help="仅执行指定 key 的实验预设",
    )
    parser.add_argument("--dry-run", action="store_true", help="仅打印命令，不实际执行")
    parser.add_argument("--fast", action="store_true", help="使用较小的 episodes/horizon 进行冒烟测试")
    args = parser.parse_args()

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    episodes = args.episodes
    horizon = args.horizon
    robust_freq = args.robust_frequency
    if args.fast:
        episodes = min(episodes, 30)
        horizon = min(horizon, 30)
        robust_freq = max(5, min(robust_freq, 10))

    presets = EXPERIMENT_PRESETS
    if args.only:
        presets = {k: EXPERIMENT_PRESETS[k] for k in args.only}

    summary_rows: List[Dict] = []
    for key, preset in presets.items():
        exp_dir = output_root / preset["name"]
        cmd = build_command(preset, episodes, horizon, robust_freq, output_root)
        if args.dry_run:
            print("[dry-run]", " ".join(cmd))
            continue
        run_command(cmd)
        collect_artifacts(exp_dir, summary_rows)

    if summary_rows:
        df = pd.DataFrame(summary_rows)
        summary_json = output_root / "suite_summary.json"
        summary_csv = output_root / "suite_summary.csv"
        df.to_json(summary_json, orient="records", indent=2)
        df.to_csv(summary_csv, index=False)
        print(f"[done] Summary saved to {summary_json} / {summary_csv}")


if __name__ == "__main__":
    main()
