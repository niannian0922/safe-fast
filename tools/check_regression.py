#!/usr/bin/env python3
"""
训练指标回归检测脚本

比如:
    python tools/check_regression.py outputs/validation_stage --max-violation 20 --max-relax 0.05
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def load_metrics(exp_dir: Path) -> Dict[str, float]:
    """为了兼顾 macOS 与 Windows，我们优先使用 pathlib 读取文件，并允许两种平台的路径格式。"""
    metrics_path = exp_dir / "metrics.json"
    if metrics_path.exists():
        with metrics_path.open(encoding="utf-8") as fh:
            records = json.load(fh)
        if records:
            return records[0]

    pkl = exp_dir / "training_results.pkl"
    if pkl.exists():
        with pkl.open("rb") as fh:
            payload = pickle.load(fh)
        history = payload.get("history", [])
        if history:
            return history[-1]
    return {}


def _is_invalid(value: float) -> bool:
    """这里判断数值是否异常，如果发现 NaN 或无穷大，直接认定结果不可用。"""
    return math.isnan(value) or math.isinf(value)


def evaluate_directory(exp_dir: Path, thresholds: dict) -> Tuple[bool, Dict[str, float]]:
    """针对单个实验目录核对关键安全指标，并输出诊断细节。"""
    metrics = load_metrics(exp_dir)
    results: Dict[str, float] = {
        "threshold_max_violation": thresholds["max_violation"],
        "threshold_max_relax": thresholds["max_relax"],
        "threshold_min_success": thresholds["min_success"],
    }
    if not metrics:
        results["error"] = "missing_metrics"
        return False, results

    max_violation = float(metrics.get("eval/max_violation", metrics.get("robust/neural_max_violation", np.nan)))
    relax_mean = float(metrics.get("eval/relax_mean", metrics.get("safety/relaxation_mean", np.nan)))
    success_rate = float(metrics.get("eval/success_rate", metrics.get("robust/blend_success_0", np.nan)))

    results["max_violation"] = max_violation
    results["relax_mean"] = relax_mean
    results["success_rate"] = success_rate

    ok = True
    if _is_invalid(max_violation) or max_violation > thresholds["max_violation"]:
        ok = False
        results["violation_alert"] = max_violation
    if _is_invalid(relax_mean) or relax_mean > thresholds["max_relax"]:
        ok = False
        results["relax_alert"] = relax_mean
    min_success = thresholds["min_success"]
    if min_success is not None:
        if _is_invalid(success_rate) or success_rate < min_success:
            ok = False
            results["success_alert"] = success_rate

    return ok, results


def main():
    parser = argparse.ArgumentParser(description="检测训练输出中的指标是否出现回归。")
    parser.add_argument("paths", nargs="+", type=Path, help="输出目录（需包含 training_results.pkl 或 metrics.json）")
    parser.add_argument("--max-violation", type=float, default=20.0, help="允许的最大违约值")
    parser.add_argument("--max-relax", type=float, default=0.05, help="松弛均值的上线")
    parser.add_argument("--min-success", type=float, default=0.9, help="成功率下限阈值")
    args = parser.parse_args()

    thresholds = {
        "max_violation": args.max_violation,
        "max_relax": args.max_relax,
        "min_success": args.min_success,
    }

    any_failure = False
    summary: List[Dict[str, object]] = []

    for path in args.paths:
        ok, stats = evaluate_directory(path, thresholds)
        record = {"experiment": str(path), **stats, "status": "ok" if ok else "fail"}
        summary.append(record)
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {path}: {stats}")
        if not ok:
            any_failure = True

    summary_path = Path("artifacts") / "regression_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(f"Summary saved to {summary_path}")

    if any_failure:
        exit(1)


if __name__ == "__main__":
    main()
