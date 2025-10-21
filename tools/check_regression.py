#!/usr/bin/env python3
"""
训练指标回归检测脚本

比如:
    python tools/check_regression.py outputs/validation_stage --max-violation 20 --max-relax 0.05
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def load_metrics(exp_dir: Path) -> Dict[str, float]:
    metrics_path = exp_dir / "metrics.json"
    if metrics_path.exists():
        with metrics_path.open() as fh:
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


def evaluate_directory(exp_dir: Path, thresholds: dict) -> Tuple[bool, Dict[str, float]]:
    metrics = load_metrics(exp_dir)
    ok = True
    results = {}

    if not metrics:
        return False, {"error": "no_metrics"}

    max_violation = float(metrics.get("eval/max_violation", metrics.get("robust/neural_max_violation", 0.0)))
    relax_mean = float(metrics.get("eval/relax_mean", metrics.get("safety/relaxation_mean", 0.0)))
    success_rate = float(metrics.get("eval/success_rate", metrics.get("robust/blend_success_0", 0.0)))

    results["max_violation"] = max_violation
    results["relax_mean"] = relax_mean
    results["success_rate"] = success_rate

    if max_violation > thresholds["max_violation"]:
        ok = False
        results["violation_alert"] = max_violation
    if relax_mean > thresholds["max_relax"]:
        ok = False
        results["relax_alert"] = relax_mean
    if thresholds["min_success"] is not None and success_rate < thresholds["min_success"]:
        ok = False
        results["success_alert"] = success_rate

    return ok, results


def main():
    parser = argparse.ArgumentParser(description="检测训练输出中的指标是否出现回归。")
    parser.add_argument("paths", nargs="+", type=Path, help="输出目录（需包含 training_results.pkl 或 metrics.json）")
    parser.add_argument("--max-violation", type=float, default=20.0, help="允许的最大违约值")
    parser.add_argument("--max-relax", type=float, default=0.05, help="松弛均值的上限")
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
    with summary_path.open("w") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(f"Summary saved to {summary_path}")

    if any_failure:
        exit(1)


if __name__ == "__main__":
    main()
