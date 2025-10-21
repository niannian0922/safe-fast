#!/usr/bin/env python3
"""汇总 stage_* 结果文件的辅助脚本。

脚本会扫描指定目录（若未提供则默认查找当前工作路径下的 `stage*_results*`），
加载其中的 ``*.pkl`` 文件，并提取终端成功率、最佳距离、最后一次 loss 等关键
指标。默认以可读表格输出；若传入 ``--json`` 则生成机器友好的 JSON。
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable

# 确保在加载 pickle 时能导入自定义 dataclass（例如 TrainingConfig）
try:  # pragma: no cover - soft dependency
    import train_safe_policy  # noqa: F401
except Exception:
    pass


def _to_scalar(value: Any) -> float | int | None:
    """将常见的数值容器转换为 Python 标量。"""

    try:
        if hasattr(value, "__array__"):
            arr = value.__array__()
            if arr.size == 0:
                return None
            return float(arr.reshape(-1)[-1])
        if isinstance(value, (list, tuple)):
            if not value:
                return None
            return float(value[-1])
        if isinstance(value, (int, float)):
            return value
    except Exception:  # pragma: no cover - best effort conversion
        return None
    return None


def summarize_pickle(path: Path) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"file": str(path)}
    try:
        data = pickle.load(path.open("rb"))
    except Exception as exc:  # pragma: no cover
        summary["error"] = f"failed to load: {exc}"
        return summary

    if isinstance(data, dict):
        for key in ("final_success_rate", "overall_success", "best_distance", "total_time"):
            if key in data:
                scalar = _to_scalar(data[key])
                if scalar is not None:
                    summary[key] = float(scalar)

        history = data.get("training_history")
        if isinstance(history, dict):
            for metric in ("loss", "distance", "safety_loss", "grad_norm"):
                if metric in history:
                    scalar = _to_scalar(history[metric])
                    if scalar is not None:
                        summary[f"{metric}_final"] = float(scalar)
            summary["history_keys"] = sorted(history.keys())

        logs = data.get("history")
        if isinstance(logs, list) and logs:
            last = logs[-1]
            for key in (
                "loss/total",
                "loss/efficiency",
                "loss/safety_soft",
                "loss/constraint_violation",
                "loss/grad_norm",
            ):
                if key in last:
                    summary[key] = float(last[key])
            summary["steps"] = len(logs)
    else:
        summary["type"] = type(data).__name__

    return summary


def iter_pickles(root: Path) -> Iterable[Path]:
    if root.is_file() and root.suffix == ".pkl":
        yield root
    elif root.is_dir():
        yield from root.glob("**/*.pkl")


def main() -> None:
    parser = argparse.ArgumentParser(description="汇总 stage_* 结果文件")
    parser.add_argument("paths", nargs="*", help="需要扫描的目录或 pickle 文件")
    parser.add_argument("--json", action="store_true", help="以 JSON 形式输出")
    args = parser.parse_args()

    if args.paths:
        candidates = [Path(p) for p in args.paths]
    else:
        cwd = Path.cwd()
        candidates = sorted(cwd.glob("stage*_results*"))

    summaries = []
    for entry in candidates:
        for pkl in iter_pickles(entry):
            summaries.append(summarize_pickle(pkl))

    if args.json:
        print(json.dumps(summaries, ensure_ascii=False, indent=2))
    else:
        for item in summaries:
            file = item.pop("file", "<unknown>")
            print(f"\n{file}")
            if "error" in item:
                print(f"  error: {item['error']}")
                continue
            for key, value in item.items():
                if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                    value = "nan"
                print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
