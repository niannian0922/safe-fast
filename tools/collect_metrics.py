#!/usr/bin/env python3
"""Collect evaluation metrics from multiple experiment directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List
import csv
import pickle


def load_metrics(exp_dir: Path) -> Dict[str, Any]:
    clean_file = exp_dir / "eval_clean.json"
    random_file = exp_dir / "eval_random.json"
    metrics = {"experiment": exp_dir.name}
    if clean_file.exists():
        with clean_file.open() as fh:
            metrics.update({f"clean_{k}": v for k, v in json.load(fh).items()})
    if random_file.exists():
        with random_file.open() as fh:
            metrics.update({f"random_{k}": v for k, v in json.load(fh).items()})
    results_file = exp_dir / "training_results.pkl"
    if results_file.exists():
        with results_file.open("rb") as fh:
            payload = pickle.load(fh)
        history = payload.get("history", [])
        if history:
            last = history[-1]
            for key in (
                "loss/cbf_safe",
                "loss/cbf_hdot",
                "loss/cbf_value",
                "loss/constraint_violation",
                "curriculum/blend_alpha",
                "curriculum/augment",
                "eval/success_rate",
                "eval/relax_mean",
                "eval/max_violation",
            ):
                if key in last:
                    metrics[key] = last[key]
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Collect evaluation metrics")
    parser.add_argument("paths", nargs="+", type=Path, help="Experiment directories")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output")
    parser.add_argument("--csv", type=Path, default=None, help="Optional CSV output")
    args = parser.parse_args()

    records: List[Dict[str, Any]] = []
    for p in args.paths:
        if p.is_dir():
            records.append(load_metrics(p))
        else:
            print(f"warning: {p} is not a directory; skipped")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as fh:
            json.dump(records, fh, indent=2)

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        if records:
            fieldnames = sorted({k for rec in records for k in rec.keys()})
            with args.csv.open("w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                writer.writeheader()
                for rec in records:
                    writer.writerow(rec)

    for rec in records:
        print(rec)


if __name__ == "__main__":
    main()
