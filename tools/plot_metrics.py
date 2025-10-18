#!/usr/bin/env python3
"""Generate plots from curriculum summary metrics."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_metrics(df: pd.DataFrame, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    df.plot.bar(x="experiment", y="random_violation_max", ax=axes[0], legend=False)
    axes[0].set_ylabel("Random Max Violation")

    df.plot.bar(x="experiment", y="loss/cbf_hdot", ax=axes[1], legend=False)
    axes[1].set_ylabel("loss/cbf_hdot")

    df.plot.bar(x="experiment", y="eval/success_rate", ax=axes[2], legend=False)
    axes[2].set_ylabel("Eval Success Rate")

    plt.tight_layout()
    fig.savefig(output_dir / "metrics_overview.png", dpi=200)


def main():
    parser = argparse.ArgumentParser(description="Plot curriculum metrics")
    parser.add_argument("summary", type=Path, help="Path to curriculum_summary.json")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/plots"))
    args = parser.parse_args()

    with args.summary.open() as fh:
        records = json.load(fh)

    df = pd.DataFrame(records)
    plot_metrics(df, args.output_dir)


if __name__ == "__main__":
    main()

