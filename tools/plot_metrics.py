#!/usr/bin/env python3
"""根据课程汇总指标生成图表。"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_metrics(df: pd.DataFrame, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    df.plot.bar(x="experiment", y="random_violation_max", ax=axes[0], legend=False)
    axes[0].set_ylabel("随机扰动下的最大违约")

    df.plot.bar(x="experiment", y="loss/cbf_hdot", ax=axes[1], legend=False)
    axes[1].set_ylabel("loss/cbf_hdot")

    df.plot.bar(x="experiment", y="eval/success_rate", ax=axes[2], legend=False)
    axes[2].set_ylabel("评估成功率")

    plt.tight_layout()
    fig.savefig(output_dir / "metrics_overview.png", dpi=200)


def main():
    parser = argparse.ArgumentParser(description="绘制课程指标图表")
    parser.add_argument("summary", type=Path, help="curriculum_summary.json 的路径")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/plots"), help="输出目录")
    args = parser.parse_args()

    with args.summary.open() as fh:
        records = json.load(fh)

    df = pd.DataFrame(records)
    plot_metrics(df, args.output_dir)


if __name__ == "__main__":
    main()
