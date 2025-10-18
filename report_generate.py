#!/usr/bin/env python3
"""
生成安全敏捷飞行阶段性报告
==========================

- 默认汇总 `tools/run_safety_suite.py` 与阶段验证 (`validation_stage`) 的结果；
- 自动导入 CSV/JSON 指标和 PNG 可视化，写入单一 PDF；
- 可通过 CLI 指定目录或输出路径。
"""

from __future__ import annotations

import argparse
import json
import pickle
import os
from typing import Optional, Sequence

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd


def add_csv_plot(pdf: PdfPages, csv_path: str, title: str, ycols: Sequence[str]):
    if not os.path.exists(csv_path):
        return
    df = pd.read_csv(csv_path)
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    for c in ycols:
        if c in df.columns:
            ax.plot(df["step"] if "step" in df.columns else df.index, df[c], label=c)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def add_image(pdf: PdfPages, img_path: str, title: Optional[str] = None):
    if not os.path.exists(img_path):
        return
    img = plt.imread(img_path)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.imshow(img)
    ax.axis("off")
    if title:
        ax.set_title(title)
    pdf.savefig(fig)
    plt.close(fig)


def add_table(pdf: PdfPages, csv_path: str, title: str, nrows: int = 10):
    if not os.path.exists(csv_path):
        return
    df = pd.read_csv(csv_path)
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    ax.set_title(title)
    tbl = ax.table(cellText=df.head(nrows).values, colLabels=df.columns, loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.2)
    pdf.savefig(fig)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate consolidated PDF report.")
    parser.add_argument(
        "--suite-dir",
        type=str,
        default=os.path.join("outputs", "safety_suite"),
        help="路径内若存在 suite_summary.csv/json/图像，将自动纳入报告。",
    )
    parser.add_argument(
        "--validation-dir",
        type=str,
        default=os.path.join("outputs", "validation_stage"),
        help="阶段二验证产物目录（训练结果 + 可视化）。",
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        default=os.path.join("outputs", "plots"),
        help="可视化图像目录。",
    )
    parser.add_argument("--output", type=str, default="artifacts/report.pdf")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    suite_summary_csv = os.path.join(args.suite_dir, "suite_summary.csv")
    suite_plot = os.path.join(args.plots_dir, "metrics_overview.png")
    validation_metrics = os.path.join(args.plots_dir, "validation_stage_metrics.png")
    validation_history = os.path.join(args.plots_dir, "validation_stage_history.png")
    validation_pickle = os.path.join(args.validation_dir, "training_results.pkl")
    validation_metrics_csv = os.path.join(args.validation_dir, "metrics.csv")

    with PdfPages(args.output) as pdf:
        add_table(pdf, suite_summary_csv, "安全训练实验 (Suite) 总览", nrows=15)
        add_image(pdf, suite_plot, "课程实验指标概览")
        add_image(pdf, validation_metrics, "验证阶段：违约/损失指标")
        add_image(pdf, validation_history, "验证阶段：损失与惩罚缩放")
        add_table(pdf, validation_metrics_csv, "验证阶段指标", nrows=10)

        # 如果存在 training_results.pkl，则简单统计 loss 序列
        if os.path.exists(validation_pickle):
            with open(validation_pickle, "rb") as fh:
                payload = pickle.load(fh)
            history = payload.get("history", [])
            if history:
                df = pd.DataFrame(history)
                cols = [c for c in df.columns if c.startswith("loss/")]
                if cols:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    df[cols].plot(ax=ax)
                    ax.set_title("验证阶段 Loss 曲线")
                    ax.set_xlabel("Episode")
                    plt.tight_layout()
                    pdf.savefig(fig)
                    plt.close(fig)

    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
