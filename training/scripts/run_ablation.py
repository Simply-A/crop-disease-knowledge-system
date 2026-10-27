# -*- coding: utf-8 -*-
"""顺序跑完 4 组消融实验，训练后汇总指标对比表。

用法：
    python training/scripts/run_ablation.py

每组训练 100 轮，共约 30 分钟；结果写入 D:/bishu/runs/<name>/。
训练完成后从 results.csv 提取 best（mAP50 最大轮）指标，打印汇总表。
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

RUNS_DIR = "D:/bishu/runs"

ABLATIONS = [
    ("A2  P2 小目标层", "ablation_p2", "training/configs/ablation_p2.yaml"),
    ("A1  CBAM 注意力", "ablation_cbam", "training/configs/ablation_cbam.yaml"),
    ("A3  Ghost 骨干", "ablation_ghost", "training/configs/ablation_ghost.yaml"),
    ("A4  三项叠加", "ablation_full", "training/configs/ablation_full.yaml"),
]


def best_metrics(results_csv: str):
    """从 results.csv 提取 mAP50 最大那一轮的指标。"""
    best = None
    with open(results_csv, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                m50 = float(row["metrics/mAP50(B)"])
            except (KeyError, ValueError):
                continue
            if best is None or m50 > best[0]:
                best = (m50, float(row.get("metrics/mAP50-95(B)", 0)),
                        float(row.get("metrics/precision(B)", 0)),
                        float(row.get("metrics/recall(B)", 0)))
    return best


def main() -> int:
    py = sys.executable
    summary = []

    for label, name, config in ABLATIONS:
        run_dir = os.path.join(RUNS_DIR, name)
        print("\n" + "=" * 62)
        print("消融实验：%s" % label)
        print("=" * 62)
        cmd = [py, "training/scripts/train.py", "--config", config,
               "--pretrained", "training/weights/yolov8n.pt"]
        print("执行：%s" % " ".join(cmd))
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            print("!! %s 训练失败（退出码 %d）" % (name, rc))
            summary.append((label, name, None))
            continue

        csv_path = os.path.join(run_dir, "results.csv")
        m = best_metrics(csv_path) if os.path.isfile(csv_path) else None
        if m:
            print("  best: mAP50=%.4f  mAP50-95=%.4f  P=%.4f  R=%.4f" % m)
            summary.append((label, name, m))
        else:
            summary.append((label, name, None))

    print("\n" + "=" * 62)
    print("消融实验汇总（best 轮指标）")
    print("=" * 62)
    print("%-18s %10s %10s %10s %10s" % ("模型", "mAP50", "mAP50-95", "P", "R"))
    print("-" * 62)
    for label, name, m in summary:
        if m:
            print("%-18s %10.4f %10.4f %10.4f %10.4f" % (label, m[0], m[1], m[2], m[3]))
        else:
            print("%-18s %10s" % (label, "失败"))
    print("=" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())
