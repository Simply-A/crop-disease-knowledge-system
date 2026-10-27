# -*- coding: utf-8 -*-
"""图片推理 —— 加载训练好的权重，对单张/目录图片做病害检测。

用法：
    python training/scripts/infer.py --weights <best.pt> --source <图片或目录> [--conf 0.25]

输出：检测结果（类别、置信度、边界框），并保存画框结果图。
"""

from __future__ import annotations

import argparse
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def main() -> int:
    ap = argparse.ArgumentParser(description="病害检测推理")
    ap.add_argument("--weights", required=True, help="训练好的 .pt 权重")
    ap.add_argument("--source", required=True, help="单张图片或图片目录")
    ap.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    ap.add_argument("--out", default=None, help="结果图输出目录（默认不保存）")
    args = ap.parse_args()

    if not os.path.isfile(args.weights):
        print("权重不存在：%s" % args.weights)
        return 1

    try:
        from ultralytics import YOLO
    except ImportError as e:
        print("缺 ultralytics：%r" % (e,))
        return 1

    model = YOLO(args.weights)
    results = model.predict(
        source=args.source,
        conf=args.conf,
        save=bool(args.out),
        project=args.out or "runs/infer",
        name="pred",
        exist_ok=True,
    )

    print()
    print("=" * 58)
    print("推理结果")
    print("=" * 58)
    for r in results:
        img = os.path.basename(r.path)
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            print("  %-40s 未检出目标" % img)
            continue
        for cls, conf in zip(boxes.cls.tolist(), boxes.conf.tolist()):
            name = model.names.get(int(cls), str(cls))
            print("  %-40s %s  conf=%.3f" % (img, name, conf))
    print()
    print("共处理 %d 张图。" % len(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
