# -*- coding: utf-8 -*-
"""验证各模型 yaml 能正确构建，输出层数/参数量/GFLOPs 用于消融对比。

用法：
    python training/scripts/verify_models.py
"""

from __future__ import annotations

import importlib.util
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(HERE, "models")


def register_custom():
    p = os.path.join(MODELS, "custom_modules.py")
    if not os.path.isfile(p):
        return
    spec = importlib.util.spec_from_file_location("crop_custom_modules", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.register_modules()


def main() -> int:
    try:
        import torch
        from ultralytics import YOLO
        from ultralytics.utils.torch_utils import get_num_params, model_info
    except ImportError as e:
        print("缺依赖：%r" % (e,))
        return 1

    register_custom()

    specs = [
        ("基线 yolov8n", "training/weights/yolov8n.pt"),
        ("改进② P2 层", os.path.join(MODELS, "yolov8n-p2.yaml")),
        ("改进① CBAM", os.path.join(MODELS, "yolov8n-cbam.yaml")),
        ("改进③ Ghost", os.path.join(MODELS, "yolov8n-ghost.yaml")),
        ("组合 P2+CBAM+Ghost", os.path.join(MODELS, "yolov8n-p2-cbam-ghost.yaml")),
    ]

    print("=" * 62)
    print("%-16s %8s %12s %12s" % ("模型", "层数", "参数量", "GFLOPs"))
    print("=" * 62)
    for label, spec in specs:
        try:
            m = YOLO(spec)
            n_params = sum(p.numel() for p in m.model.parameters())
            n_layers = len(list(m.model.modules()))
            # GFLOPs 通过 info() 获取（只对 torch 模型有效）
            gflops = "-"
            try:
                info = m.info(verbose=False)
                # info() 返回字典，含 'GFLOPs'
                gflops = "%.1f" % info.get("GFLOPs", 0)
            except Exception:
                pass
            print("%-16s %8d %12s %12s" % (label, n_layers, f"{n_params:,}", gflops))
        except Exception as e:
            print("%-16s 构建失败：%r" % (label, e))
    print("=" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())
