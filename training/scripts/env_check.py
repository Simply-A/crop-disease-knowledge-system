# -*- coding: utf-8 -*-
"""
环境自检 —— 训练前必须先跑通这个脚本。

    python code/scripts/env_check.py

重点看三行：
    CUDA 可用 : True
    设备      : NVIDIA GeForce RTX 4060 Laptop GPU
    ultralytics: 8.x.x
任何一项不对都不要开始训练（CPU 训练会慢 30 倍且不报错，很容易白跑一晚上）。
"""

import os
import sys
import shutil
import platform

# Windows 控制台默认 GBK 代码页，固定成 UTF-8，否则中文输出可能乱码
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

OK = "[OK]"
NO = "[!!]"
WARN = "[?]"

problems = []
warnings = []


def line(k, v):
    print("  %-14s: %s" % (k, v))


def head(t):
    print()
    print("=" * 58)
    print(t)
    print("=" * 58)


head("1. Python 解释器")
line("可执行文件", sys.executable)
line("版本", platform.python_version())
in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
line("虚拟环境", ("是 (%s)" % sys.prefix) if in_venv else "否 —— 建议建 venv")
if not in_venv:
    warnings.append("当前不在虚拟环境里，依赖可能装到全局。建议用 D:\\bishu\\venvs\\yolo。")
if sys.version_info[:2] >= (3, 14):
    warnings.append("Python %s 过新，ultralytics 官方 CI 只覆盖到 3.12，可能遇到兼容问题。"
                    % platform.python_version())

head("2. 深度学习框架")
try:
    import torch
    line("torch", torch.__version__)
except ImportError:
    line("torch", "未安装")
    problems.append("没有 torch。先执行：pip install torch torchvision --index-url "
                    "https://download.pytorch.org/whl/cu128")
    torch = None

if torch is not None:
    try:
        import torchvision
        line("torchvision", torchvision.__version__)

        # 版本配对：torch 2.x 官方配套 torchvision 0.(x+15)
        #   2.7↔0.22  2.8↔0.23  2.9↔0.24  2.10↔0.25  2.11↔0.26
        # 按偏移量算，不做死表 —— 否则每出一个新版本这里就误报一次。
        def _mm(v):
            return [int(x) for x in v.split("+")[0].split(".")[:2]]
        tv_mm, t_mm = _mm(torchvision.__version__), _mm(torch.__version__)
        if tv_mm[0] != 0 or t_mm[0] != 2 or tv_mm[1] != t_mm[1] + 15:
            warnings.append("torch %s 与 torchvision %s 不是官方配套版本，"
                            "可能出现 DLL 加载失败。" % (torch.__version__, torchvision.__version__))

        # 比版本号终究是猜 —— 直接调一个 C++ 扩展算子，跑得通才算真能用
        try:
            from torchvision.ops import nms
            _keep = nms(torch.tensor([[0., 0., 10., 10.], [1., 1., 11., 11.]]),
                        torch.tensor([0.9, 0.8]), 0.5)
            line("torchvision 实测", "通过 (nms -> %s)" % _keep.tolist())
        except Exception as e:
            problems.append("torchvision 装上了但算不了（%r）。版本配对有问题，"
                            "按官方配套重装 torch + torchvision。" % (e,))
    except ImportError:
        line("torchvision", "未安装")
        problems.append("缺 torchvision。执行：pip install torchvision --index-url "
                        "https://download.pytorch.org/whl/cu128")
    cuda_ok = torch.cuda.is_available()
    line("CUDA 可用", str(cuda_ok))
    line("CUDA 版本", str(torch.version.cuda))
    if cuda_ok:
        line("设备", torch.cuda.get_device_name(0))
        p = torch.cuda.get_device_properties(0)
        vram = p.total_memory / 1024 ** 3
        line("显存", "%.1f GB" % vram)
        line("算力", "sm_%d%d" % (p.major, p.minor))
        if vram < 6:
            warnings.append("显存只有 %.1f GB，训练时把 batch 调到 8 或 4，并开启 amp。" % vram)
            print("  %s 显存偏小，建议 batch=8 + amp=True" % WARN)
        else:
            print("  %s 显存够用，YOLOv8n @ 640 可用 batch=16" % OK)
        # 实际跑一次矩阵运算，确认驱动真的能work
        try:
            a = torch.randn(512, 512, device="cuda")
            b = (a @ a).sum().item()
            line("GPU 实测", "通过 (结果 %.2f)" % b)
        except Exception as e:
            line("GPU 实测", "失败: %r" % (e,))
            problems.append("CUDA 报告可用但实际运算失败，通常是驱动与 CUDA 运行时不匹配。")
    else:
        if "+cpu" in torch.__version__:
            problems.append("装的是 CPU 版 torch（版本号带 +cpu）。必须卸载重装 GPU 版，"
                            "否则训练会跑在 CPU 上。")
        else:
            problems.append("torch 检测不到 CUDA。检查 NVIDIA 驱动，或换 cu126 的 wheel 重装。")

head("3. 其余依赖")
for mod, name in [("ultralytics", "ultralytics"), ("cv2", "opencv-python"),
                  ("numpy", "numpy"), ("yaml", "pyyaml"),
                  ("matplotlib", "matplotlib"), ("PIL", "pillow"),
                  ("pandas", "pandas")]:
    try:
        m = __import__(mod)
        line(name, getattr(m, "__version__", "已安装"))
    except ImportError:
        line(name, "未安装")
        if mod == "ultralytics":
            problems.append("没有 ultralytics。执行：pip install -r code/requirements.txt")
        else:
            warnings.append("缺 %s，ultralytics 多半会自动装，必要时手动补。" % name)

if not problems:
    try:
        from ultralytics import YOLO  # noqa: F401
        line("ultralytics 导入", "正常")
    except Exception as e:
        line("ultralytics 导入", "失败: %r" % (e,))
        problems.append("ultralytics 装上了但导入失败，多半是依赖版本冲突，看上面的报错。")

head("4. 磁盘与路径")
for d, label in [("C:\\", "C盘"), ("D:\\", "D盘")]:
    if os.path.exists(d):
        t = shutil.disk_usage(d)
        line(label, "剩余 %.1f GB / 共 %.1f GB" % (t.free / 1024 ** 3, t.total / 1024 ** 3))
        if label == "C盘" and t.free / 1024 ** 3 < 30:
            warnings.append("C 盘剩余不足 30 GB，数据集和虚拟环境都放 D 盘。")

for d, label in [(r"D:\bishu\datasets", "数据集根目录"),
                 (r"C:\Users\今又是\Desktop\毕设2\code\runs", "训练产出目录")]:
    exists = os.path.exists(d)
    line(label, ("存在  %s" % d) if exists else ("不存在  %s" % d))
    if not exists and label == "数据集根目录":
        warnings.append("还没有 %s，先建目录再放数据。" % d)

head("5. 数据集现状")
found = []
for root in [r"D:\bishu\datasets", r"C:\Users\今又是\Desktop\毕设2\code\datasets"]:
    if not os.path.isdir(root):
        continue
    for name in os.listdir(root):
        p = os.path.join(root, name)
        if not os.path.isdir(p):
            continue
        n_img = n_lbl = 0
        for r2, _, fs in os.walk(p):
            for f in fs:
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    n_img += 1
                elif f.lower().endswith(".txt"):
                    n_lbl += 1
        found.append((p, n_img, n_lbl))
if found:
    for p, i, l in found:
        line(os.path.basename(p), "图 %d 张 / 标注 %d 个%s" % (i, l, "" if l else "  ← 没有标注，是分类数据集"))
else:
    print("  （还没有数据集）")
    warnings.append("数据集为空。先用 prepare_data.py 整理数据，或先下 Roboflow 的小数据集跑通流程。")

head("结论")
if problems:
    print("%s 有 %d 个必须解决的问题：" % (NO, len(problems)))
    for i, p in enumerate(problems, 1):
        print("   %d) %s" % (i, p))
    print()
    print("   修完再跑一次本脚本，直到「必须解决」为空。")
else:
    print("%s 环境就绪，可以开始训练。" % OK)
    if warnings:
        print()
        print("%s %d 条提醒（不阻塞）：" % (WARN, len(warnings)))
        for i, w in enumerate(warnings, 1):
            print("   %d) %s" % (i, w))

print()
print("下一步：python code/scripts/prepare_data.py scan --src <你的数据目录>")
sys.exit(1 if problems else 0)
