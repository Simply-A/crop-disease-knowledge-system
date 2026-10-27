# -*- coding: utf-8 -*-
"""
训练入口 —— 读 YAML 配置，调 ultralytics。

    python code/scripts/train.py --config code/configs/baseline.yaml
    python code/scripts/train.py --config code/configs/ablation_p2.yaml --epochs 50

命令行参数会覆盖配置文件里的同名项，方便临时试跑。
"""

import argparse
import importlib.util
import os
import sys
import time
import urllib.request

# Windows 控制台默认 GBK 代码页，固定成 UTF-8，否则中文输出可能乱码
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# 预训练权重缓存目录（与 code/ 同级）
WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights")

# ultralytics 的默认下载源是 github.com，国内直连必超时（curl 28/35）。
# 按实测可用性排序，末尾空串 = 直连 GitHub 兜底。
GITHUB_MIRRORS = ["https://ghfast.top/", "https://ghproxy.net/", "https://gh-proxy.com/", ""]


def _release_tag():
    """ultralytics 8.4.155 -> 资源发布标签 v8.4.0"""
    try:
        from ultralytics import __version__
        mm = __version__.split(".")[:2]
        return "v%s.%s.0" % (mm[0], mm[1])
    except Exception:
        return "v8.4.0"


def resolve_model(name):
    """把 'yolov8n.pt' 解析成本地可用路径；本地没有就走镜像下载，绕开 GitHub。

    只有 .pt 会被拦截 —— .yaml（从结构建模型）和绝对路径原样交给 ultralytics。
    """
    base = os.path.basename(name)
    if not base.endswith(".pt"):
        return name

    dst = os.path.join(WEIGHTS_DIR, base)
    if os.path.isfile(dst):
        print("  %-10s: %s（已存在）" % ("权重", dst))
        return dst

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    url = "https://github.com/ultralytics/assets/releases/download/%s/%s" % (_release_tag(), base)
    for mirror in GITHUB_MIRRORS:
        who = mirror or "直连 GitHub"
        try:
            req = urllib.request.Request(mirror + url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as r, open(dst, "wb") as f:
                while True:
                    chunk = r.read(1 << 16)
                    if not chunk:
                        break
                    f.write(chunk)
            size = os.path.getsize(dst)
            if size > 100_000:                      # 小于 100 KB 一定是错误页
                print("  %-10s: 已从 %s 下载 (%.1f MB)" % ("权重", who, size / 1024 ** 2))
                return dst
            os.remove(dst)
        except Exception as e:
            print("  %-10s: %s 失败（%s）" % ("权重", who, type(e).__name__))
            if os.path.isfile(dst):
                os.remove(dst)

    raise SystemExit(
        "无法下载预训练权重 %s —— 所有镜像均失败。\n"
        "可手动下载后放到：%s\n"
        "或改用 --model yolov8n.yaml 从零训练（无预训练，效果会明显差）。" % (base, WEIGHTS_DIR))


def _setup_cjk_rendering():
    """让 ultralytics 画图时能显示中文类别名。

    两处默认设置在国内环境下都会出问题：
      1) matplotlib 默认字体 DejaVu Sans 没有中文字形 —— 混淆矩阵、PR 曲线上的
         类别名会变成一排方框；
      2) 画检测框的 Annotator 遇到非 ASCII 类别名会去 GitHub 下 Arial.Unicode.ttf，
         离线必失败，退回 PIL 默认小字体。
    统一改成系统自带的黑体，两个问题一起解决。
    """
    import matplotlib
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    cjk = next((f for f in ("C:/Windows/Fonts/simhei.ttf",
                            "C:/Windows/Fonts/msyh.ttc",
                            "C:/Windows/Fonts/Deng.ttf") if os.path.isfile(f)), None)
    if not cjk:
        return
    try:
        import ultralytics.utils.plotting as _plotting
        _plotting.check_font = lambda font="Arial.ttf": font if os.path.isfile(str(font)) else cjk
    except Exception:
        pass


def _register_custom_modules():
    """按文件路径加载 training/models/custom_modules.py 并注册自定义模块。

    不依赖包结构（train.py 可能以任意工作目录运行），直接用 importlib 加载。
    """
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = os.path.join(here, "models", "custom_modules.py")
    if not os.path.isfile(p):
        return
    try:
        spec = importlib.util.spec_from_file_location("crop_custom_modules", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.register_modules()
        print("  %-10s: 已注册自定义模块（CBAM 等）" % "模块")
    except Exception as e:
        print("  %-10s: 自定义模块注册失败（%s）" % "模块", e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML 配置路径")
    ap.add_argument("--data", default=None, help="覆盖 data.yaml 路径")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--name", default=None, help="覆盖本次实验名")
    ap.add_argument("--model", default=None, help="覆盖模型，如 yolo11n.pt")
    ap.add_argument("--resume", action="store_true", help="从上次中断处继续")
    ap.add_argument("--pretrained", default=None,
                    help="迁移预训练权重（.pt）。仅当 --model 是 .yaml 时生效")
    args = ap.parse_args()

    try:
        import yaml
    except ImportError:
        print("缺 pyyaml，执行：pip install pyyaml")
        return 1
    try:
        import torch
        from ultralytics import YOLO
    except ImportError as e:
        print("依赖没装好：%r" % (e,))
        print("先跑 python code/scripts/env_check.py 看缺什么。")
        return 1

    cfg_path = args.config
    if not os.path.isfile(cfg_path):
        print("找不到配置：%s" % cfg_path)
        return 1
    with open(cfg_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    # 命令行覆盖
    if args.data:
        cfg["data"] = args.data
    if args.epochs:
        cfg["epochs"] = args.epochs
    if args.batch:
        cfg["batch"] = args.batch
    if args.imgsz:
        cfg["imgsz"] = args.imgsz
    if args.device is not None:
        cfg["device"] = args.device
    if args.name:
        cfg["name"] = args.name
    if args.model:
        cfg["model"] = args.model
    if args.resume:
        cfg["resume"] = True

    # 环境检查
    if not torch.cuda.is_available() and str(cfg.get("device")) != "cpu":
        print("!! CUDA 不可用，训练会落到 CPU 上，慢 30 倍。")
        print("   先跑 scripts/env_check.py 排查。要强行 CPU 训练加 --device cpu")
        return 1

    data_yaml = cfg.get("data", "")
    if not os.path.isfile(data_yaml):
        print("找不到 data.yaml：%s" % data_yaml)
        print("先跑：python code/scripts/prepare_data.py split --src <数据目录> --out <输出目录>")
        return 1

    with open(data_yaml, encoding="utf-8") as fh:
        d = yaml.safe_load(fh) or {}
    nc = d.get("nc", "?")
    names = d.get("names", {})

    print("=" * 58)
    print("实验配置")
    print("=" * 58)
    # 权重放在表头下面解析：本地没有就先下（走镜像），免得卡在 github.com
    model_path = resolve_model(cfg.pop("model"))
    for k in ("data", "epochs", "imgsz", "batch", "device", "seed"):
        print("  %-10s: %s" % (k, cfg.get(k)))
    print("  %-10s: %s (%s 类)" % ("数据集", os.path.basename(os.path.dirname(data_yaml)), nc))
    if isinstance(names, dict) and len(names) <= 6:
        print("  %-10s: %s" % ("类别", ", ".join(str(v) for v in names.values())))
    elif isinstance(names, dict):
        print("  %-10s: %s ... 共 %d 类" % ("类别", ", ".join(str(v) for v in list(names.values())[:5]), len(names)))
    print("  %-10s: %s" % ("GPU", torch.cuda.get_device_name(0)))
    print()

    _setup_cjk_rendering()

    t0 = time.time()

    # 若模型是 yaml，先注册自定义模块（CBAM 等），否则 parse_model 会因
    # `globals()[模块名]` 找不到类而报 KeyError。
    if str(model_path).endswith((".yaml", ".yml")):
        _register_custom_modules()

    model = YOLO(model_path)

    # 结构改动的自定义模型（如 yolov8n-p2.yaml）无法直接加载官方 .pt，
    # 这里按层名迁移能匹配的权重，P2 新增层随机初始化。
    if args.pretrained and str(model_path).endswith((".yaml", ".yml")):
        print("  %-10s: 迁移预训练权重 %s" % ("权重", args.pretrained))
        model.load(args.pretrained)

    # 把剩余配置整体传给 train()
    cfg.setdefault("project", "runs")
    cfg.setdefault("plots", True)
    results = model.train(**cfg)

    dt = time.time() - t0
    print()
    print("=" * 58)
    print("训练完成，耗时 %.1f 分钟" % (dt / 60))
    print("=" * 58)
    try:
        save_dir = str(results.save_dir)
        print("产出目录：%s" % save_dir)
        best = os.path.join(save_dir, "weights", "best.pt")
        if os.path.isfile(best):
            print("最佳权重：%s" % best)
        print()
        print("关键指标（写进论文的数字从这里取）：")
        for attr, label in [("results_dict", "末轮指标")]:
            rd = getattr(results, attr, None)
            if rd:
                for k, v in rd.items():
                    if any(s in k for s in ("mAP", "precision", "recall", "fitness")):
                        print("  %-24s %.4f" % (k, v))
        print()
        print("验证集评估：python -c \"from ultralytics import YOLO;"
              "YOLO(r'%s').val(data=r'%s')\"" % (best.replace("\\", "\\\\"), data_yaml))
    except Exception as e:
        print("指标读取失败（不影响训练结果）：%r" % (e,))
    return 0


if __name__ == "__main__":
    sys.exit(main())
