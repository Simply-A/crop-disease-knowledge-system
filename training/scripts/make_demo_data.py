# -*- coding: utf-8 -*-
"""
造一份「演示数据集」—— 用于在真实数据到位前验证训练链路。

为什么需要它：
    装完依赖后如果直接等数据集下载，链路是否通就是未知数；等到几万张
    数据下完才发现环境有问题，白等一场。本脚本用纯标准库合成一批带
    边界框的图片，几分钟内就能把「数据 → 训练 → 出 mAP」跑通一遍。

    **合成数据训练出的模型没有任何实际意义**，它只用来证明：
    环境、脚本、配置、显存、产出目录全都正常。

零第三方依赖（只用 zlib/struct 手写 PNG），所以在最裸的环境里也能跑。

用法：
    python code/scripts/make_demo_data.py --out D:/bishu/datasets/demo
    python code/scripts/make_demo_data.py --out D:/bishu/datasets/demo --train 64 --val 16
"""

import argparse
import json
import os
import random
import struct
import sys
import zlib

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# 类别用「作物 + 病害」完整名 —— 简称会让后续按类别过滤时同时命中多个作物
CLASSES = ["番茄早疫病", "番茄晚疫病", "马铃薯早疫病"]

# 每类的（叶片底色, 病斑色）—— 让不同类在颜色上可分，训练才不会摆烂
PALETTE = {
    0: ((58, 108, 48), (126, 74, 32)),    # 褐色同心轮纹斑
    1: ((64, 112, 56), (46, 48, 54)),     # 暗灰褐色水渍斑
    2: ((74, 116, 50), (150, 132, 58)),   # 黄褐色斑
}


def _chunk(tag: bytes, data: bytes) -> bytes:
    body = tag + data
    return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)


def write_png(path: str, w: int, h: int, rgb: bytearray) -> None:
    """把 w*h*3 的 RGB 缓冲写成 PNG（filter=0，不压缩预处理）。"""
    stride = w * 3
    raw = bytearray()
    for y in range(h):
        raw.append(0)
        raw += rgb[y * stride:(y + 1) * stride]
    out = b"\x89PNG\r\n\x1a\n"
    out += _chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    out += _chunk(b"IDAT", zlib.compress(bytes(raw), 6))
    out += _chunk(b"IEND", b"")
    with open(path, "wb") as fh:
        fh.write(out)


def _fill_rect(buf: bytearray, w: int, x: int, y: int, rw: int, rh: int, col) -> None:
    row = bytes(col) * rw
    for yy in range(y, y + rh):
        s = (yy * w + x) * 3
        buf[s:s + len(row)] = row


def make_sample(w: int, h: int, cls: int, rng: random.Random):
    """生成一张图 + 该图的框（像素坐标 x,y,bw,bh）。"""
    bg, spot = PALETTE[cls]
    buf = bytearray(bg * (w * h))

    # 叶片底纹：画几条浅色叶脉，避免图像过于平整
    vein = tuple(min(255, c + 22) for c in bg)
    for _ in range(rng.randint(3, 6)):
        vx = rng.randint(0, w - 6)
        _fill_rect(buf, w, vx, 0, rng.randint(2, 5), h, vein)

    boxes = []
    for _ in range(rng.randint(1, 3)):
        # 混入小目标：约 40% 的框边长 < 64 px（imgsz=640 下即 < 0.1 相对尺寸）
        if rng.random() < 0.4:
            bw = rng.randint(18, 62)
            bh = rng.randint(18, 62)
        else:
            bw = rng.randint(70, 170)
            bh = rng.randint(70, 170)
        bw = min(bw, w - 2)
        bh = min(bh, h - 2)
        x = rng.randint(0, w - bw - 1)
        y = rng.randint(0, h - bh - 1)
        col = tuple(max(0, min(255, c + rng.randint(-20, 20))) for c in spot)
        _fill_rect(buf, w, x, y, bw, bh, col)
        # 病斑外围一圈过渡色，让边界不那么生硬
        edge = tuple((a + b) // 2 for a, b in zip(col, bg))
        if x > 0 and y > 0:
            _fill_rect(buf, w, x - 1, y - 1, bw + 2, 1, edge)
            _fill_rect(buf, w, x - 1, y + bh, bw + 2, 1, edge)
        boxes.append((x, y, bw, bh))
    return buf, boxes


def main() -> int:
    ap = argparse.ArgumentParser(description="合成演示数据集（仅供链路验证）")
    ap.add_argument("--out", required=True, help="输出目录，如 D:/bishu/datasets/demo")
    ap.add_argument("--train", type=int, default=48, help="训练集张数（默认 48）")
    ap.add_argument("--val", type=int, default=12, help="验证集张数（默认 12）")
    ap.add_argument("--imgsz", type=int, default=640, help="图片边长（默认 640）")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = os.path.abspath(args.out)
    if os.path.isdir(root) and os.listdir(root):
        print("输出目录非空：%s" % root)
        print("换个 --out，或先把旧目录挪走（脚本不覆盖已有数据）。")
        return 1

    rng = random.Random(args.seed)
    total_box = 0
    small_box = 0

    for split, n in (("train", args.train), ("val", args.val)):
        img_dir = os.path.join(root, "images", split)
        lbl_dir = os.path.join(root, "labels", split)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        for i in range(n):
            cls = i % len(CLASSES)          # 按序轮转，保证类别均衡
            buf, boxes = make_sample(args.imgsz, args.imgsz, cls, rng)
            stem = "%s_%03d" % (split, i)
            write_png(os.path.join(img_dir, stem + ".png"), args.imgsz, args.imgsz, buf)

            lines = []
            for (x, y, bw, bh) in boxes:
                cx = (x + bw / 2.0) / args.imgsz
                cy = (y + bh / 2.0) / args.imgsz
                nw = bw / float(args.imgsz)
                nh = bh / float(args.imgsz)
                lines.append("%d %.6f %.6f %.6f %.6f" % (cls, cx, cy, nw, nh))
                total_box += 1
                if max(nw, nh) < 0.1:
                    small_box += 1
            with open(os.path.join(lbl_dir, stem + ".txt"), "w", encoding="utf-8") as fh:
                fh.write("\n".join(lines) + "\n")

    # data.yaml —— names 用 dict 形式，和 prepare_data.py 的产物保持一致
    yaml_txt = [
        "# 演示数据集（合成图，仅供训练链路验证，无任何实际意义）",
        "path: %s" % root.replace("\\", "/"),
        "train: images/train",
        "val: images/val",
        "nc: %d" % len(CLASSES),
        "names:",
    ]
    for i, name in enumerate(CLASSES):
        yaml_txt.append("  %d: %s" % (i, name))
    with open(os.path.join(root, "data.yaml"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(yaml_txt) + "\n")

    print("=" * 58)
    print("演示数据集已生成")
    print("=" * 58)
    print("  目录      : %s" % root)
    print("  训练 / 验证: %d / %d 张" % (args.train, args.val))
    print("  图片尺寸  : %d x %d" % (args.imgsz, args.imgsz))
    print("  类别      : %s" % ", ".join("%d=%s" % (i, n) for i, n in enumerate(CLASSES)))
    print("  框总数    : %d（其中相对边长 < 0.1 的小目标 %d 个，占 %.0f%%）"
          % (total_box, small_box, 100.0 * small_box / max(total_box, 1)))
    print("  data.yaml : %s" % os.path.join(root, "data.yaml"))
    print()
    print("训练命令：")
    print("  & 'D:\\bishu\\venvs\\yolo\\Scripts\\python.exe' code/scripts/train.py \\")
    print("      --config code/configs/baseline.yaml \\")
    print("      --data '%s' \\" % os.path.join(root, "data.yaml"))
    print("      --epochs 10 --batch 8 --name smoke_demo")
    print()
    print("提醒：合成数据训练出的模型没有实际意义，跑通即弃。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
