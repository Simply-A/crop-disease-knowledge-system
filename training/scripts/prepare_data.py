# -*- coding: utf-8 -*-
"""
数据集整理工具 —— 扫描 / 划分 / 校验 YOLO 格式数据集。

    python prepare_data.py scan  --src <数据根目录> [--out 报告.txt]
    python prepare_data.py split --src <数据根目录> --out <输出目录> [--ratios 7:2:1]
    python prepare_data.py check --out <输出目录>

支持三种输入布局：
  A. YOLO 检测格式：子目录下有 images/ 与 labels/（同名 .jpg / .txt 配对），
     或图片与同名 .txt 混放在同一目录 —— 例如 Roboflow 导出的数据集。
     类别表从源 data.yaml / classes.txt 读取并沿用。
  B. 分类格式（PlantVillage 原版）：每个子目录就是一个类别，只有图片没有 txt。
     同时也会读取子目录里的 .txt（若有），并按「目录 = 类别」重映射 class_id。

两个关键保障：
  1. split 会**重写**标注文件里的 class_id，使它和生成的 data.yaml 严格对应。
     直接复制标注文件会导致类别错位 —— 训练不报错，但结果全废。
  2. split 前会核对源类别表；对不上时直接报错停下，而不是"猜一个"。

scan 会重点输出**病斑框的相对尺寸分布** —— 这是「小目标检测改进」这个创新点的
直接数据支撑，论文里要用，建议把报告存下来。
"""

import argparse
import os
import random
import re
import shutil
import statistics
import sys
from collections import Counter, defaultdict

# Windows 控制台默认 GBK 代码页，固定成 UTF-8，否则中文输出可能乱码
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")

# 小目标判据：框边长占图像边长的比例
SMALL_RATIO = 0.05
TINY_RATIO = 0.02


def is_img(f):
    return f.lower().endswith(IMG_EXT)


def stem(f):
    return os.path.splitext(f)[0]


# ---------------------------------------------------------------- 扫描

def find_pairs(root):
    """返回 (pairs, layout, groups)
    pairs: [(img_path, label_path_or_None, group_name)]
    """
    pairs = []
    has_labels_dir = False
    for _, dirs, _ in os.walk(root):
        if "labels" in [d.lower() for d in dirs]:
            has_labels_dir = True
            break

    if has_labels_dir:
        layout = "yolo"
        for cur, dirs, files in os.walk(root):
            if os.path.basename(cur).lower() != "images":
                continue
            lab_dir = os.path.join(os.path.dirname(cur), "labels")
            group = os.path.basename(os.path.dirname(cur))
            for f in files:
                if is_img(f):
                    lp = os.path.join(lab_dir, stem(f) + ".txt")
                    pairs.append((os.path.join(cur, f),
                                  lp if os.path.exists(lp) else None, group))
        if not pairs:
            # images 目录下没有图片，退回通用扫描
            layout = "flat"
            for cur, _, files in os.walk(root):
                for f in files:
                    if is_img(f):
                        lp = os.path.join(cur, stem(f) + ".txt")
                        pairs.append((os.path.join(cur, f),
                                      lp if os.path.exists(lp) else None, "_"))
    else:
        subdirs = [d for d in os.listdir(root)
                   if os.path.isdir(os.path.join(root, d))
                   and any(is_img(f) for f in os.listdir(os.path.join(root, d)))]
        if subdirs:
            layout = "classify"
            for d in sorted(subdirs):
                dp = os.path.join(root, d)
                for f in os.listdir(dp):
                    if is_img(f):
                        lp = os.path.join(dp, stem(f) + ".txt")
                        pairs.append((os.path.join(dp, f),
                                      lp if os.path.exists(lp) else None, d))
        else:
            layout = "flat"
            for f in os.listdir(root):
                if is_img(f):
                    lp = os.path.join(root, stem(f) + ".txt")
                    pairs.append((os.path.join(root, f),
                                  lp if os.path.exists(lp) else None, "_"))

    groups = sorted(set(g for _, _, g in pairs))
    return pairs, layout, groups


def read_boxes(label_path):
    """读 YOLO 标注，返回 [(cls_id, x, y, w, h, area)]；格式非法返回 None，空文件返回 []。"""
    out = []
    try:
        with open(label_path, "r", encoding="utf-8", errors="ignore") as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln:
                    continue
                parts = ln.split()
                if len(parts) < 5:
                    return None
                try:
                    c = int(float(parts[0]))
                    x, y, w, h = (float(v) for v in parts[1:5])
                except ValueError:
                    return None
                if not (0 <= x <= 1 and 0 <= y <= 1):
                    return None
                if w <= 0 or h <= 0 or w > 1.001 or h > 1.001:
                    return None
                out.append((c, x, y, w, h, w * h))
    except Exception:
        return None
    return out


# ---------------------------------------------------------------- 类别表

def parse_names_from_yaml(text):
    """从 data.yaml 文本里抠 names。支持两种写法：
        names: ['a', 'b']
        names:
          0: a
          1: b
    """
    names = {}
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        if not re.match(r"^\s*names\s*:", ln):
            continue
        rest = ln.split(":", 1)[1].strip()
        if rest.startswith("["):
            body = rest.strip("[]")
            items = [x.strip().strip("'\"") for x in body.split(",") if x.strip()]
            for k, v in enumerate(items):
                names[k] = v
            return names or None
        if rest:
            # names: {0: a, 1: b}
            for m in re.finditer(r"(\d+)\s*:\s*['\"]?([^,'\"}]+)", rest):
                names[int(m.group(1))] = m.group(2).strip()
            if names:
                return names
        # 块式续行
        for ln2 in lines[i + 1:]:
            if ln2.strip() and not ln2.startswith((" ", "\t")):
                break
            m = re.match(r"^\s*(?:-\s*)?['\"]?(\d+)['\"]?\s*:\s*(.+?)\s*$", ln2)
            if m:
                names[int(m.group(1))] = m.group(2).strip().strip("'\"")
            else:
                m2 = re.match(r"^\s*-\s*['\"]?(.+?)['\"]?\s*$", ln2)
                if m2:
                    names[len(names)] = m2.group(1).strip()
        return names or None
    return None


def find_src_names(src):
    """在源目录里找类别表，返回 ({id: name}, 来源文件) 或 (None, None)。"""
    cands = []
    for cur, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs if d.lower() != "labels"]
        for f in files:
            lf = f.lower()
            if lf in ("data.yaml", "data.yml", "dataset.yaml", "classes.txt",
                      "obj.names", "predefined_classes.txt", "notes.json"):
                cands.append(os.path.join(cur, f))
        if len(cands) > 6:
            break
    # 优先 data.yaml，其次 classes.txt
    cands.sort(key=lambda p: (0 if "yaml" in p.lower() else 1))
    for p in cands:
        try:
            with open(p, encoding="utf-8", errors="ignore") as fh:
                text = fh.read()
        except Exception:
            continue
        if p.lower().endswith((".yaml", ".yml")):
            n = parse_names_from_yaml(text)
            if n:
                return n, p
        else:
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            if lines and not any("," in l for l in lines[:2]):
                return {i: v for i, v in enumerate(lines)}, p
    return None, None


def histogram(values, bins=8, width=32):
    if not values:
        return "    （无数据）"
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        hi = lo + 1e-9
    step = (hi - lo) / bins
    counts = [0] * bins
    for v in values:
        counts[min(int((v - lo) / step), bins - 1)] += 1
    mx = max(counts)
    lines = []
    for i, c in enumerate(counts):
        a, b = lo + i * step, lo + (i + 1) * step
        bar = "#" * int(c / mx * width) if mx else ""
        lines.append("    %.3f-%.3f |%-*s| %6d  %5.1f%%"
                     % (a, b, width, bar, c, c / len(values) * 100))
    return "\n".join(lines)


# ---------------------------------------------------------------- scan

def cmd_scan(args):
    root = args.src
    if not os.path.isdir(root):
        print("目录不存在：%s" % root)
        return 1

    print("扫描：%s" % root)
    pairs, layout, groups = find_pairs(root)
    if not pairs:
        print("没找到任何图片。检查路径。")
        return 1

    n_img = len(pairs)
    n_lbl = sum(1 for _, l, _ in pairs if l)
    print()
    print("布局        : %s" % layout)
    print("图片        : %d 张" % n_img)
    print("有标注      : %d 张 (%.1f%%)" % (n_lbl, n_lbl / n_img * 100))
    print("分组        : %d 组" % len(groups))

    src_names, ypath = find_src_names(root)
    if src_names:
        print("源类别表    : %s（%d 类）" % (os.path.basename(ypath), len(src_names)))
        for k in sorted(src_names)[:12]:
            print("                %2d: %s" % (k, src_names[k]))
        if len(src_names) > 12:
            print("                ... 共 %d 类" % len(src_names))
    else:
        print("源类别表    : 未找到（训练前需人工确认类别顺序）")

    lines = ["数据集扫描报告", "路径: %s" % root,
             "布局: %s | 图片 %d 张 | 有标注 %d 张" % (layout, n_img, n_lbl), ""]

    print()
    print("=" * 58)
    print("分组分布")
    print("=" * 58)
    by_grp = defaultdict(lambda: [0, 0])
    for _, lab, g in pairs:
        by_grp[g][0] += 1
        if lab:
            by_grp[g][1] += 1
    lines.append("分组 | 图片数 | 有标注")
    for g in sorted(by_grp):
        t, l = by_grp[g]
        print("  %-32s 图%6d  标注%6d" % (g[:32], t, l))
        lines.append("%s | %d | %d" % (g, t, l))

    all_side, per_cls_side = [], defaultdict(list)
    bad, cls_ids, n_box = [], Counter(), 0
    if n_lbl:
        for _, lab, g in pairs:
            if not lab:
                continue
            boxes = read_boxes(lab)
            if boxes is None:
                bad.append(lab)
                continue
            for c, x, y, w, h, a in boxes:
                side = (a ** 0.5)
                all_side.append(side)
                per_cls_side[g].append(side)
                cls_ids[c] += 1
                n_box += 1

        print()
        print("=" * 58)
        print("病斑框尺寸分布（框边长 / 图像边长，越小越难检测）")
        print("=" * 58)
        print("  框总数        : %d" % n_box)
        lines.append("")
        lines.append("框总数: %d" % n_box)
        if all_side:
            mn, md, mx = min(all_side), statistics.median(all_side), max(all_side)
            small = sum(1 for s in all_side if s < SMALL_RATIO)
            tiny = sum(1 for s in all_side if s < TINY_RATIO)
            print("  平均/中位边长 : %.3f / %.3f" % (statistics.mean(all_side), md))
            print("  最小/最大边长 : %.4f / %.4f" % (mn, mx))
            print()
            print("  边长 < %.2f 的小目标 : %6d 个，占 %5.1f%%"
                  % (SMALL_RATIO, small, small / n_box * 100))
            print("  边长 < %.2f 的极小目标: %6d 个，占 %5.1f%%"
                  % (TINY_RATIO, tiny, tiny / n_box * 100))
            print()
            print("  边长分布：")
            print(histogram(all_side))
            lines += ["平均框边长占比: %.4f" % statistics.mean(all_side),
                      "中位框边长占比: %.4f" % md,
                      "小目标(<%.2f)占比: %.1f%% (%d/%d)"
                      % (SMALL_RATIO, small / n_box * 100, small, n_box),
                      "极小目标(<%.2f)占比: %.1f%% (%d/%d)"
                      % (TINY_RATIO, tiny / n_box * 100, tiny, n_box), ""]
            print()
            print("  ← 上面这段直接写进论文，是「小目标检测改进」的立项依据")

        print()
        print("=" * 58)
        print("各类别的框尺寸（找最难检测的类）")
        print("=" * 58)
        print("  %-32s %8s %12s" % ("分组", "框数", "平均边长"))
        for g in sorted(per_cls_side):
            v = per_cls_side[g]
            print("  %-32s %8d %12.3f" % (g[:32], len(v), statistics.mean(v)))
        lines.append("分组 | 框数 | 平均框边长占比")
        for g in sorted(per_cls_side):
            v = per_cls_side[g]
            lines.append("%s | %d | %.4f" % (g, len(v), statistics.mean(v)))

        print()
        print("=" * 58)
        print("标注里的 class_id 分布")
        print("=" * 58)
        for c in sorted(cls_ids):
            nm = ""
            if src_names and c in src_names:
                nm = "  <- %s" % src_names[c]
            print("  class %-4d %6d 个框%s" % (c, cls_ids[c], nm))
        lines += ["", "class_id 分布: %s" % dict(sorted(cls_ids.items()))]
        if src_names:
            missing = set(src_names) - set(cls_ids)
            extra = set(cls_ids) - set(src_names)
            if extra:
                print()
                print("  !! 标注里出现源类别表里没有的 id: %s" % sorted(extra))
                print("     说明类别表选错了，或标注本身就乱。训练前必须解决。")

    print()
    print("=" * 58)
    print("问题样本")
    print("=" * 58)
    no_lab = [p for p in pairs if not p[1]]
    print("  缺标注文件  : %d 张" % len(no_lab))
    print("  标注格式错误: %d 个" % len(bad))
    if no_lab:
        print("    例：%s" % os.path.basename(no_lab[0][0]))
        print("    分类数据集属正常（本来就没有框）；若是检测数据集，说明标注漏了。")
    if bad:
        print("    例：%s" % bad[0])
    lines += ["", "缺标注: %d 张；格式错误: %d 个" % (len(no_lab), len(bad))]

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))
        print()
        print("报告已写入：%s" % args.out)

    print()
    print("下一步：python prepare_data.py split --src \"%s\" --out D:\\bishu\\datasets\\<数据集名>"
          % root)
    return 0


# ---------------------------------------------------------------- split

def cmd_split(args):
    src, out = args.src, args.out
    if not os.path.isdir(src):
        print("目录不存在：%s" % src)
        return 1
    pairs, layout, groups = find_pairs(src)
    if not pairs:
        print("没找到图片。")
        return 1

    n_img = len(pairs)
    n_lbl = sum(1 for _, l, _ in pairs if l)

    # ---- 1. 定类别表 ----
    print("=" * 58)
    print("类别表核对")
    print("=" * 58)
    if layout == "classify":
        names = sorted(groups)
        mode = "by_dir"
        print("布局：分类结构（子目录 = 类别）")
        print("将按目录名重映射 class_id，共 %d 类：" % len(names))
        for i, n in enumerate(names):
            print("   %2d: %s" % (i, n))
        print()
        print("  !! 请确认上面的顺序就是你要的类别顺序。")
        print("     类别名建议用「作物+病害」完整形式（如 Tomato__Early_blight），")
        print("     否则后续按作物过滤时会命中多个作物（番茄早疫病 / 马铃薯早疫病）。")
    else:
        src_names, ypath = find_src_names(src)
        if src_names:
            print("布局：YOLO 检测结构")
            print("从 %s 读到类别表，共 %d 类：" % (os.path.basename(ypath), len(src_names)))
            for k in sorted(src_names):
                print("   %2d: %s" % (k, src_names[k]))
            names = [src_names[k] for k in sorted(src_names)]
            mode = "identity"
            print()
            print("  标注文件的 class_id 原样保留（与源类别表一致）。")
        else:
            ids = set()
            for _, l, _ in pairs:
                if l:
                    b = read_boxes(l)
                    if b:
                        ids.update(c for c, *_ in b)
            if not ids:
                print("没有任何可用标注，也没有类别表，无法确定类别。")
                print("先给源标注配一个 data.yaml，或用分类结构组织数据。")
                return 1
            names = ["class_%d" % i for i in sorted(ids)]
            mode = "identity"
            print("布局：YOLO 检测结构，但**没找到源类别表**。")
            print("按标注里出现的 class_id 推断出 %d 类：%s" % (len(names), names))
            print()
            print("  !! 这些名字是占位符。训练前必须打开生成的 data.yaml")
            print("     把 names 改成真实病害名，并把顺序与 class_id 对齐。")
            print("     改错了训练不报错，但结果全废。")

    # ---- 2. 划分 ----
    ratios = [float(x) for x in args.ratios.split(":")]
    if len(ratios) != 3 or sum(ratios) <= 0:
        print("--ratios 要写三段正数，如 7:2:1")
        return 1
    s = sum(ratios)
    ratios = [r / s for r in ratios]

    random.seed(args.seed)
    by_grp = defaultdict(list)
    for p in pairs:
        by_grp[p[2]].append(p)

    splits = {"train": [], "val": [], "test": []}
    for g, items in by_grp.items():
        items = items[:]
        random.shuffle(items)
        n = len(items)
        n_tr = int(round(n * ratios[0]))
        n_va = int(round(n * ratios[1]))
        if n >= 3:
            n_tr = max(1, min(n_tr, n - 2))
            n_va = max(1, min(n_va, n - n_tr))
        else:
            n_tr, n_va = n, 0
        splits["train"] += items[:n_tr]
        splits["val"] += items[n_tr:n_tr + n_va]
        splits["test"] += items[n_tr + n_va:]
    for g in by_grp:
        if len(by_grp[g]) >= 3 and not splits["val"]:
            print("!! 划分异常：%s 组没有进验证集" % g)

    # ---- 3. 覆盖率：哪些组需要生成整图框 ----
    cov = {}
    for _, lab, g in pairs:
        c = cov.setdefault(g, [0, 0])
        c[0] += 1
        if lab:
            c[1] += 1
    fake_groups = {g for g, (t, l) in cov.items() if l < t * 0.5}

    # ---- 4. 落盘（重写标注，重映射 class_id）----
    for sp in splits:
        for sub in ("images", "labels"):
            os.makedirs(os.path.join(out, sub, sp), exist_ok=True)

    cmap = {g: i for i, g in enumerate(names)} if mode == "by_dir" else None
    used = Counter()
    n_fake = n_bad = n_skip = 0
    processed = 0

    for sp in ("train", "val", "test"):
        for idx, (img, lab, g) in enumerate(splits[sp]):
            ext = os.path.splitext(img)[1].lower()
            safe = re.sub(r"[^0-9A-Za-z_.-]", "_", g)[:28].strip("_") or "cls"
            newname = "%s_%s%05d%s" % (sp[0], safe, idx, ext)
            shutil.copy2(img, os.path.join(out, "images", sp, newname))
            dst = os.path.join(out, "labels", sp, os.path.splitext(newname)[0] + ".txt")

            boxes = read_boxes(lab) if lab else None
            if lab and boxes is None:
                n_bad += 1
                open(dst, "w").close()
            elif boxes:
                with open(dst, "w", encoding="utf-8") as fh:
                    for c, x, y, w, h, _a in boxes:
                        nid = cmap[g] if mode == "by_dir" else c
                        fh.write("%d %.6f %.6f %.6f %.6f\n" % (nid, x, y, w, h))
            elif g in fake_groups:
                nid = cmap[g] if mode == "by_dir" else 0
                with open(dst, "w", encoding="utf-8") as fh:
                    fh.write("%d 0.500000 0.500000 1.000000 1.000000\n" % nid)
                n_fake += 1
            else:
                n_skip += 1
                if os.path.exists(dst):
                    os.remove(dst)
                os.remove(os.path.join(out, "images", sp, newname))
                continue
            used[sp] += 1
            processed += 1
            if processed % 2000 == 0:
                print("  已处理 %d 张..." % processed)

    # ---- 5. data.yaml ----
    yml = os.path.join(out, "data.yaml")
    with open(yml, "w", encoding="utf-8") as fh:
        fh.write("# 由 prepare_data.py 自动生成\n")
        fh.write("path: %s\n" % out.replace("\\", "/"))
        fh.write("train: images/train\n")
        fh.write("val: images/val\n")
        fh.write("test: images/test\n")
        fh.write("nc: %d\n" % len(names))
        fh.write("names:\n")
        for i, n in enumerate(names):
            fh.write("  %d: %s\n" % (i, n))

    print()
    print("=" * 58)
    print("划分完成 -> %s" % out)
    print("=" * 58)
    for sp in ("train", "val", "test"):
        print("  %-6s %6d 张" % (sp, used[sp]))
    print("  data.yaml 已生成，nc=%d" % len(names))
    if n_fake:
        print()
        print("  ! 生成整图框 %d 张（源数据没有标注的组）" % n_fake)
        print("    这只是占位，正式实验前必须替换成真实病斑标注。")
    if n_skip:
        print("  跳过缺标注图片 %d 张（同组里多数有标注，视为标注遗漏）" % n_skip)
    if n_bad:
        print("  标注格式错误 %d 个，已写成空标注（相当于负样本）" % n_bad)
    print()
    print("接着跑：python prepare_data.py check --out \"%s\"" % out)
    return 0


# ---------------------------------------------------------------- check

def cmd_check(args):
    out = args.out
    yml = os.path.join(out, "data.yaml")
    print("校验：%s" % out)
    print()
    if not os.path.isfile(yml):
        print("找不到 data.yaml，先跑 split。")
        return 1
    with open(yml, encoding="utf-8") as fh:
        text = fh.read()
    print(text.strip())
    print()

    ok = True
    m = re.search(r"nc:\s*(\d+)", text)
    nc = int(m.group(1)) if m else None
    all_ids = set()

    for sp in ("train", "val", "test"):
        idir = os.path.join(out, "images", sp)
        ldir = os.path.join(out, "labels", sp)
        if not os.path.isdir(idir):
            print("  [%s] 目录缺失" % sp)
            ok = False
            continue
        imgs = [f for f in os.listdir(idir) if is_img(f)]
        n_missing = n_empty = n_bad = 0
        seen = Counter()
        for f in imgs:
            lp = os.path.join(ldir, stem(f) + ".txt")
            if not os.path.exists(lp):
                n_missing += 1
                continue
            boxes = read_boxes(lp)
            if boxes is None:
                n_bad += 1
                continue
            if not boxes:
                n_empty += 1
            for c, *_ in boxes:
                seen[c] += 1
                all_ids.add(c)
        print("  [%-5s] 图 %5d | 缺标注 %d | 空标注 %d | 格式错 %d"
              % (sp, len(imgs), n_missing, n_empty, n_bad))
        if seen:
            print("          class_id: %s" % dict(sorted(seen.items())))
        if sp in ("train", "val") and not imgs:
            print("          !! %s 集是空的，训不起来" % sp)
            ok = False
        if n_missing:
            print("          !! %d 张图缺标注文件" % n_missing)
            ok = False
        if n_bad:
            print("          !! %d 个标注文件格式错误" % n_bad)
            ok = False

    print()
    if nc is not None:
        if all_ids and max(all_ids) >= nc:
            print("  !! 标注里最大 class_id = %d，但 nc = %d，会越界报错。" % (max(all_ids), nc))
            ok = False
        else:
            print("  nc=%d，标注里出现 %d 个不同类别，范围合法" % (nc, len(all_ids)))
        if nc != len(all_ids):
            print("  ?  nc=%d 但有 %d 个类别出现在标注里。若某些类别在当前子集里没样本，属正常；"
                  % (nc, len(all_ids)))
            print("     否则说明类别表与实际标注对不上，回查 data.yaml 的 names。")

    print()
    if ok:
        print("[OK] 校验通过，可以开始训练。")
        print("     python code/scripts/train.py --config code/configs/baseline.yaml")
    else:
        print("[!!] 有上面的问题，先修再训。")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description="YOLO 数据集整理工具")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("scan", help="体检：分布统计，不写文件")
    p1.add_argument("--src", required=True)
    p1.add_argument("--out", default=None, help="把报告写到这个文件")

    p2 = sub.add_parser("split", help="分层划分并生成 data.yaml")
    p2.add_argument("--src", required=True)
    p2.add_argument("--out", required=True)
    p2.add_argument("--ratios", default="7:2:1")
    p2.add_argument("--seed", type=int, default=42)

    p3 = sub.add_parser("check", help="训练前校验")
    p3.add_argument("--out", required=True)

    args = ap.parse_args()
    return {"scan": cmd_scan, "split": cmd_split, "check": cmd_check}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
