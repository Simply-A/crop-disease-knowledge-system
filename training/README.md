# 检测模型训练（training/）

毕设《基于改进YOLOv8的农作物病害检测系统设计与实现》的**模型训练部分**：数据集整理、基线复现、三处改进与消融实验、图片推理。

## 目录结构

```
training/
├── scripts/
│   ├── train.py            训练入口（读 YAML 配置，调 ultralytics）
│   ├── prepare_data.py     数据集整理：scan / split / check
│   ├── make_demo_data.py   合成演示数据（验证链路，跑通即弃）
│   ├── env_check.py        环境自检（训练前必须先跑）
│   ├── verify_models.py    验证各模型 yaml 构建与参数量
│   └── infer.py            图片推理
├── configs/
│   ├── baseline.yaml       基线训练配置（YOLOv8n）
│   ├── ablation_p2.yaml    改进② P2 小目标层
│   ├── ablation_cbam.yaml  改进① CBAM 注意力
│   ├── ablation_ghost.yaml 改进③ Ghost 轻量骨干
│   └── ablation_full.yaml  三项叠加
├── models/                 模型结构 yaml 与自定义模块
│   ├── yolov8n-p2.yaml
│   ├── yolov8n-cbam.yaml
│   ├── yolov8n-ghost.yaml
│   ├── yolov8n-p2-cbam-ghost.yaml
│   └── custom_modules.py   CBAM 自定义模块
└── weights/                yolov8n.pt 预训练权重（git 忽略，缺失时自动镜像下载）
```

## 三处改进

| 编号 | 改进 | 落点 | 实现方式 | 参数量变化 |
|---|---|---|---|---|
| ① | CBAM 注意力 | 颈部 P3/P4/P5 特征 | 自定义模块 | +0.35% |
| ② | P2 小目标检测层 | 检测头新增 1/4 分辨率尺度 | 结构 yaml | +6.2% |
| ③ | GhostNet 轻量骨干 | 骨干 Conv→GhostConv、C2f→C3Ghost | 内置模块 | −22.4% |

## 数据集

主力检测数据集：番茄田间病害集（`tomato_det`，7 类 / 689 张，CC BY 4.0，田间实拍病斑框）。
数据集实体放 D 盘（`D:/bishu/datasets/`），不进 git；`data.yaml` 路径在配置文件中指定。

## 用法

环境自检 → 训练基线 → 跑消融 → 推理：

```bash
PY="D:/Develop/anaconda/envs/pytorch_gpu/python.exe"   # 或你的 GPU 环境

# 1) 自检（必须看到 CUDA 可用 = True）
$PY training/scripts/env_check.py

# 2) 基线训练
$PY training/scripts/train.py --config training/configs/baseline.yaml

# 3) 消融实验（结构改动的模型需迁移 yolov8n.pt 权重）
$PY training/scripts/train.py --config training/configs/ablation_p2.yaml \
    --pretrained training/weights/yolov8n.pt
$PY training/scripts/train.py --config training/configs/ablation_cbam.yaml \
    --pretrained training/weights/yolov8n.pt
$PY training/scripts/train.py --config training/configs/ablation_ghost.yaml \
    --pretrained training/weights/yolov8n.pt
$PY training/scripts/train.py --config training/configs/ablation_full.yaml \
    --pretrained training/weights/yolov8n.pt

# 4) 推理
$PY training/scripts/infer.py --weights <best.pt> --source <图片或目录>
```

## 关键约定

- **训练产出目录**：`project` 统一设为 `D:/bishu/runs`（放 D 盘，避免占 C 盘；git 已忽略 `runs/`）。
- **预训练权重**：`yolov8n.pt` 缺失时 `train.py` 自动走镜像下载（ghfast.top 等）。
- **类别命名**：检测类别名与知识库类别标准名保持一致，避免类别硬过滤失效。
- **中文渲染**：`train.py` 内置了 matplotlib 中文字体与 Annotator 字体修复。
- **可复现**：所有实验固定 `seed: 42`，改进只改结构、不动其他超参数。

## 消融实验方案（5 组）

| 组 | 配置 | 说明 |
|---|---|---|
| A0 | baseline | YOLOv8n 基线（对照零点） |
| A1 | ablation_cbam | +CBAM 注意力 |
| A2 | ablation_p2 | +P2 小目标层 |
| A3 | ablation_ghost | +Ghost 轻量骨干 |
| A4 | ablation_full | 三项叠加 |

每组在同一数据划分、同一超参数下训练，对比 mAP@50、mAP@50-95、Precision、Recall、Params、FLOPs。
