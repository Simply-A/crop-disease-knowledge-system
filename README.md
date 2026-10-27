# 作物病害检测与知识服务系统

面向计算机视觉与大模型应用的项目实践：计划整合农作物病害目标检测、农业知识检索和可追溯问答服务。

> **项目状态：开发中。** 已完成：基础后端（FastAPI + 测试 + CI）、检测训练模块（YOLOv8n 基线 + 三处改进 + 消融实验）、图文双模态 RAG 知识库。数据集与模型权重不提交（体积大，走本地 D 盘）；前端待开发。请勿将计划中的功能当作已完成成果。

## 改进方案（消融实验）

| 编号 | 改进 | 落点 | 参数量变化 |
|---|---|---|---|
| ① | CBAM 注意力 | 颈部 P3/P4/P5 特征 | +0.35% |
| ② | P2 小目标检测层 | 检测头新增 1/4 分辨率尺度 | +6.2% |
| ③ | GhostNet 轻量骨干 | 骨干 Conv→GhostConv、C2f→C3Ghost | −22.4% |

## 目标功能

- **病害检测**：基于 YOLO 系列模型识别图像中的作物病害；后续评估小目标检测改进。
- **知识检索**：建立农业病害文本知识库，支持分块、向量化与语义检索。
- **图文检索**：在数据和模型准备完成后，探索病害图像与文本的跨模态检索。
- **知识问答**：结合检索结果生成回答，并展示引用来源；检索证据不足时明确提示。
- **服务接口**：使用 FastAPI 提供健康检查与后续推理接口。

## 技术规划

- Python 3.11+
- FastAPI / Uvicorn / Pydantic
- PyTorch / Ultralytics YOLO（模型阶段）
- 向量检索与 RAG（知识库阶段）
- Vue 3（可选前端阶段）

## 目录结构

```text
.
├── .github/workflows/       CI（pytest）
├── backend/                 FastAPI 主服务（健康检查，检测/知识接口陆续接入）
│   └── app/
│       ├── config.py        集中配置（.env 覆盖）
│       ├── main.py          应用入口
│       └── core/logging.py  日志
├── training/                检测模型训练（基线 / 三处改进 / 消融 / 推理）
│   ├── scripts/             train.py · prepare_data.py · run_ablation.py · infer.py …
│   ├── configs/             baseline.yaml · ablation_{p2,cbam,ghost,full}.yaml
│   ├── models/              改进模型 yaml + CBAM 自定义模块
│   └── weights/             yolov8n.pt（git 忽略，缺失自动镜像下载）
├── knowledge/               图文双模态 RAG 知识库（bge-m3 + CLIP + ChromaDB）
│   ├── config.py            全局配置
│   ├── embedders.py         双编码器（文本 / 图像）
│   ├── ingest.py            建库（分块 + 编码 + 入库）
│   ├── retriever.py         融合检索（类别硬过滤 + 加权重排 + 类别投票）
│   ├── generator.py         防治建议生成（Ollama + 降级）
│   ├── api.py / cli.py      接口与命令行入口
│   └── data/docs/           病害知识文档
├── docs/ROADMAP.md          开发路线与验收清单
├── tests/                   pytest 测试
├── .env.example
├── pytest.ini
├── requirements.txt         Web 基础依赖
└── requirements-dev.txt     测试依赖
```

## 本地启动

建议使用 Python 3.11 或 3.12。

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

python -m pip install -r requirements.txt

# 开发/测试依赖
python -m pip install -r requirements-dev.txt

# 运行测试
pytest

# 启动服务
uvicorn backend.app.main:app --reload
```

打开以下地址验证服务：

- 健康检查：http://127.0.0.1:8000/health
- API 文档：http://127.0.0.1:8000/docs

## 当前 API

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 项目基本信息 |
| GET | `/health` | 服务健康状态 |

模型推理、知识库入库、检索评估和前端页面将在后续迭代中实现。只有在真实运行并完成测试后，才应将对应功能标记为完成。

## 数据与实验规范

- 使用公开数据集时，遵守其许可与使用条款，并在文档中注明来源。
- 不要提交大型数据集、模型权重、密钥、个人信息或本地环境文件。
- 记录数据集划分、随机种子、模型配置、硬件环境和评估指标。
- 对比实验应在一致的数据划分与评估设置下进行；不预先填写未经验证的准确率或性能提升。

## 开发路线

详见 [docs/ROADMAP.md](docs/ROADMAP.md)。

## 许可证

当前仓库尚未指定许可证。确定开源范围后再添加合适的 LICENSE 文件。
