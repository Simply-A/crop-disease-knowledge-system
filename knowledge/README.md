# 多模态 RAG 农艺知识库

农作物病害检测系统的知识服务模块。输入检测结果或一张病害图片，输出**有知识依据的防治建议**与**可供人工比对的参考病例图**。

文本与图像走两条独立的向量链路（bge-m3 / CLIP），在检索层按类别硬过滤后融合重排——这样既保住了中文专业文本的语义精度，又获得了跨模态的图像检索能力。

## 目录结构

```
multimodal_rag/
├── config.py            全局配置（模型、分块、权重、接口地址）
├── class_map.py         PlantVillage 目录名 ↔ 病害中文类别映射
├── embedders.py         双编码器封装（bge-m3 + CLIP，单例复用）
├── ingest.py            建库：文本切块入库 + 图像编码入库
├── retriever.py         融合检索：三种入口 + 类别投票 + 加权重排
├── generator.py         防治建议生成（提示词模板 + Ollama 调用 + 降级）
├── api.py               FastAPI 接口
├── cli.py               命令行工具
├── selftest.py          自检脚本：验证分块、类型继承、打分与提示词
├── 设计说明.md          论文与答辩素材（设计依据、实验方案、写作要点）
├── requirements.txt
├── data/
│   ├── docs/            知识文档（.md，YAML 元数据 + 正文）
│   │   ├── 番茄_早疫病.md        示例（完整）
│   │   ├── 马铃薯_晚疫病.md      示例（完整）
│   │   └── _模板.md             模板，下划线开头不会被入库
│   └── image_library/   病害症状图库，按类别分目录（见其中 README）
└── store/               Chroma 持久化目录（首次建库后自动生成）
```

## 文件速查：哪些能直接运行

**大部分文件是库模块，直接运行不会执行任何检索**（运行完没有任何输出是正常的，不是报错）。真正能作为入口运行的只有两个文件：

| 文件 | 定位 | 直接运行的效果 |
| --- | --- | --- |
| `cli.py` | ✅ **主入口** | 建库 / 检索 / 生成建议 / 启服务（子命令见下） |
| `selftest.py` | ✅ 自检入口 | 跑 10 项检查，`--light` 不需要模型权重 |
| `api.py` | ✅ 可运行 | 直接启动 FastAPI（等价于 `cli.py serve`） |
| `ingest.py` | ⚠️ 库 + 附带入口 | 直接运行 = 全量重建两个库 |
| `generator.py` | ⚠️ 库 + 附带入口 | 直接运行 = 用示例数据生成一次建议 |
| `retriever.py` | ❌ **纯库** | 打印入口提示 + 两项不依赖模型权重的打分演示 |
| `config.py` | ❌ 纯库 | 打印当前生效配置（JSON，含融合权重与模型名） |
| `class_map.py` | ❌ 纯库 | 打印类别映射表 + 跨作物别名歧义清单 |
| `embedders.py` | ❌ 纯库 | 未装依赖时报 `ModuleNotFoundError: numpy` —— 这是**正常的缺依赖提示**，装完依赖即消失 |

所以想验证"能不能跑通"，第一条命令应该是：

```bash
cd multimodal_rag
python selftest.py --light      # 不需要 chromadb/torch，几秒出结果
```

> **为什么 `retriever.py` 单独跑没反应**：它只定义了 `MultimodalRetriever` 类与几个打分函数，没有 `if __name__ == "__main__"` 分支。检索必须由 `cli.py` / `api.py` 调用才会发生，且需要先建库。

## 快速开始

> **用 VSCode 跑的话不用敲这些命令**：在项目根目录 `Ctrl+Shift+P` → `Tasks: Run Task` → 选 `RAG · 1 自检` / `RAG · 2 建库` / `RAG · 4 文本检索` 等任务即可（配置在项目根的 `.vscode/tasks.json`，与 YOLO 训练共用同一个虚拟环境 `D:\bishu\venvs\yolo`）。

### 1. 安装依赖

**推荐直接装项目根目录的合并清单** —— 检测与 RAG 都依赖 torch，一份装完两处，
不必重复下载约 3 GB：

```powershell
$py = "D:\bishu\venvs\yolo\Scripts\python.exe"
& $py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
& $py -m pip install -r C:\Users\今又是\Desktop\毕设2\requirements.txt
```

只装本模块时用本目录的清单（注意 torch 仍必须最先装）：

```bash
cd multimodal_rag
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

> **顺序不能反，这是实测过的坑**：`pip install torch`（不带索引）装到的是 CPU 版，
> 而让 `sentence-transformers` 顺带把 torch 拉进来，拿到的同样是 CPU 版。
> 实测 dry-run 解析出的是 `torch-2.14.0`、仅 124 MB —— 那是 CPU 版；GPU 版是 GB 级。
> CPU 版不会报错，只是编码慢到没法用。

首次运行会自动下载模型权重：bge-m3 约 2.2 GB、CLIP 约 600 MB。若磁盘或内存紧张，把 `config.TEXT_MODEL` 改为 `BAAI/bge-small-zh-v1.5`（约 95 MB）即可。

### 2. 准备数据

- **知识文档**：把病害资料写成 `.md` 放进 `data/docs/`，格式照抄 `_模板.md`。当前已有 2 篇示例可直接跑通。
- **症状图库**：按 `data/image_library/README.md` 里的脚本从 PlantVillage 复制图像，一个类别一个子目录。

### 3. 建库

```bash
python selftest.py --light          # 先自检：不依赖模型权重，几秒出结果
python cli.py build                 # 文本库 + 图像库全量重建
python cli.py build --text-only     # 只重建文本库（改完文档用这个）
python cli.py stats                 # 查看库内条目
```

装好依赖后直接跑 `python selftest.py`（不带参数）会自动带上端到端检查；`--light` 只跑不依赖模型的部分。自检全部通过再建库，出问题能快速定位是哪一环。

预期输出：

```
[文本库] 文档 2 个 → 切块 23 条，清理旧块 0 条，库内共 23 条
[图像库] 类别 10 个 → 入库 300 张（跳过损坏 0 张），库内共 300 张
```

### 4. 验证检索

```bash
# 文本检索（带类别硬过滤）
python cli.py text "晚疫病的防治措施" --disease 番茄晚疫病

# 以图搜病：不看文件名，只靠图像相似度投票
python cli.py image data/image_library/Tomato___Early_blight/某张图.jpg --recommend

# 生成防治建议（需先启动 Ollama）
ollama pull qwen2.5:7b
python cli.py advice 番茄早疫病 --confidence 0.93 --crop 番茄
```

以图搜病的预期输出：

```
候选病害（图像库投票）：
  [1] 番茄早疫病  投票分=0.812  最高相似度=0.941  命中=9 张
  [2] 番茄晚疫病  投票分=0.207  最高相似度=0.402  命中=3 张
```

### 5. 启动接口服务

```bash
python cli.py serve --reload
# 或 python -m uvicorn api:app --host 127.0.0.1 --port 8000 --reload
```

打开 http://127.0.0.1:8000/docs 可直接交互调试全部接口。

## 接口速查

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| GET | `/kb/health` | 服务状态与当前配置 |
| GET | `/kb/stats` | 两个向量库的条目与类别分布 |
| POST | `/kb/search/text` | 文本检索，可带 disease_name / crop_type / doc_types 过滤 |
| POST | `/kb/search/image` | 以图搜病，multipart 上传图片 |
| POST | `/kb/search/disease` | 检测结果驱动检索，返回知识块与参考图 |
| POST | `/kb/advice` | **完整闭环**：检测结果 → 检索 → 生成防治建议 |
| POST | `/kb/debug/pipeline` | 打印实际送入大模型的提示词，调参排错用 |
| POST | `/kb/reindex` | 重建索引 |

### 与检测模块对接

后端只需把 YOLOv8 的推理结果按下面的结构发给 `/kb/advice` 即可：

```bash
curl -X POST http://127.0.0.1:8000/kb/advice \
  -H "Content-Type: application/json" \
  -d '{
    "detection": {"disease_name": "番茄早疫病", "confidence": 0.93, "crop_type": "番茄"},
    "image_id": "20260911_093012.jpg",
    "top_k_text": 6,
    "top_k_image": 5
  }'
```

`image_id` 传检测时的原图文件名（需放在 `data/detected/` 下），系统会用它计算参考图的相似度排序；不传则返回该类别代表性症状图。

返回结构：

```json
{
  "disease_name": "番茄早疫病",
  "crop_type": "番茄",
  "confidence": 0.93,
  "advice": "## 一、病害确认\n检测模型判定为「番茄早疫病」…",
  "degraded": false,
  "sources": [{"index": 1, "doc_title": "番茄早疫病防治知识", "heading": "典型症状", "score": 0.79}],
  "reference_images": [{"rank": 1, "image_path": "Tomato___Early_blight/xxx.jpg", "similarity": 0.94}]
}
```

`degraded=true` 表示 Ollama 未启动，此时 `advice` 返回的是检索原文——系统不会白屏，答辩现场可以放心演示。

## 关键设计说明

**为什么不用一个 CLIP 编码两种模态。** CLIP 的文本塔对中文专业术语（"安全间隔期""稀释倍数"）语义表达很弱，用它检索防治文本会明显掉点；而 bge 系列完全不懂图像。所以文本库用 bge-m3 保精度，图像库用 CLIP 做图像相似，两者在检索层汇合。代价是要维护两套索引，收益是两条链路各自最优。

**融合打分**（`retriever.py`）：

```
final = 0.60 × 向量相似度 + 0.25 × 词面重合度 + 0.15 × 类别与类型先验
```

融合打分的第二项是**查询字面覆盖率**——查询的字符二元组有多大比例原样出现在知识块中。加这一项的原因：向量检索对"代森锰锌""嘧菌酯"这类药剂名称不敏感（它们语义相近但字面不同、且需要精确匹配），字面匹配能把这些查询拉回正确结果。这里用覆盖率而不是 Jaccard 系数：查询通常只有几个字而知识块有数百字，Jaccard 会被文档长度严重稀释（4 字查询对上 300 字文档即使完全命中也只有约 0.03），覆盖率则能给出 1.0，真正起到补偿作用。

**以图搜病的类别投票**：

```
score(类别) = 0.60 × 最高相似度 + 0.30 × 前三相似度均值 + 0.10 × 命中覆盖
```

只用最高相似度容易被单张异常样本带偏；加入前三均值和命中覆盖率后，判定结果对个别噪声样本更稳健。

**提示词的强约束。** 见 `generator.py` 的 `SYSTEM_PROMPT`：只允许依据检索资料作答、强制标注 `[n]` 来源编号、禁止编造药剂名称与剂量、资料未涉及时必须明说。这三条是针对大模型在农业领域的实际风险设计的——剂量错误是会出实事的。

## 常见问题

**建库报 UnicodeEncodeError。** Windows 控制台编码问题，先执行 `chcp 65001`，或设置环境变量 `PYTHONIOENCODING=utf-8`。

**中文路径找不到文件。** 本项目用 `pathlib` 处理路径，在 Windows 上正常工作；如果自定义脚本里出现字符串拼接路径，改成 `Path` 更稳妥。

**检索结果全是同一个文档的相邻块。** 这是分块重叠导致的近重复。把 `config.CHUNK_OVERLAP` 调小（如 40），或在返回前对同一 `doc_path` 做去重限制。

**Ollama 调用超时。** 首次调用会加载模型，耗时较长。把 `config.OLLAMA_TIMEOUT` 提到 180 秒以上，并确认 `ollama list` 中已有对应模型。

## 后续可扩展方向

1. **重排模型**：在融合层后接 bge-reranker，对 Top-20 候选做精排——这是当前性价比最高的效果提升手段。
2. **病害识别结果反馈**：把用户对建议的采纳情况写回 `feedback` 表，用于统计哪些知识条目真正被使用。
3. **多模态大模型兜底**：对 CLIP 投票分数低于阈值的图像，转交视觉大模型做二次判断，再走知识检索。
