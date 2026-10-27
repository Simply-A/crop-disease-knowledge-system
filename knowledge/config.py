"""多模态 RAG 农艺知识库 —— 全局配置。

所有可调参数集中在此，便于消融实验与论文中说明实验配置。

环境变量覆盖（可选）：
    RAG_TEXT_MODEL    文本编码模型，默认 BAAI/bge-m3
    RAG_IMAGE_MODEL   图像编码模型，默认 openai/clip-vit-base-patch32
    RAG_DEVICE        强制指定 cuda / cpu，默认自动
    RAG_STORE_DIR     向量库持久化目录
    OLLAMA_HOST       Ollama 服务地址
    OLLAMA_MODEL      生成用大模型
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------- 路径
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
DOCS_DIR = DATA_DIR / "docs"                  # 病害知识文档（.md）
IMAGE_DIR = DATA_DIR / "image_library"        # 病害症状图库（按类别分目录）
STORE_DIR = Path(os.getenv("RAG_STORE_DIR", str(BASE_DIR / "store")))

# ---------------------------------------------------------------- 编码模型
# 文本走中文语义模型，图像走 CLIP；两条链路互不干扰，在检索层汇合。
# 若显存/内存紧张，可换成轻量替代：
#   BAAI/bge-small-zh-v1.5（约 95MB）或 BAAI/bge-base-zh-v1.5（约 400MB）
TEXT_MODEL = os.getenv("RAG_TEXT_MODEL", "BAAI/bge-m3")
IMAGE_MODEL = os.getenv("RAG_IMAGE_MODEL", "openai/clip-vit-base-patch32")
DEVICE = os.getenv("RAG_DEVICE") or None      # None -> 自动选择 cuda / cpu

# ---------------------------------------------------------------- 向量库
COLL_TEXT = "kb_text"
COLL_IMAGE = "kb_image"
DISTANCE = "cosine"
UPSERT_BATCH = 256

# ---------------------------------------------------------------- 分块参数
CHUNK_SIZE = 420          # 单块最大字符数
CHUNK_MIN = 60            # 小于该长度的块丢弃
CHUNK_OVERLAP = 80        # 滑窗重叠字符数

# ---------------------------------------------------------------- 检索参数
TOP_K_TEXT = 6
TOP_K_IMAGE = 5
CANDIDATE_K_IMAGE = 15    # 以图搜病时的候选图数量
MAX_IMAGES_PER_CLASS = 30 # 每个病害类别入库的图像上限

# ---------------------------------------------------------------- 融合打分权重
# final = W_DENSE * 向量相似度 + W_LEXICAL * 词面重合度 + W_PRIOR * 类别先验
W_DENSE = 0.60
W_LEXICAL = 0.25
W_PRIOR = 0.15

# ---------------------------------------------------------------- 以图搜病的类别投票权重
# score(cls) = VOTE_MAX*最高相似度 + VOTE_MEAN_TOP3*前三均值 + VOTE_COVERAGE*命中覆盖
# 自述症状推断病害（文本冷启动）复用同一套权重，使两条冷启动链路可横向比较。
VOTE_MAX = 0.60
VOTE_MEAN_TOP3 = 0.30
VOTE_COVERAGE = 0.10

# ---------------------------------------------------------------- 自述症状推断的"不定论"门槛
# 语言描述的信息量天然低于图像，因此除给出候选排序外，还需判断是否足以定论：
#   最高候选分低于 IDENTIFY_MIN_SCORE   → 描述与知识库症状都对不上，宁可拒答
#   最高与次高分差小于 IDENTIFY_MIN_MARGIN → 描述不足以区分，应并列呈现候选
# 这两个阈值需在自述症状测试集上标定；当前取值为基于样例数据的经验估计值，
# 论文中应报告阈值敏感性分析结果。
IDENTIFY_MIN_SCORE = 0.52
IDENTIFY_MIN_MARGIN = 0.03

# ---------------------------------------------------------------- 生成模型
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))
LLM_TEMPERATURE = 0.2
LLM_TOP_P = 0.9
LLM_NUM_PREDICT = 900

# ---------------------------------------------------------------- 知识条目类型
DOC_TYPE_LABEL = {
    "symptom": "症状识别",
    "cause": "发病规律",
    "prevention": "防治措施",
    "pesticide": "用药参考",
    "management": "农事管理",
}


def ensure_dirs() -> None:
    """确保运行时所需目录存在。"""
    for d in (DOCS_DIR, IMAGE_DIR, STORE_DIR):
        d.mkdir(parents=True, exist_ok=True)


def describe() -> dict:
    """返回当前配置摘要，供 /kb/health 接口与实验记录使用。"""
    return {
        "text_model": TEXT_MODEL,
        "image_model": IMAGE_MODEL,
        "device": DEVICE or "auto",
        "store_dir": str(STORE_DIR),
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "top_k_text": TOP_K_TEXT,
        "top_k_image": TOP_K_IMAGE,
        "fusion_weights": {
            "dense": W_DENSE,
            "lexical": W_LEXICAL,
            "prior": W_PRIOR,
        },
        "llm": {"host": OLLAMA_HOST, "model": OLLAMA_MODEL},
    }


if __name__ == "__main__":
    # 本模块是库，不是脚本。直接运行时把生效配置打出来，避免"跑完什么反应都
    # 没有"被误当成出错——要跑检索请用 cli.py。
    import json

    print(__doc__.split("环境变量覆盖")[0].strip())
    print()
    print("=" * 62)
    print("当前文件定位：全局配置模块，无命令行入口。")
    print("=" * 62)
    print("下面是当前生效的配置值（可用环境变量覆盖模型 / 设备 / 向量库路径）：")
    print()
    print(json.dumps(describe(), ensure_ascii=False, indent=2))
    print()
    print("要真正跑检索，请改用下面任一条命令（在 multimodal_rag/ 目录下执行）：")
    for line in [
        "python selftest.py --light                  环境自检，不需要模型权重，几秒出结果",
        'python cli.py text "晚疫病的防治措施"        文本检索',
        "python cli.py image <图片路径> --recommend   以图搜病",
        "python cli.py build                         建库（需 chromadb + torch）",
        "python cli.py serve                         启动 FastAPI 服务",
    ]:
        print("  " + line)
