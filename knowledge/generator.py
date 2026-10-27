"""防治建议生成：把融合检索结果组装成提示词，交给本地大模型生成。

设计原则
    1. 检索资料是唯一事实来源，提示词中明确禁止引入资料之外的信息；
    2. 强制按 [n] 标注引用编号，前端可据此回溯到原始知识块，实现可追溯；
    3. 药剂名称与用量属于高风险信息，资料中没有时要求模型明确回答"资料未涉及"；
    4. Ollama 不可用时降级返回检索原文，保证系统现场演示不会白屏。
"""
from __future__ import annotations

import json
from typing import Sequence

import config

SYSTEM_PROMPT = """你是农作物病害防治辅助系统的建议生成模块。你的回答必须严格遵守以下规则：

1. 只能依据【检索资料】中的内容回答，不允许引入资料之外的任何知识。
2. 每一条结论后面必须标注来源编号，格式为 [1]、[2]，编号对应检索资料的序号。
3. 禁止编造药剂名称、稀释倍数、施用剂量与安全间隔期。若检索资料中未提供相关信息，
   直接写明"检索资料未涉及，请查阅当地植保部门发布的用药指导"。
4. 不使用"可能""大概""应该是"等模糊表述描述病害，结论要依据检索资料中的症状描述。
5. 语气专业、简洁、面向农户可读，不使用学术论文式的长句。

输出结构固定为五节，每节标题使用 Markdown 二级标题：
## 一、病害确认
## 二、判断依据
## 三、防治措施
## 四、用药参考
## 五、注意事项
"""

IDENTIFY_RULES = """
补充规则（本次输入是农户的口头症状描述，不是检测模型的输出，因此还有以下约束）：

6. 必须明确说明本结论是"依据自述症状的推断"，不得表述为确诊结论。
7. 先给出候选病害排序及各自的判别依据（自述症状里的哪些表述支持哪个候选），
   再对首选候选给出防治建议。候选之间的区分点要写清楚。
8. 若系统标记自述信息不足以区分候选，必须如实说明还需要补充哪些信息
   （例如发病部位、病斑颜色与形状、是否有霉层、作物品种与生育期），不得强行断言。

输出结构固定为五节，每节标题使用 Markdown 二级标题：
## 一、症状归纳
## 二、候选病害与判别依据
## 三、防治措施
## 四、用药参考
## 五、需补充确认的信息
"""


def build_context(chunks: Sequence[dict]) -> str:
    """把检索到的知识块拼成带编号的上下文。"""
    if not chunks:
        return "（未检索到相关资料）"
    lines: list[str] = []
    for i, c in enumerate(chunks, start=1):
        title = c.get("doc_title") or c.get("doc_path") or "知识条目"
        heading = c.get("heading") or ""
        label = c.get("doc_type_label") or ""
        source = c.get("source") or ""
        head = f"[{i}] 《{title}》" + (f" · {heading}" if heading else "")
        meta = " | ".join([x for x in (label, source) if x])
        if meta:
            head += f"（{meta}）"
        lines.append(f"{head}\n{c.get('chunk_text', '').strip()}")
    return "\n\n".join(lines)


def build_user_prompt(
    disease_name: str,
    confidence: float | None,
    chunks: Sequence[dict],
    images: Sequence[dict] | None = None,
    crop_type: str | None = None,
) -> str:
    conf = f"{confidence * 100:.2f}%" if isinstance(confidence, (int, float)) else "未提供"
    img_lines = ""
    if images:
        names = sorted({i.get("disease_name", "") for i in images if i.get("disease_name")})
        img_lines = (
            f"\n【参考病例图】系统同时从图像库检索到 {len(images)} 张相似病斑图"
            f"（类别：{'、'.join(names)}），已在前端展示供人工比对。\n"
        )
    return (
        f"【检测结果】改进 YOLOv8 检测模型输出：作物为 {crop_type or '未知'}，"
        f"病害类别为「{disease_name}」，置信度 {conf}。\n"
        f"{img_lines}\n"
        f"【检索资料】\n{build_context(chunks)}\n\n"
        f"请依据上述检索资料，为农户生成一份防治建议，严格按系统提示规定的五节结构输出，"
        f"并在每条结论后标注来源编号。"
    )


def build_identify_prompt(
    description: str,
    candidates: Sequence[dict],
    chunks: Sequence[dict],
    images: Sequence[dict] | None = None,
) -> str:
    """自述症状场景的用户提示词：输入是推断出的候选而非检测结果。

    与 build_user_prompt 的关键差别：这里必须把候选排序与各自的匹配证据一并给出，
    并要求模型输出判别依据——因为"按描述推断"本身带有不确定性，把推断过程写出来
    才能让农户判断该不该采信，也才能让答辩时的结论可核对。
    """
    lines: list[str] = []
    for i, c in enumerate(candidates, start=1):
        lines.append(
            f"{i}. 「{c.get('disease_name')}」（作物：{c.get('crop_type') or '未知'}，"
            f"匹配分 {c.get('score')}，最高相似度 {c.get('max_similarity')}，"
            f"命中 {c.get('hit_count')} 条症状描述）"
        )
    cand_block = "\n".join(lines) if lines else "（未能给出候选）"

    img_lines = ""
    if images:
        img_lines = (
            f"\n【参考症状图】系统从图像库检索到 {len(images)} 张该病害的典型症状图，"
            f"已在前端展示供人工比对。\n"
        )

    return (
        f"【农户自述症状】\n{description}\n\n"
        f"【系统按语义匹配给出的候选病害排序】\n{cand_block}\n\n"
        f"{img_lines}\n"
        f"【检索资料】\n{build_context(chunks)}\n\n"
        f"请依据上述检索资料：先归纳农户描述中的关键症状，再判断最符合的候选病害"
        f"并说明判别依据，然后给出防治建议。严格按系统提示规定的五节结构输出，"
        f"并在每条结论后标注来源编号。"
    )


def call_ollama(messages: list[dict], stream: bool = False) -> str:
    """调用 Ollama /api/chat。requests 延迟导入，便于在没有该依赖时单独测试提示词组装。"""
    import requests

    payload = {
        "model": config.OLLAMA_MODEL,
        "messages": messages,
        "stream": stream,
        "options": {
            "temperature": config.LLM_TEMPERATURE,
            "top_p": config.LLM_TOP_P,
            "num_predict": config.LLM_NUM_PREDICT,
        },
    }
    resp = requests.post(
        f"{config.OLLAMA_HOST.rstrip('/')}/api/chat",
        json=payload,
        timeout=config.OLLAMA_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    return (data.get("message") or {}).get("content", "").strip()


def fallback_advice(disease_name: str, chunks: Sequence[dict], images: Sequence[dict]) -> str:
    """降级输出：直接呈现检索到的知识原文，附来源。"""
    parts = [
        "## 一、病害确认",
        f"检测模型判定为「{disease_name}」。",
        "",
        "## 二、判断依据",
        "本地大模型服务未启动，以下为知识库直接检索结果，供人工判读：",
        "",
    ]
    for i, c in enumerate(chunks, start=1):
        parts.append(f"**[{i}] {c.get('heading') or c.get('doc_title') or '知识块'}**")
        parts.append(c.get("chunk_text", "").strip())
        parts.append("")
    if images:
        parts.append(f"## 附加参考\n图像库检索到 {len(images)} 张相似病斑图，已在前端展示。")
    return "\n".join(parts)


def generate_advice(
    disease_name: str,
    confidence: float | None = None,
    chunks: Sequence[dict] | None = None,
    images: Sequence[dict] | None = None,
    crop_type: str | None = None,
    query: str | None = None,
) -> dict:
    """生成防治建议。

    若未直接传入 chunks，则自动走一次类别驱动的检索。
    """
    if chunks is None:
        from retriever import get_retriever

        chunks = get_retriever().retrieve_text(
            query
            or f"{crop_type or ''}{disease_name} 典型症状 防治措施 推荐药剂 农事管理",
            disease_name=disease_name,
            crop_type=crop_type,
        )
    chunks = list(chunks or [])
    images = list(images or [])

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_user_prompt(disease_name, confidence, chunks, images, crop_type),
        },
    ]

    error = None
    try:
        text = call_ollama(messages)
        if not text:
            raise RuntimeError("模型返回空内容")
    except Exception as exc:  # 网络不通、模型未拉取等
        error = f"{type(exc).__name__}: {exc}"
        text = fallback_advice(disease_name, chunks, images)

    return {
        "disease_name": disease_name,
        "crop_type": crop_type,
        "confidence": confidence,
        "advice": text,
        "degraded": error is not None,
        "error": error,
        "sources": [
            {
                "index": i,
                "doc_title": c.get("doc_title"),
                "doc_path": c.get("doc_path"),
                "heading": c.get("heading"),
                "doc_type_label": c.get("doc_type_label"),
                "source": c.get("source"),
                "score": c.get("score"),
                "dense_sim": c.get("dense_sim"),
            }
            for i, c in enumerate(chunks, start=1)
        ],
        "reference_images": [
            {
                "rank": i.get("rank"),
                "image_path": i.get("image_path"),
                "disease_name": i.get("disease_name"),
                "similarity": i.get("similarity"),
            }
            for i in images
        ],
    }


if __name__ == "__main__":
    print(json.dumps(generate_advice("番茄早疫病", 0.93, crop_type="番茄"), ensure_ascii=False, indent=2))
