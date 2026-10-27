"""融合检索层：双库召回 + 类别硬过滤 + 加权重排。

本文件是**库模块**，只定义 MultimodalRetriever 类与打分函数，不提供命令行。
直接 `python retriever.py` 不会执行检索（仅打印入口提示与一次无依赖的纯函数演示）。
要跑检索请用：
    python cli.py text "晚疫病的防治措施" --disease 番茄晚疫病
    python cli.py image path/to/leaf.jpg --recommend
    python selftest.py --light        # 不需要模型权重，环境自检

三种检索入口
    1) search_by_disease  检测结果驱动（主链路）
       已知病害类别 → 类别硬过滤 → 文本库召回防治知识 + 图像库返回同类别标准症状图
    2) search_by_image    以图搜病（冷启动）
       用户直接拍图 → CLIP 编码 → 图像库 top-k → 按类别聚合投票 → 再走链路 1
    3) search_hybrid      图文联合
       文本查询与图像查询各自召回，加权合并

另加一条文本冷启动链路，与链路 2 对称：
    4) search_by_description  自述症状搜病（文本冷启动）
       用户口语描述症状 → bge-m3 编码 → 症状类知识块 top-k → 按病害类别聚合投票
       → 候选排序 + 是否足以定论 → 再走链路 1 取防治建议

融合打分（可写进论文的公式）
    final = W_DENSE * dense_sim + W_LEXICAL * lexical_overlap + W_PRIOR * prior
    其中
      dense_sim       向量余弦相似度（bge-m3 语义匹配）
      lexical_overlap 查询字面覆盖率（查询的字符二元组原样出现在块中的比例），
                      补偿向量检索对药剂名称、数字等字面信息的弱敏感
      prior           类别一致性与文档类型先验

类别投票（以图搜病与自述症状搜病共用同一条公式，便于横向比较）
    score(cls) = VOTE_MAX * max_sim + VOTE_MEAN_TOP3 * mean_top3
               + VOTE_COVERAGE * (命中数 / 最大命中数)
"""
from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

import class_map
import config


def _sim(distance: float) -> float:
    """Chroma 返回余弦距离，转换为相似度并裁剪到 [0, 1]。"""
    return float(max(0.0, min(1.0, 1.0 - distance)))


def _bigrams(text: str) -> set[str]:
    text = "".join(text.split())
    if len(text) < 2:
        return {text} if text else set()
    return {text[i : i + 2] for i in range(len(text) - 1)}


def _lexical_overlap(query: str, doc: str) -> float:
    """查询字面覆盖率：查询的字符二元组有多大比例原样出现在文档中。

    用覆盖率而非 Jaccard 系数，是因为本项要解决的具体问题是"查询里的药剂名、
    病害名是否原样出现在知识块中"。这个场景下查询往往只有几个字（如"代森锰锌"），
    而知识块有数百字，Jaccard 会被文档长度严重稀释——4 字查询对上 300 字文档即使
    完全命中，Jaccard 也仅约 0.03，乘以 0.25 的权重后等于这一项没起作用；
    覆盖率则能给出 1.0，真正实现对字面精确匹配的补偿。

    代价是对"查询词全为高频词"的情况不够敏感，但由于查询是系统按病害类别与
    意图构造的（含病害名与"防治措施""药剂"等实义词），实际影响很小。
    """
    q, d = _bigrams(query), _bigrams(doc)
    if not q or not d:
        return 0.0
    return len(q & d) / len(q)


def _build_where(
    disease_name: str | None = None,
    crop_type: str | None = None,
    doc_types: Sequence[str] | None = None,
) -> dict | None:
    conds: list[dict] = []
    if disease_name:
        conds.append({"disease_name": {"$eq": disease_name}})
    if crop_type:
        conds.append({"crop_type": {"$eq": crop_type}})
    if doc_types:
        conds.append({"doc_type": {"$in": list(doc_types)}})
    if not conds:
        return None
    return conds[0] if len(conds) == 1 else {"$and": conds}


def _aggregate_by_class(hits: list[dict]) -> list[dict]:
    """按病害类别聚合打分。

    hits 中每项需含 disease_name / crop_type / similarity，其余键原样保留在
    evidence 里（图像链路存图片路径，文本链路存知识块）。

    以图搜病与自述症状搜病共用本函数，保证两条冷启动链路的判定口径一致，
    也便于在论文中说明"同一套投票公式应用于两种模态"。

    打分同时考虑三项，是为了抑制单条异常命中：
      max_sim   最高相似度 —— 反映最像的那一条
      mean_top3 前三均值   —— 反映该类别的整体吻合程度，避免被单张异常图/单个异常块带偏
      coverage  命中覆盖   —— 反映该类别的证据是否集中，防止其它类别的少量样本挤占候选位
    """
    groups: dict[str, dict] = {}
    for h in hits:
        name = h.get("disease_name")
        if not name:
            continue
        g = groups.setdefault(
            name,
            {"disease_name": name, "crop_type": h.get("crop_type"), "sims": [], "evidence": []},
        )
        g["sims"].append(float(h.get("similarity") or 0.0))
        g["evidence"].append(h)

    if not groups:
        return []

    max_hits = max(len(g["sims"]) for g in groups.values())
    out: list[dict] = []
    for name, g in groups.items():
        sims = sorted(g["sims"], reverse=True)
        max_sim = sims[0]
        mean_top3 = float(np.mean(sims[:3]))
        coverage = len(sims) / max_hits
        score = (
            config.VOTE_MAX * max_sim
            + config.VOTE_MEAN_TOP3 * mean_top3
            + config.VOTE_COVERAGE * coverage
        )
        out.append(
            {
                "disease_name": name,
                "crop_type": g["crop_type"],
                "score": round(float(score), 4),
                "max_similarity": round(float(max_sim), 4),
                "mean_top3_similarity": round(mean_top3, 4),
                "hit_count": len(sims),
                "evidence": sorted(
                    g["evidence"], key=lambda x: x.get("similarity") or 0.0, reverse=True
                ),
            }
        )
    out.sort(key=lambda x: x["score"], reverse=True)
    return out


class MultimodalRetriever:
    """双库融合检索器。首次使用时惰性加载向量库与编码器。"""

    def __init__(self) -> None:
        self._text_col = None
        self._image_col = None

    @property
    def text_col(self):
        if self._text_col is None:
            from ingest import get_text_collection

            self._text_col = get_text_collection()
        return self._text_col

    @property
    def image_col(self):
        if self._image_col is None:
            from ingest import get_image_collection

            self._image_col = get_image_collection()
        return self._image_col

    # ------------------------------------------------------------ 文本检索
    def retrieve_text(
        self,
        query: str,
        disease_name: str | None = None,
        crop_type: str | None = None,
        doc_types: Sequence[str] | None = None,
        top_k: int | None = None,
        candidate_k: int = 24,
    ) -> list[dict]:
        """从文本库召回知识块并做融合重排。"""
        from embedders import get_text_embedder

        top_k = top_k or config.TOP_K_TEXT
        total = self.text_col.count()
        if total == 0:
            return []

        query_vec = get_text_embedder().encode_query(query)
        res = self.text_col.query(
            query_embeddings=[query_vec.tolist()],
            n_results=min(candidate_k, total),
            where=_build_where(disease_name, crop_type, doc_types),
            include=["documents", "metadatas", "distances"],
        )

        docs = res["documents"][0] if res["documents"] else []
        metas = res["metadatas"][0] if res["metadatas"] else []
        dists = res["distances"][0] if res["distances"] else []
        ids = res["ids"][0] if res["ids"] else []

        prefer = {"prevention", "pesticide", "symptom"}
        items: list[dict] = []
        for cid, text, meta, dist in zip(ids, docs, metas, dists):
            dense = _sim(dist)
            lex = _lexical_overlap(query, text)
            prior = 0.0
            if disease_name and meta.get("disease_name") == disease_name:
                prior += 0.6
            if meta.get("doc_type") in prefer:
                prior += 0.4
            score = (
                config.W_DENSE * dense
                + config.W_LEXICAL * lex
                + config.W_PRIOR * prior
            )
            items.append(
                {
                    "chunk_id": cid,
                    "chunk_text": text,
                    "dense_sim": round(dense, 4),
                    "lexical": round(lex, 4),
                    "prior": round(prior, 4),
                    "score": round(score, 4),
                    "doc_path": meta.get("doc_path"),
                    "doc_title": meta.get("doc_title"),
                    "doc_type": meta.get("doc_type"),
                    "doc_type_label": meta.get("doc_type_label"),
                    "heading": meta.get("heading"),
                    "disease_name": meta.get("disease_name"),
                    "crop_type": meta.get("crop_type"),
                    "source": meta.get("source"),
                }
            )

        items.sort(key=lambda x: x["score"], reverse=True)
        return items[:top_k]

    # ------------------------------------------------------------ 图像检索
    def _class_representatives(
        self, disease_name: str, crop_type: str | None, top_k: int
    ) -> list[dict]:
        """无查询图时，返回该类别代表性症状图（等距抽样）。"""
        where = _build_where(disease_name, crop_type)
        try:
            data = self.image_col.get(where=where, include=["metadatas"])
        except Exception:
            return []
        metas = data.get("metadatas") or []
        if not metas:
            return []
        step = max(1, len(metas) // top_k)
        picked = metas[::step][:top_k]
        return [
            {
                "image_path": m.get("image_path"),
                "disease_name": m.get("disease_name"),
                "crop_type": m.get("crop_type"),
                "similarity": None,
                "rank": i + 1,
                "source": m.get("source"),
                "is_reference": True,
            }
            for i, m in enumerate(picked)
        ]

    def retrieve_images(
        self,
        image=None,
        disease_name: str | None = None,
        crop_type: str | None = None,
        top_k: int | None = None,
    ) -> list[dict]:
        """图像库检索。

        - 传入 image：按与该图的相似度排序（若同时给定类别，则在该类别内排序）
        - 不传 image：返回该病害类别的代表性症状图
        """
        top_k = top_k or config.TOP_K_IMAGE
        if self.image_col.count() == 0:
            return []

        if image is None:
            if not disease_name:
                return []
            return self._class_representatives(disease_name, crop_type, top_k)

        from embedders import get_image_embedder

        vec = get_image_embedder().encode_image(image)
        res = self.image_col.query(
            query_embeddings=[vec.tolist()],
            n_results=min(max(top_k * 3, top_k), self.image_col.count()),
            where=_build_where(disease_name, crop_type),
            include=["metadatas", "distances"],
        )
        metas = res["metadatas"][0] if res["metadatas"] else []
        dists = res["distances"][0] if res["distances"] else []
        out = []
        for i, (m, d) in enumerate(zip(metas, dists)):
            out.append(
                {
                    "image_path": m.get("image_path"),
                    "disease_name": m.get("disease_name"),
                    "crop_type": m.get("crop_type"),
                    "similarity": round(_sim(d), 4),
                    "rank": i + 1,
                    "source": m.get("source"),
                    "is_reference": False,
                }
            )
        out.sort(key=lambda x: x["similarity"], reverse=True)
        for i, item in enumerate(out[:top_k]):
            item["rank"] = i + 1
        return out[:top_k]

    # ------------------------------------------------------------ 以图搜病
    def vote_disease(
        self, image, candidate_k: int = config.CANDIDATE_K_IMAGE
    ) -> list[dict]:
        """图像库全局检索后按类别投票，输出候选病害及置信度。"""
        if self.image_col.count() == 0:
            return []
        from embedders import get_image_embedder

        vec = get_image_embedder().encode_image(image)
        res = self.image_col.query(
            query_embeddings=[vec.tolist()],
            n_results=min(candidate_k, self.image_col.count()),
            include=["metadatas", "distances"],
        )
        metas = res["metadatas"][0] if res["metadatas"] else []
        dists = res["distances"][0] if res["distances"] else []

        hits = [
            {
                "disease_name": m.get("disease_name"),
                "crop_type": m.get("crop_type"),
                "similarity": _sim(d),
                "image_path": m.get("image_path"),
            }
            for m, d in zip(metas, dists)
        ]
        grouped = _aggregate_by_class(hits)
        for g in grouped:
            g["evidence_images"] = [e.get("image_path") for e in g.pop("evidence")[:3]]
        return grouped

    # ------------------------------------------------------------ 三种检索入口
    def search_by_disease(
        self,
        disease_name: str,
        crop_type: str | None = None,
        image=None,
        query: str | None = None,
        top_k_text: int | None = None,
        top_k_image: int | None = None,
    ) -> dict:
        """主链路：已知病害类别 → 文本知识 + 参考症状图。"""
        info = class_map.resolve_name(disease_name) or {}
        disease_name = info.get("disease_name", disease_name)
        crop_type = crop_type or info.get("crop_type")

        text_query = query or (
            f"{crop_type or ''}{disease_name} 典型症状 发病规律 防治措施 推荐药剂 用法用量 农事管理"
        )
        chunks = self.retrieve_text(
            text_query, disease_name=disease_name, crop_type=crop_type, top_k=top_k_text
        )
        images = self.retrieve_images(
            image=image, disease_name=disease_name, crop_type=crop_type, top_k=top_k_image
        )
        return {
            "mode": "by_disease",
            "disease_name": disease_name,
            "crop_type": crop_type,
            "disease_id": info.get("disease_id"),
            "query_used": text_query,
            "text_hits": chunks,
            "image_hits": images,
        }

    def search_by_image(self, image, top_k_text: int | None = None) -> dict:
        """冷启动链路：用户拍图 → 类别投票 → 用 Top-1 类别检索知识。"""
        votes = self.vote_disease(image)
        if not votes:
            return {
                "mode": "by_image",
                "candidates": [],
                "text_hits": [],
                "image_hits": [],
                "note": "图像库为空，请先执行建库",
            }
        best = votes[0]
        result = self.search_by_disease(
            best["disease_name"], best.get("crop_type"), image=image, top_k_text=top_k_text
        )
        result["mode"] = "by_image"
        result["candidates"] = votes[:3]
        result["confidence"] = best["score"]
        return result

    # ------------------------------------------------------------ 自述症状搜病
    def identify_disease_by_text(
        self,
        description: str,
        candidate_k: int = 30,
        top_k: int = 3,
    ) -> dict:
        """按用户自述症状推断候选病害（文本版的"以图搜病"）。

        只检索 symptom 类知识块，理由：防治措施、用药参考这些章节在不同病害之间
        高度相似（都写"轮作换茬、清洁田园、交替用药"），纳入投票会把各类别分数拉平；
        症状描述才是区分病害的判别性信息。

        与 vote_disease 对称——图像链路在图像库里找相似图后按类别投票，本方法在
        症状知识块里做语义检索后按类别投票，二者共用 _aggregate_by_class 的打分公式。
        """
        from embedders import get_text_embedder

        empty = {
            "description": description,
            "candidates": [],
            "ambiguous": True,
            "margin": None,
            "min_score_threshold": config.IDENTIFY_MIN_SCORE,
        }
        total = self.text_col.count()
        if total == 0:
            return {**empty, "note": "知识库为空，请先执行建库"}

        query_vec = get_text_embedder().encode_query(description)
        res = self.text_col.query(
            query_embeddings=[query_vec.tolist()],
            n_results=min(candidate_k, total),
            where=_build_where(doc_types=["symptom"]),
            include=["documents", "metadatas", "distances"],
        )
        docs = res["documents"][0] if res["documents"] else []
        metas = res["metadatas"][0] if res["metadatas"] else []
        dists = res["distances"][0] if res["distances"] else []

        hits = [
            {
                "disease_name": m.get("disease_name"),
                "crop_type": m.get("crop_type"),
                "similarity": _sim(d),
                "lexical": round(_lexical_overlap(description, text), 4),
                "heading": m.get("heading"),
                "doc_title": m.get("doc_title"),
                "doc_type_label": m.get("doc_type_label"),
                "source": m.get("source"),
                "chunk_text": text,
            }
            for text, m, d in zip(docs, metas, dists)
        ]

        candidates = _aggregate_by_class(hits)[:top_k]
        if not candidates:
            return {**empty, "note": "知识库中没有症状类知识块，无法按自述症状推断"}

        best = candidates[0]["score"]
        second = candidates[1]["score"] if len(candidates) > 1 else 0.0
        margin = round(best - second, 4)
        ambiguous = best < config.IDENTIFY_MIN_SCORE or margin < config.IDENTIFY_MIN_MARGIN

        if best < config.IDENTIFY_MIN_SCORE:
            note = (
                f"自述症状与知识库中各种病害的症状描述匹配度都不高（最高 {best:.2f}，"
                f"低于判定门槛 {config.IDENTIFY_MIN_SCORE}）。可能是知识库尚未收录该病害，"
                f"或描述信息过少。建议补充发病部位、病斑颜色与形状、是否有霉层等信息，"
                f"或直接上传病斑照片改用以图搜病。"
            )
        elif margin < config.IDENTIFY_MIN_MARGIN:
            note = (
                f"前两位候选得分接近（{candidates[0]['disease_name']} {best:.2f} / "
                f"{candidates[1]['disease_name']} {second:.2f}），仅凭当前描述难以区分，"
                f"已并列呈现供人工判断。建议补充拍摄病斑特写照片。"
            )
        else:
            note = "依据自述症状与知识库症状描述的语义匹配给出候选排序，建议结合病斑照片确认。"

        return {
            "description": description,
            "candidates": candidates,
            "ambiguous": bool(ambiguous),
            "margin": margin,
            "min_score_threshold": config.IDENTIFY_MIN_SCORE,
            "note": note,
        }

    def search_by_description(
        self,
        description: str,
        top_k_text: int | None = None,
        top_k_image: int | None = None,
    ) -> dict:
        """自述症状驱动的完整链路：描述 → 推断候选 → 按首选候选取知识与参考图。

        与 search_by_image 对称。即使系统标记为"不足以定论"，仍以 Top-1 候选组装
        参考资料，但同时返回候选列表与判定说明，由前端提示用户复核——对农户而言，
        拿到"两种可能 + 需补充确认"比什么都拿不到更有用。
        """
        ident = self.identify_disease_by_text(description)
        candidates = ident.get("candidates") or []
        if not candidates:
            return {
                "mode": "by_description",
                "description": description,
                "disease_name": None,
                "crop_type": None,
                "candidates": [],
                "ambiguous": True,
                "text_hits": [],
                "image_hits": [],
                "note": ident.get("note"),
            }

        best = candidates[0]
        # 检索查询把自述症状与防治意图拼接：农户的自述里通常不含"防治""药剂"这类词，
        # 直接拿原文去检索会偏向症状类知识块，取不到防治措施与用药参考。
        result = self.search_by_disease(
            best["disease_name"],
            best.get("crop_type"),
            query=f"{description} 典型症状 防治措施 推荐药剂 农事管理",
            top_k_text=top_k_text,
            top_k_image=top_k_image,
        )
        result["mode"] = "by_description"
        result["description"] = description
        result["candidates"] = candidates
        result["ambiguous"] = ident.get("ambiguous")
        result["margin"] = ident.get("margin")
        result["confidence"] = best["score"]
        result["note"] = ident.get("note")
        return result

    def search_hybrid(
        self,
        text: str,
        image=None,
        disease_name: str | None = None,
        crop_type: str | None = None,
        top_k_text: int | None = None,
    ) -> dict:
        """图文联合检索：两路召回后按 0.6/0.4 加权合并（权重可在 config 调整）。"""
        chunks = self.retrieve_text(
            text, disease_name=disease_name, crop_type=crop_type, top_k=(top_k_text or config.TOP_K_TEXT) * 2
        )
        votes = self.vote_disease(image) if image is not None else []
        if votes:
            voted_name = votes[0]["disease_name"]
            merge: dict[str, dict] = {}
            for c in chunks:
                key = c["chunk_id"]
                boost = 0.4 if c.get("disease_name") == voted_name else 0.0
                c["score"] = round(c["score"] * 0.6 + boost, 4)
                merge[key] = c
            chunks = sorted(merge.values(), key=lambda x: x["score"], reverse=True)

        target_disease = disease_name or (votes[0]["disease_name"] if votes else None)
        images = (
            self.retrieve_images(
                image=image,
                disease_name=target_disease,
                crop_type=crop_type,
            )
            if target_disease
            else []
        )
        top_k = top_k_text or config.TOP_K_TEXT
        return {
            "mode": "hybrid",
            "disease_name": target_disease,
            "crop_type": crop_type,
            "candidates": votes[:3],
            "text_hits": chunks[:top_k],
            "image_hits": images,
        }


_retriever: MultimodalRetriever | None = None


def get_retriever() -> MultimodalRetriever:
    global _retriever
    if _retriever is None:
        _retriever = MultimodalRetriever()
    return _retriever


if __name__ == "__main__":
    # 本模块是库，不是脚本。直接运行时给个明确的入口提示，并演示不依赖任何
    # 模型权重的纯函数部分——避免"跑完什么都没有"被误当成出错。
    print(__doc__.split("三种检索入口")[0].strip())
    print()
    print("=" * 72)
    print("当前文件定位：库模块（MultimodalRetriever 类 + 打分函数），无命令行入口。")
    print("=" * 72)
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
    print()
    print("-" * 72)
    print("以下是本模块中不依赖模型权重的打分逻辑演示：")
    print("-" * 72)

    # 1) 字面覆盖率：向量检索对药剂名不敏感，这一项负责把字面精确匹配拉回来
    q = "代森锰锌"
    for doc in ["可选用代森锰锌进行防治", "可选用丙森锌进行防治", "田间管理以通风降湿为主"]:
        print(f"  字面覆盖率  {q!r} vs {doc!r:>16} = {_lexical_overlap(q, doc):.2f}")

    # 2) 融合打分：同样的向量相似度下，"类别一致 + 防治类文档"应当排更前
    def _score(dense: float, lex: float, prior: float) -> float:
        return config.W_DENSE * dense + config.W_LEXICAL * lex + config.W_PRIOR * prior

    hit = _score(0.80, 0.20, 1.0)
    miss = _score(0.85, 0.05, 0.0)
    print()
    print(f"  融合打分  命中先验 {hit:.3f}  >  相似但无先验 {miss:.3f}  "
          f"（权重 {config.W_DENSE}/{config.W_LEXICAL}/{config.W_PRIOR}）")
    print()
    print("注：以上为纯函数演示，未加载向量库。完整链路请用 cli.py。")
