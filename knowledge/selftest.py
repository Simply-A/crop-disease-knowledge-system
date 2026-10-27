"""自检脚本：验证知识库构建链路的正确性。

分两级运行：

    轻量级（只需 pyyaml + numpy）
        校验类别映射、元数据解析、分块质量、章节类型继承、融合打分逻辑、
        提示词组装。在没有 chromadb / torch 的机器上也能跑。

    完整级（额外需要 chromadb + sentence-transformers + transformers + torch）
        真正建库并跑一次检索，验证端到端链路。

用法：
    python selftest.py            # 自动选择：依赖齐全就跑完整级
    python selftest.py --light    # 只跑轻量级
"""
from __future__ import annotations

import argparse
import sys
import traceback

PASS, FAIL = "PASS", "FAIL"
_results: list[tuple[str, str, str]] = []


def check(name: str):
    """装饰器：执行一个检查项并记录结果。"""

    def deco(fn):
        try:
            detail = fn() or ""
            _results.append((PASS, name, str(detail)))
        except AssertionError as exc:
            _results.append((FAIL, name, f"断言失败：{exc}"))
        except Exception as exc:
            _results.append((FAIL, name, f"{type(exc).__name__}: {exc}"))
            traceback.print_exc()
        return fn

    return deco


# ============================================================ 轻量级检查
@check("类别映射：目录名解析（含尾下划线与大小写容错）")
def t_class_dir():
    import class_map

    a = class_map.resolve_dir("Tomato___Early_blight")
    b = class_map.resolve_dir("corn_(MAIZE)___common_rust_")
    c = class_map.resolve_dir("不存在的目录")
    assert a and a["disease_name"] == "番茄早疫病", a
    assert a["crop_type"] == "番茄", a
    assert b and b["disease_name"] == "玉米普通锈病", b
    assert c is None, c
    return f"共登记 {len(class_map.all_names())} 个类别"


@check("类别映射：中文病名与别名反查，歧义别名拒绝解析")
def t_class_name():
    import class_map

    a = class_map.resolve_name("番茄早疫病")
    b = class_map.resolve_name("potato late blight")
    c = class_map.resolve_name("早疫病")  # 番茄与马铃薯共用该别名
    assert a and a["disease_id"] == "tomato_early_blight", a
    assert b and b["disease_name"] == "马铃薯晚疫病", b
    assert c is None, "歧义别名不应被解析为确定类别"

    ambiguous = class_map.ambiguous_names()
    assert "早疫病" in ambiguous, ambiguous
    assert sorted(ambiguous["早疫病"]) == ["番茄早疫病", "马铃薯早疫病"], ambiguous
    return f"已登记 {len(''.join(class_map.all_names()))} 字病名表；歧义别名 {list(ambiguous)} 已识别"


@check("文档解析：YAML 元数据与正文分离")
def t_front_matter():
    import config
    from ingest import parse_front_matter

    src = config.DOCS_DIR / "番茄_早疫病.md"
    assert src.exists(), f"示例文档缺失：{src}"
    meta, body = parse_front_matter(src.read_text(encoding="utf-8"))
    assert meta.get("disease_name") == "番茄早疫病", meta
    assert meta.get("crop_type") == "番茄", meta
    assert isinstance(meta.get("section_types"), dict), meta
    assert body.lstrip().startswith("#"), body[:40]
    assert "---" not in body.split("\n")[0], "正文中不应残留元数据分隔符"
    return f"元数据 {len(meta)} 项，正文 {len(body)} 字符"


@check("分块：长度受控、保留标题前缀、模板文件被排除")
def t_chunking():
    import config
    from ingest import collect_text_chunks

    col = collect_text_chunks(verbose=False)
    texts, metas = col["texts"], col["metadatas"]
    assert texts, "未产生任何知识块"
    assert col["skipped_missing_meta"] == [], col["skipped_missing_meta"]
    assert col["unresolved_names"] == [], f"示例文档的病名应能在 class_map 中解析：{col['unresolved_names']}"

    for t in texts:
        assert len(t) >= config.CHUNK_MIN, f"块过短：{len(t)}"
    carried = [m for m, t in zip(metas, texts) if not m["heading"] or t.startswith(m["heading"])]
    assert len(carried) >= 1, "没有任何块携带标题前缀"

    paths = {m["doc_path"] for m in metas}
    assert not any(p.startswith("_") for p in paths), f"模板文件被误入库：{paths}"
    assert len(paths) == col["doc_files"], (paths, col["doc_files"])
    return f"{col['doc_files']} 篇文档 → {len(texts)} 块，最长 {max(len(t) for t in texts)} 字符"


@check("分块：章节级知识类型继承（### 继承其 ## 父节）")
def t_section_types():
    from ingest import collect_text_chunks, resolve_section_type

    col = collect_text_chunks(verbose=False)
    types = {m["section_parent"] + "/" + m["heading"]: m["doc_type"] for m in col["metadatas"]}

    got = types.get("典型症状/典型症状")
    assert got == "symptom", types
    assert types.get("用药参考/用药参考") == "pesticide", types
    # 三级小节"农业防治"继承二级父节"防治措施"的类型
    assert types.get("防治措施/农业防治") == "prevention", types
    assert types.get("农事管理/农事管理") == "management", types

    # 直接测试解析函数的三级回退顺序
    st = {"防治措施": "prevention", "防治措施·化学防治": "pesticide"}
    assert resolve_section_type(st, "化学防治", "防治措施", "general") == "prevention"
    assert resolve_section_type(st, "防治措施", "防治措施", "general") == "prevention"
    assert resolve_section_type({}, "未知节", "未知父节", "general") == "general"
    return f"已正确识别 {len({m['doc_type'] for m in col['metadatas']})} 种知识类型"


@check("检索：字面覆盖率打分（药剂名等字面精确匹配）")
def t_lexical():
    from retriever import _lexical_overlap

    same = _lexical_overlap("代森锰锌", "可选用代森锰锌进行防治")
    partial = _lexical_overlap("代森锰锌防治", "可选用代森锰锌进行防治")
    other = _lexical_overlap("代森锰锌", "田间管理以通风降湿为主")
    near = _lexical_overlap("代森锰锌", "可选用丙森锌进行防治")

    assert same >= 0.9, f"完全命中应接近 1.0，实际 {same}"
    assert 0.5 < partial < 1.0, f"部分命中应落在 0~1 之间，实际 {partial}"
    assert other == 0.0, other
    assert near == 0.0, f"丙森锌与代森锰锌是两种药，不应获得字面得分（实际 {near}）"

    # 长文档不应稀释命中信号——这正是弃用 Jaccard 系数的原因
    long_doc = "田间管理要点。" * 50 + "可选用代森锰锌进行防治。"
    assert _lexical_overlap("代森锰锌", long_doc) >= 0.9, "长文档下命中信号被稀释"
    return f"命中 {same:.2f} / 部分 {partial:.2f} / 无关 {other:.2f}，长文档 300 字不稀释"


@check("检索：类别硬过滤条件构造")
def t_where():
    from retriever import _build_where

    assert _build_where() is None
    one = _build_where(disease_name="番茄早疫病")
    assert one == {"disease_name": {"$eq": "番茄早疫病"}}, one
    two = _build_where("番茄早疫病", "番茄")
    assert "$and" in two and len(two["$and"]) == 2, two
    three = _build_where("番茄早疫病", "番茄", ["prevention", "pesticide"])
    assert len(three["$and"]) == 3, three
    assert three["$and"][2]["doc_type"]["$in"] == ["prevention", "pesticide"]
    return "单条件 / $and 多条件组合均正确"


@check("检索：融合打分权重与排序方向")
def t_fusion_weights():
    import config

    total = config.W_DENSE + config.W_LEXICAL + config.W_PRIOR
    assert abs(total - 1.0) < 1e-6, f"融合权重之和应为 1.0，当前 {total}"
    v = config.VOTE_MAX + config.VOTE_MEAN_TOP3 + config.VOTE_COVERAGE
    assert abs(v - 1.0) < 1e-6, f"投票权重之和应为 1.0，当前 {v}"

    # 复算一遍打分公式，确认"类别一致 + 防治类文档"确实能拿到更高分
    def score(dense, lex, prior):
        return config.W_DENSE * dense + config.W_LEXICAL * lex + config.W_PRIOR * prior

    hit = score(0.80, 0.20, 1.0)
    miss = score(0.85, 0.05, 0.0)
    assert hit > miss, (hit, miss)
    return f"权重和校验通过；命中 {hit:.3f} > 相似但无先验 {miss:.3f}"


@check("生成：提示词包含检测结果、编号资料与强约束")
def t_prompt():
    import generator
    from ingest import collect_text_chunks

    col = collect_text_chunks(verbose=False)
    chunks = [
        {
            "doc_title": m["doc_title"],
            "heading": m["heading"],
            "doc_type_label": m["doc_type_label"],
            "source": m["source"],
            "chunk_text": t,
        }
        for m, t in list(zip(col["metadatas"], col["texts"]))[:3]
    ]
    prompt = generator.build_user_prompt("番茄早疫病", 0.93, chunks, [{"disease_name": "番茄早疫病"}], "番茄")
    assert "番茄早疫病" in prompt
    assert "93.00%" in prompt, prompt[:200]
    assert "[1]" in prompt and "[3]" in prompt, "检索资料未按编号标注"
    assert "参考病例图" in prompt
    assert "依据" in generator.SYSTEM_PROMPT and "禁止编造" in generator.SYSTEM_PROMPT
    for sec in ("病害确认", "判断依据", "防治措施", "用药参考", "注意事项"):
        assert sec in generator.SYSTEM_PROMPT, sec
    return f"提示词 {len(prompt)} 字符，含 {len(chunks)} 条编号资料"


@check("降级：大模型不可用时返回检索原文而非报错")
def t_fallback():
    import generator

    chunks = [{"doc_title": "测试文档", "heading": "防治措施", "chunk_text": "内容一二三"}]
    text = generator.fallback_advice("番茄早疫病", chunks, [])
    assert "番茄早疫病" in text and "内容一二三" in text
    out = generator.generate_advice("番茄早疫病", 0.9, chunks=chunks, images=[])
    assert out["degraded"] is True, "Ollama 未启动时应标记为降级"
    assert out["advice"], "降级时仍须返回内容"
    assert out["sources"][0]["doc_title"] == "测试文档"
    return "降级路径返回检索原文并保留来源"


# ============================================================ 完整级检查
def run_full() -> None:
    @check("端到端：建库 → 检索 → 生成（类别驱动）")
    def t_e2e():
        import ingest
        from retriever import get_retriever

        text_stats = ingest.build_text_index(reset=True, verbose=False)
        assert text_stats["chunks"] > 0, text_stats

        r = get_retriever()
        hits = r.retrieve_text("番茄早疫病的防治措施和推荐药剂", disease_name="番茄早疫病")
        assert hits, "文本检索无结果"
        assert hits[0]["disease_name"] == "番茄早疫病", hits[0]
        assert hits[0]["score"] > 0, hits[0]
        joined = " ".join(h["chunk_text"] for h in hits)
        assert ("早疫病" in joined), joined[:200]

        # 类别硬过滤必须生效：检索到的结果不应混入其他病害
        assert {h["disease_name"] for h in hits} == {"番茄早疫病"}, {h["disease_name"] for h in hits}
        return f"建库 {text_stats['chunks']} 块；Top-1 分数 {hits[0]['score']}，标题 {hits[0]['heading']}"

    @check("端到端：图像库构建与参考图召回")
    def t_image():
        import config
        import ingest

        if not any(p.is_dir() for p in config.IMAGE_DIR.iterdir()):
            return "图像库目录为空，跳过（按 data/image_library/README.md 准备数据后重跑）"
        stats = ingest.build_image_index(reset=True, verbose=False)
        assert stats["images"] > 0, stats
        from retriever import get_retriever

        imgs = get_retriever().retrieve_images(disease_name="番茄早疫病", top_k=5)
        assert imgs, "参考图召回为空"
        assert all(i["disease_name"] == "番茄早疫病" for i in imgs)
        return f"入库 {stats['images']} 张，召回 {len(imgs)} 张参考图"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--light", action="store_true", help="只跑轻量级检查")
    parser.add_argument("--full", action="store_true", help="强制跑完整级检查")
    args = parser.parse_args()

    heavy_ok = True
    if not args.light:
        try:
            import chromadb  # noqa: F401
            import torch  # noqa: F401

            heavy_ok = True
        except Exception:
            heavy_ok = False
            if args.full:
                print("[错误] --full 需要 chromadb 与 torch，请先 pip install -r requirements.txt")
                return 2

    if heavy_ok and not args.light:
        run_full()

    width = max(len(n) for _, n, _ in _results)
    print("\n" + "=" * 78)
    for status, name, detail in _results:
        mark = "✓" if status == PASS else "✗"
        print(f"{mark} {name.ljust(width)}  {detail}")
    failed = [n for s, n, _ in _results if s == FAIL]
    print("=" * 78)
    if failed:
        print(f"结果：{len(_results) - len(failed)}/{len(_results)} 项通过，失败项：{failed}")
        return 1
    tip = "（完整级）" if heavy_ok and not args.light else "（轻量级，未安装 chromadb/torch）"
    print(f"结果：全部 {len(_results)} 项检查通过 {tip}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
