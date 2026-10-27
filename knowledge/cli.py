"""命令行入口，用于建库、检索调试与实验跑批。

常用命令：
    python cli.py build                  # 全量重建文本库与图像库
    python cli.py build --text-only      # 只重建文本库
    python cli.py stats                  # 查看库内条目统计
    python cli.py text "番茄晚疫病怎么防治" --disease 番茄晚疫病
    python cli.py image path/to/leaf.jpg # 以图搜病
    python cli.py serve                  # 启动接口服务
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import config


def _print(obj) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2, default=str))


def cmd_build(args) -> None:
    import ingest

    if args.text_only:
        _print(ingest.build_text_index(reset=not args.append))
    elif args.image_only:
        _print(
            ingest.build_image_index(reset=not args.append, limit_per_class=args.per_class)
        )
    else:
        _print(ingest.build_text_index(reset=not args.append))
        _print(ingest.build_image_index(reset=not args.append, limit_per_class=args.per_class))


def cmd_stats(args) -> None:
    import ingest

    _print(ingest.stats())


def cmd_text(args) -> None:
    from retriever import get_retriever

    hits = get_retriever().retrieve_text(
        args.query,
        disease_name=args.disease,
        crop_type=args.crop,
        doc_types=args.doc_type,
        top_k=args.top_k,
    )
    if args.json:
        _print(hits)
        return
    for i, h in enumerate(hits, start=1):
        print(
            f"[{i}] score={h['score']} dense={h['dense_sim']} lex={h['lexical']} "
            f"| {h['doc_title']} · {h['heading']} · {h['doc_type_label']}"
        )
        print(f"    {h['chunk_text'][:120].replace(chr(10), ' ')}…")


def cmd_image(args) -> None:
    from PIL import Image

    from retriever import get_retriever

    path = Path(args.path)
    if not path.exists():
        sys.exit(f"图片不存在：{path}")
    image = Image.open(path).convert("RGB")
    retriever = get_retriever()

    votes = retriever.vote_disease(image)
    print("候选病害（图像库投票）：")
    for i, v in enumerate(votes[:5], start=1):
        print(
            f"  [{i}] {v['disease_name']}  投票分={v['score']}  "
            f"最高相似度={v['max_similarity']}  命中={v['hit_count']} 张"
        )

    if args.recommend and votes:
        result = retriever.search_by_image(image)
        print(f"\n以「{result.get('disease_name')}」检索到的知识块：")
        for i, h in enumerate(result.get("text_hits", []), start=1):
            print(f"  [{i}] {h['score']} | {h['doc_title']} · {h['heading']}")
            print(f"      {h['chunk_text'][:100].replace(chr(10), ' ')}…")


def cmd_advice(args) -> None:
    import generator

    out = generator.generate_advice(
        args.disease, confidence=args.confidence, crop_type=args.crop
    )
    if out["degraded"]:
        print(f"[警告] 大模型调用失败，已降级返回检索原文（{out['error']}）\n")
    print(out["advice"])
    print("\n---- 知识来源 ----")
    for s in out["sources"]:
        print(f"[{s['index']}] {s['doc_title']} · {s['heading']} （{s['source']}）")


def cmd_serve(args) -> None:
    import uvicorn

    uvicorn.run("api:app", host=args.host, port=args.port, reload=args.reload)


def main() -> None:
    parser = argparse.ArgumentParser(description="多模态 RAG 农艺知识库")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("build", help="构建向量库")
    p.add_argument("--text-only", action="store_true")
    p.add_argument("--image-only", action="store_true")
    p.add_argument("--append", action="store_true", help="增量入库，不先清空")
    p.add_argument("--per-class", type=int, default=config.MAX_IMAGES_PER_CLASS)
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("stats", help="查看库状态")
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("text", help="文本检索")
    p.add_argument("query")
    p.add_argument("--disease", default=None)
    p.add_argument("--crop", default=None)
    p.add_argument("--doc-type", nargs="*", default=None)
    p.add_argument("--top-k", type=int, default=config.TOP_K_TEXT)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_text)

    p = sub.add_parser("image", help="以图搜病")
    p.add_argument("path")
    p.add_argument("--recommend", action="store_true", help="继续检索防治知识")
    p.set_defaults(func=cmd_image)

    p = sub.add_parser("advice", help="生成防治建议")
    p.add_argument("disease")
    p.add_argument("--confidence", type=float, default=None)
    p.add_argument("--crop", default=None)
    p.set_defaults(func=cmd_advice)

    p = sub.add_parser("serve", help="启动 FastAPI 服务")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true")
    p.set_defaults(func=cmd_serve)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
