"""知识库构建：文本库 + 图像库。

文本入库流程
    读取 data/docs/*.md → 解析 YAML 元数据 → 按标题分节 → 超长节滑窗切分
    → bge-m3 编码 → 写入 Chroma 集合 kb_text

图像入库流程
    遍历 data/image_library/<类别目录>/*.jpg → 目录名解析类别元数据
    → 按类别均匀抽样至上限 → CLIP 编码 → 写入 Chroma 集合 kb_image

设计要点
    1. 两个库共用 crop_type / disease_name / disease_id 三个元数据字段，
       检索层才能用同一套过滤条件做"类别硬过滤"。
    2. chunk_id 由文档相对路径 + 序号确定性生成，重复入库为幂等覆盖；
       同时清理同一文档已失效的旧块，避免改了文档后检索到残留内容。
    3. 图像库抽样采用等距抽样而非随机抽样，保证同一类别内不同拍摄条件
       （正拍、侧拍、不同叶龄）的样本都能进库，避免同类图像高度冗余。
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import yaml

import class_map
import config

IMAGE_SUFFIX = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ============================================================ Chroma 客户端
def get_client():
    import chromadb

    config.ensure_dirs()
    return chromadb.PersistentClient(path=str(config.STORE_DIR))


def get_text_collection():
    client = get_client()
    return client.get_or_create_collection(
        name=config.COLL_TEXT, metadata={"hnsw:space": config.DISTANCE}
    )


def get_image_collection():
    client = get_client()
    return client.get_or_create_collection(
        name=config.COLL_IMAGE, metadata={"hnsw:space": config.DISTANCE}
    )


def reset_collection(name: str) -> None:
    client = get_client()
    try:
        client.delete_collection(name)
    except Exception:
        pass


# ============================================================ 文本解析与分块
def parse_front_matter(raw: str) -> tuple[dict[str, Any], str]:
    """解析 Markdown 头部的 YAML 元数据块。"""
    if not raw.lstrip().startswith("---"):
        return {}, raw
    text = raw.lstrip()
    end = text.find("\n---", 3)
    if end == -1:
        return {}, raw
    meta = yaml.safe_load(text[3:end]) or {}
    body = text[end + 4 :]
    return (meta if isinstance(meta, dict) else {}), body


def _window(text: str, size: int, overlap: int) -> list[str]:
    """对超长文本做带重叠的滑窗切分，优先在句末断开。"""
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []
    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = max(text.rfind("。", start + size // 2, end), text.rfind("\n", start + size // 2, end))
            if cut > start:
                end = cut + 1
        pieces.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [p for p in pieces if p]


def chunk_markdown(body: str) -> list[tuple[str, str, str]]:
    """按标题分节，再对超长节滑窗切分。

    返回 [(heading, parent_heading, chunk_text), ...]。
    chunk_text 保留标题前缀，使每个块自身携带语义上下文，显著改善短块的检索命中率；
    parent_heading 用于回退查询章节级文档类型（### 小节的类型继承自其 ## 父节）。
    """
    sections: list[tuple[str, str, list[str]]] = []
    current_heading = ""
    current_parent = ""
    current_lines: list[str] = []

    for line in body.splitlines():
        m = re.match(r"^(#{1,4})\s+(.*)$", line.strip())
        if m:
            if current_lines:
                sections.append((current_heading, current_parent, current_lines))
            level = len(m.group(1))
            title = m.group(2).strip()
            if level <= 2:
                current_parent = title
            current_heading = title
            current_lines = []
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_heading, current_parent, current_lines))

    chunks: list[tuple[str, str, str]] = []
    for heading, parent, lines in sections:
        text = "\n".join(lines).strip()
        if not text:
            continue
        prefix = f"{heading}\n" if heading else ""
        for piece in _window(text, config.CHUNK_SIZE - len(prefix), config.CHUNK_OVERLAP):
            full = f"{prefix}{piece}".strip()
            if len(full) >= config.CHUNK_MIN:
                chunks.append((heading, parent, full))
    return chunks


# ============================================================ 文本建库
def resolve_section_type(section_types: dict, heading: str, parent: str, default: str) -> str:
    """章节级知识类型解析：优先本节声明，其次继承父节，最后回落到文档级默认值。"""
    return str(section_types.get(heading) or section_types.get(parent) or default)


def collect_text_chunks(verbose: bool = True) -> dict:
    """扫描知识文档并生成待入库的块，不涉及编码与向量库。

    与 build_text_index 拆开的原因：这一步是纯文本处理，可在没有 chromadb、
    没有编码模型的机器上单独运行，用于校验分块质量（见 selftest.py）。
    """
    config.ensure_dirs()
    docs = sorted(
        [p for p in config.DOCS_DIR.rglob("*") if p.suffix.lower() in {".md", ".txt", ".markdown"}]
    )
    docs = [p for p in docs if not p.name.startswith("_")]  # 下划线开头为模板，不入库

    ids: list[str] = []
    texts: list[str] = []
    metas: list[dict] = []
    missing_meta: list[str] = []
    unresolved: list[str] = []

    for path in docs:
        rel = path.relative_to(config.DOCS_DIR).as_posix()
        meta, body = parse_front_matter(path.read_text(encoding="utf-8"))
        raw_name = str(meta.get("disease_name", "")).strip()
        resolved = class_map.resolve_name(raw_name)
        if resolved is None and raw_name:
            # 未登记或别名歧义：仍按原文入库，但必须告警——
            # 检测模型的类别名若与之不一致，类别硬过滤会全部落空。
            unresolved.append(f"{rel} → {raw_name}")
        resolved = resolved or {}
        disease_name = resolved.get("disease_name") or raw_name
        crop_type = resolved.get("crop_type") or str(meta.get("crop_type", "")).strip()
        disease_id = resolved.get("disease_id") or str(meta.get("disease_id", "")).strip()

        if not disease_name:
            missing_meta.append(rel)
            if verbose:
                print(f"[跳过] {rel}：缺少 disease_name 元数据")
            continue

        section_types: dict = meta.get("section_types") or {}
        default_type = str(meta.get("doc_type", "general"))

        for idx, (heading, parent, chunk) in enumerate(chunk_markdown(body)):
            sec_type = resolve_section_type(section_types, heading, parent, default_type)
            ids.append(f"{rel}::{idx}")
            texts.append(chunk)
            metas.append(
                {
                    "doc_path": rel,
                    "doc_title": str(meta.get("title", path.stem)),
                    "doc_type": sec_type,
                    "doc_type_label": config.DOC_TYPE_LABEL.get(sec_type, "综合资料"),
                    "heading": heading,
                    "section_parent": parent,
                    "disease_id": disease_id,
                    "disease_name": disease_name,
                    "crop_type": crop_type,
                    "source": str(meta.get("source", "自建知识库")),
                    "chunk_index": idx,
                }
            )

    return {
        "ids": ids,
        "texts": texts,
        "metadatas": metas,
        "doc_files": len(docs),
        "skipped_missing_meta": missing_meta,
        "unresolved_names": unresolved,
    }


def build_text_index(reset: bool = False, verbose: bool = True) -> dict:
    from embedders import get_text_embedder

    config.ensure_dirs()
    if reset:
        reset_collection(config.COLL_TEXT)
    col = get_text_collection()

    collected = collect_text_chunks(verbose=verbose)
    ids: list[str] = collected["ids"]
    texts: list[str] = collected["texts"]
    metas: list[dict] = collected["metadatas"]

    if ids:
        embedder = get_text_embedder()
        vectors = embedder.encode(texts)
        for i in range(0, len(ids), config.UPSERT_BATCH):
            sl = slice(i, i + config.UPSERT_BATCH)
            col.upsert(
                ids=ids[sl],
                embeddings=vectors[sl].tolist(),
                documents=texts[sl],
                metadatas=metas[sl],
            )

    # 清理失效块：覆盖"文档被改写"与"文档被整篇删除"两种情况，
    # 因此从库内反查既有 id，而不是只看本次扫描到的文档。
    stale: list[str] = []
    if col.count():
        existing = col.get(include=[])
        new_id_set = set(ids)
        stale = [i for i in existing["ids"] if i not in new_id_set]
        if stale:
            col.delete(ids=stale)

    stats = {
        "doc_files": collected["doc_files"],
        "chunks": len(ids),
        "skipped_missing_meta": collected["skipped_missing_meta"],
        "unresolved_names": collected["unresolved_names"],
        "stale_removed": len(stale),
        "collection_count": col.count(),
        "dim": get_text_embedder().dim if ids else None,
        "model": config.TEXT_MODEL,
    }
    if verbose:
        print(
            f"[文本库] 文档 {stats['doc_files']} 个 → 切块 {stats['chunks']} 条，"
            f"清理旧块 {stats['stale_removed']} 条，库内共 {stats['collection_count']} 条"
        )
        for item in stats["unresolved_names"]:
            print(f"[警告] 病名未在 class_map.py 登记或存在歧义：{item}")
    return stats


# ============================================================ 图像建库
def _sample_evenly(files: list[str], limit: int) -> list[str]:
    if len(files) <= limit:
        return files
    step = len(files) / limit
    return [files[int(i * step)] for i in range(limit)]


def build_image_index(
    reset: bool = False,
    limit_per_class: int = config.MAX_IMAGES_PER_CLASS,
    verbose: bool = True,
) -> dict:
    from PIL import Image

    from embedders import get_image_embedder

    config.ensure_dirs()
    if reset:
        reset_collection(config.COLL_IMAGE)
    col = get_image_collection()

    if not config.IMAGE_DIR.exists():
        raise FileNotFoundError(f"图像库目录不存在：{config.IMAGE_DIR}")

    ids: list[str] = []
    paths: list[str] = []
    metas: list[dict] = []
    unknown_dirs: list[str] = []

    for cls_dir in sorted([d for d in config.IMAGE_DIR.iterdir() if d.is_dir()]):
        meta = class_map.resolve_dir(cls_dir.name)
        if meta is None:
            unknown_dirs.append(cls_dir.name)
            continue
        # 排序保证抽样确定性；等距抽样覆盖不同拍摄条件
        files = sorted(
            [str(p) for p in cls_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIX]
        )
        picked = _sample_evenly(files, limit_per_class)
        for k, fp in enumerate(picked):
            rel = Path(fp).relative_to(config.IMAGE_DIR).as_posix()
            ids.append(f"img::{rel}")
            paths.append(fp)
            metas.append(
                {
                    "image_path": rel,
                    "image_name": Path(fp).name,
                    "disease_id": meta["disease_id"],
                    "disease_name": meta["disease_name"],
                    "crop_type": meta["crop_type"],
                    "class_dir": cls_dir.name,
                    "source": "PlantVillage",
                    "in_class_index": k,
                }
            )

    valid_ids: list[str] = []
    valid_metas: list[dict] = []
    valid_paths: list[str] = []
    images: list = []
    skipped = 0
    for i, fp in enumerate(paths):
        try:
            images.append(Image.open(fp).convert("RGB"))
            valid_ids.append(ids[i])
            valid_metas.append(metas[i])
            valid_paths.append(fp)
        except Exception:
            skipped += 1

    if valid_ids:
        embedder = get_image_embedder()
        vectors = embedder.encode_images(images)
        for i in range(0, len(valid_ids), config.UPSERT_BATCH):
            sl = slice(i, i + config.UPSERT_BATCH)
            col.upsert(
                ids=valid_ids[sl],
                embeddings=vectors[sl].tolist(),
                documents=[m["disease_name"] for m in valid_metas[sl]],
                metadatas=valid_metas[sl],
            )

    stats = {
        "classes": len({m["disease_name"] for m in valid_metas}),
        "images": len(valid_ids),
        "skipped": skipped,
        "unknown_dirs": unknown_dirs,
        "per_class_limit": limit_per_class,
        "collection_count": col.count(),
        "dim": get_image_embedder().dim if valid_ids else None,
        "model": config.IMAGE_MODEL,
    }
    if verbose:
        print(
            f"[图像库] 类别 {stats['classes']} 个 → 入库 {stats['images']} 张"
            f"（跳过损坏 {skipped} 张），库内共 {stats['collection_count']} 张"
        )
        if unknown_dirs:
            print(f"[警告] 以下目录未在 class_map.py 中登记，已跳过：{unknown_dirs}")
    return stats


# ============================================================ 库状态
def stats() -> dict:
    config.ensure_dirs()
    client = get_client()
    names = [c.name for c in client.list_collections()]
    out: dict[str, Any] = {"collections": names, "store_dir": str(config.STORE_DIR)}
    for name in (config.COLL_TEXT, config.COLL_IMAGE):
        if name in names:
            col = client.get_collection(name)
            data = col.get(include=["metadatas"])
            by_disease: dict[str, int] = {}
            for m in data["metadatas"] or []:
                key = m.get("disease_name", "未知")
                by_disease[key] = by_disease.get(key, 0) + 1
            out[name] = {"count": col.count(), "by_disease": by_disease}
        else:
            out[name] = {"count": 0, "by_disease": {}}
    return out


def build_all(reset: bool = True) -> dict:
    return {"text": build_text_index(reset=reset), "image": build_image_index(reset=reset)}


if __name__ == "__main__":
    import json

    print(json.dumps(build_all(reset=True), ensure_ascii=False, indent=2))
