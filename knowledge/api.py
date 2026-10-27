"""FastAPI 接口层。

启动：
    python -m uvicorn api:app --host 127.0.0.1 --port 8000 --reload

接口一览：
    GET  /kb/health              服务与配置状态
    GET  /kb/stats               两个向量库的条目统计
    POST /kb/search/text         文本检索（可选按病害类别过滤）
    POST /kb/search/image        以图搜病：上传图片 → 候选病害 → 相关知识
    POST /kb/search/disease      检测结果驱动检索
    POST /kb/advice              完整闭环：检测结果 → 检索 → 生成防治建议
    POST /kb/reindex             重建索引（开发调试用）
    GET  /kb/images/{path}       访问图像库中的参考图

与检测模块的接口契约：
    改进 YOLOv8 推理服务返回 detection 字段（disease_name / confidence / bbox），
    本模块据此完成知识检索与建议生成，检测与知识服务之间只依赖这个结构。
"""
from __future__ import annotations

import io
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

import class_map
import config
import generator
import ingest
from retriever import get_retriever

config.ensure_dirs()

app = FastAPI(
    title="多模态 RAG 农艺知识库",
    description="面向农作物病害检测系统的图文双库检索与防治建议生成服务",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/kb/images", StaticFiles(directory=str(config.IMAGE_DIR)), name="images")

DETECTED_IMAGE_DIR = config.DATA_DIR / "detected"
DETECTED_IMAGE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================ 模型
class TextSearchRequest(BaseModel):
    query: str = Field(..., description="查询文本")
    disease_name: str | None = Field(None, description="病害类别，作为硬过滤条件")
    crop_type: str | None = None
    doc_types: list[str] | None = Field(None, description="symptom/cause/prevention/pesticide/management")
    top_k: int = 6


class DiseaseSearchRequest(BaseModel):
    disease_name: str
    crop_type: str | None = None
    image_id: str | None = Field(None, description="检测原图文件名，用于按相似度排序参考图")
    query: str | None = None
    top_k_text: int = 6
    top_k_image: int = 5


class DetectionItem(BaseModel):
    disease_name: str
    confidence: float = 0.0
    crop_type: str | None = None
    bbox: list[float] | None = Field(None, description="[x1, y1, x2, y2]")


class AdviceRequest(BaseModel):
    detection: DetectionItem | None = None
    disease_name: str | None = None
    crop_type: str | None = None
    confidence: float | None = None
    image_id: str | None = None
    top_k_text: int = 6
    top_k_image: int = 5


class ReindexRequest(BaseModel):
    reset: bool = True
    images: bool = True
    texts: bool = True


# ============================================================ 工具
def _load_detected_image(image_id: str | None):
    if not image_id:
        return None
    path = Path(image_id)
    if not path.is_absolute():
        path = DETECTED_IMAGE_DIR / image_id
    if not path.exists():
        return None
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


async def _read_upload(file: UploadFile):
    raw = await file.read()
    try:
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"无法解析上传的图片：{exc}") from exc


# ============================================================ 接口
@app.get("/kb/health", summary="服务状态")
def health():
    st = ingest.stats()
    return {
        "status": "ok",
        "config": config.describe(),
        "collections": {
            config.COLL_TEXT: st.get(config.COLL_TEXT, {}).get("count", 0),
            config.COLL_IMAGE: st.get(config.COLL_IMAGE, {}).get("count", 0),
        },
        "disease_classes": class_map.all_names(),
    }


@app.get("/kb/stats", summary="知识库统计")
def stats():
    return ingest.stats()


@app.post("/kb/search/text", summary="文本检索")
def search_text(req: TextSearchRequest):
    hits = get_retriever().retrieve_text(
        req.query,
        disease_name=req.disease_name,
        crop_type=req.crop_type,
        doc_types=req.doc_types,
        top_k=req.top_k,
    )
    return {"mode": "text", "query": req.query, "count": len(hits), "hits": hits}


@app.post("/kb/search/disease", summary="检测结果驱动检索")
def search_disease(req: DiseaseSearchRequest):
    image = _load_detected_image(req.image_id)
    return get_retriever().search_by_disease(
        req.disease_name,
        crop_type=req.crop_type,
        image=image,
        query=req.query,
        top_k_text=req.top_k_text,
        top_k_image=req.top_k_image,
    )


@app.post("/kb/search/image", summary="以图搜病")
async def search_image(file: UploadFile = File(...)):
    image = await _read_upload(file)
    result = get_retriever().search_by_image(image)
    result["count"] = len(result.get("text_hits", []))
    return result


@app.post("/kb/advice", summary="生成防治建议（完整闭环）")
def advice(req: AdviceRequest):
    disease_name = req.disease_name or (req.detection.disease_name if req.detection else None)
    if not disease_name:
        raise HTTPException(status_code=422, detail="必须提供 disease_name 或 detection.disease_name")

    confidence = req.confidence
    crop_type = req.crop_type
    if req.detection:
        confidence = req.detection.confidence if confidence is None else confidence
        crop_type = crop_type or req.detection.crop_type

    retriever = get_retriever()
    search = retriever.search_by_disease(
        disease_name,
        crop_type=crop_type,
        image=_load_detected_image(req.image_id),
        top_k_text=req.top_k_text,
        top_k_image=req.top_k_image,
    )
    result = generator.generate_advice(
        search["disease_name"],
        confidence=confidence,
        chunks=search["text_hits"],
        images=search["image_hits"],
        crop_type=search.get("crop_type") or crop_type,
    )
    result["search_query"] = search.get("query_used")
    result["text_hit_count"] = len(search["text_hits"])
    result["image_hit_count"] = len(search["image_hits"])
    return result


@app.post("/kb/debug/pipeline", summary="查看完整检索链路（调参用）")
def debug_pipeline(req: DiseaseSearchRequest):
    retriever = get_retriever()
    search = retriever.search_by_disease(
        req.disease_name,
        crop_type=req.crop_type,
        image=_load_detected_image(req.image_id),
        query=req.query,
    )
    return {
        "config": config.describe(),
        "disease": req.disease_name,
        "query_used": search.get("query_used"),
        "text_hits": search["text_hits"],
        "image_hits": search["image_hits"],
        "prompt": generator.build_user_prompt(
            search["disease_name"], None, search["text_hits"], search["image_hits"]
        ),
    }


@app.post("/kb/reindex", summary="重建索引")
def reindex(req: ReindexRequest):
    out = {}
    if req.texts:
        out["text"] = ingest.build_text_index(reset=req.reset)
    if req.images:
        out["image"] = ingest.build_image_index(reset=req.reset)
    return out


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
