"""FastAPI 应用入口。"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.config import settings
from backend.app.core.logging import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)

app = FastAPI(
    title=settings.app_name,
    description="用于病害检测与农业知识服务的开发中 API。",
    version=settings.version,
)

# 开发期默认允许本地前端来源，上线前需收紧。
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["系统"])
def root() -> dict[str, str]:
    logger.info("GET / 被访问")
    return {
        "name": settings.app_name,
        "version": settings.version,
        "status": "scaffold",
        "message": "基础服务已启动；模型推理与知识检索功能尚待实现。",
    }


@app.get("/health", tags=["系统"])
def health() -> dict[str, str]:
    return {"status": "ok", "version": settings.version}
