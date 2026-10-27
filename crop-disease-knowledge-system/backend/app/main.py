from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="作物病害检测与知识服务系统",
    description="用于病害检测与农业知识服务的开发中 API。",
    version="0.1.0",
)

# Local development defaults. Restrict allowed origins before production deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["系统"])
def root() -> dict[str, str]:
    return {
        "name": "作物病害检测与知识服务系统",
        "version": "0.1.0",
        "status": "scaffold",
        "message": "基础服务已启动；模型推理与知识检索功能尚待实现。",
    }


@app.get("/health", tags=["系统"])
def health() -> dict[str, str]:
    return {"status": "ok"}
