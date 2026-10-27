"""应用配置：集中从环境变量与 .env 文件读取。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# 仓库根目录：backend/app/config.py 向上三级
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 加载 .env（若存在），不覆盖已有环境变量
load_dotenv(PROJECT_ROOT / ".env")


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """应用配置项（frozen，避免运行期被意外修改）。"""

    app_name: str = os.getenv("APP_NAME", "作物病害检测与知识服务系统")
    version: str = os.getenv("APP_VERSION", "0.1.0")
    debug: bool = _env_bool("APP_DEBUG", True)

    cors_origins: tuple[str, ...] = tuple(
        origin.strip()
        for origin in os.getenv(
            "CORS_ORIGINS",
            "http://localhost:5173,http://127.0.0.1:5173",
        ).split(",")
        if origin.strip()
    )

    # 数据与模型产物目录（已通过 .gitignore 排除）
    data_dir: Path = Path(os.getenv("DATA_DIR", str(PROJECT_ROOT / "data")))
    model_dir: Path = Path(os.getenv("MODEL_DIR", str(PROJECT_ROOT / "weights")))


settings = Settings()
