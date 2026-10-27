"""系统接口的自动化测试。"""

from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_root() -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "作物病害检测与知识服务系统"
    assert data["status"] == "scaffold"
    assert data["version"]


def test_health() -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["version"]
