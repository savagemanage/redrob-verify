"""Preflight must refuse silent eval against missing or wrong backends."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from fastapi import FastAPI
from uvicorn import Config, Server
import threading
import time

from harness.preflight import check_endpoint, run_preflight


@pytest.fixture
def tmp_cfg(tmp_path: Path) -> Path:
    cfg = {
        "data_root": str(tmp_path / "data"),
        "results_root": str(tmp_path / "results"),
        "endpoints": {
            "ocr": "http://127.0.0.1:17901",
            "face": "http://127.0.0.1:17902",
            "forgery": "http://127.0.0.1:17903",
            "identity": "http://127.0.0.1:17904",
        },
        "targets": {
            "tc1_cer_max": 0.21,
            "tc2_tpr_min": 0.88,
            "tc3_f1_min": 0.79,
            "tc4_sensitivity_min": 0.84,
            "tc5_accuracy_min": 0.8,
            "tc6_seconds_max": 60.0,
        },
    }
    (tmp_path / "results").mkdir()
    path = tmp_path / "config.yaml"
    path.write_text(yaml.dump(cfg), encoding="utf-8")
    return path


def test_preflight_exits_when_services_down(tmp_cfg: Path) -> None:
    with pytest.raises(SystemExit) as exc:
        run_preflight(yaml.safe_load(tmp_cfg.read_text(encoding="utf-8")))
    assert exc.value.code == 1


def _serve_meta(port: int, service: str) -> tuple[Server, threading.Thread]:
    app = FastAPI()

    @app.get("/v1/meta")
    def meta():
        return {
            "service": service,
            "version": "0.0.0",
            "backend": "test",
            "model_sha256": "none",
            "git_commit": "test",
            "started_at": "now",
        }

    config = Config(app, host="127.0.0.1", port=port, log_level="error")
    server = Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(50):
        try:
            check_endpoint("probe", f"http://127.0.0.1:{port}", service)
            break
        except RuntimeError:
            time.sleep(0.05)
    return server, thread


def test_preflight_exits_on_wrong_service(tmp_cfg: Path) -> None:
    # Bind face-looking port but advertise OCR — wrong identity for that endpoint.
    server, _ = _serve_meta(17902, "ocr")
    try:
        cfg = yaml.safe_load(tmp_cfg.read_text(encoding="utf-8"))
        # Only check the wrong one via check_endpoint contract
        with pytest.raises(RuntimeError, match="expected service"):
            check_endpoint("face", cfg["endpoints"]["face"], "face")
        # Full preflight also fails (all others down + wrong face)
        with pytest.raises(SystemExit) as exc:
            run_preflight(cfg)
        assert exc.value.code == 1
    finally:
        server.should_exit = True
