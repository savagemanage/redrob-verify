"""Backend swap: eval/calibrate paths must not hardcode a model; factory selects by config."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from services.face.backends.factory import create_backend
from services.face.backends.sface import SFaceBackend
from services.face.backends.stub import StubEmbeddingBackend
from services.face.pipeline import FacePipeline, cosine_to_unit_interval

REPO = Path(__file__).resolve().parents[1]


def test_cosine_mapping_comment_contract() -> None:
    assert cosine_to_unit_interval(-1.0) == 0.0
    assert cosine_to_unit_interval(0.0) == 0.5
    assert cosine_to_unit_interval(1.0) == 1.0


def test_factory_stub() -> None:
    be = create_backend({"backend": "stub", "stub": {"dim": 64, "seed": 1}}, repo_root=REPO)
    assert isinstance(be, StubEmbeddingBackend)
    assert be.name == "stub"
    img = np.zeros((112, 112, 3), dtype=np.uint8)
    e1 = be.embed(img)
    e2 = be.embed(img)
    assert e1.shape == (64,)
    assert np.allclose(np.linalg.norm(e1), 1.0, atol=1e-6)
    assert np.allclose(e1, e2)


def test_factory_sface_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        create_backend(
            {
                "backend": "sface",
                "sface": {
                    "yunet_path": "models/face/does_not_exist_yunet.onnx",
                    "sface_path": "models/face/does_not_exist_sface.onnx",
                },
            },
            repo_root=REPO,
        )


def test_backend_swap_stub_to_sface_config_only(tmp_path: Path) -> None:
    """Switching stub → sface is config-only; eval_face does not import a backend name."""
    stub = create_backend({"backend": "stub", "detect": {}}, repo_root=REPO)
    pipe_stub = FacePipeline(
        stub,
        quality_cfg={"min_face_side_px": 1, "min_laplacian_var": 0.0},
        detect_cfg={"fullframe_fallback": True},
    )
    img = np.full((64, 64, 3), 120, dtype=np.uint8)
    path_a = tmp_path / "a.png"
    path_b = tmp_path / "b.png"
    import cv2

    cv2.imwrite(str(path_a), img)
    cv2.imwrite(str(path_b), img[:, :, ::-1])

    out = pipe_stub.compare(str(path_a), str(path_b))
    assert out["backend"] == "stub"
    assert "similarity" in out
    assert "latency_ms" in out
    assert "quality" in out
    assert out["similarity"] is None or 0.0 <= float(out["similarity"]) <= 1.0

    with pytest.raises(FileNotFoundError):
        create_backend(
            {
                "backend": "sface",
                "sface": {
                    "yunet_path": str(tmp_path / "y.onnx"),
                    "sface_path": str(tmp_path / "s.onnx"),
                },
            },
            repo_root=REPO,
        )

    assert SFaceBackend.name == "sface"
    assert StubEmbeddingBackend.name == "stub"

    # eval_face.py must not hardcode backend module names
    eval_src = (REPO / "harness" / "eval_face.py").read_text(encoding="utf-8")
    assert "sface" not in eval_src.lower()
    assert "stub" not in eval_src.lower()
    assert "create_backend" not in eval_src


def test_eval_face_consumes_backend_agnostic_keys() -> None:
    required = {"similarity"}
    stub = create_backend({"backend": "stub"}, repo_root=REPO)
    pipe = FacePipeline(
        stub,
        quality_cfg={"min_face_side_px": 1, "min_laplacian_var": 0.0},
        detect_cfg={"fullframe_fallback": True},
    )
    img = np.full((48, 48, 3), 90, dtype=np.uint8)
    out = pipe.compare(img, img)
    assert required <= set(out.keys())
    assert out["backend"] == "stub"
