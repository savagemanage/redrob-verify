"""Unit tests for DS-3 face pair builder and freeze trial rules."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from harness.freeze import check_ds3_trial_rules
from tools.build_face_pairs import build_face_pairs, validate_counts


def _make_identities(root: Path, n: int) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        person = root / f"p{i:04d}"
        person.mkdir()
        Image.new("RGB", (32, 32), (i % 255, 40, 80)).save(person / "doc.png")
        Image.new("RGB", (32, 32), (80, i % 255, 40)).save(person / "selfie.png")
    return root


def test_validate_counts_field_band() -> None:
    validate_counts(200, 200, origin="field_collected")
    with pytest.raises(ValueError):
        validate_counts(100, 200, origin="field_collected")
    with pytest.raises(ValueError):
        validate_counts(200, 100, origin="field_collected")


def test_validate_counts_smoke_relaxed() -> None:
    validate_counts(10, 10, origin="dev_fixture")


def test_build_face_pairs_smoke_default_origin(tmp_path: Path) -> None:
    src = _make_identities(tmp_path / "ids", 10)
    out = tmp_path / "3_face"
    summary = build_face_pairs(src, out, seed=1)  # default origin=dev_fixture
    assert summary["origin"] == "dev_fixture"
    assert summary["tta_valid"] is False
    assert summary["same_pairs"] == 10
    assert summary["diff_pairs"] == 10
    rows = [
        json.loads(line)
        for line in (out / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert all(r["origin"] == "dev_fixture" for r in rows)


def test_build_face_pairs_field_scale(tmp_path: Path) -> None:
    src = _make_identities(tmp_path / "ids", 160)
    out = tmp_path / "3_face"
    summary = build_face_pairs(src, out, origin="field_collected", seed=1)
    assert summary["same_pairs"] == 160
    assert 150 <= summary["diff_pairs"] <= 250
    assert 300 <= summary["total"] <= 500
    assert summary["tta_valid"] is True
    rows = [
        json.loads(line)
        for line in (out / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert all(r["origin"] == "field_collected" for r in rows)


def test_freeze_rejects_midv_as_trial(tmp_path: Path) -> None:
    ds3 = tmp_path / "3_face"
    ds3.mkdir()
    (ds3 / "manifest.jsonl").write_text(
        json.dumps(
            {
                "id": "midv_1",
                "img_a": "a.png",
                "img_b": "b.png",
                "same": True,
                "origin": "public_dataset",
                "pair_warning": "under_variation:midv_cross_capture",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    cfg = {
        "data_root": str(tmp_path),
        "results_root": str(tmp_path / "results"),
        "thresholds": {"face": 0.5},
        "seed": 42,
    }
    (tmp_path / "results").mkdir()
    errors = check_ds3_trial_rules(cfg, {"threshold": {"face": 0.5}})
    joined = " | ".join(errors)
    assert "field_collected" in joined
    assert "under-variation" in joined or "변이" in joined or "threshold" in joined.lower()
