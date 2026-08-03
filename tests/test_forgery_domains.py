"""Forgery dual-domain schema + gate helpers."""

from __future__ import annotations

from harness.schema import ForgeryRecord


def test_forgery_eval_domain_field() -> None:
    row = ForgeryRecord.model_validate(
        {
            "id": "fg_xd_1",
            "path": "images/a.png",
            "label": 1,
            "origin": "public_dataset",
            "fabrication": "script",
            "tamper": "copy_move",
            "mask_path": "masks/a.png",
            "eval_domain": "cross_domain",
            "generator": "fmidv",
        }
    )
    assert row.eval_domain == "cross_domain"
    assert row.generator == "fmidv"


def test_forgery_defaults_without_eval_domain() -> None:
    row = ForgeryRecord.model_validate(
        {
            "id": "auth_1",
            "path": "authentic/a.png",
            "label": 0,
            "origin": "dev_fixture",
            "fabrication": "authentic",
        }
    )
    assert row.eval_domain is None
