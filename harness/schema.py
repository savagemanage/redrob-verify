"""Manifest schemas (Pydantic v2) and JSONL loaders with path + count checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from harness.origin import (
    PROVENANCE_ORIGINS,
    origin_distribution,
    tta_valid_for_records,
    warn_origin_dataset,
)

ProvenanceOrigin = Literal[
    "dev_fixture",
    "public_dataset",
    "synthetic_generated",
    "field_collected",
]


SCRIPT_BY_DOC_PREFIX: dict[str, str] = {
    "alb": "latin",
    "aze": "latin",
    "esp": "latin",
    "est": "latin",
    "fin": "latin",
    "lva": "latin",
    "svk": "latin",
    "grc": "greek",
    "rus": "cyrillic",
    "srb": "cyrillic",
}

# MIDV-2020 doc_type prefix IS the country code in the dataset naming
# (alb_id, rus_internalpassport, grc_passport, …).
COUNTRY_BY_DOC_PREFIX: dict[str, str] = {
    "alb": "alb",
    "aze": "aze",
    "esp": "esp",
    "est": "est",
    "fin": "fin",
    "lva": "lva",
    "svk": "svk",
    "grc": "grc",
    "rus": "rus",
    "srb": "srb",
}


def country_for_doc_type(doc_type: str) -> str | None:
    """Map MIDV doc code (alb_id → alb) to ISO-ish country key used in MIDV-2020."""
    prefix = str(doc_type or "").split("_", 1)[0].lower()
    return COUNTRY_BY_DOC_PREFIX.get(prefix)


def script_for_doc_type(doc_type: str) -> str | None:
    """Map MIDV doc code prefix (alb_id → alb) to script family."""
    prefix = str(doc_type or "").split("_", 1)[0].lower()
    return SCRIPT_BY_DOC_PREFIX.get(prefix)


class OcrRecord(BaseModel):
    """1_ocr — TC1"""

    id: str
    path: str
    doc_type: str
    gt_text: str
    origin: ProvenanceOrigin | str | None = None
    gt_fields: dict[str, Any] = Field(default_factory=dict)
    script: Literal["latin", "greek", "cyrillic"] | str | None = None
    capture: str | None = None
    source: str | None = None

    @model_validator(mode="after")
    def _default_script(self) -> OcrRecord:
        if self.script is None:
            inferred = script_for_doc_type(self.doc_type)
            if inferred is not None:
                object.__setattr__(self, "script", inferred)
        return self


class ForgeryRecord(BaseModel):
    """2_forgery — TC2, TC3

    ``origin`` is dataset provenance (TTA validity).
    ``fabrication`` is how the sample was produced (authentic / script / manual).
    Legacy manifests that put authentic|script|manual in ``origin`` are accepted
    and rewritten into fabrication + origin=dev_fixture|synthetic_generated.
    """

    id: str
    path: str
    label: Literal[0, 1]
    origin: str
    fabrication: Literal["authentic", "script", "manual"] | None = None
    source: str | None = None
    tamper: str | None = None
    mask_path: str | None = None
    profile: Literal["train", "test"] | None = None
    # TC2/TC3 split: in_domain = our gen_forgery; cross_domain = FMIDV (3rd party)
    eval_domain: Literal["in_domain", "cross_domain"] | None = None
    generator: str | None = None  # e.g. gen_forgery | fmidv | midv_authentic

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_origin(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        origin = data.get("origin")
        fabrication = data.get("fabrication")
        if fabrication is None and origin in ("authentic", "script", "manual"):
            data["fabrication"] = origin
            # Legacy harness fixtures used these as fabrication tags.
            data["origin"] = "dev_fixture" if origin == "authentic" else "synthetic_generated"
        elif fabrication is None and origin in PROVENANCE_ORIGINS:
            # Provenance-only rows: infer fabrication from label when possible.
            if data.get("label") == 0:
                data["fabrication"] = "authentic"
            elif data.get("label") == 1:
                data["fabrication"] = "script"
        return data

    @model_validator(mode="after")
    def _check_fields(self) -> ForgeryRecord:
        fab = self.fabrication
        if fab is None:
            raise ValueError(f"{self.id}: fabrication is required (authentic|script|manual)")
        if self.label == 0 and fab != "authentic":
            raise ValueError(f"{self.id}: label=0 requires fabrication=authentic")
        if self.label == 1 and fab == "authentic":
            raise ValueError(f"{self.id}: label=1 cannot have fabrication=authentic")
        if self.label == 1 and self.tamper is None:
            raise ValueError(f"{self.id}: label=1 requires tamper")
        if self.label == 1 and self.mask_path is None:
            raise ValueError(f"{self.id}: label=1 requires mask_path")
        return self


class FaceRecord(BaseModel):
    """3_face — TC4, TC5"""

    id: str
    img_a: str
    img_b: str
    same: bool
    origin: ProvenanceOrigin | str | None = None
    identity_id: str | None = None
    capture_a: str | None = None  # e.g. photo|scan|video_frame
    capture_b: str | None = None
    pair_warning: str | None = None


class ResumeRecord(BaseModel):
    """4_resume — TC6"""

    id: str
    path: str
    profiles: dict[str, str] = Field(default_factory=dict)
    origin: ProvenanceOrigin | str | None = None
    gt_discrepancies: list[dict[str, Any]] = Field(default_factory=list)
    anchors: dict[str, Any] = Field(default_factory=dict)


def manifest_has_dev_fixtures(records: list[Any]) -> bool:
    """Return whether a loaded manifest contains development-only fixtures."""
    return any(getattr(record, "origin", None) == "dev_fixture" for record in records)


def warn_dev_fixtures(dataset_name: str) -> None:
    """Backward-compatible alias."""
    print(
        "\n"
        + "!" * 72
        + "\n"
        + f"!!! DEV FIXTURE DATASET LOADED: {dataset_name} - NOT VALID FOR TTA SUBMISSION !!!"
        + "\n"
        + "!" * 72
    )


def _load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"manifest not found: {path}")
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {e}") from e
    return rows


def _missing_paths(dataset_root: Path, rel_paths: list[tuple[str, str]]) -> list[str]:
    broken: list[str] = []
    for record_id, rel in rel_paths:
        full = dataset_root / rel
        if not full.is_file():
            broken.append(f"{record_id}: {rel}")
    return broken


def _raise_count_mismatch(dataset: str, details: list[str]) -> None:
    msg = f"{dataset}: expected_counts mismatch with manifest:\n  " + "\n  ".join(details)
    raise ValueError(msg)


def _relaxed_counts(records: list[Any]) -> bool:
    """Skip strict expected_counts when any non-field provenance is present."""
    return bool(records) and not tta_valid_for_records(records)


def _check_total(dataset: str, actual: int, expected: Any, *, relaxed: bool = False) -> None:
    if expected is None:
        return
    if isinstance(expected, dict):
        exp_total = expected.get("total")
        if exp_total is not None and actual != int(exp_total) and not relaxed:
            _raise_count_mismatch(dataset, [f"total: expected {exp_total}, got {actual}"])
    elif actual != int(expected) and not relaxed:
        _raise_count_mismatch(dataset, [f"total: expected {expected}, got {actual}"])


def load_ocr_manifest(
    dataset_root: Path | str,
    *,
    expected: Any = None,
) -> list[OcrRecord]:
    root = Path(dataset_root)
    raw = _load_jsonl(root / "manifest.jsonl")
    records = [OcrRecord.model_validate(r) for r in raw]
    warn_origin_dataset("1_ocr", records)
    checks = [(r.id, r.path) for r in records]
    broken = _missing_paths(root, checks)
    if broken:
        raise FileNotFoundError(
            "OCR manifest references missing files:\n  " + "\n  ".join(broken)
        )
    _check_total("1_ocr", len(records), expected, relaxed=_relaxed_counts(records))
    return records


def load_forgery_manifest(
    dataset_root: Path | str,
    *,
    expected: Any = None,
) -> list[ForgeryRecord]:
    root = Path(dataset_root)
    raw = _load_jsonl(root / "manifest.jsonl")
    records = [ForgeryRecord.model_validate(r) for r in raw]
    warn_origin_dataset("2_forgery", records)
    checks = [(r.id, r.path) for r in records]
    checks.extend((r.id, r.mask_path) for r in records if r.mask_path is not None)
    broken = _missing_paths(root, checks)
    if broken:
        raise FileNotFoundError(
            "Forgery manifest references missing files:\n  " + "\n  ".join(broken)
        )
    relaxed = _relaxed_counts(records)
    _check_total("2_forgery", len(records), expected, relaxed=relaxed)
    if isinstance(expected, dict) and not relaxed:
        n0 = sum(1 for r in records if r.label == 0)
        n1 = sum(1 for r in records if r.label == 1)
        errs: list[str] = []
        if "label_0" in expected and n0 != int(expected["label_0"]):
            errs.append(f"label_0: expected {expected['label_0']}, got {n0}")
        if "label_1" in expected and n1 != int(expected["label_1"]):
            errs.append(f"label_1: expected {expected['label_1']}, got {n1}")
        if errs:
            _raise_count_mismatch("2_forgery", errs)
    return records


def load_face_manifest(
    dataset_root: Path | str,
    *,
    expected: Any = None,
) -> list[FaceRecord]:
    root = Path(dataset_root)
    raw = _load_jsonl(root / "manifest.jsonl")
    records = [FaceRecord.model_validate(r) for r in raw]
    warn_origin_dataset("3_face", records)
    checks = [(r.id, r.img_a) for r in records] + [(r.id, r.img_b) for r in records]
    broken = _missing_paths(root, checks)
    if broken:
        raise FileNotFoundError(
            "Face manifest references missing files:\n  " + "\n  ".join(broken)
        )
    relaxed = _relaxed_counts(records)
    _check_total("3_face", len(records), expected, relaxed=relaxed)
    if isinstance(expected, dict) and not relaxed:
        n_true = sum(1 for r in records if r.same)
        n_false = sum(1 for r in records if not r.same)
        errs: list[str] = []
        if "same_true" in expected and n_true != int(expected["same_true"]):
            errs.append(f"same_true: expected {expected['same_true']}, got {n_true}")
        if "same_false" in expected and n_false != int(expected["same_false"]):
            errs.append(f"same_false: expected {expected['same_false']}, got {n_false}")
        if errs:
            _raise_count_mismatch("3_face", errs)
    return records


def load_resume_manifest(
    dataset_root: Path | str,
    *,
    expected: Any = None,
) -> list[ResumeRecord]:
    root = Path(dataset_root)
    raw = _load_jsonl(root / "manifest.jsonl")
    records = [ResumeRecord.model_validate(r) for r in raw]
    warn_origin_dataset("4_resume", records)
    checks = [(r.id, r.path) for r in records]
    broken = _missing_paths(root, checks)
    if broken:
        raise FileNotFoundError(
            "Resume manifest references missing files:\n  " + "\n  ".join(broken)
        )
    _check_total("4_resume", len(records), expected, relaxed=_relaxed_counts(records))
    return records


__all__ = [
    "OcrRecord",
    "ForgeryRecord",
    "FaceRecord",
    "ResumeRecord",
    "PROVENANCE_ORIGINS",
    "manifest_has_dev_fixtures",
    "warn_dev_fixtures",
    "origin_distribution",
    "tta_valid_for_records",
    "load_ocr_manifest",
    "load_forgery_manifest",
    "load_face_manifest",
    "load_resume_manifest",
]
