"""Dataset provenance (origin) tiers for TTA validity and report banners."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

# provenance → tta_valid
ORIGIN_TTA_VALID: dict[str, bool] = {
    "dev_fixture": False,
    "public_dataset": False,
    "synthetic_generated": False,
    "field_collected": True,
}

PROVENANCE_ORIGINS = tuple(ORIGIN_TTA_VALID)

# Report banner class by origin presence (most severe wins)
BANNER_PRIORITY = ("dev_fixture", "public_dataset", "synthetic_generated")


def normalize_origin(value: str | None) -> str | None:
    if value is None:
        return None
    return str(value).strip() or None


def origin_distribution(records: Iterable[Any]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        origin = normalize_origin(getattr(record, "origin", None) if not isinstance(record, dict) else record.get("origin"))
        counts[origin or "unknown"] += 1
    return dict(sorted(counts.items()))


def tta_valid_for_origins(origins: Iterable[str | None]) -> bool:
    """TTA-valid only when every known provenance is field_collected (and non-empty)."""
    values = [normalize_origin(o) for o in origins]
    if not values or any(v is None for v in values):
        return False
    return all(ORIGIN_TTA_VALID.get(v, False) for v in values)


def tta_valid_for_records(records: list[Any]) -> bool:
    if not records:
        return False
    return tta_valid_for_origins(getattr(r, "origin", None) for r in records)


def most_severe_banner_origin(records: list[Any]) -> str | None:
    present = {
        normalize_origin(getattr(r, "origin", None))
        for r in records
    }
    for key in BANNER_PRIORITY:
        if key in present:
            return key
    return None


def warn_origin_dataset(dataset_name: str, records: list[Any]) -> None:
    banner_key = most_severe_banner_origin(records)
    if banner_key == "dev_fixture":
        print(
            "\n"
            + "!" * 72
            + f"\n!!! DEV FIXTURE DATASET LOADED: {dataset_name} - NOT VALID FOR TTA SUBMISSION !!!\n"
            + "!" * 72
        )
    elif banner_key in ("public_dataset", "synthetic_generated"):
        print(
            "\n"
            + "=" * 72
            + f"\n=== NON-FIELD DATA ({banner_key}): {dataset_name} - tta_valid=false ===\n"
            + "=" * 72
        )


def html_banner_for_results(result_payloads: list[dict[str, Any] | None]) -> str:
    """Pick the most severe banner across eval JSON payloads."""
    origins: set[str] = set()
    for payload in result_payloads:
        if not payload:
            continue
        dist = payload.get("origin_distribution") or {}
        origins.update(str(k) for k, n in dist.items() if n and k != "unknown")
        if payload.get("tta_valid") is False and "dev_fixture" in dist:
            origins.add("dev_fixture")
    for key in BANNER_PRIORITY:
        if key in origins:
            if key == "dev_fixture":
                return '<div class="dev-fixture">DEV FIXTURE - NOT VALID FOR TTA SUBMISSION</div>'
            return (
                f'<div class="non-field">{key.upper().replace("_", " ")} '
                f"- NOT VALID FOR TTA SUBMISSION (tta_valid=false)</div>"
            )
    if any(p and p.get("tta_valid") is False for p in result_payloads):
        return '<div class="non-field">NON-FIELD DATA - NOT VALID FOR TTA SUBMISSION</div>'
    return ""
