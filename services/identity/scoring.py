"""Pure deterministic identity-consistency rules."""

from __future__ import annotations

from typing import Any


def score_identity(
    resume_handles: dict[str, str],
    supplied_handles: dict[str, str],
    sources: list[dict[str, Any]],
    weights: dict[str, float],
) -> tuple[int, list[dict[str, Any]]]:
    """Score explicit handle agreement and successful public-source confirmation.

    The function has no clock, random state, or network dependency.
    """
    discrepancies: list[dict[str, Any]] = []
    available = set(resume_handles) | set(supplied_handles)
    if not available:
        return 0, [{"type": "no_profile_handle", "message": "No supported profile handle found."}]

    total = 0.0
    earned = 0.0
    results = {str(source["source"]): source for source in sources}
    for source in sorted(available):
        weight = float(weights.get(source, 1.0))
        total += weight
        resume_value = resume_handles.get(source)
        supplied_value = supplied_handles.get(source)
        if resume_value and supplied_value and resume_value.casefold() != supplied_value.casefold():
            discrepancies.append(
                {
                    "type": "handle_mismatch",
                    "source": source,
                    "resume": resume_value,
                    "submitted": supplied_value,
                }
            )
            continue
        expected = supplied_value or resume_value
        result = results.get(source, {})
        if result.get("status") == "ok":
            earned += weight
        elif result.get("status") == "not_found":
            discrepancies.append(
                {"type": "profile_not_found", "source": source, "handle": expected}
            )
        elif result.get("status") not in {"unsupported", "not_provided"}:
            discrepancies.append(
                {
                    "type": "source_unavailable",
                    "source": source,
                    "status": result.get("status", "missing"),
                }
            )
    return int(round(100 * earned / total)) if total else 0, discrepancies
