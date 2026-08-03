"""Fatal eval-response checks. Do not silently demote transport/schema failures to scores."""

from __future__ import annotations

from typing import Any


class EvalTransportError(RuntimeError):
    """HTTP or schema failure that must abort metric computation."""


def require_fields(resp: dict[str, Any], fields: list[str], *, context: str) -> None:
    missing = [f for f in fields if f not in resp]
    if missing:
        raise EvalTransportError(f"{context}: response missing fields {missing}")


def require_number(resp: dict[str, Any], field: str, *, context: str) -> float:
    require_fields(resp, [field], context=context)
    value = resp[field]
    if value is None:
        raise EvalTransportError(f"{context}: field {field!r} is null (not allowed here)")
    try:
        return float(value)
    except (TypeError, ValueError) as e:
        raise EvalTransportError(f"{context}: field {field!r} is not numeric: {value!r}") from e
