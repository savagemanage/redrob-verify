"""Preflight: verify each configured endpoint is the expected module service."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import httpx

from harness.config_util import load_config, resolve_results_root

EXPECTED_SERVICES = {
    "ocr": "ocr",
    "face": "face",
    "forgery": "forgery",
    "identity": "identity",
}


def check_endpoint(
    name: str,
    base_url: str,
    expected_service: str,
    *,
    timeout: float = 5.0,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/v1/meta"
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(url)
    except httpx.HTTPError as e:
        raise RuntimeError(f"{name}: no response from {url}: {e}") from e

    if resp.status_code == 404:
        raise RuntimeError(f"{name}: {url} returned 404 (wrong or legacy process)")
    if resp.status_code != 200:
        raise RuntimeError(f"{name}: {url} returned HTTP {resp.status_code}")

    try:
        meta = resp.json()
    except ValueError as e:
        raise RuntimeError(f"{name}: {url} returned non-JSON") from e

    if not isinstance(meta, dict):
        raise RuntimeError(f"{name}: meta is not an object")
    got = meta.get("service")
    if got != expected_service:
        raise RuntimeError(
            f"{name}: expected service={expected_service!r} at {url}, got {got!r}"
        )
    return {"endpoint": name, "url": url, "meta": meta}


def run_preflight(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = cfg or load_config()
    endpoints = cfg.get("endpoints") or {}
    results: list[dict[str, Any]] = []
    errors: list[str] = []
    for name, expected in EXPECTED_SERVICES.items():
        base = endpoints.get(name)
        if not base:
            errors.append(f"{name}: missing endpoints.{name} in config")
            continue
        try:
            results.append(check_endpoint(name, str(base), expected))
            print(f"preflight OK  {name:8s} -> {base}  service={expected}")
        except RuntimeError as e:
            errors.append(str(e))
            print(f"preflight FAIL {name}: {e}", file=sys.stderr)

    payload = {
        "ok": len(errors) == 0,
        "services": results,
        "errors": errors,
    }
    results_root = resolve_results_root(cfg)
    out = results_root / "preflight.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    if errors:
        print(f"PREFLIGHT FAILED ({len(errors)} error(s))", file=sys.stderr)
        raise SystemExit(1)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify module endpoints via /v1/meta")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    run_preflight(load_config(args.config))


if __name__ == "__main__":
    main()
