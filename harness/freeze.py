"""Collect freeze metadata: model/dataset SHA-256, git commit, seed."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harness.config_util import (
    REPO_ROOT,
    load_config,
    resolve_data_root,
    resolve_results_root,
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_tree(root: Path, patterns: tuple[str, ...] = ("**/*",)) -> str:
    """Deterministic hash of all regular files under root (sorted by relative path)."""
    files: list[Path] = []
    for pattern in patterns:
        files.extend(p for p in root.glob(pattern) if p.is_file())
    files = sorted(set(files), key=lambda p: p.relative_to(root).as_posix())
    h = hashlib.sha256()
    for path in files:
        rel = path.relative_to(root).as_posix()
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(sha256_file(path).encode("ascii"))
        h.update(b"\0")
    return h.hexdigest()


def git_commit(repo: Path | None = None) -> str:
    root = repo or REPO_ROOT
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def git_working_tree_dirty(repo: Path | None = None) -> bool:
    """True if git is unavailable, not a repo, or working tree has changes."""
    root = repo or REPO_ROOT
    try:
        # Ensure we are inside a git work tree
        subprocess.check_output(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return bool(status.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return True


def collect_freeze(
    cfg: dict[str, Any] | None = None,
    *,
    model_dir: Path | str | None = None,
) -> dict[str, Any]:
    cfg = cfg or load_config()
    data_root = resolve_data_root(cfg)
    model_path = Path(model_dir) if model_dir else (REPO_ROOT / "models")

    if model_path.is_file():
        model_sha = sha256_file(model_path)
    elif model_path.is_dir() and any(model_path.iterdir()):
        model_sha = sha256_tree(model_path)
    else:
        model_sha = "none"  # P0: no model weights yet

    dataset_sha = sha256_tree(data_root) if data_root.is_dir() else "none"

    return {
        "model_sha256": model_sha,
        "dataset_sha256": dataset_sha,
        "threshold": {
            "forgery": cfg.get("thresholds", {}).get("forgery"),
            "face": cfg.get("thresholds", {}).get("face"),
        },
        "seed": cfg.get("seed"),
        "git_commit": git_commit(),
        "git_dirty": git_working_tree_dirty(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def write_freeze(
    cfg: dict[str, Any] | None = None,
    out_path: Path | str | None = None,
) -> dict[str, Any]:
    cfg = cfg or load_config()
    results = resolve_results_root(cfg)
    payload = collect_freeze(cfg)
    dest = Path(out_path) if out_path else results / "freeze.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


# DS-3 trial scale (field_collected only)
DS3_SAME_MIN, DS3_SAME_MAX = 150, 250
DS3_DIFF_MIN, DS3_DIFF_MAX = 150, 250
DS3_TOTAL_MIN, DS3_TOTAL_MAX = 300, 500
NON_TRIAL_ORIGINS = frozenset({"dev_fixture", "public_dataset", "synthetic_generated"})


def check_strict(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("model_sha256") in (None, "", "none"):
        errors.append("model_sha256 is 'none' (no frozen model weights)")
    if payload.get("git_commit") in (None, "", "unknown"):
        errors.append("git_commit is unknown")
    if payload.get("git_dirty"):
        errors.append("git working tree is dirty")
    return errors


def find_dev_fixture_manifests(data_root: Path) -> list[Path]:
    """Return manifests under data_root containing a non-trial provenance row."""
    matches: list[Path] = []
    if not data_root.is_dir():
        return matches
    for manifest in data_root.rglob("manifest.jsonl"):
        with manifest.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("//"):
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("origin") in NON_TRIAL_ORIGINS:
                    matches.append(manifest)
                    break
    return matches


def _load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def check_ds3_trial_rules(cfg: dict[str, Any], payload: dict[str, Any]) -> list[str]:
    """DS-3 trial: field_collected only. MIDV/synthetic are calib-only — no threshold lock."""
    errors: list[str] = []
    data_root = resolve_data_root(cfg)
    face_manifest = data_root / "3_face" / "manifest.jsonl"
    rows = _load_jsonl_rows(face_manifest)
    face_threshold = (payload.get("threshold") or {}).get("face")
    if face_threshold is None:
        face_threshold = (cfg.get("thresholds") or {}).get("face")

    if not rows:
        errors.append(
            "DS-3: missing field_collected 3_face/manifest.jsonl "
            "(public/synthetic cannot substitute for trial)"
        )
        if face_threshold is not None:
            errors.append(
                "DS-3: thresholds.face is set but no field_collected face manifest — "
                "MIDV/합성 캘리브로는 임계값 확정 불가"
            )
        return errors

    origins = {str(r.get("origin") or "unknown") for r in rows}
    non_field = origins - {"field_collected"}
    if non_field:
        errors.append(
            "DS-3 trial requires origin=field_collected only; found "
            + ", ".join(sorted(non_field))
            + " (MIDV/synthetic/dev are calibration-only, not TTA trial data)"
        )

    under = [
        str(r.get("id"))
        for r in rows
        if str(r.get("pair_warning") or "").startswith("under_variation")
    ]
    if under:
        errors.append(
            "DS-3: under-variation pairs present (e.g. MIDV) — "
            "변이 과소, 임계값 확정 불가; cannot freeze trial thresholds"
        )

    n_same = sum(1 for r in rows if r.get("same") is True)
    n_diff = sum(1 for r in rows if r.get("same") is False)
    total = len(rows)
    if not (DS3_SAME_MIN <= n_same <= DS3_SAME_MAX):
        errors.append(f"DS-3 same pairs {n_same} outside [{DS3_SAME_MIN}, {DS3_SAME_MAX}]")
    if not (DS3_DIFF_MIN <= n_diff <= DS3_DIFF_MAX):
        errors.append(f"DS-3 diff pairs {n_diff} outside [{DS3_DIFF_MIN}, {DS3_DIFF_MAX}]")
    if not (DS3_TOTAL_MIN <= total <= DS3_TOTAL_MAX):
        errors.append(f"DS-3 total pairs {total} outside [{DS3_TOTAL_MIN}, {DS3_TOTAL_MAX}]")

    if face_threshold is not None and non_field:
        errors.append(
            "DS-3: thresholds.face cannot be locked from public_dataset/synthetic_generated "
            "(캘리브레이션 전용 — 임계값 확정 근거 불가)"
        )

    results_root = resolve_results_root(cfg)
    face_result = results_root / "tc4_tc5_face.json"
    if face_result.is_file():
        try:
            result = json.loads(face_result.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            result = {}
        if result.get("under_variation_warning") or result.get("threshold_lock") == "blocked":
            if face_threshold is not None:
                errors.append(
                    "DS-3: tc4_tc5_face.json marks under-variation / threshold_lock=blocked "
                    "but thresholds.face is set — 변이 과소, 임계값 확정 불가"
                )
        dist = result.get("origin_distribution") or {}
        if any(k in NON_TRIAL_ORIGINS for k, n in dist.items() if n):
            errors.append(
                "DS-3: last face eval used non-field origins "
                f"{dist} — trial freeze requires field_collected only"
            )

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Write freeze metadata")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Exit 1 if model/git dirty, non-trial manifests in data_root, "
            "or DS-3 is not field_collected trial scale (MIDV/synthetic cannot lock face thresholds)"
        ),
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    payload = write_freeze(cfg)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.strict:
        errors = check_strict(payload)
        data_root = resolve_data_root(cfg)
        dev_manifests = find_dev_fixture_manifests(data_root)
        if dev_manifests:
            errors.append(
                "non-trial provenance in data_root manifests: "
                + ", ".join(str(path.relative_to(data_root)) for path in dev_manifests)
                + " (dev_fixture/public_dataset/synthetic_generated forbidden for TTA freeze)"
            )
        errors.extend(check_ds3_trial_rules(cfg, payload))
        if errors:
            print("STRICT FREEZE FAILED:", file=sys.stderr)
            for e in errors:
                print(f"  - {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
