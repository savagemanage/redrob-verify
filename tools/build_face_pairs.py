"""Build face pairs from identity directories (doc photo + selfie).

Expected layout::

    <input>/
      <identity_id>/
        doc.jpg|png|…     # or id|photo|document under doc/
        selfie.jpg|png|…  # or live|capture under selfie/

``--origin`` (default ``dev_fixture``):
  - ``dev_fixture`` / ``public_dataset`` / ``synthetic_generated``:
      smoke / calib only → ``tta_valid=false``. No 합의서 scale gate.
      Intended for ``data/3_face`` (e.g. 10 IDs × 2).
  - ``field_collected``:
      trial scale: same 150–250, diff 150–250, total 300–500 → ``tta_valid=true``.

Never invent a second capture by augmenting one image.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any

from harness.config_util import REPO_ROOT
from harness.origin import ORIGIN_TTA_VALID, PROVENANCE_ORIGINS

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DOC_NAMES = ("doc", "id", "photo", "document", "id_photo", "aadhaar_photo")
SELFIE_NAMES = ("selfie", "live", "capture", "webcam", "phone")

# Field trial scale only (origin=field_collected)
FIELD_SAME_MIN, FIELD_SAME_MAX = 150, 250
FIELD_DIFF_MIN, FIELD_DIFF_MAX = 150, 250
FIELD_TOTAL_MIN, FIELD_TOTAL_MAX = 300, 500


def _first_image(folder: Path) -> Path | None:
    if not folder.is_dir():
        return None
    images = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    return images[0] if images else None


def _find_capture(identity_dir: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        sub = _first_image(identity_dir / name)
        if sub is not None:
            return sub
    for name in names:
        for ext in IMAGE_EXTS:
            candidate = identity_dir / f"{name}{ext}"
            if candidate.is_file():
                return candidate
    return None


def discover_identities(input_dir: Path) -> list[dict[str, Any]]:
    identities: list[dict[str, Any]] = []
    for person in sorted(
        p for p in input_dir.iterdir() if p.is_dir() and not p.name.startswith(".")
    ):
        doc = _find_capture(person, DOC_NAMES)
        selfie = _find_capture(person, SELFIE_NAMES)
        if doc is None or selfie is None:
            images = sorted(
                p for p in person.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS
            )
            if len(images) >= 2:
                doc, selfie = images[0], images[1]
            else:
                continue
        if doc.resolve() == selfie.resolve():
            continue
        identities.append({"id": person.name, "doc": doc, "selfie": selfie})
    return identities


def sample_diff_pairs(
    identity_ids: list[str],
    n_diff: int,
    rng: random.Random,
) -> list[tuple[str, str]]:
    """Balanced different-identity pairs: each id appears roughly equally as left/right."""
    n = len(identity_ids)
    if n < 2:
        return []
    max_possible = n * (n - 1) // 2
    n_diff = min(n_diff, max_possible)
    edges: list[tuple[str, str]] = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = identity_ids[i], identity_ids[j]
            if (i + j) % 2 == 0:
                edges.append((a, b))
            else:
                edges.append((b, a))
    rng.shuffle(edges)
    degree = {i: 0 for i in identity_ids}
    chosen: list[tuple[str, str]] = []
    remaining = edges[:]
    while len(chosen) < n_diff and remaining:
        remaining.sort(key=lambda ab: degree[ab[0]] + degree[ab[1]])
        a, b = remaining.pop(0)
        chosen.append((a, b))
        degree[a] += 1
        degree[b] += 1
    return chosen


def validate_counts(n_same: int, n_diff: int, *, origin: str) -> None:
    if origin != "field_collected":
        if n_same < 1:
            raise ValueError("need at least 1 same-identity pair")
        if n_diff < 1 and n_same >= 2:
            raise ValueError("need at least 1 different-identity pair when ≥2 identities")
        return
    errs: list[str] = []
    if not (FIELD_SAME_MIN <= n_same <= FIELD_SAME_MAX):
        errs.append(f"same pairs {n_same} outside [{FIELD_SAME_MIN}, {FIELD_SAME_MAX}]")
    if not (FIELD_DIFF_MIN <= n_diff <= FIELD_DIFF_MAX):
        errs.append(f"diff pairs {n_diff} outside [{FIELD_DIFF_MIN}, {FIELD_DIFF_MAX}]")
    total = n_same + n_diff
    if not (FIELD_TOTAL_MIN <= total <= FIELD_TOTAL_MAX):
        errs.append(f"total pairs {total} outside [{FIELD_TOTAL_MIN}, {FIELD_TOTAL_MAX}]")
    if errs:
        raise ValueError("; ".join(errs))


def build_face_pairs(
    input_dir: Path,
    output_dir: Path,
    *,
    origin: str = "dev_fixture",
    n_diff: int | None = None,
    seed: int = 42,
    max_identities: int | None = None,
) -> dict[str, Any]:
    if origin not in PROVENANCE_ORIGINS:
        raise ValueError(f"origin must be one of {PROVENANCE_ORIGINS}, got {origin!r}")

    identities = discover_identities(input_dir)
    if max_identities is not None:
        identities = identities[:max_identities]
    n_same = len(identities)
    if n_same < 1:
        raise ValueError(f"no identities with doc+selfie under {input_dir}")

    if origin == "field_collected":
        if n_same < FIELD_SAME_MIN:
            raise ValueError(
                f"field trial needs ≥{FIELD_SAME_MIN} identities with doc+selfie, found {n_same}"
            )
        if n_same > FIELD_SAME_MAX:
            identities = identities[:FIELD_SAME_MAX]
            n_same = len(identities)

    target_diff = n_diff if n_diff is not None else n_same
    if origin == "field_collected":
        target_diff = max(FIELD_DIFF_MIN, min(FIELD_DIFF_MAX, target_diff))
        if n_same + target_diff > FIELD_TOTAL_MAX:
            target_diff = FIELD_TOTAL_MAX - n_same
        if n_same + target_diff < FIELD_TOTAL_MIN:
            target_diff = FIELD_TOTAL_MIN - n_same
        target_diff = max(FIELD_DIFF_MIN, min(FIELD_DIFF_MAX, target_diff))
    else:
        # Smoke: match same count, capped by C(n,2)
        max_possible = n_same * (n_same - 1) // 2
        target_diff = min(max(0, target_diff), max_possible)

    validate_counts(n_same, target_diff, origin=origin)

    rng = random.Random(seed)
    by_id = {row["id"]: row for row in identities}
    id_list = [row["id"] for row in identities]
    diff_pairs = sample_diff_pairs(id_list, target_diff, rng)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    doc_out = output_dir / "doc"
    selfie_out = output_dir / "selfie"
    doc_out.mkdir(parents=True)
    selfie_out.mkdir(parents=True)

    for row in identities:
        dst_doc = doc_out / f"{row['id']}{row['doc'].suffix.lower()}"
        dst_selfie = selfie_out / f"{row['id']}{row['selfie'].suffix.lower()}"
        shutil.copy2(row["doc"], dst_doc)
        shutil.copy2(row["selfie"], dst_selfie)
        row["doc_rel"] = f"doc/{dst_doc.name}"
        row["selfie_rel"] = f"selfie/{dst_selfie.name}"

    records: list[dict[str, Any]] = []
    for i, row in enumerate(identities, start=1):
        records.append(
            {
                "id": f"fc_same_{i:04d}",
                "img_a": row["doc_rel"],
                "img_b": row["selfie_rel"],
                "same": True,
                "origin": origin,
                "identity_id": row["id"],
                "capture_a": "doc",
                "capture_b": "selfie",
            }
        )

    for i, (a, b) in enumerate(diff_pairs, start=1):
        records.append(
            {
                "id": f"fc_diff_{i:04d}",
                "img_a": by_id[a]["doc_rel"],
                "img_b": by_id[b]["selfie_rel"],
                "same": False,
                "origin": origin,
                "identity_a": a,
                "identity_b": b,
                "capture_a": "doc",
                "capture_b": "selfie",
            }
        )

    validate_counts(
        sum(1 for r in records if r["same"]),
        sum(1 for r in records if not r["same"]),
        origin=origin,
    )

    with (output_dir / "manifest.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    tta_valid = bool(ORIGIN_TTA_VALID.get(origin, False))
    return {
        "identities": n_same,
        "same_pairs": sum(1 for r in records if r["same"]),
        "diff_pairs": sum(1 for r in records if not r["same"]),
        "total": len(records),
        "origin": origin,
        "tta_valid": tta_valid,
        "purpose": (
            "sface_smoke_similarity_separation"
            if origin != "field_collected"
            else "tta_trial"
        ),
        "output": str(output_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DS-3 face pairs from identity folders")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Directory of identity subfolders (each with doc + selfie)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output 3_face root (default depends on --origin)",
    )
    parser.add_argument(
        "--origin",
        choices=list(PROVENANCE_ORIGINS),
        default="dev_fixture",
        help="Provenance tag written to every pair (default: dev_fixture → tta_valid=false)",
    )
    parser.add_argument("--n-diff", type=int, default=None, help="Target different-identity pairs")
    parser.add_argument("--max-identities", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.output is None:
        if args.origin == "field_collected":
            args.output = REPO_ROOT / "data" / "3_face"
        else:
            args.output = REPO_ROOT / "data" / "3_face"

    summary = build_face_pairs(
        args.input,
        args.output,
        origin=args.origin,
        n_diff=args.n_diff,
        seed=args.seed,
        max_identities=args.max_identities,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
