#!/usr/bin/env python3
"""
check_holdout_leakage.py

data/2_forgery_gen/ (train pool) vs data/2_forgery/ (eval n=1000) leakage checks:

  1) exact  : sha256 identical file
  2) near   : perceptual hash Hamming distance <= threshold
  3) source : trailing 4-digit index shared across train forged / eval auth|forged
              (gen_forgery builds fg_*_{i:04d} from sorted authentic[i-1];
               eval uses auth_{i:04d} / fg_{i:04d} from the same authentic pool)

Also prints an AUTHENTIC_PATH note: ForgeryNet train negatives are JPEG-recompressed
copies of data/2_forgery/authentic — same paths as eval authentic (by design in
services/forgery/train.py). That identity reuse is not visible under --train-dir
= 2_forgery_gen alone.

Usage:
    python tools/check_holdout_leakage.py \\
        --train-dir data/2_forgery_gen \\
        --eval-dir data/2_forgery \\
        --out results/report_holdout.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

try:
    from PIL import Image
    import imagehash

    HAS_IMAGEHASH = True
except ImportError:
    HAS_IMAGEHASH = False

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Trailing #### from auth_0001 / fg_0001 / fg_train_v2_easy_0044
DEFAULT_SOURCE_ID_REGEX = r"^(?:auth|fg(?:_[A-Za-z0-9]+)*)_(\d{4})$"


def iter_images(root: Path):
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in IMG_EXTS:
            continue
        # Masks are not training/eval images for leakage purposes.
        if "masks" in p.parts:
            continue
        yield p


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def phash_of(path: Path):
    if not HAS_IMAGEHASH:
        return None
    try:
        with Image.open(path) as img:
            return imagehash.phash(img)
    except Exception as e:
        print(f"[warn] phash failed: {path} ({e})", file=sys.stderr)
        return None


def extract_source_id(path: Path, pattern: str):
    m = re.match(pattern, path.stem)
    return m.group(1) if m else None


def load_manifest_doc_keys(root: Path) -> dict[str, str]:
    """Map absolute image path → authentic document stem (e.g. auth_0042).

    Prefer explicit ``source`` / authentic path stems from manifests so pool
    indices like ``fg_train_0001`` are not mistaken for document IDs.
    """
    mapping: dict[str, str] = {}
    for manifest in root.rglob("manifest.jsonl"):
        try:
            lines = manifest.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        base = manifest.parent
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            rel = row.get("path")
            if not rel:
                continue
            abs_path = (base / rel).resolve()
            source = row.get("source")
            if isinstance(source, str) and source and source not in {"midv2020", "l3i_midv2020"}:
                # gen_forgery stores authentic filename in source; ingest uses dataset tag.
                mapping[str(abs_path)] = Path(source).stem
            elif int(row.get("label", -1)) == 0:
                mapping[str(abs_path)] = Path(rel).stem
            else:
                # fg_auth_0042.png style ids from holdout rebuild
                stem = Path(rel).stem
                if stem.startswith("fg_auth_"):
                    mapping[str(abs_path)] = stem[len("fg_") :]
                elif stem.startswith("fg_") and re.match(r"^auth_\d{4}$", stem[3:]):
                    mapping[str(abs_path)] = stem[3:]
    return mapping


def build_index(
    root: Path,
    source_id_regex: str,
    label: str,
    *,
    authentic_allow: set[str] | None = None,
):
    index = []
    files = []
    for p in iter_images(root):
        if any(
            part.endswith("_pre_holdout") or part.startswith("_holdout")
            for part in p.parts
        ):
            continue
        if authentic_allow is not None and "authentic" in p.parts and p.name not in authentic_allow:
            continue
        files.append(p)
    doc_keys = load_manifest_doc_keys(root)
    print(f"[{label}] scanning {len(files)} files (manifest doc keys={len(doc_keys)})...")
    for i, p in enumerate(files, 1):
        key = doc_keys.get(str(p.resolve()))
        if key is None and "authentic" in p.parts:
            key = p.stem  # auth_0042
        if key is None:
            # Legacy fallback: trailing #### pool index (noisy after holdout).
            key = extract_source_id(p, source_id_regex)
        index.append(
            {
                "path": str(p),
                "sha256": sha256_of(p),
                "phash": phash_of(p),
                "source_id": key,
            }
        )
        if i % 500 == 0:
            print(f"  {i}/{len(files)}")
    return index


def check_holdout_split(eval_dir: Path) -> dict | None:
    split_path = eval_dir / "holdout_split.json"
    if not split_path.is_file():
        return None
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    train = set(payload.get("train_authentic") or [])
    ev = set(payload.get("eval_authentic") or [])
    return {
        "path": str(split_path),
        "train_n": len(train),
        "eval_n": len(ev),
        "train_authentic": sorted(train),
        "eval_authentic": sorted(ev),
        "document_name_overlap": sorted(train & ev),
        "document_name_overlap_count": len(train & ev),
    }


def find_exact_overlap(train_idx, eval_idx):
    train_by_hash = defaultdict(list)
    for e in train_idx:
        train_by_hash[e["sha256"]].append(e["path"])
    overlaps = []
    for e in eval_idx:
        if e["sha256"] in train_by_hash:
            overlaps.append(
                {
                    "eval_path": e["path"],
                    "train_paths": train_by_hash[e["sha256"]],
                }
            )
    return overlaps


def find_near_overlap(train_idx, eval_idx, threshold: int):
    if not HAS_IMAGEHASH:
        print(
            "[warn] imagehash not installed — skipping near-duplicate check "
            "(pip install imagehash)",
            file=sys.stderr,
        )
        return []
    overlaps = []
    train_entries = [e for e in train_idx if e["phash"] is not None]
    for e in eval_idx:
        if e["phash"] is None:
            continue
        for t in train_entries:
            dist = e["phash"] - t["phash"]
            if dist <= threshold:
                overlaps.append(
                    {
                        "eval_path": e["path"],
                        "train_path": t["path"],
                        "hamming_distance": int(dist),
                    }
                )
    return overlaps


def find_source_id_overlap(train_idx, eval_idx):
    train_ids = defaultdict(list)
    for e in train_idx:
        if e["source_id"]:
            train_ids[e["source_id"]].append(e["path"])
    overlaps = []
    seen_eval_ids = set()
    for e in eval_idx:
        sid = e["source_id"]
        if sid and sid in train_ids and sid not in seen_eval_ids:
            seen_eval_ids.add(sid)
            overlaps.append(
                {
                    "source_id": sid,
                    "eval_paths": [x["path"] for x in eval_idx if x["source_id"] == sid],
                    "train_paths": train_ids[sid],
                }
            )
    return overlaps


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--train-dir", required=True, type=Path)
    ap.add_argument("--eval-dir", required=True, type=Path)
    ap.add_argument(
        "--source-id-regex",
        default=DEFAULT_SOURCE_ID_REGEX,
        help="stem regex; group(1) = source document index",
    )
    ap.add_argument("--phash-threshold", type=int, default=6)
    ap.add_argument("--out", type=Path, default=Path("results/report_holdout.json"))
    args = ap.parse_args()

    if not args.train_dir.exists():
        sys.exit(f"train-dir missing: {args.train_dir}")
    if not args.eval_dir.exists():
        sys.exit(f"eval-dir missing: {args.eval_dir}")

    auth_dir = args.eval_dir / "authentic"
    holdout = check_holdout_split(args.eval_dir)
    if holdout and holdout["document_name_overlap_count"] == 0:
        print(
            "NOTE: holdout_split.json present — train authentic should be filtered "
            f"to {holdout['train_n']} docs (disjoint from {holdout['eval_n']} eval). "
            f"JPEG train negatives still read from {auth_dir} but only train names.\n"
        )
    else:
        print(
            "NOTE: without a clean holdout_split, train negatives (JPEG) may reuse "
            f"the same authentic files under {auth_dir} as eval "
            "(services/forgery/train.py).\n"
        )

    train_idx = build_index(args.train_dir, args.source_id_regex, "train")
    eval_allow = set(holdout["eval_authentic"]) if holdout else None
    eval_idx = build_index(
        args.eval_dir, args.source_id_regex, "eval", authentic_allow=eval_allow
    )

    exact = find_exact_overlap(train_idx, eval_idx)
    near = find_near_overlap(train_idx, eval_idx, args.phash_threshold)
    by_source = find_source_id_overlap(train_idx, eval_idx)

    n_eval_with_source_id = sum(1 for e in eval_idx if e["source_id"])
    n_train_with_source_id = sum(1 for e in train_idx if e["source_id"])
    report = {
        "train_dir": str(args.train_dir),
        "eval_dir": str(args.eval_dir),
        "source_id_regex": args.source_id_regex,
        "holdout_split": holdout,
        "n_train_files": len(train_idx),
        "n_eval_files": len(eval_idx),
        "n_train_with_extractable_source_id": n_train_with_source_id,
        "n_eval_with_extractable_source_id": n_eval_with_source_id,
        "authentic_path_reuse_in_train_code": holdout is None
        or holdout["document_name_overlap_count"] > 0,
        "authentic_path_reuse_note": (
            "With holdout_split.json, train.py restricts JPEG authentic to "
            "train_authentic names (document-disjoint from eval)."
            if holdout and holdout["document_name_overlap_count"] == 0
            else (
                "ForgeryNet train authentic samples may point at eval-dir/authentic "
                "with jpeg_quality recompress; path identity by design unless holdout."
            )
        ),
        "exact_duplicate_count": len(exact),
        "near_duplicate_count": len(near),
        "source_id_overlap_count": len(by_source),
        "exact_duplicates": exact[:50],
        "near_duplicates": near[:50],
        "source_id_overlaps": by_source[:50],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("\n=== result ===")
    print(f"train pool: {len(train_idx)}  eval: {len(eval_idx)}")
    if holdout:
        print(
            f"holdout_split: train={holdout['train_n']} eval={holdout['eval_n']} "
            f"name_overlap={holdout['document_name_overlap_count']}"
        )
    print(f"exact (sha256): {len(exact)}")
    print(f"near (phash <= {args.phash_threshold}): {len(near)}")
    print(
        f"source_id extractable: train={n_train_with_source_id} eval={n_eval_with_source_id}"
    )
    print(f"source_id overlap: {len(by_source)}")
    print(f"report: {args.out}")

    if holdout and holdout["document_name_overlap_count"]:
        print("\n[FAIL] holdout_split.json has train∩eval authentic names.")
        sys.exit(1)
    if exact or near:
        print("\n[FAIL] pixel-level leakage — eval numbers are not a clean holdout.")
        sys.exit(1)
    if by_source:
        print(
            "\n[CHECK] source_id / document overlap — same authentic document appears "
            "in train forged and eval."
        )
        sys.exit(2)
    print("\n[PASS] no exact/near/document overlap.")
    sys.exit(0)


if __name__ == "__main__":
    main()
