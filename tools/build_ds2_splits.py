"""Assemble DS-2 in-domain / cross-domain forgery manifests (no download).

Layout::

    data/2_forgery/authentic/   # MIDV authentic (from ingest)
    data/fmidv/                            # FMIDV forged IDs when available
    data/_tmp_forgery/test/       # gen_forgery output (optional)

Eval manifests under ``data/2_forgery`` (or --output).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from harness.config_util import REPO_ROOT

MIDV_AUTH = REPO_ROOT / "data" / "2_forgery" / "authentic"
FMIDV_ROOT = REPO_ROOT / "data" / "fmidv"
GEN_FORGERY_TEST = REPO_ROOT / "results" / "_tmp_forgery" / "test"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _images_under(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_splits(
    *,
    midv_root: Path = MIDV_AUTH,
    fmidv_root: Path = FMIDV_ROOT,
    gen_test_root: Path = GEN_FORGERY_TEST,
    output: Path,
    max_auth: int | None = 200,
    max_fmidv: int | None = 500,
    max_gen: int | None = 200,
) -> dict[str, Any]:
    auth = _images_under(midv_root)
    fmidv = _images_under(fmidv_root)
    gen_imgs = _images_under(gen_test_root / "images") if (gen_test_root / "images").is_dir() else _images_under(gen_test_root)

    status = {
        "midv_images": len(auth),
        "fmidv_images": len(fmidv),
        "gen_forgery_test_images": len(gen_imgs),
    }
    if not auth:
        readme = output.parent / "README_ds2_splits.md"
        output.parent.mkdir(parents=True, exist_ok=True)
        readme.write_text(
            "Need MIDV authentic images at data/2_forgery/authentic/ "
            "(run tools/ingest_midv.py) and optionally data/fmidv/ for cross-domain.\n",
            encoding="utf-8",
        )
        _write_jsonl(output / "manifest.jsonl", [])
        return {**status, "wrote": 0, "warning": "midv_missing"}

    if max_auth is not None:
        auth = auth[:max_auth]
    if max_fmidv is not None:
        fmidv = fmidv[:max_fmidv]
    if max_gen is not None:
        gen_imgs = gen_imgs[:max_gen]

    if output.exists():
        shutil.rmtree(output)
    img_dir = output / "images"
    mask_dir = output / "masks"
    img_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)

    rows: list[dict[str, Any]] = []

    def _copy(src: Path, dest_name: str) -> str:
        dst = img_dir / dest_name
        shutil.copy2(src, dst)
        return f"images/{dest_name}"

    # Shared authentic pool referenced by both domains (copied once)
    for i, src in enumerate(auth, start=1):
        rel = _copy(src, f"auth_midv_{i:05d}{src.suffix.lower()}")
        for domain in ("in_domain", "cross_domain"):
            rows.append(
                {
                    "id": f"auth_{domain}_{i:05d}",
                    "path": rel,
                    "label": 0,
                    "origin": "public_dataset",
                    "fabrication": "authentic",
                    "eval_domain": domain,
                    "generator": "midv_authentic",
                    "source": "l3i_midv2020",
                }
            )

    # in-domain forgeries: our generator
    for i, src in enumerate(gen_imgs, start=1):
        rel = _copy(src, f"fg_gen_{i:05d}{src.suffix.lower()}")
        mask_rel = None
        # Prefer sibling mask if present
        cand = gen_test_root / "masks" / f"{src.stem}_mask.png"
        if not cand.is_file():
            cand = src.parent.parent / "masks" / f"{src.stem}_mask.png"
        if cand.is_file():
            mdst = mask_dir / f"fg_gen_{i:05d}_mask.png"
            shutil.copy2(cand, mdst)
            mask_rel = f"masks/{mdst.name}"
        rows.append(
            {
                "id": f"fg_in_{i:05d}",
                "path": rel,
                "label": 1,
                "origin": "synthetic_generated",
                "fabrication": "script",
                "tamper": "mixed",  # gen_forgery: copy_move/splice/inpaint/text_replace
                "mask_path": mask_rel or rel,  # schema requires mask; prefer real mask
                "eval_domain": "in_domain",
                "generator": "gen_forgery",
                "profile": "test",
            }
        )

    # cross-domain forgeries: FMIDV (copy-move only)
    for i, src in enumerate(fmidv, start=1):
        rel = _copy(src, f"fg_fmidv_{i:05d}{src.suffix.lower()}")
        rows.append(
            {
                "id": f"fg_xd_{i:05d}",
                "path": rel,
                "label": 1,
                "origin": "public_dataset",
                "fabrication": "script",
                "tamper": "copy_move",
                "mask_path": rel,  # FMIDV may lack masks; placeholder until annotated
                "eval_domain": "cross_domain",
                "generator": "fmidv",
                "source": "l3i_fmidv",
            }
        )

    if not fmidv:
        print("WARNING: FMIDV empty — cross_domain forged rows missing; eval gate will fail")
    if not gen_imgs:
        print("WARNING: gen_forgery test images empty — run tools/gen_forgery.py --profile test")

    _write_jsonl(output / "manifest.jsonl", rows)
    meta = {
        "design": {
            "in_domain": "MIDV authentic + gen_forgery",
            "cross_domain": "MIDV authentic + FMIDV",
            "pass_gate": "cross_domain",
        },
        "counts": {
            "total": len(rows),
            "in_domain": sum(1 for r in rows if r["eval_domain"] == "in_domain"),
            "cross_domain": sum(1 for r in rows if r["eval_domain"] == "cross_domain"),
            "label_0": sum(1 for r in rows if r["label"] == 0),
            "label_1": sum(1 for r in rows if r["label"] == 1),
        },
        "sources": status,
        "license_note": "L3i-Share: commercial use needs explicit permission; awaiting reply",
    }
    (output / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DS-2 in/cross-domain forgery manifests")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "data" / "public" / "2_forgery_l3i",
    )
    parser.add_argument("--max-auth", type=int, default=200)
    parser.add_argument("--max-fmidv", type=int, default=500)
    parser.add_argument("--max-gen", type=int, default=200)
    args = parser.parse_args()
    meta = build_splits(
        output=args.output,
        max_auth=args.max_auth,
        max_fmidv=args.max_fmidv,
        max_gen=args.max_gen,
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
