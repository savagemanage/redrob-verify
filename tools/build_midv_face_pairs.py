"""Build DS-3 calibration pairs from MIDV-2020 photo/scan/video captures.

Prefer ``tools/ingest_midv.py``. Fallback layout::

    data/raw_ids/<doc_id>/{photo,scan,video}/...

Writes ``data/3_face`` with ``origin=public_dataset`` and
``pair_warning=under_variation:midv_cross_capture``.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from harness.config_util import REPO_ROOT

MIDV_ROOT = REPO_ROOT / "results" / "midv_archives" / "raw_ids"
OUT_ROOT = REPO_ROOT / "data" / "3_face"


def _collect_capture_images(identity_dir: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    mapping = {
        "photo": ("photo", "photos", "image"),
        "scan": ("scan", "scans"),
        "video_frame": ("video", "videos", "clip"),
    }
    for label, names in mapping.items():
        for name in names:
            folder = identity_dir / name
            if not folder.is_dir():
                continue
            images = sorted(
                p
                for p in folder.rglob("*")
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
            )
            if images:
                found[label] = images[0]
                break
    return found


def build_pairs() -> dict[str, int]:
    if not MIDV_ROOT.is_dir():
        OUT_ROOT.mkdir(parents=True, exist_ok=True)
        readme = OUT_ROOT / "README.md"
        readme.write_text(
            "Place MIDV under data/raw_ids/<id>/{photo,scan}/ then re-run "
            "`uv run python tools/build_midv_face_pairs.py` "
            "(or prefer tools/ingest_midv.py).\n"
            "Do not invent pairs by augmenting a single image.\n",
            encoding="utf-8",
        )
        (OUT_ROOT / "manifest.jsonl").write_text("", encoding="utf-8")
        return {"identities": 0, "pairs": 0, "status": "midv_missing"}

    out_img = OUT_ROOT / "img"
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    out_img.mkdir(parents=True)

    records = []
    pair_i = 0
    for identity_dir in sorted(p for p in MIDV_ROOT.iterdir() if p.is_dir()):
        captures = _collect_capture_images(identity_dir)
        keys = [k for k in ("photo", "scan", "video_frame") if k in captures]
        # Same-identity cross capture pairs only
        for a in range(len(keys)):
            for b in range(a + 1, len(keys)):
                pair_i += 1
                ca, cb = keys[a], keys[b]
                dst_a = out_img / f"{identity_dir.name}_{ca}{captures[ca].suffix.lower()}"
                dst_b = out_img / f"{identity_dir.name}_{cb}{captures[cb].suffix.lower()}"
                if not dst_a.is_file():
                    shutil.copy2(captures[ca], dst_a)
                if not dst_b.is_file():
                    shutil.copy2(captures[cb], dst_b)
                records.append(
                    {
                        "id": f"midv_fc_{pair_i:04d}",
                        "img_a": f"img/{dst_a.name}",
                        "img_b": f"img/{dst_b.name}",
                        "same": True,
                        "origin": "public_dataset",
                        "identity_id": identity_dir.name,
                        "capture_a": ca,
                        "capture_b": cb,
                        "pair_warning": "under_variation:midv_cross_capture",
                    }
                )
    with (OUT_ROOT / "manifest.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    meta = {
        "role": "calibration_only",
        "origin": "public_dataset",
        "tta_valid": False,
        "threshold_lock": "blocked",
        "warning": "변이 과소, 임계값 확정 불가",
        "pairs": len(records),
        "identities": len({r["identity_id"] for r in records}),
    }
    (OUT_ROOT / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    # Also expose as 3_face layout for optional config swap
    ds3 = OUT_ROOT / "3_face"
    ds3.mkdir(exist_ok=True)
    if (OUT_ROOT / "img").exists():
        target = ds3 / "img"
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(OUT_ROOT / "img", target)
    with (ds3 / "manifest.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"identities": len({r["identity_id"] for r in records}), "pairs": len(records)}


def main() -> None:
    parser = argparse.ArgumentParser(description="MIDV-2020 face pair builder (local only)")
    parser.parse_args()
    print(json.dumps(build_pairs(), indent=2))


if __name__ == "__main__":
    main()
