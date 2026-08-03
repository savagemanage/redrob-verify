#!/usr/bin/env python3
"""Ingest MIDV-2020 stage-1 archives into harness ds1/ds2/ds3 layouts.

Expects archives under::

    results/midv_archives/dataset/{scan_upright,scan_rotated,photo,templates}.tar

Writes into ``data/``::

    1_ocr/  2_forgery/  3_face/
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import tarfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from harness.config_util import DS_FACE, DS_FORGERY, DS_OCR, DS_RESUME, REPO_ROOT
from tools.gen_forgery import generate as gen_forgery_generate

FTP_ROOT = REPO_ROOT / "results" / "midv_archives"
EXTRACT_ROOT = FTP_ROOT / "extracted"
DEFAULT_OUT = REPO_ROOT / "data"
GEN_FORGERY_TMP = REPO_ROOT / "results" / "_tmp_forgery"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def _ensure_extracted(archive: Path, dest: Path) -> Path:
    marker = dest / ".extracted_ok"
    if marker.is_file() and dest.is_dir():
        return dest
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    print(f"extract {archive.name} → {dest}", flush=True)
    with tarfile.open(archive, "r") as tar:
        tar.extractall(dest)
    marker.write_text(archive.name + "\n", encoding="utf-8")
    return dest


def _load_via_annotation(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _via_regions(ann: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize VIA v2 project JSON into a list of region dicts with attrs."""
    regions: list[dict[str, Any]] = []
    # VIA project: _via_img_metadata[key].regions[]
    meta = ann.get("_via_img_metadata") or {}
    if meta:
        for _key, entry in meta.items():
            for region in entry.get("regions") or []:
                shape = region.get("shape_attributes") or {}
                attrs = region.get("region_attributes") or {}
                regions.append({"shape": shape, "attrs": attrs, "file": entry.get("filename")})
        return regions
    # Sometimes one-image export
    for region in ann.get("regions") or []:
        regions.append(
            {
                "shape": region.get("shape_attributes") or {},
                "attrs": region.get("region_attributes") or {},
                "file": ann.get("filename"),
            }
        )
    return regions


def _field_values_from_regions(regions: list[dict[str, Any]]) -> dict[str, str]:
    fields: dict[str, str] = {}
    for region in regions:
        attrs = region["attrs"]
        # Common MIDV keys: field_name / value / name
        name = (
            attrs.get("field_name")
            or attrs.get("name")
            or attrs.get("field")
            or attrs.get("key")
        )
        value = attrs.get("value") or attrs.get("field_value") or attrs.get("text")
        if name and value is not None and str(name).lower() not in {"face", "photo", "portrait"}:
            fields[str(name)] = str(value)
    return fields


def _face_quad_from_regions(regions: list[dict[str, Any]]) -> np.ndarray | None:
    for region in regions:
        attrs = {str(k).lower(): v for k, v in region["attrs"].items()}
        name = str(attrs.get("field_name") or attrs.get("name") or attrs.get("type") or "").lower()
        # Prefer oval/face over photo (holder photo is larger); use first face-ish.
        if name not in {"face", "photo", "portrait", "face_photo", "photograph"}:
            if "face" not in name and "photo" not in name and "portrait" not in name:
                continue
        if name != "face" and "face" not in name:
            continue  # skip photo/portrait boxes; use face oval geometry
        shape = region["shape"]
        st = shape.get("name")
        if st == "polygon" and "all_points_x" in shape and "all_points_y" in shape:
            xs = shape["all_points_x"]
            ys = shape["all_points_y"]
            pts = np.array([[int(x), int(y)] for x, y in zip(xs, ys)], dtype=np.int32)
            if len(pts) >= 3:
                return pts
        if st == "rect":
            x = int(shape.get("x", 0))
            y = int(shape.get("y", 0))
            w = int(shape.get("width", 0))
            h = int(shape.get("height", 0))
            return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.int32)
    return None

def _crop_quad(image: np.ndarray, quad: np.ndarray, out_size: tuple[int, int] = (160, 200)) -> np.ndarray:
    x, y, w, h = cv2.boundingRect(quad)
    x2, y2 = x + w, y + h
    x, y = max(0, x), max(0, y)
    x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
    crop = image[y:y2, x:x2]
    if crop.size == 0:
        raise ValueError("empty face crop")
    return cv2.resize(crop, out_size, interpolation=cv2.INTER_AREA)


def _iter_code_images(root: Path) -> list[tuple[str, str, Path, Path]]:
    """Yield (code, idx, image_path, annotation_json_path_for_code)."""
    images_root = root / "images"
    ann_root = root / "annotations"
    if not images_root.is_dir():
        # sometimes extracted with nested top folder
        subs = [p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")]
        for sub in subs:
            if (sub / "images").is_dir():
                return _iter_code_images(sub)
        return []
    rows: list[tuple[str, str, Path, Path]] = []
    for code_dir in sorted(p for p in images_root.iterdir() if p.is_dir()):
        code = code_dir.name
        ann_path = ann_root / f"{code}.json"
        if not ann_path.is_file():
            # per-image annotations folder variant
            ann_path = ann_root / code
        for img in sorted(code_dir.iterdir()):
            if img.suffix.lower() not in IMAGE_EXTS:
                continue
            idx = img.stem
            rows.append((code, idx, img, ann_path))
    return rows


def _regions_for_image(ann_path: Path, code: str, idx: str, filename: str) -> list[dict[str, Any]]:
    if ann_path.is_dir():
        candidate = ann_path / f"{idx}.json"
        if candidate.is_file():
            return _via_regions(_load_via_annotation(candidate))
        return []
    if not ann_path.is_file():
        return []
    ann = _load_via_annotation(ann_path)
    meta = ann.get("_via_img_metadata") or {}
    # Match by filename endings
    targets = {filename, f"{idx}.jpg", f"{idx}.jpeg", f"{code}/{filename}", f"{idx}"}
    for _key, entry in meta.items():
        fn = str(entry.get("filename") or "")
        base = Path(fn).name
        if fn in targets or base in targets or base.startswith(idx):
            regions = []
            for region in entry.get("regions") or []:
                regions.append(
                    {
                        "shape": region.get("shape_attributes") or {},
                        "attrs": region.get("region_attributes") or {},
                        "file": fn,
                    }
                )
            return regions
    # Fallback: all regions (template-level shared) — rare
    return _via_regions(ann)


def _template_fields_index(templates_root: Path) -> dict[tuple[str, str], dict[str, str]]:
    """MIDV text GT lives on templates only; scan/photo annotations are face+doc_quad."""
    index: dict[tuple[str, str], dict[str, str]] = {}
    for code, idx, img_path, ann_path in _iter_code_images(templates_root):
        regions = _regions_for_image(ann_path, code, idx, img_path.name)
        fields = _field_values_from_regions(regions)
        if fields:
            index[(code, idx)] = fields
    return index


def build_ds1(
    scan_roots: list[Path],
    photo_root: Path,
    templates_root: Path,
    out: Path,
) -> int:
    img_out = out / "img"
    if out.exists():
        shutil.rmtree(out)
    img_out.mkdir(parents=True)
    tmpl_fields = _template_fields_index(templates_root)
    print(f"template field GT docs={len(tmpl_fields)}", flush=True)
    records: list[dict[str, Any]] = []
    n = 0
    sources = [(p, "scan") for p in scan_roots] + [(photo_root, "photo")]
    for root, capture in sources:
        for code, idx, img_path, ann_path in _iter_code_images(root):
            fields = dict(tmpl_fields.get((code, idx), {}))
            gt_text = "\n".join(f"{k}: {v}" for k, v in fields.items()) if fields else ""
            rel = f"img/{capture}_{code}_{idx}{img_path.suffix.lower()}"
            shutil.copy2(img_path, out / rel)
            n += 1
            records.append(
                {
                    "id": f"ocr_{capture}_{code}_{idx}",
                    "path": rel,
                    "doc_type": code,
                    "gt_text": gt_text,
                    "gt_fields": fields,
                    "origin": "public_dataset",
                    "capture": capture,
                    "source": "midv2020",
                    "script": {
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
                    }.get(str(code).split("_", 1)[0].lower()),
                }
            )
    with (out / "manifest.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return n

def build_ds2(scan_root: Path, out: Path, *, n_each: int = 500, seed: int = 42) -> dict[str, int]:
    if out.exists():
        shutil.rmtree(out)
    auth_dir = out / "authentic"
    auth_dir.mkdir(parents=True)
    images = [t[2] for t in _iter_code_images(scan_root)]
    rng = random.Random(seed)
    rng.shuffle(images)
    images = images[:n_each]
    if len(images) < n_each:
        raise RuntimeError(f"need {n_each} scan images, found {len(images)}")

    records: list[dict[str, Any]] = []
    for i, src in enumerate(images, start=1):
        name = f"auth_{i:04d}{src.suffix.lower()}"
        shutil.copy2(src, auth_dir / name)
        records.append(
            {
                "id": f"auth_{i:04d}",
                "path": f"authentic/{name}",
                "label": 0,
                "origin": "public_dataset",
                "fabrication": "authentic",
                "eval_domain": "in_domain",
                "generator": "midv_scan",
                "source": "midv2020",
            }
        )

    # Generate forgeries into a scratch dir under midv2020, then copy in
    if GEN_FORGERY_TMP.exists():
        shutil.rmtree(GEN_FORGERY_TMP)
    forged = gen_forgery_generate(
        auth_dir,
        profile_name="test",
        difficulty="medium",
        count=n_each,
        output_root=GEN_FORGERY_TMP,
    )
    gen_root = GEN_FORGERY_TMP / "test"
    forg_img = out / "images"
    forg_mask = out / "masks"
    forg_img.mkdir(parents=True, exist_ok=True)
    forg_mask.mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(forged[:n_each], start=1):
        src = gen_root / row["path"]
        msrc = gen_root / row["mask_path"]
        dst_name = f"fg_{i:04d}{src.suffix.lower()}"
        mask_name = f"fg_{i:04d}_mask.png"
        shutil.copy2(src, forg_img / dst_name)
        shutil.copy2(msrc, forg_mask / mask_name)
        records.append(
            {
                "id": f"fg_{i:04d}",
                "path": f"images/{dst_name}",
                "label": 1,
                "origin": "synthetic_generated",
                "fabrication": "script",
                "tamper": row.get("tamper"),
                "mask_path": f"masks/{mask_name}",
                "eval_domain": "in_domain",
                "generator": "gen_forgery",
                "profile": "test",
            }
        )

    with (out / "manifest.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {
        "authentic": sum(1 for r in records if r["label"] == 0),
        "forged": sum(1 for r in records if r["label"] == 1),
    }


def build_ds3(scan_root: Path, photo_root: Path, out: Path, *, seed: int = 42) -> dict[str, int]:
    if out.exists():
        shutil.rmtree(out)
    face_dir = out / "faces"
    face_dir.mkdir(parents=True)

    def _index(root: Path) -> dict[tuple[str, str], tuple[Path, Path]]:
        mapping: dict[tuple[str, str], tuple[Path, Path]] = {}
        for code, idx, img, ann in _iter_code_images(root):
            mapping[(code, idx)] = (img, ann)
        return mapping

    scans = _index(scan_root)
    photos = _index(photo_root)
    common = sorted(set(scans) & set(photos))
    if len(common) < 2:
        raise RuntimeError(f"need paired scan/photo ids, found {len(common)}")

    crops: dict[tuple[str, str], tuple[Path, Path]] = {}
    for code, idx in common:
        scan_img, scan_ann = scans[(code, idx)]
        photo_img, photo_ann = photos[(code, idx)]
        s_regions = _regions_for_image(scan_ann, code, idx, scan_img.name)
        p_regions = _regions_for_image(photo_ann, code, idx, photo_img.name)
        s_quad = _face_quad_from_regions(s_regions)
        p_quad = _face_quad_from_regions(p_regions)
        if s_quad is None or p_quad is None:
            continue
        s_im = cv2.imread(str(scan_img))
        p_im = cv2.imread(str(photo_img))
        if s_im is None or p_im is None:
            continue
        try:
            s_crop = _crop_quad(s_im, s_quad)
            p_crop = _crop_quad(p_im, p_quad)
        except ValueError:
            continue
        s_path = face_dir / f"{code}_{idx}_scan.jpg"
        p_path = face_dir / f"{code}_{idx}_photo.jpg"
        cv2.imwrite(str(s_path), s_crop)
        cv2.imwrite(str(p_path), p_crop)
        crops[(code, idx)] = (s_path, p_path)

    ids = sorted(crops)
    if len(ids) < 2:
        raise RuntimeError("insufficient face crops from MIDV annotations")

    records: list[dict[str, Any]] = []
    for i, key in enumerate(ids, start=1):
        a, b = crops[key]
        records.append(
            {
                "id": f"fc_same_{i:04d}",
                "img_a": f"faces/{a.name}",
                "img_b": f"faces/{b.name}",
                "same": True,
                "origin": "public_dataset",
                "identity_id": f"{key[0]}_{key[1]}",
                "capture_a": "scan",
                "capture_b": "photo",
                "pair_warning": "under_variation:midv_scan_photo",
            }
        )

    rng = random.Random(seed)
    n_same = len(ids)
    # balanced different pairs
    diff_needed = n_same
    used: set[tuple[int, int]] = set()
    while len(used) < diff_needed and len(used) < n_same * (n_same - 1) // 2:
        i, j = rng.sample(range(n_same), 2)
        if i > j:
            i, j = j, i
        used.add((i, j))
    for k, (i, j) in enumerate(sorted(used), start=1):
        ka, kb = ids[i], ids[j]
        a = crops[ka][0]  # scan of A
        b = crops[kb][1]  # photo of B
        records.append(
            {
                "id": f"fc_diff_{k:04d}",
                "img_a": f"faces/{a.name}",
                "img_b": f"faces/{b.name}",
                "same": False,
                "origin": "public_dataset",
                "identity_a": f"{ka[0]}_{ka[1]}",
                "identity_b": f"{kb[0]}_{kb[1]}",
                "capture_a": "scan",
                "capture_b": "photo",
                "pair_warning": "under_variation:midv_scan_photo",
            }
        )

    with (out / "manifest.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {
        "identities": n_same,
        "same": sum(1 for r in records if r["same"]),
        "diff": sum(1 for r in records if not r["same"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest MIDV-2020 into harness datasets")
    parser.add_argument("--ftp-root", type=Path, default=FTP_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-forgery", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-extract", action="store_true")
    args = parser.parse_args()

    dataset = args.ftp_root / "dataset"
    archives = {
        "scan_upright": dataset / "scan_upright.tar",
        "scan_rotated": dataset / "scan_rotated.tar",
        "photo": dataset / "photo.tar",
        "templates": dataset / "templates.tar",
    }
    for key, path in archives.items():
        if not path.is_file():
            raise SystemExit(f"missing {path} — run tools/fetch_midv.py first")

    if not args.skip_extract:
        scan_u = _ensure_extracted(archives["scan_upright"], EXTRACT_ROOT / "scan_upright")
        scan_r = _ensure_extracted(archives["scan_rotated"], EXTRACT_ROOT / "scan_rotated")
        photo = _ensure_extracted(archives["photo"], EXTRACT_ROOT / "photo")
        templates = _ensure_extracted(archives["templates"], EXTRACT_ROOT / "templates")
    else:
        scan_u = EXTRACT_ROOT / "scan_upright"
        scan_r = EXTRACT_ROOT / "scan_rotated"
        photo = EXTRACT_ROOT / "photo"
        templates = EXTRACT_ROOT / "templates"

    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    print("=== 1_ocr ===", flush=True)
    n1 = build_ds1([scan_u, scan_r], photo, templates, out / DS_OCR)
    print(f"1_ocr rows={n1}", flush=True)

    print("=== 2_forgery ===", flush=True)
    n2 = build_ds2(scan_u, out / DS_FORGERY, n_each=args.n_forgery, seed=args.seed)
    print(n2, flush=True)

    print("=== 3_face ===", flush=True)
    n3 = build_ds3(scan_u, photo, out / DS_FACE, seed=args.seed)
    print(n3, flush=True)

    # 4_resume already present under data/ after gen_resumes; skip if missing
    ds4_dst = out / DS_RESUME
    if not ds4_dst.is_dir():
        print("WARNING: data/4_resume missing — run ./run.sh gen-resumes", flush=True)

    if GEN_FORGERY_TMP.exists():
        shutil.rmtree(GEN_FORGERY_TMP)

    summary = {"1_ocr": n1, "2_forgery": n2, "3_face": n3, "out": str(out)}
    (out / "ingest_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
