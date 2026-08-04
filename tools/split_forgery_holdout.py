#!/usr/bin/env python3
"""Document-disjoint forgery holdout: split authentic, regen train/eval, no ID overlap.

Default: 400 train / 100 eval documents (seeded). Train forgeries are regenerated
only from train authentic; eval manifest lists only held-out authentic + test
forgeries from those same documents.

Usage:
    uv run python tools/split_forgery_holdout.py \\
        --seed 42 --train-n 400 --eval-n 100 \\
        --regenerate-train --rebuild-eval
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

from harness.config_util import REPO_ROOT
from tools.gen_forgery import DIFFICULTY, PROFILES, generate

AUTH_DIR = REPO_ROOT / "data" / "2_forgery" / "authentic"
EVAL_ROOT = REPO_ROOT / "data" / "2_forgery"
GEN_ROOT = REPO_ROOT / "data" / "2_forgery_gen"
SPLIT_PATH = EVAL_ROOT / "holdout_split.json"
IMG_EXTS = {".jpg", ".jpeg", ".png"}

# Match the multi-pool layout already used on the GPU host.
TRAIN_JOBS: list[tuple[str, str]] = [
    ("train", "medium"),
    ("train", "easy"),
    ("train", "hard"),
    ("train_v2", "easy"),
    ("train_v2", "medium"),
    ("train_v2", "hard"),
    ("train_v3", "easy"),
    ("train_v3", "medium"),
    ("train_v3", "hard"),
]


def _list_authentic(auth_dir: Path) -> list[Path]:
    files = sorted(
        p for p in auth_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS
    )
    if not files:
        raise SystemExit(f"no authentic images in {auth_dir}")
    return files


def _stage_subset(auth_dir: Path, names: list[str], dest: Path) -> Path:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for name in names:
        src = auth_dir / name
        if not src.is_file():
            raise FileNotFoundError(src)
        target = dest / name
        try:
            target.hardlink_to(src)
        except OSError:
            try:
                target.symlink_to(src.resolve())
            except OSError:
                shutil.copy2(src, target)
    return dest


def write_split(
    auth_dir: Path,
    *,
    train_n: int,
    eval_n: int,
    seed: int,
    out: Path,
) -> dict:
    files = _list_authentic(auth_dir)
    if len(files) < train_n + eval_n:
        raise SystemExit(
            f"need {train_n + eval_n} authentic, found {len(files)} in {auth_dir}"
        )
    names = [p.name for p in files]
    rng = random.Random(seed)
    rng.shuffle(names)
    train_names = sorted(names[:train_n])
    eval_names = sorted(names[train_n : train_n + eval_n])
    train_set, eval_set = set(train_names), set(eval_names)
    if train_set & eval_set:
        raise RuntimeError("train/eval authentic overlap after split")
    payload = {
        "seed": seed,
        "train_n": len(train_names),
        "eval_n": len(eval_names),
        "train_authentic": train_names,
        "eval_authentic": eval_names,
        "note": "document-disjoint; train must not use eval_authentic filenames",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def regenerate_train(train_auth_dir: Path, gen_root: Path, *, no_backup: bool = False) -> list[str]:
    if gen_root.exists():
        if no_backup:
            print(f"removing existing {gen_root}", flush=True)
            shutil.rmtree(gen_root)
        else:
            # Keep a one-shot backup of the previous full-pool gen if present.
            backup = gen_root.parent / "2_forgery_gen_pre_holdout"
            if not backup.exists():
                print(f"backing up {gen_root} -> {backup}", flush=True)
                shutil.move(str(gen_root), str(backup))
            else:
                print(f"removing existing {gen_root}", flush=True)
                shutil.rmtree(gen_root)
    gen_root.mkdir(parents=True)
    folders: list[str] = []
    for profile, difficulty in TRAIN_JOBS:
        if profile not in PROFILES or difficulty not in DIFFICULTY:
            raise SystemExit(f"unknown profile/difficulty: {profile}/{difficulty}")
        records = generate(
            train_auth_dir,
            profile_name=profile,
            difficulty=difficulty,
            count=None,
            output_root=gen_root,
        )
        if difficulty == "medium" and profile in {"train", "test"}:
            folder = profile
        else:
            folder = f"{profile}_{difficulty}"
        folders.append(f"{folder}:{len(records)}")
        print(f"  train pool {folder}: {len(records)}", flush=True)
    return folders


def rebuild_eval(
    eval_auth_dir: Path,
    eval_root: Path,
    *,
    eval_names: list[str],
    no_backup: bool = False,
) -> int:
    """Replace eval images/masks/manifest with held-out authentic + test forgeries."""
    scratch = eval_root / "_holdout_gen_scratch"
    if scratch.exists():
        shutil.rmtree(scratch)
    forged = generate(
        eval_auth_dir,
        profile_name="test",
        difficulty="medium",
        count=None,
        output_root=scratch,
    )
    gen_test = scratch / "test"
    img_dir = eval_root / "images"
    mask_dir = eval_root / "masks"
    # Backup previous eval forgery pixels once (unless --no-backup for seed sweeps).
    if not no_backup:
        backup_img = eval_root / "images_pre_holdout"
        if img_dir.is_dir() and not backup_img.exists():
            print(f"backing up {img_dir} -> {backup_img}", flush=True)
            shutil.move(str(img_dir), str(backup_img))
            mask_backup = eval_root / "masks_pre_holdout"
            if mask_dir.is_dir() and not mask_backup.exists():
                shutil.move(str(mask_dir), str(mask_backup))
    if img_dir.exists():
        shutil.rmtree(img_dir)
    if mask_dir.exists():
        shutil.rmtree(mask_dir)
    img_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)

    records: list[dict] = []
    for i, name in enumerate(eval_names, start=1):
        stem = Path(name).stem
        records.append(
            {
                "id": stem,
                "path": f"authentic/{name}",
                "label": 0,
                "origin": "public_dataset",
                "fabrication": "authentic",
                "eval_domain": "in_domain",
                "generator": "midv_scan",
                "source": "midv2020",
                "holdout": "eval",
            }
        )

    for i, row in enumerate(forged, start=1):
        src = gen_test / row["path"]
        msrc = gen_test / row["mask_path"]
        source_name = row.get("source") or ""
        source_stem = Path(source_name).stem if source_name else f"{i:04d}"
        # Keep a stable id keyed to the authentic document, not pool order alone.
        dst_name = f"fg_{source_stem}{src.suffix.lower()}"
        mask_name = f"fg_{source_stem}_mask.png"
        shutil.copy2(src, img_dir / dst_name)
        shutil.copy2(msrc, mask_dir / mask_name)
        records.append(
            {
                "id": f"fg_{source_stem}",
                "path": f"images/{dst_name}",
                "label": 1,
                "origin": "synthetic_generated",
                "fabrication": "script",
                "tamper": row.get("tamper"),
                "mask_path": f"masks/{mask_name}",
                "eval_domain": "in_domain",
                "generator": "gen_forgery",
                "profile": "test",
                "source": source_name,
                "holdout": "eval",
            }
        )

    manifest = eval_root / "manifest.jsonl"
    backup_manifest = eval_root / "manifest_pre_holdout.jsonl"
    if not no_backup and manifest.is_file() and not backup_manifest.exists():
        shutil.copy2(manifest, backup_manifest)
    with manifest.open("w", encoding="utf-8", newline="\n") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    shutil.rmtree(scratch, ignore_errors=True)
    n0 = sum(1 for r in records if r["label"] == 0)
    n1 = sum(1 for r in records if r["label"] == 1)
    print(f"eval manifest: total={len(records)} label_0={n0} label_1={n1}", flush=True)
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--auth-dir", type=Path, default=AUTH_DIR)
    parser.add_argument("--eval-root", type=Path, default=EVAL_ROOT)
    parser.add_argument("--gen-root", type=Path, default=GEN_ROOT)
    parser.add_argument("--split-path", type=Path, default=SPLIT_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-n", type=int, default=400)
    parser.add_argument("--eval-n", type=int, default=100)
    parser.add_argument(
        "--regenerate-train",
        action="store_true",
        help="wipe/rebuild data/2_forgery_gen from train authentic only",
    )
    parser.add_argument(
        "--rebuild-eval",
        action="store_true",
        help="rebuild 2_forgery images+manifest from eval authentic only",
    )
    parser.add_argument(
        "--write-split-only",
        action="store_true",
        help="only write holdout_split.json (no gen)",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="do not snapshot prior gen/eval dirs (seed sweeps)",
    )
    args = parser.parse_args()

    split = write_split(
        args.auth_dir,
        train_n=args.train_n,
        eval_n=args.eval_n,
        seed=args.seed,
        out=args.split_path,
    )
    print(
        f"wrote {args.split_path} train={split['train_n']} eval={split['eval_n']} seed={split['seed']}",
        flush=True,
    )
    if args.write_split_only:
        return

    stage_root = args.eval_root / "_holdout_stage"
    train_stage = _stage_subset(
        args.auth_dir, split["train_authentic"], stage_root / "train_authentic"
    )
    eval_stage = _stage_subset(
        args.auth_dir, split["eval_authentic"], stage_root / "eval_authentic"
    )
    print(f"staged train={train_stage} eval={eval_stage}", flush=True)

    if args.regenerate_train:
        print("regenerating train forgery pools…", flush=True)
        folders = regenerate_train(train_stage, args.gen_root, no_backup=args.no_backup)
        print(f"train pools: {', '.join(folders)}", flush=True)

    if args.rebuild_eval:
        print("rebuilding eval set…", flush=True)
        rebuild_eval(
            eval_stage,
            args.eval_root,
            eval_names=split["eval_authentic"],
            no_backup=args.no_backup,
        )

    overlap = set(split["train_authentic"]) & set(split["eval_authentic"])
    print(json.dumps({"ok": True, "doc_overlap": len(overlap)}, indent=2))


if __name__ == "__main__":
    main()
