#!/usr/bin/env python3
"""Upload YuNet + SFace ONNX to savagemanage/redrob-verify-face."""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--face-dir", type=Path, default=Path("models/face"))
    ap.add_argument("--repo-id", default="savagemanage/redrob-verify-face")
    ap.add_argument("--model-card", type=Path, default=Path("services/face/MODEL_CARD.md"))
    ap.add_argument("--staging", type=Path, default=Path("results/_hf_face"))
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    yunet = args.face_dir / "face_detection_yunet_2023mar.onnx"
    sface = args.face_dir / "face_recognition_sface_2021dec.onnx"
    for path in (yunet, sface):
        if not path.is_file():
            raise SystemExit(f"missing {path} — run fetch from Zoo first or copy weights")

    if args.staging.exists():
        shutil.rmtree(args.staging)
    args.staging.mkdir(parents=True)
    shutil.copy2(yunet, args.staging / yunet.name)
    shutil.copy2(sface, args.staging / sface.name)
    if args.model_card.is_file():
        shutil.copy2(args.model_card, args.staging / "README.md")
    print("staged", sorted(p.name for p in args.staging.iterdir()))
    if args.dry_run:
        return

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise SystemExit("Set HF_TOKEN before upload")

    from huggingface_hub import HfApi, login

    login(token=token, add_to_git_credential=False)
    api = HfApi(token=token)
    api.create_repo(args.repo_id, private=args.private, exist_ok=True, repo_type="model")
    api.upload_folder(
        folder_path=str(args.staging),
        repo_id=args.repo_id,
        repo_type="model",
        commit_message="Add YuNet + SFace ONNX for redrob-verify face backend",
    )
    print(f"uploaded https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
