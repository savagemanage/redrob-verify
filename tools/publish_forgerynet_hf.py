#!/usr/bin/env python3
"""Package and upload ForgeryNet weights to Hugging Face Hub.

Usage:
  export HF_TOKEN=hf_...
  python tools/publish_forgerynet_hf.py \\
    --weights models/forgery/forgerynet_apache.pth \\
    --repo-id savagemanage/forgerynet-apache \\
    --private   # optional

Creates a staging dir with README.md (model card), config.json, and
model.safetensors (state_dict only), then creates/uploads the Hub repo.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument(
        "--repo-id",
        default="savagemanage/redrob-verify-forgery",
        help="Hub repo id (default: savagemanage/redrob-verify-forgery)",
    )
    ap.add_argument(
        "--staging",
        type=Path,
        default=None,
        help="staging directory (default: results/_hf_forgerynet)",
    )
    ap.add_argument("--model-card", type=Path, default=Path("services/forgery/MODEL_CARD.md"))
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="build staging only, do not upload")
    ap.add_argument("--revision", default="main")
    args = ap.parse_args()

    try:
        import torch
        from huggingface_hub import HfApi, login
        from safetensors.torch import save_file
    except ImportError as exc:
        raise SystemExit(
            "Need torch, huggingface_hub, safetensors. "
            "Try: uv sync --extra forgery && uv pip install huggingface_hub safetensors"
        ) from exc

    weights = args.weights
    if not weights.is_file():
        raise SystemExit(f"weights not found: {weights}")

    staging = args.staging or Path("results/_hf_forgerynet")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    ckpt = torch.load(weights, map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state = ckpt["model_state"]
        meta = {k: ckpt[k] for k in ckpt if k != "model_state"}
    else:
        state = ckpt
        meta = {}

    # HF prefers contiguous CPU tensors
    state = {k: v.detach().cpu().contiguous() for k, v in state.items()}
    save_file(state, staging / "model.safetensors")
    # Also ship the native training checkpoint for drop-in config.yaml paths.
    shutil.copy2(weights, staging / "forgerynet_apache.pth")

    config = {
        "architectures": ["ForgeryNet"],
        "model_type": "forgery_net",
        "image_size": int(meta.get("image_size") or 320),
        "loc_head": True,
        "backbone": "resnet50",
        "backbone_init": "torchvision.ResNet50_Weights.IMAGENET1K_V2",
        "num_labels": 1,
        "id2label": {"0": "authentic", "1": "forged"},
        "label2id": {"authentic": 0, "forged": 1},
        "recommended_threshold": 0.45,
        "train_meta": {
            k: (float(v) if isinstance(v, (float,)) else v)
            for k, v in meta.items()
            if k in {"epoch", "auc", "joint_interval_width", "image_size", "imagenet", "roc_gate"}
        },
        "code_repository": "https://github.com/savagemanage/redrob-verify",
        "code_entry": "services/forgery/model.py",
        "license": "apache-2.0",
    }
    # Make JSON-safe
    def _sanitize(obj):
        if isinstance(obj, dict):
            return {str(k): _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_sanitize(v) for v in obj]
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        return str(obj)

    (staging / "config.json").write_text(json.dumps(_sanitize(config), indent=2) + "\n", encoding="utf-8")

    card = args.model_card
    if card.is_file():
        shutil.copy2(card, staging / "README.md")
    else:
        (staging / "README.md").write_text("# ForgeryNet\n", encoding="utf-8")

    # Tiny usage hint for Hub visitors who clone weights only
    (staging / "load_example.py").write_text(
        '''"""Load ForgeryNet weights from this Hub repo (requires redrob-verify code)."""
from pathlib import Path
import torch
from safetensors.torch import load_file

# pip/uv install from https://github.com/savagemanage/redrob-verify
from services.forgery.model import ForgeryNet

def load(repo_dir: str | Path = ".") -> ForgeryNet:
    cfg = __import__("json").loads(Path(repo_dir, "config.json").read_text())
    model = ForgeryNet(imagenet=False, loc_head=True)
    state = load_file(Path(repo_dir) / "model.safetensors")
    model.load_state_dict(state)
    model.eval()
    print("image_size", cfg.get("image_size"), "threshold", cfg.get("recommended_threshold"))
    return model

if __name__ == "__main__":
    load()
''',
        encoding="utf-8",
    )

    print(f"staged -> {staging}")
    print(" files:", sorted(p.name for p in staging.iterdir()))
    print(" params:", sum(v.numel() for v in state.values()))

    if args.dry_run:
        print("dry-run: skip upload")
        return

    import os

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise SystemExit("Set HF_TOKEN (or HUGGING_FACE_HUB_TOKEN) before upload")

    login(token=token, add_to_git_credential=False)
    api = HfApi(token=token)
    api.create_repo(args.repo_id, private=args.private, exist_ok=True, repo_type="model")
    api.upload_folder(
        folder_path=str(staging),
        repo_id=args.repo_id,
        repo_type="model",
        revision=args.revision,
        commit_message="Add ForgeryNet Apache checkpoint (safetensors + model card)",
    )
    print(f"uploaded https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
