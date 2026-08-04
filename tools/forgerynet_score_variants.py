#!/usr/bin/env python3
"""Score ForgeryNet checkpoint with classifier and loc-fusion variants on eval set."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from harness.config_util import load_config, resolve_data_root, resolve_results_root
from harness.sweep import f1, fpr, precision, sweep_thresholds, tpr
from services.forgery.model import ForgeryNet
from services.forgery.train import (
    _auc,
    _cached_tensor,
    _roc_gate,
    _samples_eval,
)


def _score_batch(model: ForgeryNet, images: torch.Tensor) -> dict[str, torch.Tensor]:
    logits, loc = model.forward_features(images)
    det = torch.sigmoid(logits.float())
    out: dict[str, torch.Tensor] = {"det": det}
    if loc is not None:
        # loc is (N,1,h,w) at encoder resolution
        prob = torch.sigmoid(loc.float()).flatten(1)
        out["loc_mean"] = prob.mean(dim=1)
        out["loc_max"] = prob.max(dim=1).values
        k = max(1, prob.shape[1] // 10)
        topk = torch.topk(prob, k=k, dim=1).values.mean(dim=1)
        out["loc_topk10"] = topk
        out["fuse_mean"] = 0.5 * det + 0.5 * out["loc_mean"]
        out["fuse_max"] = 0.5 * det + 0.5 * out["loc_max"]
        out["fuse_topk"] = 0.5 * det + 0.5 * out["loc_topk10"]
        out["fuse_max70"] = 0.3 * det + 0.7 * out["loc_max"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--weights", default="models/forgery/forgerynet_apache.pth")
    ap.add_argument("--image-size", type=int, default=320)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", default="results/forgerynet_score_variants.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_root = resolve_data_root(cfg)
    results_root = resolve_results_root(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    samples = _samples_eval(data_root)
    cache_dir = data_root / "_cache" / f"forgery_{args.image_size}"
    images = torch.stack([_cached_tensor(s.path, args.image_size, cache_dir) for s in samples])
    labels_t = torch.tensor([s.label for s in samples], dtype=torch.float32)
    loader = DataLoader(TensorDataset(images, labels_t), batch_size=args.batch_size, shuffle=False)

    ckpt = torch.load(args.weights, map_location=device, weights_only=True)
    state = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt
    model = ForgeryNet(imagenet=False, loc_head=True).to(device).eval()
    model.load_state_dict(state)

    buckets: dict[str, list[float]] = {}
    labels: list[int] = []
    with torch.inference_mode():
        for batch_images, batch_labels in loader:
            batch_images = batch_images.to(device, non_blocking=True)
            scored = _score_batch(model, batch_images)
            for name, tensor in scored.items():
                buckets.setdefault(name, []).extend(tensor.cpu().tolist())
            labels.extend(batch_labels.int().tolist())

    tpr_min = float(cfg["targets"]["tc2_tpr_min"])
    f1_min = float(cfg["targets"]["tc3_f1_min"])
    summary = {
        "weights": args.weights,
        "n": len(labels),
        "image_size": args.image_size,
        "variants": {},
    }
    for name, scores in buckets.items():
        sweep = sweep_thresholds(
            scores,
            labels,
            {"tpr": tpr, "fpr": fpr, "precision": precision, "f1": f1},
            {"tpr": tpr_min, "f1": f1_min},
        )
        width = 0.0 if sweep.infeasible else float(sweep.t_max - sweep.t_min)  # type: ignore[operator]
        gate = _roc_gate(scores, labels)
        pos = [s for s, y in zip(scores, labels) if y == 1]
        neg = [s for s, y in zip(scores, labels) if y == 0]
        row = {
            "auc": _auc(labels, scores),
            "mean_pos": float(np.mean(pos)),
            "mean_neg": float(np.mean(neg)),
            "roc_gate": gate,
            "joint_interval": None if sweep.infeasible else [sweep.t_min, sweep.t_max],
            "joint_interval_width": width,
            "recommended": sweep.recommended,
            "recommended_metrics": sweep.recommended_metrics,
        }
        summary["variants"][name] = row
        print(
            f"{name:12} auc={row['auc']:.4f} joint_w={width:.4f} "
            f"roc={gate['passed']} fpr@tpr088={gate['fpr']:.3f} "
            f"pos={row['mean_pos']:.3f} neg={row['mean_neg']:.3f}",
            flush=True,
        )

    out = Path(args.out)
    if not out.is_absolute():
        out = results_root / out.name if out.parent == Path("results") else Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
