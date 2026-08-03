#!/usr/bin/env python3
"""Eval TruFor checkpoint on 2_forgery with TC2/TC3 sweep."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parents[1]
TRUFOR_SRC = Path.home() / "TruFor" / "test_docker" / "src"
if str(TRUFOR_SRC) not in sys.path:
    sys.path.insert(0, str(TRUFOR_SRC))

from config import _C as config  # noqa: E402
from config import update_config  # noqa: E402


class SampleDS(Dataset):
    def __init__(self, items: list[tuple[Path, int]], size: int):
        self.items = items
        self.size = size

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        path, label = self.items[i]
        img = Image.open(path).convert("RGB")
        if self.size > 0:
            img = img.resize((self.size, self.size), Image.Resampling.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 256.0
        t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        return t, torch.tensor(label, dtype=torch.float32), str(path)


def auc(labels: list[int], scores: list[float]) -> float:
    pos = sum(labels)
    neg = len(labels) - pos
    if not pos or not neg:
        return 0.0
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and scores[order[end]] == scores[order[start]]:
            end += 1
        avg = (start + end + 1) / 2.0
        for k in range(start, end):
            ranks[order[k]] = avg
        start = end
    return (sum(ranks[i] for i, lab in enumerate(labels) if lab == 1) - pos * (pos + 1) / 2) / (
        pos * neg
    )


def sweep(labels: list[int], scores: list[float], tpr_min: float = 0.88, f1_min: float = 0.79):
    best = None
    feasible = []
    for ti in range(0, 101):
        t = ti / 100.0
        tp = sum(1 for lab, s in zip(labels, scores) if s >= t and lab == 1)
        fp = sum(1 for lab, s in zip(labels, scores) if s >= t and lab == 0)
        fn = sum(1 for lab, s in zip(labels, scores) if s < t and lab == 1)
        tn = sum(1 for lab, s in zip(labels, scores) if s < t and lab == 0)
        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        prec = tp / max(tp + fp, 1)
        f1 = 0.0 if prec + tpr == 0 else 2 * prec * tpr / (prec + tpr)
        row = dict(t=t, tpr=tpr, fpr=fpr, precision=prec, f1=f1)
        if tpr >= tpr_min and f1 >= f1_min:
            feasible.append(row)
        key = (
            1 if (tpr >= tpr_min and f1 >= f1_min) else 0,
            min(tpr / tpr_min, f1 / f1_min if f1_min else 0),
            f1,
            tpr,
        )
        if best is None or key > best[0]:
            best = (key, row)
    width = 0.0
    if feasible:
        width = max(r["t"] for r in feasible) - min(r["t"] for r in feasible)
    at88 = None
    for ti in range(0, 101):
        t = ti / 100.0
        tp = sum(1 for lab, s in zip(labels, scores) if s >= t and lab == 1)
        fp = sum(1 for lab, s in zip(labels, scores) if s >= t and lab == 0)
        fn = sum(1 for lab, s in zip(labels, scores) if s < t and lab == 1)
        tn = sum(1 for lab, s in zip(labels, scores) if s < t and lab == 0)
        tpr = tp / max(tp + fn, 1)
        if tpr >= 0.88:
            fpr = fp / max(fp + tn, 1)
            prec = tp / max(tp + fp, 1)
            f1 = 0.0 if prec + tpr == 0 else 2 * prec * tpr / (prec + tpr)
            at88 = dict(t=t, tpr=tpr, fpr=fpr, precision=prec, f1=f1)
            break
    return dict(
        infeasible=not feasible,
        width=width,
        best_effort=best[1],
        at_tpr88=at88,
        auc=auc(labels, scores),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--data-root", type=Path, default=REPO / "data")
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--max-eval", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    class NS:
        pass

    ns = NS()
    ns.opts = []
    update_config(config, ns)

    from models.cmx.builder_np_conf import myEncoderDecoder as confcmx

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = confcmx(cfg=config)
    ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(sd, strict=True)
    model = model.to(device).eval()
    epoch = ckpt.get("epoch") if isinstance(ckpt, dict) else None
    print(f"loaded {args.weights} epoch={epoch}", flush=True)

    root = args.data_root / "2_forgery"
    items: list[tuple[Path, int]] = []
    for line in (root / "manifest.jsonl").read_text().splitlines():
        r = json.loads(line)
        items.append((root / r["path"], int(r["label"])))
    if args.max_eval and args.max_eval > 0:
        rng = random.Random(args.seed)
        pos = [x for x in items if x[1] == 1]
        neg = [x for x in items if x[1] == 0]
        n = args.max_eval // 2
        items = rng.sample(neg, min(n, len(neg))) + rng.sample(pos, min(n, len(pos)))
    print(f"n={len(items)} pos={sum(1 for _, lab in items if lab == 1)} size={args.image_size}", flush=True)

    loader = DataLoader(SampleDS(items, args.image_size), batch_size=1, shuffle=False, num_workers=0)
    labels: list[int] = []
    scores: list[float] = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for i, (rgb, lab, _path) in enumerate(loader):
            rgb = rgb.to(device)
            pred, _conf, det, _npp = model(rgb)
            if det is None:
                loc = F.softmax(pred, dim=1)[:, 1]
                score = loc.flatten(1).max(dim=1).values
            else:
                score = torch.sigmoid(det).view(-1)
            labels.append(int(lab.item()))
            scores.append(float(score.item()))
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(items)}", flush=True)
    elapsed = time.perf_counter() - t0
    metrics = sweep(labels, scores)
    pos_s = [s for lab, s in zip(labels, scores) if lab == 1]
    neg_s = [s for lab, s in zip(labels, scores) if lab == 0]
    out = dict(
        tag=args.tag,
        weights=str(args.weights),
        epoch=epoch,
        n=len(labels),
        image_size=args.image_size,
        max_eval=args.max_eval,
        seed=args.seed,
        elapsed_s=elapsed,
        mean_pos=float(np.mean(pos_s)),
        mean_neg=float(np.mean(neg_s)),
        **metrics,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in out if k != "weights"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
