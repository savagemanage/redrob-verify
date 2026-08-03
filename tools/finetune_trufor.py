#!/usr/bin/env python3
"""Fine-tune TruFor on redrob-verify gen_forgery train + MIDV authentic.

Selects checkpoints by TC2/TC3 joint feasibility on the in-domain eval set
(integrity score sweep), falling back to AUC.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
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


@dataclass(frozen=True)
class Sample:
    image: Path
    mask: Path | None  # None => authentic (all-zero mask)
    label: int


def _load_rgb(path: Path, size: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 256.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _load_mask(path: Path | None, size: int) -> torch.Tensor:
    if path is None:
        return torch.zeros(1, size, size, dtype=torch.float32)
    m = Image.open(path).convert("L").resize((size, size), Image.Resampling.NEAREST)
    arr = (np.asarray(m, dtype=np.float32) > 127).astype(np.float32)
    return torch.from_numpy(arr)[None]


class ForgeryFineTuneDS(Dataset):
    def __init__(self, samples: list[Sample], size: int):
        self.samples = samples
        self.size = size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        return _load_rgb(s.image, self.size), _load_mask(s.mask, self.size), torch.tensor(
            s.label, dtype=torch.float32
        )


def build_train_samples(data_root: Path, forged_root: Path) -> list[Sample]:
    authentic_dir = data_root / "2_forgery" / "authentic"
    authentic = sorted(
        p for p in authentic_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    eval_auth: set[Path] = set()
    man = data_root / "2_forgery" / "manifest.jsonl"
    if man.is_file():
        for line in man.read_text().splitlines():
            r = json.loads(line)
            if int(r.get("label", 1)) == 0:
                eval_auth.add((data_root / "2_forgery" / r["path"]).resolve())
    held = [p for p in authentic if p.resolve() not in eval_auth]
    if not held and len(authentic) >= 2:
        rng = random.Random(17101)
        order = authentic[:]
        rng.shuffle(order)
        held = order[: len(order) // 2]
        print(f"WARNING: overlap; using seeded half n={len(held)}", flush=True)
    forged: list[Sample] = []
    for line in (forged_root / "manifest.jsonl").read_text().splitlines():
        r = json.loads(line)
        if int(r.get("label", 1)) != 1:
            continue
        img = forged_root / r["path"]
        mask_path = forged_root / r["mask_path"] if r.get("mask_path") else None
        forged.append(Sample(img, mask_path if mask_path and mask_path.is_file() else None, 1))
    neg = [Sample(p, None, 0) for p in held]
    print(f"train forged={len(forged)} authentic={len(neg)}", flush=True)
    return neg + forged


def build_eval_samples(data_root: Path) -> list[Sample]:
    root = data_root / "2_forgery"
    out: list[Sample] = []
    for line in (root / "manifest.jsonl").read_text().splitlines():
        r = json.loads(line)
        out.append(Sample(root / r["path"], None, int(r["label"])))
    return out


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


def sweep_joint(labels: list[int], scores: list[float], tpr_min: float, f1_min: float):
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
    return dict(infeasible=not feasible, width=width, best_effort=best[1], auc=auc(labels, scores))


@torch.no_grad()
def eval_scores(model, loader, device):
    model.eval()
    labels, scores = [], []
    for rgb, _mask, lab in loader:
        rgb = rgb.to(device, non_blocking=True)
        pred, _conf, det, _npp = model(rgb)
        if det is None:
            loc = F.softmax(pred, dim=1)[:, 1]
            score = loc.flatten(1).max(dim=1).values
        else:
            score = torch.sigmoid(det).view(-1)
        labels.extend(int(x) for x in lab.tolist())
        scores.extend(float(x) for x in score.detach().cpu().tolist())
    return labels, scores


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=REPO / "data")
    ap.add_argument(
        "--train-forged-root", type=Path, default=REPO / "data" / "2_forgery_gen" / "train"
    )
    ap.add_argument(
        "--weights",
        type=Path,
        default=Path.home() / "TruFor/TruFor_train_test/pretrained_models/weights/trufor.pth.tar",
    )
    ap.add_argument(
        "--output", type=Path, default=REPO / "models" / "forgery" / "trufor_finetuned.pth.tar"
    )
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--loc-weight", type=float, default=1.0)
    ap.add_argument("--det-weight", type=float, default=1.0)
    ap.add_argument("--tpr-min", type=float, default=0.88)
    ap.add_argument("--f1-min", type=float, default=0.79)
    ap.add_argument("--eval-every", type=int, default=1)
    ap.add_argument("--max-eval", type=int, default=0, help="0=full eval set")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"device={device} {torch.cuda.get_device_name(0) if device.type == 'cuda' else ''}",
        flush=True,
    )

    class NS:
        pass

    ns = NS()
    ns.opts = []
    update_config(config, ns)

    from models.cmx.builder_np_conf import myEncoderDecoder as confcmx

    model = confcmx(cfg=config)
    ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model = model.to(device)
    print(f"loaded {args.weights}", flush=True)

    train_samples = build_train_samples(args.data_root, args.train_forged_root)
    eval_samples = build_eval_samples(args.data_root)
    if args.max_eval and args.max_eval > 0:
        rng = random.Random(args.seed)
        pos = [s for s in eval_samples if s.label == 1]
        neg = [s for s in eval_samples if s.label == 0]
        n = args.max_eval // 2
        eval_samples = rng.sample(neg, min(n, len(neg))) + rng.sample(pos, min(n, len(pos)))

    train_loader = DataLoader(
        ForgeryFineTuneDS(train_samples, args.image_size),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    eval_loader = DataLoader(
        ForgeryFineTuneDS(eval_samples, args.image_size),
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(args.epochs, 1))

    history: list[dict] = []
    best_key = None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results_path = REPO / "results" / "trufor_finetune.json"

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        t0 = time.perf_counter()
        for rgb, mask, lab in train_loader:
            rgb = rgb.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            lab = lab.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            pred, _conf, det, _npp = model(rgb)
            loc_logits = pred[:, 1:2]
            if loc_logits.shape[-2:] != mask.shape[-2:]:
                loc_logits = F.interpolate(
                    loc_logits, size=mask.shape[-2:], mode="bilinear", align_corners=False
                )
            loc_loss = F.binary_cross_entropy_with_logits(loc_logits, mask)
            if det is None:
                det_loss = loc_loss * 0.0
            else:
                det_loss = F.binary_cross_entropy_with_logits(det.view(-1), lab.view(-1))
            loss = args.loc_weight * loc_loss + args.det_weight * det_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            losses.append(float(loss.detach().item()))
        sched.step()
        train_s = time.perf_counter() - t0
        row: dict = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "train_s": train_s,
            "lr": float(optim.param_groups[0]["lr"]),
        }

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            labels, scores = eval_scores(model, eval_loader, device)
            metrics = sweep_joint(labels, scores, args.tpr_min, args.f1_min)
            row.update(
                {
                    "val_auc": metrics["auc"],
                    "val_infeasible": metrics["infeasible"],
                    "val_width": metrics["width"],
                    "val_best": metrics["best_effort"],
                }
            )
            print(
                f"epoch={epoch} loss={row['loss']:.4f} auc={metrics['auc']:.4f} "
                f"infeas={metrics['infeasible']} width={metrics['width']:.4f} "
                f"best t={metrics['best_effort']['t']:.2f} "
                f"tpr={metrics['best_effort']['tpr']:.3f} f1={metrics['best_effort']['f1']:.3f} "
                f"train_s={train_s:.1f}",
                flush=True,
            )
            key = (
                0 if metrics["infeasible"] else 1,
                metrics["width"],
                metrics["auc"],
                metrics["best_effort"]["f1"],
            )
            if best_key is None or key > best_key:
                best_key = key
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "epoch": epoch,
                        "metrics": metrics,
                        "image_size": args.image_size,
                        "args": {
                            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                        },
                    },
                    args.output,
                )
                print(f"  saved best -> {args.output}", flush=True)
        else:
            print(f"epoch={epoch} loss={row['loss']:.4f} train_s={train_s:.1f}", flush=True)

        history.append(row)
        results_path.write_text(
            json.dumps({"history": history, "best": str(args.output), "best_key": best_key}, indent=2)
            + "\n"
        )

    print(json.dumps({"best": str(args.output), "best_key": best_key}), flush=True)


if __name__ == "__main__":
    main()
