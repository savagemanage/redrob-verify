"""Train the forgery multi-stream model; select checkpoints by TC2/TC3 interval width.

Best checkpoint = max joint feasible interval width (NOT accuracy).
Per-epoch logs: AUC, ROC gate (TPR≥0.88 ∧ FPR≤0.348), joint interval width.

Speed: MIDV scans are ~2480×3507. Decoding them every epoch is the bottleneck
(~2s/batch). Resized 224² tensors are disk-cached once and kept in RAM.
FFT/HOG stay batched torch ops on GPU (~3ms each); they are not per-image
Python loops and are cheap once inputs are 224².
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from harness.config_util import REPO_ROOT, load_config, resolve_data_root, resolve_results_root
from harness.schema import load_forgery_manifest
from harness.sweep import f1, precision, sweep_thresholds, tpr
from services.forgery.model import ForgeryNet

# ROC gate from work order (aligned with OPERATING_POINTS first row)
ROC_GATE_TPR = 0.88
ROC_GATE_FPR = 0.348


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int
    mask_path: Path | None = None
    # If set, decode via in-memory JPEG recompress so train auth ≠ eval pixels.
    jpeg_quality: int | None = None


def _cache_key(path: Path, image_size: int, jpeg_quality: int | None = None) -> str:
    stat = path.stat()
    payload = f"{path.resolve()}|{stat.st_mtime_ns}|{stat.st_size}|{image_size}|jq={jpeg_quality}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _load_resized_rgb(path: Path, image_size: int, jpeg_quality: int | None = None) -> Tensor:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        if jpeg_quality is not None:
            import io

            buf = io.BytesIO()
            rgb.save(buf, format="JPEG", quality=int(jpeg_quality))
            buf.seek(0)
            rgb = Image.open(buf).convert("RGB")
        pixels = np.asarray(
            rgb.resize((image_size, image_size), Image.Resampling.BILINEAR),
            dtype=np.float32,
        ) / 255.0
    return torch.from_numpy(pixels).permute(2, 0, 1).contiguous()


def _load_resized_mask(path: Path | None, image_size: int) -> Tensor:
    if path is None or not path.is_file():
        return torch.zeros(1, image_size, image_size, dtype=torch.float32)
    with Image.open(path) as image:
        arr = np.asarray(
            image.convert("L").resize((image_size, image_size), Image.Resampling.NEAREST),
            dtype=np.float32,
        )
    return torch.from_numpy((arr > 127).astype(np.float32))[None]


def _cached_tensor(
    path: Path, image_size: int, cache_dir: Path, jpeg_quality: int | None = None
) -> Tensor:
    """Load resized CHW float32 from disk cache, building it on first access."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{_cache_key(path, image_size, jpeg_quality)}.pt"
    if cache_path.is_file():
        return torch.load(cache_path, map_location="cpu", weights_only=True)
    tensor = _load_resized_rgb(path, image_size, jpeg_quality=jpeg_quality)
    torch.save(tensor, cache_path)
    return tensor


def _dice_loss(logits: Tensor, targets: Tensor, eps: float = 1.0) -> Tensor:
    probs = torch.sigmoid(logits.float())
    dims = tuple(range(1, probs.ndim))
    inter = (probs * targets).sum(dim=dims)
    union = probs.sum(dim=dims) + targets.sum(dim=dims)
    dice = (2.0 * inter + eps) / (union + eps)
    return 1.0 - dice.mean()


def _materialize(
    samples: list[Sample],
    image_size: int,
    cache_dir: Path,
    *,
    label: str,
    device: torch.device,
    gpu_resident: bool,
    with_masks: bool,
) -> TensorDataset:
    """Decode once (or hit cache); optionally pin the full split on the GPU."""
    images: list[Tensor] = []
    masks: list[Tensor] = []
    labels: list[int] = []
    t0 = time.perf_counter()
    hits = 0
    for index, sample in enumerate(samples, start=1):
        cache_path = cache_dir / f"{_cache_key(sample.path, image_size, sample.jpeg_quality)}.pt"
        if cache_path.is_file():
            hits += 1
        images.append(
            _cached_tensor(sample.path, image_size, cache_dir, jpeg_quality=sample.jpeg_quality)
        )
        if with_masks:
            masks.append(_load_resized_mask(sample.mask_path, image_size))
        labels.append(sample.label)
        if index % 200 == 0 or index == len(samples):
            print(f"  cache[{label}] {index}/{len(samples)} hits={hits}", flush=True)
    stacked = torch.stack(images)
    label_tensor = torch.tensor(labels, dtype=torch.float32)
    if with_masks:
        mask_tensor = torch.stack(masks)
        if gpu_resident and device.type == "cuda":
            stacked = stacked.to(device, non_blocking=True)
            mask_tensor = mask_tensor.to(device, non_blocking=True)
            label_tensor = label_tensor.to(device, non_blocking=True)
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        print(
            f"  cache[{label}] done n={len(samples)} hits={hits} "
            f"shape={tuple(stacked.shape)} masks={tuple(mask_tensor.shape)} "
            f"gpu_resident={gpu_resident} s={elapsed:.1f}",
            flush=True,
        )
        return TensorDataset(stacked, mask_tensor, label_tensor)
    if gpu_resident and device.type == "cuda":
        stacked = stacked.to(device, non_blocking=True)
        label_tensor = label_tensor.to(device, non_blocking=True)
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    print(
        f"  cache[{label}] done n={len(samples)} hits={hits} "
        f"shape={tuple(stacked.shape)} gpu_resident={gpu_resident} s={elapsed:.1f}",
        flush=True,
    )
    return TensorDataset(stacked, label_tensor)


def _auc(labels: list[int], scores: list[float]) -> float:
    """Rank AUC with tied-score averaging, avoiding an additional sklearn dependency."""
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return 0.0
    order = sorted(range(len(scores)), key=lambda index: scores[index])
    ranks = [0.0] * len(scores)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and scores[order[end]] == scores[order[start]]:
            end += 1
        mean_rank = (start + 1 + end) / 2.0
        for position in range(start, end):
            ranks[order[position]] = mean_rank
        start = end
    return (
        sum(rank for rank, label in zip(ranks, labels, strict=True) if label) - positives * (positives + 1) / 2
    ) / (positives * negatives)


def _fpr(scores: list[float], labels: list[int], threshold: float) -> float:
    fp = sum(score >= threshold and label == 0 for score, label in zip(scores, labels, strict=True))
    negatives = sum(label == 0 for label in labels)
    return fp / negatives if negatives else 0.0


def _roc_gate(scores: list[float], labels: list[int]) -> dict[str, float | bool | None]:
    """True if some threshold satisfies TPR≥0.88 and FPR≤0.348 simultaneously."""
    thresholds = sorted(set(scores), reverse=True)
    best: dict[str, float | bool | None] = {
        "passed": False,
        "threshold": None,
        "tpr": 0.0,
        "fpr": 1.0,
    }
    for threshold in thresholds:
        observed_tpr = tpr(scores, labels, threshold)
        observed_fpr = _fpr(scores, labels, threshold)
        if observed_tpr >= ROC_GATE_TPR and observed_fpr <= ROC_GATE_FPR:
            return {
                "passed": True,
                "threshold": threshold,
                "tpr": observed_tpr,
                "fpr": observed_fpr,
            }
        if observed_tpr >= ROC_GATE_TPR and observed_fpr < float(best["fpr"]):  # type: ignore[arg-type]
            best = {
                "passed": False,
                "threshold": threshold,
                "tpr": observed_tpr,
                "fpr": observed_fpr,
            }
    return best


def _samples_train(
    data_root: Path,
    train_forged_roots: list[Path],
    *,
    auth_jpeg_quality: int = 72,
) -> list[Sample]:
    """Authentic = JPEG-recompressed MIDV scans; forged = gen_forgery train pools.

    Eval authentic stay pristine originals. Train uses JPEG recompress so negatives
    are not pixel-identical to the eval set (full authentic pool overlaps eval).
    """
    authentic_dir = data_root / "2_forgery" / "authentic"
    authentic = [
        Sample(path, 0, None, jpeg_quality=auth_jpeg_quality)
        for path in sorted(authentic_dir.glob("*"))
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    forged: list[Sample] = []
    for train_forged_root in train_forged_roots:
        forged_records = load_forgery_manifest(train_forged_root)
        forged.extend(
            Sample(
                train_forged_root / record.path,
                1,
                (train_forged_root / record.mask_path) if record.mask_path else None,
            )
            for record in forged_records
            if record.label == 1
        )
    if not authentic:
        raise RuntimeError(f"no authentic images in {authentic_dir}")
    if not forged:
        roots = ", ".join(str(r) for r in train_forged_roots)
        raise RuntimeError(
            f"no train forgeries in [{roots}]; run: "
            "uv run python -m tools.gen_forgery --profile train --output-root data/2_forgery_gen"
        )
    print(
        f"train authentic={len(authentic)} (jpeg_q={auth_jpeg_quality}) forged={len(forged)} "
        f"roots={len(train_forged_roots)}",
        flush=True,
    )
    return authentic + forged


def _resolve_train_forged_roots(data_root: Path, explicit: list[Path] | None) -> list[Path]:
    if explicit:
        return explicit
    pool = data_root / "2_forgery_gen"
    if not pool.is_dir():
        return [data_root / "2_forgery_gen" / "train"]
    roots = sorted(
        p for p in pool.iterdir() if p.is_dir() and (p / "manifest.jsonl").is_file() and p.name != "test"
    )
    return roots or [pool / "train"]


def _samples_eval(data_root: Path) -> list[Sample]:
    eval_root = data_root / "2_forgery"
    return [
        Sample(eval_root / record.path, int(record.label), None)
        for record in load_forgery_manifest(eval_root)
    ]


def _resolve_device(*, allow_cpu: bool) -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"device={device} name={torch.cuda.get_device_name(0)}", flush=True)
        return device
    message = (
        "CUDA unavailable — forgery training requires GPU. "
        "Host: install torch with CUDA. "
        "Container: set gpus: all and use a CUDA base image."
    )
    if allow_cpu:
        print(f"WARNING: {message} Continuing on CPU because --allow-cpu.", flush=True)
        return torch.device("cpu")
    raise RuntimeError(message)


def _evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device, *, use_amp: bool
) -> tuple[list[int], list[float]]:
    labels: list[int] = []
    scores: list[float] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            images = batch[0].to(device, non_blocking=True)
            batch_labels = batch[-1]
            with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                logits = model(images)
            scores.extend(torch.sigmoid(logits.float()).cpu().tolist())
            labels.extend(batch_labels.int().tolist())
    return labels, scores


def _profile_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    *,
    use_amp: bool,
) -> dict[str, float]:
    """Time data / aux / forward / backward for one train+val pass. Numbers, not guesses."""
    from torch.profiler import ProfilerActivity, profile, record_function

    model.train()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")
    data_s = forward_s = backward_s = aux_s = opt_s = 0.0
    steps = 0
    if device.type == "cuda":
        torch.cuda.synchronize()
    epoch_t0 = time.perf_counter()
    for batch in train_loader:
        t_data = time.perf_counter()
        images = batch[0].to(device, non_blocking=True)
        labels = batch[-1].to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        data_s += time.perf_counter() - t_data

        optimizer.zero_grad(set_to_none=True)

        t_aux = time.perf_counter()
        with torch.no_grad():
            _ = model.fft_magnitude(images)
            _ = model.hog(images)
        if device.type == "cuda":
            torch.cuda.synchronize()
        aux_s += time.perf_counter() - t_aux

        t_fwd = time.perf_counter()
        with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
            logits = model(images)
            loss = loss_fn(logits, labels)
        if device.type == "cuda":
            torch.cuda.synchronize()
        forward_s += time.perf_counter() - t_fwd

        t_bwd = time.perf_counter()
        scaler.scale(loss).backward()
        if device.type == "cuda":
            torch.cuda.synchronize()
        backward_s += time.perf_counter() - t_bwd

        t_opt = time.perf_counter()
        scaler.step(optimizer)
        scaler.update()
        if device.type == "cuda":
            torch.cuda.synchronize()
        opt_s += time.perf_counter() - t_opt
        steps += 1

    t_val = time.perf_counter()
    _evaluate(model, validation_loader, device, use_amp=use_amp)
    if device.type == "cuda":
        torch.cuda.synchronize()
    val_s = time.perf_counter() - t_val
    total_s = time.perf_counter() - epoch_t0

    model.train()
    batch_images, batch_labels = next(iter(train_loader))
    batch_images = batch_images.to(device)
    batch_labels = batch_labels.to(device)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False) as prof:
        for _ in range(3):
            with record_function("train_step"):
                optimizer.zero_grad(set_to_none=True)
                with record_function("forward"):
                    with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                        logits = model(batch_images)
                        loss = loss_fn(logits, batch_labels)
                with record_function("backward"):
                    scaler.scale(loss).backward()
                with record_function("optimizer"):
                    scaler.step(optimizer)
                    scaler.update()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20), flush=True)

    breakdown = {
        "steps": float(steps),
        "data_s": data_s,
        "aux_fft_hog_s": aux_s,
        "forward_s": forward_s,
        "backward_s": backward_s,
        "optimizer_s": opt_s,
        "val_s": val_s,
        "train_plus_val_s": total_s,
        "ms_per_step": (total_s - val_s) / max(steps, 1) * 1000.0,
    }
    print("profile_epoch " + json.dumps(breakdown), flush=True)
    return breakdown


def train(args: argparse.Namespace) -> dict:
    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = _resolve_device(allow_cpu=bool(args.allow_cpu))
    use_amp = bool(args.amp) and device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.use_deterministic_algorithms(False)

    data_root = resolve_data_root(cfg)
    explicit_roots: list[Path] | None = None
    if getattr(args, "train_forged_roots", None):
        explicit_roots = [Path(p) for p in args.train_forged_roots]
    elif args.train_forged_root:
        explicit_roots = [Path(args.train_forged_root)]
    train_forged_roots = _resolve_train_forged_roots(data_root, explicit_roots)
    auth_jpeg_quality = int(getattr(args, "auth_jpeg_quality", 72))
    train_samples = _samples_train(
        data_root, train_forged_roots, auth_jpeg_quality=auth_jpeg_quality
    )
    validation_samples = _samples_eval(data_root)
    if args.max_samples is not None and int(args.max_samples) > 0:
        rng = random.Random(seed)
        rng.shuffle(train_samples)
        train_samples = train_samples[: int(args.max_samples)]

    print(
        f"train n={len(train_samples)} "
        f"(pos={sum(s.label for s in train_samples)} neg={sum(1 - s.label for s in train_samples)}) "
        f"val n={len(validation_samples)} device={device} amp={use_amp}",
        flush=True,
    )
    if len({s.label for s in train_samples}) < 2:
        raise RuntimeError("train set has a single label - check data loader / paths")

    cache_dir = Path(args.cache_dir) if args.cache_dir else data_root / "_cache" / f"forgery_{args.image_size}"
    print(f"image_cache={cache_dir}", flush=True)
    gpu_resident = bool(args.gpu_resident) and device.type == "cuda"
    loc_weight = float(getattr(args, "loc_weight", 1.0))
    use_masks = loc_weight > 0.0
    train_ds = _materialize(
        train_samples,
        args.image_size,
        cache_dir,
        label="train",
        device=device,
        gpu_resident=gpu_resident,
        with_masks=use_masks,
    )
    val_ds = _materialize(
        validation_samples,
        args.image_size,
        cache_dir,
        label="val",
        device=device,
        gpu_resident=gpu_resident,
        with_masks=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda" and not gpu_resident,
    )
    validation_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda" and not gpu_resident,
    )
    model = ForgeryNet(imagenet=bool(args.imagenet), loc_head=use_masks).to(device)
    n_pos = sum(1 for s in train_samples if s.label == 1)
    n_neg = len(train_samples) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device, dtype=torch.float32)
    print(
        f"imagenet={bool(args.imagenet)} loc_weight={loc_weight} "
        f"pos_weight={float(pos_weight):.3f} (neg={n_neg} pos={n_pos})",
        flush=True,
    )
    try:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, fused=device.type == "cuda"
        )
    except TypeError:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    det_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loc_loss_fn = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    best_width, best_auc, history = -1.0, -1.0, []
    collapse_streak = 0
    output_path = Path(args.output or (REPO_ROOT / "models" / "forgery" / "forgerynet_apache.pth"))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    profile_row: dict[str, float] | None = None
    if args.profile_one_epoch:
        profile_row = _profile_one_epoch(
            model, optimizer, det_loss_fn, train_loader, validation_loader, device, use_amp=use_amp
        )

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        epoch_t0 = time.perf_counter()
        for batch in train_loader:
            if use_masks:
                images, masks, labels = batch
                masks = masks.to(device, non_blocking=True)
            else:
                images, labels = batch
                masks = None
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if args.augment:
                # Geometry / photometric noise that keeps FFT/HOG meaningful.
                if torch.rand(1, device=device).item() < 0.5:
                    images = torch.flip(images, dims=(-1,))
                    if masks is not None:
                        masks = torch.flip(masks, dims=(-1,))
                if torch.rand(1, device=device).item() < 0.5:
                    images = torch.flip(images, dims=(-2,))
                    if masks is not None:
                        masks = torch.flip(masks, dims=(-2,))
                # Brightness / contrast jitter (broadcast over NCHW).
                bright = 1.0 + 0.15 * (torch.rand(images.size(0), 1, 1, 1, device=device) * 2 - 1)
                contrast = 1.0 + 0.15 * (torch.rand(images.size(0), 1, 1, 1, device=device) * 2 - 1)
                mean = images.mean(dim=(-2, -1), keepdim=True)
                images = (mean + (images - mean) * contrast) * bright
                images = (images + 0.02 * torch.randn_like(images)).clamp(0.0, 1.0)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits, loc_logits = model.forward_features(images)
                det_loss = det_loss_fn(logits, labels)
                if use_masks and loc_logits is not None and masks is not None:
                    if loc_logits.shape[-2:] != masks.shape[-2:]:
                        loc_logits = F.interpolate(
                            loc_logits, size=masks.shape[-2:], mode="bilinear", align_corners=False
                        )
                    loc_bce = loc_loss_fn(loc_logits, masks)
                    loc_dice = _dice_loss(loc_logits, masks)
                    loc_loss = 0.5 * loc_bce + 0.5 * loc_dice
                else:
                    loc_loss = det_loss * 0.0
                loss = det_loss + loc_weight * loc_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().float().item()))
        scheduler.step()
        labels, scores = _evaluate(model, validation_loader, device, use_amp=use_amp)
        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_s = time.perf_counter() - epoch_t0
        score_std = float(np.std(scores)) if scores else 0.0
        if score_std < 1e-6:
            collapse_streak += 1
            print(
                f"WARNING epoch={epoch}: constant scores~={scores[0] if scores else 'n/a'} "
                f"(streak={collapse_streak})",
                flush=True,
            )
        else:
            collapse_streak = 0
        sweep = sweep_thresholds(
            scores,
            labels,
            {"tpr": tpr, "precision": precision, "f1": f1},
            {"tpr": float(cfg["targets"]["tc2_tpr_min"]), "f1": float(cfg["targets"]["tc3_f1_min"])},
        )
        width = 0.0 if sweep.infeasible else float(sweep.t_max - sweep.t_min)  # type: ignore[operator]
        gate = _roc_gate(scores, labels)
        auc = _auc(labels, scores)
        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)) if losses else None,
            "auc": auc,
            "score_std": score_std,
            "roc_gate": gate,
            "joint_interval": None if sweep.infeasible else [sweep.t_min, sweep.t_max],
            "joint_interval_width": width,
            "epoch_seconds": epoch_s,
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        print(
            f"epoch={epoch} loss={row['loss']:.4f} auc={auc:.4f} "
            f"roc_gate={gate['passed']} (tpr={gate['tpr']:.4f} fpr={gate['fpr']:.4f}) "
            f"joint_width={width:.4f} epoch_s={epoch_s:.2f} lr={row['lr']:.2e}",
            flush=True,
        )
        if width > best_width or (width == best_width and auc > best_auc):
            best_width = width
            best_auc = auc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "joint_interval_width": width,
                    "auc": auc,
                    "roc_gate": gate,
                    "image_size": args.image_size,
                    "imagenet": bool(args.imagenet),
                },
                output_path,
            )
            print(f"  saved best -> {output_path}", flush=True)
        if collapse_streak >= args.collapse_patience:
            print(f"early stop: score collapse for {collapse_streak} epochs", flush=True)
            break

    result = {
        "device": device.type,
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "best_checkpoint": str(output_path),
        "best_joint_interval_width": best_width,
        "best_auc": best_auc,
        "train_n": len(train_samples),
        "val_n": len(validation_samples),
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "amp": use_amp,
        "gpu_resident": gpu_resident,
        "imagenet": bool(args.imagenet),
        "learning_rate": args.learning_rate,
        "cache_dir": str(cache_dir),
        "profile_one_epoch": profile_row,
        "epochs": history,
    }
    results = resolve_results_root(cfg)
    results.mkdir(parents=True, exist_ok=True)
    (results / "forgery_train.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the forgery multi-stream model")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=320, help="must match serving image_size")
    parser.add_argument("--max-samples", type=int, default=None, help="cap train set (debug only)")
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--collapse-patience", type=int, default=2)
    parser.add_argument("--loc-weight", type=float, default=2.0, help="mask BCE+Dice weight (0 disables loc head)")
    parser.add_argument("--auth-jpeg-quality", type=int, default=72, help="JPEG recompress quality for train authentic")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--train-forged-root",
        type=Path,
        default=None,
        help="single train-profile gen_forgery dir (default: auto-discover data/2_forgery_gen/*)",
    )
    parser.add_argument(
        "--train-forged-roots",
        type=Path,
        nargs="*",
        default=None,
        help="explicit list of forged train dirs (overrides auto-discover)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="resized-tensor cache (default: data/_cache/forgery_<size>)",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="do not abort when CUDA is unavailable (debug only)",
    )
    parser.add_argument(
        "--profile-one-epoch",
        action="store_true",
        help="run timed train+val pass + torch.profiler before normal epochs",
    )
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True, help="CUDA autocast FP16")
    parser.add_argument(
        "--gpu-resident",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="keep cached 224 tensors on GPU (uses ~1GB VRAM; off by default on 8GB laptops)",
    )
    parser.add_argument(
        "--imagenet",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="initialize ResNet-50 from ImageNet (not a forgery detector checkpoint)",
    )
    parser.add_argument(
        "--augment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="light train-time flip + noise",
    )
    result = train(parser.parse_args())
    print(
        json.dumps(
            {
                "device": result["device"],
                "best_joint_interval_width": result["best_joint_interval_width"],
                "best_auc": result.get("best_auc"),
                "best_checkpoint": result["best_checkpoint"],
                "profile_one_epoch": result.get("profile_one_epoch"),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
