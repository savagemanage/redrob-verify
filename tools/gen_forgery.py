"""Generate synthetic document forgeries for DS-2 training / in-domain eval.

Writes under ``output_root/<profile>/`` (default ``results/_tmp_forgery``).
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from harness.config_util import REPO_ROOT

DEFAULT_OUTPUT = REPO_ROOT / "results" / "_tmp_forgery"
TAMPERS = ("copy_move", "splice", "inpaint", "text_replace")


@dataclass(frozen=True)
class Profile:
    seed: int
    patch_fraction: tuple[float, float]
    opacity: tuple[float, float]


PROFILES = {
    # Cover test-profile patch/opacity range; keep a distinct seed from "test".
    "train": Profile(seed=17_101, patch_fraction=(0.05, 0.26), opacity=(0.40, 0.90)),
    "train_v2": Profile(seed=17_202, patch_fraction=(0.04, 0.28), opacity=(0.35, 0.92)),
    "train_v3": Profile(seed=17_303, patch_fraction=(0.06, 0.24), opacity=(0.38, 0.88)),
    "test": Profile(seed=91_337, patch_fraction=(0.10, 0.26), opacity=(0.42, 0.74)),
}
DIFFICULTY = {
    "easy": (0.90, 1.00),
    "medium": (0.60, 0.82),
    "hard": (0.30, 0.58),
}


def _font(size: int) -> ImageFont.ImageFont:
    for candidate in ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/calibri.ttf"):
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def _box(rng: random.Random, width: int, height: int, fraction: float) -> tuple[int, int, int, int]:
    side = max(12, int(min(width, height) * fraction))
    patch_w = min(width - 2, side + rng.randrange(max(1, side // 2)))
    patch_h = min(height - 2, side + rng.randrange(max(1, side // 2)))
    x = rng.randrange(1, max(2, width - patch_w))
    y = rng.randrange(1, max(2, height - patch_h))
    return x, y, patch_w, patch_h


def _blend(
    target: np.ndarray, patch: np.ndarray, mask: np.ndarray, x: int, y: int, alpha: float
) -> None:
    h, w = patch.shape[:2]
    th, tw = target.shape[:2]
    h = min(h, th - y)
    w = min(w, tw - x)
    if h <= 0 or w <= 0:
        return
    patch = patch[:h, :w]
    mask = mask[:h, :w]
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    feather = cv2.GaussianBlur(mask, (0, 0), max(0.75, min(w, h) / 14))
    weight = (feather.astype(np.float32) / 255.0 * alpha)[..., None]
    roi = target[y : y + h, x : x + w].astype(np.float32)
    target[y : y + h, x : x + w] = np.clip(
        roi * (1.0 - weight) + patch.astype(np.float32) * weight, 0, 255
    ).astype(np.uint8)


def _copy_move(image: np.ndarray, rng: random.Random, fraction: float, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    out, h, w = image.copy(), image.shape[0], image.shape[1]
    sx, sy, pw, ph = _box(rng, w, h, fraction)
    dx, dy, _, _ = _box(rng, w, h, fraction)
    dx, dy = min(dx, w - pw), min(dy, h - ph)
    patch = image[sy : sy + ph, sx : sx + pw].copy()
    patch = cv2.warpAffine(patch, cv2.getRotationMatrix2D((pw / 2, ph / 2), rng.uniform(-3, 3), 1), (pw, ph))
    local_mask = np.full((ph, pw), 255, dtype=np.uint8)
    _blend(out, patch, local_mask, dx, dy, alpha)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[dy : dy + ph, dx : dx + pw] = local_mask
    return out, mask


def _splice(
    image: np.ndarray, donor: np.ndarray, rng: random.Random, fraction: float, alpha: float
) -> tuple[np.ndarray, np.ndarray]:
    out, h, w = image.copy(), image.shape[0], image.shape[1]
    x, y, pw, ph = _box(rng, w, h, fraction)
    donor = cv2.resize(donor, (w, h), interpolation=cv2.INTER_AREA)
    # Same (pw, ph) as destination — second _box() can pick a different size and break blend.
    dx = rng.randrange(0, max(1, w - pw + 1))
    dy = rng.randrange(0, max(1, h - ph + 1))
    patch = donor[dy : dy + ph, dx : dx + pw]
    if patch.shape[0] != ph or patch.shape[1] != pw:
        patch = cv2.resize(patch, (pw, ph), interpolation=cv2.INTER_AREA)
    local_mask = np.full((ph, pw), 255, dtype=np.uint8)
    _blend(out, patch, local_mask, x, y, alpha)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y : y + ph, x : x + pw] = local_mask
    return out, mask


def _inpaint(image: np.ndarray, rng: random.Random, fraction: float) -> tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    x, y, pw, ph = _box(rng, w, h, fraction)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (x + pw // 2, y + ph // 2), (pw // 2, ph // 2), 0, 0, 360, 255, -1)
    return cv2.inpaint(image, mask, 5, cv2.INPAINT_TELEA), mask


def _text_replace(image: np.ndarray, rng: random.Random, fraction: float, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    x, y, pw, ph = _box(rng, w, h, fraction)
    pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    overlay = pil.copy()
    draw = ImageDraw.Draw(overlay)
    draw.rectangle((x, y, x + pw, y + ph), fill=(245, 239, 221))
    draw.text((x + 3, y + max(1, ph // 4)), f"VALID {rng.randrange(10_000, 99_999)}", fill=(25, 25, 25), font=_font(max(9, ph // 4)))
    changed = cv2.cvtColor(np.asarray(overlay), cv2.COLOR_RGB2BGR)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y : y + ph, x : x + pw] = 255
    out = image.copy()
    _blend(out, changed[y : y + ph, x : x + pw], mask[y : y + ph, x : x + pw], x, y, alpha)
    return out, mask


def _tamper(
    kind: str, image: np.ndarray, donor: np.ndarray, rng: random.Random, fraction: float, alpha: float
) -> tuple[np.ndarray, np.ndarray]:
    if kind == "copy_move":
        return _copy_move(image, rng, fraction, alpha)
    if kind == "splice":
        return _splice(image, donor, rng, fraction, alpha)
    if kind == "inpaint":
        return _inpaint(image, rng, fraction)
    return _text_replace(image, rng, fraction, alpha)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def generate(
    input_dir: Path,
    *,
    profile_name: str,
    difficulty: str,
    count: int | None,
    output_root: Path | None = None,
) -> list[dict]:
    profile = PROFILES[profile_name]
    sources = sorted(path for path in input_dir.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if not sources:
        raise ValueError(f"no images found in {input_dir}")
    if count is not None:
        sources = sources[:count]

    # Folder per profile×difficulty so train pools can be concatenated without clobbering.
    # Legacy paths: train/medium → train, test/medium → test.
    if difficulty == "medium" and profile_name in {"train", "test"}:
        folder = profile_name
    else:
        folder = f"{profile_name}_{difficulty}"
    root = (output_root or DEFAULT_OUTPUT) / folder
    image_dir, mask_dir = root / "images", root / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(profile.seed + (hash(difficulty) % 10_000))
    alpha_min, alpha_max = DIFFICULTY[difficulty]
    records: list[dict] = []
    for index, source in enumerate(sources, start=1):
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        donor_path = sources[(index * 7) % len(sources)]
        donor = cv2.imread(str(donor_path), cv2.IMREAD_COLOR)
        if image is None or donor is None:
            raise ValueError(f"could not decode fixture image {source}")
        tamper = TAMPERS[(index - 1) % len(TAMPERS)]
        fraction = rng.uniform(*profile.patch_fraction)
        alpha = rng.uniform(*profile.opacity) * rng.uniform(alpha_min, alpha_max)
        forged, mask = _tamper(tamper, image, donor, rng, fraction, alpha)
        sample_id = f"fg_{folder}_{index:04d}"
        forged_path, mask_path = image_dir / f"{sample_id}.png", mask_dir / f"{sample_id}_mask.png"
        cv2.imwrite(str(forged_path), forged)
        cv2.imwrite(str(mask_path), mask)
        records.append(
            {
                "id": sample_id,
                "path": f"images/{forged_path.name}",
                "label": 1,
                "origin": "synthetic_generated",
                "fabrication": "script",
                "tamper": tamper,
                "mask_path": f"masks/{mask_path.name}",
                "profile": profile_name,
                "eval_domain": "in_domain",
                "generator": "gen_forgery",
                # Authentic filename this forgery was built from (document holdout key).
                "source": source.name,
            }
        )
    _write_jsonl(root / "manifest.jsonl", records)
    return records


def refresh_eval_manifest(
    test_records: list[dict],
    authentic_dir: Path,
    *,
    eval_root: Path,
    authentic_names: list[str] | None = None,
) -> None:
    """Merge authentic rows with test forgery rows for harness evaluation."""
    if authentic_names is not None:
        authentic = [authentic_dir / name for name in authentic_names]
        missing = [str(p) for p in authentic if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"authentic missing: {missing[:5]}")
    else:
        authentic = sorted(
            path for path in authentic_dir.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
    records = [
        {
            "id": path.stem,
            "path": f"authentic/{path.name}",
            "label": 0,
            "origin": "public_dataset",
            "fabrication": "authentic",
        }
        for path in authentic
    ]
    for record in test_records:
        records.append(dict(record))
    eval_root.mkdir(parents=True, exist_ok=True)
    _write_jsonl(eval_root / "manifest.jsonl", records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate document forgery samples")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=REPO_ROOT / "data" / "2_forgery" / "authentic",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--profile", choices=tuple(PROFILES), required=True)
    parser.add_argument("--difficulty", choices=tuple(DIFFICULTY), default="medium")
    parser.add_argument("--count", type=int, default=None, help="limit source images")
    parser.add_argument("--refresh-eval", action="store_true", help="merge test set into 2_forgery manifest")
    args = parser.parse_args()
    records = generate(
        args.input_dir,
        profile_name=args.profile,
        difficulty=args.difficulty,
        count=args.count,
        output_root=args.output_root,
    )
    if args.refresh_eval:
        if args.profile != "test":
            parser.error("--refresh-eval requires --profile test")
        refresh_eval_manifest(
            records,
            args.input_dir,
            eval_root=REPO_ROOT / "data" / "2_forgery",
        )
    print(f"Generated {len(records)} {args.profile}/{args.difficulty} forgeries under {records and args.output_root}")
    if records:
        # Recompute folder the same way generate() does for the log line.
        if args.difficulty == "medium" and args.profile in {"train", "test"}:
            folder = args.profile
        else:
            folder = f"{args.profile}_{args.difficulty}"
        print(f"  -> {args.output_root / folder}")


if __name__ == "__main__":
    main()
