"""TruFor backend (CMX integrity detection @ 512², /256.0 preprocess)."""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import HTTPException
from PIL import Image, UnidentifiedImageError


class TruForDetector:
    name = "trufor"

    def __init__(
        self,
        weights_path: Path,
        device: torch.device,
        *,
        trufor_src: Path,
        image_size: int = 512,
    ):
        self.weights_path = weights_path
        self.device = device
        self.image_size = image_size
        self.trufor_src = trufor_src
        self.status = "untrained"

        if not trufor_src.is_dir():
            raise FileNotFoundError(
                f"TruFor source not found at {trufor_src}. "
                "Set forgery.trufor_src or TRUFOR_SRC (compose mounts this at /trufor_src)."
            )
        src = str(trufor_src.resolve())
        if src not in sys.path:
            sys.path.insert(0, src)

        from config import _C as config  # noqa: E402
        from config import update_config  # noqa: E402
        from models.cmx.builder_np_conf import myEncoderDecoder as confcmx  # noqa: E402

        class _NS:
            pass

        ns = _NS()
        ns.opts = []
        # TruFor's update_config merges ./trufor.yaml relative to CWD.
        prev_cwd = Path.cwd()
        try:
            os.chdir(src)
            update_config(config, ns)
            self.model = confcmx(cfg=config).to(device).eval()
            if weights_path.is_file():
                # TruFor checkpoints include non-tensor metadata; weights_only=False required.
                checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
                state = (
                    checkpoint["state_dict"]
                    if isinstance(checkpoint, dict) and "state_dict" in checkpoint
                    else checkpoint
                )
                self.model.load_state_dict(state, strict=True)
                self.status = "checkpoint"
        finally:
            os.chdir(prev_cwd)

    def score(self, content: bytes) -> float:
        try:
            image = (
                Image.open(io.BytesIO(content))
                .convert("RGB")
                .resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
            )
        except (UnidentifiedImageError, OSError) as exc:
            raise HTTPException(status_code=422, detail="unable to decode image") from exc
        # Match TruFor train/eval tools: float32 / 256.0 (not 255).
        pixels = np.asarray(image, dtype=np.float32) / 256.0
        tensor = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            pred, _conf, det, _npp = self.model(tensor)
            if det is None:
                loc = F.softmax(pred, dim=1)[:, 1]
                value = loc.flatten(1).max(dim=1).values
            else:
                value = torch.sigmoid(det).view(-1)
            return float(value.item())
