"""ForgeryNet backend (ResNet-50 + FFT + HOG @ 224²)."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import torch
from fastapi import HTTPException
from PIL import Image, UnidentifiedImageError

from services.forgery.model import ForgeryNet


class ForgeryNetDetector:
    name = "forgery_net"

    def __init__(self, weights_path: Path, device: torch.device, image_size: int = 224):
        self.weights_path = weights_path
        self.device = device
        self.image_size = image_size
        self.model = ForgeryNet().to(device).eval()
        self.status = "untrained"
        if weights_path.is_file():
            checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
            state = (
                checkpoint.get("model_state", checkpoint)
                if isinstance(checkpoint, dict)
                else checkpoint
            )
            self.model.load_state_dict(state)
            self.status = "checkpoint"

    def score(self, content: bytes) -> float:
        try:
            image = (
                Image.open(io.BytesIO(content))
                .convert("RGB")
                .resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
            )
        except (UnidentifiedImageError, OSError) as exc:
            raise HTTPException(status_code=422, detail="unable to decode image") from exc
        pixels = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            return float(torch.sigmoid(self.model(tensor)).item())
