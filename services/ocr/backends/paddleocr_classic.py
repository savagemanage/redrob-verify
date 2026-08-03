"""Classic PaddleOCR (PP-OCRv3/v4 detection+recognition) — preserved for A/B."""

from __future__ import annotations

from typing import Any

import numpy as np


class PaddleOcrClassicBackend:
    name = "paddleocr_classic"
    license = "Apache-2.0"
    model_version = "paddleocr-classic-en"

    def __init__(self, lang: str = "en") -> None:
        from paddleocr import PaddleOCR  # type: ignore[import-not-found]

        lang = (lang or "en").strip().lower()
        if lang in {"", "ch", "chinese", "chinese_cht"}:
            raise ValueError(
                f"OCR lang={lang!r} would use a Chinese model on Latin text; use lang='en'"
            )
        self.lang = lang
        # paddleocr 2.x / 3.x constructor kwargs differ; prefer silent classic path
        try:
            self.engine = PaddleOCR(use_angle_cls=True, lang=self.lang, show_log=False)
        except TypeError:
            self.engine = PaddleOCR(use_angle_cls=True, lang=self.lang)
        self._weight_hint: str | None = None

    def model_sha256(self) -> str:
        # Classic downloads under ~/.paddleocr; pin identity via version string if no file.
        return self._weight_hint or f"classic:{self.lang}:{self.model_version}"

    def extract(self, image: np.ndarray) -> dict[str, Any]:
        # paddleocr 2.x: .ocr(); 3.x classic: .predict() / .ocr()
        if hasattr(self.engine, "ocr"):
            try:
                result = self.engine.ocr(image, cls=True)
            except TypeError:
                result = self.engine.ocr(image)
        else:
            result = self.engine.predict(image)
        fields: dict[str, dict[str, Any]] = {}
        lines: list[str] = []
        idx = 0
        for page in _iter_classic_pages(result):
            for item in page:
                text, conf, box = _parse_classic_item(item)
                if text is None:
                    continue
                bbox = _quad_to_xyxy(box)
                name = f"line_{idx:03d}"
                idx += 1
                if bbox is None:
                    fields[name] = {"value": None, "bbox": None, "confidence": conf}
                else:
                    fields[name] = {"value": text, "bbox": bbox, "confidence": conf}
                    lines.append(text)
        return {
            "fields": fields,
            "text": "\n".join(lines),
            "script": None,
            "reason": None,
        }


def _iter_classic_pages(result: Any) -> list[list[Any]]:
    if result is None:
        return []
    if isinstance(result, list):
        # [[lines...]] or [lines...]
        if result and isinstance(result[0], list) and result[0] and not isinstance(result[0][0], (list, tuple)):
            return [result]
        return result
    return []


def _parse_classic_item(item: Any) -> tuple[str | None, float | None, Any]:
    if not item:
        return None, None, None
    if isinstance(item, dict):
        text = item.get("transcription") or item.get("text") or item.get("rec_text")
        conf = item.get("score") or item.get("confidence") or item.get("rec_score")
        box = item.get("points") or item.get("box") or item.get("dt_polys")
        return (str(text) if text is not None else None), (float(conf) if conf is not None else None), box
    if len(item) < 2:
        return None, None, None
    box = item[0]
    payload = item[1]
    if isinstance(payload, (list, tuple)) and payload:
        text = str(payload[0])
        conf = float(payload[1]) if len(payload) > 1 else None
        return text, conf, box
    return None, None, box


def _quad_to_xyxy(box: Any) -> list[float] | None:
    try:
        pts = np.asarray(box, dtype=np.float32).reshape(-1, 2)
        if pts.size < 4:
            return None
        x0, y0 = float(pts[:, 0].min()), float(pts[:, 1].min())
        x1, y1 = float(pts[:, 0].max()), float(pts[:, 1].max())
        return [x0, y0, x1, y1]
    except Exception:
        return None
