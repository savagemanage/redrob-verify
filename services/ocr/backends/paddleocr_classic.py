"""Classic PaddleOCR (PP-OCRv3/v4/v5/v6 detection+recognition) — preserved for A/B."""

from __future__ import annotations

from typing import Any

import numpy as np


class PaddleOcrClassicBackend:
    name = "paddleocr_classic"
    license = "Apache-2.0"
    model_version = "paddleocr-classic"

    def __init__(self, lang: str = "en") -> None:
        from paddleocr import PaddleOCR  # type: ignore[import-not-found]

        lang = (lang or "en").strip().lower()
        if lang in {"", "ch", "chinese", "chinese_cht"}:
            raise ValueError(
                f"OCR lang={lang!r} would use a Chinese model on Latin text; use lang='en'"
            )
        self.lang = lang
        # PaddleOCR 3.x removed show_log / use_angle_cls; try modern kwargs first.
        last_error: Exception | None = None
        for kwargs in (
            {
                "lang": self.lang,
                "use_doc_orientation_classify": False,
                "use_doc_unwarping": False,
                "use_textline_orientation": True,
            },
            {"lang": self.lang, "use_textline_orientation": True},
            {"lang": self.lang},
            {"lang": self.lang, "use_angle_cls": True},  # 2.x
        ):
            try:
                self.engine = PaddleOCR(**kwargs)
                break
            except (TypeError, ValueError) as error:
                last_error = error
                self.engine = None  # type: ignore[assignment]
        else:
            raise ValueError(f"PaddleOCR classic init failed: {last_error}")
        self._weight_hint: str | None = None

    def model_sha256(self) -> str:
        # Classic downloads under ~/.paddlex / ~/.paddleocr; pin via lang+version.
        return self._weight_hint or f"classic:{self.lang}:{self.model_version}"

    def extract(self, image: np.ndarray) -> dict[str, Any]:
        # Prefer 3.x predict(); fall back to 2.x ocr().
        result: Any
        if hasattr(self.engine, "predict"):
            try:
                result = self.engine.predict(image)
            except TypeError:
                result = self.engine.predict(input=image)
        elif hasattr(self.engine, "ocr"):
            try:
                result = self.engine.ocr(image, cls=True)
            except TypeError:
                result = self.engine.ocr(image)
        else:
            result = None

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
        # 3.x OCRResult objects often expose rec_texts directly
        if not lines:
            for text, conf, box in _iter_paddlex_texts(result):
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


def _iter_paddlex_texts(result: Any) -> list[tuple[str, float | None, Any]]:
    out: list[tuple[str, float | None, Any]] = []
    for res in result or [] if isinstance(result, list) else [result]:
        if res is None:
            continue
        data = res
        if hasattr(res, "json") and callable(res.json):
            try:
                data = res.json
            except Exception:
                data = res
        if isinstance(data, dict) and "res" in data:
            data = data.get("res") or data
        if not isinstance(data, dict):
            # OCRResult-like: attribute access
            texts = getattr(res, "rec_texts", None) or getattr(data, "rec_texts", None)
            scores = getattr(res, "rec_scores", None) or getattr(data, "rec_scores", None)
            boxes = (
                getattr(res, "rec_polys", None)
                or getattr(res, "dt_polys", None)
                or getattr(res, "rec_boxes", None)
            )
            if texts:
                for i, text in enumerate(list(texts)):
                    conf = None
                    if scores is not None and i < len(scores):
                        try:
                            conf = float(scores[i])
                        except Exception:
                            conf = None
                    box = boxes[i] if boxes is not None and i < len(boxes) else None
                    out.append((str(text), conf, box))
            continue
        texts = data.get("rec_texts") or data.get("texts") or []
        scores = data.get("rec_scores") or data.get("scores") or []
        boxes = data.get("rec_polys") or data.get("dt_polys") or data.get("rec_boxes") or []
        for i, text in enumerate(list(texts)):
            conf = float(scores[i]) if i < len(scores) else None
            box = boxes[i] if i < len(boxes) else None
            out.append((str(text), conf, box))
    return out


def _iter_classic_pages(result: Any) -> list[list[Any]]:
    if result is None:
        return []
    if isinstance(result, list):
        # [[lines...]] or [lines...]
        if (
            result
            and isinstance(result[0], list)
            and result[0]
            and not isinstance(result[0][0], (list, tuple))
        ):
            return [result]
        # list of OCRResult / dict — not classic pages
        if result and not isinstance(result[0], (list, tuple)):
            return []
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
