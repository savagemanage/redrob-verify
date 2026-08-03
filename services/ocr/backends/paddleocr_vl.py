"""PaddleOCR-VL-1.6 backend — pinned; no auto-latest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

# Hard pin — never float to "latest"
MODEL_ID = "PaddlePaddle/PaddleOCR-VL-1.6"
PIPELINE_VERSION = "v1.6"


class PaddleOcrVlBackend:
    name = "paddleocr_vl"
    license = "Apache-2.0"
    model_version = "1.6"

    def __init__(self, *, model_dir: Path | None = None) -> None:
        from paddleocr import PaddleOCRVL  # type: ignore[import-not-found]

        kwargs: dict[str, Any] = {"pipeline_version": PIPELINE_VERSION}
        if model_dir is not None and model_dir.is_dir():
            kwargs["vl_rec_model_dir"] = str(model_dir)
        self.engine = PaddleOCRVL(**kwargs)
        self.model_id = MODEL_ID
        self._sha_cache: str | None = None
        self._model_dir = model_dir

    def model_sha256(self) -> str:
        if self._sha_cache:
            return self._sha_cache
        candidates: list[Path] = []
        if self._model_dir is not None:
            candidates.append(self._model_dir / "config.json")
        home = Path.home()
        candidates.extend(
            home.glob(".cache/huggingface/hub/models--PaddlePaddle--PaddleOCR-VL-1.6/**/config.json")
        )
        candidates.extend(Path("/root/.paddlex/official_models/PaddleOCR-VL-1.6").glob("config.json"))
        for path in candidates:
            if path.is_file():
                h = hashlib.sha256()
                h.update(path.read_bytes())
                h.update(f"{MODEL_ID}:{PIPELINE_VERSION}".encode())
                self._sha_cache = h.hexdigest()
                return self._sha_cache
        self._sha_cache = hashlib.sha256(f"{MODEL_ID}:{PIPELINE_VERSION}".encode()).hexdigest()
        return self._sha_cache

    def extract(self, image: np.ndarray) -> dict[str, Any]:
        # VL + 8GB VRAM: MIDV scans hang if resolution / pixel budget too high.
        # DocLayoutV3 returns zero boxes on most ID/passport cards, so skip layout
        # and run whole-image OCR prompt with a tight pixel budget.
        image = _downscale_long_side(image, max_side=960)
        results = self.engine.predict(
            image,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_layout_detection=False,
            prompt_label="ocr",
            vlm_extra_args={"ocr_max_pixels": 1280 * 1280, "ocr_min_pixels": 100 * 100},
        )
        parsed = _parse_vl_results(results)
        # If still empty, one more try with layout + image-block OCR (some scans work).
        if not parsed["lines"]:
            results = self.engine.predict(
                image,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_ocr_for_image_block=True,
                layout_shape_mode="auto",
            )
            parsed = _parse_vl_results(results)
            if parsed["lines"]:
                parsed["reason"] = "layout_retry_after_ocr_prompt_empty"
            else:
                parsed["reason"] = "vl_empty_after_ocr_prompt_and_layout"
        else:
            parsed["reason"] = "ocr_prompt_no_layout"
        return {
            "fields": parsed["fields"],
            "text": "\n".join(parsed["lines"]),
            "script": None,
            "reason": parsed.get("reason"),
            "model_id": self.model_id,
            "pipeline_version": PIPELINE_VERSION,
        }


def _parse_vl_results(results: Any) -> dict[str, Any]:
    fields: dict[str, dict[str, Any]] = {}
    lines: list[str] = []
    idx = 0
    for res in results or []:
        for block in _blocks_from_result(res):
            label = str(block.get("label") or "block")
            content = block.get("content")
            score = block.get("score")
            bbox = _coord_to_xyxy(block.get("bbox"))
            name = f"{label}_{idx:03d}"
            idx += 1
            conf = float(score) if score is not None else None
            if bbox is None:
                # Absolute rule: no bbox → null value (hallucination defense)
                # Whole-image OCR prompt often has no layout bbox — keep text.
                value = None if content is None else str(content).strip()
                if value and label.lower() in {"ocr", "text", "block"}:
                    fields[name] = {
                        "value": value,
                        "bbox": [0.0, 0.0, 1.0, 1.0],
                        "confidence": conf,
                    }
                    lines.append(value)
                else:
                    fields[name] = {"value": None, "bbox": None, "confidence": conf}
            else:
                value = None if content is None else str(content).strip()
                fields[name] = {"value": value or None, "bbox": bbox, "confidence": conf}
                if value:
                    lines.append(value)
    return {"fields": fields, "lines": lines, "reason": None}


def _downscale_long_side(image: np.ndarray, *, max_side: int) -> np.ndarray:
    """Limit VL input resolution to avoid VRAM thrash / hung generation on scans."""
    height, width = image.shape[:2]
    long_side = max(height, width)
    if long_side <= max_side:
        return image
    scale = max_side / float(long_side)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    import cv2

    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _normalize_block(block: Any) -> dict[str, Any] | None:
    """Map PaddleOCRVLBlock / JSON dict into {label, content, bbox, score}."""
    if block is None:
        return None
    if isinstance(block, dict):
        label = block.get("label") or block.get("block_label") or block.get("type") or "block"
        content = (
            block.get("content")
            if block.get("content") is not None
            else block.get("block_content")
            if block.get("block_content") is not None
            else block.get("text")
        )
        bbox = block.get("bbox") or block.get("block_bbox") or block.get("coordinate") or block.get("box")
        score = block.get("score") or block.get("confidence")
        return {"label": label, "content": content, "bbox": bbox, "score": score}
    # PaddleOCRVLBlock-like objects expose .label / .content / .bbox
    label = getattr(block, "label", None) or getattr(block, "block_label", None) or "block"
    content = getattr(block, "content", None)
    if content is None:
        content = getattr(block, "block_content", None)
    bbox = getattr(block, "bbox", None)
    if bbox is None:
        bbox = getattr(block, "block_bbox", None)
    score = getattr(block, "score", None)
    if content is None and bbox is None and not hasattr(block, "label"):
        return None
    return {"label": label, "content": content, "bbox": bbox, "score": score}


def _blocks_from_result(res: Any) -> list[dict[str, Any]]:
    raw_list = _raw_parsing_list(res)
    out: list[dict[str, Any]] = []
    for item in raw_list:
        normalized = _normalize_block(item)
        if normalized is not None:
            out.append(normalized)
    return out


def _raw_parsing_list(res: Any) -> list[Any]:
    if res is None:
        return []
    if isinstance(res, list):
        return res
    if isinstance(res, dict):
        # unwrap {"res": {...}} JSON wrapper from .json property
        if "res" in res and isinstance(res["res"], dict) and "parsing_res_list" not in res:
            return _raw_parsing_list(res["res"])
        for key in ("parsing_res_list", "layout_det_res", "boxes", "blocks"):
            val = res.get(key)
            if isinstance(val, list) and val:
                return val
            if isinstance(val, dict) and isinstance(val.get("boxes"), list):
                return val["boxes"]
        return []
    # Prefer mapping access used by PaddleOCRVLResult
    try:
        if "parsing_res_list" in res:  # type: ignore[operator]
            val = res["parsing_res_list"]
            if isinstance(val, list):
                return val
    except Exception:
        pass
    for attr in ("json", "res", "data"):
        payload = getattr(res, attr, None)
        if callable(payload):
            try:
                payload = payload()
            except Exception:
                payload = None
        if isinstance(payload, dict):
            found = _raw_parsing_list(payload)
            if found:
                return found
        if isinstance(payload, str):
            try:
                found = _raw_parsing_list(json.loads(payload))
                if found:
                    return found
            except Exception:
                pass
    try:
        return _raw_parsing_list(dict(res))
    except Exception:
        return []


def _coord_to_xyxy(coord: Any) -> list[float] | None:
    if coord is None:
        return None
    try:
        arr = np.asarray(coord, dtype=np.float32).reshape(-1)
        if arr.size >= 4:
            if arr.size == 4:
                return [float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])]
            pts = arr.reshape(-1, 2)
            return [
                float(pts[:, 0].min()),
                float(pts[:, 1].min()),
                float(pts[:, 0].max()),
                float(pts[:, 1].max()),
            ]
    except Exception:
        return None
    return None
