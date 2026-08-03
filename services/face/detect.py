"""Face detection, 5-point landmark estimate, and affine alignment."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Reference 5-point template for 112x112 ArcFace-style alignment (eyes, nose, mouth corners)
ARCFACE_5POINT_REF = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


@dataclass
class FaceDetection:
    box: tuple[int, int, int, int]  # x, y, w, h
    landmarks: np.ndarray  # (5, 2) float32
    score: float
    source: str  # "haar" | "yunet" | "fullframe_fallback"
    raw_face: np.ndarray | None = None  # YuNet detector row for alignCrop


def _landmarks_from_box(x: int, y: int, w: int, h: int) -> np.ndarray:
    """Approximate 5-point landmarks from a detection box (no learned landmark model)."""
    return np.array(
        [
            [x + 0.3 * w, y + 0.35 * h],
            [x + 0.7 * w, y + 0.35 * h],
            [x + 0.5 * w, y + 0.55 * h],
            [x + 0.35 * w, y + 0.78 * h],
            [x + 0.65 * w, y + 0.78 * h],
        ],
        dtype=np.float32,
    )


def detect_faces(
    bgr: np.ndarray,
    *,
    fullframe_fallback: bool = False,
    min_side: int = 24,
) -> list[FaceDetection]:
    """Detect faces with OpenCV Haar; optionally fall back to full-frame for stub/dummy assets."""
    if bgr.size == 0:
        return []
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    h, w = gray.shape[:2]
    found: list[FaceDetection] = []
    try:
        cascade_path = getattr(cv2, "data", None)
        cascade_cls = getattr(cv2, "CascadeClassifier", None)
        if cascade_cls is not None and cascade_path is not None:
            cascade = cascade_cls(
                cascade_path.haarcascades + "haarcascade_frontalface_default.xml"
            )
            if not cascade.empty() and min(h, w) >= min_side:
                boxes = cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=3, minSize=(min_side, min_side)
                )
                for (x, y, bw, bh) in boxes:
                    found.append(
                        FaceDetection(
                            box=(int(x), int(y), int(bw), int(bh)),
                            landmarks=_landmarks_from_box(int(x), int(y), int(bw), int(bh)),
                            score=1.0,
                            source="haar",
                        )
                    )
    except Exception:
        # OpenCV builds without Haar (e.g. some 5.x wheels) — fall through
        found = []
    if found:
        found.sort(key=lambda d: d.box[2] * d.box[3], reverse=True)
        return found
    if fullframe_fallback and h > 0 and w > 0:
        return [
            FaceDetection(
                box=(0, 0, w, h),
                landmarks=_landmarks_from_box(0, 0, w, h),
                score=0.0,
                source="fullframe_fallback",
            )
        ]
    return []


def align_face(
    bgr: np.ndarray,
    landmarks: np.ndarray,
    *,
    out_size: tuple[int, int] = (112, 112),
) -> np.ndarray:
    """Similarity transform from 5 landmarks to ArcFace reference template."""
    src = np.asarray(landmarks, dtype=np.float32).reshape(5, 2)
    dst = ARCFACE_5POINT_REF.copy()
    if out_size != (112, 112):
        sx = out_size[0] / 112.0
        sy = out_size[1] / 112.0
        dst = dst * np.array([sx, sy], dtype=np.float32)
    m, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if m is None:
        x = int(src[:, 0].min())
        y = int(src[:, 1].min())
        w = int(max(1.0, float(src[:, 0].max() - src[:, 0].min())))
        h = int(max(1.0, float(src[:, 1].max() - src[:, 1].min())))
        crop = bgr[max(0, y) : y + h, max(0, x) : x + w]
        if crop.size == 0:
            crop = bgr
        return cv2.resize(crop, out_size, interpolation=cv2.INTER_LINEAR)
    return cv2.warpAffine(bgr, m, out_size, flags=cv2.INTER_LINEAR)
