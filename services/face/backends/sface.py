"""OpenCV Zoo SFace + YuNet backend (Apache-2.0).

Detection: FaceDetectorYN
Alignment: FaceRecognizerSF.alignCrop (do not hand-roll affine)
Embedding: FaceRecognizerSF.feature → 128-d
Match: FaceRecognizerSF.match(..., FR_COSINE) in [-1, 1]
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from services.face.detect import FaceDetection
from services.face.quality import assess_quality_yunet


class SFaceBackend:
    name = "sface"
    license = "Apache-2.0 (OpenCV Zoo face_detection_yunet + face_recognition_sface)"

    def __init__(
        self,
        *,
        yunet_path: Path,
        sface_path: Path,
        score_threshold: float = 0.6,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
    ) -> None:
        if not yunet_path.is_file():
            raise FileNotFoundError(f"YuNet model missing: {yunet_path}")
        if not sface_path.is_file():
            raise FileNotFoundError(f"SFace model missing: {sface_path}")
        self.yunet_path = yunet_path
        self.sface_path = sface_path
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self._detector: cv2.FaceDetectorYN | None = None
        self._recognizer = cv2.FaceRecognizerSF.create(str(sface_path), "")

    def _detector_for(self, width: int, height: int) -> cv2.FaceDetectorYN:
        if self._detector is None:
            self._detector = cv2.FaceDetectorYN.create(
                str(self.yunet_path),
                "",
                (width, height),
                self.score_threshold,
                self.nms_threshold,
                self.top_k,
            )
        else:
            self._detector.setInputSize((width, height))
        return self._detector

    def detect(self, bgr: np.ndarray) -> FaceDetection | None:
        h, w = bgr.shape[:2]
        detector = self._detector_for(w, h)
        _retval, faces = detector.detect(bgr)
        if faces is None or len(faces) == 0:
            return None
        # Pick highest-score face
        best = max(faces, key=lambda row: float(row[-1]))
        x, y, bw, bh = [int(round(v)) for v in best[:4]]
        landmarks = np.array(
            [
                [best[4], best[5]],
                [best[6], best[7]],
                [best[8], best[9]],
                [best[10], best[11]],
                [best[12], best[13]],
            ],
            dtype=np.float32,
        )
        return FaceDetection(
            box=(x, y, bw, bh),
            landmarks=landmarks,
            score=float(best[-1]),
            source="yunet",
            raw_face=best,
        )

    def feature_from_detection(
        self, bgr: np.ndarray, det: FaceDetection
    ) -> np.ndarray:
        face_row = det.raw_face
        if face_row is None:
            # Reconstruct FaceDetectorYN row: x,y,w,h + 10 landmark coords + score
            x, y, bw, bh = det.box
            lm = det.landmarks.reshape(-1)
            face_row = np.array(
                [x, y, bw, bh, *lm.tolist(), det.score], dtype=np.float32
            )
        aligned = self._recognizer.alignCrop(bgr, face_row)
        feat = self._recognizer.feature(aligned)
        return np.asarray(feat, dtype=np.float32).reshape(-1)

    def match_cosine(self, feat_a: np.ndarray, feat_b: np.ndarray) -> float:
        score = float(
            self._recognizer.match(
                feat_a.reshape(1, -1),
                feat_b.reshape(1, -1),
                cv2.FaceRecognizerSF_FR_COSINE,
            )
        )
        return score

    def process(
        self,
        bgr: np.ndarray,
        *,
        quality_cfg: dict[str, Any],
    ) -> tuple[np.ndarray | None, dict[str, Any], str | None]:
        det = self.detect(bgr)
        if det is None:
            return None, {}, "no_face_detected"
        quality = assess_quality_yunet(bgr, det, **_quality_kwargs(quality_cfg))
        try:
            feat = self.feature_from_detection(bgr, det)
        except cv2.error as error:
            return None, quality, f"align_or_feature_failed: {error}"
        return feat, quality, None

    def embed(self, img: np.ndarray) -> np.ndarray:
        """Protocol compatibility: expect an already-aligned crop when possible."""
        feat = self._recognizer.feature(img)
        return np.asarray(feat, dtype=np.float32).reshape(-1)


def _quality_kwargs(cfg: dict[str, Any]) -> dict[str, float]:
    return {
        "min_face_side_px": float(cfg.get("min_face_side_px", 40)),
        "min_laplacian_var": float(cfg.get("min_laplacian_var", 50.0)),
        "max_yaw_deg": float(cfg.get("max_yaw_deg", 30.0)),
        "min_eye_distance_ratio": float(cfg.get("min_eye_distance_ratio", 0.05)),
    }
