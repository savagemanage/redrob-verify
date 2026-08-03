"""Detection-stage quality gating. Does NOT decide match/non-match."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from services.face.detect import FaceDetection


def laplacian_variance(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def estimate_yaw_from_symmetry(landmarks: np.ndarray) -> float:
    """Approximate yaw from left/right landmark symmetry (degrees)."""
    pts = np.asarray(landmarks, dtype=np.float64).reshape(5, 2)
    left_eye, right_eye, nose, mouth_l, mouth_r = pts
    eye_mid = (left_eye + right_eye) / 2.0
    inter_ocular = np.linalg.norm(right_eye - left_eye) + 1e-6
    # Horizontal nose offset vs eye midpoint → yaw proxy
    return float(np.clip((nose[0] - eye_mid[0]) / inter_ocular * 60.0, -90.0, 90.0))


def estimate_yaw_pitch_deg(landmarks: np.ndarray) -> tuple[float, float]:
    """Rough yaw/pitch from 5-point landmarks (no 3D head-pose model)."""
    pts = np.asarray(landmarks, dtype=np.float64).reshape(5, 2)
    left_eye, right_eye, nose, mouth_l, mouth_r = pts
    eye_mid = (left_eye + right_eye) / 2.0
    inter_ocular = np.linalg.norm(right_eye - left_eye) + 1e-6
    yaw = float(np.clip((nose[0] - eye_mid[0]) / inter_ocular * 60.0, -90.0, 90.0))
    mouth_mid = (mouth_l + mouth_r) / 2.0
    face_h = abs(mouth_mid[1] - eye_mid[1]) + 1e-6
    expected_nose_y = eye_mid[1] + 0.45 * (mouth_mid[1] - eye_mid[1])
    pitch = float(np.clip((expected_nose_y - nose[1]) / face_h * 40.0, -90.0, 90.0))
    return yaw, pitch


def assess_quality(
    bgr: np.ndarray,
    det: FaceDetection,
    *,
    min_face_side_px: float,
    min_laplacian_var: float,
    max_yaw_deg: float,
    max_pitch_deg: float,
) -> dict[str, Any]:
    x, y, w, h = det.box
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    face_gray = gray[max(0, y) : y + h, max(0, x) : x + w]
    blur = laplacian_variance(face_gray) if face_gray.size else 0.0
    yaw, pitch = estimate_yaw_pitch_deg(det.landmarks)
    face_side = float(min(w, h))

    flags: list[str] = []
    if face_side < min_face_side_px:
        flags.append("face_too_small")
    if blur < min_laplacian_var:
        flags.append("too_blurry")
    if abs(yaw) > max_yaw_deg:
        flags.append("yaw_out_of_range")
    if abs(pitch) > max_pitch_deg:
        flags.append("pitch_out_of_range")
    if det.source == "fullframe_fallback":
        flags.append("fullframe_fallback")

    return {
        "face_side_px": face_side,
        "laplacian_var": blur,
        "yaw_deg": yaw,
        "pitch_deg": pitch,
        "detect_score": det.score,
        "detect_source": det.source,
        "box": {"x": x, "y": y, "w": w, "h": h},
        "gates": {
            "min_face_side_px": min_face_side_px,
            "min_laplacian_var": min_laplacian_var,
            "max_yaw_deg": max_yaw_deg,
            "max_pitch_deg": max_pitch_deg,
        },
        "flags": flags,
        "passed": len([f for f in flags if f != "fullframe_fallback"]) == 0,
    }


def assess_quality_yunet(
    bgr: np.ndarray,
    det: FaceDetection,
    *,
    min_face_side_px: float,
    min_laplacian_var: float,
    max_yaw_deg: float,
    min_eye_distance_ratio: float,
) -> dict[str, Any]:
    """Quality from YuNet 5-point landmarks. Informational only — never decides match."""
    x, y, w, h = det.box
    img_h, img_w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    face_gray = gray[max(0, y) : y + h, max(0, x) : x + w]
    blur = laplacian_variance(face_gray) if face_gray.size else 0.0
    pts = np.asarray(det.landmarks, dtype=np.float64).reshape(5, 2)
    left_eye, right_eye = pts[0], pts[1]
    eye_distance = float(np.linalg.norm(right_eye - left_eye))
    eye_distance_ratio = eye_distance / float(max(img_w, 1))
    yaw = estimate_yaw_from_symmetry(det.landmarks)
    face_side = float(min(w, h))

    flags: list[str] = []
    if face_side < min_face_side_px:
        flags.append("face_too_small")
    if eye_distance_ratio < min_eye_distance_ratio:
        flags.append("eye_distance_too_small")
    if blur < min_laplacian_var:
        flags.append("too_blurry")
    if abs(yaw) > max_yaw_deg:
        flags.append("yaw_out_of_range")

    return {
        "face_side_px": face_side,
        "eye_distance_px": eye_distance,
        "eye_distance_ratio": eye_distance_ratio,
        "laplacian_var": blur,
        "yaw_deg": yaw,
        "landmark_symmetry_yaw_deg": yaw,
        "detect_score": det.score,
        "detect_source": det.source,
        "box": {"x": x, "y": y, "w": w, "h": h},
        "gates": {
            "min_face_side_px": min_face_side_px,
            "min_laplacian_var": min_laplacian_var,
            "max_yaw_deg": max_yaw_deg,
            "min_eye_distance_ratio": min_eye_distance_ratio,
        },
        "flags": flags,
        "passed": len(flags) == 0,
    }
