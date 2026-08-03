"""OpenCV document-image preprocessing steps for the OCR service."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def deskew(image: np.ndarray) -> np.ndarray:
    """Correct small page rotations using the median Hough-line angle."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=max(40, image.shape[1] // 8),
        maxLineGap=15,
    )
    if lines is None:
        return image

    angles = [
        np.degrees(np.arctan2(y2 - y1, x2 - x1))
        for x1, y1, x2, y2 in lines.reshape(-1, 4)
        if abs(np.degrees(np.arctan2(y2 - y1, x2 - x1))) <= 15
    ]
    if not angles:
        return image
    angle = float(np.median(angles))
    height, width = image.shape[:2]
    transform = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    return cv2.warpAffine(
        image,
        transform,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def perspective_warp(image: np.ndarray) -> np.ndarray:
    """Find a large four-corner page border and flatten it when reliable."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    image_area = image.shape[0] * image.shape[1]

    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(contour) < image_area * 0.2:
            break
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) != 4:
            continue
        points = _order_points(approx.reshape(4, 2).astype(np.float32))
        width = int(max(np.linalg.norm(points[1] - points[0]), np.linalg.norm(points[2] - points[3])))
        height = int(max(np.linalg.norm(points[3] - points[0]), np.linalg.norm(points[2] - points[1])))
        if width < 32 or height < 32:
            return image
        target = np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype=np.float32,
        )
        return cv2.warpPerspective(image, cv2.getPerspectiveTransform(points, target), (width, height))
    return image


def remove_background(image: np.ndarray) -> np.ndarray:
    """Flatten uneven paper illumination by dividing by a blurred background."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=31, sigmaY=31)
    normalized = cv2.divide(gray, background, scale=255)
    return cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)


def normalize_and_binarize(image: np.ndarray) -> np.ndarray:
    """Contrast-normalize grayscale text then apply adaptive binarization."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    normalized = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    binary = cv2.adaptiveThreshold(
        normalized,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def sharpen(image: np.ndarray) -> np.ndarray:
    """Apply unsharp masking while retaining natural document edges."""
    blurred = cv2.GaussianBlur(image, (0, 0), 2.0)
    return cv2.addWeighted(image, 1.5, blurred, -0.5, 0)


def preprocess_image(image: np.ndarray, settings: dict[str, Any] | None = None) -> tuple[np.ndarray, list[str]]:
    """Apply configured preprocessing in document-safe order."""
    enabled = settings or {}
    steps: list[tuple[str, Any]] = [
        ("deskew", deskew),
        ("perspective", perspective_warp),
        ("bg_remove", remove_background),
        ("binarize", normalize_and_binarize),
        ("sharpen", sharpen),
    ]
    output = image
    applied: list[str] = []
    for name, step in steps:
        if bool(enabled.get(name, False)):
            output = step(output)
            applied.append(name)
    return output, applied


def _order_points(points: np.ndarray) -> np.ndarray:
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).reshape(-1)
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(diffs)]
    ordered[3] = points[np.argmax(diffs)]
    return ordered
