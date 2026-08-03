"""OCR backends package."""

from services.ocr.backends.factory import create_backend

__all__ = ["create_backend"]
