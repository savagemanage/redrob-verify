"""Forgery detection backends."""

from services.forgery.backends.factory import create_detector

__all__ = ["create_detector"]
