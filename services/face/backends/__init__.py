"""Re-export face backends."""

from services.face.backends.factory import create_backend
from services.face.backends.sface import SFaceBackend
from services.face.backends.stub import StubEmbeddingBackend

__all__ = ["StubEmbeddingBackend", "SFaceBackend", "create_backend"]
