"""Backward-compatible alias. The implementation now lives in models.backend."""

from models.backend import PycudaBackend as TRTEngine

__all__ = ["TRTEngine"]
