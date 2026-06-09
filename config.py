"""Backward-compatible re-export.

The canonical definitions now live in :mod:`models.labels` (class names are loaded
from ``data/labels/*.txt``). This shim keeps ``import config`` / ``from config import
...`` working for existing scripts.
"""

from models.labels import (
    ALPHA,
    CLASSES_CLS,
    CLASSES_DET,
    CLASSES_OBB,
    CLASSES_SEG,
    COLORS,
    COLORS_OBB,
    KPS_COLORS,
    LIMB_COLORS,
    MASK_COLORS,
    SKELETON,
)

__all__ = [
    "ALPHA",
    "CLASSES_CLS",
    "CLASSES_DET",
    "CLASSES_OBB",
    "CLASSES_SEG",
    "COLORS",
    "COLORS_OBB",
    "KPS_COLORS",
    "LIMB_COLORS",
    "MASK_COLORS",
    "SKELETON",
]
