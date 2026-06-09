"""Class names, palettes and pose skeleton.

Class names are loaded from ``data/labels/*.txt`` (shared with the C++ side);
colours/skeleton are defined here. ``config.py`` re-exports everything for
backward compatibility.
"""

import random
from pathlib import Path

import numpy as np

random.seed(0)

_LABELS_DIR = Path(__file__).resolve().parent.parent / "data" / "labels"


def load_labels(name: str) -> tuple[str, ...]:
    """Load labels from ``data/labels/<name>.txt`` (or a direct path), one per line."""
    path = Path(name)
    if not path.is_file():
        path = _LABELS_DIR / f"{name}.txt"
    return tuple(line for line in path.read_text().splitlines() if line)


# classification (ImageNet-1k), detection/segmentation (COCO-80), obb (DOTA-15)
CLASSES_CLS = load_labels("imagenet")
CLASSES_DET = load_labels("coco")
CLASSES_SEG = CLASSES_DET
CLASSES_OBB = load_labels("dota")

COLORS = {cls: [random.randint(0, 255) for _ in range(3)] for cls in CLASSES_DET}
COLORS_OBB = {cls: [random.randint(0, 255) for _ in range(3)] for cls in CLASSES_OBB}

# colours for segmentation mask overlays, normalised to [0, 1]
MASK_COLORS = (
    np.array(
        [
            (255, 56, 56),
            (255, 157, 151),
            (255, 112, 31),
            (255, 178, 29),
            (207, 210, 49),
            (72, 249, 10),
            (146, 204, 23),
            (61, 219, 134),
            (26, 147, 52),
            (0, 212, 187),
            (44, 153, 168),
            (0, 194, 255),
            (52, 69, 147),
            (100, 115, 255),
            (0, 24, 236),
            (132, 56, 255),
            (82, 0, 133),
            (203, 56, 255),
            (255, 149, 200),
            (255, 55, 199),
        ],
        dtype=np.float32,
    )
    / 255.0
)

ALPHA = 0.5  # segmentation mask blend factor

# pose: 17 keypoints, 19 skeleton edges
KPS_COLORS = [
    [0, 255, 0], [0, 255, 0], [0, 255, 0], [0, 255, 0], [0, 255, 0],
    [255, 128, 0], [255, 128, 0], [255, 128, 0], [255, 128, 0], [255, 128, 0], [255, 128, 0],
    [51, 153, 255], [51, 153, 255], [51, 153, 255], [51, 153, 255], [51, 153, 255], [51, 153, 255],
]  # fmt: skip

SKELETON = [
    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
    [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7],
]  # fmt: skip

LIMB_COLORS = [
    [51, 153, 255], [51, 153, 255], [51, 153, 255], [51, 153, 255],
    [255, 51, 255], [255, 51, 255], [255, 51, 255],
    [255, 128, 0], [255, 128, 0], [255, 128, 0], [255, 128, 0], [255, 128, 0],
    [0, 255, 0], [0, 255, 0], [0, 255, 0], [0, 255, 0], [0, 255, 0], [0, 255, 0], [0, 255, 0],
]  # fmt: skip
