"""Per-task inference handlers shared by the unified ``infer.py`` entry point.

Each task module exposes two functions:

- ``preprocess(bgr, ctx) -> (tensor, meta)`` — letterbox/resize + blob for one
  image; ``tensor`` is ``[1, 3, H, W]`` so a batch is just a concatenation.
- ``postprocess(data, meta, draw, ctx) -> bool`` — decode one image's engine
  output slice, draw onto ``draw`` in place, return whether anything was found.

``infer.py`` runs the engine in between, so a batch of N images is one engine
call on the stacked ``[N, 3, H, W]`` tensor followed by N per-image
``postprocess`` calls (see ``slice_batch``). The torch and numpy (cudart/pycuda)
paths reuse ``models.torch_utils`` / ``models.utils`` respectively, keeping the
original numeric behaviour of both.
"""

from dataclasses import dataclass


def slice_batch(data, i: int):
    """One image's slice of a batched engine output, keeping the leading dim
    so the per-task postprocess functions stay batch-shaped (``data[0]`` etc.)."""
    if isinstance(data, tuple):
        return tuple(t[i : i + 1] for t in data)
    return data[i : i + 1]


@dataclass
class Context:
    torch: bool  # True for the torch backend, False for cudart/pycuda
    device: object  # torch.device for the torch backend, else None
    conf_thres: float
    iou_thres: float
    height: int  # network input H
    width: int  # network input W


def get_task(name: str):
    from . import cls, det, obb, pose, seg

    tasks = {"det": det, "seg": seg, "pose": pose, "obb": obb, "cls": cls}
    if name not in tasks:
        raise ValueError(f"unknown task '{name}', choose from {sorted(tasks)}")
    return tasks[name]


# Output binding orders that the torch TRTModule must request via set_desired().
DESIRED_OUTPUTS = {
    "det": ["num_dets", "bboxes", "scores", "labels"],
    "seg": ["outputs", "proto"],
}
