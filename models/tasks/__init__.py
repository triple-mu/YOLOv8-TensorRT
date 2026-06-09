"""Per-task inference handlers shared by the unified ``infer.py`` entry point.

Each task module exposes ``process(engine, bgr, draw, ctx) -> bool`` which runs
preprocess -> inference -> postprocess -> render for one image, drawing onto
``draw`` in place and returning whether any object was found. The torch and
numpy (cudart/pycuda) paths reuse ``models.torch_utils`` / ``models.utils``
respectively, keeping the original numeric behaviour of both.
"""

from dataclasses import dataclass


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
