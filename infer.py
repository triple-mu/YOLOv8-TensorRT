"""Unified inference entry point.

Replaces the former infer-{det,seg,pose,obb,cls}[-without-torch].py scripts with a
single command parameterised by --task and --backend, e.g.

    python infer.py --task det  --backend torch  --engine yolov8s.engine --imgs data
    python infer.py --task seg  --backend cudart --engine yolov8s-seg.engine --imgs data
    python infer.py --task pose --backend pycuda --engine yolov8s-pose.engine --imgs data --show
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from models.tasks import DESIRED_OUTPUTS, Context, get_task, slice_batch
from models.utils import path_to_list


def build_engine(backend: str, weight: str, device_str: str, task: str):
    """Construct the inference backend; returns (engine, Context-without-thresholds)."""
    if backend == "torch":
        import torch

        from models import TRTModule

        device = torch.device(device_str)
        engine = TRTModule(weight, device)
        if task in DESIRED_OUTPUTS:
            engine.set_desired(DESIRED_OUTPUTS[task])
        return engine, True, device
    if backend == "cudart":
        try:
            from models.cudart_api import TRTEngine
        except ImportError as e:
            raise SystemExit("backend 'cudart' needs cuda-python: pip install cuda-python") from e
        return TRTEngine(weight), False, None
    if backend == "pycuda":
        try:
            from models.pycuda_api import TRTEngine
        except ImportError as e:
            raise SystemExit("backend 'pycuda' needs pycuda: pip install pycuda") from e
        return TRTEngine(weight), False, None
    raise SystemExit(f"unknown backend '{backend}' (choose torch / cudart / pycuda)")


def run_engine(engine, batch_tensor, is_torch: bool, device):
    if is_torch:
        import torch

        return engine(torch.asarray(batch_tensor, device=device))
    return engine(np.ascontiguousarray(batch_tensor))


def main(args: argparse.Namespace) -> None:
    task = get_task(args.task)
    engine, is_torch, device = build_engine(args.backend, args.engine, args.device, args.task)
    height, width = engine.inp_info[0].shape[-2:]
    ctx = Context(is_torch, device, args.conf_thres, args.iou_thres, height, width)

    # The engine's batch dimension drives batching: a fixed batch (e.g. 2) is
    # used as-is, a dynamic one (-1) follows --batch. Inputs are stacked into a
    # single [N, 3, H, W] tensor for one engine call, then decoded per image.
    engine_batch = engine.inp_info[0].shape[0]
    dynamic = engine_batch is None or engine_batch < 1
    batch = args.batch if dynamic else engine_batch
    if not dynamic and args.batch != engine_batch:
        print(f"engine has a fixed batch of {engine_batch}, ignoring --batch {args.batch}")

    images = path_to_list(args.imgs)
    save_dir = Path(args.out_dir)
    if not args.show:
        save_dir.mkdir(parents=True, exist_ok=True)

    for start in range(0, len(images), batch):
        bgrs, names = [], []
        for image in images[start : start + batch]:
            bgr = cv2.imread(str(image))
            if bgr is None:
                print(f"{image}: cannot read image, skipping")
                continue
            bgrs.append(bgr)
            names.append(image)
        if not bgrs:
            continue

        tensors, metas = zip(*(task.preprocess(bgr, ctx) for bgr in bgrs))
        tensors = list(tensors)
        n = len(tensors)
        if not dynamic and n < batch:  # fixed-batch engine: pad the short final chunk
            tensors += [tensors[-1]] * (batch - n)
        data = run_engine(engine, np.concatenate(tensors, 0), is_torch, device)

        for i in range(n):
            draw = bgrs[i].copy()
            if not task.postprocess(slice_batch(data, i), metas[i], draw, ctx):
                print(f"{names[i]}: no object!")
                continue
            if args.show:
                cv2.imshow("result", draw)
                cv2.waitKey(0)
            else:
                cv2.imwrite(str(save_dir / names[i].name), draw)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 TensorRT inference")
    parser.add_argument("--task", required=True, choices=["det", "seg", "pose", "obb", "cls"], help="Task type")
    parser.add_argument("--backend", default="torch", choices=["torch", "cudart", "pycuda"], help="Inference backend")
    parser.add_argument("--engine", required=True, type=str, help="Engine file")
    parser.add_argument("--imgs", required=True, type=str, help="Image file or directory")
    parser.add_argument("--show", action="store_true", help="Show results instead of saving")
    parser.add_argument("--out-dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="Confidence threshold (seg/pose/obb)")
    parser.add_argument("--iou-thres", type=float, default=0.65, help="NMS IoU threshold (seg/pose/obb)")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device (torch backend)")
    parser.add_argument("--batch", type=int, default=1, help="Images per engine call (dynamic-batch engines only)")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
