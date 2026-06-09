from models import TRTModule  # isort:skip
import argparse
from pathlib import Path

import cv2
import torch

from config import CLASSES_OBB, COLORS_OBB
from models.torch_utils import obb_postprocess
from models.utils import blob, letterbox, path_to_list


def main(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    Engine = TRTModule(args.engine, device)
    H, W = Engine.inp_info[0].shape[-2:]

    images = path_to_list(args.imgs)
    save_path = Path(args.out_dir)

    if not args.show and not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)

    for image in images:
        save_image = save_path / image.name
        bgr = cv2.imread(str(image))
        draw = bgr.copy()
        bgr, ratio, dwdh = letterbox(bgr, (W, H))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = blob(rgb, return_seg=False)
        dwdh = torch.asarray(dwdh, dtype=torch.float32, device=device)
        tensor = torch.asarray(tensor, device=device)
        # inference
        data = Engine(tensor)

        points, scores, labels = obb_postprocess(data, args.conf_thres, args.iou_thres)
        if points.numel() == 0:
            # if no points
            print(f"{image}: no object!")
            continue
        points -= dwdh
        points /= ratio

        for point, score, label in zip(points, scores, labels):
            point = point.round().int().cpu().numpy()
            label = int(label)
            score = float(score)
            cls = CLASSES_OBB[label]
            color = COLORS_OBB[cls]
            cv2.polylines(draw, [point], True, color, 2)
            cv2.putText(
                draw,
                f"{cls}:{score:.3f}",
                (point[0, 0], point[0, 1] - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                [225, 255, 255],
                thickness=2,
            )

        if args.show:
            cv2.imshow("result", draw)
            cv2.waitKey(0)
        else:
            cv2.imwrite(str(save_image), draw)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=str, help="Engine file")
    parser.add_argument("--imgs", type=str, help="Images file")
    parser.add_argument("--show", action="store_true", help="Show the detection results")
    parser.add_argument("--out-dir", type=str, default="./output", help="Path to output file")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.65, help="Confidence threshold")
    parser.add_argument("--device", type=str, default="cuda:0", help="TensorRT infer device")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
