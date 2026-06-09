from models import TRTModule  # isort:skip
import argparse
from pathlib import Path

import cv2
import torch

from config import COLORS, KPS_COLORS, LIMB_COLORS, SKELETON
from models.torch_utils import pose_postprocess
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
        dw, dh = int(dwdh[0]), int(dwdh[1])
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = blob(rgb, return_seg=False)
        dwdh = torch.asarray(dwdh * 2, dtype=torch.float32, device=device)
        tensor = torch.asarray(tensor, device=device)
        # inference
        data = Engine(tensor)

        bboxes, scores, kpts = pose_postprocess(data, args.conf_thres, args.iou_thres)
        if bboxes.numel() == 0:
            # if no bounding box
            print(f"{image}: no object!")
            continue
        bboxes -= dwdh
        bboxes /= ratio

        for bbox, score, kpt in zip(bboxes, scores, kpts):
            bbox = bbox.round().int().tolist()
            color = COLORS["person"]
            cv2.rectangle(draw, bbox[:2], bbox[2:], color, 2)
            cv2.putText(
                draw,
                f"person:{score:.3f}",
                (bbox[0], bbox[1] - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                [225, 255, 255],
                thickness=2,
            )
            for i in range(19):
                if i < 17:
                    px, py, ps = kpt[i]
                    if ps > 0.5:
                        kcolor = KPS_COLORS[i]
                        px = round(float(px - dw) / ratio)
                        py = round(float(py - dh) / ratio)
                        cv2.circle(draw, (px, py), 5, kcolor, -1)
                xi, yi = SKELETON[i]
                pos1_s = kpt[xi - 1][2]
                pos2_s = kpt[yi - 1][2]
                if pos1_s > 0.5 and pos2_s > 0.5:
                    limb_color = LIMB_COLORS[i]
                    pos1_x = round(float(kpt[xi - 1][0] - dw) / ratio)
                    pos1_y = round(float(kpt[xi - 1][1] - dh) / ratio)

                    pos2_x = round(float(kpt[yi - 1][0] - dw) / ratio)
                    pos2_y = round(float(kpt[yi - 1][1] - dh) / ratio)

                    cv2.line(draw, (pos1_x, pos1_y), (pos2_x, pos2_y), limb_color, 2)
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
