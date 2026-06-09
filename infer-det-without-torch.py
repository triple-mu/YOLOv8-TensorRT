import argparse
from pathlib import Path

import cv2
import numpy as np

from config import CLASSES_DET, COLORS
from models.utils import blob, det_postprocess, letterbox, path_to_list


def main(args: argparse.Namespace) -> None:
    if args.method == "cudart":
        from models.cudart_api import TRTEngine
    elif args.method == "pycuda":
        from models.pycuda_api import TRTEngine
    else:
        raise NotImplementedError

    Engine = TRTEngine(args.engine)
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
        dwdh = np.array(dwdh * 2, dtype=np.float32)
        tensor = np.ascontiguousarray(tensor)
        # inference
        data = Engine(tensor)

        bboxes, scores, labels = det_postprocess(data)
        if bboxes.size == 0:
            # if no bounding box
            print(f"{image}: no object!")
            continue
        bboxes -= dwdh
        bboxes /= ratio

        for bbox, score, label in zip(bboxes, scores, labels):
            bbox = bbox.round().astype(np.int32).tolist()
            cls_id = int(label)
            cls = CLASSES_DET[cls_id]
            color = COLORS[cls]

            text = f"{cls}:{score:.3f}"
            x1, y1, x2, y2 = bbox

            (_w, _h), _bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1)
            _y1 = min(y1 + 1, draw.shape[0])

            cv2.rectangle(draw, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(draw, (x1, _y1), (x1 + _w, _y1 + _h + _bl), (0, 0, 255), -1)
            cv2.putText(draw, text, (x1, _y1 + _h), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        if args.show:
            cv2.imshow("result", draw)
            cv2.waitKey(0)
        else:
            cv2.imwrite(str(save_image), draw)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=str, help="Engine file")
    parser.add_argument("--imgs", type=str, help="Images file")
    parser.add_argument("--show", action="store_true", help="Show the detection results")
    parser.add_argument("--out-dir", type=str, default="./output", help="Path to output file")
    parser.add_argument("--method", type=str, default="cudart", help="CUDART pipeline")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
