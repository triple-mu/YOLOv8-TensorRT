import argparse
from pathlib import Path

import cv2
import numpy as np

from config import CLASSES_CLS
from models.utils import blob, path_to_list


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
        bgr = cv2.resize(bgr, (W, H))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = blob(rgb, return_seg=False)
        tensor = np.ascontiguousarray(tensor)
        # inference
        data = Engine(tensor)

        data = data[0]
        score = data.max().item()
        cls_id = data.argmax().item()
        cls = CLASSES_CLS[cls_id]

        text = f"{cls}:{score:.3f}"
        (_w, _h), _bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1)
        _y1 = min(10, draw.shape[0])

        cv2.rectangle(draw, (10, _y1), (10 + _w, _y1 + _h + _bl), (0, 0, 255), -1)
        cv2.putText(draw, text, (10, _y1 + _h), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

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
