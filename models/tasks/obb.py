import cv2
import numpy as np

from config import CLASSES_OBB, COLORS_OBB
from models.utils import blob, letterbox


def process(engine, bgr, draw, ctx) -> bool:
    img, ratio, dwdh = letterbox(bgr, (ctx.width, ctx.height))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = blob(rgb, return_seg=False)

    if ctx.torch:
        import torch

        from models.torch_utils import obb_postprocess

        dwdh_arr = torch.asarray(dwdh, dtype=torch.float32, device=ctx.device)
        data = engine(torch.asarray(tensor, device=ctx.device))
        points, scores, labels = obb_postprocess(data, ctx.conf_thres, ctx.iou_thres)
        empty = points.numel() == 0
    else:
        from models.utils import obb_postprocess

        dwdh_arr = np.array(dwdh, dtype=np.float32)
        data = engine(np.ascontiguousarray(tensor))
        points, scores, labels = obb_postprocess(data, ctx.conf_thres, ctx.iou_thres)
        empty = points.size == 0

    if empty:
        return False
    points -= dwdh_arr
    points /= ratio

    for point, score, label in zip(points, scores, labels):
        point = point.round().int().cpu().numpy() if ctx.torch else point.round().astype(np.int32)
        cls = CLASSES_OBB[int(label)]
        color = COLORS_OBB[cls]
        cv2.polylines(draw, [point], True, color, 2)
        cv2.putText(
            draw,
            f"{cls}:{float(score):.3f}",
            (point[0, 0], point[0, 1] - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            [225, 255, 255],
            2,
        )
    return True
