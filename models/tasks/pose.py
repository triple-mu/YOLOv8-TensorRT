import cv2
import numpy as np

from config import COLORS, KPS_COLORS, LIMB_COLORS, SKELETON
from models.utils import blob, letterbox

# Single-class pose ("person") is the common case; a multi-class model reports
# its other classes by index. `COLORS` is keyed by the COCO class order, so
# index 0 is always "person" and the single-class path stays byte-identical.
POSE_NAMES = ["person"]
POSE_COLORS = list(COLORS.values())


def preprocess(bgr, ctx):
    img, ratio, dwdh = letterbox(bgr, (ctx.width, ctx.height))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = blob(rgb, return_seg=False)
    return tensor, (ratio, dwdh)


def postprocess(data, meta, draw, ctx) -> bool:
    ratio, dwdh = meta
    dw, dh = int(dwdh[0]), int(dwdh[1])

    if ctx.torch:
        import torch

        from models.torch_utils import pose_postprocess

        dwdh_arr = torch.asarray(dwdh * 2, dtype=torch.float32, device=ctx.device)
        bboxes, scores, kpts, labels = pose_postprocess(data, ctx.conf_thres, ctx.iou_thres)
        empty = bboxes.numel() == 0
    else:
        from models.utils import pose_postprocess

        dwdh_arr = np.array(dwdh * 2, dtype=np.float32)
        bboxes, scores, kpts, labels = pose_postprocess(data, ctx.conf_thres, ctx.iou_thres)
        empty = bboxes.size == 0

    if empty:
        return False
    bboxes -= dwdh_arr
    bboxes /= ratio

    for bbox, score, kpt, label in zip(bboxes, scores, kpts, labels):
        bbox = (bbox.round().int() if ctx.torch else bbox.round().astype(np.int32)).tolist()
        label = int(label)
        name = POSE_NAMES[label] if label < len(POSE_NAMES) else str(label)
        cv2.rectangle(draw, bbox[:2], bbox[2:], POSE_COLORS[label % len(POSE_COLORS)], 2)
        cv2.putText(
            draw,
            f"{name}:{float(score):.3f}",
            (bbox[0], bbox[1] - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            [225, 255, 255],
            2,
        )
        for i in range(19):
            if i < 17:
                px, py, ps = kpt[i]
                if ps > 0.5:
                    cv2.circle(
                        draw, (round(float(px - dw) / ratio), round(float(py - dh) / ratio)), 5, KPS_COLORS[i], -1
                    )
            xi, yi = SKELETON[i]
            if kpt[xi - 1][2] > 0.5 and kpt[yi - 1][2] > 0.5:
                p1 = (round(float(kpt[xi - 1][0] - dw) / ratio), round(float(kpt[xi - 1][1] - dh) / ratio))
                p2 = (round(float(kpt[yi - 1][0] - dw) / ratio), round(float(kpt[yi - 1][1] - dh) / ratio))
                cv2.line(draw, p1, p2, LIMB_COLORS[i], 2)
    return True
