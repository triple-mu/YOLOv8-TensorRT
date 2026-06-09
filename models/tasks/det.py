import cv2
import numpy as np

from config import CLASSES_DET, COLORS
from models.utils import blob, letterbox


def process(engine, bgr, draw, ctx) -> bool:
    img, ratio, dwdh = letterbox(bgr, (ctx.width, ctx.height))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = blob(rgb, return_seg=False)

    if ctx.torch:
        import torch

        from models.torch_utils import det_postprocess

        dwdh_arr = torch.asarray(dwdh * 2, dtype=torch.float32, device=ctx.device)
        data = engine(torch.asarray(tensor, device=ctx.device))
        bboxes, scores, labels = det_postprocess(data)
        empty = bboxes.numel() == 0
    else:
        from models.utils import det_postprocess

        dwdh_arr = np.array(dwdh * 2, dtype=np.float32)
        data = engine(np.ascontiguousarray(tensor))
        bboxes, scores, labels = det_postprocess(data)
        empty = bboxes.size == 0

    if empty:
        return False
    bboxes -= dwdh_arr
    bboxes /= ratio

    for bbox, score, label in zip(bboxes, scores, labels):
        bbox = (bbox.round().int() if ctx.torch else bbox.round().astype(np.int32)).tolist()
        cls = CLASSES_DET[int(label)]
        color = COLORS[cls]
        text = f"{cls}:{float(score):.3f}"
        x1, y1, x2, y2 = bbox
        (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1)
        y1c = min(y1 + 1, draw.shape[0])
        cv2.rectangle(draw, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(draw, (x1, y1c), (x1 + tw, y1c + th + bl), (0, 0, 255), -1)
        cv2.putText(draw, text, (x1, y1c + th), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    return True
