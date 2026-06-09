import cv2
import numpy as np

from config import ALPHA, CLASSES_SEG, COLORS, MASK_COLORS
from models.utils import blob, letterbox


def process(engine, bgr, draw, ctx) -> bool:
    img, ratio, dwdh = letterbox(bgr, (ctx.width, ctx.height))
    dw, dh = int(dwdh[0]), int(dwdh[1])
    h, w = ctx.height, ctx.width
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor, seg_img = blob(rgb, return_seg=True)
    seg_img = seg_img[dh : h - dh, dw : w - dw, [2, 1, 0]]

    if ctx.torch:
        import torch

        from models.torch_utils import seg_postprocess

        dwdh_arr = torch.asarray(dwdh * 2, dtype=torch.float32, device=ctx.device)
        data = engine(torch.asarray(tensor, device=ctx.device))
        seg_img = torch.asarray(seg_img, device=ctx.device)
        bboxes, scores, labels, masks = seg_postprocess(data, img.shape[:2], ctx.conf_thres, ctx.iou_thres)
        if bboxes.numel() == 0:
            return False
        masks = masks[:, dh : h - dh, dw : w - dw, :]
        mask_colors = torch.asarray(MASK_COLORS, device=ctx.device)[(labels % len(MASK_COLORS)).long()]
        mask_colors = mask_colors.view(-1, 1, 1, 3) * ALPHA
        mask_colors = masks @ mask_colors
        inv_alph_masks = (1 - masks * 0.5).cumprod(0)
        mcs = (mask_colors * inv_alph_masks).sum(0) * 2
        seg_img = (seg_img * inv_alph_masks[-1] + mcs) * 255
        seg_img = seg_img.cpu().numpy().astype(np.uint8)
    else:
        from models.utils import seg_postprocess

        dwdh_arr = np.array(dwdh * 2, dtype=np.float32)
        data = engine(np.ascontiguousarray(tensor))
        bboxes, scores, labels, masks = seg_postprocess(data, img.shape[:2], ctx.conf_thres, ctx.iou_thres)
        if bboxes.size == 0:
            return False
        masks = masks[:, dh : h - dh, dw : w - dw, :]
        mask_colors = MASK_COLORS[labels % len(MASK_COLORS)].reshape(-1, 1, 1, 3) * ALPHA
        mask_colors = masks @ mask_colors
        inv_alph_masks = (1 - masks * 0.5).cumprod(0)
        mcs = (mask_colors * inv_alph_masks).sum(0) * 2
        seg_img = ((seg_img * inv_alph_masks[-1] + mcs) * 255).astype(np.uint8)

    draw[:] = cv2.resize(seg_img, draw.shape[:2][::-1])
    bboxes -= dwdh_arr
    bboxes /= ratio
    for bbox, score, label in zip(bboxes, scores, labels):
        bbox = (bbox.round().int() if ctx.torch else bbox.round().astype(np.int32)).tolist()
        cls = CLASSES_SEG[int(label)]
        cv2.rectangle(draw, bbox[:2], bbox[2:], COLORS[cls], 2)
        cv2.putText(
            draw,
            f"{cls}:{float(score):.3f}",
            (bbox[0], bbox[1] - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            [225, 255, 255],
            2,
        )
    return True
