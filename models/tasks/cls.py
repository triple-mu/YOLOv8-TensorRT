import cv2

from config import CLASSES_CLS
from models.utils import blob


def preprocess(bgr, ctx):
    img = cv2.resize(bgr, (ctx.width, ctx.height))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = blob(rgb, return_seg=False)
    return tensor, None


def postprocess(data, meta, draw, ctx) -> bool:
    if ctx.torch:
        score, cls_id = data[0].max(0)
        score, cls_id = float(score), int(cls_id)
    else:
        probs = data[0]
        cls_id = int(probs.argmax())
        score = float(probs[cls_id])

    text = f"{CLASSES_CLS[cls_id]}:{score:.3f}"
    (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1)
    y = min(10, draw.shape[0])
    cv2.rectangle(draw, (10, y), (10 + tw, y + th + bl), (0, 0, 255), -1)
    cv2.putText(draw, text, (10, y + th), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    return True
