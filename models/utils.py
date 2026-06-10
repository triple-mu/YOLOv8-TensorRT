from pathlib import Path

import cv2
import numpy as np
from numpy import ndarray

# image suffixs
SUFFIXS = (".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp", ".pfm")

# angle scale
ANGLE_SCALE = 1 / np.pi * 180.0


def letterbox(
    im: ndarray, new_shape: tuple | list = (640, 640), color: tuple | list = (114, 114, 114)
) -> tuple[ndarray, float, tuple[float, float]]:
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    # new_shape: [width, height]

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[1], new_shape[1] / shape[0])
    # Compute padding [width, height]
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[0] - new_unpad[0], new_shape[1] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, r, (dw, dh)


def blob(im: ndarray, return_seg: bool = False) -> ndarray | tuple:
    seg = None
    if return_seg:
        seg = im.astype(np.float32) / 255
    im = im.transpose([2, 0, 1])
    im = im[np.newaxis, ...]
    im = np.ascontiguousarray(im).astype(np.float32) / 255
    if return_seg:
        return im, seg
    else:
        return im


def sigmoid(x: ndarray) -> ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def path_to_list(images_path: str | Path) -> list:
    if isinstance(images_path, str):
        images_path = Path(images_path)
    if not images_path.exists():
        raise FileNotFoundError(f"path does not exist: {images_path}")
    if images_path.is_dir():
        images = [i.absolute() for i in images_path.iterdir() if i.suffix in SUFFIXS]
    else:
        if images_path.suffix not in SUFFIXS:
            raise ValueError(f"unsupported image suffix: {images_path.suffix}")
        images = [images_path.absolute()]
    return images


def crop_mask(masks: ndarray, bboxes: ndarray) -> ndarray:
    n, h, w = masks.shape
    x1, y1, x2, y2 = np.split(bboxes[:, :, None], [1, 2, 3], 1)  # x1 shape(1,1,n)
    r = np.arange(w, dtype=x1.dtype)[None, None, :]  # rows shape(1,w,1)
    c = np.arange(h, dtype=x1.dtype)[None, :, None]  # cols shape(h,1,1)

    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))


def box_iou(box1: ndarray, box2: ndarray) -> float:
    x11, y11, x21, y21 = box1
    x12, y12, x22, y22 = box2
    x1 = max(x11, x12)
    y1 = max(y11, y12)
    x2 = min(x21, x22)
    y2 = min(y21, y22)
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    union_area = (x21 - x11) * (y21 - y11) + (x22 - x12) * (y22 - y12) - inter_area
    return max(0, inter_area / union_area)


def NMSBoxes(boxes: ndarray, scores: ndarray, labels: ndarray, iou_thres: float, agnostic: bool = False):
    num_boxes = boxes.shape[0]
    order = np.argsort(scores)[::-1]
    boxes = boxes[order]
    labels = labels[order]

    indices = []

    for i in range(num_boxes):
        box_a = boxes[i]
        label_a = labels[i]
        keep = True
        for j in indices:
            box_b = boxes[j]
            label_b = labels[j]
            if not agnostic and label_a != label_b:
                continue
            if box_iou(box_a, box_b) > iou_thres:
                keep = False
        if keep:
            indices.append(i)

    indices = np.array(indices, dtype=np.int32)
    return order[indices]


def det_postprocess(data: tuple[ndarray, ndarray, ndarray, ndarray]):
    if len(data) != 4:
        raise ValueError(f"det_postprocess expects 4 output tensors, got {len(data)}")
    num_dets, bboxes, scores, labels = (i[0] for i in data)
    nums = num_dets.item()
    if nums == 0:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int32)
    # check score negative
    scores[scores < 0] = 1 + scores[scores < 0]
    bboxes = bboxes[:nums]
    scores = scores[:nums]
    labels = labels[:nums]
    return bboxes, scores, labels


def seg_postprocess(
    data: tuple[ndarray], shape: tuple | list, conf_thres: float = 0.25, iou_thres: float = 0.65
) -> tuple[ndarray, ndarray, ndarray, ndarray]:
    assert len(data) == 2
    h, w = shape[0] // 4, shape[1] // 4  # 4x downsampling
    outputs, proto = (i[0] for i in data)
    bboxes, scores, labels, maskconf = np.split(outputs, [4, 5, 6], 1)
    scores, labels = scores.squeeze(), labels.squeeze()
    idx = scores > conf_thres
    if not idx.any():  # no bounding boxes or seg were created
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
            np.empty((0, 0, 0, 0), dtype=np.int32),
        )

    bboxes, scores, labels, maskconf = bboxes[idx], scores[idx], labels[idx], maskconf[idx]
    cvbboxes = np.concatenate([bboxes[:, :2], bboxes[:, 2:] - bboxes[:, :2]], 1)
    labels = labels.astype(np.int32)
    v0, v1 = map(int, (cv2.__version__).split(".")[:2])
    assert v0 == 4, "OpenCV version is wrong"
    if v1 > 6:
        idx = cv2.dnn.NMSBoxesBatched(cvbboxes, scores, labels, conf_thres, iou_thres)
    else:
        idx = cv2.dnn.NMSBoxes(cvbboxes, scores, conf_thres, iou_thres)
    bboxes, scores, labels, maskconf = bboxes[idx], scores[idx], labels[idx], maskconf[idx]
    masks = sigmoid(maskconf @ proto).reshape(-1, h, w)
    masks = crop_mask(masks, bboxes / 4.0)
    masks = masks.transpose([1, 2, 0])
    masks = cv2.resize(masks, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    masks = masks.transpose(2, 0, 1)
    masks = np.ascontiguousarray((masks > 0.5)[..., None], dtype=np.float32)
    return bboxes, scores, labels, masks


def pose_postprocess(
    data: tuple | ndarray, conf_thres: float = 0.25, iou_thres: float = 0.65
) -> tuple[ndarray, ndarray, ndarray, ndarray]:
    if isinstance(data, tuple):
        assert len(data) == 1
        data = data[0]
    outputs = np.transpose(data[0], (1, 0))
    num_cls = outputs.shape[-1] - 4 - 51  # 51 = 17 keypoints x 3; usually 1 (person)
    bboxes, cls, kpts = np.split(outputs, [4, 4 + num_cls], 1)
    scores = cls.max(-1)
    labels = cls.argmax(-1)
    idx = scores > conf_thres
    if not idx.any():  # no bounding boxes were created
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0, 0, 0), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        )
    bboxes, scores, kpts, labels = bboxes[idx], scores[idx], kpts[idx], labels[idx]
    xycenter, wh = np.split(bboxes, [2], -1)
    cvbboxes = np.concatenate([xycenter - 0.5 * wh, wh], -1)
    nms = cv2.dnn.NMSBoxes(cvbboxes, scores, conf_thres, iou_thres)
    cvbboxes, scores, kpts, labels = cvbboxes[nms], scores[nms], kpts[nms], labels[nms]
    cvbboxes[:, 2:] += cvbboxes[:, :2]
    return cvbboxes, scores, kpts.reshape(nms.shape[0], -1, 3), labels


def obb_postprocess(
    data: tuple | ndarray, conf_thres: float = 0.25, iou_thres: float = 0.65
) -> tuple[ndarray, ndarray, ndarray]:
    if isinstance(data, tuple):
        assert len(data) == 1
        data = data[0]
    outputs = np.transpose(data[0], (1, 0))
    num_cls = outputs.shape[-1] - 5
    bboxes, scores, angles = np.split(outputs, [4, num_cls + 4], 1)
    scores, labels = scores.max(-1), scores.argmax(-1)
    scores, labels, angles = scores.squeeze(), labels.squeeze(), angles.squeeze()
    idx = scores > conf_thres
    if not idx.any():  # no obbs were created
        return np.empty((0, 4, 2), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int32)
    bboxes, scores, labels, angles = bboxes[idx], scores[idx], labels[idx], angles[idx] * ANGLE_SCALE
    cvrbboxes = [[(xc, yc), (w, h), a] for (xc, yc, w, h), a in zip(bboxes, angles)]
    idx = cv2.dnn.NMSBoxesRotated(cvrbboxes, scores, conf_thres, iou_thres)
    points = np.array([cv2.boxPoints(cvrbboxes[i]) for i in idx], dtype=np.float32)
    return points, scores, labels
