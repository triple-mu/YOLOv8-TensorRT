import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops import batched_nms, nms

from .utils import obb_postprocess as np_obb_postprocess


def seg_postprocess(
    data: tuple[Tensor], shape: tuple | list, conf_thres: float = 0.25, iou_thres: float = 0.65
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    assert len(data) == 2
    h, w = shape[0] // 4, shape[1] // 4  # 4x downsampling
    outputs, proto = data[0][0], data[1][0]
    bboxes, scores, labels, maskconf = outputs.split([4, 1, 1, 32], 1)
    scores, labels = scores.squeeze(), labels.squeeze()
    idx = scores > conf_thres
    if not idx.any():  # no bounding boxes or seg were created
        return bboxes.new_zeros((0, 4)), scores.new_zeros((0,)), labels.new_zeros((0,)), bboxes.new_zeros((0, 0, 0, 0))
    bboxes, scores, labels, maskconf = bboxes[idx], scores[idx], labels[idx], maskconf[idx]
    idx = batched_nms(bboxes, scores, labels, iou_thres)
    bboxes, scores, labels, maskconf = bboxes[idx], scores[idx], labels[idx].int(), maskconf[idx]
    masks = (maskconf @ proto).sigmoid().view(-1, h, w)
    masks = crop_mask(masks, bboxes / 4.0)
    masks = F.interpolate(masks[None], shape, mode="bilinear", align_corners=False)[0]
    masks = masks.gt_(0.5)[..., None]
    return bboxes, scores, labels, masks


def pose_postprocess(
    data: tuple | Tensor, conf_thres: float = 0.25, iou_thres: float = 0.65
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if isinstance(data, tuple):
        assert len(data) == 1
        data = data[0]
    outputs = torch.transpose(data[0], 0, 1).contiguous()
    num_cls = outputs.shape[-1] - 4 - 51  # 51 = 17 keypoints x 3; usually 1 (person)
    bboxes, cls, kpts = outputs.split([4, num_cls, 51], 1)
    scores, labels = cls.max(-1)
    idx = scores > conf_thres
    if not idx.any():  # no bounding boxes were created
        return (
            bboxes.new_zeros((0, 4)),
            scores.new_zeros((0,)),
            bboxes.new_zeros((0, 0, 0)),
            labels.new_zeros((0,)),
        )
    bboxes, scores, kpts, labels = bboxes[idx], scores[idx], kpts[idx], labels[idx]
    xycenter, wh = bboxes.chunk(2, -1)
    bboxes = torch.cat([xycenter - 0.5 * wh, xycenter + 0.5 * wh], -1)
    keep = nms(bboxes, scores, iou_thres)
    bboxes, scores, kpts, labels = bboxes[keep], scores[keep], kpts[keep], labels[keep]
    return bboxes, scores, kpts.reshape(keep.shape[0], -1, 3), labels


def det_postprocess(data: tuple[Tensor, Tensor, Tensor, Tensor]):
    assert len(data) == 4
    iou_thres: float = 0.65  # noqa F841
    num_dets, bboxes, scores, labels = data[0][0], data[1][0], data[2][0], data[3][0]
    nums = num_dets.item()
    if nums == 0:
        return bboxes.new_zeros((0, 4)), scores.new_zeros((0,)), labels.new_zeros((0,))
    # check score negative
    scores[scores < 0] = 1 + scores[scores < 0]
    bboxes = bboxes[:nums]
    scores = scores[:nums]
    labels = labels[:nums]

    return bboxes, scores, labels


def obb_postprocess(
    data: tuple | Tensor, conf_thres: float = 0.25, iou_thres: float = 0.65
) -> tuple[Tensor, Tensor, Tensor]:
    if isinstance(data, tuple):
        assert len(data) == 1
        data = data[0]
    device = data.device
    points, scores, labels = np_obb_postprocess(data.cpu().numpy(), conf_thres, iou_thres)
    return torch.from_numpy(points).to(device), torch.from_numpy(scores).to(device), torch.from_numpy(labels).to(device)


def crop_mask(masks: Tensor, bboxes: Tensor) -> Tensor:
    n, h, w = masks.shape
    x1, y1, x2, y2 = torch.chunk(bboxes[:, :, None], 4, 1)  # x1 shape(1,1,n)
    r = torch.arange(w, device=masks.device, dtype=x1.dtype)[None, None, :]  # rows shape(1,w,1)
    c = torch.arange(h, device=masks.device, dtype=x1.dtype)[None, :, None]  # cols shape(h,1,1)

    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))
