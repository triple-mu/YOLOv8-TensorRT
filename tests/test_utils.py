"""Unit tests for the torch-free pure functions in models.utils."""

import numpy as np

from models.utils import NMSBoxes, blob, box_iou, det_postprocess, letterbox, sigmoid


def test_letterbox_shape_and_padding():
    im = np.zeros((480, 640, 3), dtype=np.uint8)
    out, ratio, (dw, dh) = letterbox(im, (640, 640))
    assert out.shape == (640, 640, 3)
    assert ratio == 1.0  # 640-wide image fits exactly
    assert dw == 0 and dh == 80  # pad top/bottom only


def test_letterbox_preserves_aspect_ratio():
    im = np.zeros((1000, 500, 3), dtype=np.uint8)
    out, ratio, _ = letterbox(im, (640, 640))
    assert out.shape == (640, 640, 3)
    assert abs(ratio - 0.64) < 1e-6  # min(640/500, 640/1000) = 0.64


def test_blob_normalises_to_nchw():
    im = np.full((32, 48, 3), 255, dtype=np.uint8)
    out = blob(im)
    assert out.shape == (1, 3, 32, 48)
    assert out.dtype == np.float32
    assert np.allclose(out, 1.0)


def test_blob_return_seg():
    im = np.full((16, 16, 3), 127, dtype=np.uint8)
    out, seg = blob(im, return_seg=True)
    assert out.shape == (1, 3, 16, 16)
    assert seg.shape == (16, 16, 3)
    assert np.allclose(seg, 127 / 255)


def test_sigmoid_range():
    x = np.array([-30.0, 0.0, 30.0], dtype=np.float32)
    assert np.allclose(sigmoid(x), [0.0, 0.5, 1.0], atol=1e-6)


def test_box_iou():
    box = np.array([0, 0, 10, 10], dtype=np.float32)
    assert box_iou(box, box) == 1.0
    assert box_iou(box, np.array([20, 20, 30, 30], dtype=np.float32)) == 0.0


def test_nmsboxes_suppresses_overlap():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [100, 100, 110, 110]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    labels = np.array([0, 0, 0], dtype=np.int32)
    keep = NMSBoxes(boxes, scores, labels, iou_thres=0.5, agnostic=True)
    assert 0 in keep and 2 in keep and 1 not in keep  # box 1 overlaps box 0


def test_nmsboxes_class_aware_keeps_different_labels():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    labels = np.array([0, 1], dtype=np.int32)
    keep = NMSBoxes(boxes, scores, labels, iou_thres=0.5, agnostic=False)
    assert set(keep.tolist()) == {0, 1}  # different classes are not suppressed


def test_det_postprocess_truncates_to_num_dets():
    num_dets = np.array([[2]], dtype=np.int32)
    bboxes = np.array([[[0, 0, 1, 1], [2, 2, 3, 3], [4, 4, 5, 5]]], dtype=np.float32)
    scores = np.array([[0.9, 0.8, 0.7]], dtype=np.float32)
    labels = np.array([[1, 2, 3]], dtype=np.int32)
    b, s, le = det_postprocess((num_dets, bboxes, scores, labels))
    assert b.shape == (2, 4) and s.shape == (2,) and le.tolist() == [1, 2]


def test_det_postprocess_empty():
    num_dets = np.array([[0]], dtype=np.int32)
    z = np.zeros((1, 1, 4), dtype=np.float32)
    b, s, le = det_postprocess((num_dets, z, z[..., 0], z[..., 0].astype(np.int32)))
    assert b.shape == (0, 4) and s.shape == (0,) and le.shape == (0,)
