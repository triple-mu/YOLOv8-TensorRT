import warnings
from collections import OrderedDict

import numpy as np
import tensorrt as trt

warnings.filterwarnings(action="ignore", category=DeprecationWarning)


def trtweight(weights: np.ndarray) -> trt.Weights:
    weights = weights.astype(weights.dtype.name)
    return trt.Weights(weights)


def get_width(x: int, gw: float, divisor: int = 8) -> int:
    return int(np.ceil(x * gw / divisor) * divisor)


def get_depth(x: int, gd: float) -> int:
    return max(int(round(x * gd)), 1)


def Conv2d(
    network: trt.INetworkDefinition,
    weights: OrderedDict,
    input: trt.ITensor,
    out_channel: int,
    ksize: int,
    stride: int,
    group: int,
    layer_name: str,
) -> trt.ILayer:
    padding = ksize // 2
    conv_w = trtweight(weights[layer_name + ".weight"])
    conv_b = trtweight(weights[layer_name + ".bias"])
    conv = network.add_convolution_nd(
        input, num_output_maps=out_channel, kernel_shape=trt.DimsHW(ksize, ksize), kernel=conv_w, bias=conv_b
    )
    assert conv, "Add convolution_nd layer failed"
    conv.stride_nd = trt.DimsHW(stride, stride)
    conv.padding_nd = trt.DimsHW(padding, padding)
    conv.num_groups = group
    return conv


def Conv(
    network: trt.INetworkDefinition,
    weights: OrderedDict,
    input: trt.ITensor,
    out_channel: int,
    ksize: int,
    stride: int,
    group: int,
    layer_name: str,
) -> trt.ILayer:
    padding = ksize // 2
    if ksize > 3:
        padding -= 1
    conv_w = trtweight(weights[layer_name + ".conv.weight"])
    conv_b = trtweight(weights[layer_name + ".conv.bias"])

    conv = network.add_convolution_nd(
        input, num_output_maps=out_channel, kernel_shape=trt.DimsHW(ksize, ksize), kernel=conv_w, bias=conv_b
    )
    assert conv, "Add convolution_nd layer failed"
    conv.stride_nd = trt.DimsHW(stride, stride)
    conv.padding_nd = trt.DimsHW(padding, padding)
    conv.num_groups = group

    sigmoid = network.add_activation(conv.get_output(0), trt.ActivationType.SIGMOID)
    assert sigmoid, "Add activation layer failed"
    dot_product = network.add_elementwise(conv.get_output(0), sigmoid.get_output(0), trt.ElementWiseOperation.PROD)
    assert dot_product, "Add elementwise layer failed"
    return dot_product


def Bottleneck(
    network: trt.INetworkDefinition,
    weights: OrderedDict,
    input: trt.ITensor,
    c1: int,
    c2: int,
    shortcut: bool,
    group: int,
    scale: float,
    layer_name: str,
) -> trt.ILayer:
    c_ = int(c2 * scale)
    conv1 = Conv(network, weights, input, c_, 3, 1, 1, layer_name + ".cv1")
    conv2 = Conv(network, weights, conv1.get_output(0), c2, 3, 1, group, layer_name + ".cv2")
    if shortcut and c1 == c2:
        ew = network.add_elementwise(input, conv2.get_output(0), op=trt.ElementWiseOperation.SUM)
        assert ew, "Add elementwise layer failed"
        return ew
    return conv2


def C2f(
    network: trt.INetworkDefinition,
    weights: OrderedDict,
    input: trt.ITensor,
    cout: int,
    n: int,
    shortcut: bool,
    group: int,
    scale: float,
    layer_name: str,
) -> trt.ILayer:
    c_ = int(cout * scale)  # e:expand param
    conv1 = Conv(network, weights, input, 2 * c_, 1, 1, 1, layer_name + ".cv1")
    y1 = conv1.get_output(0)

    b, _, h, w = y1.shape
    slice = network.add_slice(y1, (0, c_, 0, 0), (b, c_, h, w), (1, 1, 1, 1))
    assert slice, "Add slice layer failed"
    y2 = slice.get_output(0)

    input_tensors = [y1]
    for i in range(n):
        b = Bottleneck(network, weights, y2, c_, c_, shortcut, group, 1.0, layer_name + ".m." + str(i))
        y2 = b.get_output(0)
        input_tensors.append(y2)

    cat = network.add_concatenation(input_tensors)
    assert cat, "Add concatenation layer failed"

    conv2 = Conv(network, weights, cat.get_output(0), cout, 1, 1, 1, layer_name + ".cv2")
    return conv2


def SPPF(
    network: trt.INetworkDefinition,
    weights: OrderedDict,
    input: trt.ITensor,
    c1: int,
    c2: int,
    ksize: int,
    layer_name: str,
) -> trt.ILayer:
    c_ = c1 // 2
    conv1 = Conv(network, weights, input, c_, 1, 1, 1, layer_name + ".cv1")

    pool1 = network.add_pooling_nd(conv1.get_output(0), trt.PoolingType.MAX, trt.DimsHW(ksize, ksize))
    assert pool1, "Add pooling_nd layer failed"
    pool1.padding_nd = trt.DimsHW(ksize // 2, ksize // 2)
    pool1.stride_nd = trt.DimsHW(1, 1)

    pool2 = network.add_pooling_nd(pool1.get_output(0), trt.PoolingType.MAX, trt.DimsHW(ksize, ksize))
    assert pool2, "Add pooling_nd layer failed"
    pool2.padding_nd = trt.DimsHW(ksize // 2, ksize // 2)
    pool2.stride_nd = trt.DimsHW(1, 1)

    pool3 = network.add_pooling_nd(pool2.get_output(0), trt.PoolingType.MAX, trt.DimsHW(ksize, ksize))
    assert pool3, "Add pooling_nd layer failed"
    pool3.padding_nd = trt.DimsHW(ksize // 2, ksize // 2)
    pool3.stride_nd = trt.DimsHW(1, 1)

    input_tensors = [conv1.get_output(0), pool1.get_output(0), pool2.get_output(0), pool3.get_output(0)]
    cat = network.add_concatenation(input_tensors)
    assert cat, "Add concatenation layer failed"
    conv2 = Conv(network, weights, cat.get_output(0), c2, 1, 1, 1, layer_name + ".cv2")
    return conv2


def Detect(
    network: trt.INetworkDefinition,
    weights: OrderedDict,
    input: list | tuple,
    s: list | tuple,
    layer_name: str,
    reg_max: int = 16,
    fp16: bool = True,
    iou: float = 0.65,
    conf: float = 0.25,
    topk: int = 100,
) -> trt.ILayer:
    bboxes_branch = []
    scores_branch = []
    anchors = []
    strides = []
    for i, (inp, stride) in enumerate(zip(input, s)):
        h, w = inp.shape[2:]
        sx = np.arange(0, w).astype(np.float16 if fp16 else np.float32) + 0.5
        sy = np.arange(0, h).astype(np.float16 if fp16 else np.float32) + 0.5
        sy, sx = np.meshgrid(sy, sx)
        a = np.ascontiguousarray(np.stack((sy, sx), -1).reshape(-1, 2))
        anchors.append(a)
        strides.append(np.full((1, h * w), stride, dtype=np.float16 if fp16 else np.float32))
        c2 = weights[f"{layer_name}.cv2.{i}.0.conv.weight"].shape[0]
        c3 = weights[f"{layer_name}.cv3.{i}.0.conv.weight"].shape[0]
        nc = weights[f"{layer_name}.cv3.0.2.weight"].shape[0]
        reg_max_x4 = weights[layer_name + f".cv2.{i}.2.weight"].shape[0]
        assert reg_max_x4 == reg_max * 4
        b_Conv_0 = Conv(network, weights, inp, c2, 3, 1, 1, layer_name + f".cv2.{i}.0")
        b_Conv_1 = Conv(network, weights, b_Conv_0.get_output(0), c2, 3, 1, 1, layer_name + f".cv2.{i}.1")
        b_Conv_2 = Conv2d(network, weights, b_Conv_1.get_output(0), reg_max_x4, 1, 1, 1, layer_name + f".cv2.{i}.2")

        b_out = b_Conv_2.get_output(0)
        b_shape = network.add_constant(
            [
                4,
            ],
            np.array(b_out.shape[0:1] + (4, reg_max, -1), dtype=np.int32),
        )
        assert b_shape, "Add constant layer failed"
        b_shuffle = network.add_shuffle(b_out)
        assert b_shuffle, "Add shuffle layer failed"
        b_shuffle.set_input(1, b_shape.get_output(0))
        b_shuffle.second_transpose = (0, 3, 1, 2)

        bboxes_branch.append(b_shuffle.get_output(0))

        s_Conv_0 = Conv(network, weights, inp, c3, 3, 1, 1, layer_name + f".cv3.{i}.0")
        s_Conv_1 = Conv(network, weights, s_Conv_0.get_output(0), c3, 3, 1, 1, layer_name + f".cv3.{i}.1")
        s_Conv_2 = Conv2d(network, weights, s_Conv_1.get_output(0), nc, 1, 1, 1, layer_name + f".cv3.{i}.2")
        s_out = s_Conv_2.get_output(0)
        s_shape = network.add_constant(
            [
                3,
            ],
            np.array(s_out.shape[0:2] + (-1,), dtype=np.int32),
        )
        assert s_shape, "Add constant layer failed"
        s_shuffle = network.add_shuffle(s_out)
        assert s_shuffle, "Add shuffle layer failed"
        s_shuffle.set_input(1, s_shape.get_output(0))
        s_shuffle.second_transpose = (0, 2, 1)

        scores_branch.append(s_shuffle.get_output(0))

    Cat_bboxes = network.add_concatenation(bboxes_branch)
    assert Cat_bboxes, "Add concatenation layer failed"
    Cat_scores = network.add_concatenation(scores_branch)
    assert Cat_scores, "Add concatenation layer failed"
    Cat_scores.axis = 1

    Softmax = network.add_softmax(Cat_bboxes.get_output(0))
    assert Softmax, "Add softmax layer failed"
    Softmax.axes = 1 << 3

    SCORES = network.add_activation(Cat_scores.get_output(0), trt.ActivationType.SIGMOID)
    assert SCORES, "Add activation layer failed"

    reg_max = np.arange(0, reg_max).astype(np.float16 if fp16 else np.float32).reshape((1, 1, -1, 1))
    constant = network.add_constant(reg_max.shape, reg_max)
    assert constant, "Add constant layer failed"
    Matmul = network.add_matrix_multiply(
        Softmax.get_output(0), trt.MatrixOperation.NONE, constant.get_output(0), trt.MatrixOperation.NONE
    )
    assert Matmul, "Add matrix_multiply layer failed"
    pre_bboxes = network.add_gather(
        Matmul.get_output(0),
        network.add_constant(
            [
                1,
            ],
            np.array([0], dtype=np.int32),
        ).get_output(0),
        3,
    )
    assert pre_bboxes, "Add gather layer failed"
    pre_bboxes.num_elementwise_dims = 1

    pre_bboxes_tensor = pre_bboxes.get_output(0)
    b, c, _ = pre_bboxes_tensor.shape
    slice_x1y1 = network.add_slice(pre_bboxes_tensor, (0, 0, 0), (b, c, 2), (1, 1, 1))
    assert slice_x1y1, "Add slice layer failed"
    slice_x2y2 = network.add_slice(pre_bboxes_tensor, (0, 0, 2), (b, c, 2), (1, 1, 1))
    assert slice_x2y2, "Add slice layer failed"
    anchors = np.concatenate(anchors, 0)[np.newaxis]
    anchors = network.add_constant(anchors.shape, anchors)
    assert anchors, "Add constant layer failed"
    strides = np.concatenate(strides, 1)[..., np.newaxis]
    strides = network.add_constant(strides.shape, strides)
    assert strides, "Add constant layer failed"

    Sub = network.add_elementwise(anchors.get_output(0), slice_x1y1.get_output(0), trt.ElementWiseOperation.SUB)
    assert Sub, "Add elementwise layer failed"
    Add = network.add_elementwise(anchors.get_output(0), slice_x2y2.get_output(0), trt.ElementWiseOperation.SUM)
    assert Add, "Add elementwise layer failed"
    x1y1 = Sub.get_output(0)
    x2y2 = Add.get_output(0)

    Cat_bboxes_ = network.add_concatenation([x1y1, x2y2])
    assert Cat_bboxes_, "Add concatenation layer failed"
    Cat_bboxes_.axis = 2

    BBOXES = network.add_elementwise(Cat_bboxes_.get_output(0), strides.get_output(0), trt.ElementWiseOperation.PROD)
    assert BBOXES, "Add elementwise layer failed"
    plugin_creator = trt.get_plugin_registry().get_plugin_creator("EfficientNMS_TRT", "1")
    assert plugin_creator, "Plugin EfficientNMS_TRT is not registried"

    background_class = trt.PluginField("background_class", np.array(-1, np.int32), trt.PluginFieldType.INT32)
    box_coding = trt.PluginField("box_coding", np.array(0, np.int32), trt.PluginFieldType.INT32)
    iou_threshold = trt.PluginField("iou_threshold", np.array(iou, dtype=np.float32), trt.PluginFieldType.FLOAT32)
    max_output_boxes = trt.PluginField("max_output_boxes", np.array(topk, np.int32), trt.PluginFieldType.INT32)
    plugin_version = trt.PluginField("plugin_version", np.array("1"), trt.PluginFieldType.CHAR)
    score_activation = trt.PluginField("score_activation", np.array(0, np.int32), trt.PluginFieldType.INT32)
    score_threshold = trt.PluginField("score_threshold", np.array(conf, dtype=np.float32), trt.PluginFieldType.FLOAT32)

    batched_nms_op = plugin_creator.create_plugin(
        name="batched_nms",
        field_collection=trt.PluginFieldCollection(
            [
                background_class,
                box_coding,
                iou_threshold,
                max_output_boxes,
                plugin_version,
                score_activation,
                score_threshold,
            ]
        ),
    )

    batched_nms = network.add_plugin_v2(inputs=[BBOXES.get_output(0), SCORES.get_output(0)], plugin=batched_nms_op)

    batched_nms.get_output(0).name = "num_dets"
    batched_nms.get_output(1).name = "bboxes"
    batched_nms.get_output(2).name = "scores"
    batched_nms.get_output(3).name = "labels"

    return batched_nms
