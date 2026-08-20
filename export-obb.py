"""Export a YOLOv8-obb ONNX with the YoloObbPostprocess plugin node appended.

Like export-pose.py: the raw ultralytics obb output [1, C, A] (C = 4 + nc + 1: xc,yc,w,h
| nc scores | angle) is transposed to [1, A, C] and fed to the plugin via
onnx_graphsurgeon. The plugin does decode + rotated NMS + top-k and emits the kept
center boxes + angles. Build with libyolov8_plugins.so loaded.
"""

import argparse

import numpy as np
import onnx
import onnx_graphsurgeon as gs
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--weights", type=str, required=True, help="PyTorch yolov8-obb weights")
    parser.add_argument("--opset", type=int, default=11, help="ONNX opset version")
    parser.add_argument("--iou-thres", type=float, default=0.65, help="NMS IoU threshold")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="Score threshold")
    parser.add_argument("--topk", type=int, default=100, help="Max detections")
    parser.add_argument("--imgsz", type=int, default=1024, help="Square input size")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    onnx_path = YOLO(args.weights).export(format="onnx", opset=args.opset, simplify=True, imgsz=args.imgsz)

    graph = gs.import_onnx(onnx.load(onnx_path))
    raw = graph.outputs[0]  # [1, C, A] = (4 + nc + 1, anchors)
    c, a = int(raw.shape[1]), int(raw.shape[2])

    transposed = gs.Variable("obb_in", dtype=np.float32, shape=[1, a, c])
    graph.nodes.append(gs.Node("Transpose", attrs={"perm": [0, 2, 1]}, inputs=[raw], outputs=[transposed]))

    topk = args.topk
    num_dets = gs.Variable("num_dets", np.int32, [1, 1])
    bboxes = gs.Variable("bboxes", np.float32, [1, topk, 4])
    scores = gs.Variable("scores", np.float32, [1, topk])
    labels = gs.Variable("labels", np.int32, [1, topk])
    angles = gs.Variable("angles", np.float32, [1, topk, 1])
    node = gs.Node(
        "YoloObbPostprocess",
        name="YoloObbPostprocess_0",
        attrs={
            "score_threshold": float(args.conf_thres),
            "iou_threshold": float(args.iou_thres),
            "max_output_boxes": int(topk),
            "background_class": -1,
            "box_coding": 0,
            "score_activation": 0,
            "plugin_version": "1",
        },
        inputs=[transposed],
        outputs=[num_dets, bboxes, scores, labels, angles],
    )
    node.domain = "TRT"
    graph.nodes.append(node)
    graph.outputs = [num_dets, bboxes, scores, labels, angles]
    graph.cleanup().toposort()

    model = gs.export_onnx(graph)
    model.opset_import.append(onnx.helper.make_opsetid("TRT", 1))
    onnx.save(model, onnx_path)
    print(f"ONNX export success (plugin), saved as {onnx_path}")


if __name__ == "__main__":
    main(parse_args())
