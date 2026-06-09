import argparse
from io import BytesIO

import onnx
import torch
from ultralytics import YOLO

from models.common import PostDetect, optim

try:
    import onnxsim
except ImportError:
    onnxsim = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--weights", type=str, required=True, help="PyTorch yolov8 weights")
    parser.add_argument("--iou-thres", type=float, default=0.65, help="IOU threshoud for NMS plugin")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="CONF threshoud for NMS plugin")
    parser.add_argument("--topk", type=int, default=100, help="Max number of detection bboxes")
    parser.add_argument("--opset", type=int, default=11, help="ONNX opset version")
    parser.add_argument("--sim", action="store_true", help="simplify onnx model")
    parser.add_argument(
        "--input-shape", nargs="+", type=int, default=[1, 3, 640, 640], help="Model input shape only for api builder"
    )
    parser.add_argument("--device", type=str, default="cpu", help="Export ONNX device")
    args = parser.parse_args()
    assert len(args.input_shape) == 4
    PostDetect.conf_thres = args.conf_thres
    PostDetect.iou_thres = args.iou_thres
    PostDetect.topk = args.topk
    return args


def main(args):
    b = args.input_shape[0]
    YOLOv8 = YOLO(args.weights)
    model = YOLOv8.model.fuse().eval()
    for m in model.modules():
        optim(m)
        m.to(args.device)
    model.to(args.device)
    fake_input = torch.randn(args.input_shape).to(args.device)
    for _ in range(2):
        model(fake_input)
    save_path = args.weights.replace(".pt", ".onnx")
    with BytesIO() as f:
        torch.onnx.export(
            model,
            fake_input,
            f,
            opset_version=args.opset,
            input_names=["images"],
            output_names=["num_dets", "bboxes", "scores", "labels"],
            dynamo=False,
        )
        f.seek(0)
        onnx_model = onnx.load(f)
    onnx.checker.check_model(onnx_model)
    shapes = [b, 1, b, args.topk, 4, b, args.topk, b, args.topk]
    for i in onnx_model.graph.output:
        for j in i.type.tensor_type.shape.dim:
            j.dim_param = str(shapes.pop(0))
    if args.sim:
        try:
            onnx_model, check = onnxsim.simplify(onnx_model)
            assert check, "assert check failed"
        except Exception as e:
            print(f"Simplifier failure: {e}")
    onnx.save(onnx_model, save_path)
    print(f"ONNX export success, saved as {save_path}")


if __name__ == "__main__":
    main(parse_args())
