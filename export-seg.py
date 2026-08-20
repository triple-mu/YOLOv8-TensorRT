import argparse
from io import BytesIO

import onnx
import torch
from ultralytics import YOLO

from models.common import PostSeg, optim

try:
    import onnxsim
except ImportError:
    onnxsim = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--weights", type=str, required=True, help="PyTorch yolov8 weights")
    parser.add_argument("--opset", type=int, default=11, help="ONNX opset version")
    parser.add_argument("--sim", action="store_true", help="simplify onnx model")
    parser.add_argument(
        "--input-shape", nargs="+", type=int, default=[1, 3, 640, 640], help="Model input shape only for api builder"
    )
    parser.add_argument("--device", type=str, default="cpu", help="Export ONNX device")
    parser.add_argument(
        "--plugin", action="store_true", help="Emit the YoloSegPostprocess plugin (decode+NMS+mask coeffs in-engine)"
    )
    parser.add_argument("--iou-thres", type=float, default=0.65, help="IoU threshold (plugin)")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="Score threshold (plugin)")
    parser.add_argument("--topk", type=int, default=100, help="Max detections (plugin)")
    args = parser.parse_args()
    if len(args.input_shape) != 4:
        raise ValueError(f"--input-shape needs 4 values, got {len(args.input_shape)}")
    PostSeg.plugin = args.plugin
    PostSeg.iou_thres = args.iou_thres
    PostSeg.conf_thres = args.conf_thres
    PostSeg.topk = args.topk
    return args


def main(args: argparse.Namespace) -> None:
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
    output_names = (
        ["num_dets", "bboxes", "scores", "labels", "mask_coeffs", "proto"] if args.plugin else ["outputs", "proto"]
    )
    with BytesIO() as f:
        torch.onnx.export(
            model,
            fake_input,
            f,
            opset_version=args.opset,
            input_names=["images"],
            output_names=output_names,
            dynamo=False,
        )
        f.seek(0)
        onnx_model = onnx.load(f)
    onnx.checker.check_model(onnx_model)
    if args.sim:
        try:
            onnx_model, check = onnxsim.simplify(onnx_model)
            if not check:
                raise RuntimeError("onnxsim simplification check failed")
        except Exception as e:
            print(f"Simplifier failure: {e}")
    onnx.save(onnx_model, save_path)
    print(f"ONNX export success, saved as {save_path}")


if __name__ == "__main__":
    main(parse_args())
