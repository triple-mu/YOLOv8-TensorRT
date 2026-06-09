import argparse

from models import EngineBuilder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True, help="Weights file")
    parser.add_argument("--iou-thres", type=float, default=0.65, help="IOU threshoud for NMS plugin")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="CONF threshoud for NMS plugin")
    parser.add_argument("--topk", type=int, default=100, help="Max number of detection bboxes")
    parser.add_argument(
        "--input-shape", nargs="+", type=int, default=[1, 3, 640, 640], help="Model input shape only for api builder"
    )
    parser.add_argument("--fp16", action="store_true", help="Build model with fp16 mode")
    parser.add_argument("--device", type=str, default="cuda:0", help="TensorRT builder device")
    parser.add_argument("--seg", action="store_true", help="Build seg model by onnx")
    args = parser.parse_args()
    assert len(args.input_shape) == 4
    return args


def main(args):
    builder = EngineBuilder(args.weights, args.device)
    builder.seg = args.seg
    builder.build(
        fp16=args.fp16,
        input_shape=args.input_shape,
        iou_thres=args.iou_thres,
        conf_thres=args.conf_thres,
        topk=args.topk,
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
