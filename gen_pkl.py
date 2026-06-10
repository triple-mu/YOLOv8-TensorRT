import argparse
import pickle
from collections import OrderedDict

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--weights", type=str, required=True, help="YOLOv8 pytorch weights")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output file")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    model = YOLO(args.weights)
    model.model.fuse()
    yolov8 = model.model.model

    strides = yolov8[-1].stride.detach().cpu().numpy()
    reg_max = yolov8[-1].dfl.conv.weight.shape[1]

    state_dict = OrderedDict(
        GD=model.model.yaml["depth_multiple"],
        GW=model.model.yaml["width_multiple"],
        strides=strides,
        reg_max=reg_max,
    )

    for name, value in yolov8.state_dict().items():
        value = value.detach().cpu().numpy()
        i = int(name.split(".")[0])
        layer = yolov8[i]
        module_name = layer.type.split(".")[-1]
        state_dict[module_name + "." + name] = value

    with open(args.output, "wb") as f:
        pickle.dump(state_dict, f)


if __name__ == "__main__":
    main(parse_args())
