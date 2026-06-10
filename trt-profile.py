from models import TRTModule, TRTProfilerV0  # isort:skip
import argparse

import torch


def profile(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    Engine = TRTModule(args.engine, device)
    profiler = TRTProfilerV0()
    Engine.set_profiler(profiler)
    random_input = torch.randn(Engine.inp_info[0].shape, device=device)
    _ = Engine(random_input)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=str, help="Engine file")
    parser.add_argument("--device", type=str, default="cuda:0", help="TensorRT infer device")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    profile(args)
