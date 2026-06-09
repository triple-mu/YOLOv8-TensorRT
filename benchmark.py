"""Throughput / latency benchmark for a serialized TensorRT engine.

Times the full host-to-host inference call (H2D + execute + D2H) over the
torch-free cudart backend, so it works on any engine without PyTorch.

    python benchmark.py --engine yolov8s.engine --runs 200
"""

import argparse
import time

import numpy as np

from models.backend import CudartBackend


def main(args: argparse.Namespace) -> None:
    engine = CudartBackend(args.engine)
    info = engine.inp_info[0]
    shape = tuple(1 if d < 0 else d for d in info.shape)
    x = np.zeros(shape, dtype=info.dtype)

    for _ in range(args.warmup):
        engine(x)

    times = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        engine(x)
        times.append((time.perf_counter() - t0) * 1e3)

    t = np.asarray(times)
    print(f"engine: {args.engine}  input: {shape}")
    print(f"runs={args.runs}  warmup={args.warmup}")
    print(f"latency  mean={t.mean():.3f} ms  p50={np.percentile(t, 50):.3f}  p99={np.percentile(t, 99):.3f}")
    print(f"throughput  {1000.0 / t.mean():.1f} qps")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark a TensorRT engine")
    parser.add_argument("--engine", required=True, type=str, help="Engine file")
    parser.add_argument("--runs", type=int, default=200, help="Timed iterations")
    parser.add_argument("--warmup", type=int, default=50, help="Warmup iterations")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
