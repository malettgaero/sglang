"""Benchmark script for the scale kernel.

Compares performance of the custom CUDA kernel vs. PyTorch native implementation.
Usage:
    python benchmark.py
    python benchmark.py --dtype float16 --sizes 1024 4096 16384
"""

import argparse
import time

import torch

from . import scale
from .kernel_test import torch_scale


def benchmark_fn(fn, *args, warmup=20, iters=100, **kwargs):
    """Run a function repeatedly and return median latency in milliseconds."""
    # Warmup
    for _ in range(warmup):
        fn(*args, **kwargs)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        fn(*args, **kwargs)
    torch.cuda.synchronize()
    end = time.perf_counter()

    return (end - start) / iters * 1000  # ms


def run_benchmark(size: int, dtype: torch.dtype, scalar: float = 2.0):
    """Benchmark scale kernel vs torch for a given tensor size."""
    x = torch.randn(size, dtype=dtype, device="cuda")

    sgl_ms = benchmark_fn(scale, x, scalar)
    torch_ms = benchmark_fn(torch_scale, x, scalar)

    speedup = torch_ms / sgl_ms
    print(
        f"size={size:>10,}  dtype={str(dtype):<20}  "
        f"sgl={sgl_ms:.4f}ms  torch={torch_ms:.4f}ms  speedup={speedup:.2f}x"
    )


def main():
    parser = argparse.ArgumentParser(description="Benchmark scale kernel")
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[1024, 4096, 16384, 65536, 262144, 1048576],
        help="Tensor sizes to benchmark",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="float32",
        help="Data type for tensors",
    )
    parser.add_argument("--scalar", type=float, default=2.0, help="Scale factor")
    args = parser.parse_args()

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    if not torch.cuda.is_available():
        print("CUDA is not available. Skipping benchmark.")
        return

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Scalar: {args.scalar}")
    print("-" * 80)

    for size in args.sizes:
        run_benchmark(size, dtype, args.scalar)


if __name__ == "__main__":
    main()
