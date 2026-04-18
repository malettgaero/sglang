"""Benchmark script for the scale kernel.

Compares performance of the custom CUDA kernel against a pure PyTorch
implementation across various tensor sizes and dtypes.

Usage:
    python kernel_bench.py
    python kernel_bench.py --dtype float16 --warmup 50 --iters 200
"""

import argparse
import time
from typing import Callable

import torch

from . import scale, scale_inplace


def torch_scale(x: torch.Tensor, factor: float) -> torch.Tensor:
    """Reference PyTorch implementation."""
    return x * factor


def benchmark_fn(
    fn: Callable,
    *args,
    warmup: int = 25,
    iters: int = 100,
    **kwargs,
) -> float:
    """Run a function repeatedly and return median latency in microseconds."""
    # Warmup
    for _ in range(warmup):
        fn(*args, **kwargs)
    torch.cuda.synchronize()

    # Timed runs
    start = time.perf_counter()
    for _ in range(iters):
        fn(*args, **kwargs)
    torch.cuda.synchronize()
    end = time.perf_counter()

    elapsed_us = (end - start) / iters * 1e6
    return elapsed_us


def run_benchmark(
    sizes: list[int],
    dtype: torch.dtype,
    warmup: int,
    iters: int,
) -> None:
    """Run benchmarks across tensor sizes and print a formatted table."""
    factor = 2.0
    device = "cuda"

    print(f"\nBenchmark: scale kernel  |  dtype={dtype}  |  device={device}")
    print(f"{'Size':>12}  {'PyTorch (us)':>14}  {'sgl-kernel (us)':>16}  {'Speedup':>8}")
    print("-" * 58)

    for size in sizes:
        x = torch.randn(size, dtype=dtype, device=device)

        torch_us = benchmark_fn(torch_scale, x, factor, warmup=warmup, iters=iters)
        sgl_us = benchmark_fn(scale, x, factor, warmup=warmup, iters=iters)
        speedup = torch_us / sgl_us

        print(f"{size:>12,}  {torch_us:>14.2f}  {sgl_us:>16.2f}  {speedup:>8.2f}x")

    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the scale kernel.")
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="float32",
        help="Tensor dtype to benchmark (default: float32)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=25,
        help="Number of warmup iterations (default: 25)",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=100,
        help="Number of timed iterations (default: 100)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    sizes = [1_024, 16_384, 131_072, 1_048_576, 8_388_608, 67_108_864]

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run benchmarks.")

    run_benchmark(sizes, dtype=dtype, warmup=args.warmup, iters=args.iters)


if __name__ == "__main__":
    main()
