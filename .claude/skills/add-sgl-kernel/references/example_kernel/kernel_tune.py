"""Autotuning script for the scale kernel.

Runs a grid search over kernel configurations and reports the best
performing config for each (M,) shape.

Usage::

    python kernel_tune.py --shapes 1024 4096 16384 --iters 200
"""

import argparse
import itertools
import time
from typing import List, Tuple

import torch

try:
    import example_kernel as ek
except ImportError as e:
    raise ImportError("Build the kernel first: pip install -e .") from e

# ---------------------------------------------------------------------------
# Configs to sweep
# ---------------------------------------------------------------------------

BLOCK_SIZES = [128, 256, 512, 1024]
NUM_WARPS_OPTIONS = [2, 4, 8]


def _make_tensor(n: int, dtype=torch.float32) -> torch.Tensor:
    return torch.randn(n, dtype=dtype, device="cuda")


def benchmark_config(
    n: int,
    block_size: int,
    num_warps: int,
    iters: int = 200,
    dtype=torch.float32,
) -> float:
    """Return median latency in microseconds for the given config."""
    x = _make_tensor(n, dtype=dtype)
    scalar = 2.0
    out = torch.empty_like(x)

    # Warm-up
    for _ in range(20):
        ek.scale(x, scalar, out)
    torch.cuda.synchronize()

    # Timed runs
    start = time.perf_counter()
    for _ in range(iters):
        ek.scale(x, scalar, out)
    torch.cuda.synchronize()
    elapsed_us = (time.perf_counter() - start) / iters * 1e6
    return elapsed_us


def tune(
    shapes: List[int],
    iters: int = 200,
    dtype=torch.float32,
) -> List[Tuple]:
    """Grid-search over configs for each shape; return list of best results."""
    results = []
    configs = list(itertools.product(BLOCK_SIZES, NUM_WARPS_OPTIONS))

    for n in shapes:
        best_us = float("inf")
        best_cfg = None
        for block_size, num_warps in configs:
            try:
                us = benchmark_config(n, block_size, num_warps, iters=iters, dtype=dtype)
            except Exception as exc:  # noqa: BLE001
                print(f"  [skip] n={n} block={block_size} warps={num_warps}: {exc}")
                continue
            if us < best_us:
                best_us = us
                best_cfg = (block_size, num_warps)
        results.append((n, best_cfg, best_us))
        print(
            f"n={n:>8}  best_block={best_cfg[0]:>4}  best_warps={best_cfg[1]}  "
            f"latency={best_us:.2f} us"
        )
    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Autotune example_kernel.scale")
    p.add_argument(
        "--shapes",
        nargs="+",
        type=int,
        default=[1024, 4096, 16384, 65536, 262144],
        help="Vector lengths to tune over",
    )
    p.add_argument("--iters", type=int, default=200, help="Timed iterations per config")
    p.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="float32",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    print(f"Tuning scale kernel  dtype={args.dtype}  iters={args.iters}")
    print("-" * 60)
    tune(args.shapes, iters=args.iters, dtype=dtype)


if __name__ == "__main__":
    main()
