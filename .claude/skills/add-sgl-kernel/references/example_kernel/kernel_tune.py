"""Auto-tuning script for the scale kernel.

Runs a grid search over block sizes and other launch parameters,
reports the best configuration for each (dtype, size) combination.

Usage:
    python kernel_tune.py
    python kernel_tune.py --dtype float16 --sizes 1048576 4194304
"""

import argparse
import itertools
from typing import List, Tuple

import torch

try:
    from . import scale as _scale_op
except ImportError:
    from kernel import scale as _scale_op


DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

# Tuning grid
BLOCK_SIZES = [128, 256, 512, 1024]
ELEMS_PER_THREAD = [1, 2, 4, 8]
NUM_WARMUP = 5
NUM_ITERS = 50


def _make_tensor(numel: int, dtype: torch.dtype) -> torch.Tensor:
    return torch.randn(numel, dtype=dtype, device="cuda")


def benchmark_config(
    x: torch.Tensor,
    scalar: float,
    block_size: int,
    elems_per_thread: int,
) -> float:
    """Return median latency in microseconds for a given config."""
    out = torch.empty_like(x)

    # Warmup
    for _ in range(NUM_WARMUP):
        _scale_op(x, scalar, out=out, block_size=block_size, elems_per_thread=elems_per_thread)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(NUM_ITERS):
        _scale_op(x, scalar, out=out, block_size=block_size, elems_per_thread=elems_per_thread)
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) * 1e3 / NUM_ITERS  # µs


def tune(
    sizes: List[int],
    dtypes: List[torch.dtype],
) -> dict:
    """Grid-search over configs; return best per (dtype, size)."""
    results = {}

    for dtype, numel in itertools.product(dtypes, sizes):
        x = _make_tensor(numel, dtype)
        best_time = float("inf")
        best_cfg: Tuple[int, int] = (BLOCK_SIZES[0], ELEMS_PER_THREAD[0])

        for bs, ept in itertools.product(BLOCK_SIZES, ELEMS_PER_THREAD):
            try:
                t = benchmark_config(x, 2.0, bs, ept)
            except Exception:
                # Some configs may be invalid for small tensors
                continue
            if t < best_time:
                best_time = t
                best_cfg = (bs, ept)

        key = (str(dtype).split(".")[1], numel)
        results[key] = {"block_size": best_cfg[0], "elems_per_thread": best_cfg[1], "us": round(best_time, 3)}
        print(f"  dtype={key[0]:>10s}  numel={numel:>10,}  best_block={best_cfg[0]:>5}  ept={best_cfg[1]}  latency={best_time:.3f} µs")

    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tune scale kernel launch parameters")
    p.add_argument("--dtype", nargs="+", default=list(DTYPE_MAP.keys()), choices=list(DTYPE_MAP.keys()))
    p.add_argument("--sizes", nargs="+", type=int, default=[65536, 262144, 1048576, 4194304])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dtypes = [DTYPE_MAP[d] for d in args.dtype]

    print(f"Tuning on {torch.cuda.get_device_name(0)}")
    print(f"dtypes={args.dtype}  sizes={args.sizes}\n")

    results = tune(args.sizes, dtypes)

    print("\n=== Summary ===")
    for (dtype, numel), cfg in results.items():
        print(f"  {dtype:>10s}  {numel:>10,}  -> block_size={cfg['block_size']}, elems_per_thread={cfg['elems_per_thread']}  ({cfg['us']} µs)")


if __name__ == "__main__":
    main()
