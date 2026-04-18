"""Profiling script for the scale kernel using PyTorch profiler and NSight.

Usage:
    python kernel_profile.py --mode torch   # PyTorch profiler trace
    python kernel_profile.py --mode nsight  # NSight Systems markers only
"""

import argparse
import torch
from pathlib import Path

from . import scale


def _make_tensor(shape, dtype=torch.float32, device="cuda"):
    return torch.randn(shape, dtype=dtype, device=device)


def profile_torch(size: int, dtype: torch.dtype, warmup: int, steps: int, out_dir: str):
    """Run PyTorch profiler and export a Chrome trace."""
    x = _make_tensor((size,), dtype=dtype)
    factor = 2.0

    # Warmup
    for _ in range(warmup):
        scale(x, factor)
    torch.cuda.synchronize()

    trace_path = Path(out_dir) / f"scale_trace_{dtype}_n{size}.json"
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        with_flops=True,
    ) as prof:
        for _ in range(steps):
            with torch.profiler.record_function("scale_kernel"):
                scale(x, factor)
        torch.cuda.synchronize()

    prof.export_chrome_trace(str(trace_path))
    print(f"[torch profiler] trace saved to {trace_path}")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


def profile_nsight(size: int, dtype: torch.dtype, warmup: int, steps: int):
    """Emit NVTX ranges for NSight Systems / NSight Compute capture."""
    try:
        import nvtx
        _nvtx = nvtx
    except ImportError:
        import contextlib

        class _Stub:
            @staticmethod
            def annotate(msg, color=None):
                return contextlib.nullcontext()

        _nvtx = _Stub()
        print("[nsight] nvtx package not found – ranges will not appear in NSight.")

    x = _make_tensor((size,), dtype=dtype)
    factor = 2.0

    for _ in range(warmup):
        scale(x, factor)
    torch.cuda.synchronize()

    torch.cuda.cudart().cudaProfilerStart()
    for i in range(steps):
        with _nvtx.annotate(f"scale_iter_{i}", color="blue"):
            scale(x, factor)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    print(f"[nsight] completed {steps} iterations – attach NSight to capture ranges.")


def parse_args():
    p = argparse.ArgumentParser(description="Profile the scale kernel")
    p.add_argument("--mode", choices=["torch", "nsight"], default="torch")
    p.add_argument("--size", type=int, default=4 * 1024 * 1024, help="Number of elements")
    p.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp32")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--out-dir", default=".", help="Directory for trace files (torch mode)")
    return p.parse_args()


def main():
    args = parse_args()

    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required for profiling.")

    print(f"Profiling scale kernel | mode={args.mode} size={args.size} dtype={args.dtype}")

    if args.mode == "torch":
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        profile_torch(args.size, dtype, args.warmup, args.steps, args.out_dir)
    else:
        profile_nsight(args.size, dtype, args.warmup, args.steps)


if __name__ == "__main__":
    main()
