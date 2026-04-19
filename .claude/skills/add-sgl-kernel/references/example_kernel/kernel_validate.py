"""Validation script for the scale kernel.

Runs correctness checks and shape/dtype validation against the PyTorch reference
implementation. Useful for quick sanity checks after modifying the kernel.

Usage:
    python kernel_validate.py
    python kernel_validate.py --verbose
    python kernel_validate.py --shapes 128,256 512,1024
"""

import argparse
import sys
from typing import List, Tuple

import torch

from . import scale, scale_inplace


def torch_scale(x: torch.Tensor, factor: float) -> torch.Tensor:
    """Reference PyTorch implementation."""
    return x * factor


def _make_tensor(
    shape: Tuple[int, ...],
    dtype: torch.dtype = torch.float32,
    device: str = "cuda",
) -> torch.Tensor:
    return torch.randn(shape, dtype=dtype, device=device)


def check(
    result: torch.Tensor,
    expected: torch.Tensor,
    label: str,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    verbose: bool = False,
) -> bool:
    """Compare two tensors and print a pass/fail message."""
    ok = torch.allclose(result, expected, atol=atol, rtol=rtol)
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label}")
    if not ok and verbose:
        diff = (result - expected).abs()
        print(f"         max_diff={diff.max().item():.3e}  mean_diff={diff.mean().item():.3e}")
    return ok


def validate_correctness(verbose: bool = False) -> bool:
    """Check numerical correctness for common shapes and dtypes."""
    print("Correctness validation")
    passed = True
    configs = [
        ((1024,), torch.float32, 2.0),
        ((512, 512), torch.float32, 0.5),
        ((128, 128, 4), torch.float32, -1.0),
        ((1024,), torch.float16, 2.0),
        ((512, 512), torch.bfloat16, 3.14),
    ]
    for shape, dtype, factor in configs:
        x = _make_tensor(shape, dtype=dtype)
        expected = torch_scale(x, factor)
        result = scale(x, factor)
        label = f"scale  shape={shape} dtype={dtype} factor={factor}"
        passed &= check(result, expected, label, atol=1e-3, rtol=1e-3, verbose=verbose)

        # inplace variant
        x_ip = x.clone()
        scale_inplace(x_ip, factor)
        label_ip = f"scale_inplace  shape={shape} dtype={dtype} factor={factor}"
        passed &= check(x_ip, expected, label_ip, atol=1e-3, rtol=1e-3, verbose=verbose)

    return passed


def validate_shapes(shapes: List[Tuple[int, ...]], verbose: bool = False) -> bool:
    """Validate a user-supplied list of shapes."""
    print("Custom shape validation")
    passed = True
    factor = 1.5
    for shape in shapes:
        x = _make_tensor(shape)
        expected = torch_scale(x, factor)
        result = scale(x, factor)
        label = f"scale  shape={shape}"
        passed &= check(result, expected, label, verbose=verbose)
    return passed


def validate_edge_cases(verbose: bool = False) -> bool:
    """Check edge cases: zero tensor, single element, large values."""
    print("Edge-case validation")
    passed = True

    # zero tensor
    x = torch.zeros(256, device="cuda")
    passed &= check(scale(x, 99.0), x, "zero tensor", verbose=verbose)

    # single element
    x = torch.tensor([3.0], device="cuda")
    passed &= check(scale(x, 2.0), torch.tensor([6.0], device="cuda"), "single element", verbose=verbose)

    # large values (overflow check for fp16 skipped intentionally)
    x = _make_tensor((1024,), dtype=torch.float32)
    x = x * 1e4
    passed &= check(scale(x, 1.0), torch_scale(x, 1.0), "large values factor=1", verbose=verbose)

    return passed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate the scale kernel")
    p.add_argument("--verbose", action="store_true", help="Print diff stats on failure")
    p.add_argument(
        "--shapes",
        nargs="+",
        metavar="DIM,DIM",
        help="Extra shapes to validate, e.g. 128,256 1024,1024",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available — skipping validation.")
        sys.exit(1)

    results = []
    results.append(validate_correctness(verbose=args.verbose))
    results.append(validate_edge_cases(verbose=args.verbose))

    if args.shapes:
        parsed = [tuple(int(d) for d in s.split(",")) for s in args.shapes]
        results.append(validate_shapes(parsed, verbose=args.verbose))

    print()
    if all(results):
        print("All validations PASSED.")
    else:
        print("Some validations FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
