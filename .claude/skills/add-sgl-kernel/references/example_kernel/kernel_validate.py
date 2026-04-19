"""Validation script for the scale kernel.

Runs correctness checks and shape validation against a PyTorch reference
implementation. Use this to verify the kernel behaves correctly before
running benchmarks.

Usage:
    python kernel_validate.py
    python kernel_validate.py --dtype float16
    python kernel_validate.py --verbose
"""

import argparse
from typing import Optional

import torch

from kernel import scale, scale_inplace


def torch_scale(x: torch.Tensor, s: float) -> torch.Tensor:
    """Reference PyTorch implementation of scale."""
    return x * s


def _make_tensor(
    shape: tuple,
    dtype: torch.dtype = torch.float32,
    device: str = "cuda",
) -> torch.Tensor:
    return torch.randn(shape, dtype=dtype, device=device)


def check(
    actual: torch.Tensor,
    expected: torch.Tensor,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    label: str = "",
) -> bool:
    """Compare two tensors and return True if they are close."""
    if not torch.allclose(actual, expected.to(actual.dtype), atol=atol, rtol=rtol):
        max_diff = (actual - expected.to(actual.dtype)).abs().max().item()
        print(f"  FAIL {label}: max_diff={max_diff:.3e} (atol={atol}, rtol={rtol})")
        return False
    return True


def validate_correctness(dtype: torch.dtype = torch.float32, verbose: bool = False) -> bool:
    """Validate scale output matches PyTorch reference."""
    shapes = [(1024,), (128, 256), (4, 32, 64)]
    scalars = [0.5, 1.0, -2.0, 0.0]
    passed = 0
    total = 0

    for shape in shapes:
        for s in scalars:
            x = _make_tensor(shape, dtype=dtype)
            expected = torch_scale(x, s)
            actual = scale(x, s)
            label = f"shape={shape} s={s} dtype={dtype}"
            ok = check(actual, expected, label=label)
            if verbose:
                status = "PASS" if ok else "FAIL"
                print(f"  [{status}] {label}")
            passed += int(ok)
            total += 1

    print(f"validate_correctness: {passed}/{total} passed")
    return passed == total


def validate_shapes(verbose: bool = False) -> bool:
    """Validate that output shape matches input shape."""
    shapes = [(1,), (100,), (8, 16), (2, 4, 8, 16)]
    passed = 0

    for shape in shapes:
        x = _make_tensor(shape)
        out = scale(x, 2.0)
        ok = out.shape == x.shape
        if verbose:
            status = "PASS" if ok else "FAIL"
            print(f"  [{status}] shape={shape} -> out.shape={out.shape}")
        passed += int(ok)

    print(f"validate_shapes: {passed}/{len(shapes)} passed")
    return passed == len(shapes)


def validate_inplace(verbose: bool = False) -> bool:
    """Validate that scale_inplace modifies the tensor in-place."""
    x = _make_tensor((256,))
    x_orig = x.clone()
    ptr_before = x.data_ptr()
    scale_inplace(x, 3.0)
    ptr_after = x.data_ptr()

    same_ptr = ptr_before == ptr_after
    correct = torch.allclose(x, x_orig * 3.0, atol=1e-5)
    ok = same_ptr and correct

    if verbose:
        print(f"  same data_ptr: {same_ptr}")
        print(f"  correct values: {correct}")

    status = "PASS" if ok else "FAIL"
    print(f"validate_inplace: [{status}]")
    return ok


def validate_dtypes(verbose: bool = False) -> bool:
    """Validate supported dtypes."""
    dtypes = [torch.float32, torch.float16, torch.bfloat16]
    passed = 0

    for dtype in dtypes:
        try:
            x = _make_tensor((512,), dtype=dtype)
            out = scale(x, 0.5)
            ok = out.dtype == dtype
        except Exception as e:
            if verbose:
                print(f"  FAIL dtype={dtype}: {e}")
            ok = False

        if verbose:
            status = "PASS" if ok else "FAIL"
            print(f"  [{status}] dtype={dtype}")
        passed += int(ok)

    print(f"validate_dtypes: {passed}/{len(dtypes)} passed")
    return passed == len(dtypes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate scale kernel")
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="float32",
        help="Primary dtype for correctness validation",
    )
    parser.add_argument("--verbose", action="store_true", help="Print per-case results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for kernel validation")

    results = [
        validate_correctness(dtype=dtype, verbose=args.verbose),
        validate_shapes(verbose=args.verbose),
        validate_inplace(verbose=args.verbose),
        validate_dtypes(verbose=args.verbose),
    ]

    total = len(results)
    passed = sum(results)
    print(f"\nOverall: {passed}/{total} validation suites passed")
    if passed < total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
