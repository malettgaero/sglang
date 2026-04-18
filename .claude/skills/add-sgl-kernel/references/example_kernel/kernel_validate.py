"""Validation script for the scale kernel.

Runs a suite of correctness checks comparing the custom CUDA kernel
against the reference PyTorch implementation across various dtypes,
shapes, and edge cases. Useful for CI or pre-release checks.
"""

import sys
import torch
from kernel import scale, scale_inplace


def torch_scale(x: torch.Tensor, s: float) -> torch.Tensor:
    return x * s


def _make_tensor(shape, dtype=torch.float32, device="cuda"):
    return torch.randn(shape, dtype=dtype, device=device)


def check(name: str, passed: bool) -> bool:
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}")
    return passed


def validate_correctness() -> bool:
    """Basic correctness across dtypes."""
    print("validate_correctness")
    ok = True
    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        x = _make_tensor((1024,), dtype=dtype)
        s = 3.14
        ref = torch_scale(x, s)
        out = scale(x, s)
        close = torch.allclose(out.float(), ref.float(), atol=1e-2, rtol=1e-2)
        ok &= check(f"dtype={dtype}", close)
    return ok


def validate_shapes() -> bool:
    """Various tensor shapes."""
    print("validate_shapes")
    ok = True
    shapes = [(1,), (256,), (1024,), (4, 256), (8, 128, 64), (1, 1, 1)]
    for shape in shapes:
        x = _make_tensor(shape)
        s = 2.0
        ref = torch_scale(x, s)
        out = scale(x, s)
        close = torch.allclose(out, ref, atol=1e-5)
        ok &= check(f"shape={shape}", close)
    return ok


def validate_inplace() -> bool:
    """Inplace variant produces same result as out-of-place."""
    print("validate_inplace")
    ok = True
    x = _make_tensor((512,))
    s = 0.5
    expected = scale(x, s)
    scale_inplace(x, s)
    close = torch.allclose(x, expected, atol=1e-5)
    ok &= check("inplace matches out-of-place", close)
    return ok


def validate_scale_zero() -> bool:
    """Scaling by zero yields all zeros."""
    print("validate_scale_zero")
    x = _make_tensor((256,))
    out = scale(x, 0.0)
    passed = torch.all(out == 0.0).item()
    return check("scale by 0.0", passed)


def validate_scale_one() -> bool:
    """Scaling by one is identity."""
    print("validate_scale_one")
    x = _make_tensor((256,))
    out = scale(x, 1.0)
    passed = torch.allclose(out, x, atol=1e-6)
    return check("scale by 1.0 (identity)", passed)


def validate_negative_scale() -> bool:
    """Negative scale flips signs."""
    print("validate_negative_scale")
    x = _make_tensor((256,))
    s = -2.5
    ref = torch_scale(x, s)
    out = scale(x, s)
    passed = torch.allclose(out, ref, atol=1e-5)
    return check("negative scale", passed)


def validate_non_contiguous() -> bool:
    """Non-contiguous input tensor."""
    print("validate_non_contiguous")
    x = _make_tensor((512,))[::2]  # stride-2 slice
    s = 1.5
    ref = torch_scale(x.contiguous(), s)
    out = scale(x, s)
    passed = torch.allclose(out, ref, atol=1e-5)
    return check("non-contiguous input", passed)


def main() -> int:
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available, skipping validation.")
        return 1

    suites = [
        validate_correctness,
        validate_shapes,
        validate_inplace,
        validate_scale_zero,
        validate_scale_one,
        validate_negative_scale,
        validate_non_contiguous,
    ]

    results = [fn() for fn in suites]
    total = len(results)
    passed = sum(results)
    print(f"\nResult: {passed}/{total} suites passed.")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
