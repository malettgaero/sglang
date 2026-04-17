"""Tests for the example scale kernel.

Runs correctness, shape mismatch, and CPU-input checks against
the reference torch implementation so the custom CUDA kernel can
be validated end-to-end before being wired into sglang.
"""

import pytest
import torch

# The extension is built via `python setup.py install` or `pip install -e .`
try:
    import example_kernel  # type: ignore
    HAS_KERNEL = True
except ImportError:
    HAS_KERNEL = False


def torch_scale(x: torch.Tensor, factor: float) -> torch.Tensor:
    """Reference implementation using plain PyTorch."""
    return x * factor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tensor(*shape, dtype=torch.float32, device="cuda"):
    return torch.randn(*shape, dtype=dtype, device=device)


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_KERNEL, reason="example_kernel not built")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("shape", [(128,), (32, 64), (4, 16, 256)])
@pytest.mark.parametrize("factor", [0.5, 1.0, 2.0, -3.14])
def test_scale_correctness(shape, factor):
    x = _make_tensor(*shape)
    expected = torch_scale(x, factor)
    actual = example_kernel.scale(x, factor)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not HAS_KERNEL, reason="example_kernel not built")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_scale_out_param():
    """Kernel should write into a pre-allocated output buffer when provided."""
    x = _make_tensor(256)
    out = torch.empty_like(x)
    result = example_kernel.scale(x, 3.0, out=out)
    expected = torch_scale(x, 3.0)
    # result and out should be the same tensor
    assert result.data_ptr() == out.data_ptr()
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Half / BFloat16 precision
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_KERNEL, reason="example_kernel not built")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_scale_half_precision(dtype):
    x = _make_tensor(512, dtype=dtype)
    expected = torch_scale(x, 0.25)
    actual = example_kernel.scale(x, 0.25)
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_KERNEL, reason="example_kernel not built")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_scale_shape_mismatch():
    """Passing a mismatched output buffer should raise an exception."""
    x = _make_tensor(64)
    out = torch.empty(128, device="cuda")
    with pytest.raises((RuntimeError, ValueError)):
        example_kernel.scale(x, 1.0, out=out)


@pytest.mark.skipif(not HAS_KERNEL, reason="example_kernel not built")
def test_scale_cpu_input():
    """CPU tensors should raise an error — kernel requires CUDA inputs."""
    x = torch.randn(64)  # CPU tensor
    with pytest.raises((RuntimeError, ValueError)):
        example_kernel.scale(x, 1.0)


# ---------------------------------------------------------------------------
# Performance smoke-test (not a benchmark, just ensures no crash at scale)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_KERNEL, reason="example_kernel not built")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_scale_large_tensor():
    x = _make_tensor(1024, 1024)  # 4 MB fp32
    out = example_kernel.scale(x, 0.1)
    assert out.shape == x.shape
    assert out.dtype == x.dtype
