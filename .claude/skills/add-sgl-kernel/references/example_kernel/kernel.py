"""Python bindings and utilities for the example scale kernel.

This module loads the compiled CUDA extension and exposes a clean
Python API for the scale operation.
"""

import os
import torch
from pathlib import Path

# Attempt to load the pre-built extension; fall back to JIT compilation.
try:
    import example_kernel_cuda as _C
except ImportError:
    from torch.utils.cpp_extension import load

    _ext_dir = Path(__file__).parent
    _C = load(
        name="example_kernel_cuda",
        sources=[str(_ext_dir / "kernel.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )


def scale(
    x: torch.Tensor,
    factor: float,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Element-wise scale: out = x * factor.

    Args:
        x:      Input tensor. Must be on a CUDA device.
        factor: Scalar multiplier.
        out:    Optional pre-allocated output tensor with the same shape and
                dtype as *x*. If *None* a new tensor is allocated.

    Returns:
        Tensor with each element of *x* multiplied by *factor*.

    Raises:
        ValueError: If *x* is not on a CUDA device.
        ValueError: If *out* shape or dtype does not match *x*.
    """
    if x.device.type != "cuda":
        raise ValueError(f"Input tensor must be on a CUDA device, got {x.device}")

    if out is None:
        out = torch.empty_like(x)
    else:
        if out.shape != x.shape:
            raise ValueError(
                f"Output shape {out.shape} does not match input shape {x.shape}"
            )
        if out.dtype != x.dtype:
            raise ValueError(
                f"Output dtype {out.dtype} does not match input dtype {x.dtype}"
            )
        if out.device != x.device:
            raise ValueError(
                f"Output device {out.device} does not match input device {x.device}"
            )

    _C.scale(x, factor, out)
    return out


def scale_inplace(x: torch.Tensor, factor: float) -> torch.Tensor:
    """In-place variant: x *= factor.

    Args:
        x:      Input/output tensor on a CUDA device.
        factor: Scalar multiplier.

    Returns:
        The same tensor *x* after scaling.
    """
    return scale(x, factor, out=x)
