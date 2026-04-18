# Example SGL Kernel

This directory contains a minimal end-to-end example of adding a custom CUDA kernel to sglang via the `sgl-kernel` extension mechanism.

The example implements a simple element-wise `scale` operation (`output = input * scalar`) to demonstrate the full workflow without distracting domain complexity.

## Files

| File | Purpose |
|------|---------|
| `kernel.cu` | CUDA kernel implementation |
| `kernel.py` | Python bindings and public API |
| `setup.py` | Build script (compiles the `.cu` file) |
| `__init__.py` | Package exports |
| `kernel_test.py` | Correctness tests (pytest) |
| `kernel_bench.py` | Microbenchmark (Python timing) |
| `kernel_bench.sh` | Shell wrapper to run benchmarks across shapes |
| `kernel_profile.py` | Nsight Systems / torch.profiler integration |

## Quick Start

```bash
# 1. Build the extension
pip install -e . --no-build-isolation

# 2. Run correctness tests
pytest kernel_test.py -v

# 3. Run benchmarks
bash kernel_bench.sh

# 4. Profile with torch.profiler
python kernel_profile.py --mode torch

# 5. Profile with Nsight Systems
python kernel_profile.py --mode nsight
```

## How It Works

### CUDA Kernel (`kernel.cu`)

The kernel is a straightforward grid-stride loop:

```cpp
template <typename scalar_t>
__global__ void scale_kernel(const scalar_t* input, scalar_t* output,
                              float scalar, int64_t n) {
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += blockDim.x * gridDim.x) {
        output[i] = static_cast<scalar_t>(input[i] * scalar);
    }
}
```

Key design decisions:
- **Grid-stride loop** — handles arbitrarily large tensors with a fixed launch config.
- **Template on `scalar_t`** — supports `float32`, `float16`, and `bfloat16` without code duplication.
- **Separate `output` buffer** — avoids in-place aliasing issues; a thin Python wrapper provides `scale_inplace`.

### Python Bindings (`kernel.py`)

`kernel.py` loads the compiled extension with `torch.utils.cpp_extension.load` (JIT) or via the installed package, then exposes:

```python
def scale(input: torch.Tensor, scalar: float,
          out: Optional[torch.Tensor] = None) -> torch.Tensor:
    ...

def scale_inplace(input: torch.Tensor, scalar: float) -> torch.Tensor:
    ...
```

Both functions validate inputs (device, contiguity, dtype) before dispatching to the C++ extension.

## Adapting to a New Kernel

1. Copy this directory and rename files to match your operation.
2. Replace the `scale_kernel` implementation in `kernel.cu` with your logic.
3. Update the C++ dispatcher in `kernel.cu` and the Python API in `kernel.py`.
4. Add correctness tests in `kernel_test.py` covering edge cases and supported dtypes.
5. Update `setup.py` if you need additional CUDA libraries (e.g., `cublas`, `cutlass`).

## Supported Dtypes

| dtype | Supported |
|-------|-----------|
| `float32` | ✅ |
| `float16` | ✅ |
| `bfloat16` | ✅ |
| `int32` / `int64` | ❌ (cast to float first) |

## Performance Notes

For memory-bound kernels like `scale`, performance is dominated by DRAM bandwidth. On an A100 80 GB:

- Peak bandwidth: ~2 TB/s
- Expected throughput at large sizes: >90% of peak
- Overhead vs. `torch.mul`: <5% at sizes ≥ 1 M elements

Run `bash kernel_bench.sh` to reproduce these numbers on your hardware.
