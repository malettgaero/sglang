// Example SGLang custom CUDA kernel: element-wise scale
// Demonstrates the pattern for adding new kernels to sgl-kernels

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// CUDA kernel
// ---------------------------------------------------------------------------

template <typename scalar_t>
__global__ void scale_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const float scale,
    const int64_t numel) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numel) {
    output[idx] = static_cast<scalar_t>(static_cast<float>(input[idx]) * scale);
  }
}

// ---------------------------------------------------------------------------
// Host-side launcher
// ---------------------------------------------------------------------------

torch::Tensor scale_cuda(const torch::Tensor& input, float scale_factor) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");

  auto output = torch::empty_like(input);
  const int64_t numel = input.numel();

  if (numel == 0) {
    return output;
  }

  const int threads = 256;
  const int blocks = (numel + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      input.scalar_type(),
      "scale_cuda",
      [&]() {
        scale_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            scale_factor,
            numel);
      });

  // Check for kernel launch errors
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return output;
}

// In-place variant — writes result back into `output` buffer provided by caller
void scale_cuda_out(
    const torch::Tensor& input,
    torch::Tensor& output,
    float scale_factor) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(output.is_cuda(), "output must be a CUDA tensor");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
  TORCH_CHECK(input.sizes() == output.sizes(), "input and output must have the same shape");
  TORCH_CHECK(input.scalar_type() == output.scalar_type(), "input and output must have the same dtype");

  const int64_t numel = input.numel();
  if (numel == 0) return;

  const int threads = 256;
  const int blocks = (numel + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      input.scalar_type(),
      "scale_cuda_out",
      [&]() {
        scale_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            scale_factor,
            numel);
      });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---------------------------------------------------------------------------
// Pybind11 bindings
// ---------------------------------------------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "Example SGLang scale kernel";
  m.def("scale", &scale_cuda, "Element-wise scale (returns new tensor)",
        py::arg("input"), py::arg("scale_factor"));
  m.def("scale_out", &scale_cuda_out, "Element-wise scale into pre-allocated output",
        py::arg("input"), py::arg("output"), py::arg("scale_factor"));
}
