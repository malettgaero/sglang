from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch


def get_cuda_arch_flags():
    """Get CUDA architecture flags based on available GPU."""
    if not torch.cuda.is_available():
        return []
    capability = torch.cuda.get_device_capability()
    major, minor = capability
    arch = f"{major}{minor}"
    return [f"-gencode=arch=compute_{arch},code=sm_{arch}"]


setup(
    name="example_sgl_kernel",
    version="0.1.0",
    description="Example custom CUDA kernel for SGLang",
    ext_modules=[
        CUDAExtension(
            name="example_sgl_kernel",
            sources=[
                "csrc/example_kernel.cu",
                "csrc/example_kernel_binding.cpp",
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-std=c++17",
                    *get_cuda_arch_flags(),
                ],
            },
            include_dirs=["csrc/include"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.8",
)