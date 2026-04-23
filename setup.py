from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="cuda_ops",
    ext_modules=[
        CUDAExtension(
            name="cuda_ops._C",
            sources=[
                "csrc/bindings.cpp",
                "kernels/categorical_cross_entropy.cu",
                "kernels/dot_product.cu",
                "kernels/prefix_sum.cu",
                "kernels/matmul.cu",
            ],
            extra_compile_args={
                "cxx":  ["-O3"],
                "nvcc": ["-O3", "--use_fast_math"]
            }
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    packages=["cuda_ops"],
)