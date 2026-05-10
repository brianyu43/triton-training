"""Build the flash-attention PyTorch extension.

Build in-place:
    cd extension
    python setup.py build_ext --inplace

After build, the shared object `mylib_ext*.so` lands next to this setup.py.
Importing the module registers the ops under `torch.ops.mylib`.
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="mylib_ext",
    ext_modules=[
        CUDAExtension(
            name="mylib_ext",
            sources=["csrc/flash_attention_ext.cu"],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "-lineinfo",
                    "-gencode=arch=compute_75,code=sm_75",
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
