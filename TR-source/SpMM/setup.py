import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# 目标架构由环境变量 TR_CUDA_ARCHS 控制 (分号或逗号分隔):
#   RTX 3090 -> 8.6   RTX 4090 -> 8.9   A100 -> 8.0
_archs = os.environ.get("TR_CUDA_ARCHS", "")
_nvcc = ["-O3"]
if _archs:
    for _a in _archs.replace(",", ";").split(";"):
        _a = _a.strip().replace(".", "")
        if _a:
            _nvcc.append(f"-gencode=arch=compute_{_a},code=sm_{_a}")

setup(
    name='TR_SPMM',
    ext_modules=[
        CUDAExtension(
            name='TR_SPMM',
            sources=['./TRkernel.cu', './TR.cpp'],
            extra_compile_args={'cxx': ['-O3'], 'nvcc': _nvcc},
        ),
    ],
    cmdclass={'build_ext': BuildExtension})
