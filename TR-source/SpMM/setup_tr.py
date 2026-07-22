from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name='TR_SPMM',
    ext_modules=[
        CUDAExtension(
            name='TR_SPMM',
            sources=[
                './TRkernel.cu',
                './TR.cpp',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '-gencode=arch=compute_80,code=sm_80',   # RTX 3090 (Ampere SM 8.0)
                    '-gencode=arch=compute_86,code=sm_86',   # RTX 3090 (Ampere SM 8.6)
                ],
            },
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
