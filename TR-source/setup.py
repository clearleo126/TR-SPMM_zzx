from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension


setup(
    name='Matching24',
    ext_modules=[
        CppExtension(
            name='matching_utils',
            sources=['./Block/block_matching_zhy.cpp'],
            extra_compile_args=['-O3', '-march=native', '-fopenmp'],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
