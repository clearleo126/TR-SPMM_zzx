from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

setup(
    name='Libra6',
    ext_modules=[

        CUDAExtension(
            name='Libra6SDDMM', 
            sources=[
            './mGATkernel.cu',
            './mGAT.cpp',
            ]
         ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })


