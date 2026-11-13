from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

setup(
    name='Libra6',
    ext_modules=[

        CUDAExtension(
            name='Libra6SpMM', 
            sources=[
            './mGCNkernel.cu',
            './mGCN.cpp',
            ]
         ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })


