from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

setup(
    name='Libra5',
    ext_modules=[
        CppExtension(
            name='Libra5Block', 
            sources=[
            './Block/block.cpp'
            ],
            extra_compile_args=['-O3', '-fopenmp', '-mcx16'],
            # libraries=["numa", "tcmalloc_minimal"]
            # [port] block.cpp 未使用任何 numa_* 符号, 去掉 -lnuma 依赖以便在无 libnuma-dev 的机器上编译
         ),
        CUDAExtension(
            name='Libra5BenchmarkGCN', 
            sources=[
            './benchmarkGCN/mGCNkernel.cu',
            './benchmarkGCN/mGCN.cpp',
            ]
         ),
        # CUDAExtension(
        #     name='Libra5BenchmarkGAT', 
        #     sources=[
        #     './benchmarkGAT/mGATkernel.cu',
        #     './benchmarkGAT/mGAT.cpp',
        #     ]
        #  ),
        # CUDAExtension(
        #     name='Libra5SpMM', 
        #     sources=[
        #     './SpMM/mGCNkernel.cu',
        #     './SpMM/mGCN.cpp',
        #     ]
        #  ),
        # CUDAExtension(
        #     name='LibraGCN_new', 
        #     sources=[
        #     './GCN/GCNkernel.cu',
        #     './GCN/GCN.cpp',
        #     ]
        #  ),
        # CUDAExtension(
        #     name='LibraAGNN_new', 
        #     sources=[
        #     './GAT/mGATkernel.cu',
        #     './GAT/mGAT.cpp',
        #     ]
        #  ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })


