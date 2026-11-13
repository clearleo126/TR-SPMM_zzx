# from setuptools import setup
# from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# nvcc_flags = [
#     "-O3",
#     # 例：H100
#     "-gencode=arch=compute_120,code=sm_120",
# ]

# setup(
#     name="Libra6_AMD",
#     ext_modules=[
#         CUDAExtension(
#             name="Libra6SpMMamd",
#             sources=["./mGCNkernel.cu", "./mGCN.cpp"],
#             extra_compile_args={"nvcc": nvcc_flags, "cxx": ["-O3"]},
#         )
#     ],
#     cmdclass={"build_ext": BuildExtension},
# )
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

AMD_ARCH = os.environ.get("AMD_ARCH", "gfx1100")

hip_flags = [
    "-O3",
    f"--offload-arch=gfx1100",
]

setup(
    name="Libra6AMD",
    ext_modules=[
        CUDAExtension(
            name="Libra6SpMMamd",
            # 强烈建议：把 .hip 改名成 .cu（内容仍然是 HIP 也行）
            sources=["./mGCNkernel.cu", "./mGCN.cpp"],
            extra_compile_args={
                "cxx": ["-O3"],
                # ROCm 下仍用 key="nvcc" 传给 HIP 编译链
                "nvcc": hip_flags,
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)