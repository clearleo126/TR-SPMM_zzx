#!/bin/bash
# A100 上的环境搭建 (Python 3.9 + torch cu128 + 依赖 + libMHCUDA)
# 注意: 本机的 LD_LIBRARY_PATH 被上层工具注入了一个旧 libstdc++, 会遮蔽系统库,
#      凡是编译/运行 g++/cmake/libMHCUDA 前都必须清掉.
set -e
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64

uv python install 3.9
uv venv --python 3.9 /root/venv39
uv pip install --python /root/venv39/bin/python torch \
    --index-url https://download.pytorch.org/whl/cu128
uv pip install --python /root/venv39/bin/python \
    numpy scipy datasketch ogb Cython pybind11 setuptools ninja wheel

# libMHCUDA 2.3.0: TCA_reorder_gpu.py 的 GPU Weighted-MinHash 依赖
rm -rf /root/minhashcuda
git clone https://github.com/src-d/minhashcuda.git /root/minhashcuda
cd /root/minhashcuda
/root/venv39/bin/python setup.py build
cp -f libMHCUDA.so /root/venv39/lib/python3.9/site-packages/
/root/venv39/bin/python -c "import libMHCUDA; print('libMHCUDA ok')"
