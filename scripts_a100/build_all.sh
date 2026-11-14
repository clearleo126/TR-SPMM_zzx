#!/bin/bash
# 编译 4 家 backend (全部 --inplace). A100 / CUDA 12.8 / Python 3.9 (venv39).
# 前置: scripts_a100/env_setup.sh 已装好 torch cu128 + libMHCUDA.
set -x
REPO=$(cd "$(dirname "$0")/.." && pwd)
source /root/venv39/bin/activate
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64
export MAX_JOBS=48
export TORCH_CUDA_ARCH_LIST="8.0"

cd "$REPO/TR-source" && python setup_24matching.py build_ext --inplace || exit 1
cd "$REPO/TR-source/SpMM" && python setup.py build_ext --inplace || exit 1
cd "$REPO/baseline/mp/preprocessing" && bash build.sh || exit 1
cd "$REPO/baseline/mp" && python setup_mp.py build_ext --inplace || exit 1
cd "$REPO/baseline/libra/Libra-source" && python setup.py build_ext --inplace || exit 1
cd "$REPO/baseline/libra/Libra-source/SpMM" && python setup.py build_ext --inplace || exit 1
cd "$REPO/baseline/dtc" && python setup.py build_ext --inplace || exit 1
echo "BUILD_ALL_DONE"
