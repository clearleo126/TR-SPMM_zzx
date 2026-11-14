#!/bin/bash
# 取 DTC 官方数据集 (SuiteSparse 官网 2026-11 离线时的替代源).
# origin_matrices.tar.gz: 8 个矩阵的 COO npz (src_li/dst_li/num_nodes)
# reordered_matrices.tar.gz: 官方 TCA 重排后的 COO (本次未用于计时, 因为要求现场生成)
set -e
mkdir -p /dev/shm/dtc && cd /dev/shm/dtc
GIT_LFS_SKIP_SMUDGE=0 git clone https://github.com/fan1997/dtc_datasets.git
cd dtc_datasets
mkdir -p ../origin
tar xzf origin_matrices.tar.gz -C ../origin
ls -l ../origin
