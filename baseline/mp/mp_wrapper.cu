// MP-SpMM pybind 封装: 把官方 CLI 基准程序暴露为 PyTorch 算子 (GPU 直通).
//
// 官方实现 (mpspmm/SpMM/spmm_sp_new.cu) 是 main() 程序: 从磁盘读预处理结果,
// CPU 侧 rand 初始化 B, 每次 cudaMalloc + H2D, 跑 50 次后打印时间, 没有
// Python/Tensor 接口. 这里保留其 kernel 与预处理产物格式, 只替换外壳:
//   - 输入为 torch CUDA tensor (rhs 不做 CPU 往返)
//   - 输出为 torch CUDA tensor
//   - metadata 压缩与 H2D 只付一次 (见 mp_wrap.py 的 packed 缓存)
// 指令序列与官方完全一致, 因此计时可与官方 CLI 对照.
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <unordered_map>

#include "config.h"
#include "kernels.cu"   // 单 TU 编译, 避免跨 TU kernel 链接需要的 -rdc

// ============================================================================
// metadata: int32 (0-3) → 2-bit packed uint32 (逐字复刻官方 storeArrayInUint32)
// ============================================================================
static int valueToStorage(int v) {
    switch (v) {
        case 0: return 0b00;
        case 1: return 0b01;
        case 2: return 0b10;
        case 3: return 0b11;
        default: return 0;
    }
}

static void storeArrayInUint32(const int* metadata_old, uint32_t* metadata_new,
                               int block_row, int block_col, int block_start) {
    int start_row = (block_start + block_col) * tile_M * tile_K / 2;
    for (int r = 0; r < 16 / 2; ++r) {
        for (int c = 0; c < 8 / 2; ++c) {
            int v00 = metadata_old[start_row + r * tile_K / 2 + c * 2];
            int v01 = metadata_old[start_row + r * tile_K / 2 + c * 2 + 1];
            int v10 = metadata_old[start_row + (r + 8) * tile_K / 2 + c * 2];
            int v11 = metadata_old[start_row + (r + 8) * tile_K / 2 + c * 2 + 1];
            int s00 = valueToStorage(v00), s01 = valueToStorage(v01);
            int s10 = valueToStorage(v10), s11 = valueToStorage(v11);
            int idx = r + (block_start + block_col) * 8;
            metadata_new[idx] |= (s00 << (c * 4));
            metadata_new[idx] |= (s01 << (c * 4 + 2));
            metadata_new[idx] |= (s10 << (c * 4 + 16));
            metadata_new[idx] |= (s11 << (c * 4 + 16 + 2));
        }
    }
}

static std::vector<uint32_t> pack_metadata(const int* meta, int64_t n,
                                           const int* offset, int64_t n_off) {
    std::vector<uint32_t> out((size_t)(n / 16), 0u);
    for (int64_t row = 0; row + 1 < n_off; ++row) {
        int block_start = offset[row];
        int cnt = offset[row + 1] - block_start;
        for (int bc = 0; bc < cnt; ++bc)
            storeArrayInUint32(meta, out.data(), (int)row, bc, block_start);
    }
    return out;
}

// ============================================================================
// 预处理产物常驻显存 (与 TR.cpp 的 g_tr_prep_map 同思路)
//
// 官方 main() 每次调用都 cudaMalloc + 全量 H2D, 在端到端循环里这部分墙钟
// 会完全压过 kernel 本身 (实测 ddi 上 5.8 ms vs kernel 0.2 ms). 这里把
// metadata/A/col 的 device 缓冲按 CPU tensor 的 data_ptr 缓存, 只有 rhs
// 每次刷新, 使三家 baseline 的调用边界一致.
// ============================================================================
struct MPPrep {
    int* d_off = nullptr;
    uint32_t* d_meta = nullptr;
    half* d_a = nullptr;
    int* d_col = nullptr;
    half* d_b = nullptr;
    int64_t K = 0, N = 0;
};

static std::unordered_map<const void*, MPPrep> g_mp_prep;

static void mp_prep_free(MPPrep& p) {
    if (p.d_off) cudaFree(p.d_off);
    if (p.d_meta) cudaFree(p.d_meta);
    if (p.d_a) cudaFree(p.d_a);
    if (p.d_col) cudaFree(p.d_col);
    if (p.d_b) cudaFree(p.d_b);
    p = MPPrep();
}

static void mp_prepare(torch::Tensor tcblocks_offset, torch::Tensor opd_a,
                       torch::Tensor packed_meta, torch::Tensor old_col_all,
                       int64_t k_cols, int64_t dim_n) {
    const void* key = tcblocks_offset.data_ptr();
    auto it = g_mp_prep.find(key);
    if (it != g_mp_prep.end() && it->second.K == k_cols && it->second.N == dim_n)
        return;   // 命中: 预处理产物已在显存
    if (it != g_mp_prep.end()) { mp_prep_free(it->second); g_mp_prep.erase(it); }

    MPPrep p;
    p.K = k_cols; p.N = dim_n;
    cudaMalloc(&p.d_off, tcblocks_offset.numel() * sizeof(int));
    cudaMalloc(&p.d_meta, std::max<int64_t>(packed_meta.numel(), 1) * sizeof(uint32_t));
    cudaMalloc(&p.d_a, std::max<int64_t>(opd_a.numel(), 1) * sizeof(half));
    cudaMalloc(&p.d_col, std::max<int64_t>(old_col_all.numel(), 1) * sizeof(int));
    cudaMalloc(&p.d_b, std::max<int64_t>(k_cols * dim_n, 1) * sizeof(half));

    cudaMemcpy(p.d_off, tcblocks_offset.data_ptr<int>(), tcblocks_offset.numel() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(p.d_meta, reinterpret_cast<const uint32_t*>(packed_meta.data_ptr<int>()),
               packed_meta.numel() * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(p.d_a, opd_a.data_ptr<at::Half>(), opd_a.numel() * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(p.d_col, old_col_all.data_ptr<int>(), old_col_all.numel() * sizeof(int), cudaMemcpyHostToDevice);
    g_mp_prep[key] = p;
}

static void mp_clear_prep() {
    for (auto& kv : g_mp_prep) mp_prep_free(kv.second);
    g_mp_prep.clear();
}

// ============================================================================
// forward
// ============================================================================
static torch::Tensor mp_forward(
    torch::Tensor tcblocks_offset,   // [num_windows+1] int32 (CPU)
    torch::Tensor opd_a,             // [N] float16, 已转 half
    torch::Tensor packed_meta,       // [N/16] uint32 (CPU, 已压缩)
    torch::Tensor old_col_all,       // [N] int32, -1 已替换为 0
    torch::Tensor rhs,               // [K, N] float16, CPU 或 CUDA
    int64_t m_rows, int64_t k_cols, int64_t dim_n) {

    TORCH_CHECK(dim_n == 128, "mp_forward currently supports dimN=128 only");
    const int M = (int)((m_rows + 15) & ~15LL);   // 与官方 main 一致: pad 到 16
    const int K = (int)k_cols;
    const int N = (int)dim_n;

    mp_prepare(tcblocks_offset, opd_a, packed_meta, old_col_all, k_cols, dim_n);
    MPPrep& p = g_mp_prep.at(tcblocks_offset.data_ptr());

    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(rhs.device());
    // kernel 逐 window 整块写入 C, 但无 block 的 window 会提前 return,
    // 那些行必须保持 0, 故仍需 zeros (与 TR kernel 内部的 memset 对等)
    auto out = torch::zeros({M, N}, opts);
    float* d_c = out.data_ptr<float>();

    // rhs 为 CUDA 时零拷贝直用调用方缓冲 (此前每次多付一份 K*N half 的 D2D)
    half* b_use = p.d_b;
    if (rhs.is_cuda()) {
        b_use = const_cast<half*>(reinterpret_cast<const half*>(rhs.data_ptr<at::Half>()));
    } else {
        cudaMemcpy(p.d_b, rhs.data_ptr<at::Half>(),
                   (int64_t)K * N * sizeof(half), cudaMemcpyHostToDevice);
    }

    dim3 grid((M + tile_M - 1) / tile_M, 1, 1);
    int warpCount = (N + tile_N - 1) / tile_N;
    dim3 block(WARP_SIZE, warpCount / 4, 1);
    cudaFuncSetAttribute(sparse_mma_kernel_base_Buint64_Cfloat4,
                         cudaFuncAttributePreferredSharedMemoryCarveout, 0);
    sparse_mma_kernel_base_Buint64_Cfloat4<<<grid, block>>>(
        M, N, K, p.d_a, p.d_meta, p.d_off, p.d_col, b_use, d_c);
    cudaDeviceSynchronize();
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "mp kernel launch failed");

    return out.narrow(0, 0, m_rows);
}

static std::vector<uint32_t> mp_pack_metadata(torch::Tensor metadata, torch::Tensor offset) {
    return pack_metadata(metadata.data_ptr<int>(), metadata.numel(),
                         offset.data_ptr<int>(), offset.numel());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &mp_forward, "MP-SpMM forward (dimN=128)");
    m.def("pack_metadata", &mp_pack_metadata, "pack int32 metadata to 2-bit uint32");
    m.def("prepare", &mp_prepare, "cache preprocessed metadata on device");
    m.def("clear_prep", &mp_clear_prep, "release cached device buffers");
}
