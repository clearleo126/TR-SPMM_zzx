/**
 * TRkernel.cu — TC (Dense MMA) + SPTC (mma.sp) 双流并行 CUDA 内核
 *
 * 架构:
 *   - TC  kernel: 稠密 Tensor Core GEMM (mma.sync.aligned.m16n8k8)
 *                 处理 TC 16xK 稠密块, 从 sparse local_id 格式重建
 *   - SPTC kernel: 结构化稀疏 Tensor Core GEMM (mma.sp.sync.aligned.m16n8k16)
 *                  处理 SPTC 16x16 2:4 稀疏块, 使用预压缩元数据
 *   - 双 CUDA Stream 并行执行 TC 和 SPTC 内核
 *
 * 参考:
 *   - mGCNkernel_online.cu (TR-SPMM, 流并行模式)
 *   - kernels.cu (MP-SpMM_SC25, mma.sp 内核模式)
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>


// ============================================================================
// 辅助函数
// ============================================================================

static __device__ __forceinline__ uint32_t pack_half2_u32(half lo, half hi) {
    return (uint32_t)__half_as_ushort(lo) | ((uint32_t)__half_as_ushort(hi) << 16);
}

static bool env_enabled(const char* name, bool default_value) {
    const char* value = std::getenv(name);
    if (value == nullptr) return default_value;
    return std::strcmp(value, "1") == 0 || std::strcmp(value, "true") == 0;
}


// ============================================================================
// TC Kernel: 稠密 Tensor Core GEMM (m16n8k8)
//
// 将 TC 块 (16×dense_threshold) 从 sparse local_id 格式重建为稠密矩阵,
// 然后使用常规 mma.sync 执行稠密 GEMM.
//
// Grid:  (num_windows, feature_tiles) 2D
// Block: (128, 1, 1)  4 warps × 32 lanes
//
// 每 block 处理 1 个窗口 × 32 特征列.
// 每 warp 处理 8 特征列 (4 次 mma, 每次处理 2 列).
// ============================================================================

__global__ void tr_tc_kernel(
    const int* __restrict__ tc_window_offset,   // [num_windows+1]
    const int* __restrict__ tc_offset,           // [total_nnz] NNZ prefix sum per block
    const uint8_t* __restrict__ tc_local_id,     // [total_nnz] r*16+c
    const int* __restrict__ tc_col_old,          // [num_blocks * dense_threshold]
    const half* __restrict__ tc_value,           // [total_nnz]
    const half* __restrict__ rhs_matrix,         // [K * dimN] B matrix
    float* __restrict__ output_matrix,           // [M * dimN]
    int num_windows,
    int dense_threshold,                          // TC block K-dimension (typically 8)
    int dimN,
    int mOri,
    int kOri,
    int feature_tiles)
{
    int window_id = blockIdx.x;
    int feature_tile_id = blockIdx.y;
    if (window_id >= num_windows || feature_tile_id >= feature_tiles) return;

    int block_start = __ldg(tc_window_offset + window_id);
    int block_end   = __ldg(tc_window_offset + window_id + 1);
    if (block_start >= block_end) return;

    int feature_base = feature_tile_id * 32;
    if (feature_base >= dimN) return;

    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int groupID = lane >> 2;
    int tid_in_group = lane & 3;
    int warp_feat_base = feature_base + warp_id * 8;

    // 共享内存: 16×16 dense A tile (每个 TC block 重建)
    __shared__ half smem_A[16][16];

    // 预加载窗口行偏移
    int window_row = window_id * 16;

    // 遍历该窗口的每个 TC block
    for (int blk = block_start; blk < block_end; ++blk) {
        // ---- 1. 清空 smem_A ----
        for (int i = lane; i < 256; i += 32) {
            int r = i / 16;
            int c = i % 16;
            smem_A[r][c] = __float2half(0.0f);
        }

        // ---- 2. 从 sparse 格式散射 A 值到 smem_A ----
        // tc_offset 是 CSR 格式前缀和, 以 0 开头: [0, cumsum0, cumsum1, ...]
        // 因此 tc_offset[blk] 是 block blk 之前的累计 NNZ,
        //    tc_offset[blk+1] 是 block blk 之后的累计 NNZ
        int nnz_start = __ldg(tc_offset + blk);
        int nnz_end   = __ldg(tc_offset + blk + 1);
        int nnz_count = nnz_end - nnz_start;

        for (int n = lane; n < nnz_count; n += 32) {
            uint8_t local_id = __ldg(tc_local_id + nnz_start + n);
            int r = local_id / 16;
            int c = local_id % 16;
            half val = __ldg(tc_value + nnz_start + n);
            smem_A[r][c] = val;
        }
        __syncthreads();

        // ---- 3. 遍历 K 维度 (每次处理 8 个 K 列), 累加到一个累加器 ----
        const int* blk_col = tc_col_old + blk * dense_threshold;
        int k_chunks = (dense_threshold + 7) / 8;

        // 单一累加器 (所有 k-chunks 累加到此)
        float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;

        for (int kc = 0; kc < k_chunks; ++kc) {
            int k_start = kc * 8;
            if (warp_feat_base >= dimN) break;

            // ---- 3a. 加载 A (从 smem_A, 格式适配 m16n8k8) ----
            int a_col0 = k_start + tid_in_group * 2;
            int a_col1 = a_col0 + 1;
            uint32_t a0, a1;
            if (a_col0 < dense_threshold && a_col1 < dense_threshold) {
                a0 = pack_half2_u32(smem_A[groupID][a_col0], smem_A[groupID][a_col1]);
                a1 = pack_half2_u32(smem_A[groupID + 8][a_col0], smem_A[groupID + 8][a_col1]);
            } else if (a_col0 < dense_threshold) {
                a0 = pack_half2_u32(smem_A[groupID][a_col0], __float2half(0.0f));
                a1 = pack_half2_u32(smem_A[groupID + 8][a_col0], __float2half(0.0f));
            } else {
                a0 = 0;
                a1 = 0;
            }

            // ---- 3b. 加载 B (8×8 tile, 从 rhs_matrix) ----
            int k_idx0 = k_start + tid_in_group * 2;
            int k_idx1 = k_idx0 + 1;

            half b0_val = __float2half(0.0f);
            half b1_val = __float2half(0.0f);

            if (k_idx0 < dense_threshold && k_idx1 < dense_threshold) {
                int global_k0 = __ldg(blk_col + k_idx0);
                int global_k1 = __ldg(blk_col + k_idx1);
                int n_col = warp_feat_base + groupID;

                if (global_k0 >= 0 && global_k0 < kOri && n_col < dimN)
                    b0_val = __ldg(rhs_matrix + global_k0 * dimN + n_col);
                if (global_k1 >= 0 && global_k1 < kOri && n_col < dimN)
                    b1_val = __ldg(rhs_matrix + global_k1 * dimN + n_col);
            } else if (k_idx0 < dense_threshold) {
                int global_k0 = __ldg(blk_col + k_idx0);
                int n_col = warp_feat_base + groupID;
                if (global_k0 >= 0 && global_k0 < kOri && n_col < dimN)
                    b0_val = __ldg(rhs_matrix + global_k0 * dimN + n_col);
            }
            uint32_t b = pack_half2_u32(b0_val, b1_val);

            // ---- 3c. MMA (累加到 c0-c3) ----
            asm volatile(
                "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};\n"
                : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                : "r"(a0), "r"(a1), "r"(b));
        }

        // ---- 4. 写回输出 (所有 k-chunks 累加后写一次) ----
        int row0 = window_row + groupID;
        int row1 = row0 + 8;
        int out_col0 = warp_feat_base + tid_in_group * 2;
        int out_col1 = out_col0 + 1;

        if (row0 < mOri) {
            if (out_col0 < dimN)
                atomicAdd(output_matrix + row0 * dimN + out_col0, c0);
            if (out_col1 < dimN)
                atomicAdd(output_matrix + row0 * dimN + out_col1, c1);
        }
        if (row1 < mOri) {
            if (out_col0 < dimN)
                atomicAdd(output_matrix + row1 * dimN + out_col0, c2);
            if (out_col1 < dimN)
                atomicAdd(output_matrix + row1 * dimN + out_col1, c3);
        }
        __syncthreads();  // 为下一个 TC block 做准备
    }
}


// ============================================================================
// SPTC Kernel: 结构化稀疏 Tensor Core GEMM (mma.sp m16n8k16)
//
// 处理 SPTC 16×16 2:4 结构化稀疏块.
// A 值和元数据已在 CPU 端预打包, 直接使用 mma.sp.
//
// Grid:  (num_windows, feature_tiles) 2D
// Block: (128, 1, 1)  4 warps × 32 lanes
//
// 每 block 处理 1 个窗口 × 32 特征列.
// 每 warp 处理 8 特征列 (1 次 mma.sp per tile per warp).
//
// 参考: MP-SpMM_SC25 kernels.cu sparse_mma_kernel_base_Bhalf2_Cfloat2
// ============================================================================

__global__ void tr_sptc_kernel(
    const int* __restrict__ sptc_window_offset,  // [num_windows+1] CSR pointer
    const half* __restrict__ sptc_value,          // [num_blocks * 128] fp16 A values
    const uint32_t* __restrict__ sptc_packed_meta,// [num_blocks * 8] packed metadata
    const int* __restrict__ sptc_col_old,         // [num_blocks * 16] column indices
    const half* __restrict__ rhs_matrix,          // [K * dimN] B matrix
    float* __restrict__ output_matrix,            // [M * dimN]
    int num_windows,
    int dimN,
    int mOri,
    int kOri,
    int feature_tiles)
{
#if __CUDA_ARCH__ >= 800
    int window_id = blockIdx.x;
    int feature_tile_id = blockIdx.y;
    if (window_id >= num_windows || feature_tile_id >= feature_tiles) return;

    int block_start = __ldg(sptc_window_offset + window_id);
    int block_end   = __ldg(sptc_window_offset + window_id + 1);
    if (block_start >= block_end) return;

    int feature_base = feature_tile_id * 32;
    if (feature_base >= dimN) return;

    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int groupID = lane >> 2;           // 0..7
    int tid_in_group = lane & 3;       // 0..3
    int warp_feat_base = feature_base + warp_id * 8;

    // A 值索引 (与 block_matching_zhy.cpp 的 sptc_value 布局一致)
    // 布局: [block][row=0..15][nnz=0..7], 即每 block 128 个 half
    int sparse_A_idx_0 = groupID * 8 + tid_in_group * 2;
    int sparse_A_idx_1 = sparse_A_idx_0 + 64;  // rows 8-15 offset

    // 输出行
    int window_row = window_id * 16;
    int row0 = window_row + groupID;
    int row1 = row0 + 8;

    // 遍历该窗口的每个 SPTC block
    for (int blk = block_start; blk < block_end; ++blk) {
        // ---- 加载 A 值 ----
        int a_offset = blk * 128;
        uint32_t a0 = __ldg(reinterpret_cast<const uint32_t*>(sptc_value + a_offset + sparse_A_idx_0));
        uint32_t a1 = __ldg(reinterpret_cast<const uint32_t*>(sptc_value + a_offset + sparse_A_idx_1));

        // ---- 加载 metadata ----
        uint32_t meta = __ldg(sptc_packed_meta + blk * 8 + groupID);

        // ---- 加载 B 值 ----
        // 每个 SPTC block 有 16 列, 线程按组加载
        const int* blk_col = sptc_col_old + blk * 16;
        uint32_t b0, b1;

        {
            // B 矩阵映射: 16 列 × 8 特征 → ldmatrix 格式
            // 线程加载 4 个 B 值 (每 K-column 1 个, 共 4 个 K-columns)
            // mma.sp.m16n8k16 的 B 是 16×8 (col-major)
            // 线程 (groupID 0..7, tid_in_group 0..3) 加载:
            //   B[tid_in_group*2][groupID] 和 B[tid_in_group*2+1][groupID]
            //   B[tid_in_group*2+8][groupID] 和 B[tid_in_group*2+9][groupID]

            int k0 = tid_in_group * 2;
            int k1 = k0 + 1;
            int k2 = k0 + 8;
            int k3 = k0 + 9;
            int n_col0 = warp_feat_base + groupID;

            half bv[4] = {__float2half(0.0f), __float2half(0.0f),
                          __float2half(0.0f), __float2half(0.0f)};

            // 加载列索引并取值
            int gk0 = (k0 < 16) ? __ldg(blk_col + k0) : -1;
            int gk1 = (k1 < 16) ? __ldg(blk_col + k1) : -1;
            int gk2 = (k2 < 16) ? __ldg(blk_col + k2) : -1;
            int gk3 = (k3 < 16) ? __ldg(blk_col + k3) : -1;

            if (gk0 >= 0 && gk0 < kOri && n_col0 < dimN)
                bv[0] = __ldg(rhs_matrix + gk0 * dimN + n_col0);
            if (gk1 >= 0 && gk1 < kOri && n_col0 < dimN)
                bv[1] = __ldg(rhs_matrix + gk1 * dimN + n_col0);
            if (gk2 >= 0 && gk2 < kOri && n_col0 < dimN)
                bv[2] = __ldg(rhs_matrix + gk2 * dimN + n_col0);
            if (gk3 >= 0 && gk3 < kOri && n_col0 < dimN)
                bv[3] = __ldg(rhs_matrix + gk3 * dimN + n_col0);

            b0 = pack_half2_u32(bv[0], bv[1]);
            b1 = pack_half2_u32(bv[2], bv[3]);
        }

        // ---- MMA (mma.sp) ----
        float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;
        asm volatile(
            "mma.sp.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0, %1, %2, %3}, {%4, %5}, {%6, %7}, {%0, %1, %2, %3}, %8, 0x1;\n"
            : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
            : "r"(a0), "r"(a1), "r"(b0), "r"(b1), "r"(meta));

        // ---- 写回输出 ----
        int out_col0 = warp_feat_base + tid_in_group * 2;
        int out_col1 = out_col0 + 1;

        if (row0 < mOri) {
            if (out_col0 < dimN)
                atomicAdd(output_matrix + row0 * dimN + out_col0, c0);
            if (out_col1 < dimN)
                atomicAdd(output_matrix + row0 * dimN + out_col1, c1);
        }
        if (row1 < mOri) {
            if (out_col0 < dimN)
                atomicAdd(output_matrix + row1 * dimN + out_col0, c2);
            if (out_col1 < dimN)
                atomicAdd(output_matrix + row1 * dimN + out_col1, c3);
        }
    }
#endif
}


// ============================================================================
// 辅助 kernel: 矩阵加法 (TC 和 SPTC 结果合并)
// ============================================================================

__global__ void add_matrix_kernel(
    float* __restrict__ dst,
    const float* __restrict__ src,
    long long elements)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;
    for (; idx < elements; idx += stride) {
        dst[idx] += src[idx];
    }
}


// ============================================================================
// 辅助函数: TC kernel launch
// ============================================================================

static void launch_tc_kernel(
    const int* tc_window_offset,
    const int* tc_offset,
    const uint8_t* tc_local_id,
    const int* tc_col_old,
    const half* tc_value,
    const half* rhs_matrix,
    float* output_matrix,
    int num_windows,
    int num_tc_blocks,
    int dense_threshold,
    int dimN,
    int mOri,
    int kOri,
    cudaStream_t stream)
{
    if (num_tc_blocks <= 0 || dimN <= 0 || mOri <= 0) return;

    int feature_tiles = (dimN + 31) / 32;
    dim3 block_dim(128, 1, 1);
    dim3 grid_dim(num_windows, feature_tiles, 1);

    tr_tc_kernel<<<grid_dim, block_dim, 0, stream>>>(
        tc_window_offset,
        tc_offset,
        tc_local_id,
        tc_col_old,
        tc_value,
        rhs_matrix,
        output_matrix,
        num_windows,
        dense_threshold,
        dimN,
        mOri,
        kOri,
        feature_tiles);
}


// ============================================================================
// 辅助函数: SPTC kernel launch
// ============================================================================

static void launch_sptc_kernel(
    const int* sptc_window_offset,
    const half* sptc_value,
    const uint32_t* sptc_packed_meta,
    const int* sptc_col_old,
    const half* rhs_matrix,
    float* output_matrix,
    int num_windows,
    int num_sptc_blocks,
    int dimN,
    int mOri,
    int kOri,
    cudaStream_t stream)
{
    if (num_sptc_blocks <= 0 || dimN <= 0 || mOri <= 0) return;

    int feature_tiles = (dimN + 31) / 32;
    dim3 block_dim(128, 1, 1);
    dim3 grid_dim(num_windows, feature_tiles, 1);

    tr_sptc_kernel<<<grid_dim, block_dim, 0, stream>>>(
        sptc_window_offset,
        sptc_value,
        sptc_packed_meta,
        sptc_col_old,
        rhs_matrix,
        output_matrix,
        num_windows,
        dimN,
        mOri,
        kOri,
        feature_tiles);
}


// ============================================================================
// 主入口: TC + SPTC 双流并行
//
// 参数:
//   (所有 GPU 指针已分配并填充数据)
//   num_windows, num_sptc_blocks, num_tc_blocks, dense_threshold, dimN, mOri, kOri
//   epoches: benchmark 迭代次数
//
// 返回: 平均耗时 (ms)
// ============================================================================

extern "C" float tr_spmm_forward(
    // SPTC GPU pointers
    int* d_sptc_window_offset,
    half* d_sptc_value,
    uint32_t* d_sptc_packed_meta,
    int* d_sptc_col_old,
    // TC GPU pointers
    int* d_tc_window_offset,
    int* d_tc_offset,
    uint8_t* d_tc_local_id,
    int* d_tc_col_old,
    half* d_tc_value,
    // B matrix (shared)
    half* d_rhs_matrix,
    // Output
    float* d_output_tc,
    float* d_output_sptc,
    // Dimensions
    int num_windows,
    int window_size,
    int dimN,
    int mOri,
    int kOri,
    int num_sptc_blocks,
    int num_tc_blocks,
    int dense_threshold,
    int epoches)
{
    long long elements = (long long)mOri * dimN;
    cudaMemset(d_output_tc, 0, (size_t)elements * sizeof(float));
    cudaMemset(d_output_sptc, 0, (size_t)elements * sizeof(float));
    if (epoches <= 0 || dimN <= 0 || mOri <= 0) return 0.0f;

    // ---- 创建双流 ----
    cudaStream_t stream_tc, stream_sptc;
    cudaStreamCreateWithFlags(&stream_tc, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&stream_sptc, cudaStreamNonBlocking);

    // ---- Warmup (10 iterations) ----
    for (int iter = 0; iter < 10; ++iter) {
        launch_tc_kernel(
            d_tc_window_offset, d_tc_offset, d_tc_local_id,
            d_tc_col_old, d_tc_value,
            d_rhs_matrix, d_output_tc,
            num_windows, num_tc_blocks, dense_threshold,
            dimN, mOri, kOri, stream_tc);

        launch_sptc_kernel(
            d_sptc_window_offset, d_sptc_value, d_sptc_packed_meta,
            d_sptc_col_old,
            d_rhs_matrix, d_output_sptc,
            num_windows, num_sptc_blocks,
            dimN, mOri, kOri, stream_sptc);
    }
    cudaDeviceSynchronize();
    {
        cudaError_t warmup_err = cudaGetLastError();
        if (warmup_err != cudaSuccess) {
            printf("TR SPMM warmup GPU error: %s\n", cudaGetErrorString(warmup_err));
        }
    }
    cudaMemset(d_output_tc, 0, (size_t)elements * sizeof(float));
    cudaMemset(d_output_sptc, 0, (size_t)elements * sizeof(float));
    cudaDeviceSynchronize();

    // ---- 计时 Benchmark ----
    float spmm_ms = 0.0f;
    cudaEvent_t spmm_start, spmm_end;
    cudaEventCreate(&spmm_start);
    cudaEventCreate(&spmm_end);

    cudaEventRecord(spmm_start, stream_tc);
    for (int iter = 0; iter < epoches; ++iter) {
        launch_tc_kernel(
            d_tc_window_offset, d_tc_offset, d_tc_local_id,
            d_tc_col_old, d_tc_value,
            d_rhs_matrix, d_output_tc,
            num_windows, num_tc_blocks, dense_threshold,
            dimN, mOri, kOri, stream_tc);

        launch_sptc_kernel(
            d_sptc_window_offset, d_sptc_value, d_sptc_packed_meta,
            d_sptc_col_old,
            d_rhs_matrix, d_output_sptc,
            num_windows, num_sptc_blocks,
            dimN, mOri, kOri, stream_sptc);
    }

    // 必须等待两个独立流全部完成！
    cudaStreamSynchronize(stream_tc);
    cudaStreamSynchronize(stream_sptc);

    cudaEventRecord(spmm_end, stream_tc);
    cudaEventSynchronize(spmm_end);
    cudaEventElapsedTime(&spmm_ms, spmm_start, spmm_end);
    float spmm_ms_avg = spmm_ms / (float)epoches;

    // ---- 清理: 重置并最后一次运行获取实际结果 ----
    cudaMemset(d_output_tc, 0, (size_t)elements * sizeof(float));
    cudaMemset(d_output_sptc, 0, (size_t)elements * sizeof(float));
    cudaDeviceSynchronize();

    launch_tc_kernel(
        d_tc_window_offset, d_tc_offset, d_tc_local_id,
        d_tc_col_old, d_tc_value,
        d_rhs_matrix, d_output_tc,
        num_windows, num_tc_blocks, dense_threshold,
        dimN, mOri, kOri, stream_tc);

    launch_sptc_kernel(
        d_sptc_window_offset, d_sptc_value, d_sptc_packed_meta,
        d_sptc_col_old,
        d_rhs_matrix, d_output_sptc,
        num_windows, num_sptc_blocks,
        dimN, mOri, kOri, stream_sptc);
    
    // ---- 精准等待 TC 和 SPTC 两个流完成 ----
    cudaStreamSynchronize(stream_tc);
    cudaStreamSynchronize(stream_sptc);

    // ---- 合并 TC 和 SPTC 结果: output_tc += output_sptc ----
    if (elements > 0) {
        int threads = 256;
        int blocks = (int)((elements + threads - 1) / threads);
        if (blocks > 65535) blocks = 65535;
        add_matrix_kernel<<<blocks, threads>>>(d_output_tc, d_output_sptc, elements);
        cudaDeviceSynchronize();
    }

    // ---- 清理 ----
    cudaEventDestroy(spmm_start);
    cudaEventDestroy(spmm_end);
    cudaStreamDestroy(stream_tc);
    cudaStreamDestroy(stream_sptc);

    return spmm_ms_avg;
}
