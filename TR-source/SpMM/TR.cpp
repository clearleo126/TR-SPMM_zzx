/**
 * TR.cpp — TC + SPTC 双路 GPU SpMM Pybind11 接口
 *
 * 功能:
 *   1. 接收 Python 端传递的 TC / SPTC 分块 tensor
 *   2. CPU 端压缩 SPTC 元数据 (int → 2-bit packed uint32)
 *   3. 将数据传输到 GPU 显存
 *   4. 调用 TRkernel.cu 执行双流并行矩阵乘
 *   5. 返回结果矩阵和耗时
 *
 * 参考:
 *   - mGCN_online.cpp (TR-SPMM, Pybind11 接口模式)
 *   - spmm_sp_new.cu (MP-SpMM_SC25, 元数据压缩算法)
 */

#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <vector>
#include <cstring>

// ============================================================================
// CUDA kernel 声明 (定义在 TRkernel.cu)
// ============================================================================

extern "C" float tr_spmm_forward(
    int* d_sptc_chunk_win,            // [num_sptc_chunks] 每 chunk 的窗口 id
    int* d_sptc_chunk_beg,            // [num_sptc_chunks] 块起始
    int* d_sptc_chunk_end,            // [num_sptc_chunks] 块结束
    uint8_t* d_sptc_chunk_atomic,     // [num_sptc_chunks] 1=写回需 atomicAdd
    half* d_sptc_value,
    uint32_t* d_sptc_packed_meta,
    int* d_sptc_col_old,
    int* d_tc_chunk_win,              // [num_chunks] 每 chunk 的窗口 id
    int* d_tc_chunk_beg,              // [num_chunks] 块起始
    int* d_tc_chunk_end,              // [num_chunks] 块结束
    uint8_t* d_tc_chunk_atomic,       // [num_chunks] 1=写回需 atomicAdd
    half* d_tc_value_dense,           // [num_tc_blocks * 256] 16×16 稠密 tiles
    int* d_tc_col_old,                // [num_tc_blocks * 16] (0 pad)
    half* d_rhs_matrix,
    float* d_output,
    int num_windows,
    int window_size,
    int dimN,
    int mOri,
    int kOri,
    int num_sptc_blocks,
    int num_sptc_chunks,
    int num_tc_blocks,
    int num_tc_chunks,
    int dense_threshold,
    int epoches,
    int mode,
    int warmup);


// ============================================================================
// 辅助宏
// ============================================================================

#define CHECK_CPU(x) TORCH_CHECK(!x.is_cuda(), #x " must be a CPU tensor")
#define ENSURE_CONTIGUOUS(x) if (!(x).is_contiguous()) { (x) = (x).contiguous(); }
#define CHECK_DTYPE(x, expected) TORCH_CHECK(x.dtype() == expected, #x " has unexpected dtype")


// ============================================================================
// GPU 内存管理辅助
// ============================================================================

inline cudaError_t checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}

template <typename T>
static void cuda_malloc_or_dummy(T** ptr, int64_t elements) {
    int64_t safe_elements = elements > 0 ? elements : 1;
    checkCuda(cudaMalloc(ptr, safe_elements * (int64_t)sizeof(T)));
}

template <typename T>
static void cuda_copy_h2d_if_needed(T* dst, const T* src, int64_t elements) {
    if (elements > 0) {
        checkCuda(cudaMemcpy(dst, src, elements * (int64_t)sizeof(T), cudaMemcpyHostToDevice));
    }
}


// ============================================================================
// SPTC 元数据压缩: int32 (0-3) → 2-bit packed uint32
//
// 算法参考: MP-SpMM_SC25/mpspmm/SpMM/spmm_sp_new.cu storeArrayInUint32
//
// 每 SPTC block (16×16): 128 个 metadata 条目 → 8 个 uint32_t
// 每个 uint32_t 编码 rows r 和 r+8 的 2:4 稀疏模式:
//   bits [0:1], [2:3], [4:5], [6:7]     → row r,   groups 0-3
//   bits [16:17], [18:19], [20:21], [22:23] → row r+8, groups 0-3
// ============================================================================

static inline int value_to_storage(int value) {
    switch (value) {
        case 0: return 0b00;
        case 1: return 0b01;
        case 2: return 0b10;
        case 3: return 0b11;
        default: return 0;
    }
}

static std::vector<uint32_t> compress_sptc_metadata(
    const int* metadata_old,
    const int* window_offset,
    int num_windows)
{
    int total_blocks = window_offset[num_windows];
    if (total_blocks == 0) return {};

    std::vector<uint32_t> metadata_new(total_blocks * 8, 0);

    for (int win = 0; win < num_windows; ++win) {
        int block_start = window_offset[win];
        int block_end   = window_offset[win + 1];

        for (int block_col = 0; block_col < (block_end - block_start); ++block_col) {
            int block_idx = block_start + block_col;
            int start_row = block_idx * 128;  // 16 rows * 8 entries = 128 per block

            for (int r = 0; r < 8; ++r) {
                int row_low  = r;
                int row_high = r + 8;
                uint32_t val = 0;

                for (int c = 0; c < 4; ++c) {
                    int idx_lower0 = start_row + row_low  * 8 + c * 2;
                    int idx_lower1 = start_row + row_low  * 8 + c * 2 + 1;
                    int idx_upper0 = start_row + row_high * 8 + c * 2;
                    int idx_upper1 = start_row + row_high * 8 + c * 2 + 1;

                    int sv_lower0 = value_to_storage(metadata_old[idx_lower0]);
                    int sv_lower1 = value_to_storage(metadata_old[idx_lower1]);
                    int sv_upper0 = value_to_storage(metadata_old[idx_upper0]);
                    int sv_upper1 = value_to_storage(metadata_old[idx_upper1]);

                    val |= (sv_lower0 << (c * 4));
                    val |= (sv_lower1 << (c * 4 + 2));
                    val |= (sv_upper0 << (c * 4 + 16));
                    val |= (sv_upper1 << (c * 4 + 16 + 2));
                }

                int idx = r + block_idx * 8;
                metadata_new[idx] = val;
            }
        }
    }

    return metadata_new;
}


// ============================================================================
// 主入口: Python → GPU → Python
// ============================================================================

std::vector<torch::Tensor> tr_spmm_forward_py(
    // ---- SPTC tensors (CPU) ----
    torch::Tensor sptc_window_offset,   // [num_windows+1] int32
    torch::Tensor sptc_value,           // [total_sptc_nnz] float16
    torch::Tensor sptc_metadata,        // [total_sptc_nnz] int32 (raw, 0-3)
    torch::Tensor sptc_col_old,         // [num_sptc_blocks * 16] int32
    // ---- TC tensors (CPU) ----
    torch::Tensor tc_window_offset,     // [num_windows+1] int32
    torch::Tensor tc_offset,            // [total_tc_nnz] int32
    torch::Tensor tc_local_id,          // [total_tc_nnz] uint8
    torch::Tensor tc_col_old,           // [num_tc_blocks * dense_threshold] int32
    torch::Tensor tc_value,             // [total_tc_nnz] float16
    // ---- B matrix (CPU) ----
    torch::Tensor rhs_matrix,           // [K * dimN] float16
    // ---- Dimensions ----
    int window_size,
    const int dimN,
    const int mOri,
    const int kOri,
    const int num_windows,
    const int num_sptc_blocks,
    const int num_tc_blocks,
    const int dense_threshold,
    int epoches,
    int mode,
    int warmup)
{
    // ====================================================================
    // 0. 输入验证
    // ====================================================================
    CHECK_CPU(sptc_window_offset);
    CHECK_CPU(sptc_value);
    CHECK_CPU(sptc_metadata);
    CHECK_CPU(sptc_col_old);
    CHECK_CPU(tc_window_offset);
    CHECK_CPU(tc_offset);
    CHECK_CPU(tc_local_id);
    CHECK_CPU(tc_col_old);
    CHECK_CPU(tc_value);
    CHECK_CPU(rhs_matrix);

    ENSURE_CONTIGUOUS(sptc_window_offset);
    ENSURE_CONTIGUOUS(sptc_value);
    ENSURE_CONTIGUOUS(sptc_metadata);
    ENSURE_CONTIGUOUS(sptc_col_old);
    ENSURE_CONTIGUOUS(tc_window_offset);
    ENSURE_CONTIGUOUS(tc_offset);
    ENSURE_CONTIGUOUS(tc_local_id);
    ENSURE_CONTIGUOUS(tc_col_old);
    ENSURE_CONTIGUOUS(tc_value);
    ENSURE_CONTIGUOUS(rhs_matrix);

    CHECK_DTYPE(sptc_window_offset, torch::kInt32);
    CHECK_DTYPE(sptc_value, torch::kFloat16);
    CHECK_DTYPE(sptc_metadata, torch::kInt32);
    CHECK_DTYPE(sptc_col_old, torch::kInt32);
    CHECK_DTYPE(tc_window_offset, torch::kInt32);
    CHECK_DTYPE(tc_offset, torch::kInt32);
    CHECK_DTYPE(tc_local_id, torch::kUInt8);
    CHECK_DTYPE(tc_col_old, torch::kInt32);
    CHECK_DTYPE(tc_value, torch::kFloat16);
    CHECK_DTYPE(rhs_matrix, torch::kFloat16);

    TORCH_CHECK(window_size > 0, "window_size must be positive");
    TORCH_CHECK(dimN > 0, "dimN must be positive");
    TORCH_CHECK(mOri > 0, "mOri must be positive");
    TORCH_CHECK(kOri > 0, "kOri must be positive");
    TORCH_CHECK(epoches > 0, "epoches must be positive");
    TORCH_CHECK(num_windows > 0, "num_windows must be positive");
    TORCH_CHECK(rhs_matrix.numel() == (int64_t)kOri * dimN,
                "rhs_matrix must have kOri * dimN elements");
    TORCH_CHECK(sptc_window_offset.size(0) == num_windows + 1,
                "sptc_window_offset size mismatch");
    TORCH_CHECK(tc_window_offset.size(0) == num_windows + 1,
                "tc_window_offset size mismatch");

    // ====================================================================
    // 1. 压缩 SPTC 元数据 (CPU 端)
    // ====================================================================
    auto sptc_metadata_compress_start = std::chrono::high_resolution_clock::now();

    const int* sptc_metadata_ptr = sptc_metadata.data_ptr<int>();
    const int* sptc_window_ptr   = sptc_window_offset.data_ptr<int>();

    std::vector<uint32_t> sptc_packed_meta_vec = compress_sptc_metadata(
        sptc_metadata_ptr, sptc_window_ptr, num_windows);

    auto sptc_metadata_compress_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> compress_elapsed =
        sptc_metadata_compress_end - sptc_metadata_compress_start;
    printf("[TR.cpp] SPTC metadata compression: %.4f ms\n",
           compress_elapsed.count() * 1000.0);

    int64_t num_packed_meta = (int64_t)sptc_packed_meta_vec.size();

    // ====================================================================
    // 2. 获取 CPU 数据指针
    // ====================================================================
    const int* sptc_window_offset_ = sptc_window_offset.data_ptr<int>();
    const half* sptc_value_ = reinterpret_cast<const half*>(sptc_value.data_ptr<at::Half>());
    const int* sptc_col_old_ = sptc_col_old.data_ptr<int>();

    const int* tc_window_offset_ = tc_window_offset.data_ptr<int>();
    const int* tc_offset_ = tc_offset.data_ptr<int>();
    const uint8_t* tc_local_id_ = tc_local_id.data_ptr<uint8_t>();
    const int* tc_col_old_ = tc_col_old.data_ptr<int>();
    const half* tc_value_ = reinterpret_cast<const half*>(tc_value.data_ptr<at::Half>());

    const half* rhs_matrix_ = reinterpret_cast<const half*>(rhs_matrix.data_ptr<at::Half>());

    // ====================================================================
    // 2.4 清洗 SPTC col_old: 匹配算法用 -1 表示虚拟全0列 (A=0, mma 贡献为 0),
    //     替换为 0 使内核 B 加载保持合法地址, 可走无越界检查的快路径.
    // ====================================================================
    std::vector<int> sptc_col_old_clean((size_t)sptc_col_old.numel());
    for (int64_t i = 0; i < sptc_col_old.numel(); ++i)
        sptc_col_old_clean[(size_t)i] = (sptc_col_old_[i] < 0) ? 0 : sptc_col_old_[i];

    // ====================================================================
    // 2.5 构建 TC 稠密 A tiles (CPU 端): [num_tc_blocks * 256] half, 16×16
    //     布局 [r*16+c] 与 tc_local_id 一致, 未用列补 0;
    //     col_old 补齐到 16/block (0 填充: 对应 A=0, mma 贡献 0, 且 B 加载
    //     保持合法地址, 可走快路径).
    //     → 内核每线程直接 __ldg 加载 A fragment, 免去 smem scatter + 同步.
    // ====================================================================
    auto tc_densify_start = std::chrono::high_resolution_clock::now();
    TORCH_CHECK(dense_threshold <= 16,
                "dense_threshold must be <= 16 for dense-tile layout");
    std::vector<half> tc_value_dense_vec((size_t)num_tc_blocks * 256, __float2half(0.0f));
    std::vector<int>  tc_col_old_pad_vec((size_t)num_tc_blocks * 16, 0);
    if (num_tc_blocks > 0) {
        for (int blk = 0; blk < num_tc_blocks; ++blk) {
            half* tile = tc_value_dense_vec.data() + (size_t)blk * 256;
            int nnz_start = tc_offset_[blk];
            int nnz_end   = tc_offset_[blk + 1];
            for (int n = nnz_start; n < nnz_end; ++n) {
                int local_id = tc_local_id_[n];   // r*16+c, c < dense_threshold
                tile[local_id] = tc_value_[n];
            }
            int* pad_col = tc_col_old_pad_vec.data() + (size_t)blk * 16;
            for (int k = 0; k < dense_threshold; ++k)
                pad_col[k] = tc_col_old_[blk * dense_threshold + k];
        }
    }
    auto tc_densify_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> densify_elapsed = tc_densify_end - tc_densify_start;
    printf("[TR.cpp] TC dense-tile build: %.4f ms\n",
           densify_elapsed.count() * 1000.0);

    // ====================================================================
    // 2.6 窗口内分块 (chunking): 把每窗口的 TC 块切成 ≤TC_CHUNK 的 chunk
    //     消除 straggler 窗口 (单窗口可达数千块) 造成的长尾.
    //     tc_chunk_atomic[c]: 该 chunk 的写回是否需要 atomicAdd
    //       (窗口有多个 TC chunk 或窗口同时有 SPTC 块 → 需要原子写)
    // ====================================================================
    const int TC_CHUNK = 64;
    std::vector<int> tc_chunk_win, tc_chunk_beg, tc_chunk_end;
    std::vector<uint8_t> tc_chunk_atomic;
    tc_chunk_win.reserve(num_windows + num_tc_blocks / TC_CHUNK + 16);
    tc_chunk_beg.reserve(tc_chunk_win.capacity());
    tc_chunk_end.reserve(tc_chunk_win.capacity());
    tc_chunk_atomic.reserve(tc_chunk_win.capacity());
    for (int w = 0; w < num_windows; ++w) {
        int b0 = tc_window_offset_[w];
        int b1 = tc_window_offset_[w + 1];
        if (b0 >= b1) continue;
        int n_chunks = (b1 - b0 + TC_CHUNK - 1) / TC_CHUNK;
        bool has_sptc = (sptc_window_offset_[w + 1] > sptc_window_offset_[w]);
        for (int b = b0; b < b1; b += TC_CHUNK) {
            tc_chunk_win.push_back(w);
            tc_chunk_beg.push_back(b);
            tc_chunk_end.push_back(b + TC_CHUNK < b1 ? b + TC_CHUNK : b1);
            tc_chunk_atomic.push_back((n_chunks > 1 || has_sptc) ? 1 : 0);
        }
    }
    int num_tc_chunks = (int)tc_chunk_win.size();
    printf("[TR.cpp] TC chunks: %d (blocks: %d, CHUNK=%d)\n",
           num_tc_chunks, num_tc_blocks, TC_CHUNK);

    // ====================================================================
    // 2.7 SPTC 窗口内分块 (同 TC): 每窗口 SPTC 块切成 ≤SPTC_CHUNK 的 chunk
    //     sptc_chunk_atomic[c]: 需要 atomicAdd 的标记 (多 chunk 或窗口有 TC)
    // ====================================================================
    const int SPTC_CHUNK = 64;
    std::vector<int> sptc_chunk_win, sptc_chunk_beg, sptc_chunk_end;
    std::vector<uint8_t> sptc_chunk_atomic;
    sptc_chunk_win.reserve(num_windows + num_sptc_blocks / SPTC_CHUNK + 16);
    sptc_chunk_beg.reserve(sptc_chunk_win.capacity());
    sptc_chunk_end.reserve(sptc_chunk_win.capacity());
    sptc_chunk_atomic.reserve(sptc_chunk_win.capacity());
    for (int w = 0; w < num_windows; ++w) {
        int b0 = sptc_window_offset_[w];
        int b1 = sptc_window_offset_[w + 1];
        if (b0 >= b1) continue;
        int n_chunks = (b1 - b0 + SPTC_CHUNK - 1) / SPTC_CHUNK;
        bool has_tc = (tc_window_offset_[w + 1] > tc_window_offset_[w]);
        for (int b = b0; b < b1; b += SPTC_CHUNK) {
            sptc_chunk_win.push_back(w);
            sptc_chunk_beg.push_back(b);
            sptc_chunk_end.push_back(b + SPTC_CHUNK < b1 ? b + SPTC_CHUNK : b1);
            sptc_chunk_atomic.push_back((n_chunks > 1 || has_tc) ? 1 : 0);
        }
    }
    int num_sptc_chunks = (int)sptc_chunk_win.size();
    printf("[TR.cpp] SPTC chunks: %d (blocks: %d, CHUNK=%d)\n",
           num_sptc_chunks, num_sptc_blocks, SPTC_CHUNK);

    // ====================================================================
    // 3. 分配 GPU 内存
    // ====================================================================
    int *d_sptc_chunk_win, *d_sptc_chunk_beg, *d_sptc_chunk_end;
    uint8_t *d_sptc_chunk_atomic;
    half *d_sptc_value;
    uint32_t *d_sptc_packed_meta;
    int *d_sptc_col_old;
    int *d_tc_chunk_win, *d_tc_chunk_beg, *d_tc_chunk_end;
    uint8_t *d_tc_chunk_atomic;
    int *d_tc_col_old;
    half *d_tc_value_dense;
    half *d_rhs_matrix;
    float *d_output;
    // 对齐16的倍数
    int mOri_padded = num_windows * window_size;

    cuda_malloc_or_dummy(&d_sptc_chunk_win, (int64_t)num_sptc_chunks);
    cuda_malloc_or_dummy(&d_sptc_chunk_beg, (int64_t)num_sptc_chunks);
    cuda_malloc_or_dummy(&d_sptc_chunk_end, (int64_t)num_sptc_chunks);
    cuda_malloc_or_dummy(&d_sptc_chunk_atomic, (int64_t)num_sptc_chunks);
    cuda_malloc_or_dummy(&d_sptc_value, sptc_value.numel());
    cuda_malloc_or_dummy(&d_sptc_packed_meta, num_packed_meta);
    cuda_malloc_or_dummy(&d_sptc_col_old, sptc_col_old.numel());
    cuda_malloc_or_dummy(&d_tc_chunk_win, (int64_t)num_tc_chunks);
    cuda_malloc_or_dummy(&d_tc_chunk_beg, (int64_t)num_tc_chunks);
    cuda_malloc_or_dummy(&d_tc_chunk_end, (int64_t)num_tc_chunks);
    cuda_malloc_or_dummy(&d_tc_chunk_atomic, (int64_t)num_tc_chunks);
    cuda_malloc_or_dummy(&d_tc_col_old, (int64_t)num_tc_blocks * 16);
    cuda_malloc_or_dummy(&d_tc_value_dense, (int64_t)num_tc_blocks * 256);
    cuda_malloc_or_dummy(&d_rhs_matrix, (int64_t)kOri * dimN);
    cuda_malloc_or_dummy(&d_output, (int64_t)mOri_padded * dimN);

    // ====================================================================
    // 4. 拷贝数据到 GPU
    // ====================================================================
    auto h2d_start = std::chrono::high_resolution_clock::now();
    cuda_copy_h2d_if_needed(d_sptc_chunk_win, sptc_chunk_win.data(), (int64_t)num_sptc_chunks);
    cuda_copy_h2d_if_needed(d_sptc_chunk_beg, sptc_chunk_beg.data(), (int64_t)num_sptc_chunks);
    cuda_copy_h2d_if_needed(d_sptc_chunk_end, sptc_chunk_end.data(), (int64_t)num_sptc_chunks);
    cuda_copy_h2d_if_needed(d_sptc_chunk_atomic, sptc_chunk_atomic.data(), (int64_t)num_sptc_chunks);
    cuda_copy_h2d_if_needed(d_sptc_value, sptc_value_, sptc_value.numel());
    cuda_copy_h2d_if_needed(d_sptc_col_old, sptc_col_old_clean.data(), sptc_col_old.numel());
    cuda_copy_h2d_if_needed(d_tc_chunk_win, tc_chunk_win.data(), (int64_t)num_tc_chunks);
    cuda_copy_h2d_if_needed(d_tc_chunk_beg, tc_chunk_beg.data(), (int64_t)num_tc_chunks);
    cuda_copy_h2d_if_needed(d_tc_chunk_end, tc_chunk_end.data(), (int64_t)num_tc_chunks);
    cuda_copy_h2d_if_needed(d_tc_chunk_atomic, tc_chunk_atomic.data(), (int64_t)num_tc_chunks);
    cuda_copy_h2d_if_needed(d_tc_col_old, tc_col_old_pad_vec.data(), (int64_t)num_tc_blocks * 16);
    cuda_copy_h2d_if_needed(d_tc_value_dense, tc_value_dense_vec.data(), (int64_t)num_tc_blocks * 256);
    cuda_copy_h2d_if_needed(d_rhs_matrix, rhs_matrix_, (int64_t)kOri * dimN);

    // 压缩后的元数据 (需要从 vector 拷贝)
    if (num_packed_meta > 0) {
        checkCuda(cudaMemcpy(d_sptc_packed_meta, sptc_packed_meta_vec.data(),
                   num_packed_meta * sizeof(uint32_t), cudaMemcpyHostToDevice));
    }
    auto h2d_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> h2d_elapsed = h2d_end - h2d_start;
    printf("[TR.cpp] H2D copy: %.4f ms\n", h2d_elapsed.count() * 1000.0);

    // ====================================================================
    // 5. 启动 CUDA 内核
    // ====================================================================
    float spmm_ms_avg = tr_spmm_forward(
        d_sptc_chunk_win,
        d_sptc_chunk_beg,
        d_sptc_chunk_end,
        d_sptc_chunk_atomic,
        d_sptc_value,
        d_sptc_packed_meta,
        d_sptc_col_old,
        d_tc_chunk_win,
        d_tc_chunk_beg,
        d_tc_chunk_end,
        d_tc_chunk_atomic,
        d_tc_value_dense,
        d_tc_col_old,
        d_rhs_matrix,
        d_output,
        num_windows,
        window_size,
        dimN,
        mOri,
        kOri,
        num_sptc_blocks,
        num_sptc_chunks,
        num_tc_blocks,
        num_tc_chunks,
        dense_threshold,
        epoches,
        mode,
        warmup);

    // ====================================================================
    // 6. 拷贝结果回 CPU
    // ====================================================================
    auto output_matrix = torch::empty({mOri_padded, dimN}, torch::kFloat32).to(torch::kCPU);
    float* output_ptr = output_matrix.data_ptr<float>();

    checkCuda(cudaMemcpy(output_ptr, d_output,
               (int64_t)mOri_padded * dimN * sizeof(float), cudaMemcpyDeviceToHost));

    // ====================================================================
    // 7. 释放 GPU 内存
    // ====================================================================
    cudaFree(d_sptc_chunk_win);
    cudaFree(d_sptc_chunk_beg);
    cudaFree(d_sptc_chunk_end);
    cudaFree(d_sptc_chunk_atomic);
    cudaFree(d_sptc_value);
    cudaFree(d_sptc_packed_meta);
    cudaFree(d_sptc_col_old);
    cudaFree(d_tc_chunk_win);
    cudaFree(d_tc_chunk_beg);
    cudaFree(d_tc_chunk_end);
    cudaFree(d_tc_chunk_atomic);
    cudaFree(d_tc_col_old);
    cudaFree(d_tc_value_dense);
    cudaFree(d_rhs_matrix);
    cudaFree(d_output);
    cudaDeviceSynchronize();

    return {output_matrix.narrow(0, 0, mOri), torch::tensor(spmm_ms_avg)};
}


// ============================================================================
// Pybind11 模块注册
// ============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = R"pbdoc(
        TR_SPMM: TC + SPTC Dual-Routing GPU SpMM

        功能:
          将 block_matching_zhy.cpp 分块后的 TC/SPTC 数据传入 GPU,
          使用双 CUDA Stream 并行执行:
            - TC  kernel: 稠密 Tensor Core GEMM (mma.sync.aligned.m16n8k8)
            - SPTC kernel: 结构化稀疏 Tensor Core GEMM (mma.sp.sync.aligned.m16n8k16)

        接口:
          forward(sptc_window_offset, sptc_value, sptc_metadata,
                  sptc_col_old, tc_window_offset, tc_offset,
                  tc_local_id, tc_col_old, tc_value,
                  rhs_matrix, window_size, dimN, mOri, kOri,
                  num_windows, num_sptc_blocks, num_tc_blocks,
                  dense_threshold, epoches) -> [output_matrix, elapsed_ms]
    )pbdoc";

    m.def("forward", &tr_spmm_forward_py,
          py::arg("sptc_window_offset"),
          py::arg("sptc_value"),
          py::arg("sptc_metadata"),
          py::arg("sptc_col_old"),
          py::arg("tc_window_offset"),
          py::arg("tc_offset"),
          py::arg("tc_local_id"),
          py::arg("tc_col_old"),
          py::arg("tc_value"),
          py::arg("rhs_matrix"),
          py::arg("window_size"),
          py::arg("dimN"),
          py::arg("mOri"),
          py::arg("kOri"),
          py::arg("num_windows"),
          py::arg("num_sptc_blocks"),
          py::arg("num_tc_blocks"),
          py::arg("dense_threshold"),
          py::arg("epoches"),
          py::arg("mode") = 0,
          py::arg("warmup") = 50,
          R"pbdoc(
对分块后的 TC/SPTC 数据执行 GPU 矩阵乘运算.

参数:
    sptc_window_offset: [num_windows+1] int32, SPTC block 的窗口偏移 (CSR)
    sptc_value:         [total_sptc_nnz] float16, SPTC A 值 (128/block)
    sptc_metadata:      [total_sptc_nnz] int32, SPTC 元数据 (0-3, 128/block)
    sptc_col_old:       [num_sptc_blocks*16] int32, SPTC 原始列索引
    tc_window_offset:   [num_windows+1] int32, TC block 的窗口偏移 (CSR)
    tc_offset:          [total_tc_nnz] int32, TC NNZ 前缀和
    tc_local_id:        [total_tc_nnz] uint8, TC 局部位置 (r*16+c)
    tc_col_old:         [num_tc_blocks*dense_threshold] int32, TC 原始列索引
    tc_value:           [total_tc_nnz] float16, TC A 值
    rhs_matrix:         [kOri * dimN] float16, B 矩阵 (行优先)
    window_size:        窗口大小 (默认 16)
    dimN:               特征维度 N
    mOri:               原始矩阵行数 M
    kOri:               原始矩阵列数 K
    num_windows:        窗口总数
    num_sptc_blocks:    SPTC block 总数
    num_tc_blocks:      TC block 总数
    dense_threshold:    TC block K 维度 (默认 8)
    epoches:            Benchmark 迭代次数

返回:
    [output_matrix [mOri, dimN] float32, elapsed_ms float32]
          )pbdoc");
}