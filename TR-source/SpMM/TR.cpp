/**
 * TR.cpp — DR-SpMM 主实现: TC + SPTC 双路 GPU SpMM Pybind11 接口
 *
 * 论文正文口径 (所有优化路径默认全开): 窗口行内 2:4 结构稀疏走 SPTC,
 * 孤儿列汇聚稠密块走 TC, 含窗口级融合 / chunk 路由 / 双流并发, 以及
 * 设备端预打包缓存 (按 data_ptr 索引) 与 CUDA 输入零拷贝.
 *
 * 消融版本 (ablation/chunk 参数可关闭各组件) 见 ../TR-abl-source/.
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
#include <unordered_map>

// ============================================================================
// CUDA kernel 声明 (定义在 TRkernel.cu)
// ============================================================================

extern "C" float tr_spmm_forward(
    int* d_sptc_chunk_win,            // [num_sptc_chunks] 每 chunk 的窗口 id
    int* d_sptc_chunk_beg,            // [num_sptc_chunks] 块起始
    int* d_sptc_chunk_end,            // [num_sptc_chunks] 块结束
    uint8_t* d_sptc_chunk_atomic,     // [num_sptc_chunks] 1=写回需 atomicAdd
    int* d_sptc_block_row,            // [num_sptc_blocks] 每块输出行首 (window_id*16)
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
    int warmup,
    int ablation,
    int chunk_size,
    int serial_route,
    int num_fused,
    int* d_fused_win, int* d_fused_tc_beg, int* d_fused_tc_end,
    int* d_fused_sp_beg, int* d_fused_sp_end);


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
// 预打包缓存 (训练循环用)
//
// forward() 每次都会重做 SPTC 元数据压缩 / TC 稠密化 / chunk 表构建 / H2D,
// 在 N=128 的小矩阵上这部分可达 kernel 本身的 200 倍. 同一份 packed 结构在
// 训练中被反复调用时, 只有 rhs_matrix 会变, 其余全部可复用.
//
// 命中判据: 四个源 tensor 的 data_ptr + 窗口数 + 维度 + 路由参数完全一致.
// 未命中时释放旧缓存并整体重建.
// ============================================================================
struct TRPreparedState {
    bool valid = false;
    const void *k_meta = nullptr, *k_sw = nullptr, *k_tcw = nullptr, *k_tcc = nullptr;
    int64_t k_sw_size = 0;
    int k_dimN = 0, k_mOri = 0, k_kOri = 0, k_mode = 0, k_abl = 0, k_chunk = 0, k_dt = 0;

    int *d_sptc_chunk_win = nullptr, *d_sptc_chunk_beg = nullptr, *d_sptc_chunk_end = nullptr;
    uint8_t *d_sptc_chunk_atomic = nullptr;
    int *d_sptc_block_row = nullptr;
    half *d_sptc_value = nullptr;
    uint32_t *d_sptc_packed_meta = nullptr;
    int *d_sptc_col_old = nullptr;
    int *d_tc_chunk_win = nullptr, *d_tc_chunk_beg = nullptr, *d_tc_chunk_end = nullptr;
    uint8_t *d_tc_chunk_atomic = nullptr;
    int *d_tc_col_old = nullptr;
    half *d_tc_value_dense = nullptr;
    int *d_fused_win = nullptr, *d_fused_tc_beg = nullptr, *d_fused_tc_end = nullptr;
    int *d_fused_sp_beg = nullptr, *d_fused_sp_end = nullptr;
    half *d_rhs_matrix = nullptr;
    float *d_output = nullptr;

    int num_sptc_chunks = 0, num_tc_chunks = 0, num_fused = 0, mOri_padded = 0;
    int num_sptc_blocks = 0, num_tc_blocks = 0;
    bool serial_route = false;

    void release() {
        if (!valid) return;
        cudaFree(d_sptc_chunk_win); cudaFree(d_sptc_chunk_beg); cudaFree(d_sptc_chunk_end);
        cudaFree(d_sptc_chunk_atomic); cudaFree(d_sptc_block_row); cudaFree(d_sptc_value);
        cudaFree(d_sptc_packed_meta); cudaFree(d_sptc_col_old);
        cudaFree(d_tc_chunk_win); cudaFree(d_tc_chunk_beg); cudaFree(d_tc_chunk_end);
        cudaFree(d_tc_chunk_atomic); cudaFree(d_tc_col_old); cudaFree(d_tc_value_dense);
        cudaFree(d_fused_win); cudaFree(d_fused_tc_beg); cudaFree(d_fused_tc_end);
        cudaFree(d_fused_sp_beg); cudaFree(d_fused_sp_end);
        cudaFree(d_rhs_matrix); cudaFree(d_output);
        *this = TRPreparedState();
    }
};

// 训练循环中 A 与 A^T 的 packed 结构会交替调用 (fwd 用 A, bwd 用 A^T),
// 故必须多槽缓存, 单槽每次都会被覆盖.
static std::unordered_map<const void*, TRPreparedState> g_tr_prep_map;

static TRPreparedState* tr_prep_lookup(const void* key) {
    auto it = g_tr_prep_map.find(key);
    return it == g_tr_prep_map.end() ? nullptr : &it->second;
}

static void tr_prep_store(const void* key, const TRPreparedState& st) {
    auto it = g_tr_prep_map.find(key);
    if (it != g_tr_prep_map.end()) it->second.release();
    if (g_tr_prep_map.size() >= 8) {   // 防止换矩阵时无限累积
        for (auto& kv : g_tr_prep_map) kv.second.release();
        g_tr_prep_map.clear();
    }
    g_tr_prep_map[key] = st;
}

// 显式清空预打包缓存 (换矩阵时调用, 避免多个矩阵的 packed 同时驻留显存)
static void tr_prep_clear_all() {
    for (auto& kv : g_tr_prep_map) kv.second.release();
    g_tr_prep_map.clear();
}


// 计算指定类型 (TC/SPTC) 的 block 分块大小 (每 CTA 处理的 block 数上限)
//
// chunking 的净收益 = straggler 拖尾均衡收益 − 跨 chunk atomic 写回开销.
// 仅看平均窗口大小无法区分两种"大窗"矩阵:
//   - mip1:    TC avg=22  max=8299 (max/avg=375) → 少数巨窗拖尾, 必须切
//   - protein: SPTC avg=307 max=899 (max/avg=2.9) → 窗口大但均匀, 且
//              8284 个窗口已足够填满 GPU, 再切只增加 atomic → 不该切
// 因此判据用三维 (偏斜度, 并行度, 单窗工作量):
//   1) 偏斜度: 存在拖尾窗口 (max ≥ 8×avg 且 max > 64) → 必须切分
//   2) 并行度: 窗口数 ≥ 16 CTA/SM, 且均窗 ≥ 128 块 (单窗工作量)
//      → 整窗单 CTA, 避免 atomic
//   3) 其余 (窗口少 / 无拖尾 / 中窗) → 仍切分以暴露并行度
// 返回 (1<<30) 表示"不切分"(整窗单 CTA).
int compute_chunk_size(int num_windows, int num_blocks, int max_window) {
    if (num_blocks <= 0 || num_windows <= 0) return 64;
    const int NO_CHUNK = 1 << 30;
    double avg_window = (double)num_blocks / num_windows;

    // 1) 拖尾窗口主导 → 切分, 档位仍按平均窗口大小选
    bool has_straggler = (max_window >= 8.0 * (avg_window > 1.0 ? avg_window : 1.0))
                         && (max_window > 64);
    if (!has_straggler) {
        // 2) 无拖尾 + 窗口数足够多 → 不切分 (消除 atomic 写回)
        //    要求 16 CTA/SM 以上: 整窗单 CTA 时每 CTA 需串行处理整窗块,
        //    只有 CTA 数足够多才能把窗口间负载差异摊平 (实测 629 窗口 /
        //    每窗 115 块时整窗单 CTA 反而慢 3 倍, 而 8284 窗口 / 每窗 307
        //    块时整窗更快).
        int sm_count = 82;  // 回退值: RTX 3090
        int dev = 0;
        cudaDeviceProp prop;
        if (cudaGetDevice(&dev) == cudaSuccess &&
            cudaGetDeviceProperties(&prop, dev) == cudaSuccess) {
            sm_count = prop.multiProcessorCount;
        }
        //    单窗工作量要求 >= 128 块: 实测均窗 20~40 块的中窗整窗单 CTA 反而
        //    偏慢, 而均窗 >= 128 块时整窗更快 (net100 均窗 37 块: 整窗 0.257ms
        //    vs chunk64 0.231ms; net150 39 块: 0.389 vs 0.325; GaAsH6 23 块:
        //    0.261 vs 0.240; protein 均窗 307 块则整窗最优). 128 为实测拐点.
        //    注意切分只发生在窗口内部, chunk 不会跨窗口合并.
        if (num_windows >= 16 * sm_count && avg_window >= 128.0) return NO_CHUNK;
    }
    // 3) 切分档位
    //    巨窗档 (均窗 > 512 块) 取 512 而非 128: 这类矩阵整窗块数虽大, 但
    //    窗口数少 (< GPU 容量), 需要的是"够多的 CTA"而非"够细的块".
    //    实测 12month1 (SPTC 均窗 1133 块/780 窗): 128->4.722ms,
    //    256->4.552ms, 512->4.468ms, 1024->4.514ms;
    //    spal_004 (均窗 3932 块/638 窗): 128->12.767ms, 256->11.933ms,
    //    512->11.877ms. 全量 559 个矩阵中仅这 2 个进入该档.
    if (avg_window < 32.0) return 32;
    if (avg_window > 512.0) return 512;
    return 64;
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
    int warmup,
    int ablation,
    int chunk_size)
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
    // rhs_matrix 可为 CPU 或 CUDA tensor:
    //   Python 侧对 B 矩阵直接在 GPU 上生成 (避免 CPU fp16 randn + H2D 拷贝),
    //   此时 data_ptr 为设备指针, 后续用 DeviceToDevice 拷贝.
    TORCH_CHECK(rhs_matrix.is_cpu() || rhs_matrix.is_cuda(),
                "rhs_matrix must be a CPU or CUDA tensor");

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
    // 1. [预打包缓存] 命中则跳过全部 CPU 预处理与 H2D, 只更新 rhs
    // ====================================================================
    int *d_sptc_chunk_win = nullptr, *d_sptc_chunk_beg = nullptr, *d_sptc_chunk_end = nullptr;
    uint8_t *d_sptc_chunk_atomic = nullptr;
    int *d_sptc_block_row = nullptr;
    half *d_sptc_value = nullptr;
    uint32_t *d_sptc_packed_meta = nullptr;
    int *d_sptc_col_old = nullptr;
    int *d_tc_chunk_win = nullptr, *d_tc_chunk_beg = nullptr, *d_tc_chunk_end = nullptr;
    uint8_t *d_tc_chunk_atomic = nullptr;
    int *d_tc_col_old = nullptr;
    half *d_tc_value_dense = nullptr;
    int *d_fused_win = nullptr, *d_fused_tc_beg = nullptr, *d_fused_tc_end = nullptr;
    int *d_fused_sp_beg = nullptr, *d_fused_sp_end = nullptr;
    half *d_rhs_matrix = nullptr;
    float *d_output = nullptr;
    int mOri_padded = num_windows * window_size;
    int num_sptc_chunks = 0, num_tc_chunks = 0, num_fused = 0;
    bool serial_route = false;
    const half* rhs_matrix_ = reinterpret_cast<const half*>(rhs_matrix.data_ptr<at::Half>());

    const void* prep_key = (const void*)sptc_metadata.data_ptr<int>();
    TRPreparedState* prep = tr_prep_lookup(prep_key);
    bool prep_hit = prep && prep->valid &&
                    prep->k_meta == prep_key &&
                    prep->k_sw   == (const void*)sptc_window_offset.data_ptr<int>() &&
                    prep->k_tcw  == (const void*)tc_window_offset.data_ptr<int>() &&
                    prep->k_tcc  == (const void*)tc_col_old.data_ptr<int>() &&
                    prep->k_sw_size == (int64_t)sptc_window_offset.numel() &&
                    prep->k_dimN == dimN && prep->k_mOri == mOri &&
                    prep->k_kOri == kOri && prep->k_mode == mode &&
                    prep->k_abl == ablation && prep->k_chunk == chunk_size &&
                    prep->k_dt == dense_threshold;

    if (prep_hit) {
        d_sptc_chunk_win = prep->d_sptc_chunk_win;
        d_sptc_chunk_beg = prep->d_sptc_chunk_beg;
        d_sptc_chunk_end = prep->d_sptc_chunk_end;
        d_sptc_chunk_atomic = prep->d_sptc_chunk_atomic;
        d_sptc_block_row = prep->d_sptc_block_row;
        d_sptc_value = prep->d_sptc_value;
        d_sptc_packed_meta = prep->d_sptc_packed_meta;
        d_sptc_col_old = prep->d_sptc_col_old;
        d_tc_chunk_win = prep->d_tc_chunk_win;
        d_tc_chunk_beg = prep->d_tc_chunk_beg;
        d_tc_chunk_end = prep->d_tc_chunk_end;
        d_tc_chunk_atomic = prep->d_tc_chunk_atomic;
        d_tc_col_old = prep->d_tc_col_old;
        d_tc_value_dense = prep->d_tc_value_dense;
        d_fused_win = prep->d_fused_win;
        d_fused_tc_beg = prep->d_fused_tc_beg;
        d_fused_tc_end = prep->d_fused_tc_end;
        d_fused_sp_beg = prep->d_fused_sp_beg;
        d_fused_sp_end = prep->d_fused_sp_end;
        d_rhs_matrix = prep->d_rhs_matrix;
        d_output = prep->d_output;
        mOri_padded = prep->mOri_padded;
        num_sptc_chunks = prep->num_sptc_chunks;
        num_tc_chunks = prep->num_tc_chunks;
        num_fused = prep->num_fused;
        serial_route = prep->serial_route;
    } else {

    // ====================================================================
    // 1b. 压缩 SPTC 元数据 (CPU 端)
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

    // rhs_matrix_ 已在函数开头定义 (共用段使用)

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
    // 2.6 窗口内分块 (chunking): 按窗口大小分档
    //     [E-chunk 扫描] chunk_size: 0=自适应(auto, 由 compute_chunk_size 按
    //     矩阵窗口分布选档), -1=整窗单 CTA(禁用 chunking), >0=固定每 CTA
    //     处理 chunk_size 个 block.
    //     [消融 no-E] 强制每窗口单 CTA (无 chunk 间接层).
    //     Tier 1 (小窗, 整窗单 CTA): 窗口块数 ≤ base_chunk 时整个 window 一个
    //       CTA; 若该窗口仅 TC 有块 (无 SPTC 共窗) → 完全无 atomic (单 CTA
    //       寄存器累加 + 单次写回, 无 block 竞争). 若窗口同时有 SPTC 块,
    //       因双流并发写同 row 区, 仍须 atomic 保正确.
    //     Tier 2 (大窗 / straggler): 按 straggler_chunk 分片, 多 CTA 并行;
    //       分片粒度取 max(默认, base_chunk), 避免巨窗矩阵的 base 档位被架空.
    //     tc_chunk_atomic[c]: 该 chunk 写回是否需要 atomicAdd.
    // ====================================================================
    const bool no_chunk = (ablation == 5) || (chunk_size < 0);  // [消融 no-E / chunk=-1]
    const int fixed_chunk = (chunk_size > 0) ? chunk_size : 0;  // [E-chunk 扫描]
    // 最大窗口块数 (compute_chunk_size 的偏斜度判据)
    int tc_max_window = 0;
    for (int w = 0; w < num_windows; ++w) {
        int n = tc_window_offset_[w + 1] - tc_window_offset_[w];
        if (n > tc_max_window) tc_max_window = n;
    }
    const int tc_base_chunk = no_chunk ? (1 << 30)
                            : (fixed_chunk > 0 ? fixed_chunk
                                               : compute_chunk_size(num_windows, num_tc_blocks,
                                                                    tc_max_window));
    // straggler 分片粒度跟随 base 档位: 巨窗矩阵 (base=512) 若仍按 128 切,
    //   等于 base 档位被 Tier2 架空 (实测 12month1 base=512+straggler=64 反而
    //   从 4.72 退化到 4.94ms). 故取 max(默认, base).
    const int tc_straggler_chunk = no_chunk ? 128
                                 : (fixed_chunk > 0 ? fixed_chunk
                                                    : (tc_base_chunk > 128 ? tc_base_chunk : 128));
    std::vector<int> tc_chunk_win, tc_chunk_beg, tc_chunk_end;
    std::vector<uint8_t> tc_chunk_atomic;
    tc_chunk_win.reserve(num_windows + num_tc_blocks / tc_base_chunk + 16);
    tc_chunk_beg.reserve(tc_chunk_win.capacity());
    tc_chunk_end.reserve(tc_chunk_win.capacity());
    tc_chunk_atomic.reserve(tc_chunk_win.capacity());
    for (int w = 0; w < num_windows; ++w) {
        int b0 = tc_window_offset_[w];
        int b1 = tc_window_offset_[w + 1];
        if (b0 >= b1) continue;
        bool has_sptc = (sptc_window_offset_[w + 1] > sptc_window_offset_[w]);
        int n_blocks = b1 - b0;
        if (n_blocks <= tc_base_chunk) {
            // Tier 1: 整窗单 CTA
            tc_chunk_win.push_back(w);
            tc_chunk_beg.push_back(b0);
            tc_chunk_end.push_back(b1);
            tc_chunk_atomic.push_back(has_sptc ? 1 : 0);
            continue;
        }
        // Tier 2: straggler 大窗分片, atomic 写回
        int chunk = tc_straggler_chunk;
        int n_chunks = (n_blocks + chunk - 1) / chunk;
        for (int b = b0; b < b1; b += chunk) {
            tc_chunk_win.push_back(w);
            tc_chunk_beg.push_back(b);
            tc_chunk_end.push_back(b + chunk < b1 ? b + chunk : b1);
            tc_chunk_atomic.push_back((n_chunks > 1 || has_sptc) ? 1 : 0);
        }
    }
    num_tc_chunks = (int)tc_chunk_win.size();
    printf("[TR.cpp] TC chunks: %d (blocks: %d, CHUNK=%d, straggler=%d)\n",
           num_tc_chunks, num_tc_blocks, tc_base_chunk, tc_straggler_chunk);

    // ====================================================================
    // 2.7 SPTC 窗口内分块 (Tier1 整窗单 CTA + Tier2 大窗分片)
    //     (flat chunking 实验已证伪: 跨窗口 flush 写回 + atomic 退化,
    //      恢复窗口粒度, 每窗单 CTA 整窗 RC 累加 + 单次写回)
    //     sptc_chunk_atomic[c]: 1=窗口含 TC 双流写, 需 atomicAdd
    // ====================================================================
    // 2.7.0 每块输出行号表 (kernel flush_rc 用; 窗口版下每块行号=窗口首行)
    std::vector<int> sptc_block_row((size_t)num_sptc_blocks, 0);
    {
        const int* sw = sptc_window_offset_;
        for (int w = 0; w < num_windows; ++w) {
            for (int b = sw[w]; b < sw[w + 1]; ++b)
                sptc_block_row[(size_t)b] = w * window_size;
        }
    }

    int sptc_max_window = 0;
    for (int w = 0; w < num_windows; ++w) {
        int n = sptc_window_offset_[w + 1] - sptc_window_offset_[w];
        if (n > sptc_max_window) sptc_max_window = n;
    }
    const int sptc_base_chunk = no_chunk ? (1 << 30)
                              : (fixed_chunk > 0 ? fixed_chunk
                                                 : compute_chunk_size(num_windows, num_sptc_blocks,
                                                                      sptc_max_window));
    const int sptc_straggler_chunk = no_chunk ? 64
                                   : (fixed_chunk > 0 ? fixed_chunk
                                                      : (sptc_base_chunk > 64 ? sptc_base_chunk : 64));
    std::vector<int> sptc_chunk_win, sptc_chunk_beg, sptc_chunk_end;
    std::vector<uint8_t> sptc_chunk_atomic;
    sptc_chunk_win.reserve(num_windows + num_sptc_blocks / sptc_base_chunk + 16);
    sptc_chunk_beg.reserve(sptc_chunk_win.capacity());
    sptc_chunk_end.reserve(sptc_chunk_win.capacity());
    sptc_chunk_atomic.reserve(sptc_chunk_win.capacity());
    for (int w = 0; w < num_windows; ++w) {
        int b0 = sptc_window_offset_[w];
        int b1 = sptc_window_offset_[w + 1];
        if (b0 >= b1) continue;
        bool has_tc = (tc_window_offset_[w + 1] > tc_window_offset_[w]);
        int n_blocks = b1 - b0;
        if (n_blocks <= sptc_base_chunk) {
            // Tier 1：整窗单 CTA
            sptc_chunk_win.push_back(w);
            sptc_chunk_beg.push_back(b0);
            sptc_chunk_end.push_back(b1);
            sptc_chunk_atomic.push_back(has_tc ? 1 : 0);
            continue;
        }
        // Tier 2: straggler 大窗分片
        int chunk = sptc_straggler_chunk;
        int n_chunks = (n_blocks + chunk - 1) / chunk;
        for (int b = b0; b < b1; b += chunk) {
            sptc_chunk_win.push_back(w);
            sptc_chunk_beg.push_back(b);
            sptc_chunk_end.push_back(b + chunk < b1 ? b + chunk : b1);
            sptc_chunk_atomic.push_back((n_chunks > 1 || has_tc) ? 1 : 0);
        }
    }
    num_sptc_chunks = (int)sptc_chunk_win.size();
    printf("[TR.cpp] SPTC chunks: %d (blocks: %d, CHUNK=%d, straggler=%d)\n",
           num_sptc_chunks, num_sptc_blocks, sptc_base_chunk, sptc_straggler_chunk);

    // ====================================================================
    // 2.8 [消融 no-F] 写回全部 atomicAdd: 把所有 chunk 的 atomic 标记强制置 1
    //     (单写者 float4 快写被禁用, 等效于旧版无条件原子写回)
    // ====================================================================
    if (ablation == 6) {
        for (auto& f : tc_chunk_atomic)   f = 1;
        for (auto& f : sptc_chunk_atomic) f = 1;
    }

    // ====================================================================
    // 2.9 [方案A] 高并行矩阵的串行双路 + TC 直写
    //
    // 背景: 当两路的 CTA 数都远大于 SM 数时, 任一 kernel 单独就能占满 GPU,
    //   双流并发实际没有重叠 (nsys 实测 PR02R: SPTC 启动时刻 = TC 结束时刻),
    //   但两路仍然都走 atomicAdd 写回, 白白付出 CAS 竞争代价.
    // 做法: 两路都足够大时改为"TC 先直写 → SPTC 后 atomic 累加"的串行执行.
    //   TC 的 Tier1 chunk (该窗口仅此 1 个 TC chunk) 可安全直写, 因为
    //   SPTC 在其之后才开始, 且 output 已清零.
    //   窗口内多 TC chunk (Tier2) 仍需 atomic, 属路内冲突.
    // 判据:
    //   a) 两路 chunk 数均 >= 32×SM —— 即每 SM ≥32 个 chunk (约 4 个满波),
    //      每路单独就能占满 GPU, 无空档可填, 并发收益为零 (实测 8×SM 时
    //      enron/rajat23 仍有可观重叠, 串行反慢 20-40%)
    //   b) 两路最大窗口块数均 <= 512 —— 无超大拖尾窗口.
    //      若存在巨窗 (如 mip1 TC 单窗 8299 块), 并发时另一路可填充其尾部空隙,
    //      串行反而丢失这部分重叠 (实测 mip1 串行慢 12.6%), 故不串行.
    // ====================================================================
    serial_route = false;
    {
        int sm_count = 82;  // 回退值: RTX 3090
        int dev = 0;
        cudaDeviceProp prop;
        if (cudaGetDevice(&dev) == cudaSuccess &&
            cudaGetDeviceProperties(&prop, dev) == cudaSuccess) {
            sm_count = prop.multiProcessorCount;
        }
        const int serial_min_chunks = 32 * sm_count;
        const int serial_max_window = 512;
        serial_route = (mode == 0) && (ablation != 10) &&   // 10 = no-AB(已串行)
                       (num_tc_chunks >= serial_min_chunks) &&
                       (num_sptc_chunks >= serial_min_chunks) &&
                       (tc_max_window <= serial_max_window) &&
                       (sptc_max_window <= serial_max_window);
        if (serial_route) {
            std::vector<int> tc_chunks_per_win(num_windows, 0);
            for (int w : tc_chunk_win) tc_chunks_per_win[w]++;
            for (size_t c = 0; c < tc_chunk_win.size(); ++c) {
                if (tc_chunks_per_win[tc_chunk_win[c]] == 1)
                    tc_chunk_atomic[c] = 0;   // 该窗口唯一 TC 写者 → 直写
            }
        }
        printf("[TR.cpp] 路由模式: %s (TC chunks=%d, SPTC chunks=%d, 阈值=%d)\n",
               serial_route ? "串行(TC直写→SPTC累加)" : "双流并发",
               num_tc_chunks, num_sptc_chunks, serial_min_chunks);
    }

    // ====================================================================
    // 2.10 [融合] 融合窗口表: 所有含 SPTC 或 TC 的窗口 (Tier1 语义).
    //   融合 kernel 每窗一个 CTA, 内部先跑 TC 块再跑 SPTC 块, 单写者直写.
    //   仅当窗口足够小 (两路都 <= 融合阈值) 才进表; 大窗仍走双路 kernel.
    //   TR_FUSED 环境变量在 kernel 侧读取, 这里只准备窗口表.
    // ====================================================================
    std::vector<uint8_t> exclude_w1;       // [num_windows] 1=大窗, 留在双路
    std::vector<int> fused_win;            // [num_fused] 窗口 id (升序)
    std::vector<int> fused_tc_beg, fused_tc_end;   // 该窗 TC 块范围 (无则相等)
    std::vector<int> fused_sp_beg, fused_sp_end;   // 该窗 SPTC 块范围 (无则相等)
    {
        // 融合阈值: 窗口任一路块数超过它 → 留给原双路 (大窗需 straggler 分片)
        // 融合阈值可用环境变量覆盖 (实验用): TR_FUSE_MAX
        int FUSE_MAX_PER_ROUTE = 128;
        {
            const char* fm = std::getenv("TR_FUSE_MAX");
            if (fm && fm[0]) {
                int v = std::atoi(fm);
                if (v > 0) FUSE_MAX_PER_ROUTE = v;
            }
        }
        bool want_fused = (chunk_size >= 0);   // 预留给外部开关; 仅 -1(整窗) 时为 false
        (void)want_fused;
        fused_win.reserve(num_windows);
        fused_tc_beg.reserve(num_windows); fused_tc_end.reserve(num_windows);
        fused_sp_beg.reserve(num_windows); fused_sp_end.reserve(num_windows);
        // exclude_w1[w] = 1 → 该窗口留在双路 chunk 表; 0 → 已进融合表.
        exclude_w1.assign(num_windows, 1);
        for (int w = 0; w < num_windows; ++w) {
            int tb0 = tc_window_offset_[w], tb1 = tc_window_offset_[w + 1];
            int sb0 = sptc_window_offset_[w], sb1 = sptc_window_offset_[w + 1];
            int ntc = tb1 - tb0, nsp = sb1 - sb0;
            // 纯 SPTC 窗不进融合: 本窗没有 TC 写者, 不存在跨路 atomic 可省,
            //   融合只是把 chunk 并行压成每窗单 CTA (实测慢 0.3~0.9%).
            if (ntc == 0) continue;
            if (ntc > FUSE_MAX_PER_ROUTE || nsp > FUSE_MAX_PER_ROUTE) continue;  // 大窗留双路
            fused_win.push_back(w);
            fused_tc_beg.push_back(tb0); fused_tc_end.push_back(tb1);
            fused_sp_beg.push_back(sb0); fused_sp_end.push_back(sb1);
            exclude_w1[w] = 0;
        }
        // 从双路 chunk 表剔除已被融合的窗口 (排除的大窗保留), 避免重复计算.
        //   TR_FUSED 在 kernel 侧读取; 这里只做表过滤, 未启用时不改任何行为.
        {
            const char* fv = std::getenv("TR_FUSED");
            if (fv && fv[0] && fv[0] != '0') {
                auto filter_chunks = [&](std::vector<int>& cw, std::vector<int>& cb,
                                         std::vector<int>& ce, std::vector<uint8_t>& ca) {
                    std::vector<int> nw, nb, ne2;
                    std::vector<uint8_t> na;
                    nw.reserve(cw.size()); nb.reserve(cw.size());
                    ne2.reserve(cw.size()); na.reserve(cw.size());
                    for (size_t i = 0; i < cw.size(); ++i) {
                        if (exclude_w1[cw[i]]) {
                            nw.push_back(cw[i]); nb.push_back(cb[i]);
                            ne2.push_back(ce[i]); na.push_back(ca[i]);
                        }
                    }
                    cw.swap(nw); cb.swap(nb); ce.swap(ne2); ca.swap(na);
                };
                filter_chunks(tc_chunk_win, tc_chunk_beg, tc_chunk_end, tc_chunk_atomic);
                filter_chunks(sptc_chunk_win, sptc_chunk_beg, sptc_chunk_end, sptc_chunk_atomic);
                num_tc_chunks = (int)tc_chunk_win.size();
                num_sptc_chunks = (int)sptc_chunk_win.size();
                printf("[TR.cpp] 融合后双路 chunk: TC %d, SPTC %d (仅大窗)\n",
                       num_tc_chunks, num_sptc_chunks);
            }
        }
        printf("[TR.cpp] 融合窗口: %d / %d (阈值 %d 块/路)\n",
               (int)fused_win.size(), num_windows, FUSE_MAX_PER_ROUTE);
    }

    // ====================================================================
    // 3. 分配 GPU 内存
    // ====================================================================
    cuda_malloc_or_dummy(&d_sptc_chunk_win, (int64_t)num_sptc_chunks);
    cuda_malloc_or_dummy(&d_sptc_chunk_beg, (int64_t)num_sptc_chunks);
    cuda_malloc_or_dummy(&d_sptc_chunk_end, (int64_t)num_sptc_chunks);
    cuda_malloc_or_dummy(&d_sptc_chunk_atomic, (int64_t)num_sptc_chunks);
    cuda_malloc_or_dummy(&d_sptc_block_row, (int64_t)num_sptc_blocks);
    cuda_malloc_or_dummy(&d_sptc_value, sptc_value.numel());
    cuda_malloc_or_dummy(&d_sptc_packed_meta, num_packed_meta);
    cuda_malloc_or_dummy(&d_sptc_col_old, sptc_col_old.numel());
    cuda_malloc_or_dummy(&d_tc_chunk_win, (int64_t)num_tc_chunks);
    cuda_malloc_or_dummy(&d_tc_chunk_beg, (int64_t)num_tc_chunks);
    cuda_malloc_or_dummy(&d_tc_chunk_end, (int64_t)num_tc_chunks);
    cuda_malloc_or_dummy(&d_tc_chunk_atomic, (int64_t)num_tc_chunks);
    // [融合] 融合窗口表
    num_fused = (int)fused_win.size();
    cuda_malloc_or_dummy(&d_fused_win, (int64_t)num_fused);
    cuda_malloc_or_dummy(&d_fused_tc_beg, (int64_t)num_fused);
    cuda_malloc_or_dummy(&d_fused_tc_end, (int64_t)num_fused);
    cuda_malloc_or_dummy(&d_fused_sp_beg, (int64_t)num_fused);
    cuda_malloc_or_dummy(&d_fused_sp_end, (int64_t)num_fused);
    cuda_malloc_or_dummy(&d_tc_col_old, (int64_t)num_tc_blocks * 16);
    cuda_malloc_or_dummy(&d_tc_value_dense, (int64_t)num_tc_blocks * 256);
    // rhs 为 CUDA 时直接读写调用方缓冲, 私有 rhs/output 缓冲不再分配
    // (YeastH 一档就能省下 ~2.4 GB 常驻显存, 多槽缓存下翻倍)
    if (!rhs_matrix.is_cuda()) {
        cuda_malloc_or_dummy(&d_rhs_matrix, (int64_t)kOri * dimN);
        cuda_malloc_or_dummy(&d_output, (int64_t)mOri_padded * dimN);
    }

    // ====================================================================
    // 4. 拷贝数据到 GPU
    // ====================================================================
    auto h2d_start = std::chrono::high_resolution_clock::now();
    cuda_copy_h2d_if_needed(d_sptc_chunk_win, sptc_chunk_win.data(), (int64_t)num_sptc_chunks);
    cuda_copy_h2d_if_needed(d_sptc_chunk_beg, sptc_chunk_beg.data(), (int64_t)num_sptc_chunks);
    cuda_copy_h2d_if_needed(d_sptc_chunk_end, sptc_chunk_end.data(), (int64_t)num_sptc_chunks);
    cuda_copy_h2d_if_needed(d_sptc_chunk_atomic, sptc_chunk_atomic.data(), (int64_t)num_sptc_chunks);
    cuda_copy_h2d_if_needed(d_sptc_block_row, sptc_block_row.data(), (int64_t)num_sptc_blocks);
    cuda_copy_h2d_if_needed(d_sptc_value, sptc_value_, sptc_value.numel());
    cuda_copy_h2d_if_needed(d_sptc_col_old, sptc_col_old_clean.data(), sptc_col_old.numel());
    cuda_copy_h2d_if_needed(d_tc_chunk_win, tc_chunk_win.data(), (int64_t)num_tc_chunks);
    cuda_copy_h2d_if_needed(d_tc_chunk_beg, tc_chunk_beg.data(), (int64_t)num_tc_chunks);
    cuda_copy_h2d_if_needed(d_tc_chunk_end, tc_chunk_end.data(), (int64_t)num_tc_chunks);
    cuda_copy_h2d_if_needed(d_tc_chunk_atomic, tc_chunk_atomic.data(), (int64_t)num_tc_chunks);
    cuda_copy_h2d_if_needed(d_fused_win, fused_win.data(), (int64_t)num_fused);
    cuda_copy_h2d_if_needed(d_fused_tc_beg, fused_tc_beg.data(), (int64_t)num_fused);
    cuda_copy_h2d_if_needed(d_fused_tc_end, fused_tc_end.data(), (int64_t)num_fused);
    cuda_copy_h2d_if_needed(d_fused_sp_beg, fused_sp_beg.data(), (int64_t)num_fused);
    cuda_copy_h2d_if_needed(d_fused_sp_end, fused_sp_end.data(), (int64_t)num_fused);
    cuda_copy_h2d_if_needed(d_tc_col_old, tc_col_old_pad_vec.data(), (int64_t)num_tc_blocks * 16);
    cuda_copy_h2d_if_needed(d_tc_value_dense, tc_value_dense_vec.data(), (int64_t)num_tc_blocks * 256);
    // rhs 在共用段统一刷新 (见下方), 此处不拷贝

    // 压缩后的元数据 (需要从 vector 拷贝)
    if (num_packed_meta > 0) {
        checkCuda(cudaMemcpy(d_sptc_packed_meta, sptc_packed_meta_vec.data(),
                   num_packed_meta * sizeof(uint32_t), cudaMemcpyHostToDevice));
    }
    auto h2d_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> h2d_elapsed = h2d_end - h2d_start;
    printf("[TR.cpp] H2D copy: %.4f ms\n", h2d_elapsed.count() * 1000.0);

    // 把本次准备结果登记为缓存, 供后续同 packed 调用复用
    {
        TRPreparedState st;
        st.valid = true;
        st.k_meta = prep_key;
        st.k_sw   = (const void*)sptc_window_offset.data_ptr<int>();
        st.k_tcw  = (const void*)tc_window_offset.data_ptr<int>();
        st.k_tcc  = (const void*)tc_col_old.data_ptr<int>();
        st.k_sw_size = (int64_t)sptc_window_offset.numel();
        st.k_dimN = dimN; st.k_mOri = mOri; st.k_kOri = kOri;
        st.k_mode = mode; st.k_abl = ablation;
        st.k_chunk = chunk_size; st.k_dt = dense_threshold;
        st.d_sptc_chunk_win = d_sptc_chunk_win;
        st.d_sptc_chunk_beg = d_sptc_chunk_beg;
        st.d_sptc_chunk_end = d_sptc_chunk_end;
        st.d_sptc_chunk_atomic = d_sptc_chunk_atomic;
        st.d_sptc_block_row = d_sptc_block_row;
        st.d_sptc_value = d_sptc_value;
        st.d_sptc_packed_meta = d_sptc_packed_meta;
        st.d_sptc_col_old = d_sptc_col_old;
        st.d_tc_chunk_win = d_tc_chunk_win;
        st.d_tc_chunk_beg = d_tc_chunk_beg;
        st.d_tc_chunk_end = d_tc_chunk_end;
        st.d_tc_chunk_atomic = d_tc_chunk_atomic;
        st.d_tc_col_old = d_tc_col_old;
        st.d_tc_value_dense = d_tc_value_dense;
        st.d_fused_win = d_fused_win;
        st.d_fused_tc_beg = d_fused_tc_beg;
        st.d_fused_tc_end = d_fused_tc_end;
        st.d_fused_sp_beg = d_fused_sp_beg;
        st.d_fused_sp_end = d_fused_sp_end;
        st.d_rhs_matrix = d_rhs_matrix;
        st.d_output = d_output;
        st.mOri_padded = mOri_padded;
        st.num_sptc_chunks = num_sptc_chunks;
        st.num_tc_chunks = num_tc_chunks;
        st.num_fused = num_fused;
        st.serial_route = serial_route;
        tr_prep_store(prep_key, st);
    }
    }  // end of prepare (prep_hit == false)

    // --------------------------------------------------------------------
    // 4.5 零拷贝 I/O
    //   rhs 与输出都可能是常驻显存的调用方 tensor. 此前每次调用都把 rhs
    //   拷进私有 d_rhs_matrix (K*N half), 再把 d_output 拷回输出张量
    //   (M*N float) —— 在 N=128 的大图上等于每次多付两倍矩阵流量的 D2D,
    //   量级远超 kernel 本身, 且与其它 baseline 的调用边界不一致.
    //   rhs 为 CUDA 时直接让 kernel 读写调用方缓冲; 仅 CPU 输入才走私有缓冲.
    // --------------------------------------------------------------------
    auto out_opts = torch::TensorOptions().dtype(torch::kFloat32);
    if (rhs_matrix.is_cuda()) out_opts = out_opts.device(rhs_matrix.device());
    auto output_matrix = torch::empty({mOri_padded, dimN}, out_opts);
    float* output_ptr = output_matrix.data_ptr<float>();

    const bool zero_copy = rhs_matrix.is_cuda();
    if (!zero_copy) {
        // 缓存可能来自一次 CUDA 调用 (当时未分配私有缓冲), 这里补齐
        if (d_rhs_matrix == nullptr)
            cuda_malloc_or_dummy(&d_rhs_matrix, (int64_t)kOri * dimN);
        if (d_output == nullptr)
            cuda_malloc_or_dummy(&d_output, (int64_t)mOri_padded * dimN);
    }
    half*  rhs_use = zero_copy ? const_cast<half*>(rhs_matrix_) : d_rhs_matrix;
    float* out_use = zero_copy ? output_ptr : d_output;

    if (!zero_copy && (int64_t)kOri * dimN > 0) {
        checkCuda(cudaMemcpy(d_rhs_matrix, rhs_matrix_,
                             (int64_t)kOri * dimN * sizeof(half),
                             cudaMemcpyHostToDevice));
    }


    // ====================================================================
    // 5. 启动 CUDA 内核
    // ====================================================================
    float spmm_ms_avg = tr_spmm_forward(
        d_sptc_chunk_win,
        d_sptc_chunk_beg,
        d_sptc_chunk_end,
        d_sptc_chunk_atomic,
        d_sptc_block_row,
        d_sptc_value,
        d_sptc_packed_meta,
        d_sptc_col_old,
        d_tc_chunk_win,
        d_tc_chunk_beg,
        d_tc_chunk_end,
        d_tc_chunk_atomic,
        d_tc_value_dense,
        d_tc_col_old,
        rhs_use,
        out_use,
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
        warmup,
        ablation,
        chunk_size,
        serial_route ? 1 : 0,
        num_fused,
        d_fused_win, d_fused_tc_beg, d_fused_tc_end, d_fused_sp_beg, d_fused_sp_end);

    // ====================================================================
    // 6. 结果回传
    // ====================================================================
    // 零拷贝路径: kernel 已直接写入 output_matrix, 无需再搬一次.
    // CPU 输入路径: 仍需 D2H.
    if (!zero_copy) {
        checkCuda(cudaMemcpy(output_ptr, d_output,
                   (int64_t)mOri_padded * dimN * sizeof(float),
                   cudaMemcpyDeviceToHost));
    }

    // ====================================================================
    // 7. GPU 内存归缓存所有 (由下一次 miss / 进程退出统一释放)
    // ====================================================================
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
          默认双 CUDA Stream 并行执行 (高并行矩阵走方案 A: TC 直写 → SPTC 累加):
            - TC  kernel: 稠密 Tensor Core GEMM (mma.sync.aligned.m16n8k8)
            - SPTC kernel: 结构化稀疏 Tensor Core GEMM (mma.sp.sync.aligned.m16n8k16)

        接口:
          forward(sptc_window_offset, sptc_value, sptc_metadata,
                  sptc_col_old, tc_window_offset, tc_offset,
                  tc_local_id, tc_col_old, tc_value,
                  rhs_matrix, window_size, dimN, mOri, kOri,
                  num_windows, num_sptc_blocks, num_tc_blocks,
                  dense_threshold, epoches,
                  mode=0, warmup=50, ablation=0, chunk_size=0)
              -> [output_matrix, elapsed_ms]
    )pbdoc";

    m.def("clear_prep_cache", &tr_prep_clear_all,
          "释放预打包缓存占用的显存 (换矩阵时调用)");

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
          py::arg("ablation") = 0,
          py::arg("chunk_size") = 0,
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
    dense_threshold:    TC block 的残余列宽 (K 方向, 默认 16, 上限 16)
    epoches:            Benchmark 迭代次数
    mode:               0=双路全跑, 1=仅 TC, 2=仅 SPTC (非 0 时关闭方案 A 串行路由)
    warmup:             预热迭代次数
    ablation:           消融编号 (0=完整版; 5=no-E 整窗单 CTA; 6=no-F 全 atomic)
    chunk_size:         0=自适应分档, -1=整窗单 CTA, >0=固定 chunk 块数

返回:
    [output_matrix [mOri, dimN] float32, elapsed_ms float32]
          )pbdoc");
}