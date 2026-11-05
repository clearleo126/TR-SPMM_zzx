#include <torch/extension.h>

#include <cuda_fp16.h>

#include <cuda_runtime.h>

#include <assert.h>

#include <algorithm>

#include <cstdint>

#include <vector>

#include <unordered_map>

#include <chrono>


#define CHECK_CPU(x) TORCH_CHECK(!x.is_cuda(), #x " must be a CPU tensor")

// 强制内存连续: 对于非连续 tensor 自动调用 .contiguous() 修复,
// 避免因 view/transpose 等操作导致 cudaMemcpy 读到非连续内存而 segfault
#define ENSURE_CONTIGUOUS(x) \
    if (!(x).is_contiguous()) { \
        (x) = (x).contiguous(); \
    }

#define CHECK_DTYPE(x, expected_dtype) TORCH_CHECK(x.dtype() == expected_dtype, #x " has an unexpected dtype")



inline cudaError_t checkCuda(cudaError_t result) {

    if (result != cudaSuccess) {

        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));

        assert(result == cudaSuccess);

    }

    return result;

}



float spmm_forward_fp16_sptc_online_kernel(

    int* s_column,

    float* s_value,

    int8_t* s_pos,

    int* s_window,

    half* rhs_matrix,

    float* output_matrix,

    int groups,

    int window,

    int dimN,

    int mOri,

    int kOri,

    int epoches);



float spmm_forward_fp16_sptc_mma_online_kernel(

    int* s_column,

    float* s_value,

    int8_t* s_pos,

    int* s_offsets,

    int* tile_window,

    int* tile_group,

    half* rhs_matrix,

    float* output_matrix,

    int num_tiles,

    int window,

    int dimN,

    int mOri,

    int kOri,

    int epoches);

float spmm_forward_fp16_sptc_mma_packed_online_kernel(
    int* s_row_ptr,
    int* s_tile_column,
    uint32_t* s_packed_a,
    uint32_t* s_packed_meta,
    half* rhs_matrix,
    float* output_matrix,
    int num_tiles,
    int num_windows,
    int window,
    int dimN,
    int mOri,
    int kOri,
    int epoches);

// 并行 kernel: TC + CUDA + SPTC (无 Warp Stacking, window=16 native)
float spmm_forward_fp16_tcu_cuda_sptc_mma_parallel_online_kernel(
    int * t_row_offset,
    int * t_blockNew_offset,
    half * t_value,
    int * t_column,
    int* t_window_row,
    int * t_atomic,
    long * t_binary,

    int * c_row_offset,
    int * c_row,
    int * c_atomic,
    int * c_column,
    half * c_value,

    int * c_row_offset_short,
    int * c_row_short,
    int * c_atomic_short,
    int * c_column_short,
    half * c_value_short,

    int* s_column,
    float* s_value,
    int8_t* s_pos,
    int* s_offsets,
    int* tile_window,
    int* tile_group,
    int* s_tile_column,
    uint32_t* s_packed_a,
    uint32_t* s_packed_meta,
    int* s_tile_window,

    int* s_window_tile_offset,   // v2 kernel: 窗口→tile范围 [num_windows+1]

    half * rhs_matrix,
    float * output_matrix,
    float * sptc_output_matrix,

    const int parts_t,
    const int parts_c,
    const int partsize_c,
    const int parts_c_short,
    const int num_tiles,
    const int num_windows,        // v2 kernel: 窗口总数
    const int window,
    const int dimN,
    const int mOri,
    const int kOri,
    int epoches,
    float* tc_ms_avg_out,
    float* cuda_long_ms_avg_out,
    float* cuda_short_ms_avg_out,
    float* sptc_ms_avg_out);



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



std::vector<torch::Tensor> spmm_forward_fp16_sptc_online(

    torch::Tensor s_column,

    torch::Tensor s_value,

    torch::Tensor s_pos,

    torch::Tensor s_window,

    torch::Tensor rhs_matrix,

    int window,

    const int dimN,

    const int mOri,

    const int kOri,

    int epoches)

{

    CHECK_CPU(s_column);

    CHECK_CPU(s_value);

    CHECK_CPU(s_pos);

    CHECK_CPU(s_window);

    CHECK_CPU(rhs_matrix);

    ENSURE_CONTIGUOUS(s_column);

    ENSURE_CONTIGUOUS(s_value);

    ENSURE_CONTIGUOUS(s_pos);

    ENSURE_CONTIGUOUS(s_window);

    ENSURE_CONTIGUOUS(rhs_matrix);

    CHECK_DTYPE(s_column, torch::kInt32);

    CHECK_DTYPE(s_value, torch::kFloat32);

    CHECK_DTYPE(s_pos, torch::kInt8);

    CHECK_DTYPE(s_window, torch::kInt32);

    CHECK_DTYPE(rhs_matrix, torch::kFloat16);

    TORCH_CHECK(window > 0, "window must be positive");

    TORCH_CHECK(dimN > 0, "dimN must be positive");

    TORCH_CHECK(mOri > 0, "mOri must be positive");

    TORCH_CHECK(kOri > 0, "kOri must be positive");

    TORCH_CHECK(epoches > 0, "epoches must be positive");

    TORCH_CHECK(s_column.dim() == 2 && s_column.size(1) == 4, "s_column must have shape [groups, 4]");

    TORCH_CHECK(s_value.dim() == 3 && s_value.size(1) == window && s_value.size(2) == 2,

                "s_value must have shape [groups, window, 2]");

    TORCH_CHECK(s_pos.dim() == 3 && s_pos.size(1) == window && s_pos.size(2) == 2,

                "s_pos must have shape [groups, window, 2]");

    TORCH_CHECK(s_window.dim() == 1, "s_window must be 1-D");

    TORCH_CHECK(rhs_matrix.numel() == (int64_t)kOri * dimN, "rhs_matrix must have kOri * dimN elements");



    int groups = s_window.size(0);

    TORCH_CHECK(s_column.size(0) == groups, "s_column group count mismatch");

    TORCH_CHECK(s_value.size(0) == groups, "s_value group count mismatch");

    TORCH_CHECK(s_pos.size(0) == groups, "s_pos group count mismatch");



    auto output_matrix = torch::zeros({mOri, dimN}, torch::kFloat32).to(torch::kCPU);



    int* s_column_ = s_column.data_ptr<int>();

    float* s_value_ = s_value.data_ptr<float>();

    int8_t* s_pos_ = s_pos.data_ptr<int8_t>();

    int* s_window_ = s_window.data_ptr<int>();

    half* rhs_matrix_ = reinterpret_cast<half*>(rhs_matrix.data_ptr<at::Half>());

    float* output_matrix_ = output_matrix.data_ptr<float>();



    int* d_s_column;

    float* d_s_value;

    int8_t* d_s_pos;

    int* d_s_window;

    half* d_rhs_matrix;

    float* d_output_matrix;



    cuda_malloc_or_dummy(&d_s_column, s_column.numel());

    cuda_malloc_or_dummy(&d_s_value, s_value.numel());

    cuda_malloc_or_dummy(&d_s_pos, s_pos.numel());

    cuda_malloc_or_dummy(&d_s_window, s_window.numel());

    cuda_malloc_or_dummy(&d_rhs_matrix, (int64_t)kOri * dimN);

    cuda_malloc_or_dummy(&d_output_matrix, (int64_t)mOri * dimN);



    cuda_copy_h2d_if_needed(d_s_column, s_column_, s_column.numel());

    cuda_copy_h2d_if_needed(d_s_value, s_value_, s_value.numel());

    cuda_copy_h2d_if_needed(d_s_pos, s_pos_, s_pos.numel());

    cuda_copy_h2d_if_needed(d_s_window, s_window_, s_window.numel());

    // rhs 支持 CPU/GPU 两种来源
    if ((int64_t)kOri * dimN > 0) {
        checkCuda(cudaMemcpy(d_rhs_matrix, rhs_matrix_,
                             (int64_t)kOri * dimN * sizeof(half),
                             rhs_matrix.is_cuda() ? cudaMemcpyDeviceToDevice
                                                  : cudaMemcpyHostToDevice));
    }



    float spmm_ms_avg = spmm_forward_fp16_sptc_online_kernel(

        d_s_column,

        d_s_value,

        d_s_pos,

        d_s_window,

        d_rhs_matrix,

        d_output_matrix,

        groups,

        window,

        dimN,

        mOri,

        kOri,

        epoches);



    checkCuda(cudaMemcpy(output_matrix_, d_output_matrix, (int64_t)mOri * dimN * sizeof(float), cudaMemcpyDeviceToHost));



    cudaFree(d_s_column);

    cudaFree(d_s_value);

    cudaFree(d_s_pos);

    cudaFree(d_s_window);

    cudaFree(d_rhs_matrix);

    cudaFree(d_output_matrix);

    cudaDeviceSynchronize();



    return {output_matrix, torch::tensor(spmm_ms_avg)};

}



std::vector<torch::Tensor> spmm_forward_fp16_sptc_mma_online(

    torch::Tensor s_column,

    torch::Tensor s_value,

    torch::Tensor s_pos,

    torch::Tensor s_offsets,

    torch::Tensor rhs_matrix,

    int window,

    const int dimN,

    const int mOri,

    const int kOri,

    int epoches)

{

    CHECK_CPU(s_column);

    CHECK_CPU(s_value);

    CHECK_CPU(s_pos);

    CHECK_CPU(s_offsets);

    CHECK_CPU(rhs_matrix);

    ENSURE_CONTIGUOUS(s_column);

    ENSURE_CONTIGUOUS(s_value);

    ENSURE_CONTIGUOUS(s_pos);

    ENSURE_CONTIGUOUS(s_offsets);

    ENSURE_CONTIGUOUS(rhs_matrix);

    CHECK_DTYPE(s_column, torch::kInt32);

    CHECK_DTYPE(s_value, torch::kFloat32);

    CHECK_DTYPE(s_pos, torch::kInt8);

    CHECK_DTYPE(s_offsets, torch::kInt32);

    CHECK_DTYPE(rhs_matrix, torch::kFloat16);

    TORCH_CHECK(window > 0 && window <= 16, "mma path requires 0 < window <= 16");

    TORCH_CHECK(dimN > 0, "dimN must be positive");

    TORCH_CHECK(mOri > 0, "mOri must be positive");

    TORCH_CHECK(kOri > 0, "kOri must be positive");

    TORCH_CHECK(epoches > 0, "epoches must be positive");

    TORCH_CHECK(s_column.dim() == 2 && s_column.size(1) == 4, "s_column must have shape [groups, 4]");

    TORCH_CHECK(s_value.dim() == 3 && s_value.size(1) == window && s_value.size(2) == 2,

                "s_value must have shape [groups, window, 2]");

    TORCH_CHECK(s_pos.dim() == 3 && s_pos.size(1) == window && s_pos.size(2) == 2,

                "s_pos must have shape [groups, window, 2]");

    TORCH_CHECK(s_offsets.dim() == 1 && s_offsets.size(0) >= 1, "s_offsets must be 1-D with at least one item");

    TORCH_CHECK(rhs_matrix.numel() == (int64_t)kOri * dimN, "rhs_matrix must have kOri * dimN elements");



    int groups = s_column.size(0);

    TORCH_CHECK(s_value.size(0) == groups, "s_value group count mismatch");

    TORCH_CHECK(s_pos.size(0) == groups, "s_pos group count mismatch");

    int* s_offsets_cpu = s_offsets.data_ptr<int>();

    int num_windows = s_offsets.size(0) - 1;

    TORCH_CHECK(s_offsets_cpu[0] == 0, "s_offsets must start with 0");

    TORCH_CHECK(s_offsets_cpu[num_windows] == groups, "s_offsets end must equal group count");

    std::vector<int> tile_window_host;

    std::vector<int> tile_group_host;

    tile_window_host.reserve((groups + 3) / 4);

    tile_group_host.reserve((groups + 3) / 4);

    for (int win = 0; win < num_windows; ++win) {

        int group_count = s_offsets_cpu[win + 1] - s_offsets_cpu[win];

        TORCH_CHECK(group_count >= 0, "s_offsets must be non-decreasing");

        for (int group = s_offsets_cpu[win]; group < s_offsets_cpu[win + 1]; group += 4) {

            tile_window_host.push_back(win);

            tile_group_host.push_back(group);

        }

    }

    int num_tiles = (int)tile_window_host.size();



    auto output_matrix = torch::zeros({mOri, dimN}, torch::kFloat32).to(torch::kCPU);



    int* s_column_ = s_column.data_ptr<int>();

    float* s_value_ = s_value.data_ptr<float>();

    int8_t* s_pos_ = s_pos.data_ptr<int8_t>();

    half* rhs_matrix_ = reinterpret_cast<half*>(rhs_matrix.data_ptr<at::Half>());

    float* output_matrix_ = output_matrix.data_ptr<float>();



    int* d_s_column;

    float* d_s_value;

    int8_t* d_s_pos;

    int* d_s_offsets;

    int* d_tile_window;

    int* d_tile_group;

    half* d_rhs_matrix;

    float* d_output_matrix;



    cuda_malloc_or_dummy(&d_s_column, s_column.numel());

    cuda_malloc_or_dummy(&d_s_value, s_value.numel());

    cuda_malloc_or_dummy(&d_s_pos, s_pos.numel());

    cuda_malloc_or_dummy(&d_s_offsets, s_offsets.numel());

    cuda_malloc_or_dummy(&d_tile_window, tile_window_host.size());

    cuda_malloc_or_dummy(&d_tile_group, tile_group_host.size());

    cuda_malloc_or_dummy(&d_rhs_matrix, (int64_t)kOri * dimN);

    cuda_malloc_or_dummy(&d_output_matrix, (int64_t)mOri * dimN);



    cuda_copy_h2d_if_needed(d_s_column, s_column_, s_column.numel());

    cuda_copy_h2d_if_needed(d_s_value, s_value_, s_value.numel());

    cuda_copy_h2d_if_needed(d_s_pos, s_pos_, s_pos.numel());

    cuda_copy_h2d_if_needed(d_s_offsets, s_offsets_cpu, s_offsets.numel());

    cuda_copy_h2d_if_needed(d_tile_window, tile_window_host.data(), tile_window_host.size());

    cuda_copy_h2d_if_needed(d_tile_group, tile_group_host.data(), tile_group_host.size());

    cuda_copy_h2d_if_needed(d_rhs_matrix, rhs_matrix_, (int64_t)kOri * dimN);



    float spmm_ms_avg = spmm_forward_fp16_sptc_mma_online_kernel(

        d_s_column,

        d_s_value,

        d_s_pos,

        d_s_offsets,

        d_tile_window,

        d_tile_group,

        d_rhs_matrix,

        d_output_matrix,

        num_tiles,

        window,

        dimN,

        mOri,

        kOri,

        epoches);



    checkCuda(cudaMemcpy(output_matrix_, d_output_matrix, (int64_t)mOri * dimN * sizeof(float), cudaMemcpyDeviceToHost));



    cudaFree(d_s_column);

    cudaFree(d_s_value);

    cudaFree(d_s_pos);

    cudaFree(d_s_offsets);

    cudaFree(d_tile_window);

    cudaFree(d_tile_group);

    cudaFree(d_rhs_matrix);

    cudaFree(d_output_matrix);

    cudaDeviceSynchronize();



    return {output_matrix, torch::tensor(spmm_ms_avg)};

}

std::vector<torch::Tensor> spmm_forward_fp16_sptc_mma_packed_online(
    torch::Tensor s_tile_column,
    torch::Tensor s_packed_a,
    torch::Tensor s_packed_meta,
    torch::Tensor s_window_tile_offset,
    torch::Tensor rhs_matrix,
    int window,
    const int dimN,
    const int mOri,
    const int kOri,
    int epoches)
{
    CHECK_CPU(s_tile_column);
    CHECK_CPU(s_packed_a);
    CHECK_CPU(s_packed_meta);
    CHECK_CPU(s_window_tile_offset);
    CHECK_CPU(rhs_matrix);
    ENSURE_CONTIGUOUS(s_tile_column);
    ENSURE_CONTIGUOUS(s_packed_a);
    ENSURE_CONTIGUOUS(s_packed_meta);
    ENSURE_CONTIGUOUS(s_window_tile_offset);
    ENSURE_CONTIGUOUS(rhs_matrix);
    CHECK_DTYPE(s_tile_column, torch::kInt32);
    CHECK_DTYPE(s_packed_a, torch::kInt32);
    CHECK_DTYPE(s_packed_meta, torch::kInt32);
    CHECK_DTYPE(s_window_tile_offset, torch::kInt32);
    CHECK_DTYPE(rhs_matrix, torch::kFloat16);
    TORCH_CHECK(window > 0 && window <= 16, "packed mma path requires 0 < window <= 16");
    TORCH_CHECK(dimN > 0, "dimN must be positive");
    TORCH_CHECK(mOri > 0, "mOri must be positive");
    TORCH_CHECK(kOri > 0, "kOri must be positive");
    TORCH_CHECK(epoches > 0, "epoches must be positive");
    TORCH_CHECK(s_tile_column.dim() == 2 && s_tile_column.size(1) == 16,
                "s_tile_column must have shape [tiles, 16]");
    TORCH_CHECK(s_packed_a.dim() == 4 && s_packed_a.size(1) == 8 &&
                s_packed_a.size(2) == 4 && s_packed_a.size(3) == 2,
                "s_packed_a must have shape [tiles, 8, 4, 2]");
    TORCH_CHECK(s_packed_meta.dim() == 2 && s_packed_meta.size(1) == 8,
                "s_packed_meta must have shape [tiles, 8]");
    TORCH_CHECK(s_window_tile_offset.dim() == 1 && s_window_tile_offset.size(0) >= 2,
                "s_window_tile_offset must be 1-D with at least 2 elements");
    TORCH_CHECK(rhs_matrix.numel() == (int64_t)kOri * dimN, "rhs_matrix must have kOri * dimN elements");

    int num_tiles = s_tile_column.size(0);
    int num_windows = s_window_tile_offset.size(0) - 1;
    TORCH_CHECK(s_packed_a.size(0) == num_tiles, "s_packed_a tile count mismatch");
    TORCH_CHECK(s_packed_meta.size(0) == num_tiles, "s_packed_meta tile count mismatch");

    auto output_matrix = torch::zeros({mOri, dimN}, torch::kFloat32).to(torch::kCPU);

    int* s_tile_column_ = s_tile_column.data_ptr<int>();
    uint32_t* s_packed_a_ = reinterpret_cast<uint32_t*>(s_packed_a.data_ptr<int>());
    uint32_t* s_packed_meta_ = reinterpret_cast<uint32_t*>(s_packed_meta.data_ptr<int>());
    int* s_window_tile_offset_ = s_window_tile_offset.data_ptr<int>();
    half* rhs_matrix_ = reinterpret_cast<half*>(rhs_matrix.data_ptr<at::Half>());
    float* output_matrix_ = output_matrix.data_ptr<float>();

    int* d_s_tile_column;
    uint32_t* d_s_packed_a;
    uint32_t* d_s_packed_meta;
    int* d_s_window_tile_offset;
    half* d_rhs_matrix;
    float* d_output_matrix;

    cuda_malloc_or_dummy(&d_s_tile_column, s_tile_column.numel());
    cuda_malloc_or_dummy(&d_s_packed_a, s_packed_a.numel());
    cuda_malloc_or_dummy(&d_s_packed_meta, s_packed_meta.numel());
    cuda_malloc_or_dummy(&d_s_window_tile_offset, s_window_tile_offset.numel());
    cuda_malloc_or_dummy(&d_rhs_matrix, (int64_t)kOri * dimN);
    cuda_malloc_or_dummy(&d_output_matrix, (int64_t)mOri * dimN);

    cuda_copy_h2d_if_needed(d_s_tile_column, s_tile_column_, s_tile_column.numel());
    cuda_copy_h2d_if_needed(d_s_packed_a, s_packed_a_, s_packed_a.numel());
    cuda_copy_h2d_if_needed(d_s_packed_meta, s_packed_meta_, s_packed_meta.numel());
    cuda_copy_h2d_if_needed(d_s_window_tile_offset, s_window_tile_offset_, s_window_tile_offset.numel());
    cuda_copy_h2d_if_needed(d_rhs_matrix, rhs_matrix_, (int64_t)kOri * dimN);

    float spmm_ms_avg = spmm_forward_fp16_sptc_mma_packed_online_kernel(
        d_s_window_tile_offset,
        d_s_tile_column,
        d_s_packed_a,
        d_s_packed_meta,
        d_rhs_matrix,
        d_output_matrix,
        num_tiles,
        num_windows,
        window,
        dimN,
        mOri,
        kOri,
        epoches);

    checkCuda(cudaMemcpy(output_matrix_, d_output_matrix, (int64_t)mOri * dimN * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_s_tile_column);
    cudaFree(d_s_packed_a);
    cudaFree(d_s_packed_meta);
    cudaFree(d_s_window_tile_offset);
    cudaFree(d_rhs_matrix);
    cudaFree(d_output_matrix);
    cudaDeviceSynchronize();

    return {output_matrix, torch::tensor(spmm_ms_avg)};
}

// ============================================================================
// Libra 预处理产物的 device 缓存
//
// 官方 online 管线每次调用都重新 cudaMalloc ~30 个缓冲并全量 H2D. 在端到端
// 循环里这部分墙钟 (Yeast 上 ~120 ms/次, 而 kernel 只有 2 ms) 会完全压过
// 计算本身, 且与 TR / MP 的调用边界不一致. 这里按 t_row_offset 的 data_ptr
// 缓存, 命中时只刷新 rhs. A 与 A^T 各占一槽, 由 unordered_map 区分.
// ============================================================================
struct LibraPrepState {
    bool valid = false;
    int64_t k_dimN = 0, k_mOri = 0, k_kOri = 0, k_window = 0;
    int64_t n_row_offset = 0, n_column = 0, n_t_value = 0, n_s_column = 0;

    int *d_t_row_offset = nullptr, *d_t_blockNew_offset = nullptr, *d_t_column = nullptr;
    int *d_t_window_row = nullptr, *d_t_atomic = nullptr;
    long *d_t_binary = nullptr;
    half *d_t_value = nullptr;
    int *d_c_row_offsets = nullptr, *d_c_row = nullptr;
    int *d_c_atomic = nullptr, *d_c_column = nullptr;
    int *d_c_row_offsets_short = nullptr, *d_c_row_short = nullptr;
    int *d_c_atomic_short = nullptr, *d_c_column_short = nullptr;
    half *d_c_value = nullptr, *d_c_value_short = nullptr;
    int *d_s_column = nullptr, *d_s_offsets = nullptr;
    int *d_tile_window = nullptr, *d_tile_group = nullptr;
    int *d_s_tile_window = nullptr, *d_s_tile_column = nullptr;
    int *d_s_window_tile_offset = nullptr;
    uint32_t *d_s_packed_a = nullptr, *d_s_packed_meta = nullptr;
    float *d_s_value = nullptr;
    int8_t *d_s_pos = nullptr;
    half *d_rhs_matrix = nullptr;
    float *d_output_matrix = nullptr, *d_sptc_output_matrix = nullptr;
};

static std::unordered_map<const void*, LibraPrepState> g_libra_prep;
static const size_t kLibraMaxSlots = 8;

static void libra_prep_free(LibraPrepState& s) {
    cudaFree(s.d_t_row_offset);
    cudaFree(s.d_t_blockNew_offset);
    cudaFree(s.d_t_column);
    cudaFree(s.d_t_window_row);
    cudaFree(s.d_t_atomic);
    cudaFree(s.d_t_binary);
    cudaFree(s.d_t_value);
    cudaFree(s.d_c_row_offsets);
    cudaFree(s.d_c_row);
    cudaFree(s.d_c_atomic);
    cudaFree(s.d_c_column);
    cudaFree(s.d_c_value);
    cudaFree(s.d_c_row_offsets_short);
    cudaFree(s.d_c_row_short);
    cudaFree(s.d_c_atomic_short);
    cudaFree(s.d_c_column_short);
    cudaFree(s.d_c_value_short);
    cudaFree(s.d_s_column);
    cudaFree(s.d_s_value);
    cudaFree(s.d_s_pos);
    cudaFree(s.d_s_offsets);
    cudaFree(s.d_tile_window);
    cudaFree(s.d_tile_group);
    cudaFree(s.d_s_tile_window);
    cudaFree(s.d_s_tile_column);
    cudaFree(s.d_s_packed_a);
    cudaFree(s.d_s_packed_meta);
    cudaFree(s.d_s_window_tile_offset);
    cudaFree(s.d_rhs_matrix);
    cudaFree(s.d_output_matrix);
    cudaFree(s.d_sptc_output_matrix);
    s = LibraPrepState();
}

static void libra_clear_prep() {
    for (auto& kv : g_libra_prep) libra_prep_free(kv.second);
    g_libra_prep.clear();
}

// 并行入口: TC + CUDA + SPTC (无 Warp Stacking, window=16 native)
std::vector<torch::Tensor> spmm_forward_fp16_tcu_cuda_sptc_mma_parallel_online(
    torch::Tensor t_row_offset,
    torch::Tensor t_blockNew_offset,
    torch::Tensor t_column,
    torch::Tensor t_value,
    torch::Tensor t_window_row,
    torch::Tensor t_atomic,
    torch::Tensor t_binary,

    torch::Tensor c_row_offsets,
    torch::Tensor c_row,
    torch::Tensor c_atomic,
    torch::Tensor c_column,
    torch::Tensor c_value,

    torch::Tensor c_row_offsets_short,
    torch::Tensor c_row_short,
    torch::Tensor c_atomic_short,
    torch::Tensor c_column_short,
    torch::Tensor c_value_short,

    torch::Tensor s_column,
    torch::Tensor s_value,
    torch::Tensor s_pos,
    torch::Tensor s_offsets,
    torch::Tensor s_tile_window,
    torch::Tensor s_tile_column,
    torch::Tensor s_packed_a,
    torch::Tensor s_packed_meta,
    torch::Tensor s_window_tile_offset,  // v2 kernel
    torch::Tensor rhs_matrix,

    const int parts_t,
    const int parts_c,
    const int partsize_c,
    const int parts_c_short,
    int window,
    const int dimN,
    const int mOri,
    const int kOri,
    int epoches)
{
    CHECK_CPU(t_row_offset);
    CHECK_CPU(t_blockNew_offset);
    CHECK_CPU(t_column);
    CHECK_CPU(t_value);
    CHECK_CPU(t_window_row);
    CHECK_CPU(t_atomic);
    CHECK_CPU(t_binary);
    CHECK_CPU(c_row_offsets);
    CHECK_CPU(c_row);
    CHECK_CPU(c_atomic);
    CHECK_CPU(c_column);
    CHECK_CPU(c_value);
    CHECK_CPU(c_row_offsets_short);
    CHECK_CPU(c_row_short);
    CHECK_CPU(c_atomic_short);
    CHECK_CPU(c_column_short);
    CHECK_CPU(c_value_short);
    CHECK_CPU(s_column);
    CHECK_CPU(s_value);
    CHECK_CPU(s_pos);
    CHECK_CPU(s_offsets);
    CHECK_CPU(s_tile_window);
    CHECK_CPU(s_tile_column);
    CHECK_CPU(s_packed_a);
    CHECK_CPU(s_packed_meta);
    CHECK_CPU(s_window_tile_offset);
    // rhs_matrix 允许 CUDA tensor (GPU 直通)

    ENSURE_CONTIGUOUS(t_row_offset);
    ENSURE_CONTIGUOUS(t_blockNew_offset);
    ENSURE_CONTIGUOUS(t_column);
    ENSURE_CONTIGUOUS(t_value);
    ENSURE_CONTIGUOUS(t_window_row);
    ENSURE_CONTIGUOUS(t_atomic);
    ENSURE_CONTIGUOUS(t_binary);
    ENSURE_CONTIGUOUS(c_row_offsets);
    ENSURE_CONTIGUOUS(c_row);
    ENSURE_CONTIGUOUS(c_atomic);
    ENSURE_CONTIGUOUS(c_column);
    ENSURE_CONTIGUOUS(c_value);
    ENSURE_CONTIGUOUS(c_row_offsets_short);
    ENSURE_CONTIGUOUS(c_row_short);
    ENSURE_CONTIGUOUS(c_atomic_short);
    ENSURE_CONTIGUOUS(c_column_short);
    ENSURE_CONTIGUOUS(c_value_short);
    ENSURE_CONTIGUOUS(s_column);
    ENSURE_CONTIGUOUS(s_value);
    ENSURE_CONTIGUOUS(s_pos);
    ENSURE_CONTIGUOUS(s_offsets);
    ENSURE_CONTIGUOUS(s_tile_window);
    ENSURE_CONTIGUOUS(s_tile_column);
    ENSURE_CONTIGUOUS(s_packed_a);
    ENSURE_CONTIGUOUS(s_packed_meta);
    ENSURE_CONTIGUOUS(s_window_tile_offset);
    ENSURE_CONTIGUOUS(rhs_matrix);

    CHECK_DTYPE(t_row_offset, torch::kInt32);
    CHECK_DTYPE(t_blockNew_offset, torch::kInt32);
    CHECK_DTYPE(t_column, torch::kInt32);
    CHECK_DTYPE(t_value, torch::kFloat16);
    CHECK_DTYPE(t_window_row, torch::kInt32);
    CHECK_DTYPE(t_atomic, torch::kInt32);
    CHECK_DTYPE(t_binary, torch::kInt64);
    CHECK_DTYPE(c_row_offsets, torch::kInt32);
    CHECK_DTYPE(c_row, torch::kInt32);
    CHECK_DTYPE(c_atomic, torch::kInt32);
    CHECK_DTYPE(c_column, torch::kInt32);
    CHECK_DTYPE(c_value, torch::kFloat16);
    CHECK_DTYPE(c_row_offsets_short, torch::kInt32);
    CHECK_DTYPE(c_row_short, torch::kInt32);
    CHECK_DTYPE(c_atomic_short, torch::kInt32);
    CHECK_DTYPE(c_column_short, torch::kInt32);
    CHECK_DTYPE(c_value_short, torch::kFloat16);
    CHECK_DTYPE(s_column, torch::kInt32);
    CHECK_DTYPE(s_value, torch::kFloat32);
    CHECK_DTYPE(s_pos, torch::kInt8);
    CHECK_DTYPE(s_offsets, torch::kInt32);
    CHECK_DTYPE(s_tile_window, torch::kInt32);
    CHECK_DTYPE(s_tile_column, torch::kInt32);
    CHECK_DTYPE(s_packed_a, torch::kInt32);
    CHECK_DTYPE(s_packed_meta, torch::kInt32);
    CHECK_DTYPE(rhs_matrix, torch::kFloat16);

    TORCH_CHECK(window > 0 && window <= 16, "parallel mma path requires 0 < window <= 16");
    TORCH_CHECK(dimN > 0, "dimN must be positive");
    TORCH_CHECK(mOri > 0, "mOri must be positive");
    TORCH_CHECK(kOri > 0, "kOri must be positive");
    TORCH_CHECK(epoches > 0, "epoches must be positive");
    TORCH_CHECK(rhs_matrix.numel() == (int64_t)kOri * dimN, "rhs_matrix must have kOri * dimN elements");
    TORCH_CHECK(s_column.dim() == 2 && s_column.size(1) == 4, "s_column must have shape [groups, 4]");
    TORCH_CHECK(s_value.dim() == 3 && s_value.size(1) == window && s_value.size(2) == 2,
                "s_value must have shape [groups, window, 2]");
    TORCH_CHECK(s_pos.dim() == 3 && s_pos.size(1) == window && s_pos.size(2) == 2,
                "s_pos must have shape [groups, window, 2]");
    TORCH_CHECK(s_offsets.dim() == 1 && s_offsets.size(0) >= 1, "s_offsets must be 1-D with at least one item");
    TORCH_CHECK(s_tile_window.dim() == 1, "s_tile_window must be 1-D");
    TORCH_CHECK(s_tile_column.dim() == 2 && s_tile_column.size(1) == 16,
                "s_tile_column must have shape [tiles, 16]");
    TORCH_CHECK(s_packed_a.dim() == 4 && s_packed_a.size(1) == 8 &&
                s_packed_a.size(2) == 4 && s_packed_a.size(3) == 2,
                "s_packed_a must have shape [tiles, 8, 4, 2]");
    TORCH_CHECK(s_packed_meta.dim() == 2 && s_packed_meta.size(1) == 8,
                "s_packed_meta must have shape [tiles, 8]");

    int groups = s_column.size(0);
    TORCH_CHECK(s_value.size(0) == groups, "s_value group count mismatch");
    TORCH_CHECK(s_pos.size(0) == groups, "s_pos group count mismatch");
    int* s_offsets_cpu = s_offsets.data_ptr<int>();
    int num_windows = s_offsets.size(0) - 1;
    TORCH_CHECK(s_offsets_cpu[0] == 0, "s_offsets must start with 0");
    TORCH_CHECK(s_offsets_cpu[num_windows] == groups, "s_offsets end must equal group count");

    std::vector<int> tile_window_host;
    std::vector<int> tile_group_host;
    tile_window_host.reserve((groups + 3) / 4);
    tile_group_host.reserve((groups + 3) / 4);
    for (int win = 0; win < num_windows; ++win) {
        int group_count = s_offsets_cpu[win + 1] - s_offsets_cpu[win];
        TORCH_CHECK(group_count >= 0, "s_offsets must be non-decreasing");
        for (int group = s_offsets_cpu[win]; group < s_offsets_cpu[win + 1]; group += 4) {
            tile_window_host.push_back(win);
            tile_group_host.push_back(group);
        }
    }
    int num_tiles = (int)tile_window_host.size();
    int packed_num_tiles = s_tile_window.size(0);
    TORCH_CHECK(s_tile_column.size(0) == packed_num_tiles, "s_tile_column tile count mismatch");
    TORCH_CHECK(s_packed_a.size(0) == packed_num_tiles, "s_packed_a tile count mismatch");
    TORCH_CHECK(s_packed_meta.size(0) == packed_num_tiles, "s_packed_meta tile count mismatch");
    TORCH_CHECK(packed_num_tiles == num_tiles,
                "packed tile count must match offsets-derived tile count");

    // 输出分配在与输入相同的 device 上 (输入为 CUDA 时免 D2H 往返)
    auto out_opts = torch::TensorOptions().dtype(torch::kFloat32);
    if (rhs_matrix.is_cuda()) out_opts = out_opts.device(rhs_matrix.device());
    auto _t0 = std::chrono::high_resolution_clock::now();
    auto output_matrix = torch::zeros({mOri, dimN}, out_opts);
    float* output_matrix_ = output_matrix.data_ptr<float>();
    auto _t1 = std::chrono::high_resolution_clock::now();
    const bool libra_timing = []() {
        const char* v = getenv("LIBRA_TIMING");
        return v && v[0] && v[0] != '0';
    }();

    int* t_row_offset_ = t_row_offset.data_ptr<int>();
    int* t_blockNew_offset_ = t_blockNew_offset.data_ptr<int>();
    int* t_column_ = t_column.data_ptr<int>();
    half* t_value_ = reinterpret_cast<half*>(t_value.data_ptr<at::Half>());
    int* t_window_row_ = t_window_row.data_ptr<int>();
    int* t_atomic_ = t_atomic.data_ptr<int>();
    long* t_binary_ = t_binary.data_ptr<long>();

    int* c_row_offsets_ = c_row_offsets.data_ptr<int>();
    int* c_row_ = c_row.data_ptr<int>();
    int* c_atomic_ = c_atomic.data_ptr<int>();
    int* c_column_ = c_column.data_ptr<int>();
    half* c_value_ = reinterpret_cast<half*>(c_value.data_ptr<at::Half>());

    int* c_row_offsets_short_ = c_row_offsets_short.data_ptr<int>();
    int* c_row_short_ = c_row_short.data_ptr<int>();
    int* c_atomic_short_ = c_atomic_short.data_ptr<int>();
    int* c_column_short_ = c_column_short.data_ptr<int>();
    half* c_value_short_ = reinterpret_cast<half*>(c_value_short.data_ptr<at::Half>());

    int* s_column_ = s_column.data_ptr<int>();
    float* s_value_ = s_value.data_ptr<float>();
    int8_t* s_pos_ = s_pos.data_ptr<int8_t>();
    int* s_tile_window_ = s_tile_window.data_ptr<int>();
    int* s_tile_column_ = s_tile_column.data_ptr<int>();
    uint32_t* s_packed_a_ = reinterpret_cast<uint32_t*>(s_packed_a.data_ptr<int>());
    uint32_t* s_packed_meta_ = reinterpret_cast<uint32_t*>(s_packed_meta.data_ptr<int>());
    int* s_window_tile_offset_ = s_window_tile_offset.data_ptr<int>();
    half* rhs_matrix_ = reinterpret_cast<half*>(rhs_matrix.data_ptr<at::Half>());

    // ---- 复用上一次的 device 缓冲 (按 t_row_offset 指针 + 形状校验) ----
    const void* libra_key = (const void*)t_row_offset.data_ptr<int>();
    auto libra_it = g_libra_prep.find(libra_key);
    bool libra_hit = libra_it != g_libra_prep.end() && libra_it->second.valid &&
                     libra_it->second.k_dimN == dimN &&
                     libra_it->second.k_mOri == mOri &&
                     libra_it->second.k_kOri == kOri &&
                     libra_it->second.k_window == window &&
                     libra_it->second.n_row_offset == (int64_t)t_row_offset.numel() &&
                     libra_it->second.n_column == (int64_t)t_column.numel() &&
                     libra_it->second.n_t_value == (int64_t)t_value.numel() &&
                     libra_it->second.n_s_column == (int64_t)s_column.numel();

    if (!libra_hit) {
        if (libra_it != g_libra_prep.end()) {
            libra_prep_free(libra_it->second);
            g_libra_prep.erase(libra_it);
        }
        if (g_libra_prep.size() >= kLibraMaxSlots) libra_clear_prep();

        LibraPrepState st;
        st.valid = true;
        st.k_dimN = dimN; st.k_mOri = mOri; st.k_kOri = kOri; st.k_window = window;
        st.n_row_offset = (int64_t)t_row_offset.numel();
        st.n_column = (int64_t)t_column.numel();
        st.n_t_value = (int64_t)t_value.numel();
        st.n_s_column = (int64_t)s_column.numel();

        cuda_malloc_or_dummy(&st.d_t_row_offset, t_row_offset.numel());
        cuda_malloc_or_dummy(&st.d_t_blockNew_offset, t_blockNew_offset.numel());
        cuda_malloc_or_dummy(&st.d_t_column, t_column.numel());
        cuda_malloc_or_dummy(&st.d_t_value, t_value.numel());
        cuda_malloc_or_dummy(&st.d_t_window_row, t_window_row.numel());
        cuda_malloc_or_dummy(&st.d_t_atomic, t_atomic.numel());
        cuda_malloc_or_dummy(&st.d_t_binary, t_binary.numel());
        cuda_malloc_or_dummy(&st.d_c_row_offsets, c_row_offsets.numel());
        cuda_malloc_or_dummy(&st.d_c_row, c_row.numel());
        cuda_malloc_or_dummy(&st.d_c_atomic, c_atomic.numel());
        cuda_malloc_or_dummy(&st.d_c_column, c_column.numel());
        cuda_malloc_or_dummy(&st.d_c_value, c_value.numel());
        cuda_malloc_or_dummy(&st.d_c_row_offsets_short, c_row_offsets_short.numel());
        cuda_malloc_or_dummy(&st.d_c_row_short, c_row_short.numel());
        cuda_malloc_or_dummy(&st.d_c_atomic_short, c_atomic_short.numel());
        cuda_malloc_or_dummy(&st.d_c_column_short, c_column_short.numel());
        cuda_malloc_or_dummy(&st.d_c_value_short, c_value_short.numel());
        cuda_malloc_or_dummy(&st.d_s_column, s_column.numel());
        cuda_malloc_or_dummy(&st.d_s_value, s_value.numel());
        cuda_malloc_or_dummy(&st.d_s_pos, s_pos.numel());
        cuda_malloc_or_dummy(&st.d_s_offsets, s_offsets.numel());
        cuda_malloc_or_dummy(&st.d_tile_window, tile_window_host.size());
        cuda_malloc_or_dummy(&st.d_tile_group, tile_group_host.size());
        cuda_malloc_or_dummy(&st.d_s_tile_window, s_tile_window.numel());
        cuda_malloc_or_dummy(&st.d_s_tile_column, s_tile_column.numel());
        cuda_malloc_or_dummy(&st.d_s_packed_a, s_packed_a.numel());
        cuda_malloc_or_dummy(&st.d_s_packed_meta, s_packed_meta.numel());
        cuda_malloc_or_dummy(&st.d_s_window_tile_offset, s_window_tile_offset.numel());
        cuda_malloc_or_dummy(&st.d_rhs_matrix, (int64_t)kOri * dimN);
        cuda_malloc_or_dummy(&st.d_sptc_output_matrix, (int64_t)mOri * dimN);

        cuda_copy_h2d_if_needed(st.d_t_row_offset, t_row_offset_, t_row_offset.numel());
        cuda_copy_h2d_if_needed(st.d_t_blockNew_offset, t_blockNew_offset_, t_blockNew_offset.numel());
        cuda_copy_h2d_if_needed(st.d_t_column, t_column_, t_column.numel());
        cuda_copy_h2d_if_needed(st.d_t_value, t_value_, t_value.numel());
        cuda_copy_h2d_if_needed(st.d_t_window_row, t_window_row_, t_window_row.numel());
        cuda_copy_h2d_if_needed(st.d_t_atomic, t_atomic_, t_atomic.numel());
        cuda_copy_h2d_if_needed(st.d_t_binary, t_binary_, t_binary.numel());
        cuda_copy_h2d_if_needed(st.d_c_row_offsets, c_row_offsets_, c_row_offsets.numel());
        cuda_copy_h2d_if_needed(st.d_c_row, c_row_, c_row.numel());
        cuda_copy_h2d_if_needed(st.d_c_atomic, c_atomic_, c_atomic.numel());
        cuda_copy_h2d_if_needed(st.d_c_column, c_column_, c_column.numel());
        cuda_copy_h2d_if_needed(st.d_c_value, c_value_, c_value.numel());
        cuda_copy_h2d_if_needed(st.d_c_row_offsets_short, c_row_offsets_short_, c_row_offsets_short.numel());
        cuda_copy_h2d_if_needed(st.d_c_row_short, c_row_short_, c_row_short.numel());
        cuda_copy_h2d_if_needed(st.d_c_atomic_short, c_atomic_short_, c_atomic_short.numel());
        cuda_copy_h2d_if_needed(st.d_c_column_short, c_column_short_, c_column_short.numel());
        cuda_copy_h2d_if_needed(st.d_c_value_short, c_value_short_, c_value_short.numel());
        cuda_copy_h2d_if_needed(st.d_s_column, s_column_, s_column.numel());
        cuda_copy_h2d_if_needed(st.d_s_value, s_value_, s_value.numel());
        cuda_copy_h2d_if_needed(st.d_s_pos, s_pos_, s_pos.numel());
        cuda_copy_h2d_if_needed(st.d_s_offsets, s_offsets_cpu, s_offsets.numel());
        cuda_copy_h2d_if_needed(st.d_tile_window, tile_window_host.data(), tile_window_host.size());
        cuda_copy_h2d_if_needed(st.d_tile_group, tile_group_host.data(), tile_group_host.size());
        cuda_copy_h2d_if_needed(st.d_s_tile_window, s_tile_window_, s_tile_window.numel());
        cuda_copy_h2d_if_needed(st.d_s_tile_column, s_tile_column_, s_tile_column.numel());
        cuda_copy_h2d_if_needed(st.d_s_packed_a, s_packed_a_, s_packed_a.numel());
        cuda_copy_h2d_if_needed(st.d_s_packed_meta, s_packed_meta_, s_packed_meta.numel());
        cuda_copy_h2d_if_needed(st.d_s_window_tile_offset, s_window_tile_offset_, s_window_tile_offset.numel());

        g_libra_prep[libra_key] = st;
    } else {
        // 上一次 kernel 可能仍在使用这些缓冲 (异步流), 复用前先对齐
        cudaDeviceSynchronize();
    }

    LibraPrepState& LP = g_libra_prep[libra_key];
    int *d_t_row_offset = LP.d_t_row_offset;
    int *d_t_blockNew_offset = LP.d_t_blockNew_offset;
    int *d_t_column = LP.d_t_column;
    int *d_t_window_row = LP.d_t_window_row;
    int *d_t_atomic = LP.d_t_atomic;
    long *d_t_binary = LP.d_t_binary;
    half *d_t_value = LP.d_t_value;
    int *d_c_row_offsets = LP.d_c_row_offsets;
    int *d_c_row = LP.d_c_row;
    int *d_c_atomic = LP.d_c_atomic;
    int *d_c_column = LP.d_c_column;
    int *d_c_row_offsets_short = LP.d_c_row_offsets_short;
    int *d_c_row_short = LP.d_c_row_short;
    int *d_c_atomic_short = LP.d_c_atomic_short;
    int *d_c_column_short = LP.d_c_column_short;
    half *d_c_value = LP.d_c_value;
    half *d_c_value_short = LP.d_c_value_short;
    int *d_s_column = LP.d_s_column;
    float *d_s_value = LP.d_s_value;
    int8_t *d_s_pos = LP.d_s_pos;
    int *d_s_offsets = LP.d_s_offsets;
    int *d_tile_window = LP.d_tile_window;
    int *d_tile_group = LP.d_tile_group;
    int *d_s_tile_window = LP.d_s_tile_window;
    int *d_s_tile_column = LP.d_s_tile_column;
    uint32_t *d_s_packed_a = LP.d_s_packed_a;
    uint32_t *d_s_packed_meta = LP.d_s_packed_meta;
    int *d_s_window_tile_offset = LP.d_s_window_tile_offset;
    half *d_rhs_matrix = LP.d_rhs_matrix;
    float *d_sptc_output_matrix = LP.d_sptc_output_matrix;

    // rhs 与输出零拷贝直通调用方 tensor (与 TR / MP 对齐)
    const bool libra_zero_copy = rhs_matrix.is_cuda();
    half *rhs_use = libra_zero_copy ? rhs_matrix_ : d_rhs_matrix;
    float *out_use = libra_zero_copy ? output_matrix_ : LP.d_output_matrix;

    if (!libra_zero_copy && (int64_t)kOri * dimN > 0) {
        checkCuda(cudaMemcpy(d_rhs_matrix, rhs_matrix_,
                             (int64_t)kOri * dimN * sizeof(half),
                             cudaMemcpyHostToDevice));
    }

    float tc_ms_avg = 0.0f;
    float cuda_long_ms_avg = 0.0f;
    float cuda_short_ms_avg = 0.0f;
    float sptc_ms_avg = 0.0f;
    auto _t2 = std::chrono::high_resolution_clock::now();
    float spmm_ms_avg = spmm_forward_fp16_tcu_cuda_sptc_mma_parallel_online_kernel(
        d_t_row_offset,
        d_t_blockNew_offset,
        d_t_value,
        d_t_column,
        d_t_window_row,
        d_t_atomic,
        d_t_binary,
        d_c_row_offsets,
        d_c_row,
        d_c_atomic,
        d_c_column,
        d_c_value,
        d_c_row_offsets_short,
        d_c_row_short,
        d_c_atomic_short,
        d_c_column_short,
        d_c_value_short,
        d_s_column,
        d_s_value,
        d_s_pos,
        d_s_offsets,
        d_tile_window,
        d_tile_group,
        d_s_tile_column,
        d_s_packed_a,
        d_s_packed_meta,
        d_s_tile_window,
        d_s_window_tile_offset,
        rhs_use,
        out_use,
        d_sptc_output_matrix,
        parts_t,
        parts_c,
        partsize_c,
        parts_c_short,
        num_tiles,
        num_windows,
        window,
        dimN,
        mOri,
        kOri,
        epoches,
        &tc_ms_avg,
        &cuda_long_ms_avg,
        &cuda_short_ms_avg,
        &sptc_ms_avg);

    // 零拷贝路径: kernel 已写入 output_matrix_; 仅 CPU 输入需 D2H
    if (!libra_zero_copy) {
        checkCuda(cudaMemcpy(output_matrix_, LP.d_output_matrix,
                             (int64_t)mOri * dimN * sizeof(float),
                             cudaMemcpyDeviceToHost));
    }

    // device 缓冲归缓存所有 (由 libra_clear_prep / 下一次 miss 统一释放)
    cudaDeviceSynchronize();
    if (libra_timing) {
        auto _t3 = std::chrono::high_resolution_clock::now();
        auto ms = [](auto a, auto b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        printf("[Libra timing] zeros=%.3f  pre-kernel=%.3f  kernel=%.3f  sync=%.3f\n",
               ms(_t0, _t1), ms(_t1, _t2), ms(_t2, _t3), ms(_t3, _t3));
    }

    return {
        output_matrix,
        torch::tensor(spmm_ms_avg),
        torch::tensor(tc_ms_avg),
        torch::tensor(cuda_long_ms_avg),
        torch::tensor(cuda_short_ms_avg),
        torch::tensor(sptc_ms_avg)
    };
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

    m.def("forward_fp16_sptc_online", &spmm_forward_fp16_sptc_online, "Libra online SPTC 2:4 SpMM (window=16)");

    m.def("forward_fp16_sptc_mma_online", &spmm_forward_fp16_sptc_mma_online, "Libra online SPTC 2:4 SpMM with mma.sp (window=16)");

    m.def("forward_fp16_sptc_mma_packed_online", &spmm_forward_fp16_sptc_mma_packed_online, "Libra online packed SPTC 2:4 SpMM with mma.sp (window=16 native)");

    m.def("forward_fp16_tcu_cuda_sptc_mma_parallel_online", &spmm_forward_fp16_tcu_cuda_sptc_mma_parallel_online, "Libra online TC/CUDA + SPTC mma.sp parallel SpMM (window=16)");

    m.def("clear_prep_cache", &libra_clear_prep, "release cached device buffers of the online pipeline");

}
