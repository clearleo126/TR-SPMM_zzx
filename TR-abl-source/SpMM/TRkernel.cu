/**
 * TRkernel.cu - TC (Dense MMA) + SPTC (mma.sp) 双流并行 CUDA 内核
 *
 * 架构 (最终版):
 *   - TC  kernel: 稠密 Tensor Core GEMM (mma.sync.aligned.m16n8k16).
 *     每 CTA 覆盖全部 N 列 (grid.y 移除), A fragment 每块加载一次由 4 次
 *     mma 复用; B 用 uint64 宽加载; 窗口内按 TC_CHUNK 分块并行消除 straggler.
 *   - SPTC kernel: 结构化稀疏 Tensor Core GEMM (mma.sp.sync.aligned.m16n8k16).
 *     A/meta 寄存器双缓冲 + 每 CTA 覆盖全部 N (Buint64 宽加载, 同 MP).
 *     窗口内按 SPTC_CHUNK 分块并行.
 *   - 双 CUDA Stream 并行执行 TC/SPTC, 共享 d_output.
 *     单写者窗口 (该窗口仅一个 chunk 且另一流无块) 用普通 vector store
 *     写回, 否则用 atomicAdd (float2/float4).
 *
 * 主要优化:
 *   [A] SPTC: A/meta 寄存器双缓冲流水线, 隐藏 global load 延迟.
 *   [B] SPTC: 每 CTA 覆盖全部 N, 同一块 A 只加载 1 次, 16 次 mma.sp 复用
 *       (消除旧版 grid.y=feature_tiles 造成的 A 重复加载).
 *   [C] SPTC: B 用 4×uint64 合并宽加载 (MP Buint64 布局).
 *   [D] TC: 稠密 A tile 直接加载, 免去 smem scatter + 同步 (scatter 延迟
 *       曾主导 TC 耗时).
 *   [E] TC/SPTC: 窗口内分块 (chunking), 每 CTA 至多处理 CHUNK 块, 消除
 *       straggler 窗口 (mip1 单窗口最多 8299 个 TC 块).
 *   [F] 写回: 单写者窗口普通 store, 多写者窗口 atomicAdd.
 *   [G] TC: 每 CTA 覆盖全部 N (grid 改 (num_chunks,1)), A fragment 块级
 *       加载一次, 4 次 mma 复用, 消除 feature_tiles 倍 A 重复加载.
 *   [H] TC: B 改 4×uint64 宽加载 (每线程 4 连续 N 列, 交错布局同 SPTC).
 *   [I] TC: m16n8k16 (dense_threshold>8) 替代 2×m16n8k8, mma 指令数减半.
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>


// ============================================================================
// 消融实验 (ablation) 开关 —— 外部参数控制要移除的单项优化
//   0  = baseline (全部优化启用)
//   1  = no-A: 去掉 SPTC A/meta 寄存器双缓冲流水线
//   2  = no-B: SPTC 退回 grid.y=feature_tiles (A 重复加载, 旧版布局)
//   3  = no-C: SPTC B 退回标量逐列加载 (16 次 __ldg half)
//   4  = no-D: TC A 退回 smem scatter 重建 (旧版路径)
//   5  = no-E: 禁用窗口内分块 (每窗口一个 CTA, 串行处理)
//   6  = no-F: 写回全部 atomicAdd (去掉单写者 float4 快写)
//   7  = no-G: TC 退回 grid.y=feature_tiles (A 重复加载, 旧版布局)
//   8  = no-H: TC B 退回标量逐列加载 (旧版 16 次 __ldg half)
//   9  = no-I: TC 强制 2×m16n8k8 (不用 m16n8k16)
//   10 = no-AB: 同时去掉 TC 与 SPTC 双流并行 (SPTC 先行, TC 后行, 串行)
// ============================================================================

#define ABLATION_BASELINE           0
#define ABLATION_NO_A               1
#define ABLATION_NO_B               2
#define ABLATION_NO_C               3
#define ABLATION_NO_D               4
#define ABLATION_NO_E               5
#define ABLATION_NO_F               6
#define ABLATION_NO_G               7
#define ABLATION_NO_H               8
#define ABLATION_NO_I               9
#define ABLATION_NO_STREAMS         10


// ============================================================================
// 辅助函数
// ============================================================================

static __device__ __forceinline__ uint32_t pack_half2_u32(half lo, half hi) {
    return (uint32_t)__half_as_ushort(lo) | ((uint32_t)__half_as_ushort(hi) << 16);
}

// 64-bit CAS 实现的 float2 向量化原子加
// 要求 addr 8B 对齐 (out_col0 为偶数时天然满足)
static __device__ __forceinline__ void atomic_add_float2(float* addr, float a, float b) {
    unsigned long long* addr64 = reinterpret_cast<unsigned long long*>(addr);
    unsigned long long old = *addr64;
    unsigned long long assumed;
    do {
        assumed = old;
        float f0 = __uint_as_float((unsigned int)assumed);
        float f1 = __uint_as_float((unsigned int)(assumed >> 32));
        f0 += a;
        f1 += b;
        old = atomicCAS(addr64, assumed,
              (unsigned long long)__float_as_uint(f0) |
              ((unsigned long long)__float_as_uint(f1) << 32));
    } while (assumed != old);
}

// ============================================================================
// TC Kernel - 稠密 A tile 直接加载 + 每 CTA 覆盖全部 N
//
// Grid:  (num_chunks, 1)   —— chunk 化消除 straggler
// Block: (32, warp_count, 1)  warp_count = ceil(dimN/32), 每 warp 覆盖 32 列 N
//
// [G] A 复用: 旧版 grid.y = feature_tiles, 同一 A tile 被独立加载
//     feature_tiles 次 (dimN=128 时 A DRAM 流量 4 倍). 现在每 CTA 覆盖全部 N,
//     A fragment 每块加载一次, 由 4 次 mma 复用, 跨 warp 走 L1 广播.
// [H] B 向量化: 每线程 4 个连续 N 列 (交错布局, 同 SPTC), B 用 4×uint64
//     合并宽加载, 覆盖 4 次 mma 的 B fragment.
// [I] m16n8k16: dense_threshold>8 时单次 mma 覆盖 K=16, 指令数减半;
//     dense_threshold<=8 走 k8 变体 tr_tc_kernel_k8 (A 只有低 8 列有数据).
//
// A 数据由 TR.cpp CPU 端展开为 [num_tc_blocks*256] 16×16 zero-padded tile
// (布局 [r*16+c] 与 tc_local_id 一致), 内核每线程直接 __ldg 加载 A fragment,
// 无 smem / 无 __syncthreads; col_old 补齐 16/block (0 pad → B 加载合法).
// ============================================================================

__global__ void tr_tc_kernel_k16(
    const int* __restrict__ tc_chunk_win,      // [num_chunks] 每 chunk 的窗口 id
    const int* __restrict__ tc_chunk_beg,      // [num_chunks] 块起始 (含)
    const int* __restrict__ tc_chunk_end,      // [num_chunks] 块结束 (不含)
    const uint8_t* __restrict__ tc_chunk_atomic, // [num_chunks] 1=写回需 atomicAdd
    const half* __restrict__ tc_value_dense,   // [num_tc_blocks * 256] 16×16 tiles
    const int* __restrict__ tc_col_old,        // [num_tc_blocks * 16] (0 pad)
    const half* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix,
    int num_chunks,
    int dimN)
{
#if __CUDA_ARCH__ >= 800
    int chunk_id = blockIdx.x;
    if (chunk_id >= num_chunks) return;

    int window_id = __ldg(tc_chunk_win + chunk_id);
    int block_start = __ldg(tc_chunk_beg + chunk_id);
    int block_end   = __ldg(tc_chunk_end + chunk_id);
    if (block_start >= block_end) return;

    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.y;          // 每 warp 覆盖 32 列 N
    int groupID = lane >> 2;            // 0..7
    int tid_in_group = lane & 3;        // 0..3

    int warp_feat_base = warp_id * 32;  // 该 warp 的 N 起始偏移
    if (warp_feat_base >= dimN) return;

    int window_row = window_id * 16;
    int row0 = window_row + groupID;
    int row1 = row0 + 8;

    // 每线程 4 个连续 N 列 (交错, 同 SPTC): mma j 用列 dense_B_idx_base+j
    const int dense_B_idx_base = groupID * 4 + warp_feat_base;
    bool valid4 = (dense_B_idx_base + 3 < dimN);

    // C 累加器: 4 次 mma × 每线程 4 float
    float RC[16] = {0.0f, 0.0f, 0.0f, 0.0f,
                    0.0f, 0.0f, 0.0f, 0.0f,
                    0.0f, 0.0f, 0.0f, 0.0f,
                    0.0f, 0.0f, 0.0f, 0.0f};

    // 每块: 加载 A fragment (4×uint32, 4 次 mma 共享) + B (4×uint64) + 4 次 mma
    for (int blk = block_start; blk < block_end; ++blk) {
        const half* tile = tc_value_dense + blk * 256;
        const int* bcol = tc_col_old + blk * 16;

        // A fragment (m16n8k16): 行 groupID / groupID+8, 列 2t..2t+1 与 2t+8..2t+9
        uint32_t a0 = __ldg(reinterpret_cast<const uint32_t*>(tile + groupID * 16 + tid_in_group * 2));
        uint32_t a1 = __ldg(reinterpret_cast<const uint32_t*>(tile + (groupID + 8) * 16 + tid_in_group * 2));
        uint32_t a2 = __ldg(reinterpret_cast<const uint32_t*>(tile + groupID * 16 + tid_in_group * 2 + 8));
        uint32_t a3 = __ldg(reinterpret_cast<const uint32_t*>(tile + (groupID + 8) * 16 + tid_in_group * 2 + 8));

        // B 行索引: K 位置 2t,2t+1 (低 8) 与 2t+8,2t+9 (高 8)
        int gk0 = __ldg(bcol + tid_in_group * 2);
        int gk1 = __ldg(bcol + tid_in_group * 2 + 1);
        int gk2 = __ldg(bcol + tid_in_group * 2 + 8);
        int gk3 = __ldg(bcol + tid_in_group * 2 + 9);

        // B 宽加载: 4 行 × 连续 4 列 = 4×uint64, 重组为 4 次 mma 的 B fragment
        uint32_t RB[8];
        if (valid4) {
            const uint64_t* src0 = reinterpret_cast<const uint64_t*>(rhs_matrix + gk0 * dimN + dense_B_idx_base);
            const uint64_t* src1 = reinterpret_cast<const uint64_t*>(rhs_matrix + gk1 * dimN + dense_B_idx_base);
            const uint64_t* src2 = reinterpret_cast<const uint64_t*>(rhs_matrix + gk2 * dimN + dense_B_idx_base);
            const uint64_t* src3 = reinterpret_cast<const uint64_t*>(rhs_matrix + gk3 * dimN + dense_B_idx_base);
            uint64_t t0 = src0[0], t1 = src1[0], t2 = src2[0], t3 = src3[0];

            // RB[j*2+0] = pack(B[gk0][base+j], B[gk1][base+j])
            // RB[j*2+1] = pack(B[gk2][base+j], B[gk3][base+j])
            const uint32_t lo0 = (uint32_t)t0, lo1 = (uint32_t)t1;
            const uint32_t lo2 = (uint32_t)t2, lo3 = (uint32_t)t3;
            const uint32_t hi0 = (uint32_t)(t0 >> 32), hi1 = (uint32_t)(t1 >> 32);
            const uint32_t hi2 = (uint32_t)(t2 >> 32), hi3 = (uint32_t)(t3 >> 32);
            RB[0] = (lo0 & 0xFFFFu) | (lo1 << 16);
            RB[1] = (lo2 & 0xFFFFu) | (lo3 << 16);
            RB[2] = (lo0 >> 16) | (lo1 & 0xFFFF0000u);
            RB[3] = (lo2 >> 16) | (lo3 & 0xFFFF0000u);
            RB[4] = (hi0 & 0xFFFFu) | (hi1 << 16);
            RB[5] = (hi2 & 0xFFFFu) | (hi3 << 16);
            RB[6] = (hi0 >> 16) | (hi1 & 0xFFFF0000u);
            RB[7] = (hi2 >> 16) | (hi3 & 0xFFFF0000u);
        } else {
            // 慢路径: 标量逐列读取 (边界/非法 gk)
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                int nc = dense_B_idx_base + j;
                half bv0 = __ldg(rhs_matrix + gk0 * dimN + nc);
                half bv1 = __ldg(rhs_matrix + gk1 * dimN + nc);
                half bv2 = __ldg(rhs_matrix + gk2 * dimN + nc);
                half bv3 = __ldg(rhs_matrix + gk3 * dimN + nc);
                RB[j * 2 + 0] = pack_half2_u32(bv0, bv1);
                RB[j * 2 + 1] = pack_half2_u32(bv2, bv3);
            }
        }

        // 4 次 mma (m16n8k16): 共享 A fragment, 每次用不同 N 列
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(RC[j * 4 + 0]), "+f"(RC[j * 4 + 1]),
                  "+f"(RC[j * 4 + 2]), "+f"(RC[j * 4 + 3])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                  "r"(RB[j * 2 + 0]), "r"(RB[j * 2 + 1]));
        }
    }

    // 写回 (同 SPTC 布局): 每线程 8 列 (列 = warp_feat_base + tid*8 + {0..7}),
    //   每列 2 行 (row0, row1). 单写者窗口 float4 直写, 否则 float2 atomicAdd.
    //   行 groupID:   列 c+0..3 ← RC[0],RC[4],RC[8],RC[12]
    //                 列 c+4..7 ← RC[1],RC[5],RC[9],RC[13]
    //   行 groupID+8: 列 c+0..3 ← RC[2],RC[6],RC[10],RC[14]
    //                 列 c+4..7 ← RC[3],RC[7],RC[11],RC[15]
    int c = warp_feat_base + tid_in_group * 8;
    bool need_atomic = (__ldg(tc_chunk_atomic + chunk_id) != 0);

    if (need_atomic) {
        atomic_add_float2(&output_matrix[row0 * dimN + c + 0], RC[0], RC[4]);
        atomic_add_float2(&output_matrix[row0 * dimN + c + 2], RC[8], RC[12]);
        atomic_add_float2(&output_matrix[row0 * dimN + c + 4], RC[1], RC[5]);
        atomic_add_float2(&output_matrix[row0 * dimN + c + 6], RC[9], RC[13]);

        atomic_add_float2(&output_matrix[row1 * dimN + c + 0], RC[2], RC[6]);
        atomic_add_float2(&output_matrix[row1 * dimN + c + 2], RC[10], RC[14]);
        atomic_add_float2(&output_matrix[row1 * dimN + c + 4], RC[3], RC[7]);
        atomic_add_float2(&output_matrix[row1 * dimN + c + 6], RC[11], RC[15]);
    } else {
        *reinterpret_cast<float4*>(&output_matrix[row0 * dimN + c + 0]) =
            make_float4(RC[0], RC[4], RC[8], RC[12]);
        *reinterpret_cast<float4*>(&output_matrix[row0 * dimN + c + 4]) =
            make_float4(RC[1], RC[5], RC[9], RC[13]);

        *reinterpret_cast<float4*>(&output_matrix[row1 * dimN + c + 0]) =
            make_float4(RC[2], RC[6], RC[10], RC[14]);
        *reinterpret_cast<float4*>(&output_matrix[row1 * dimN + c + 4]) =
            make_float4(RC[3], RC[7], RC[11], RC[15]);
    }
#endif
}


// ============================================================================
// TC k16 消融变体 (共用参数表)
//   - no-H: B 退回标量逐列加载 (旧版, 每线程 16 次 half __ldg)
//   - no-I: 强制 2×m16n8k8 (等价于 k8 路径, 但走同一分块结构)
// ============================================================================

__global__ void tr_tc_kernel_k16_noh(
    const int* __restrict__ tc_chunk_win,
    const int* __restrict__ tc_chunk_beg,
    const int* __restrict__ tc_chunk_end,
    const uint8_t* __restrict__ tc_chunk_atomic,
    const half* __restrict__ tc_value_dense,
    const int* __restrict__ tc_col_old,
    const half* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix,
    int num_chunks,
    int dimN)
{
#if __CUDA_ARCH__ >= 800
    int chunk_id = blockIdx.x;
    if (chunk_id >= num_chunks) return;

    int window_id = __ldg(tc_chunk_win + chunk_id);
    int block_start = __ldg(tc_chunk_beg + chunk_id);
    int block_end   = __ldg(tc_chunk_end + chunk_id);
    if (block_start >= block_end) return;

    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.y;
    int groupID = lane >> 2;
    int tid_in_group = lane & 3;

    int warp_feat_base = warp_id * 32;
    if (warp_feat_base >= dimN) return;

    int window_row = window_id * 16;
    int row0 = window_row + groupID;
    int row1 = row0 + 8;

    const int dense_B_idx_base = groupID * 4 + warp_feat_base;

    float RC[16] = {0.0f, 0.0f, 0.0f, 0.0f,
                    0.0f, 0.0f, 0.0f, 0.0f,
                    0.0f, 0.0f, 0.0f, 0.0f,
                    0.0f, 0.0f, 0.0f, 0.0f};

    for (int blk = block_start; blk < block_end; ++blk) {
        const half* tile = tc_value_dense + blk * 256;
        const int* bcol = tc_col_old + blk * 16;

        uint32_t a0 = __ldg(reinterpret_cast<const uint32_t*>(tile + groupID * 16 + tid_in_group * 2));
        uint32_t a1 = __ldg(reinterpret_cast<const uint32_t*>(tile + (groupID + 8) * 16 + tid_in_group * 2));
        uint32_t a2 = __ldg(reinterpret_cast<const uint32_t*>(tile + groupID * 16 + tid_in_group * 2 + 8));
        uint32_t a3 = __ldg(reinterpret_cast<const uint32_t*>(tile + (groupID + 8) * 16 + tid_in_group * 2 + 8));

        int gk0 = __ldg(bcol + tid_in_group * 2);
        int gk1 = __ldg(bcol + tid_in_group * 2 + 1);
        int gk2 = __ldg(bcol + tid_in_group * 2 + 8);
        int gk3 = __ldg(bcol + tid_in_group * 2 + 9);

        // [消融 no-H] B 标量逐列加载: 每线程 16 次 half __ldg
        uint32_t RB[8];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            int nc = dense_B_idx_base + j;
            half bv0 = __ldg(rhs_matrix + gk0 * dimN + nc);
            half bv1 = __ldg(rhs_matrix + gk1 * dimN + nc);
            half bv2 = __ldg(rhs_matrix + gk2 * dimN + nc);
            half bv3 = __ldg(rhs_matrix + gk3 * dimN + nc);
            RB[j * 2 + 0] = pack_half2_u32(bv0, bv1);
            RB[j * 2 + 1] = pack_half2_u32(bv2, bv3);
        }

#pragma unroll
        for (int j = 0; j < 4; ++j) {
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(RC[j * 4 + 0]), "+f"(RC[j * 4 + 1]),
                  "+f"(RC[j * 4 + 2]), "+f"(RC[j * 4 + 3])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                  "r"(RB[j * 2 + 0]), "r"(RB[j * 2 + 1]));
        }
    }

    // 写回 (同 baseline)
    int c = warp_feat_base + tid_in_group * 8;
    bool need_atomic = (__ldg(tc_chunk_atomic + chunk_id) != 0);

    if (need_atomic) {
        atomic_add_float2(&output_matrix[row0 * dimN + c + 0], RC[0], RC[4]);
        atomic_add_float2(&output_matrix[row0 * dimN + c + 2], RC[8], RC[12]);
        atomic_add_float2(&output_matrix[row0 * dimN + c + 4], RC[1], RC[5]);
        atomic_add_float2(&output_matrix[row0 * dimN + c + 6], RC[9], RC[13]);

        atomic_add_float2(&output_matrix[row1 * dimN + c + 0], RC[2], RC[6]);
        atomic_add_float2(&output_matrix[row1 * dimN + c + 2], RC[10], RC[14]);
        atomic_add_float2(&output_matrix[row1 * dimN + c + 4], RC[3], RC[7]);
        atomic_add_float2(&output_matrix[row1 * dimN + c + 6], RC[11], RC[15]);
    } else {
        *reinterpret_cast<float4*>(&output_matrix[row0 * dimN + c + 0]) =
            make_float4(RC[0], RC[4], RC[8], RC[12]);
        *reinterpret_cast<float4*>(&output_matrix[row0 * dimN + c + 4]) =
            make_float4(RC[1], RC[5], RC[9], RC[13]);

        *reinterpret_cast<float4*>(&output_matrix[row1 * dimN + c + 0]) =
            make_float4(RC[2], RC[6], RC[10], RC[14]);
        *reinterpret_cast<float4*>(&output_matrix[row1 * dimN + c + 4]) =
            make_float4(RC[3], RC[7], RC[11], RC[15]);
    }
#endif
}


// [消融 no-D] TC A 退回 smem scatter 重建: 每块 256 half 全局读 → smem 写 →
//   __syncthreads ×2 → 从 smem 读 A fragment (复现旧版 D 优化前的开销模式)
__global__ void tr_tc_kernel_k16_nod(
    const int* __restrict__ tc_chunk_win,
    const int* __restrict__ tc_chunk_beg,
    const int* __restrict__ tc_chunk_end,
    const uint8_t* __restrict__ tc_chunk_atomic,
    const half* __restrict__ tc_value_dense,
    const int* __restrict__ tc_col_old,
    const half* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix,
    int num_chunks,
    int dimN)
{
#if __CUDA_ARCH__ >= 800
    int chunk_id = blockIdx.x;
    if (chunk_id >= num_chunks) return;

    int window_id = __ldg(tc_chunk_win + chunk_id);
    int block_start = __ldg(tc_chunk_beg + chunk_id);
    int block_end   = __ldg(tc_chunk_end + chunk_id);
    if (block_start >= block_end) return;

    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.y;
    int groupID = lane >> 2;
    int tid_in_group = lane & 3;

    int warp_feat_base = warp_id * 32;
    if (warp_feat_base >= dimN) return;

    int window_row = window_id * 16;
    int row0 = window_row + groupID;
    int row1 = row0 + 8;

    const int dense_B_idx_base = groupID * 4 + warp_feat_base;

    __shared__ half tile_smem[256];

    float RC[16] = {0.0f, 0.0f, 0.0f, 0.0f,
                    0.0f, 0.0f, 0.0f, 0.0f,
                    0.0f, 0.0f, 0.0f, 0.0f,
                    0.0f, 0.0f, 0.0f, 0.0f};

    for (int blk = block_start; blk < block_end; ++blk) {
        const half* tile = tc_value_dense + blk * 256;
        const int* bcol = tc_col_old + blk * 16;

        // [消融 no-D] 全 CTA 协作把 16×16 tile 搬入 smem
        //   tile = 256 half = 128 uint32; 按 blockDim.x 循环覆盖 (CTA 可能
        //   只有 32 线程, 如 dimN<=32)
        {
            const uint32_t* src32 = reinterpret_cast<const uint32_t*>(tile);
            uint32_t* dst32 = reinterpret_cast<uint32_t*>(tile_smem);
            for (int idx = threadIdx.x; idx < 128; idx += blockDim.x)
                dst32[idx] = __ldg(src32 + idx);
        }
        __syncthreads();

        uint32_t a0 = *reinterpret_cast<const uint32_t*>(&tile_smem[groupID * 16 + tid_in_group * 2]);
        uint32_t a1 = *reinterpret_cast<const uint32_t*>(&tile_smem[(groupID + 8) * 16 + tid_in_group * 2]);
        uint32_t a2 = *reinterpret_cast<const uint32_t*>(&tile_smem[groupID * 16 + tid_in_group * 2 + 8]);
        uint32_t a3 = *reinterpret_cast<const uint32_t*>(&tile_smem[(groupID + 8) * 16 + tid_in_group * 2 + 8]);
        __syncthreads();

        int gk0 = __ldg(bcol + tid_in_group * 2);
        int gk1 = __ldg(bcol + tid_in_group * 2 + 1);
        int gk2 = __ldg(bcol + tid_in_group * 2 + 8);
        int gk3 = __ldg(bcol + tid_in_group * 2 + 9);

        uint32_t RB[8];
        if (dense_B_idx_base + 3 < dimN) {
            const uint64_t bytes0 = (uint64_t)(rhs_matrix + gk0 * dimN + dense_B_idx_base);
            uint64_t t0, t1, t2, t3;
            if ((bytes0 & 0x7) == 0) {
                t0 = __ldg(reinterpret_cast<const uint64_t*>(rhs_matrix + gk0 * dimN + dense_B_idx_base));
                t1 = __ldg(reinterpret_cast<const uint64_t*>(rhs_matrix + gk1 * dimN + dense_B_idx_base));
                t2 = __ldg(reinterpret_cast<const uint64_t*>(rhs_matrix + gk2 * dimN + dense_B_idx_base));
                t3 = __ldg(reinterpret_cast<const uint64_t*>(rhs_matrix + gk3 * dimN + dense_B_idx_base));
            } else {
                const uint32_t* e0 = reinterpret_cast<const uint32_t*>(rhs_matrix + gk0 * dimN + dense_B_idx_base);
                const uint32_t* e1 = reinterpret_cast<const uint32_t*>(rhs_matrix + gk1 * dimN + dense_B_idx_base);
                const uint32_t* e2 = reinterpret_cast<const uint32_t*>(rhs_matrix + gk2 * dimN + dense_B_idx_base);
                const uint32_t* e3 = reinterpret_cast<const uint32_t*>(rhs_matrix + gk3 * dimN + dense_B_idx_base);
                t0 = (uint64_t)__ldg(e0 + 1) << 32 | __ldg(e0);
                t1 = (uint64_t)__ldg(e1 + 1) << 32 | __ldg(e1);
                t2 = (uint64_t)__ldg(e2 + 1) << 32 | __ldg(e2);
                t3 = (uint64_t)__ldg(e3 + 1) << 32 | __ldg(e3);
            }
            const uint32_t lo0 = (uint32_t)t0, lo1 = (uint32_t)t1;
            const uint32_t lo2 = (uint32_t)t2, lo3 = (uint32_t)t3;
            const uint32_t hi0 = (uint32_t)(t0 >> 32), hi1 = (uint32_t)(t1 >> 32);
            const uint32_t hi2 = (uint32_t)(t2 >> 32), hi3 = (uint32_t)(t3 >> 32);
            RB[0] = (lo0 & 0xFFFFu) | (lo1 << 16);
            RB[1] = (lo2 & 0xFFFFu) | (lo3 << 16);
            RB[2] = (lo0 >> 16) | (lo1 & 0xFFFF0000u);
            RB[3] = (lo2 >> 16) | (lo3 & 0xFFFF0000u);
            RB[4] = (hi0 & 0xFFFFu) | (hi1 << 16);
            RB[5] = (hi2 & 0xFFFFu) | (hi3 << 16);
            RB[6] = (hi0 >> 16) | (hi1 & 0xFFFF0000u);
            RB[7] = (hi2 >> 16) | (hi3 & 0xFFFF0000u);
        } else {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                int nc = dense_B_idx_base + j;
                half bv0 = __ldg(rhs_matrix + gk0 * dimN + nc);
                half bv1 = __ldg(rhs_matrix + gk1 * dimN + nc);
                half bv2 = __ldg(rhs_matrix + gk2 * dimN + nc);
                half bv3 = __ldg(rhs_matrix + gk3 * dimN + nc);
                RB[j * 2 + 0] = pack_half2_u32(bv0, bv1);
                RB[j * 2 + 1] = pack_half2_u32(bv2, bv3);
            }
        }

#pragma unroll
        for (int j = 0; j < 4; ++j) {
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(RC[j * 4 + 0]), "+f"(RC[j * 4 + 1]),
                  "+f"(RC[j * 4 + 2]), "+f"(RC[j * 4 + 3])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                  "r"(RB[j * 2 + 0]), "r"(RB[j * 2 + 1]));
        }
    }

    int c = warp_feat_base + tid_in_group * 8;
    bool need_atomic = (__ldg(tc_chunk_atomic + chunk_id) != 0);

    if (need_atomic) {
        atomic_add_float2(&output_matrix[row0 * dimN + c + 0], RC[0], RC[4]);
        atomic_add_float2(&output_matrix[row0 * dimN + c + 2], RC[8], RC[12]);
        atomic_add_float2(&output_matrix[row0 * dimN + c + 4], RC[1], RC[5]);
        atomic_add_float2(&output_matrix[row0 * dimN + c + 6], RC[9], RC[13]);

        atomic_add_float2(&output_matrix[row1 * dimN + c + 0], RC[2], RC[6]);
        atomic_add_float2(&output_matrix[row1 * dimN + c + 2], RC[10], RC[14]);
        atomic_add_float2(&output_matrix[row1 * dimN + c + 4], RC[3], RC[7]);
        atomic_add_float2(&output_matrix[row1 * dimN + c + 6], RC[11], RC[15]);
    } else {
        *reinterpret_cast<float4*>(&output_matrix[row0 * dimN + c + 0]) =
            make_float4(RC[0], RC[4], RC[8], RC[12]);
        *reinterpret_cast<float4*>(&output_matrix[row0 * dimN + c + 4]) =
            make_float4(RC[1], RC[5], RC[9], RC[13]);

        *reinterpret_cast<float4*>(&output_matrix[row1 * dimN + c + 0]) =
            make_float4(RC[2], RC[6], RC[10], RC[14]);
        *reinterpret_cast<float4*>(&output_matrix[row1 * dimN + c + 4]) =
            make_float4(RC[3], RC[7], RC[11], RC[15]);
    }
#endif
}


// [消融 no-G] 退回旧版 grid.y=feature_tiles 布局: 每 CTA 单 warp 32 列,
//   同一窗口的 A fragment 被独立加载 feature_tiles 次.
__global__ void tr_tc_kernel_k16_nog(
    const int* __restrict__ tc_chunk_win,
    const int* __restrict__ tc_chunk_beg,
    const int* __restrict__ tc_chunk_end,
    const uint8_t* __restrict__ tc_chunk_atomic,
    const half* __restrict__ tc_value_dense,
    const int* __restrict__ tc_col_old,
    const half* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix,
    int num_chunks,
    int dimN)
{
#if __CUDA_ARCH__ >= 800
    int chunk_id = blockIdx.x;
    int feature_tile_id = blockIdx.y;
    if (feature_tile_id >= (dimN + 31) / 32) return;
    if (chunk_id >= num_chunks) return;

    int window_id = __ldg(tc_chunk_win + chunk_id);
    int block_start = __ldg(tc_chunk_beg + chunk_id);
    int block_end   = __ldg(tc_chunk_end + chunk_id);
    if (block_start >= block_end) return;

    int lane = threadIdx.x & 31;
    int groupID = lane >> 2;
    int tid_in_group = lane & 3;

    int warp_feat_base = feature_tile_id * 32;
    if (warp_feat_base >= dimN) return;

    int window_row = window_id * 16;
    int row0 = window_row + groupID;
    int row1 = row0 + 8;

    const int dense_B_idx_base = groupID * 4 + warp_feat_base;

    float RC[16] = {0.0f, 0.0f, 0.0f, 0.0f,
                    0.0f, 0.0f, 0.0f, 0.0f,
                    0.0f, 0.0f, 0.0f, 0.0f,
                    0.0f, 0.0f, 0.0f, 0.0f};

    for (int blk = block_start; blk < block_end; ++blk) {
        const half* tile = tc_value_dense + blk * 256;
        const int* bcol = tc_col_old + blk * 16;

        uint32_t a0 = __ldg(reinterpret_cast<const uint32_t*>(tile + groupID * 16 + tid_in_group * 2));
        uint32_t a1 = __ldg(reinterpret_cast<const uint32_t*>(tile + (groupID + 8) * 16 + tid_in_group * 2));
        uint32_t a2 = __ldg(reinterpret_cast<const uint32_t*>(tile + groupID * 16 + tid_in_group * 2 + 8));
        uint32_t a3 = __ldg(reinterpret_cast<const uint32_t*>(tile + (groupID + 8) * 16 + tid_in_group * 2 + 8));

        int gk0 = __ldg(bcol + tid_in_group * 2);
        int gk1 = __ldg(bcol + tid_in_group * 2 + 1);
        int gk2 = __ldg(bcol + tid_in_group * 2 + 8);
        int gk3 = __ldg(bcol + tid_in_group * 2 + 9);

        uint32_t RB[8];
        if (dense_B_idx_base + 3 < dimN) {
            const uint64_t bytes0 = (uint64_t)(rhs_matrix + gk0 * dimN + dense_B_idx_base);
            uint64_t t0, t1, t2, t3;
            if ((bytes0 & 0x7) == 0) {
                t0 = __ldg(reinterpret_cast<const uint64_t*>(rhs_matrix + gk0 * dimN + dense_B_idx_base));
                t1 = __ldg(reinterpret_cast<const uint64_t*>(rhs_matrix + gk1 * dimN + dense_B_idx_base));
                t2 = __ldg(reinterpret_cast<const uint64_t*>(rhs_matrix + gk2 * dimN + dense_B_idx_base));
                t3 = __ldg(reinterpret_cast<const uint64_t*>(rhs_matrix + gk3 * dimN + dense_B_idx_base));
            } else {
                const uint32_t* e0 = reinterpret_cast<const uint32_t*>(rhs_matrix + gk0 * dimN + dense_B_idx_base);
                const uint32_t* e1 = reinterpret_cast<const uint32_t*>(rhs_matrix + gk1 * dimN + dense_B_idx_base);
                const uint32_t* e2 = reinterpret_cast<const uint32_t*>(rhs_matrix + gk2 * dimN + dense_B_idx_base);
                const uint32_t* e3 = reinterpret_cast<const uint32_t*>(rhs_matrix + gk3 * dimN + dense_B_idx_base);
                t0 = (uint64_t)__ldg(e0 + 1) << 32 | __ldg(e0);
                t1 = (uint64_t)__ldg(e1 + 1) << 32 | __ldg(e1);
                t2 = (uint64_t)__ldg(e2 + 1) << 32 | __ldg(e2);
                t3 = (uint64_t)__ldg(e3 + 1) << 32 | __ldg(e3);
            }
            const uint32_t lo0 = (uint32_t)t0, lo1 = (uint32_t)t1;
            const uint32_t lo2 = (uint32_t)t2, lo3 = (uint32_t)t3;
            const uint32_t hi0 = (uint32_t)(t0 >> 32), hi1 = (uint32_t)(t1 >> 32);
            const uint32_t hi2 = (uint32_t)(t2 >> 32), hi3 = (uint32_t)(t3 >> 32);
            RB[0] = (lo0 & 0xFFFFu) | (lo1 << 16);
            RB[1] = (lo2 & 0xFFFFu) | (lo3 << 16);
            RB[2] = (lo0 >> 16) | (lo1 & 0xFFFF0000u);
            RB[3] = (lo2 >> 16) | (lo3 & 0xFFFF0000u);
            RB[4] = (hi0 & 0xFFFFu) | (hi1 << 16);
            RB[5] = (hi2 & 0xFFFFu) | (hi3 << 16);
            RB[6] = (hi0 >> 16) | (hi1 & 0xFFFF0000u);
            RB[7] = (hi2 >> 16) | (hi3 & 0xFFFF0000u);
        } else {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                int nc = dense_B_idx_base + j;
                half bv0 = __ldg(rhs_matrix + gk0 * dimN + nc);
                half bv1 = __ldg(rhs_matrix + gk1 * dimN + nc);
                half bv2 = __ldg(rhs_matrix + gk2 * dimN + nc);
                half bv3 = __ldg(rhs_matrix + gk3 * dimN + nc);
                RB[j * 2 + 0] = pack_half2_u32(bv0, bv1);
                RB[j * 2 + 1] = pack_half2_u32(bv2, bv3);
            }
        }

#pragma unroll
        for (int j = 0; j < 4; ++j) {
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(RC[j * 4 + 0]), "+f"(RC[j * 4 + 1]),
                  "+f"(RC[j * 4 + 2]), "+f"(RC[j * 4 + 3])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                  "r"(RB[j * 2 + 0]), "r"(RB[j * 2 + 1]));
        }
    }

    int c = warp_feat_base + tid_in_group * 8;
    bool need_atomic = (__ldg(tc_chunk_atomic + chunk_id) != 0);

    if (need_atomic) {
        atomic_add_float2(&output_matrix[row0 * dimN + c + 0], RC[0], RC[4]);
        atomic_add_float2(&output_matrix[row0 * dimN + c + 2], RC[8], RC[12]);
        atomic_add_float2(&output_matrix[row0 * dimN + c + 4], RC[1], RC[5]);
        atomic_add_float2(&output_matrix[row0 * dimN + c + 6], RC[9], RC[13]);

        atomic_add_float2(&output_matrix[row1 * dimN + c + 0], RC[2], RC[6]);
        atomic_add_float2(&output_matrix[row1 * dimN + c + 2], RC[10], RC[14]);
        atomic_add_float2(&output_matrix[row1 * dimN + c + 4], RC[3], RC[7]);
        atomic_add_float2(&output_matrix[row1 * dimN + c + 6], RC[11], RC[15]);
    } else {
        *reinterpret_cast<float4*>(&output_matrix[row0 * dimN + c + 0]) =
            make_float4(RC[0], RC[4], RC[8], RC[12]);
        *reinterpret_cast<float4*>(&output_matrix[row0 * dimN + c + 4]) =
            make_float4(RC[1], RC[5], RC[9], RC[13]);

        *reinterpret_cast<float4*>(&output_matrix[row1 * dimN + c + 0]) =
            make_float4(RC[2], RC[6], RC[10], RC[14]);
        *reinterpret_cast<float4*>(&output_matrix[row1 * dimN + c + 4]) =
            make_float4(RC[3], RC[7], RC[11], RC[15]);
    }
#endif
}


// [消融 no-I] 8×m16n8k16 替代 4×m16n8k16: 同样数学结果, mma 指令数翻倍
__global__ void tr_tc_kernel_k16_noi(
    const int* __restrict__ tc_chunk_win,
    const int* __restrict__ tc_chunk_beg,
    const int* __restrict__ tc_chunk_end,
    const uint8_t* __restrict__ tc_chunk_atomic,
    const half* __restrict__ tc_value_dense,
    const int* __restrict__ tc_col_old,
    const half* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix,
    int num_chunks,
    int dimN)
{
#if __CUDA_ARCH__ >= 800
    int chunk_id = blockIdx.x;
    if (chunk_id >= num_chunks) return;

    int window_id = __ldg(tc_chunk_win + chunk_id);
    int block_start = __ldg(tc_chunk_beg + chunk_id);
    int block_end   = __ldg(tc_chunk_end + chunk_id);
    if (block_start >= block_end) return;

    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.y;
    int groupID = lane >> 2;
    int tid_in_group = lane & 3;

    int warp_feat_base = warp_id * 32;
    if (warp_feat_base >= dimN) return;

    int window_row = window_id * 16;
    int row0 = window_row + groupID;
    int row1 = row0 + 8;

    const int dense_B_idx_base = groupID * 4 + warp_feat_base;

    float RC[16] = {0.0f, 0.0f, 0.0f, 0.0f,
                    0.0f, 0.0f, 0.0f, 0.0f,
                    0.0f, 0.0f, 0.0f, 0.0f,
                    0.0f, 0.0f, 0.0f, 0.0f};

    for (int blk = block_start; blk < block_end; ++blk) {
        const half* tile = tc_value_dense + blk * 256;
        const int* bcol = tc_col_old + blk * 16;

        uint32_t a0 = __ldg(reinterpret_cast<const uint32_t*>(tile + groupID * 16 + tid_in_group * 2));
        uint32_t a1 = __ldg(reinterpret_cast<const uint32_t*>(tile + (groupID + 8) * 16 + tid_in_group * 2));
        uint32_t a2 = __ldg(reinterpret_cast<const uint32_t*>(tile + groupID * 16 + tid_in_group * 2 + 8));
        uint32_t a3 = __ldg(reinterpret_cast<const uint32_t*>(tile + (groupID + 8) * 16 + tid_in_group * 2 + 8));

        int gk0 = __ldg(bcol + tid_in_group * 2);
        int gk1 = __ldg(bcol + tid_in_group * 2 + 1);
        int gk2 = __ldg(bcol + tid_in_group * 2 + 8);
        int gk3 = __ldg(bcol + tid_in_group * 2 + 9);

        // B 宽加载同 baseline (no-I 只改 mma 形状, 不改加载)
        uint32_t RB[8];
        if (dense_B_idx_base + 3 < dimN) {
            const uint64_t bytes0 = (uint64_t)(rhs_matrix + gk0 * dimN + dense_B_idx_base);
            uint64_t t0, t1, t2, t3;
            if ((bytes0 & 0x7) == 0) {
                t0 = __ldg(reinterpret_cast<const uint64_t*>(rhs_matrix + gk0 * dimN + dense_B_idx_base));
                t1 = __ldg(reinterpret_cast<const uint64_t*>(rhs_matrix + gk1 * dimN + dense_B_idx_base));
                t2 = __ldg(reinterpret_cast<const uint64_t*>(rhs_matrix + gk2 * dimN + dense_B_idx_base));
                t3 = __ldg(reinterpret_cast<const uint64_t*>(rhs_matrix + gk3 * dimN + dense_B_idx_base));
            } else {
                const uint32_t* e0 = reinterpret_cast<const uint32_t*>(rhs_matrix + gk0 * dimN + dense_B_idx_base);
                const uint32_t* e1 = reinterpret_cast<const uint32_t*>(rhs_matrix + gk1 * dimN + dense_B_idx_base);
                const uint32_t* e2 = reinterpret_cast<const uint32_t*>(rhs_matrix + gk2 * dimN + dense_B_idx_base);
                const uint32_t* e3 = reinterpret_cast<const uint32_t*>(rhs_matrix + gk3 * dimN + dense_B_idx_base);
                t0 = (uint64_t)__ldg(e0 + 1) << 32 | __ldg(e0);
                t1 = (uint64_t)__ldg(e1 + 1) << 32 | __ldg(e1);
                t2 = (uint64_t)__ldg(e2 + 1) << 32 | __ldg(e2);
                t3 = (uint64_t)__ldg(e3 + 1) << 32 | __ldg(e3);
            }
            const uint32_t lo0 = (uint32_t)t0, lo1 = (uint32_t)t1;
            const uint32_t lo2 = (uint32_t)t2, lo3 = (uint32_t)t3;
            const uint32_t hi0 = (uint32_t)(t0 >> 32), hi1 = (uint32_t)(t1 >> 32);
            const uint32_t hi2 = (uint32_t)(t2 >> 32), hi3 = (uint32_t)(t3 >> 32);
            RB[0] = (lo0 & 0xFFFFu) | (lo1 << 16);
            RB[1] = (lo2 & 0xFFFFu) | (lo3 << 16);
            RB[2] = (lo0 >> 16) | (lo1 & 0xFFFF0000u);
            RB[3] = (lo2 >> 16) | (lo3 & 0xFFFF0000u);
            RB[4] = (hi0 & 0xFFFFu) | (hi1 << 16);
            RB[5] = (hi2 & 0xFFFFu) | (hi3 << 16);
            RB[6] = (hi0 >> 16) | (hi1 & 0xFFFF0000u);
            RB[7] = (hi2 >> 16) | (hi3 & 0xFFFF0000u);
        } else {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                int nc = dense_B_idx_base + j;
                half bv0 = __ldg(rhs_matrix + gk0 * dimN + nc);
                half bv1 = __ldg(rhs_matrix + gk1 * dimN + nc);
                half bv2 = __ldg(rhs_matrix + gk2 * dimN + nc);
                half bv3 = __ldg(rhs_matrix + gk3 * dimN + nc);
                RB[j * 2 + 0] = pack_half2_u32(bv0, bv1);
                RB[j * 2 + 1] = pack_half2_u32(bv2, bv3);
            }
        }

        // 8× m16n8k8: 前 4 次累 K 低半 (A 低, B 行 gk0/gk1), 后 4 次累 K 高半
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            asm volatile(
                "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};\n"
                : "+f"(RC[j * 4 + 0]), "+f"(RC[j * 4 + 1]),
                  "+f"(RC[j * 4 + 2]), "+f"(RC[j * 4 + 3])
                : "r"(a0), "r"(a1), "r"(RB[j * 2 + 0]));
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            asm volatile(
                "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};\n"
                : "+f"(RC[j * 4 + 0]), "+f"(RC[j * 4 + 1]),
                  "+f"(RC[j * 4 + 2]), "+f"(RC[j * 4 + 3])
                : "r"(a2), "r"(a3), "r"(RB[j * 2 + 1]));
        }
    }

    int c = warp_feat_base + tid_in_group * 8;
    bool need_atomic = (__ldg(tc_chunk_atomic + chunk_id) != 0);

    if (need_atomic) {
        atomic_add_float2(&output_matrix[row0 * dimN + c + 0], RC[0], RC[4]);
        atomic_add_float2(&output_matrix[row0 * dimN + c + 2], RC[8], RC[12]);
        atomic_add_float2(&output_matrix[row0 * dimN + c + 4], RC[1], RC[5]);
        atomic_add_float2(&output_matrix[row0 * dimN + c + 6], RC[9], RC[13]);

        atomic_add_float2(&output_matrix[row1 * dimN + c + 0], RC[2], RC[6]);
        atomic_add_float2(&output_matrix[row1 * dimN + c + 2], RC[10], RC[14]);
        atomic_add_float2(&output_matrix[row1 * dimN + c + 4], RC[3], RC[7]);
        atomic_add_float2(&output_matrix[row1 * dimN + c + 6], RC[11], RC[15]);
    } else {
        *reinterpret_cast<float4*>(&output_matrix[row0 * dimN + c + 0]) =
            make_float4(RC[0], RC[4], RC[8], RC[12]);
        *reinterpret_cast<float4*>(&output_matrix[row0 * dimN + c + 4]) =
            make_float4(RC[1], RC[5], RC[9], RC[13]);

        *reinterpret_cast<float4*>(&output_matrix[row1 * dimN + c + 0]) =
            make_float4(RC[2], RC[6], RC[10], RC[14]);
        *reinterpret_cast<float4*>(&output_matrix[row1 * dimN + c + 4]) =
            make_float4(RC[3], RC[7], RC[11], RC[15]);
    }
#endif
}


// k8 变体: dense_threshold <= 8 (A tile 只有低 8 列有数据), 单 mma 覆盖 K=8
//   ABL: 0=baseline, 1=no-D(smem scatter), 2=no-G(feature_tiles grid),
//        3=no-H(scalar B). no-I 对 k8 无意义 (K 本来就是 8).
template <int ABL>
__global__ void tr_tc_kernel_k8_abl(
    const int* __restrict__ tc_chunk_win,
    const int* __restrict__ tc_chunk_beg,
    const int* __restrict__ tc_chunk_end,
    const uint8_t* __restrict__ tc_chunk_atomic,
    const half* __restrict__ tc_value_dense,
    const int* __restrict__ tc_col_old,
    const half* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix,
    int num_chunks,
    int dimN)
{
    int chunk_id = blockIdx.x;
    if (chunk_id >= num_chunks) return;

    int window_id = __ldg(tc_chunk_win + chunk_id);
    int block_start = __ldg(tc_chunk_beg + chunk_id);
    int block_end   = __ldg(tc_chunk_end + chunk_id);
    if (block_start >= block_end) return;

    int lane = threadIdx.x & 31;
    int groupID = lane >> 2;
    int tid_in_group = lane & 3;

    // [消融 no-G] blockIdx.y = feature_tile, 单 warp 32 列; baseline: warp_id 覆盖全 N
    int warp_id = (ABL == 2) ? 0 : (int)threadIdx.y;
    int feature_tile = (ABL == 2) ? (int)blockIdx.y : 0;

    int warp_feat_base = (ABL == 2) ? feature_tile * 32 : warp_id * 32;
    if (warp_feat_base >= dimN) return;

    int window_row = window_id * 16;
    int row0 = window_row + groupID;
    int row1 = row0 + 8;

    const int dense_B_idx_base = groupID * 4 + warp_feat_base;
    bool valid4 = (dense_B_idx_base + 3 < dimN);

    __shared__ half tile_smem[ABL == 1 ? 256 : 1];

    float RC[16] = {0.0f, 0.0f, 0.0f, 0.0f,
                    0.0f, 0.0f, 0.0f, 0.0f,
                    0.0f, 0.0f, 0.0f, 0.0f,
                    0.0f, 0.0f, 0.0f, 0.0f};

    for (int blk = block_start; blk < block_end; ++blk) {
        const half* tile = tc_value_dense + blk * 256;
        const int* bcol = tc_col_old + blk * 16;

        uint32_t a0, a1;
        if (ABL == 1) {
            // [消融 no-D] smem scatter 重建
            //   tile = 256 half = 128 uint32; 按 blockDim.x 循环覆盖
            {
                const uint32_t* src32 = reinterpret_cast<const uint32_t*>(tile);
                uint32_t* dst32 = reinterpret_cast<uint32_t*>(tile_smem);
                for (int idx = threadIdx.x; idx < 128; idx += blockDim.x)
                    dst32[idx] = __ldg(src32 + idx);
            }
            __syncthreads();
            a0 = *reinterpret_cast<const uint32_t*>(&tile_smem[groupID * 16 + tid_in_group * 2]);
            a1 = *reinterpret_cast<const uint32_t*>(&tile_smem[(groupID + 8) * 16 + tid_in_group * 2]);
            __syncthreads();
        } else {
            a0 = __ldg(reinterpret_cast<const uint32_t*>(tile + groupID * 16 + tid_in_group * 2));
            a1 = __ldg(reinterpret_cast<const uint32_t*>(tile + (groupID + 8) * 16 + tid_in_group * 2));
        }

        int gk0 = __ldg(bcol + tid_in_group * 2);
        int gk1 = __ldg(bcol + tid_in_group * 2 + 1);

        uint32_t RB[4];
        if (ABL == 3) {
            // [消融 no-H] B 标量逐列加载
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                int nc = dense_B_idx_base + j;
                half bv0 = __ldg(rhs_matrix + gk0 * dimN + nc);
                half bv1 = __ldg(rhs_matrix + gk1 * dimN + nc);
                RB[j] = pack_half2_u32(bv0, bv1);
            }
        } else if (valid4) {
            const uint64_t bytes0 = (uint64_t)(rhs_matrix + gk0 * dimN + dense_B_idx_base);
            uint64_t t0, t1;
            if ((bytes0 & 0x7) == 0) {
                t0 = __ldg(reinterpret_cast<const uint64_t*>(rhs_matrix + gk0 * dimN + dense_B_idx_base));
                t1 = __ldg(reinterpret_cast<const uint64_t*>(rhs_matrix + gk1 * dimN + dense_B_idx_base));
            } else {
                const uint32_t* e0 = reinterpret_cast<const uint32_t*>(rhs_matrix + gk0 * dimN + dense_B_idx_base);
                const uint32_t* e1 = reinterpret_cast<const uint32_t*>(rhs_matrix + gk1 * dimN + dense_B_idx_base);
                t0 = (uint64_t)__ldg(e0 + 1) << 32 | __ldg(e0);
                t1 = (uint64_t)__ldg(e1 + 1) << 32 | __ldg(e1);
            }

            // RB[j] = pack(B[gk0][base+j], B[gk1][base+j])
            const uint32_t lo0 = (uint32_t)t0, lo1 = (uint32_t)t1;
            const uint32_t hi0 = (uint32_t)(t0 >> 32), hi1 = (uint32_t)(t1 >> 32);
            RB[0] = (lo0 & 0xFFFFu) | (lo1 << 16);
            RB[1] = (lo0 >> 16) | (lo1 & 0xFFFF0000u);
            RB[2] = (hi0 & 0xFFFFu) | (hi1 << 16);
            RB[3] = (hi0 >> 16) | (hi1 & 0xFFFF0000u);
        } else {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                int nc = dense_B_idx_base + j;
                half bv0 = __ldg(rhs_matrix + gk0 * dimN + nc);
                half bv1 = __ldg(rhs_matrix + gk1 * dimN + nc);
                RB[j] = pack_half2_u32(bv0, bv1);
            }
        }

        // 4 次 mma (m16n8k8): 共享 A fragment, 每次用不同 N 列
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            asm volatile(
                "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};\n"
                : "+f"(RC[j * 4 + 0]), "+f"(RC[j * 4 + 1]),
                  "+f"(RC[j * 4 + 2]), "+f"(RC[j * 4 + 3])
                : "r"(a0), "r"(a1), "r"(RB[j]));
        }
    }

    // 写回 (与 k16 相同布局)
    int c = warp_feat_base + tid_in_group * 8;
    bool need_atomic = (__ldg(tc_chunk_atomic + chunk_id) != 0);

    if (need_atomic) {
        atomic_add_float2(&output_matrix[row0 * dimN + c + 0], RC[0], RC[4]);
        atomic_add_float2(&output_matrix[row0 * dimN + c + 2], RC[8], RC[12]);
        atomic_add_float2(&output_matrix[row0 * dimN + c + 4], RC[1], RC[5]);
        atomic_add_float2(&output_matrix[row0 * dimN + c + 6], RC[9], RC[13]);

        atomic_add_float2(&output_matrix[row1 * dimN + c + 0], RC[2], RC[6]);
        atomic_add_float2(&output_matrix[row1 * dimN + c + 2], RC[10], RC[14]);
        atomic_add_float2(&output_matrix[row1 * dimN + c + 4], RC[3], RC[7]);
        atomic_add_float2(&output_matrix[row1 * dimN + c + 6], RC[11], RC[15]);
    } else {
        *reinterpret_cast<float4*>(&output_matrix[row0 * dimN + c + 0]) =
            make_float4(RC[0], RC[4], RC[8], RC[12]);
        *reinterpret_cast<float4*>(&output_matrix[row0 * dimN + c + 4]) =
            make_float4(RC[1], RC[5], RC[9], RC[13]);

        *reinterpret_cast<float4*>(&output_matrix[row1 * dimN + c + 0]) =
            make_float4(RC[2], RC[6], RC[10], RC[14]);
        *reinterpret_cast<float4*>(&output_matrix[row1 * dimN + c + 4]) =
            make_float4(RC[3], RC[7], RC[11], RC[15]);
    }
}


// ============================================================================
// SPTC Kernel - A/meta 寄存器双缓冲 + 每 warp 32 列 (MP Buint64 布局) + chunking
//
// Grid:  (num_chunks, 1) 1D —— 窗口内分块, 每 CTA 覆盖全部 N 列
// Block: (32, warp_count, 1)  warp_count = ceil(dimN / 32), 每 warp 32 列 N
//
// [方案A] A/meta 寄存器双缓冲:
//   RA[4], RM[2], __ldg 预取
//
// [方案B] 每 CTA 覆盖全部 N (核心优化):
//   旧版 grid.y = feature_tiles, 每个 32 列 CTA 各加载同一批 A 块,
//   A 的 DRAM/L2 流量被重复 feature_tiles 次.
//   现在每 CTA 覆盖全部 N, 同一块 A 只加载 1 次, 由 16 次 mma.sp 复用.
//
// [方案C] B 列交错宽加载 (同 MP Buint64 版):
//   每线程持有连续 4 列 B (4×uint64 合并读), 4 次 mma.sp 各取 1 列,
//   消除标量逐列加载的 16 次独立内存事务.
//
// [方案D] 写回: float2 atomicAdd 到共享 output_matrix
//
// [方案F] 窗口内分块 (同 TC): SPTC 块在窗口间也不均 (mip1 最大 1441),
//   分块后每 CTA 至多处理 SPTC_CHUNK 块, 消除长尾 straggler.
// ============================================================================

__global__ void tr_sptc_kernel(
    const int* __restrict__ sptc_chunk_win,     // [num_chunks] 每 chunk 的窗口 id (诊断)
    const int* __restrict__ sptc_chunk_beg,     // [num_chunks] 块起始 (含)
    const int* __restrict__ sptc_chunk_end,     // [num_chunks] 块结束 (不含)
    const uint8_t* __restrict__ sptc_chunk_atomic, // [num_chunks] 1=写回需 atomicAdd
    const int* __restrict__ sptc_block_row,     // [num_sptc_blocks] 每块输出行首
    const half* __restrict__ sptc_value,
    const uint32_t* __restrict__ sptc_packed_meta,
    const int* __restrict__ sptc_col_old,
    const half* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix,          // 与 TC 共享, 用 atomicAdd
    int num_chunks,
    int dimN,
    int mOri,
    int kOri)
{
#if __CUDA_ARCH__ >= 800
    int chunk_id = blockIdx.x;
    if (chunk_id >= num_chunks) return;

    int window_id = __ldg(sptc_chunk_win + chunk_id);
    int block_start = __ldg(sptc_chunk_beg + chunk_id);
    int block_end   = __ldg(sptc_chunk_end + chunk_id);
    if (block_start >= block_end) return;

    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.y;          // 每 warp 覆盖 32 列 N
    int groupID = lane >> 2;            // 0..7
    int tid_in_group = lane & 3;        // 0..3

    int warp_feat_base = warp_id * 32;  // 该 warp 的 N 起始偏移
    if (warp_feat_base >= dimN) return;

    // CTA 级输出行 (整窗单 CTA: 所有块同窗口)
    int window_row = window_id * 16;
    int row0 = window_row + groupID;
    int row1 = row0 + 8;

    // A 值索引 (块内偏移, 固定)
    const int sparse_A_idx_0 = groupID * 8 + tid_in_group * 2;
    const int sparse_A_idx_1 = sparse_A_idx_0 + 64;

    // B 行索引 (块内 K 方向, 固定)
    const int row_b0 = tid_in_group * 2;
    const int row_b1 = row_b0 + 1;
    const int row_b2 = row_b0 + 8;
    const int row_b3 = row_b0 + 9;

    // B 连续 4 列基址 (MP Buint64 布局): 每线程 4 列 × 4 次 mma
    const int dense_B_idx_base = groupID * 4 + warp_feat_base;

    // [方案A] A/meta 寄存器双缓冲
    uint32_t RA[4];   // RA[cur*2], RA[cur*2+1], RA[next*2], RA[next*2+1]
    uint32_t RM[2];   // RM[cur], RM[next]

    // C 累加器: 4 次 mma × 每线程 4 float (RC[j*4 .. j*4+3])
    float RC[16] = {0.0f, 0.0f, 0.0f, 0.0f,
                    0.0f, 0.0f, 0.0f, 0.0f,
                    0.0f, 0.0f, 0.0f, 0.0f,
                    0.0f, 0.0f, 0.0f, 0.0f};

    bool need_atomic = (__ldg(sptc_chunk_atomic + chunk_id) != 0);

    // flush: 把整窗累积的 RC 写回 (MP 布局: 每线程 8 列 × 2 行)
    auto flush_rc = [&]() {
        int c = warp_feat_base + tid_in_group * 8;
        if (need_atomic) {
            atomic_add_float2(&output_matrix[row0 * dimN + c + 0], RC[0],  RC[4]);
            atomic_add_float2(&output_matrix[row0 * dimN + c + 2], RC[8],  RC[12]);
            atomic_add_float2(&output_matrix[row0 * dimN + c + 4], RC[1],  RC[5]);
            atomic_add_float2(&output_matrix[row0 * dimN + c + 6], RC[9],  RC[13]);
            atomic_add_float2(&output_matrix[row1 * dimN + c + 0], RC[2],  RC[6]);
            atomic_add_float2(&output_matrix[row1 * dimN + c + 2], RC[10], RC[14]);
            atomic_add_float2(&output_matrix[row1 * dimN + c + 4], RC[3],  RC[7]);
            atomic_add_float2(&output_matrix[row1 * dimN + c + 6], RC[11], RC[15]);
        } else {
            *reinterpret_cast<float4*>(&output_matrix[row0 * dimN + c + 0]) =
                make_float4(RC[0], RC[4], RC[8], RC[12]);
            *reinterpret_cast<float4*>(&output_matrix[row0 * dimN + c + 4]) =
                make_float4(RC[1], RC[5], RC[9], RC[13]);
            *reinterpret_cast<float4*>(&output_matrix[row1 * dimN + c + 0]) =
                make_float4(RC[2], RC[6], RC[10], RC[14]);
            *reinterpret_cast<float4*>(&output_matrix[row1 * dimN + c + 4]) =
                make_float4(RC[3], RC[7], RC[11], RC[15]);
        }
#pragma unroll
        for (int t = 0; t < 16; ++t) RC[t] = 0.0f;
    };

    // =========================================================================
    // Prologue: 预加载 block_start 的 A/meta 到 cur 槽 (槽 0)
    // =========================================================================
    {
        int a_offset = block_start * 128;
        RA[0] = __ldg(reinterpret_cast<const uint32_t*>(sptc_value + a_offset + sparse_A_idx_0));
        RA[1] = __ldg(reinterpret_cast<const uint32_t*>(sptc_value + a_offset + sparse_A_idx_1));
        RM[0] = __ldg(sptc_packed_meta + block_start * 8 + groupID);
    }
    __syncwarp();

    // =========================================================================
    // Main loop: i 从 block_start 到 block_end-1
    //   cur 槽 = (i - block_start) & 1  ← A 与 B 同属块 i
    //   next 槽 = cur ^ 1               ← 预取块 i+1 的 A/meta
    //   (整窗单 CTA: 所有块累加到同一 RC, 最后 flush 一次写回)
    // =========================================================================
    for (int i = block_start; i < block_end; ++i) {
        int cur  = (i - block_start) & 1;
        int next = cur ^ 1;
        bool has_next = (i + 1 < block_end);

        // [方案A] 预取下一块 A/meta 到 next 槽 (与当前块 mma 并行)
        if (has_next) {
            int a_offset = (i + 1) * 128;
            RA[next * 2]     = __ldg(reinterpret_cast<const uint32_t*>(sptc_value + a_offset + sparse_A_idx_0));
            RA[next * 2 + 1] = __ldg(reinterpret_cast<const uint32_t*>(sptc_value + a_offset + sparse_A_idx_1));
            RM[next] = __ldg(sptc_packed_meta + (i + 1) * 8 + groupID);
        }

        // [方案C] B 列交错宽加载: 每线程 4 行 × 连续 4 列 = 4×uint64
        //   旧版每列单独 __ldg (16 次独立事务); 现在合并为 4 次 8B 读
        const int* blk_col = sptc_col_old + i * 16;
        int gk0 = __ldg(blk_col + row_b0);
        int gk1 = __ldg(blk_col + row_b1);
        int gk2 = __ldg(blk_col + row_b2);
        int gk3 = __ldg(blk_col + row_b3);

        uint32_t RB[8];
        bool valid4 = (dense_B_idx_base + 3 < dimN);

        if (valid4) {
            // 快路径: 4 个 uint64 连续读 (8B 对齐要求: dimN % 4 == 0)
            //   col_old 由 CPU 端补齐为合法索引 (pad=0), 无需越界检查
            const uint64_t* src0 = reinterpret_cast<const uint64_t*>(rhs_matrix + gk0 * dimN + dense_B_idx_base);
            const uint64_t* src1 = reinterpret_cast<const uint64_t*>(rhs_matrix + gk1 * dimN + dense_B_idx_base);
            const uint64_t* src2 = reinterpret_cast<const uint64_t*>(rhs_matrix + gk2 * dimN + dense_B_idx_base);
            const uint64_t* src3 = reinterpret_cast<const uint64_t*>(rhs_matrix + gk3 * dimN + dense_B_idx_base);
            uint64_t t0 = src0[0], t1 = src1[0], t2 = src2[0], t3 = src3[0];

            // 重组: mma j 用 4 行的第 j 列 (低 half 在前)
            // RB[j*2+0] = pack(B[row_b0][base+j], B[row_b1][base+j])
            // RB[j*2+1] = pack(B[row_b2][base+j], B[row_b3][base+j])
            const uint32_t lo0 = (uint32_t)t0, lo1 = (uint32_t)t1;
            const uint32_t lo2 = (uint32_t)t2, lo3 = (uint32_t)t3;
            const uint32_t hi0 = (uint32_t)(t0 >> 32), hi1 = (uint32_t)(t1 >> 32);
            const uint32_t hi2 = (uint32_t)(t2 >> 32), hi3 = (uint32_t)(t3 >> 32);
            // uint64 内: 低 32bit = 列 +0/+1 两个 half, 高 32bit = 列 +2/+3
            RB[0] = (lo0 & 0xFFFFu) | (lo1 << 16);
            RB[1] = (lo2 & 0xFFFFu) | (lo3 << 16);
            RB[2] = (lo0 >> 16) | (lo1 & 0xFFFF0000u);
            RB[3] = (lo2 >> 16) | (lo3 & 0xFFFF0000u);
            RB[4] = (hi0 & 0xFFFFu) | (hi1 << 16);
            RB[5] = (hi2 & 0xFFFFu) | (hi3 << 16);
            RB[6] = (hi0 >> 16) | (hi1 & 0xFFFF0000u);
            RB[7] = (hi2 >> 16) | (hi3 & 0xFFFF0000u);
        } else {
            // 慢路径: 标量逐列读取 (边界/非法 gk)
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                int nc = dense_B_idx_base + j;

                half bv0 = __ldg(rhs_matrix + gk0 * dimN + nc);
                half bv1 = __ldg(rhs_matrix + gk1 * dimN + nc);
                half bv2 = __ldg(rhs_matrix + gk2 * dimN + nc);
                half bv3 = __ldg(rhs_matrix + gk3 * dimN + nc);
                RB[j * 2 + 0] = pack_half2_u32(bv0, bv1);
                RB[j * 2 + 1] = pack_half2_u32(bv2, bv3);
            }
        }

        __syncwarp();

        // 4 次 mma.sp: 每次用 4 行的 1 列 (共享 cur 槽的 A/meta)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            asm volatile(
                "mma.sp.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0, %1, %2, %3}, {%4, %5}, {%6, %7}, {%0, %1, %2, %3}, %8, 0x1;\n"
                : "+f"(RC[j * 4 + 0]), "+f"(RC[j * 4 + 1]),
                  "+f"(RC[j * 4 + 2]), "+f"(RC[j * 4 + 3])
                : "r"(RA[cur * 2]), "r"(RA[cur * 2 + 1]),
                  "r"(RB[j * 2 + 0]), "r"(RB[j * 2 + 1]),
                  "r"(RM[cur]));
        }
    }

    // 写回整窗累积结果 (一次 flush)
    flush_rc();
#endif
}


// ============================================================================
// SPTC 消融变体
//   - no-A: 无寄存器双缓冲 (A/meta 随用随载)
//   - no-B: 退回 grid.y=feature_tiles 布局 (每 CTA 32 列 N, A 重复加载)
//   - no-C: B 退回标量逐列加载 (16 次 half __ldg)
//   - no-F: 写回无条件 atomicAdd
// ============================================================================

__global__ void tr_sptc_kernel_noa(
    const int* __restrict__ sptc_chunk_win,
    const int* __restrict__ sptc_chunk_beg,
    const int* __restrict__ sptc_chunk_end,
    const uint8_t* __restrict__ sptc_chunk_atomic,
    const int* __restrict__ sptc_block_row,
    const half* __restrict__ sptc_value,
    const uint32_t* __restrict__ sptc_packed_meta,
    const int* __restrict__ sptc_col_old,
    const half* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix,
    int num_chunks,
    int dimN,
    int mOri,
    int kOri)
{
#if __CUDA_ARCH__ >= 800
    int chunk_id = blockIdx.x;
    if (chunk_id >= num_chunks) return;

    int window_id = __ldg(sptc_chunk_win + chunk_id);
    int block_start = __ldg(sptc_chunk_beg + chunk_id);
    int block_end   = __ldg(sptc_chunk_end + chunk_id);
    if (block_start >= block_end) return;

    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.y;
    int groupID = lane >> 2;
    int tid_in_group = lane & 3;

    int warp_feat_base = warp_id * 32;
    if (warp_feat_base >= dimN) return;

    int window_row = window_id * 16;
    int row0 = window_row + groupID;
    int row1 = row0 + 8;

    const int sparse_A_idx_0 = groupID * 8 + tid_in_group * 2;
    const int sparse_A_idx_1 = sparse_A_idx_0 + 64;
    const int row_b0 = tid_in_group * 2;
    const int row_b1 = row_b0 + 1;
    const int row_b2 = row_b0 + 8;
    const int row_b3 = row_b0 + 9;
    const int dense_B_idx_base = groupID * 4 + warp_feat_base;

    float RC[16] = {0.0f, 0.0f, 0.0f, 0.0f,
                    0.0f, 0.0f, 0.0f, 0.0f,
                    0.0f, 0.0f, 0.0f, 0.0f,
                    0.0f, 0.0f, 0.0f, 0.0f};

    bool need_atomic = (__ldg(sptc_chunk_atomic + chunk_id) != 0);

    auto flush_rc = [&]() {
        int c = warp_feat_base + tid_in_group * 8;
        if (need_atomic) {
            atomic_add_float2(&output_matrix[row0 * dimN + c + 0], RC[0],  RC[4]);
            atomic_add_float2(&output_matrix[row0 * dimN + c + 2], RC[8],  RC[12]);
            atomic_add_float2(&output_matrix[row0 * dimN + c + 4], RC[1],  RC[5]);
            atomic_add_float2(&output_matrix[row0 * dimN + c + 6], RC[9],  RC[13]);
            atomic_add_float2(&output_matrix[row1 * dimN + c + 0], RC[2],  RC[6]);
            atomic_add_float2(&output_matrix[row1 * dimN + c + 2], RC[10], RC[14]);
            atomic_add_float2(&output_matrix[row1 * dimN + c + 4], RC[3],  RC[7]);
            atomic_add_float2(&output_matrix[row1 * dimN + c + 6], RC[11], RC[15]);
        } else {
            *reinterpret_cast<float4*>(&output_matrix[row0 * dimN + c + 0]) =
                make_float4(RC[0], RC[4], RC[8], RC[12]);
            *reinterpret_cast<float4*>(&output_matrix[row0 * dimN + c + 4]) =
                make_float4(RC[1], RC[5], RC[9], RC[13]);
            *reinterpret_cast<float4*>(&output_matrix[row1 * dimN + c + 0]) =
                make_float4(RC[2], RC[6], RC[10], RC[14]);
            *reinterpret_cast<float4*>(&output_matrix[row1 * dimN + c + 4]) =
                make_float4(RC[3], RC[7], RC[11], RC[15]);
        }
#pragma unroll
        for (int t = 0; t < 16; ++t) RC[t] = 0.0f;
    };

    // [消融 no-A] A/meta 随用随载, 无预取槽
    for (int i = block_start; i < block_end; ++i) {
        int a_offset = i * 128;
        uint32_t a0 = __ldg(reinterpret_cast<const uint32_t*>(sptc_value + a_offset + sparse_A_idx_0));
        uint32_t a1 = __ldg(reinterpret_cast<const uint32_t*>(sptc_value + a_offset + sparse_A_idx_1));
        uint32_t meta = __ldg(sptc_packed_meta + i * 8 + groupID);

        const int* blk_col = sptc_col_old + i * 16;
        int gk0 = __ldg(blk_col + row_b0);
        int gk1 = __ldg(blk_col + row_b1);
        int gk2 = __ldg(blk_col + row_b2);
        int gk3 = __ldg(blk_col + row_b3);

        uint32_t RB[8];
        bool valid4 = (dense_B_idx_base + 3 < dimN);

        if (valid4) {
            const uint64_t bytes0 = (uint64_t)(rhs_matrix + gk0 * dimN + dense_B_idx_base);
            uint64_t t0, t1, t2, t3;
            if ((bytes0 & 0x7) == 0) {
                t0 = __ldg(reinterpret_cast<const uint64_t*>(rhs_matrix + gk0 * dimN + dense_B_idx_base));
                t1 = __ldg(reinterpret_cast<const uint64_t*>(rhs_matrix + gk1 * dimN + dense_B_idx_base));
                t2 = __ldg(reinterpret_cast<const uint64_t*>(rhs_matrix + gk2 * dimN + dense_B_idx_base));
                t3 = __ldg(reinterpret_cast<const uint64_t*>(rhs_matrix + gk3 * dimN + dense_B_idx_base));
            } else {
                const uint32_t* e0 = reinterpret_cast<const uint32_t*>(rhs_matrix + gk0 * dimN + dense_B_idx_base);
                const uint32_t* e1 = reinterpret_cast<const uint32_t*>(rhs_matrix + gk1 * dimN + dense_B_idx_base);
                const uint32_t* e2 = reinterpret_cast<const uint32_t*>(rhs_matrix + gk2 * dimN + dense_B_idx_base);
                const uint32_t* e3 = reinterpret_cast<const uint32_t*>(rhs_matrix + gk3 * dimN + dense_B_idx_base);
                t0 = (uint64_t)__ldg(e0 + 1) << 32 | __ldg(e0);
                t1 = (uint64_t)__ldg(e1 + 1) << 32 | __ldg(e1);
                t2 = (uint64_t)__ldg(e2 + 1) << 32 | __ldg(e2);
                t3 = (uint64_t)__ldg(e3 + 1) << 32 | __ldg(e3);
            }
            const uint32_t lo0 = (uint32_t)t0, lo1 = (uint32_t)t1;
            const uint32_t lo2 = (uint32_t)t2, lo3 = (uint32_t)t3;
            const uint32_t hi0 = (uint32_t)(t0 >> 32), hi1 = (uint32_t)(t1 >> 32);
            const uint32_t hi2 = (uint32_t)(t2 >> 32), hi3 = (uint32_t)(t3 >> 32);
            RB[0] = (lo0 & 0xFFFFu) | (lo1 << 16);
            RB[1] = (lo2 & 0xFFFFu) | (lo3 << 16);
            RB[2] = (lo0 >> 16) | (lo1 & 0xFFFF0000u);
            RB[3] = (lo2 >> 16) | (lo3 & 0xFFFF0000u);
            RB[4] = (hi0 & 0xFFFFu) | (hi1 << 16);
            RB[5] = (hi2 & 0xFFFFu) | (hi3 << 16);
            RB[6] = (hi0 >> 16) | (hi1 & 0xFFFF0000u);
            RB[7] = (hi2 >> 16) | (hi3 & 0xFFFF0000u);
        } else {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                int nc = dense_B_idx_base + j;
                half bv0 = __ldg(rhs_matrix + gk0 * dimN + nc);
                half bv1 = __ldg(rhs_matrix + gk1 * dimN + nc);
                half bv2 = __ldg(rhs_matrix + gk2 * dimN + nc);
                half bv3 = __ldg(rhs_matrix + gk3 * dimN + nc);
                RB[j * 2 + 0] = pack_half2_u32(bv0, bv1);
                RB[j * 2 + 1] = pack_half2_u32(bv2, bv3);
            }
        }

        __syncwarp();

#pragma unroll
        for (int j = 0; j < 4; ++j) {
            asm volatile(
                "mma.sp.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0, %1, %2, %3}, {%4, %5}, {%6, %7}, {%0, %1, %2, %3}, %8, 0x1;\n"
                : "+f"(RC[j * 4 + 0]), "+f"(RC[j * 4 + 1]),
                  "+f"(RC[j * 4 + 2]), "+f"(RC[j * 4 + 3])
                : "r"(a0), "r"(a1),
                  "r"(RB[j * 2 + 0]), "r"(RB[j * 2 + 1]),
                  "r"(meta));
        }
    }

    flush_rc();
#endif
}


// [消融 no-B] 退回旧版 grid.y=feature_tiles 布局: 每 CTA 只覆盖 32 列 N,
//   同一窗口的 A 块被独立加载 feature_tiles 次.
__global__ void tr_sptc_kernel_nob(
    const int* __restrict__ sptc_chunk_win,
    const int* __restrict__ sptc_chunk_beg,
    const int* __restrict__ sptc_chunk_end,
    const uint8_t* __restrict__ sptc_chunk_atomic,
    const int* __restrict__ sptc_block_row,
    const half* __restrict__ sptc_value,
    const uint32_t* __restrict__ sptc_packed_meta,
    const int* __restrict__ sptc_col_old,
    const half* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix,
    int num_chunks,
    int dimN,
    int mOri,
    int kOri)
{
#if __CUDA_ARCH__ >= 800
    int chunk_id = blockIdx.x;
    int feature_tile_id = blockIdx.y;
    if (feature_tile_id >= (dimN + 31) / 32) return;
    if (chunk_id >= num_chunks) return;

    int window_id = __ldg(sptc_chunk_win + chunk_id);
    int block_start = __ldg(sptc_chunk_beg + chunk_id);
    int block_end   = __ldg(sptc_chunk_end + chunk_id);
    if (block_start >= block_end) return;

    int lane = threadIdx.x & 31;
    int warp_id = 0;    // 单 warp: 只负责本 feature tile 的 32 列
    int groupID = lane >> 2;
    int tid_in_group = lane & 3;

    int warp_feat_base = feature_tile_id * 32;
    if (warp_feat_base >= dimN) return;

    int window_row = window_id * 16;
    int row0 = window_row + groupID;
    int row1 = row0 + 8;

    const int sparse_A_idx_0 = groupID * 8 + tid_in_group * 2;
    const int sparse_A_idx_1 = sparse_A_idx_0 + 64;
    const int row_b0 = tid_in_group * 2;
    const int row_b1 = row_b0 + 1;
    const int row_b2 = row_b0 + 8;
    const int row_b3 = row_b0 + 9;
    const int dense_B_idx_base = groupID * 4 + warp_feat_base;

    uint32_t RA[4];
    uint32_t RM[2];

    float RC[16] = {0.0f, 0.0f, 0.0f, 0.0f,
                    0.0f, 0.0f, 0.0f, 0.0f,
                    0.0f, 0.0f, 0.0f, 0.0f,
                    0.0f, 0.0f, 0.0f, 0.0f};

    bool need_atomic = (__ldg(sptc_chunk_atomic + chunk_id) != 0);

    auto flush_rc = [&]() {
        int c = warp_feat_base + tid_in_group * 8;
        if (need_atomic) {
            atomic_add_float2(&output_matrix[row0 * dimN + c + 0], RC[0],  RC[4]);
            atomic_add_float2(&output_matrix[row0 * dimN + c + 2], RC[8],  RC[12]);
            atomic_add_float2(&output_matrix[row0 * dimN + c + 4], RC[1],  RC[5]);
            atomic_add_float2(&output_matrix[row0 * dimN + c + 6], RC[9],  RC[13]);
            atomic_add_float2(&output_matrix[row1 * dimN + c + 0], RC[2],  RC[6]);
            atomic_add_float2(&output_matrix[row1 * dimN + c + 2], RC[10], RC[14]);
            atomic_add_float2(&output_matrix[row1 * dimN + c + 4], RC[3],  RC[7]);
            atomic_add_float2(&output_matrix[row1 * dimN + c + 6], RC[11], RC[15]);
        } else {
            *reinterpret_cast<float4*>(&output_matrix[row0 * dimN + c + 0]) =
                make_float4(RC[0], RC[4], RC[8], RC[12]);
            *reinterpret_cast<float4*>(&output_matrix[row0 * dimN + c + 4]) =
                make_float4(RC[1], RC[5], RC[9], RC[13]);
            *reinterpret_cast<float4*>(&output_matrix[row1 * dimN + c + 0]) =
                make_float4(RC[2], RC[6], RC[10], RC[14]);
            *reinterpret_cast<float4*>(&output_matrix[row1 * dimN + c + 4]) =
                make_float4(RC[3], RC[7], RC[11], RC[15]);
        }
#pragma unroll
        for (int t = 0; t < 16; ++t) RC[t] = 0.0f;
    };

    {
        int a_offset = block_start * 128;
        RA[0] = __ldg(reinterpret_cast<const uint32_t*>(sptc_value + a_offset + sparse_A_idx_0));
        RA[1] = __ldg(reinterpret_cast<const uint32_t*>(sptc_value + a_offset + sparse_A_idx_1));
        RM[0] = __ldg(sptc_packed_meta + block_start * 8 + groupID);
    }
    __syncwarp();

    for (int i = block_start; i < block_end; ++i) {
        int cur  = (i - block_start) & 1;
        int next = cur ^ 1;
        bool has_next = (i + 1 < block_end);

        if (has_next) {
            int a_offset = (i + 1) * 128;
            RA[next * 2]     = __ldg(reinterpret_cast<const uint32_t*>(sptc_value + a_offset + sparse_A_idx_0));
            RA[next * 2 + 1] = __ldg(reinterpret_cast<const uint32_t*>(sptc_value + a_offset + sparse_A_idx_1));
            RM[next] = __ldg(sptc_packed_meta + (i + 1) * 8 + groupID);
        }

        const int* blk_col = sptc_col_old + i * 16;
        int gk0 = __ldg(blk_col + row_b0);
        int gk1 = __ldg(blk_col + row_b1);
        int gk2 = __ldg(blk_col + row_b2);
        int gk3 = __ldg(blk_col + row_b3);

        uint32_t RB[8];
        bool valid4 = (dense_B_idx_base + 3 < dimN);

        if (valid4) {
            const uint64_t bytes0 = (uint64_t)(rhs_matrix + gk0 * dimN + dense_B_idx_base);
            uint64_t t0, t1, t2, t3;
            if ((bytes0 & 0x7) == 0) {
                t0 = __ldg(reinterpret_cast<const uint64_t*>(rhs_matrix + gk0 * dimN + dense_B_idx_base));
                t1 = __ldg(reinterpret_cast<const uint64_t*>(rhs_matrix + gk1 * dimN + dense_B_idx_base));
                t2 = __ldg(reinterpret_cast<const uint64_t*>(rhs_matrix + gk2 * dimN + dense_B_idx_base));
                t3 = __ldg(reinterpret_cast<const uint64_t*>(rhs_matrix + gk3 * dimN + dense_B_idx_base));
            } else {
                const uint32_t* e0 = reinterpret_cast<const uint32_t*>(rhs_matrix + gk0 * dimN + dense_B_idx_base);
                const uint32_t* e1 = reinterpret_cast<const uint32_t*>(rhs_matrix + gk1 * dimN + dense_B_idx_base);
                const uint32_t* e2 = reinterpret_cast<const uint32_t*>(rhs_matrix + gk2 * dimN + dense_B_idx_base);
                const uint32_t* e3 = reinterpret_cast<const uint32_t*>(rhs_matrix + gk3 * dimN + dense_B_idx_base);
                t0 = (uint64_t)__ldg(e0 + 1) << 32 | __ldg(e0);
                t1 = (uint64_t)__ldg(e1 + 1) << 32 | __ldg(e1);
                t2 = (uint64_t)__ldg(e2 + 1) << 32 | __ldg(e2);
                t3 = (uint64_t)__ldg(e3 + 1) << 32 | __ldg(e3);
            }
            const uint32_t lo0 = (uint32_t)t0, lo1 = (uint32_t)t1;
            const uint32_t lo2 = (uint32_t)t2, lo3 = (uint32_t)t3;
            const uint32_t hi0 = (uint32_t)(t0 >> 32), hi1 = (uint32_t)(t1 >> 32);
            const uint32_t hi2 = (uint32_t)(t2 >> 32), hi3 = (uint32_t)(t3 >> 32);
            RB[0] = (lo0 & 0xFFFFu) | (lo1 << 16);
            RB[1] = (lo2 & 0xFFFFu) | (lo3 << 16);
            RB[2] = (lo0 >> 16) | (lo1 & 0xFFFF0000u);
            RB[3] = (lo2 >> 16) | (lo3 & 0xFFFF0000u);
            RB[4] = (hi0 & 0xFFFFu) | (hi1 << 16);
            RB[5] = (hi2 & 0xFFFFu) | (hi3 << 16);
            RB[6] = (hi0 >> 16) | (hi1 & 0xFFFF0000u);
            RB[7] = (hi2 >> 16) | (hi3 & 0xFFFF0000u);
        } else {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                int nc = dense_B_idx_base + j;
                half bv0 = __ldg(rhs_matrix + gk0 * dimN + nc);
                half bv1 = __ldg(rhs_matrix + gk1 * dimN + nc);
                half bv2 = __ldg(rhs_matrix + gk2 * dimN + nc);
                half bv3 = __ldg(rhs_matrix + gk3 * dimN + nc);
                RB[j * 2 + 0] = pack_half2_u32(bv0, bv1);
                RB[j * 2 + 1] = pack_half2_u32(bv2, bv3);
            }
        }

        __syncwarp();

#pragma unroll
        for (int j = 0; j < 4; ++j) {
            asm volatile(
                "mma.sp.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0, %1, %2, %3}, {%4, %5}, {%6, %7}, {%0, %1, %2, %3}, %8, 0x1;\n"
                : "+f"(RC[j * 4 + 0]), "+f"(RC[j * 4 + 1]),
                  "+f"(RC[j * 4 + 2]), "+f"(RC[j * 4 + 3])
                : "r"(RA[cur * 2]), "r"(RA[cur * 2 + 1]),
                  "r"(RB[j * 2 + 0]), "r"(RB[j * 2 + 1]),
                  "r"(RM[cur]));
        }
    }

    flush_rc();
#endif
}


// [消融 no-C] B 退回标量逐列加载 (其余同 baseline 双缓冲 + 全 N)
__global__ void tr_sptc_kernel_noc(
    const int* __restrict__ sptc_chunk_win,
    const int* __restrict__ sptc_chunk_beg,
    const int* __restrict__ sptc_chunk_end,
    const uint8_t* __restrict__ sptc_chunk_atomic,
    const int* __restrict__ sptc_block_row,
    const half* __restrict__ sptc_value,
    const uint32_t* __restrict__ sptc_packed_meta,
    const int* __restrict__ sptc_col_old,
    const half* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix,
    int num_chunks,
    int dimN,
    int mOri,
    int kOri)
{
#if __CUDA_ARCH__ >= 800
    int chunk_id = blockIdx.x;
    if (chunk_id >= num_chunks) return;

    int window_id = __ldg(sptc_chunk_win + chunk_id);
    int block_start = __ldg(sptc_chunk_beg + chunk_id);
    int block_end   = __ldg(sptc_chunk_end + chunk_id);
    if (block_start >= block_end) return;

    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.y;
    int groupID = lane >> 2;
    int tid_in_group = lane & 3;

    int warp_feat_base = warp_id * 32;
    if (warp_feat_base >= dimN) return;

    int window_row = window_id * 16;
    int row0 = window_row + groupID;
    int row1 = row0 + 8;

    const int sparse_A_idx_0 = groupID * 8 + tid_in_group * 2;
    const int sparse_A_idx_1 = sparse_A_idx_0 + 64;
    const int row_b0 = tid_in_group * 2;
    const int row_b1 = row_b0 + 1;
    const int row_b2 = row_b0 + 8;
    const int row_b3 = row_b0 + 9;
    const int dense_B_idx_base = groupID * 4 + warp_feat_base;

    uint32_t RA[4];
    uint32_t RM[2];

    float RC[16] = {0.0f, 0.0f, 0.0f, 0.0f,
                    0.0f, 0.0f, 0.0f, 0.0f,
                    0.0f, 0.0f, 0.0f, 0.0f,
                    0.0f, 0.0f, 0.0f, 0.0f};

    bool need_atomic = (__ldg(sptc_chunk_atomic + chunk_id) != 0);

    auto flush_rc = [&]() {
        int c = warp_feat_base + tid_in_group * 8;
        if (need_atomic) {
            atomic_add_float2(&output_matrix[row0 * dimN + c + 0], RC[0],  RC[4]);
            atomic_add_float2(&output_matrix[row0 * dimN + c + 2], RC[8],  RC[12]);
            atomic_add_float2(&output_matrix[row0 * dimN + c + 4], RC[1],  RC[5]);
            atomic_add_float2(&output_matrix[row0 * dimN + c + 6], RC[9],  RC[13]);
            atomic_add_float2(&output_matrix[row1 * dimN + c + 0], RC[2],  RC[6]);
            atomic_add_float2(&output_matrix[row1 * dimN + c + 2], RC[10], RC[14]);
            atomic_add_float2(&output_matrix[row1 * dimN + c + 4], RC[3],  RC[7]);
            atomic_add_float2(&output_matrix[row1 * dimN + c + 6], RC[11], RC[15]);
        } else {
            *reinterpret_cast<float4*>(&output_matrix[row0 * dimN + c + 0]) =
                make_float4(RC[0], RC[4], RC[8], RC[12]);
            *reinterpret_cast<float4*>(&output_matrix[row0 * dimN + c + 4]) =
                make_float4(RC[1], RC[5], RC[9], RC[13]);
            *reinterpret_cast<float4*>(&output_matrix[row1 * dimN + c + 0]) =
                make_float4(RC[2], RC[6], RC[10], RC[14]);
            *reinterpret_cast<float4*>(&output_matrix[row1 * dimN + c + 4]) =
                make_float4(RC[3], RC[7], RC[11], RC[15]);
        }
#pragma unroll
        for (int t = 0; t < 16; ++t) RC[t] = 0.0f;
    };

    {
        int a_offset = block_start * 128;
        RA[0] = __ldg(reinterpret_cast<const uint32_t*>(sptc_value + a_offset + sparse_A_idx_0));
        RA[1] = __ldg(reinterpret_cast<const uint32_t*>(sptc_value + a_offset + sparse_A_idx_1));
        RM[0] = __ldg(sptc_packed_meta + block_start * 8 + groupID);
    }
    __syncwarp();

    for (int i = block_start; i < block_end; ++i) {
        int cur  = (i - block_start) & 1;
        int next = cur ^ 1;
        bool has_next = (i + 1 < block_end);

        if (has_next) {
            int a_offset = (i + 1) * 128;
            RA[next * 2]     = __ldg(reinterpret_cast<const uint32_t*>(sptc_value + a_offset + sparse_A_idx_0));
            RA[next * 2 + 1] = __ldg(reinterpret_cast<const uint32_t*>(sptc_value + a_offset + sparse_A_idx_1));
            RM[next] = __ldg(sptc_packed_meta + (i + 1) * 8 + groupID);
        }

        const int* blk_col = sptc_col_old + i * 16;
        int gk0 = __ldg(blk_col + row_b0);
        int gk1 = __ldg(blk_col + row_b1);
        int gk2 = __ldg(blk_col + row_b2);
        int gk3 = __ldg(blk_col + row_b3);

        // [消融 no-C] B 标量逐列加载
        uint32_t RB[8];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            int nc = dense_B_idx_base + j;
            half bv0 = __ldg(rhs_matrix + gk0 * dimN + nc);
            half bv1 = __ldg(rhs_matrix + gk1 * dimN + nc);
            half bv2 = __ldg(rhs_matrix + gk2 * dimN + nc);
            half bv3 = __ldg(rhs_matrix + gk3 * dimN + nc);
            RB[j * 2 + 0] = pack_half2_u32(bv0, bv1);
            RB[j * 2 + 1] = pack_half2_u32(bv2, bv3);
        }

        __syncwarp();

#pragma unroll
        for (int j = 0; j < 4; ++j) {
            asm volatile(
                "mma.sp.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0, %1, %2, %3}, {%4, %5}, {%6, %7}, {%0, %1, %2, %3}, %8, 0x1;\n"
                : "+f"(RC[j * 4 + 0]), "+f"(RC[j * 4 + 1]),
                  "+f"(RC[j * 4 + 2]), "+f"(RC[j * 4 + 3])
                : "r"(RA[cur * 2]), "r"(RA[cur * 2 + 1]),
                  "r"(RB[j * 2 + 0]), "r"(RB[j * 2 + 1]),
                  "r"(RM[cur]));
        }
    }

    flush_rc();
#endif
}


// ============================================================================
// 辅助函数: TC kernel launch
// ============================================================================

static void launch_tc_kernel(
    const int* tc_chunk_win,
    const int* tc_chunk_beg,
    const int* tc_chunk_end,
    const uint8_t* tc_chunk_atomic,
    const half* tc_value_dense,          // [num_tc_blocks * 256] 16×16 tiles
    const int* tc_col_old,               // [num_tc_blocks * 16] (0 pad)
    const half* rhs_matrix,
    float* output_matrix,
    int num_chunks,
    int num_tc_blocks,
    int dense_threshold,
    int dimN,
    int mOri,
    int kOri,
    int ablation,
    cudaStream_t stream)
{
    if (num_chunks <= 0 || num_tc_blocks <= 0 || dimN <= 0 || mOri <= 0) return;

    // [G] 每 CTA 覆盖全部 N: block = (32, warp_count)
    // [消融 no-G] 退回 grid.y=feature_tiles, block=(32,1)
    int warp_count = (dimN + 31) / 32;
    bool nog = (ablation == ABLATION_NO_G);
    dim3 block_dim(32, nog ? 1 : warp_count, 1);
    dim3 grid_dim(num_chunks, nog ? warp_count : 1, 1);

    // [消融 no-I] 强制 8×m16n8k8 (即使 dense_threshold>8)
    bool noi = (ablation == ABLATION_NO_I);

    // [消融 no-F] 写回全部 atomicAdd 由内核内 ablation 分支处理 (共用 baseline 内核,
    //   通过模板/宏不行 → no-F 用单独内核; TC 侧与 SPTC 侧同 flag)
    if (dense_threshold > 8) {
        if (ablation == ABLATION_NO_H) {
            tr_tc_kernel_k16_noh<<<grid_dim, block_dim, 0, stream>>>(
                tc_chunk_win, tc_chunk_beg, tc_chunk_end, tc_chunk_atomic,
                tc_value_dense, tc_col_old, rhs_matrix, output_matrix,
                num_chunks, dimN);
        } else if (ablation == ABLATION_NO_D) {
            tr_tc_kernel_k16_nod<<<grid_dim, block_dim, 0, stream>>>(
                tc_chunk_win, tc_chunk_beg, tc_chunk_end, tc_chunk_atomic,
                tc_value_dense, tc_col_old, rhs_matrix, output_matrix,
                num_chunks, dimN);
        } else if (nog) {
            tr_tc_kernel_k16_nog<<<grid_dim, block_dim, 0, stream>>>(
                tc_chunk_win, tc_chunk_beg, tc_chunk_end, tc_chunk_atomic,
                tc_value_dense, tc_col_old, rhs_matrix, output_matrix,
                num_chunks, dimN);
        } else if (noi) {
            tr_tc_kernel_k16_noi<<<grid_dim, block_dim, 0, stream>>>(
                tc_chunk_win, tc_chunk_beg, tc_chunk_end, tc_chunk_atomic,
                tc_value_dense, tc_col_old, rhs_matrix, output_matrix,
                num_chunks, dimN);
        } else {
            tr_tc_kernel_k16<<<grid_dim, block_dim, 0, stream>>>(
                tc_chunk_win, tc_chunk_beg, tc_chunk_end, tc_chunk_atomic,
                tc_value_dense, tc_col_old, rhs_matrix, output_matrix,
                num_chunks, dimN);
        }
    } else {
        // dense_threshold<=8: k8 模板变体覆盖 baseline / no-D / no-G / no-H
        //   (no-I 对 k8 无意义: K 本来就是 8, k8 baseline 即为 no-I 语义)
        const int warp_eff = nog ? 1 : warp_count;
        dim3 block_dim8(32, warp_eff, 1);
        dim3 grid_dim8(num_chunks, nog ? warp_count : 1, 1);
        if (ablation == ABLATION_NO_D) {
            tr_tc_kernel_k8_abl<1><<<grid_dim8, block_dim8, 0, stream>>>(
                tc_chunk_win, tc_chunk_beg, tc_chunk_end, tc_chunk_atomic,
                tc_value_dense, tc_col_old, rhs_matrix, output_matrix,
                num_chunks, dimN);
        } else if (nog) {
            tr_tc_kernel_k8_abl<2><<<grid_dim8, block_dim8, 0, stream>>>(
                tc_chunk_win, tc_chunk_beg, tc_chunk_end, tc_chunk_atomic,
                tc_value_dense, tc_col_old, rhs_matrix, output_matrix,
                num_chunks, dimN);
        } else if (ablation == ABLATION_NO_H) {
            tr_tc_kernel_k8_abl<3><<<grid_dim8, block_dim8, 0, stream>>>(
                tc_chunk_win, tc_chunk_beg, tc_chunk_end, tc_chunk_atomic,
                tc_value_dense, tc_col_old, rhs_matrix, output_matrix,
                num_chunks, dimN);
        } else {
            tr_tc_kernel_k8_abl<0><<<grid_dim8, block_dim8, 0, stream>>>(
                tc_chunk_win, tc_chunk_beg, tc_chunk_end, tc_chunk_atomic,
                tc_value_dense, tc_col_old, rhs_matrix, output_matrix,
                num_chunks, dimN);
        }
    }
}


// ============================================================================
// 辅助函数: SPTC kernel launch
// ============================================================================

static void launch_sptc_kernel(
    const int* sptc_chunk_win,
    const int* sptc_chunk_beg,
    const int* sptc_chunk_end,
    const uint8_t* sptc_chunk_atomic,
    const int* sptc_block_row,
    const half* sptc_value,
    const uint32_t* sptc_packed_meta,
    const int* sptc_col_old,
    const half* rhs_matrix,
    float* output_matrix,
    int num_chunks,
    int num_sptc_blocks,
    int dimN,
    int mOri,
    int kOri,
    int ablation,
    cudaStream_t stream)
{
    if (num_chunks <= 0 || num_sptc_blocks <= 0 || dimN <= 0 || mOri <= 0) return;

    // [B] 每 CTA 覆盖全部 N: block = (32, warp_count)
    // [消融 no-B] 退回 grid.y=feature_tiles, block=(32,1)
    int warp_count = (dimN + 31) / 32;
    bool nob = (ablation == ABLATION_NO_B);
    dim3 block_dim(32, nob ? 1 : warp_count, 1);
    dim3 grid_dim(num_chunks, nob ? warp_count : 1, 1);

    if (ablation == ABLATION_NO_A) {
        tr_sptc_kernel_noa<<<grid_dim, block_dim, 0, stream>>>(
            sptc_chunk_win, sptc_chunk_beg, sptc_chunk_end, sptc_chunk_atomic,
            sptc_block_row, sptc_value, sptc_packed_meta, sptc_col_old,
            rhs_matrix, output_matrix, num_chunks, dimN, mOri, kOri);
    } else if (nob) {
        tr_sptc_kernel_nob<<<grid_dim, block_dim, 0, stream>>>(
            sptc_chunk_win, sptc_chunk_beg, sptc_chunk_end, sptc_chunk_atomic,
            sptc_block_row, sptc_value, sptc_packed_meta, sptc_col_old,
            rhs_matrix, output_matrix, num_chunks, dimN, mOri, kOri);
    } else if (ablation == ABLATION_NO_C) {
        tr_sptc_kernel_noc<<<grid_dim, block_dim, 0, stream>>>(
            sptc_chunk_win, sptc_chunk_beg, sptc_chunk_end, sptc_chunk_atomic,
            sptc_block_row, sptc_value, sptc_packed_meta, sptc_col_old,
            rhs_matrix, output_matrix, num_chunks, dimN, mOri, kOri);
    } else {
        // 无动态 smem: 内核为纯寄存器实现, 不传 smem 以保持最大 occupancy
        tr_sptc_kernel<<<grid_dim, block_dim, 0, stream>>>(
            sptc_chunk_win, sptc_chunk_beg, sptc_chunk_end, sptc_chunk_atomic,
            sptc_block_row, sptc_value, sptc_packed_meta, sptc_col_old,
            rhs_matrix, output_matrix, num_chunks, dimN, mOri, kOri);
    }
}


// ============================================================================
// 主入口: TC + SPTC 双流并行
//
// [C] 合并输出: TC/SPTC 共用 d_output, 各自用 atomicAdd 写回.
//     去掉了原来的 add_matrix_kernel 全矩阵合并 pass.
// ============================================================================

extern "C" float tr_spmm_forward(
    // SPTC GPU pointers
    int* d_sptc_chunk_win,             // [num_sptc_chunks] 每 chunk 的窗口 id
    int* d_sptc_chunk_beg,             // [num_sptc_chunks] 块起始
    int* d_sptc_chunk_end,             // [num_sptc_chunks] 块结束
    uint8_t* d_sptc_chunk_atomic,      // [num_sptc_chunks] 1=写回需 atomicAdd
    int* d_sptc_block_row,             // [num_sptc_blocks] 每块输出行首
    half* d_sptc_value,
    uint32_t* d_sptc_packed_meta,
    int* d_sptc_col_old,
    // TC GPU pointers
    int* d_tc_chunk_win,               // [num_chunks] 每 chunk 的窗口 id
    int* d_tc_chunk_beg,               // [num_chunks] 块起始
    int* d_tc_chunk_end,               // [num_chunks] 块结束
    uint8_t* d_tc_chunk_atomic,        // [num_chunks] 1=写回需 atomicAdd
    half* d_tc_value_dense,            // [num_tc_blocks * 256] 16×16 稠密 tiles
    int* d_tc_col_old,                 // [num_tc_blocks * 16] (0 pad)
    // B matrix (shared)
    half* d_rhs_matrix,
    // Output - 单块共享 (原来的 d_output_tc)
    float* d_output,
    // Dimensions
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
    int mode,   // 0=Full, 1=TC-only, 2=SPTC-only (诊断模式)
    int warmup,
    int ablation)
{
    long long elements = (long long)mOri * dimN;
    cudaMemset(d_output, 0, (size_t)elements * sizeof(float));
    if (epoches <= 0 || dimN <= 0 || mOri <= 0) return 0.0f;

    // ---- 创建双流 (消融 no-AB: 单流串行) ----
    cudaStream_t stream_tc, stream_sptc;
    if (ablation == ABLATION_NO_STREAMS) {
        stream_tc = 0;                       // 默认流
        stream_sptc = 0;
    } else {
        cudaStreamCreateWithFlags(&stream_tc, cudaStreamNonBlocking);
        cudaStreamCreateWithFlags(&stream_sptc, cudaStreamNonBlocking);
    }

    // ---- Warmup ----
    for (int iter = 0; iter < warmup; ++iter) {
        cudaMemsetAsync(d_output, 0, (size_t)elements * sizeof(float), stream_tc);
        cudaStreamSynchronize(stream_tc);

        // mode: 0=Full, 1=TC-only, 2=SPTC-only
        if (mode != 2) {
            launch_tc_kernel(
                d_tc_chunk_win, d_tc_chunk_beg, d_tc_chunk_end, d_tc_chunk_atomic,
                d_tc_value_dense, d_tc_col_old,
                d_rhs_matrix, d_output,
                num_tc_chunks, num_tc_blocks, dense_threshold,
                dimN, mOri, kOri, ablation, stream_tc);
        }
        if (mode != 1) {
            launch_sptc_kernel(
                d_sptc_chunk_win, d_sptc_chunk_beg, d_sptc_chunk_end, d_sptc_chunk_atomic,
                d_sptc_block_row,
                d_sptc_value, d_sptc_packed_meta,
                d_sptc_col_old,
                d_rhs_matrix, d_output,
                num_sptc_chunks, num_sptc_blocks,
                dimN, mOri, kOri, ablation, stream_sptc);
        }

        cudaStreamSynchronize(stream_tc);
        cudaStreamSynchronize(stream_sptc);
    }
    cudaDeviceSynchronize();
    {
        cudaError_t warmup_err = cudaGetLastError();
        if (warmup_err != cudaSuccess) {
            printf("TR SPMM warmup GPU error: %s\n", cudaGetErrorString(warmup_err));
        }
    }

    // ---- 计时 Benchmark (含双流均衡诊断) ----
    // 注: 计时循环内不做 memset, 保持 kernel-only 测量 (与原始 benchmark 一致).
    //     输出 buffer 在循环前清零一次; 迭代间原子累加不影响计时 (每迭代结果相同),
    //     最终结果由后面的"干净运行"重新计算.
    cudaMemset(d_output, 0, (size_t)elements * sizeof(float));
    cudaDeviceSynchronize();

    float spmm_ms = 0.0f;
    cudaEvent_t spmm_start, spmm_end;
    cudaEvent_t tc_event_start, tc_event_end;
    cudaEvent_t sptc_event_start, sptc_event_end;
    cudaEventCreate(&spmm_start);
    cudaEventCreate(&spmm_end);
    cudaEventCreate(&tc_event_start);
    cudaEventCreate(&tc_event_end);
    cudaEventCreate(&sptc_event_start);
    cudaEventCreate(&sptc_event_end);

    cudaEventRecord(spmm_start, stream_tc);
    cudaEventRecord(tc_event_start, stream_tc);
    cudaEventRecord(sptc_event_start, stream_sptc);
    for (int iter = 0; iter < epoches; ++iter) {
        // 计时循环内不做 memset/事件同步: 两条流完全独立并发执行, 测量纯 kernel 时间.
        // mode: 0=Full, 1=TC-only, 2=SPTC-only
        if (mode != 2) {
            launch_tc_kernel(
                d_tc_chunk_win, d_tc_chunk_beg, d_tc_chunk_end, d_tc_chunk_atomic,
                d_tc_value_dense, d_tc_col_old,
                d_rhs_matrix, d_output,
                num_tc_chunks, num_tc_blocks, dense_threshold,
                dimN, mOri, kOri, ablation, stream_tc);
        }
        if (mode != 1) {
            launch_sptc_kernel(
                d_sptc_chunk_win, d_sptc_chunk_beg, d_sptc_chunk_end, d_sptc_chunk_atomic,
                d_sptc_block_row,
                d_sptc_value, d_sptc_packed_meta,
                d_sptc_col_old,
                d_rhs_matrix, d_output,
                num_sptc_chunks, num_sptc_blocks,
                dimN, mOri, kOri, ablation, stream_sptc);
        }
    }

    // 分别在两条流上记录结束时刻
    cudaEventRecord(tc_event_end, stream_tc);
    cudaEventRecord(sptc_event_end, stream_sptc);

    cudaStreamSynchronize(stream_tc);
    cudaStreamSynchronize(stream_sptc);
    cudaEventRecord(spmm_end, stream_tc);
    cudaEventSynchronize(spmm_end);
    cudaEventElapsedTime(&spmm_ms, spmm_start, spmm_end);
    float spmm_ms_avg = spmm_ms / (float)epoches;

    // ---- 双流均衡诊断输出 ----
    {
        float tc_ms = 0.0f, sptc_ms = 0.0f;
        cudaEventElapsedTime(&tc_ms, tc_event_start, tc_event_end);
        cudaEventElapsedTime(&sptc_ms, sptc_event_start, sptc_event_end);
        float tc_avg = tc_ms / (float)epoches;
        float sptc_avg = sptc_ms / (float)epoches;
        const char* mode_name = (mode == 0) ? "Full" :
                                (mode == 1) ? "TC-only" : "SPTC-only";
        const char* ablation_name =
            (ablation == ABLATION_NO_A)       ? "no-A(no-sptc-db)" :
            (ablation == ABLATION_NO_B)       ? "no-B(sptc-feat-tiles)" :
            (ablation == ABLATION_NO_C)       ? "no-C(sptc-scalar-B)" :
            (ablation == ABLATION_NO_D)       ? "no-D(tc-smem)" :
            (ablation == ABLATION_NO_E)       ? "no-E(no-chunk)" :
            (ablation == ABLATION_NO_F)       ? "no-F(all-atomic)" :
            (ablation == ABLATION_NO_G)       ? "no-G(tc-feat-tiles)" :
            (ablation == ABLATION_NO_H)       ? "no-H(tc-scalar-B)" :
            (ablation == ABLATION_NO_I)       ? "no-I(tc-2xk8)" :
            (ablation == ABLATION_NO_STREAMS) ? "no-AB(serial)" :
                                                "baseline";
        printf("[TR-SPMM 双流均衡诊断]  epoches=%d mode=%s ablation=%s\n",
               epoches, mode_name, ablation_name);
        printf("  TC 流:    %7.3f ms total,  avg %8.4f ms/iter\n", tc_ms, tc_avg);
        printf("  SPTC 流:  %7.3f ms total,  avg %8.4f ms/iter\n", sptc_ms, sptc_avg);
        if (mode == 0) {
            printf("  耗时比 (TC/SPTC): %.2f×\n", tc_ms / (sptc_ms + 1e-6f));
            printf("  耗时比 (SPTC/TC): %.2f×\n", sptc_ms / (tc_ms + 1e-6f));
        }
        printf("  实际总耗时:       %.3f ms (avg %.4f ms/iter)\n", spmm_ms, spmm_ms_avg);
    }

    // ---- 最后一次干净运行获取实际结果 (mode-aware) ----
    cudaMemset(d_output, 0, (size_t)elements * sizeof(float));
    cudaDeviceSynchronize();

    if (mode != 2) {
        launch_tc_kernel(
            d_tc_chunk_win, d_tc_chunk_beg, d_tc_chunk_end, d_tc_chunk_atomic,
            d_tc_value_dense, d_tc_col_old,
            d_rhs_matrix, d_output,
            num_tc_chunks, num_tc_blocks, dense_threshold,
            dimN, mOri, kOri, ablation, stream_tc);
    }
    if (mode != 1) {
        launch_sptc_kernel(
            d_sptc_chunk_win, d_sptc_chunk_beg, d_sptc_chunk_end, d_sptc_chunk_atomic,
            d_sptc_block_row,
            d_sptc_value, d_sptc_packed_meta,
            d_sptc_col_old,
            d_rhs_matrix, d_output,
            num_sptc_chunks, num_sptc_blocks,
            dimN, mOri, kOri, ablation, stream_sptc);
    }

    cudaStreamSynchronize(stream_tc);
    cudaStreamSynchronize(stream_sptc);

    // ---- 清理 ----
    cudaEventDestroy(spmm_start);
    cudaEventDestroy(spmm_end);
    cudaEventDestroy(tc_event_start);
    cudaEventDestroy(tc_event_end);
    cudaEventDestroy(sptc_event_start);
    cudaEventDestroy(sptc_event_end);
    if (ablation != ABLATION_NO_STREAMS) {
        cudaStreamDestroy(stream_tc);
        cudaStreamDestroy(stream_sptc);
    }

    return spmm_ms_avg;
}


	