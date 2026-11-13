#include <stdio.h>
#include <cstdint>
#include <iostream>
#include <torch/extension.h>
// #include "./spmm_utils/dense_tile.h"
#include "./spmm_utils/compute.h"
// #include "./spmm_utils/output_tile.h"
#define mma_k = 8
#include <cuda_fp16.h>
#include <cuda_runtime.h>

static inline void checkHip(hipError_t e) {
    if (e != hipSuccess) {
        printf("HIP error: %s\n", hipGetErrorString(e));
        abort();
    }
}

/*
CUDA-only
*/
template <int Tile_N>
__global__ void spmm_forward_fp16_csr_v2_kernel_amd_wave64(
    const int* __restrict__ c_row_offset,
    const int* __restrict__ c_row,
    const int* __restrict__ c_atomic,
    const int* __restrict__ c_column,
    const half* __restrict__ c_value,
    const half* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix,
    int dimN,
    int nOri,
    int mOri,
    int splitk,
    int parts,
    int partsize)
{
    int c_part_offset_vec = (blockIdx.z * splitk) + blockIdx.y;
    if (c_part_offset_vec >= parts) return;

    // AMD wave64
    int lane_id = threadIdx.x & 63;   // 0..63
    int wave_id = threadIdx.x >> 6;   // threadIdx.x / 64

    int out_row_offset = __ldg(c_row + c_part_offset_vec);
    if (out_row_offset < mOri) {
        int c_row_offset_vec = __ldg(c_row_offset + c_part_offset_vec);
        int nonzeros = __ldg(c_row_offset + c_part_offset_vec + 1) - c_row_offset_vec;

        extern __shared__ at::Half sparse_2[];   
        half* sparse = reinterpret_cast<half*>(sparse_2);
        int*  sparse_col = (int*)&sparse_2[partsize];

        if (nonzeros != 0) {
            if (threadIdx.x < nonzeros) {
                sparse[threadIdx.x]     = __ldg(c_value  + c_row_offset_vec + threadIdx.x);
                sparse_col[threadIdx.x] = __ldg(c_column + c_row_offset_vec + threadIdx.x);
            }
            __syncthreads();


            cudaComputeUtils_fp16_trans compute(
                reinterpret_cast<const float2*>(rhs_matrix),
                sparse_col,
                sparse,
                output_matrix + out_row_offset * nOri,
                nOri,
                lane_id
            );

            compute.cudaCompute(nonzeros, __ldg(c_atomic + c_part_offset_vec));
        }
    }
}

template <int Tile_N>
__global__ void spmm_forward_fp16_csr_v2_kernel_amd_wave64_short(
    const int* __restrict__ c_row_offset,
    const int* __restrict__ c_row,
    const int* __restrict__ c_atomic,
    const int* __restrict__ c_column,
    const half* __restrict__ c_value,
    const half* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix,
    int dimN,
    int nOri,
    int mOri,
    int splitk,
    int parts)
{
    int c_part_offset_vec = (blockIdx.z * splitk) + blockIdx.y;
    if (c_part_offset_vec >= parts) return;

    int lane_id = threadIdx.x & 63;
    int wave_id = threadIdx.x >> 6;

    int out_row_offset = __ldg(c_row + c_part_offset_vec);
    if (out_row_offset < mOri) {
        int c_row_offset_vec = __ldg(c_row_offset + c_part_offset_vec);
        int nonzeros = __ldg(c_row_offset + c_part_offset_vec + 1) - c_row_offset_vec;

        if (nonzeros != 0) {
            cudaComputeUtils_fp16_trans_short compute(
                reinterpret_cast<const float2*>(rhs_matrix),
                c_column + c_row_offset_vec,
                c_value  + c_row_offset_vec,
                output_matrix + out_row_offset * nOri,
                nOri,
                lane_id
            );

            compute.cudaCompute(nonzeros, __ldg(c_atomic + c_part_offset_vec));
        }
    }
}

// fp16
template <int Tile_N>
__global__ void spmm_forward_fp16_csr_v2_kernel_cuda(
    const int* __restrict__ c_row_offset,
    const int* __restrict__ c_row,
    const int* __restrict__ c_atomic,
    const int* __restrict__ c_column,
    const half* __restrict__ c_value,
    const half* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix,
    int dimN,
    int nOri,
    int mOri,
    int splitk,
    int parts,
    int partsize)
{
    int c_part_offset_vec = (blockIdx.z*splitk)+blockIdx.y;
    if(c_part_offset_vec>=parts) return;

    //判断执行tcu还是cuda

    //cuda
    int warpin_id = threadIdx.x%32;
    int warp_id = threadIdx.x/32;
    // if((warp_id+1)*64 > (nOri + 32))
    // return;
    // int c_part_offset_vec = m_index_vec; 
    // if(c_part_offset_vec>=parts) return;
    //当前cuda需要计算的行
    int out_row_offset = __ldg(c_row + c_part_offset_vec);
    if(out_row_offset<mOri)
    {
        int c_row_offset_vec = __ldg(c_row_offset + c_part_offset_vec); 
        int nonzeros = __ldg(c_row_offset + c_part_offset_vec + 1) -  c_row_offset_vec;

        // if(blockIdx.y==1 && threadIdx.x==0) printf("%d, %d\n",out_row_offset, nonzeros);
        // __shared__ float c_part_result[dimN];
        extern __shared__ at::Half sparse_2[];
        //half * sparse = sparse_;
        half * sparse =reinterpret_cast<half *>(sparse_2);
        int * sparse_col = (int *) & sparse_2[partsize];
        // *(c_part_result) = 0.0;
        //进行cuda计算
        if(nonzeros!=0) 
        {  
            
            // __shared__ float sparse[16*4];      
            if(threadIdx.x<nonzeros){
                *(sparse + threadIdx.x) = __ldg(c_value + c_row_offset_vec + threadIdx.x);
                *(sparse_col + threadIdx.x) = __ldg(c_column + c_row_offset_vec + threadIdx.x);               
            }
            __syncthreads();

            cudaComputeUtils_fp16_trans compute(
            reinterpret_cast<const float2 *>(rhs_matrix),
            sparse_col,
            sparse,
            output_matrix + out_row_offset*nOri,
            nOri,
            warpin_id);           

            compute.cudaCompute(nonzeros,
            __ldg(c_atomic + c_part_offset_vec));    

        }
    } 
}

template <int Tile_N>
__global__ void spmm_forward_fp16_csr_v2_kernel_cuda_short(
    const int* __restrict__ c_row_offset,
    const int* __restrict__ c_row,
    const int* __restrict__ c_atomic,
    const int* __restrict__ c_column,
    const half* __restrict__ c_value,
    const half* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix,
    int dimN,
    int nOri,
    int mOri,
    int splitk,
    int parts)
{
    int c_part_offset_vec = (blockIdx.z*splitk)+blockIdx.y;
    if(c_part_offset_vec>=parts) return;
    //cuda
    int warpin_id = threadIdx.x%32;
    int warp_id = threadIdx.x/32;

    // if(swizzle)
    // c_part_offset_vec = __ldg(c_part_offset + c_part_offset_vec);
    if(c_part_offset_vec>=parts) return;
    //当前cuda需要计算的行
    int out_row_offset = __ldg(c_row + c_part_offset_vec);
    if(out_row_offset<mOri)
    {
        // int dimN_index = blockIdx.x * Tile_N;

        int c_row_offset_vec = __ldg(c_row_offset + c_part_offset_vec); 
        int nonzeros = __ldg(c_row_offset + c_part_offset_vec + 1) -  c_row_offset_vec;
        // if(m_index_vec==0 && threadIdx.x==0)
        // printf("%d\n", nonzeros);
        //进行cuda计算
        if(nonzeros!=0) 
        {  
            cudaComputeUtils_fp16_trans_short compute(
            reinterpret_cast<const float2 *>(rhs_matrix),
            c_column + c_row_offset_vec,
            c_value + c_row_offset_vec,
            output_matrix + out_row_offset*nOri,
            nOri,
            warpin_id);           

            compute.cudaCompute(nonzeros,
            __ldg(c_atomic + c_part_offset_vec));    

        }
    } 
    
    
}


//cuda 长短行 fp16
float spmm_forward_fp16_cuda_kernel_split(

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

    half * rhs_matrix,
    float * output_matrix,

    int partsize,
    const int dimN,
    const int mOri,
    int epoches,
    int parts,
    int parts_short,
    bool swizzle)
{

    int n1=dimN;
    if((dimN%64)!=0) n1=((dimN/64)+1)*64;
    int grid_x = (n1/128)+1;
    if(n1%128==0) grid_x-=1;

    int windows =  parts;
    int splitk = 0;
    if(windows<500000) splitk=8;
    else splitk=((windows/1250000)+1)*20;

    int windows_short =  parts_short;
    int splitk_short = 0;
    if(windows_short<500000) splitk_short=8;
    else splitk_short=((windows_short/1250000)+1)*20;

    dim3 grid_dim_c(grid_x, splitk ,((windows/splitk)+1));
    dim3 grid_dim_c_short(grid_x, splitk_short ,((windows_short/splitk_short)+1));
    dim3 block_dim_c(32, 1, 1);
    // int warpSize = 32;
    int sharedmemory = partsize*(sizeof(float)+ sizeof(int));
    cudaStream_t stream1,stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    for(int iter=0; iter<10; ++iter){
        spmm_forward_fp16_csr_v2_kernel_cuda<128><<<grid_dim_c, block_dim_c, sharedmemory, stream1>>>(
            c_row_offset, 
            c_row,
            c_atomic,
            c_column,
            c_value, 
            rhs_matrix,  
            output_matrix,
            n1, dimN, mOri, splitk, parts, partsize);
        
        spmm_forward_fp16_csr_v2_kernel_cuda_short<128><<<grid_dim_c_short, block_dim_c, 0, stream2>>>(
            c_row_offset_short, 
            c_row_short,
            c_atomic_short,
            c_column_short,
            c_value_short, 
            rhs_matrix,  
            output_matrix,
            n1, dimN, mOri, splitk_short, parts_short);
    }
    cudaDeviceSynchronize();

    //测试kernel
    float spmm_ms_avg = 0.0f;
    float spmm_ms = 0.0f;
    cudaEvent_t spmm_start;
    cudaEvent_t spmm_end;
    cudaEventCreate(&spmm_start);
    cudaEventCreate(&spmm_end);
    cudaEventRecord(spmm_start);
    for(int iter=0; iter<epoches; ++iter){
        spmm_forward_fp16_csr_v2_kernel_cuda<128><<<grid_dim_c, block_dim_c, sharedmemory, stream1>>>(
            c_row_offset, 
            c_row,
            c_atomic,
            c_column,
            c_value, 
            rhs_matrix,  
            output_matrix,
            n1, dimN, mOri, splitk, parts,partsize);
        
        spmm_forward_fp16_csr_v2_kernel_cuda_short<128><<<grid_dim_c_short, block_dim_c, 0, stream2>>>(
            c_row_offset_short, 
            c_row_short,
            c_atomic_short,
            c_column_short,
            c_value_short, 
            rhs_matrix,  
            output_matrix,
            n1, dimN, mOri, splitk_short, parts_short);
    }
    cudaEventRecord(spmm_end);
    cudaEventSynchronize(spmm_end);
    cudaEventElapsedTime(&spmm_ms, spmm_start, spmm_end);
    cudaEventDestroy(spmm_start);
    cudaEventDestroy(spmm_end);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    //计算时间 ms
    spmm_ms_avg = spmm_ms/(float)epoches;


    return spmm_ms_avg;
}

float spmm_forward_fp16_hip_kernel_split(
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

    half * rhs_matrix,
    float * output_matrix,

    int partsize,
    const int dimN,
    const int mOri,
    int epoches,
    int parts,
    int parts_short,
    bool swizzle)
{
    int n1 = dimN;
    if ((dimN % 64) != 0) n1 = ((dimN / 64) + 1) * 64;

    int grid_x = (n1 / 128) + 1;
    if (n1 % 128 == 0) grid_x -= 1;

    int windows = parts;
    int splitk = 0;
    if (windows < 500000) splitk = 8;
    else splitk = ((windows / 1250000) + 1) * 20;

    int windows_short = parts_short;
    int splitk_short = 0;
    if (windows_short < 500000) splitk_short = 8;
    else splitk_short = ((windows_short / 1250000) + 1) * 20;

    dim3 grid_dim_c(grid_x, splitk, (windows / splitk) + 1);
    dim3 grid_dim_c_short(grid_x, splitk_short, (windows_short / splitk_short) + 1);
    // dim3 block_dim_c(64, 1, 1);

    int sharedmemory = partsize * (sizeof(float) + sizeof(int));

    hipStream_t stream1, stream2;
    checkHip(hipStreamCreate(&stream1));
    checkHip(hipStreamCreate(&stream2));

    // warmup
    dim3 block_dim_c(64, 1, 1); // AMD wave64

    for(int iter=0; iter<10; ++iter){
        spmm_forward_fp16_csr_v2_kernel_amd_wave64<128><<<grid_dim_c, block_dim_c, sharedmemory, stream1>>>(
            c_row_offset, c_row, c_atomic, c_column, c_value,
            rhs_matrix, output_matrix,
            n1, dimN, mOri, splitk, parts, partsize);

        spmm_forward_fp16_csr_v2_kernel_amd_wave64_short<128><<<grid_dim_c_short, block_dim_c, 0, stream2>>>(
            c_row_offset_short, c_row_short, c_atomic_short, c_column_short, c_value_short,
            rhs_matrix, output_matrix,
            n1, dimN, mOri, splitk_short, parts_short);
    }
    checkHip(hipDeviceSynchronize());

    // timing
    float spmm_ms = 0.0f;
    hipEvent_t spmm_start, spmm_end;
    checkHip(hipEventCreate(&spmm_start));
    checkHip(hipEventCreate(&spmm_end));

    checkHip(hipEventRecord(spmm_start, nullptr));

    for (int iter = 0; iter < epoches; ++iter) {
        spmm_forward_fp16_csr_v2_kernel_amd_wave64<128><<<grid_dim_c, block_dim_c, sharedmemory, stream1>>>(
            c_row_offset, c_row, c_atomic, c_column, c_value,
            rhs_matrix, output_matrix,
            n1, dimN, mOri, splitk, parts, partsize);

        spmm_forward_fp16_csr_v2_kernel_amd_wave64_short<128><<<grid_dim_c_short, block_dim_c, 0, stream2>>>(
            c_row_offset_short, c_row_short, c_atomic_short, c_column_short, c_value_short,
            rhs_matrix, output_matrix,
            n1, dimN, mOri, splitk_short, parts_short);
    }

    // 确保两条 stream 完成后再打点 end
    checkHip(hipStreamSynchronize(stream1));
    checkHip(hipStreamSynchronize(stream2));

    checkHip(hipEventRecord(spmm_end, nullptr));
    checkHip(hipEventSynchronize(spmm_end));
    checkHip(hipEventElapsedTime(&spmm_ms, spmm_start, spmm_end));

    checkHip(hipEventDestroy(spmm_start));
    checkHip(hipEventDestroy(spmm_end));
    checkHip(hipStreamDestroy(stream1));
    checkHip(hipStreamDestroy(stream2));

    return spmm_ms / (float)epoches;
}

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocwmma/rocwmma.hpp>

#ifndef LDG
#define LDG(p) (*(p))
#endif

using namespace rocwmma;


template <int Tile_N>
__global__ void spmm_forward_fp16_csr_v2_kernel_tcu_cc_hip_wmma(
    const int* __restrict__ t_window_offset,
    const int* __restrict__ t_block_offset,
    const half* __restrict__ t_value,
    const int* __restrict__ t_column,
    const long* __restrict__ t_binary,
    const int* __restrict__ t_window_row,
    const int* __restrict__ t_atomic,
    const half* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix,
    int dimN,
    int windows,
    int nOri,
    int mOri,
    int splitk,
    int grid_x)
{
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;

    const int lane_id = threadIdx.x & 31;
    const int wave_id = threadIdx.x >> 5;

    if (wave_id != 0) return; // 当前骨架只示例一个 wave

    int m_index_vec = (blockIdx.z * splitk) + blockIdx.y;
    if (m_index_vec >= windows) return;

    int dimN_index = blockIdx.x * Tile_N;
    if (dimN_index >= dimN) return;

    int t_win_offset = LDG(t_window_offset + m_index_vec);
    int tcu_blocks   = LDG(t_window_offset + m_index_vec + 1) - t_win_offset;
    if (tcu_blocks == 0) return;

    // rocWMMA fragment
    using fragA_t = fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major>;
    using fragB_t = fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, col_major>;
    using fragC_t = fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float>;

    fragA_t fragA;
    fragB_t fragB;
    fragC_t fragC;

    fill_fragment(fragC, 0.0f);

    __shared__ half ldsA[WMMA_M * WMMA_K]; // row-major: 16x16
    __shared__ half ldsB[WMMA_K * WMMA_N]; // col-major or transformed layout as needed

    for (int bi = 0; bi < tcu_blocks; ++bi)
    {
        int value_offset = LDG(t_block_offset + t_win_offset + bi);
        uint64_t binary  = (uint64_t)LDG(t_binary + t_win_offset + bi);

        // ---------------------------
        // Step 1. 清空 A/B tile
        // ---------------------------
        for (int idx = lane_id; idx < WMMA_M * WMMA_K; idx += 32)
            ldsA[idx] = __float2half(0.0f);

        for (int idx = lane_id; idx < WMMA_K * WMMA_N; idx += 32)
            ldsB[idx] = __float2half(0.0f);

        __syncthreads();

        for (int elem = lane_id; elem < WMMA_M * WMMA_K; elem += 32)
        {
            int bit = (binary >> (elem & 63)) & 1ull;
            if (bit)
            {
                int nz_before = __popcll(binary & ((1ull << (elem & 63)) - 1ull));
                ldsA[elem] = LDG(t_value + value_offset + nz_before);
            }
        }

        const int* t_column_block = t_column + (t_win_offset + bi) * 16;

        for (int idx = lane_id; idx < WMMA_K * WMMA_N; idx += 32)
        {
            int k_local = idx / WMMA_N;  // 0..15
            int n_local = idx % WMMA_N;  // 0..15

            int rhs_col = LDG(t_column_block + k_local);
            int rhs_n   = dimN_index + n_local;

            if (rhs_col >= 0 && rhs_n < nOri)
            {
                // B[k, n]
                ldsB[k_local * WMMA_N + n_local] =
                    LDG(rhs_matrix + rhs_col * nOri + rhs_n);
            }
        }

        __syncthreads();

        // ---------------------------
        // Step 4. 用 rocWMMA 调 matrix core
        // ---------------------------
        load_matrix_sync(fragA, ldsA, WMMA_K);
        load_matrix_sync(fragB, ldsB, WMMA_N);
        mma_sync(fragC, fragA, fragB, fragC);

        __syncthreads();
    }

    // ---------------------------
    // Step 5. 将 accumulator 写回 LDS / global
    // ---------------------------
    __shared__ float ldsC[WMMA_M * WMMA_N];
    store_matrix_sync(ldsC, fragC, WMMA_N, mem_row_major);

    __syncthreads();

    int cur_m_index_vec = LDG(t_window_row + m_index_vec);
    int cur_t_atomic    = LDG(t_atomic + m_index_vec);

    int base_row = cur_m_index_vec * WMMA_M;
    int base_col = dimN_index;

    for (int idx = lane_id; idx < WMMA_M * WMMA_N; idx += 32)
    {
        int r = idx / WMMA_N;
        int c = idx % WMMA_N;

        int out_r = base_row + r;
        int out_c = base_col + c;

        if (out_r < mOri && out_c < nOri)
        {
            float val = ldsC[idx];
            float* outp = output_matrix + out_r * nOri + out_c;

            if (cur_t_atomic == 0)
                *outp = val;
            else
                atomicAdd(outp, val);
        }
    }
}

/*
TCU-CUDA
*/

#ifndef LDG
#define LDG(p) (*(p))
#endif

template <int Tile_N>
__global__ void spmm_forward_fp16_csr_v2_kernel_tcu_cc_hip(
    const int* __restrict__ t_window_offset,
    const int* __restrict__ t_block_offset,
    const half* __restrict__ t_value,
    const int* __restrict__ t_column,
    const long* __restrict__ t_binary,      
    const int* __restrict__ t_window_row,
    const int* __restrict__ t_atomic,
    const half* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix,
    int dimN,
    int windows,
    int nOri,
    int mOri,
    int splitk,
    int grid_x)
{
    int m_index_vec = (blockIdx.z * splitk) + blockIdx.y;
    if (m_index_vec >= windows) return;

    int dimN_index = blockIdx.x * Tile_N;

    constexpr int WARPS_PER_BLOCK = 4;

    __shared__ half smem_A[WARPS_PER_BLOCK][32 * 2];
    __shared__ half smem_B[WARPS_PER_BLOCK][32 * 4];


    int t_win_offset = LDG(t_window_offset + m_index_vec);
    int tcu_blocks   = LDG(t_window_offset + m_index_vec + 1) - t_win_offset;
    if (tcu_blocks == 0) return;

    int warp_id   = threadIdx.x >> 5;     // /32
    int warpin_id = threadIdx.x & 31;     // %32

    if ((dimN_index + ((warp_id + 1) * 16)) > dimN) return;

    uint32_t output_fragment_u32[2] = {0u, 0u};
    half* output_fragment = reinterpret_cast<half*>(output_fragment_u32);


    half sparse_fragment1[4];
    half dense_fragment[4];

    const int* t_column_ = t_column + t_win_offset * 8 + ((warpin_id & 3) * 2);

    const half2* matrix_base_ = reinterpret_cast<const half2*>(rhs_matrix + dimN_index);

    for (int bi = 0; bi < tcu_blocks; bi++) {
        int value_offset = LDG(t_block_offset + t_win_offset + bi);
        uint64_t binary  = LDG(t_binary + t_win_offset + bi);

        // 读两位
        uint64_t temp = (binary >> (uint64_t)(warpin_id * 2));
        uint64_t a = 1ull;
        uint64_t mask = (a << (uint64_t)(warpin_id * 2));

        int bit0 = (int)(temp & 1ull);
        int block_offset = -1;

        if (bit0) {
            block_offset = __popcll(binary & (mask - 1ull));
            sparse_fragment1[0] = LDG(t_value + value_offset + block_offset);
        } else {
            sparse_fragment1[0] = __float2half(0.0f);
        }

        int bit1 = (int)((temp >> 1) & 1ull);
        if (bit1) {
            if (block_offset == -1) {
                mask = (a << (uint64_t)(warpin_id * 2 + 1));
                block_offset = __popcll(binary & (mask - 1ull));
            }
            sparse_fragment1[1] = LDG(t_value + value_offset + block_offset + 1);
        } else {
            sparse_fragment1[1] = __float2half(0.0f);
        }

        long col_temp0 = (long)LDG(t_column_ + 0);
        long col_temp1 = (long)LDG(t_column_ + 1);
        t_column_ += 8;

        int col_offset = (warp_id << 3) + (warpin_id >> 2); // (warpin_id/4)

        // 取 dense 的 2×half2 -> 4×half
        // i=0
        if (col_temp0 != -1) {
            long offset = col_temp0 * (nOri / 2);
            half2 v = LDG(matrix_base_ + offset + col_offset);
            dense_fragment[0] = v.x;
            dense_fragment[2] = v.y;
        } else {
            dense_fragment[0] = __float2half(0.0f);
            dense_fragment[2] = __float2half(0.0f);
        }
        // i=1
        if (col_temp1 != -1) {
            long offset = col_temp1 * (nOri / 2);
            half2 v = LDG(matrix_base_ + offset + col_offset);
            dense_fragment[1] = v.x;
            dense_fragment[3] = v.y;
        } else {
            dense_fragment[1] = __float2half(0.0f);
            dense_fragment[3] = __float2half(0.0f);
        }

        __syncwarp(); // wave32

        // 写入 shared
        #pragma unroll
        for (int t = 0; t < 2; ++t) smem_A[warp_id][warpin_id * 2 + t] = sparse_fragment1[t];
        #pragma unroll
        for (int t = 0; t < 4; ++t) smem_B[warp_id][warpin_id * 4 + t] = dense_fragment[t];

        __syncwarp();

        for (int mm = 0; mm < 4; mm++) {
            int peer_s = (warpin_id & 3) * 8 + mm;

            #pragma unroll
            for (int t = 0; t < 2; ++t) {
                sparse_fragment1[t]     = smem_A[warp_id][peer_s * 2 + t];
                sparse_fragment1[t + 2] = smem_A[warp_id][(peer_s + 4) * 2 + t];
            }

            int peer_d = (warpin_id >> 2) * 4 + mm;

            #pragma unroll
            for (int t = 0; t < 4; ++t) {
                dense_fragment[t] = smem_B[warp_id][peer_d * 4 + t];
            }

            // 4 个 half 输出累加（half）
            output_fragment[0] = __hadd(output_fragment[0], __hmul(sparse_fragment1[0], dense_fragment[0]));
            output_fragment[0] = __hadd(output_fragment[0], __hmul(sparse_fragment1[1], dense_fragment[1]));

            output_fragment[1] = __hadd(output_fragment[1], __hmul(sparse_fragment1[2], dense_fragment[0]));
            output_fragment[1] = __hadd(output_fragment[1], __hmul(sparse_fragment1[3], dense_fragment[1]));

            output_fragment[2] = __hadd(output_fragment[2], __hmul(sparse_fragment1[0], dense_fragment[2]));
            output_fragment[2] = __hadd(output_fragment[2], __hmul(sparse_fragment1[1], dense_fragment[3]));

            output_fragment[3] = __hadd(output_fragment[3], __hmul(sparse_fragment1[2], dense_fragment[2]));
            output_fragment[3] = __hadd(output_fragment[3], __hmul(sparse_fragment1[3], dense_fragment[3]));
        }

        __syncwarp();
    }

    int cur_m_index_vec = LDG(t_window_row + m_index_vec);
    int cur_t_atomic    = LDG(t_atomic + m_index_vec);

    int row = (cur_m_index_vec << 3) + (warpin_id & 3) * 2;
    int col = dimN_index + warp_id * 16 + (warpin_id >> 2) * 2; 

    if (row < mOri) {
        float* outp = output_matrix + (row * nOri) + col;

        float o0 = __half2float(output_fragment[0]);
        float o1 = __half2float(output_fragment[2]);
        float o2 = __half2float(output_fragment[1]);
        float o3 = __half2float(output_fragment[3]);

        if (cur_t_atomic == 0) {
            if (col < nOri)        outp[0] = o0;
            if ((col + 1) < nOri)  outp[1] = o1;

            if ((row + 1) < mOri) {
                outp += nOri;
                if (col < nOri)        outp[0] = o2;
                if ((col + 1) < nOri)  outp[1] = o3;
            }
        } else {
            if (col < nOri)        atomicAdd(outp + 0, o0);
            if ((col + 1) < nOri)  atomicAdd(outp + 1, o1);

            if ((row + 1) < mOri) {
                outp += nOri;
                if (col < nOri)        atomicAdd(outp + 0, o2);
                if ((col + 1) < nOri)  atomicAdd(outp + 1, o3);
            }
        }
    }
}

float spmm_forward_fp16_tcu_cuda_kernel_hip(
    int * t_row_offset,
    int * t_blockNew_offset,
    half * t_value, 
    int * t_column, 
    int* t_window_row,
    int * t_atomic,
    long *  t_binary,

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

    half * rhs_matrix,
    float * output_matrix,

    const int parts_t,
    const int parts_c,
    const int partsize_c,
    const int parts_c_short,
    const int dimN,
    const int mOri,
    int epoches)
{
        int dev; 
    cudaGetDevice(&dev);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    int numSM = prop.multiProcessorCount;   // 应该是 128


    //tcu
    int n1_t=dimN;
    if((dimN%16)!=0) n1_t=((dimN/16)+1)*16;
    int grid_x_t = (n1_t/64)+1;
    if(n1_t%64==0) grid_x_t-=1;
    int splitk_t = 0;
    if(parts_t<500000) splitk_t=8;
    else splitk_t=((parts_t/1250000)+1)*20;
    // 4是每个block中的warp数量
    dim3 grid_dim_t(grid_x_t, splitk_t ,((parts_t/splitk_t)+1));
    dim3 block_dim_t(128, 1, 1);

    //cuda
    int n1_c=dimN;
    if((dimN%64)!=0) n1_c=((dimN/64)+1)*64;
    //每个block默认处理128
    int grid_x_c = (n1_c/128)+1;
    if(n1_c%128==0) grid_x_c-=1;

    int windows =  parts_c;
    int splitk_c = 0;
    if(windows<500000) splitk_c=8;
    else splitk_c=((windows/1250000)+1)*20;

    int windows_short =  parts_c_short;
    int splitk_c_short = 0;
    if(windows_short<500000) splitk_c_short=8;
    else splitk_c_short=((windows_short/1250000)+1)*20;

    dim3 grid_dim_c(grid_x_c, splitk_c ,((windows/splitk_c)+1));
    dim3 grid_dim_c_short(grid_x_c, splitk_c_short ,((windows_short/splitk_c_short)+1));
    dim3 block_dim_c(64, 1, 1);
    dim3 block_dim(128, 1, 1);
    int sharedmemory = partsize_c*(sizeof(half)+ sizeof(int));


    hipStream_t stream1, stream2, stream3;
    checkHip(hipStreamCreate(&stream1));
    checkHip(hipStreamCreate(&stream2));
    checkHip(hipStreamCreate(&stream3));

    // warmup
    for(int iter=0; iter<10; ++iter){
        spmm_forward_fp16_csr_v2_kernel_tcu_cc_hip<64><<<grid_dim_t, block_dim_t, 0, stream1>>>(
                    t_row_offset, 
                    t_blockNew_offset,
                    t_value, 
                    t_column, 
                    t_binary,
                    t_window_row,
                    t_atomic,
                    rhs_matrix,  
                    output_matrix,
                    n1_t, parts_t, dimN, mOri, splitk_t, grid_x_t);

        spmm_forward_fp16_csr_v2_kernel_amd_wave64<128><<<grid_dim_c, block_dim_c, sharedmemory, stream2>>>(
            c_row_offset, c_row, c_atomic, c_column, c_value,
            rhs_matrix, output_matrix,
            n1_c, dimN, mOri, splitk_c, parts_c,partsize_c);

        spmm_forward_fp16_csr_v2_kernel_amd_wave64_short<128><<<grid_dim_c_short, block_dim_c, 0, stream3>>>(
            c_row_offset_short, c_row_short, c_atomic_short, c_column_short, c_value_short,
            rhs_matrix, output_matrix,
            n1_c, dimN, mOri, splitk_c_short, parts_c_short);
    }
    checkHip(hipDeviceSynchronize());

    hipEvent_t start, stop;
    checkHip(hipEventCreate(&start));
    checkHip(hipEventCreate(&stop));
    checkHip(hipEventRecord(start, nullptr));

    for(int iter=0; iter<epoches; ++iter){
        spmm_forward_fp16_csr_v2_kernel_tcu_cc_hip<64><<<grid_dim_t, block_dim_t, 0, stream1>>>(
                    t_row_offset, 
                    t_blockNew_offset,
                    t_value, 
                    t_column, 
                    t_binary,
                    t_window_row,
                    t_atomic,
                    rhs_matrix,  
                    output_matrix,
                    n1_t, parts_t, dimN, mOri, splitk_t, grid_x_t);

        spmm_forward_fp16_csr_v2_kernel_amd_wave64<128><<<grid_dim_c, block_dim_c, sharedmemory, stream2>>>(
            c_row_offset, c_row, c_atomic, c_column, c_value,
            rhs_matrix, output_matrix,
            n1_c, dimN, mOri, splitk_c, parts_c,partsize_c);

        spmm_forward_fp16_csr_v2_kernel_amd_wave64_short<128><<<grid_dim_c_short, block_dim_c, 0, stream3>>>(
            c_row_offset_short, c_row_short, c_atomic_short, c_column_short, c_value_short,
            rhs_matrix, output_matrix,
            n1_c, dimN, mOri, splitk_c_short, parts_c_short);
    }

    checkHip(hipEventRecord(stop, nullptr));
    checkHip(hipEventSynchronize(stop));
    float ms = 0.f;
    checkHip(hipEventElapsedTime(&ms, start, stop));

    checkHip(hipEventDestroy(start));
    checkHip(hipEventDestroy(stop));
    checkHip(hipStreamDestroy(stream1));
    checkHip(hipStreamDestroy(stream2));
    checkHip(hipStreamDestroy(stream3));
    return ms / epoches;
}


/*
TCU-only
*/
float spmm_forward_fp16_tcu_bcrs_part_kernel_hip(
    int * t_row_offset,
    int * t_blockNew_offset,
    half * t_value, 
    int * t_column, 
    int* t_window_row,
    int * t_atomic,
    long *  t_binary,
    
    half * rhs_matrix,
    float * output_matrix,

    const int parts_t,
    const int dimN,
    const int mOri,
    int epoches)
{
    //tcu
    int n1_t=dimN;
    int grid_x_t = (n1_t/128)+1;
    if(n1_t%128==0) grid_x_t-=1;
    int splitk_t = 0;
    if(parts_t<500000) splitk_t=8;
    else splitk_t=((parts_t/1250000)+1)*20;
    // 4是每个block中的warp数量
    dim3 grid_dim_t(grid_x_t, splitk_t ,((parts_t/splitk_t)+1));
    dim3 block_dim_t(256, 1, 1);


    for(int iter=0; iter<10; ++iter){
        spmm_forward_fp16_csr_v2_kernel_tcu_cc_hip<64><<<grid_dim_t, block_dim_t>>>(
                    t_row_offset, 
                    t_blockNew_offset,
                    t_value, 
                    t_column, 
                    t_binary,
                    t_window_row,
                    t_atomic,
                    rhs_matrix,  
                    output_matrix,
                    n1_t, parts_t, dimN, mOri, splitk_t, grid_x_t);
    }
    cudaDeviceSynchronize();
    //测试kernel
    float spmm_ms_avg = 0.0f;
    float spmm_ms = 0.0f;
    cudaEvent_t spmm_start;
    cudaEvent_t spmm_end;
    cudaEventCreate(&spmm_start);
    cudaEventCreate(&spmm_end); 
    cudaEventRecord(spmm_start);
    for(int iter=0; iter<epoches; ++iter){
        spmm_forward_fp16_csr_v2_kernel_tcu_cc_hip<64><<<grid_dim_t, block_dim_t>>>(
                    t_row_offset, 
                    t_blockNew_offset,
                    t_value, 
                    t_column, 
                    t_binary,
                    t_window_row,
                    t_atomic,
                    rhs_matrix,  
                    output_matrix,
                    n1_t, parts_t, dimN, mOri, splitk_t, grid_x_t);
    }
    cudaEventRecord(spmm_end);
    cudaEventSynchronize(spmm_end);
    cudaEventElapsedTime(&spmm_ms, spmm_start, spmm_end);
    cudaEventDestroy(spmm_start);
    cudaEventDestroy(spmm_end);
    //计算时间 ms
    spmm_ms_avg = spmm_ms/(float)epoches;


    return spmm_ms_avg;
}
