#include <stdio.h>
#include <mma.h>
#include <cstdint>
#include <iostream>
#include <torch/extension.h>
#include "./spmm_utils/dense_tile.h"
#include "./spmm_utils/compute.h"
#include "./spmm_utils/output_tile.h"
#define mma_k = 8
#include <cuda_fp16.h>
#include <cuda_runtime.h>

template <int Tile_N>
__global__ void spmm_forward_tf32_csr_v2_kernel_tcu_pad(
    const int* __restrict__ t_window_offset,
    const int* __restrict__ t_block_offset,
    const float* __restrict__ t_value,
    const int* __restrict__ t_column,
    const int* __restrict__ t_row,
    // const int* __restrict__ t_col,
    // const int* t_window_row,
    const float* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix,
    int dimN,
    int windows,
    int nOri,
    int mOri,
    int splitk,
    int grid_x)
{
    int m_index_vec = (blockIdx.z*splitk)+blockIdx.y;
    if(m_index_vec>=windows)
    return;
    int dimN_index = blockIdx.x * Tile_N;
    //判断执行tcu还是cuda

    //tcu
    // 需要计算的TCU block个数tcu_blocks
    int t_win_offset = __ldg(t_window_offset + m_index_vec);
    int tcu_blocks = __ldg(t_window_offset + m_index_vec + 1) - t_win_offset; 
    if(tcu_blocks==0) return;

    int warp_id = threadIdx.x>>5;
    if((dimN_index+(((warp_id)+1)*16))>dimN)  return;
    int warpin_id = threadIdx.x%32;
    //用于TCU计算的结果
    float t_output_fragment[4] = {0.0, 0.0, 0.0, 0.0}; 
    //稀疏的块, 16*8
    __shared__ float sparse[32];
    // __shared__ int sparse_to_col[4];
    float sparse_fragment[1] = {0.0};
    float dense_fragment[2] = {0.0, 0.0};
    uint32_t * sparse_fragment_ = reinterpret_cast<uint32_t*>(sparse_fragment);
    uint32_t * dense_fragment_ = reinterpret_cast<uint32_t*>(dense_fragment);
    const int * t_column_ = t_column + t_win_offset*4;
    //读取稠密矩阵的行偏移
    int col_offset = dimN_index + (warp_id*16) + (warpin_id/4);
    const float * matrix_base_ = rhs_matrix + col_offset;
    //循环遍历每个block
    for(int i=0; i<tcu_blocks; i++)
    {
         __syncthreads();
        //block内非零元的数量
        int value_offset = __ldg(t_block_offset + t_win_offset + i);
        int nnz_block = __ldg(t_block_offset + t_win_offset + i + 1) - value_offset;
        //block中的所有warp一起把稀疏数据搬运到sparse, sparse_to_col
        //block内部的每个线程初始化sparse tile 为0
        if(threadIdx.x < 32){
            sparse[threadIdx.x] = 0;
        }
        __syncthreads();
        // 获取列索引
        // if(threadIdx.x < 4){
        //     sparse_to_col[threadIdx.x] = __ldg(t_column_ + threadIdx.x);
        // }
        // t_column_ += 4;
        //搬运稀疏数据
        if(threadIdx.x<nnz_block)
        {
            float v = __ldg(t_value + value_offset + threadIdx.x);
            int row = __ldg(t_row + value_offset + threadIdx.x);
            // int col = __ldg(t_col + value_offset + threadIdx.x);
            *(sparse + row) = v;
        }
         __syncthreads();
        //搬运dense数据
        int col =  __ldg(t_column_ + (threadIdx.x%4));
        t_column_ += 4;
        for(int d=0;d<2;d++)
        {
            if((col_offset + d*8) < nOri)
            { 
                if(col != -1)
                {
                    *(dense_fragment + d) = __ldg(matrix_base_ + (col*nOri) +  d*8);
                }
            }
        }
        //读取稀疏数据

        *(sparse_fragment) = *(sparse + warpin_id);
        __syncwarp();

        //MMA计算
        asm volatile(
        "mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
            : "=f"(t_output_fragment[0]), "=f"(t_output_fragment[1]), "=f"(t_output_fragment[2]), "=f"(t_output_fragment[3])
            : "r"(dense_fragment_[0]), "r"(dense_fragment_[1]), "r"(sparse_fragment_[0]), "f"(t_output_fragment[0]), "f"(t_output_fragment[1]), "f"(t_output_fragment[2]), "f"(t_output_fragment[3]));
        
    }
    //原子写入gloabl
    // int cur_m_index_vec = __ldg(t_window_row + m_index_vec);

    int row=(m_index_vec << 3)+  (warpin_id%4)*2;
    int col=dimN_index + warp_id*16 + + (warpin_id/4);
    //结果矩阵的块内列偏移为转置矩阵的行偏移
    if(row<mOri)
    {
        float * output_matrix_ = output_matrix +(row*nOri)+col;
        if(col<nOri)
        *(output_matrix_) = t_output_fragment[0];
        if((col+8)<nOri)
        *((output_matrix_+8))= t_output_fragment[2];
        // if(col<nOri)
        // *(output_matrix_ ) =  t_output_fragment[2*j];
        // if((col+1)<nOri)
        // *(output_matrix_+1 ) =  t_output_fragment[1+2*j];
        if((row+1)<mOri)
        {
            output_matrix_ += nOri;
            if(col<nOri)
            *(output_matrix_) = t_output_fragment[1];
            if((col+8)<nOri)
            *((output_matrix_+8)) =  t_output_fragment[3];
        }
    }
}
//tcu
float spmm_forward_tf32_tcu_kernel(
    int * t_windowNew_offset,
    int * t_blockNew_offset,
    float * t_value, 
    int * t_column, 
    int*  t_row,
    // int*  t_col,
    // int*  t_window_row,

    float * rhs_matrix,
    float * output_matrix,

    const int dimM,
    const int dimN,
    const int mOri,
    int epoches)
{
    //n1为按8补齐后的dimN
    int n1=dimN;
    if((dimN%16)!=0) n1=((dimN/16)+1)*16;
    int grid_x = (n1/64)+1;
    if(n1%64==0) grid_x-=1;

    // int windows = boundary + (parts/(4*grid_x)) + 1;
    // int windows = boundary;
    int splitk = 0;
    if(dimM<500000) splitk=8;
    else splitk=((dimM/1250000)+1)*20;

    // 4是每个block中的warp数量
    dim3 grid_dim(grid_x, splitk ,((dimM/splitk)+1));
    dim3 block_dim(128, 1, 1);
    // printf("%d %d %d %d\n", boundary, parts, grid_x, windows);
    for(int iter=0; iter<10; ++iter){
        spmm_forward_tf32_csr_v2_kernel_tcu_pad<64><<<grid_dim, block_dim>>>(
            t_windowNew_offset, 
            t_blockNew_offset,
            t_value, 
            t_column, 
            t_row, 
            // t_col,
            // t_window_row,
            rhs_matrix,  
            output_matrix,
            n1, dimM, dimN, mOri, splitk, grid_x);
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
        spmm_forward_tf32_csr_v2_kernel_tcu_pad<64><<<grid_dim, block_dim>>>(
            t_windowNew_offset, 
            t_blockNew_offset,
            t_value, 
            t_column, 
            t_row, 
            // t_col,
            // t_window_row,
            rhs_matrix,  
            output_matrix,
            n1, dimM, dimN, mOri, splitk, grid_x);
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

template <int Tile_N>
__global__ void spmm_forward_tf32_csr_v2_kernel_tcu_pad_sgt(
    const int* __restrict__ t_window_offset,
    const int* __restrict__ node_pointer,
    const float* __restrict__ t_value,
    const int* __restrict__ t_column,
    const int* __restrict__ t_row,
    const int* __restrict__ t_col,
    // const int* t_window_row,
    const float* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix,
    int dimN,
    int windows,
    int nOri,
    int mOri,
    int splitk,
    int grid_x)
{
    int m_index_vec = (blockIdx.z*splitk)+blockIdx.y;
    if(m_index_vec>=windows)
    return;
    int dimN_index = blockIdx.x * Tile_N;
    //判断执行tcu还是cuda

    //tcu
    // 需要计算的TCU block个数tcu_blocks
    int tcu_blocks = __ldg(t_window_offset + m_index_vec);
    // int tcu_blocks = __ldg(t_window_offset + m_index_vec + 1) - t_win_offset; 
    if(tcu_blocks==0) return;

    int warp_id = threadIdx.x>>5;
    if((dimN_index+(((warp_id)+1)*16))>dimN)  return;
    int warpin_id = threadIdx.x%32;
    //用于TCU计算的结果
    float t_output_fragment[4] =  {0.0, 0.0, 0.0, 0.0}; 
    //稀疏的块, 16*8
    __shared__ float sparse[32];
    __shared__ int sparse_to_col[4];
    float sparse_fragment[1] = {0.0};
    float dense_fragment[2] = {0.0, 0.0};
    uint32_t * sparse_fragment_ = reinterpret_cast<uint32_t*>(sparse_fragment);
    uint32_t * dense_fragment_ = reinterpret_cast<uint32_t*>(dense_fragment);
    // const int * t_column_ = t_column + t_win_offset*4;
    //读取稠密矩阵的行偏移
    int col_offset = dimN_index + (warp_id*16) + (warpin_id/4);
    const float * matrix_base_ = rhs_matrix + col_offset;
    int value_offset = __ldg(node_pointer + m_index_vec*8);
    int nnz_block = __ldg(node_pointer + m_index_vec*8 + 8) - value_offset;
    //循环遍历每个block
    for(int i=0; i<tcu_blocks; i++)
    {
         __syncthreads();
        //block内非零元的数量
        //block中的所有warp一起把稀疏数据搬运到sparse, sparse_to_col
        //block内部的每个线程初始化sparse tile 为0
        if(threadIdx.x < 32){
            sparse[threadIdx.x] = 0.0;
        }
        if(threadIdx.x < 4){
            sparse_to_col[threadIdx.x] = -1;
        }
        __syncthreads();
        //每个block 128线程
        int ites = (nnz_block/128)+1; 
        if((nnz_block%128) == 0) ites-=1;
        for(int q=0; q<ites; q++)
        {
            int cur = q*128 + threadIdx.x;
            if(cur <nnz_block)
            {
                int col = __ldg(t_col + value_offset + cur);
                    //tf32 8x4划块
                if(col < (i+1)*4 && col>=(i*4)){
                    float v = __ldg(t_value + value_offset + cur);
                    int row = __ldg(t_row + value_offset + cur);
                    int colum =  __ldg(t_column + value_offset + cur);
                    *(sparse + row*4 + col%4) = v;
                    // 可能会写冲突
                    *(sparse_to_col+col%4) = colum;
                }
            }
        }
         __syncthreads();
        //          if(m_index_vec==0 && threadIdx.x==0){
        //     for(int q =0;q<32;q++)
        //     printf("%f ", *(sparse+q));
        //     printf("\n");
        //     for(int q =0;q<4;q++)
        //     printf("%d ", *(sparse_to_col+q));
        //     printf("\n");
        //      printf("%d %d ", i, tcu_blocks);
        //     printf("\n");
        // }
        //搬运dense数据
        int col =  *(sparse_to_col + (threadIdx.x%4));
        for(int d=0;d<2;d++)
        {
            if((col_offset + d*8) < nOri)
            { 
                if(col != -1)
                {
                    *(dense_fragment + d) = __ldg(matrix_base_ + (col*nOri) +  d*8);
                }
            }
        }
        //读取稀疏数据

        *(sparse_fragment) = *(sparse + warpin_id);
        __syncwarp();

        //MMA计算
        asm volatile(
        "mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
            : "=f"(t_output_fragment[0]), "=f"(t_output_fragment[1]), "=f"(t_output_fragment[2]), "=f"(t_output_fragment[3])
            : "r"(dense_fragment_[0]), "r"(dense_fragment_[1]), "r"(sparse_fragment_[0]), "f"(t_output_fragment[0]), "f"(t_output_fragment[1]), "f"(t_output_fragment[2]), "f"(t_output_fragment[3]));
        
    }
    //原子写入gloabl
    // int cur_m_index_vec = __ldg(t_window_row + m_index_vec);

    int row=(m_index_vec << 3)+  (warpin_id%4)*2;
    int col=dimN_index + warp_id*16 + + (warpin_id/4);
    //结果矩阵的块内列偏移为转置矩阵的行偏移
    if(row<mOri)
    {
        float * output_matrix_ = output_matrix +(row*nOri)+col;
        if(col<nOri)
        *(output_matrix_) = t_output_fragment[0];
        if((col+8)<nOri)
        *((output_matrix_+8))= t_output_fragment[2];
        // if(col<nOri)
        // *(output_matrix_ ) =  t_output_fragment[2*j];
        // if((col+1)<nOri)
        // *(output_matrix_+1 ) =  t_output_fragment[1+2*j];
        if((row+1)<mOri)
        {
            output_matrix_ += nOri;
            if(col<nOri)
            *(output_matrix_) = t_output_fragment[1];
            if((col+8)<nOri)
            *((output_matrix_+8)) =  t_output_fragment[3];
        }
    }
}
float spmm_forward_tf32_tcu_kernel_sgt(
    int * t_windowNew_offset,
    int * t_blockNew_offset,
    float * t_value, 
    int * t_column, 
    int*  t_row,
    int*  t_col,
    // int*  t_window_row,

    float * rhs_matrix,
    float * output_matrix,

    const int dimM,
    const int dimN,
    const int mOri,
    int epoches)
{
    //n1为按8补齐后的dimN
    int n1=dimN;
    if((dimN%16)!=0) n1=((dimN/16)+1)*16;
    int grid_x = (n1/64)+1;
    if(n1%64==0) grid_x-=1;

    // int windows = boundary + (parts/(4*grid_x)) + 1;
    // int windows = boundary;
    int splitk = 0;
    if(dimM<500000) splitk=8;
    else splitk=((dimM/1250000)+1)*20;

    // 4是每个block中的warp数量
    dim3 grid_dim(grid_x, splitk ,((dimM/splitk)+1));
    dim3 block_dim(128, 1, 1);
    // printf("%d %d %d %d\n", boundary, parts, grid_x, windows);
    for(int iter=0; iter<10; ++iter){
        spmm_forward_tf32_csr_v2_kernel_tcu_pad_sgt<64><<<grid_dim, block_dim>>>(
            t_windowNew_offset, 
            t_blockNew_offset,
            t_value, 
            t_column, 
            t_row, 
            t_col,
            // t_window_row,
            rhs_matrix,  
            output_matrix,
            n1, dimM, dimN, mOri, splitk, grid_x);
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
        spmm_forward_tf32_csr_v2_kernel_tcu_pad_sgt<64><<<grid_dim, block_dim>>>(
            t_windowNew_offset, 
            t_blockNew_offset,
            t_value, 
            t_column, 
            t_row, 
            t_col,
            // t_window_row,
            rhs_matrix,  
            output_matrix,
            n1, dimM, dimN, mOri, splitk, grid_x);
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


template <int Tile_N>
__global__ void spmm_forward_fp16_csr_v2_kernel_tcu_pad(
    const int* __restrict__ t_window_offset,
    const int* __restrict__ t_block_offset,
    const int* __restrict__ t_column,
    const half* __restrict__ t_value,
    const int* __restrict__ t_row,
    //const int* __restrict__ t_col,
    // const int* t_window_row,
    const half* __restrict__ rhs_matrix,
    half* __restrict__ output_matrix,
    int dimN,
    int windows,
    int nOri,
    int mOri,
    int splitk,
    int grid_x)
{
    int m_index_vec = (blockIdx.z*splitk)+blockIdx.y;
    if(m_index_vec>=windows)
    return;
    int dimN_index = blockIdx.x * Tile_N;
    //判断执行tcu还是cuda

    //tcu
    // 需要计算的TCU block个数tcu_blocks
    int t_win_offset = __ldg(t_window_offset + m_index_vec);
    int tcu_blocks = __ldg(t_window_offset + m_index_vec + 1) - t_win_offset; 
    if(tcu_blocks==0) return;

    int warp_id = threadIdx.x>>5;
    if((dimN_index+(((warp_id)+1)*16))>dimN)  return;
    int warpin_id = threadIdx.x%32;
    //用于TCU计算的结果
    uint32_t output_fragment_[2] = {0,0};
    half * output_fragment = reinterpret_cast<half *>(output_fragment_);
    //稀疏的块, 16*8
    __shared__ at::Half sparse_[64];
    half * sparse = reinterpret_cast<half *>(sparse_);
    // __shared__ int sparse_to_col[8];

    float sparse_fragment[1] = {0.0};
    at::Half dense_fragment1_[4] = {0.0, 0.0, 0.0, 0.0};
    half * dense_fragment = reinterpret_cast<half *>(dense_fragment1_);
    uint32_t * sparse_fragment_ = reinterpret_cast<uint32_t*>(sparse_fragment);
    uint32_t * dense_fragment_ = reinterpret_cast<uint32_t*>(dense_fragment);
    
    const int * t_column_ = t_column + t_win_offset*8;
    // uint32_t * t_output_fragment_ = reinterpret_cast<uint32_t*>(t_output_fragment);
    //读取稠密矩阵的行偏移
    int col_offset = dimN_index + (warp_id<<4) + (warpin_id/4);
    const half * matrix_base_ = rhs_matrix + col_offset;
    //循环遍历每个block
    for(int i=0; i<tcu_blocks; i++)
    {
        __syncthreads();
        //block内非零元的数量
        int value_offset = __ldg(t_block_offset + t_win_offset + i);
        int nnz_block = __ldg(t_block_offset + t_win_offset + i + 1) - value_offset;
        if(threadIdx.x < 64){
            sparse[threadIdx.x] = __float2half(0.0f);
        }
        __syncthreads();
        // 获取列索引
        // if(threadIdx.x < 8){
        //     sparse_to_col[threadIdx.x] = __ldg(t_column_ + threadIdx.x);
        // }
        // t_column_ += 8;
        //搬运稀疏数据
        if(threadIdx.x<nnz_block)
        {
            half v = __ldg(t_value + value_offset + threadIdx.x);
            int row = __ldg(t_row + value_offset + threadIdx.x);
            // int col = __ldg(t_col + value_offset + threadIdx.x);
            *(sparse + row) = v;
        }
         __syncthreads();
        //搬运dense数据
        // for(int d=0;d<2;d++)
        // {
        //     if((col_offset + d*8) < nOri)
        //     { 
        //             if(sparse_to_col[(threadIdx.x%4)*2] != -1)
        //             {
        //                 *(dense_fragment + 2*d) = __ldg(matrix_base_ + (sparse_to_col[(threadIdx.x%4)*2]*nOri) + d*8);
        //             }

        //             if(sparse_to_col[(threadIdx.x%4)*2 + 1] != -1)
        //             {
        //                 *(dense_fragment + 2*d + 1) = __ldg(matrix_base_ + (sparse_to_col[(threadIdx.x%4)*2 + 1]*nOri) + d*8);
        //             }
        //     }

        // } 
        int col =  __ldg(t_column_ + (threadIdx.x%4)*2);
        int col1 =  __ldg(t_column_ + (threadIdx.x%4)*2 + 1);
        t_column_ += 8;
        for(int d=0;d<2;d++)
        {
            if((col_offset + d*8) < nOri)
            { 
                    if(col != -1)
                    {
                        *(dense_fragment + 2*d) = __ldg(matrix_base_ + (col*nOri) + d*8);
                    }

                    if(col1 != -1)
                    {
                        *(dense_fragment + 2*d + 1) = __ldg(matrix_base_ + (col1*nOri) + d*8);
                    }
            }

        } 
        //读取稀疏数据
        float *sparse_ = reinterpret_cast<float *>(sparse);
        *(sparse_fragment) = *(sparse_ + warpin_id);
        __syncwarp();

        //MMA计算
        asm("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 \t"
                "{%0,%1}, \t"
                "{%2,%3}, \t"
                "{%4}, \t"
                "{%0,%1}; ":
                "+r"(output_fragment_[0]), "+r"(output_fragment_[1]):
                "r"(dense_fragment_[0]),  "r"(dense_fragment_[1]),
                "r"(sparse_fragment_[0])
            );  
        
    }

    

    // int cur_m_index_vec = __ldg(t_window_row + m_index_vec);

    int row=(m_index_vec << 3)+  (warpin_id%4)*2;
    int col=dimN_index + warp_id*16 + + (warpin_id/4);

    if(row<mOri)
    {
        half * output_matrix_ = output_matrix +(row*nOri)+col;

            if(col<nOri)
            *(output_matrix_ ) = output_fragment[0];
            if((col+8)<nOri)
            *(output_matrix_+8) =  output_fragment[2];
            if((row+1)<mOri)
            {
                output_matrix_ += nOri;
                if(col<nOri)
                *(output_matrix_) = output_fragment[1];
                if((col+8)<nOri)
                *(output_matrix_+8) = output_fragment[3];
            }
        
    }
}

float spmm_forward_fp16_tcu_kernel(
    int * t_windowNew_offset,
    int * t_blockNew_offset,
    int * t_column, 
    half * t_value, 
    int*  t_row,
    // int*  t_col,
    // int*  t_window_row,

    half * rhs_matrix,
    half * output_matrix,

    const int dimM,
    const int dimN,
    const int mOri,
    int epoches)
{
    //n1为按8补齐后的dimN
    int n1=dimN;
    if((dimN%16)!=0) n1=((dimN/16)+1)*16;
    int grid_x = (n1/64)+1;
    if(n1%64==0) grid_x-=1;

    // int windows = boundary + (parts/(4*grid_x)) + 1;
    // int windows = dimM/8;
    int splitk = 0;
    if(dimM<500000) splitk=8;
    else splitk=((dimM/1250000)+1)*20;

    // 4是每个block中的warp数量
    dim3 grid_dim(grid_x, splitk ,((dimM/splitk)+1));
    dim3 block_dim(128, 1, 1);
    // printf("%d %d %d %d\n", boundary, parts, grid_x, windows);
    for(int iter=0; iter<10; ++iter){
        spmm_forward_fp16_csr_v2_kernel_tcu_pad<64><<<grid_dim, block_dim>>>(
            t_windowNew_offset, 
            t_blockNew_offset,
            t_column, 
            t_value, 
            t_row, 
            //t_col,
            // t_window_row,
            rhs_matrix,  
            output_matrix,
            n1, dimM, dimN, mOri, splitk, grid_x);
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

    // cudaStream_t stream1,stream2;
    // cudaStreamCreate(&stream1);
    // cudaStreamCreate(&stream2);
    // printf("%d, %d, %d, %d\n", n1, dimM, dimN, mOri);
    for(int iter=0; iter<epoches; ++iter){
        spmm_forward_fp16_csr_v2_kernel_tcu_pad<64><<<grid_dim, block_dim>>>(
            t_windowNew_offset, 
            t_blockNew_offset,
            t_column, 
            t_value, 
            t_row, 
            //t_col,
            // t_window_row,
            rhs_matrix,  
            output_matrix,
            n1, dimM, dimN, mOri, splitk, grid_x);
    }
    cudaEventRecord(spmm_end);
    cudaEventSynchronize(spmm_end);
    cudaEventElapsedTime(&spmm_ms, spmm_start, spmm_end);
    cudaEventDestroy(spmm_start);
    cudaEventDestroy(spmm_end);
    // cudaStreamDestroy(stream1);
    // cudaStreamDestroy(stream2);
    //计算时间 ms
    spmm_ms_avg = spmm_ms/(float)epoches;


    return spmm_ms_avg;
}

float spmm_forward_fp16_tcu_kernel_stream(
    int * t_windowNew_offset,
    int * t_blockNew_offset,
    int * t_column, 
    half * t_value, 
    int*  t_row,
    int * t_windowNew_offset1,
    int * t_blockNew_offset1,
    int * t_column1, 
    half * t_value1, 
    int*  t_row1,

    half * rhs_matrix,
    half * output_matrix,
    half * output_matrix1,

    const int dimM,
    const int dimM1,
    const int dimN,
    const int mOri,
    const int mOri1,
    int epoches)
{
    //n1为按8补齐后的dimN
    int n1=dimN;
    if((dimN%16)!=0) n1=((dimN/16)+1)*16;
    int grid_x = (n1/64)+1;
    if(n1%64==0) grid_x-=1;

    // int windows = boundary + (parts/(4*grid_x)) + 1;
    // int windows = dimM/8;
    int splitk = 0;
    if(dimM<500000) splitk=8;
    else splitk=((dimM/1250000)+1)*20;

    // 4是每个block中的warp数量
    dim3 grid_dim(grid_x, splitk ,((dimM/splitk)+1));
    dim3 grid_dim1(grid_x, splitk ,((dimM1/splitk)+1));
    dim3 block_dim(128, 1, 1);

    for(int iter=0; iter<10; ++iter){
        spmm_forward_fp16_csr_v2_kernel_tcu_pad<64><<<grid_dim, block_dim>>>(
            t_windowNew_offset, 
            t_blockNew_offset,
            t_column, 
            t_value, 
            t_row, 
            //t_col,
            // t_window_row,
            rhs_matrix,  
            output_matrix,
            n1, dimM, dimN, mOri, splitk, grid_x);
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

    cudaStream_t stream1,stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    // printf("%d, %d, %d, %d\n", n1, dimM, dimN, mOri);
    for(int iter=0; iter<epoches; ++iter){
        spmm_forward_fp16_csr_v2_kernel_tcu_pad<64><<<grid_dim, block_dim, 0, stream1>>>(
            t_windowNew_offset, 
            t_blockNew_offset,
            t_column, 
            t_value, 
            t_row, 
            //t_col,
            // t_window_row,
            rhs_matrix,  
            output_matrix,
            n1, dimM, dimN, mOri, splitk, grid_x);

        spmm_forward_fp16_csr_v2_kernel_tcu_pad<64><<<grid_dim1, block_dim, 0, stream2>>>(
            t_windowNew_offset1, 
            t_blockNew_offset1,
            t_column1, 
            t_value1, 
            t_row1, 
            //t_col,
            // t_window_row,
            rhs_matrix,  
            output_matrix1,
            n1, dimM1, dimN, mOri1, splitk, grid_x);
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
template <int Tile_N>
__global__ void spmm_forward_fp16_csr_v2_kernel_tcu_pad_sgt(
    const int* __restrict__ t_window_offset,
    const int* __restrict__ node_pointer,
    const int* __restrict__ t_column,
    const half* __restrict__ t_value,
    const int* __restrict__ t_row,
    const int* __restrict__ t_col,
    // const int* t_window_row,
    const half* __restrict__ rhs_matrix,
    half* __restrict__ output_matrix,
    int dimN,
    int windows,
    int nOri,
    int mOri,
    int splitk,
    int grid_x)
{
    int m_index_vec = (blockIdx.z*splitk)+blockIdx.y;
    if(m_index_vec>=windows)
    return;
    int dimN_index = blockIdx.x * Tile_N;
    //判断执行tcu还是cuda

    //tcu
    // 需要计算的TCU block个数tcu_blocks
    int tcu_blocks = __ldg(t_window_offset + m_index_vec);
    // int tcu_blocks = __ldg(t_window_offset + m_index_vec + 1) - t_win_offset; 
    if(tcu_blocks==0) return;

    int warp_id = threadIdx.x>>5;
    if((dimN_index+(((warp_id)+1)*16))>dimN)  return;
    int warpin_id = threadIdx.x%32;
    //用于TCU计算的结果
    uint32_t output_fragment_[2] = {0,0};
    half * output_fragment = reinterpret_cast<half *>(output_fragment_);
    //稀疏的块, 16*8
    __shared__ at::Half sparse_[64];
    half * sparse = reinterpret_cast<half *>(sparse_);
    __shared__ int sparse_to_col[8];

    float sparse_fragment[1] = {0.0};
    at::Half dense_fragment1_[4] = {0.0, 0.0, 0.0, 0.0};
    half * dense_fragment = reinterpret_cast<half *>(dense_fragment1_);
    uint32_t * sparse_fragment_ = reinterpret_cast<uint32_t*>(sparse_fragment);
    uint32_t * dense_fragment_ = reinterpret_cast<uint32_t*>(dense_fragment);
    
    // const int * t_column_ = t_column + t_win_offset*8;
    // uint32_t * t_output_fragment_ = reinterpret_cast<uint32_t*>(t_output_fragment);
    //读取稠密矩阵的行偏移
    int col_offset = dimN_index + (warp_id<<4) + (warpin_id/4);
    const half * matrix_base_ = rhs_matrix + col_offset;
    int value_offset = __ldg(node_pointer + m_index_vec*8);
    int nnz_block = __ldg(node_pointer + m_index_vec*8 + 8) - value_offset;
    //循环遍历每个block
    for(int i=0; i<tcu_blocks; i++)
    {
        __syncthreads();
        //block内非零元的数量
        if(threadIdx.x < 64){
            sparse[threadIdx.x] = __float2half(0.0f);
        }
        if(threadIdx.x < 8){
            sparse_to_col[threadIdx.x] = -1;
        }
        __syncthreads();
        int ites = (nnz_block/128)+1; 
        if((nnz_block%128) == 0) ites-=1;
        for(int q=0; q<ites; q++)
        {
            int cur = q*128 + threadIdx.x;
            if(cur <nnz_block)
            {
                int col = __ldg(t_col + value_offset + cur);
                //tf32 8x8划块
                if(col < (i+1)*8 && col>=(i*8)){
                    half v = __ldg(t_value + value_offset + cur);
                    int row = __ldg(t_row + value_offset + cur);
                    int colum =  __ldg(t_column + value_offset + cur);
                    *(sparse + row*8 + col%8) = v;
                    // 可能会写冲突
                    *(sparse_to_col+col%8) = colum;
                }
            }
        }
         __syncthreads();

        int col =  *(sparse_to_col + (threadIdx.x%4)*2);
        int col1 =  *(sparse_to_col + (threadIdx.x%4)*2 + 1);
        for(int d=0;d<2;d++)
        {
            if((col_offset + d*8) < nOri)
            { 
                    if(col != -1)
                    {
                        *(dense_fragment + 2*d) = __ldg(matrix_base_ + (col*nOri) + d*8);
                    }

                    if(col1 != -1)
                    {
                        *(dense_fragment + 2*d + 1) = __ldg(matrix_base_ + (col1*nOri) + d*8);
                    }
            }

        } 
        //读取稀疏数据
        float *sparse_ = reinterpret_cast<float *>(sparse);
        *(sparse_fragment) = *(sparse_ + warpin_id);
        __syncwarp();

        //MMA计算
        asm("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 \t"
                "{%0,%1}, \t"
                "{%2,%3}, \t"
                "{%4}, \t"
                "{%0,%1}; ":
                "+r"(output_fragment_[0]), "+r"(output_fragment_[1]):
                "r"(dense_fragment_[0]),  "r"(dense_fragment_[1]),
                "r"(sparse_fragment_[0])
            );  
        
    }

    

    // int cur_m_index_vec = __ldg(t_window_row + m_index_vec);

    int row=(m_index_vec << 3)+  (warpin_id%4)*2;
    int col=dimN_index + warp_id*16 + + (warpin_id/4);

    if(row<mOri)
    {
        half * output_matrix_ = output_matrix +(row*nOri)+col;

            if(col<nOri)
            *(output_matrix_ ) = output_fragment[0];
            if((col+8)<nOri)
            *(output_matrix_+8) =  output_fragment[2];
            if((row+1)<mOri)
            {
                output_matrix_ += nOri;
                if(col<nOri)
                *(output_matrix_) = output_fragment[1];
                if((col+8)<nOri)
                *(output_matrix_+8) = output_fragment[3];
            }
        
    }
}

float spmm_forward_fp16_tcu_kernel_sgt(
    int * t_windowNew_offset,
    int * t_blockNew_offset,
    int * t_column, 
    half * t_value, 
    int*  t_row,
    int*  t_col,
    // int*  t_window_row,

    half * rhs_matrix,
    half * output_matrix,

    const int dimM,
    const int dimN,
    const int mOri,
    int epoches)
{
    //n1为按8补齐后的dimN
    int n1=dimN;
    if((dimN%16)!=0) n1=((dimN/16)+1)*16;
    int grid_x = (n1/64)+1;
    if(n1%64==0) grid_x-=1;

    // int windows = boundary + (parts/(4*grid_x)) + 1;
    // int windows = dimM/8;
    int splitk = 0;
    if(dimM<500000) splitk=8;
    else splitk=((dimM/1250000)+1)*20;

    // 4是每个block中的warp数量
    dim3 grid_dim(grid_x, splitk ,((dimM/splitk)+1));
    dim3 block_dim(128, 1, 1);
    // printf("%d %d %d %d\n", boundary, parts, grid_x, windows);
    for(int iter=0; iter<10; ++iter){
        spmm_forward_fp16_csr_v2_kernel_tcu_pad_sgt<64><<<grid_dim, block_dim>>>(
            t_windowNew_offset, 
            t_blockNew_offset,
            t_column, 
            t_value, 
            t_row, 
            t_col,
            // t_window_row,
            rhs_matrix,  
            output_matrix,
            n1, dimM, dimN, mOri, splitk, grid_x);
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

    // cudaStream_t stream1,stream2;
    // cudaStreamCreate(&stream1);
    // cudaStreamCreate(&stream2);
    // printf("%d, %d, %d, %d\n", n1, dimM, dimN, mOri);
    for(int iter=0; iter<epoches; ++iter){
        spmm_forward_fp16_csr_v2_kernel_tcu_pad_sgt<64><<<grid_dim, block_dim>>>(
            t_windowNew_offset, 
            t_blockNew_offset,
            t_column, 
            t_value, 
            t_row, 
            t_col,
            // t_window_row,
            rhs_matrix,  
            output_matrix,
            n1, dimM, dimN, mOri, splitk, grid_x);
    }
    cudaEventRecord(spmm_end);
    cudaEventSynchronize(spmm_end);
    cudaEventElapsedTime(&spmm_ms, spmm_start, spmm_end);
    cudaEventDestroy(spmm_start);
    cudaEventDestroy(spmm_end);
    // cudaStreamDestroy(stream1);
    // cudaStreamDestroy(stream2);
    //计算时间 ms
    spmm_ms_avg = spmm_ms/(float)epoches;


    return spmm_ms_avg;
}



template <int Tile_N>
__global__ void spmm_forward_tf32_csr_v2_kernel_tcu_pad_binary(
    const int* __restrict__ t_window_offset,
    const int* __restrict__ t_block_offset,
    const float* __restrict__ t_value,
    const int* __restrict__ t_column,
    const int* __restrict__ t_binary,

    const float* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix,
    int dimN,
    int windows,
    int nOri,
    int mOri,
    int splitk,
    int grid_x)
{
    int m_index_vec = (blockIdx.z*splitk)+blockIdx.y;
    if(m_index_vec>=windows)
    return;
    int dimN_index = blockIdx.x * Tile_N;
    //判断执行tcu还是cuda

    //tcu
    // 需要计算的TCU block个数tcu_blocks
    int t_win_offset = __ldg(t_window_offset + m_index_vec);
    int tcu_blocks = __ldg(t_window_offset + m_index_vec + 1) - t_win_offset; 
    if(tcu_blocks==0) return;

    int warp_id = threadIdx.x>>5;
    if((dimN_index+(((warp_id)+1)*16))>dimN)  return;
    int warpin_id = threadIdx.x%32;
    //用于TCU计算的结果
    float t_output_fragment[4] = {0.0, 0.0, 0.0, 0.0}; 
    //稀疏的块, 16*8
    // __shared__ float sparse[32];
    // __shared__ int sparse_to_col[4];
    float sparse_fragment[1] = {0.0};
    float dense_fragment[2] = {0.0, 0.0};
    uint32_t * sparse_fragment_ = reinterpret_cast<uint32_t*>(sparse_fragment);
    uint32_t * dense_fragment_ = reinterpret_cast<uint32_t*>(dense_fragment);
    const int * t_column_ = t_column + t_win_offset*4;
    //读取稠密矩阵的行偏移
    int col_offset = dimN_index + (warp_id*16) + (warpin_id/4);
    const float * matrix_base_ = rhs_matrix + col_offset;
    //循环遍历每个block
    for(int i=0; i<tcu_blocks; i++)
    {
        //  __syncthreads();
         sparse_fragment[0]=0.0;
        //block内非零元的数量
        int value_offset = __ldg(t_block_offset + t_win_offset + i);
        // int nnz_block = __ldg(t_block_offset + t_win_offset + i + 1) - value_offset;
        // if(threadIdx.x < 4){
        //     sparse_to_col[threadIdx.x] = __ldg(t_column_ + threadIdx.x);
        // }
        // t_column_ += 4;
        //  __syncthreads();
        //搬运稀疏数据
        // if(threadIdx.x < 32){
            //warp内的线程均读同一个值binary
            int binary = __ldg(t_binary + t_win_offset + i);
            //取线程对应的二进制位
            int fifthBit = (binary >> warpin_id) & 1;
            if(fifthBit == 1){
                //记录块内偏移
                int mask = (1 << warpin_id) -1;
                int block_offset = __popc(binary & mask);
                sparse_fragment[0] = __ldg(t_value + value_offset + block_offset);

                // int block_offset = 0;
                // for (int m = 0; m < warpin_id; m++)
                // {
                //     if (binary & (1 << m))
                //     {
                //         block_offset++;
                //     }
                // }
                // sparse_fragment[0] = __ldg(t_value + value_offset + block_offset);
            }
        // }
        //搬运dense数据
        int col =  __ldg(t_column_ + (threadIdx.x%4));
        t_column_ += 4;
        for(int d=0;d<2;d++)
        {
            if((col_offset + d*8) < nOri)
            { 
                if(col != -1)
                {
                    *(dense_fragment + d) = __ldg(matrix_base_ + (col*nOri) +  d*8);
                }
            }
        }
        //读取稀疏数据

        // *(sparse_fragment) = *(sparse + warpin_id);
        __syncwarp();

        //MMA计算
        asm volatile(
        "mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
            : "=f"(t_output_fragment[0]), "=f"(t_output_fragment[1]), "=f"(t_output_fragment[2]), "=f"(t_output_fragment[3])
            : "r"(dense_fragment_[0]), "r"(dense_fragment_[1]), "r"(sparse_fragment_[0]), "f"(t_output_fragment[0]), "f"(t_output_fragment[1]), "f"(t_output_fragment[2]), "f"(t_output_fragment[3]));
        
    }
    //原子写入gloabl
    // int cur_m_index_vec = __ldg(t_window_row + m_index_vec);

    int row=(m_index_vec << 3)+  (warpin_id%4)*2;
    int col=dimN_index + warp_id*16 + + (warpin_id/4);
    //结果矩阵的块内列偏移为转置矩阵的行偏移
    if(row<mOri)
    {
        float * output_matrix_ = output_matrix +(row*nOri)+col;
        if(col<nOri)
        *(output_matrix_) = t_output_fragment[0];
        if((col+8)<nOri)
        *((output_matrix_+8))= t_output_fragment[2];
        // if(col<nOri)
        // *(output_matrix_ ) =  t_output_fragment[2*j];
        // if((col+1)<nOri)
        // *(output_matrix_+1 ) =  t_output_fragment[1+2*j];
        if((row+1)<mOri)
        {
            output_matrix_ += nOri;
            if(col<nOri)
            *(output_matrix_) = t_output_fragment[1];
            if((col+8)<nOri)
            *((output_matrix_+8)) =  t_output_fragment[3];
        }
    }
}
float spmm_forward_tf32_tcu_kernel_binary(
    int * t_windowNew_offset,
    int * t_blockNew_offset,
    float * t_value, 
    int * t_column, 
    int *  t_binary,

    float * rhs_matrix,
    float * output_matrix,

    const int dimM,
    const int dimN,
    const int mOri,
    int epoches)
{
    //n1为按8补齐后的dimN
    int n1=dimN;
    if((dimN%16)!=0) n1=((dimN/16)+1)*16;
    int grid_x = (n1/64)+1;
    if(n1%64==0) grid_x-=1;

    // int windows = boundary + (parts/(4*grid_x)) + 1;
    // int windows = boundary;
    int splitk = 0;
    if(dimM<500000) splitk=8;
    else splitk=((dimM/1250000)+1)*20;

    // 4是每个block中的warp数量
    dim3 grid_dim(grid_x, splitk ,((dimM/splitk)+1));
    dim3 block_dim(128, 1, 1);
    // printf("%d %d %d %d\n", boundary, parts, grid_x, windows);
    for(int iter=0; iter<10; ++iter){
        spmm_forward_tf32_csr_v2_kernel_tcu_pad_binary<64><<<grid_dim, block_dim>>>(
            t_windowNew_offset, 
            t_blockNew_offset,
            t_value, 
            t_column, 
            t_binary,
            rhs_matrix,  
            output_matrix,
            n1, dimM, dimN, mOri, splitk, grid_x);
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
        spmm_forward_tf32_csr_v2_kernel_tcu_pad_binary<64><<<grid_dim, block_dim>>>(
            t_windowNew_offset, 
            t_blockNew_offset,
            t_value, 
            t_column, 
            t_binary,
            rhs_matrix,  
            output_matrix,
            n1, dimM, dimN, mOri, splitk, grid_x);
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

template <int Tile_N>
__global__ void spmm_forward_fp16_csr_v2_kernel_tcu_pad_binary(
    const int* __restrict__ t_window_offset,
    const int* __restrict__ t_block_offset,
    const int* __restrict__ t_column,
    const half* __restrict__ t_value,
    const long* __restrict__ t_binary,
    // const int* t_window_row,
    const half* __restrict__ rhs_matrix,
    half* __restrict__ output_matrix,
    int dimN,
    int windows,
    int nOri,
    int mOri,
    int splitk,
    int grid_x)
{
    int m_index_vec = (blockIdx.z*splitk)+blockIdx.y;
    if(m_index_vec>=windows)
    return;
    int dimN_index = blockIdx.x * Tile_N;
    //判断执行tcu还是cuda

    //tcu
    // 需要计算的TCU block个数tcu_blocks
    int t_win_offset = __ldg(t_window_offset + m_index_vec);
    int tcu_blocks = __ldg(t_window_offset + m_index_vec + 1) - t_win_offset; 
    if(tcu_blocks==0) return;

    int warp_id = threadIdx.x>>5;
    if((dimN_index+(((warp_id)+1)*16))>dimN)  return;
    int warpin_id = threadIdx.x%32;
    //用于TCU计算的结果
    uint32_t output_fragment_[2] = {0,0};
    half * output_fragment = reinterpret_cast<half *>(output_fragment_);
    //稀疏的块, 16*8
    // __shared__ at::Half sparse_[64];
    // half * sparse = reinterpret_cast<half *>(sparse_);
    // __shared__ int sparse_to_col[8];

    float sparse_fragment[1] = {0.0};
    at::Half dense_fragment1_[4] = {0.0, 0.0, 0.0, 0.0};
    half * sparse_fragment1 = reinterpret_cast<half *>(sparse_fragment);
    half * dense_fragment = reinterpret_cast<half *>(dense_fragment1_);
    uint32_t * sparse_fragment_ = reinterpret_cast<uint32_t*>(sparse_fragment);
    uint32_t * dense_fragment_ = reinterpret_cast<uint32_t*>(dense_fragment);
    
    const int * t_column_ = t_column + t_win_offset*8;
    // uint32_t * t_output_fragment_ = reinterpret_cast<uint32_t*>(t_output_fragment);
    //读取稠密矩阵的行偏移
    int col_offset = dimN_index + (warp_id<<4) + (warpin_id/4);
    const half * matrix_base_ = rhs_matrix + col_offset;
    //循环遍历每个block
    for(int i=0; i<tcu_blocks; i++)
    {
        // __syncthreads();
        //block内非零元的数量
        int value_offset = __ldg(t_block_offset + t_win_offset + i);
        // int nnz_block = __ldg(t_block_offset + t_win_offset + i + 1) - value_offset;
        long binary = __ldg(t_binary + t_win_offset + i);

        // 取二进制位
        //先取第一个非零元的值
        //取线程对应的二进制位
        // int fifthBit = (binary >> (warpin_id*2)) & 1;
        // long a= 1;
        // if(fifthBit == 1){
        //     long mask = (a << (warpin_id*2)) -1;
        //     int block_offset = __popcll(binary & mask);
        //     sparse_fragment1[0] = __ldg(t_value + value_offset + block_offset);

        //     fifthBit = (binary >> (warpin_id*2 + 1)) & 1;
        //     if(fifthBit == 1){
        //         sparse_fragment1[1] = __ldg(t_value + value_offset + block_offset + 1);
        //     }else{
        //         sparse_fragment1[1]=__float2half(0.0);
        //     }
        // }else{
        //     sparse_fragment1[0]=__float2half(0.0);
        //     //取第二个非零元的值
        //     fifthBit = (binary >> (warpin_id*2 + 1)) & 1;
        //     if(fifthBit == 1){
        //         long mask = (a << (warpin_id*2 + 1)) -1;
        //         int block_offset = __popcll(binary & mask);
        //         sparse_fragment1[1] = __ldg(t_value + value_offset + block_offset);
        //     }else{
        //         sparse_fragment1[1]=__float2half(0.0);
        //     }
        // }


        // for(int h=0;h<2;h++)
        // {   sparse_fragment1[h]=__float2half(0.0);
        //     //取线程对应的二进制位
        //     int fifthBit = (binary >> (warpin_id*2 + h)) & 1;
        //     if(fifthBit == 1){
        //         //记录块内偏移
        //         long a= 1;
        //         long mask = (a << (warpin_id*2 + h)) -1;
        //         int block_offset = __popcll(binary & mask);
        //         sparse_fragment1[h] = __ldg(t_value + value_offset + block_offset);
        //     }
        // }

        long temp = (binary >> (warpin_id*2));
        long a= 1;
        long mask = (a << (warpin_id*2));
        int fifthBit = ((temp) & 1);
        int block_offset = -1;
        if(fifthBit == 1){
            block_offset = __popcll(binary & (mask-1));
            sparse_fragment1[0] = __ldg(t_value + value_offset + block_offset);
        }else{
            sparse_fragment1[0]=__float2half(0.0);
        }
        fifthBit = ((temp>>1) & 1);
        if(fifthBit == 1){
            if(block_offset==-1)
            {
                mask = (a << ((warpin_id*2)+1));
                block_offset = __popcll(binary & (mask-1));
            }
            sparse_fragment1[1] = __ldg(t_value + value_offset + block_offset + 1);
        }else{
            sparse_fragment1[1]=__float2half(0.0);
        }
        //搬运稀疏数据
        // if(threadIdx.x<nnz_block)
        // {
        //     half v = __ldg(t_value + value_offset + threadIdx.x);
        //     int row = __ldg(t_row + value_offset + threadIdx.x);
        //     int col = __ldg(t_col + value_offset + threadIdx.x);
        //     *(sparse + row*8 + col) = v;
        // }
        //  __syncthreads();
        //搬运dense数据
        int col =  __ldg(t_column_ + (threadIdx.x%4)*2);
        int col1 =  __ldg(t_column_ + (threadIdx.x%4)*2 + 1);
        t_column_ += 8;
        for(int d=0;d<2;d++)
        {
            if((col_offset + d*8) < nOri)
            { 
                    if(col != -1)
                    {
                        *(dense_fragment + 2*d) = __ldg(matrix_base_ + (col*nOri) + d*8);
                    }

                    if(col1 != -1)
                    {
                        *(dense_fragment + 2*d + 1) = __ldg(matrix_base_ + (col1*nOri) + d*8);
                    }
            }

        } 
        
        //读取稀疏数据
        // float *sparse_ = reinterpret_cast<float *>(sparse);
        // *(sparse_fragment) = *(sparse_ + warpin_id);
        __syncwarp();

        //MMA计算
        asm("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 \t"
                "{%0,%1}, \t"
                "{%2,%3}, \t"
                "{%4}, \t"
                "{%0,%1}; ":
                "+r"(output_fragment_[0]), "+r"(output_fragment_[1]):
                "r"(dense_fragment_[0]),  "r"(dense_fragment_[1]),
                "r"(sparse_fragment_[0])
            );  
        
    }

    

    // int cur_m_index_vec = __ldg(t_window_row + m_index_vec);

    int row=(m_index_vec << 3)+  (warpin_id%4)*2;
    int col=dimN_index + warp_id*16 + + (warpin_id/4);

    if(row<mOri)
    {
        half * output_matrix_ = output_matrix +(row*nOri)+col;

            if(col<nOri)
            *(output_matrix_ ) = output_fragment[0];
            if((col+8)<nOri)
            *(output_matrix_+8) =  output_fragment[2];
            if((row+1)<mOri)
            {
                output_matrix_ += nOri;
                if(col<nOri)
                *(output_matrix_) = output_fragment[1];
                if((col+8)<nOri)
                *(output_matrix_+8) = output_fragment[3];
            }
        
    }
}

// tcu fp16 binary
float spmm_forward_fp16_tcu_kernel_binary(
    int * t_windowNew_offset,
    int * t_blockNew_offset,
    int * t_column, 
    half * t_value, 
    long *  t_binary,

    half * rhs_matrix,
    half * output_matrix,

    const int dimM,
    const int dimN,
    const int mOri,
    int epoches)
{
    //n1为按8补齐后的dimN
    int n1=dimN;
    if((dimN%16)!=0) n1=((dimN/16)+1)*16;
    int grid_x = (n1/64)+1;
    if(n1%64==0) grid_x-=1;

    // int windows = boundary + (parts/(4*grid_x)) + 1;
    // int windows = dimM/8;
    int splitk = 0;
    if(dimM<500000) splitk=8;
    else splitk=((dimM/1250000)+1)*20;

    // 4是每个block中的warp数量
    dim3 grid_dim(grid_x, splitk ,((dimM/splitk)+1));
    dim3 block_dim(128, 1, 1);
    // printf("%d %d %d %d\n", boundary, parts, grid_x, windows);
    for(int iter=0; iter<10; ++iter){
        spmm_forward_fp16_csr_v2_kernel_tcu_pad_binary<64><<<grid_dim, block_dim>>>(
            t_windowNew_offset, 
            t_blockNew_offset,
            t_column, 
            t_value, 
            t_binary,
            // t_window_row,
            rhs_matrix,  
            output_matrix,
            n1, dimM, dimN, mOri, splitk, grid_x);
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
    // printf("%d, %d, %d, %d\n", n1, dimM, dimN, mOri);
    for(int iter=0; iter<epoches; ++iter){
        spmm_forward_fp16_csr_v2_kernel_tcu_pad_binary<64><<<grid_dim, block_dim>>>(
            t_windowNew_offset, 
            t_blockNew_offset,
            t_column, 
            t_value, 
            t_binary,
            // t_window_row,
            rhs_matrix,  
            output_matrix,
            n1, dimM, dimN, mOri, splitk, grid_x);
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

template <int Tile_N>
__global__ void spmm_forward_tf32_csr_v2_kernel_tcu(
    const int* __restrict__ t_window_offset,
    const int* __restrict__ t_block_offset,
    const float* __restrict__ t_value,
    const int* __restrict__ t_column,
    const int* __restrict__ t_binary,
    const int* t_window_row,
    const int* t_atomic,
    const float* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix,
    int dimN,
    int windows,
    int nOri,
    int mOri,
    int splitk,
    int grid_x)
{
    int m_index_vec = (blockIdx.z*splitk)+blockIdx.y;
    if(m_index_vec>=windows)
    return;
    int dimN_index = blockIdx.x * Tile_N;
    //判断执行tcu还是cuda

    //tcu
    // 需要计算的TCU block个数tcu_blocks
    int t_win_offset = __ldg(t_window_offset + m_index_vec);
    int tcu_blocks = __ldg(t_window_offset + m_index_vec + 1) - t_win_offset; 
    if(tcu_blocks==0) return;

        int warp_id = threadIdx.x>>5;
        if((dimN_index+(((warp_id)+1)*16))>dimN)  return;
        int warpin_id = threadIdx.x%32;
        //用于TCU计算的结果
        float t_output_fragment[4] = {0.0, 0.0, 0.0, 0.0}; 
        //稀疏的块, 16*8
        // __shared__ float sparse[32];
        // __shared__ int sparse_to_col[4];
        float sparse_fragment[1] = {0.0};
        float dense_fragment[2] = {0.0, 0.0};
        uint32_t * sparse_fragment_ = reinterpret_cast<uint32_t*>(sparse_fragment);
        uint32_t * dense_fragment_ = reinterpret_cast<uint32_t*>(dense_fragment);
        const int * t_column_ = t_column + t_win_offset*4;
        //读取稠密矩阵的行偏移
        int col_offset = dimN_index + (warp_id<<4) + (warpin_id/4);
        const float * matrix_base_ = rhs_matrix + col_offset;
        //循环遍历每个block
        for(int i=0; i<tcu_blocks; i++)
        {
            sparse_fragment[0]=0.0;
            //block内非零元的数量
            int value_offset = __ldg(t_block_offset + t_win_offset + i);
            //搬运稀疏数据
            // if(threadIdx.x < 32){
                //warp内的线程均读同一个值binary
                int binary = __ldg(t_binary + t_win_offset + i);
                //取线程对应的二进制位
                int fifthBit = (binary >> warpin_id) & 1;
                if(fifthBit == 1){
                    //记录块内偏移
                    int mask = (1 << warpin_id) -1;
                    int block_offset = __popc(binary & mask);
                    sparse_fragment[0] = __ldg(t_value + value_offset + block_offset);
                }
            // }
            //搬运dense数据
            int col =  __ldg(t_column_ + (threadIdx.x%4));
            t_column_ += 4;
            for(int d=0;d<2;d++)
            {
                if((col_offset + d*8) < nOri)
                { 
                    if(col != -1)
                    {
                        *(dense_fragment + d) = __ldg(matrix_base_ + (col*nOri) +  d*8);
                    }
                }
            }
            //读取稀疏数据

            // *(sparse_fragment) = *(sparse + warpin_id);
            __syncwarp();
            //MMA计算
            asm volatile(
            "mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
                : "=f"(t_output_fragment[0]), "=f"(t_output_fragment[1]), "=f"(t_output_fragment[2]), "=f"(t_output_fragment[3])
                : "r"(dense_fragment_[0]), "r"(dense_fragment_[1]), "r"(sparse_fragment_[0]), "f"(t_output_fragment[0]), "f"(t_output_fragment[1]), "f"(t_output_fragment[2]), "f"(t_output_fragment[3]));
            
        }
        // if(threadIdx.x==0 && m_index_vec==0)
        // printf("%f, %f, %f, %f\n", t_output_fragment[0], t_output_fragment[1], t_output_fragment[2], t_output_fragment[3]);
        //原子写入gloabl
        int cur_m_index_vec = __ldg(t_window_row + m_index_vec);
        int cur_t_atomic = __ldg(t_atomic + m_index_vec);
        int row=(cur_m_index_vec << 3)+  (warpin_id%4)*2;
        int col=dimN_index + warp_id*16 + + (warpin_id/4);

        if(row<mOri)
        {
            float * output_matrix_ = output_matrix +(row*nOri)+col;
            if(cur_t_atomic==0)
            {
                if(col<nOri)
                *(output_matrix_ ) = t_output_fragment[0];
                if((col+8)<nOri)
                *(output_matrix_+8) =  t_output_fragment[2];
                if((row+1)<mOri)
                {
                    output_matrix_ += nOri;
                    if(col<nOri)
                    *(output_matrix_) = t_output_fragment[1];
                    if((col+8)<nOri)
                    *(output_matrix_+8) =  t_output_fragment[3];
                }
            }else{
                if(col<nOri)
                atomicAdd(output_matrix_ , t_output_fragment[0]);
                if((col+8)<nOri)
                atomicAdd(output_matrix_+8, t_output_fragment[2]);
                if((row+1)<mOri)
                {
                    output_matrix_ += nOri;
                    if(col<nOri)
                    atomicAdd(output_matrix_ , t_output_fragment[1]);
                    if((col+8)<nOri)
                    atomicAdd(output_matrix_+8 , t_output_fragment[3]);
                }
            }
        }
}

template <int Tile_N>
__global__ void spmm_forward_tf32_csr_v2_kernel_tcu_csr(
    const int* __restrict__ t_window_offset,
    const int* __restrict__ t_block_offset,
    const float* __restrict__ t_value,
    const int* __restrict__ t_column,
    const int* __restrict__ t_row,
    const int* t_window_row,
    const int* t_atomic,
    const float* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix,
    int dimN,
    int windows,
    int nOri,
    int mOri,
    int splitk,
    int grid_x)
{
    int m_index_vec = (blockIdx.z*splitk)+blockIdx.y;
    if(m_index_vec>=windows)
    return;
    int dimN_index = blockIdx.x * Tile_N;
    //判断执行tcu还是cuda

    //tcu
    // 需要计算的TCU block个数tcu_blocks
    int t_win_offset = __ldg(t_window_offset + m_index_vec);
    int tcu_blocks = __ldg(t_window_offset + m_index_vec + 1) - t_win_offset; 
    if(tcu_blocks==0) return;

    int warp_id = threadIdx.x>>5;
    if((dimN_index+(((warp_id)+1)*16))>dimN)  return;
    int warpin_id = threadIdx.x%32;
    //用于TCU计算的结果
    float t_output_fragment[4] = {0.0, 0.0, 0.0, 0.0}; 
    //稀疏的块, 16*8
    __shared__ float sparse[32];
    // __shared__ int sparse_to_col[4];
    float sparse_fragment[1] = {0.0};
    float dense_fragment[2] = {0.0, 0.0};
    uint32_t * sparse_fragment_ = reinterpret_cast<uint32_t*>(sparse_fragment);
    uint32_t * dense_fragment_ = reinterpret_cast<uint32_t*>(dense_fragment);
    const int * t_column_ = t_column + t_win_offset*4;
    //读取稠密矩阵的行偏移
    int col_offset = dimN_index + (warp_id*16) + (warpin_id/4);
    const float * matrix_base_ = rhs_matrix + col_offset;
        //循环遍历每个block
    for(int i=0; i<tcu_blocks; i++)
    {
         __syncthreads();
        //block内非零元的数量
        int value_offset = __ldg(t_block_offset + t_win_offset + i);
        int nnz_block = __ldg(t_block_offset + t_win_offset + i + 1) - value_offset;
        //block中的所有warp一起把稀疏数据搬运到sparse, sparse_to_col
        //block内部的每个线程初始化sparse tile 为0
        if(threadIdx.x < 32){
            sparse[threadIdx.x] = 0.0;
        }
        __syncthreads();
        // 获取列索引
        // if(threadIdx.x < 4){
        //     sparse_to_col[threadIdx.x] = __ldg(t_column_ + threadIdx.x);
        // }
        // t_column_ += 4;
        //搬运稀疏数据
        if(threadIdx.x<nnz_block)
        {
            float v = __ldg(t_value + value_offset + threadIdx.x);
            int row = __ldg(t_row + value_offset + threadIdx.x);
            // int col = __ldg(t_col + value_offset + threadIdx.x);
            *(sparse + row) = v;
        }
         __syncthreads();
        //搬运dense数据
        int col =  __ldg(t_column_ + (threadIdx.x%4));
        t_column_ += 4;
        for(int d=0;d<2;d++)
        {
            if((col_offset + d*8) < nOri)
            { 
                if(col != -1)
                {
                    *(dense_fragment + d) = __ldg(matrix_base_ + (col*nOri) +  d*8);
                }
            }
        }
        //读取稀疏数据

        *(sparse_fragment) = *(sparse + warpin_id);
        __syncwarp();

        //MMA计算
        asm volatile(
        "mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
            : "=f"(t_output_fragment[0]), "=f"(t_output_fragment[1]), "=f"(t_output_fragment[2]), "=f"(t_output_fragment[3])
            : "r"(dense_fragment_[0]), "r"(dense_fragment_[1]), "r"(sparse_fragment_[0]), "f"(t_output_fragment[0]), "f"(t_output_fragment[1]), "f"(t_output_fragment[2]), "f"(t_output_fragment[3]));
        
    }
        // if(threadIdx.x==0 && m_index_vec==0)
        // printf("%f, %f, %f, %f\n", t_output_fragment[0], t_output_fragment[1], t_output_fragment[2], t_output_fragment[3]);
        //原子写入gloabl
        int cur_m_index_vec = __ldg(t_window_row + m_index_vec);
        int cur_t_atomic = __ldg(t_atomic + m_index_vec);
        int row=(cur_m_index_vec << 3)+  (warpin_id%4)*2;
        int col=dimN_index + warp_id*16 + + (warpin_id/4);

        if(row<mOri)
        {
            float * output_matrix_ = output_matrix +(row*nOri)+col;
            if(cur_t_atomic==0)
            {
                if(col<nOri)
                *(output_matrix_ ) = t_output_fragment[0];
                if((col+8)<nOri)
                *(output_matrix_+8) =  t_output_fragment[2];
                if((row+1)<mOri)
                {
                    output_matrix_ += nOri;
                    if(col<nOri)
                    *(output_matrix_) = t_output_fragment[1];
                    if((col+8)<nOri)
                    *(output_matrix_+8) =  t_output_fragment[3];
                }
            }else{
                if(col<nOri)
                atomicAdd(output_matrix_ , t_output_fragment[0]);
                if((col+8)<nOri)
                atomicAdd(output_matrix_+8, t_output_fragment[2]);
                if((row+1)<mOri)
                {
                    output_matrix_ += nOri;
                    if(col<nOri)
                    atomicAdd(output_matrix_ , t_output_fragment[1]);
                    if((col+8)<nOri)
                    atomicAdd(output_matrix_+8 , t_output_fragment[3]);
                }
            }
        }
}


template <int Tile_N>
__global__ void spmm_forward_fp16_csr_v2_kernel_tcu(
    const int* __restrict__ t_window_offset,
    const int* __restrict__ t_block_offset,
    const half* __restrict__ t_value,
    const int* __restrict__ t_column,
    const long* __restrict__ t_binary,
    const int* t_window_row,
    const int* t_atomic,
    const half* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix,
    int dimN,
    int windows,
    int nOri,
    int mOri,
    int splitk,
    int grid_x)
{
    int m_index_vec = (blockIdx.z*splitk)+blockIdx.y;
    if(m_index_vec>=windows)
    return;
    int dimN_index = blockIdx.x * Tile_N;
    //判断执行tcu还是cuda

    //tcu
    // 需要计算的TCU block个数tcu_blocks
     int t_win_offset = __ldg(t_window_offset + m_index_vec);
    int tcu_blocks = __ldg(t_window_offset + m_index_vec + 1) - t_win_offset; 
    if(tcu_blocks==0) return;

    int warp_id = threadIdx.x>>5;
    if((dimN_index+(((warp_id)+1)*16))>dimN)  return;
    int warpin_id = threadIdx.x%32;
    //用于TCU计算的结果
    uint32_t output_fragment_[2] = {0,0};
    half * output_fragment = reinterpret_cast<half *>(output_fragment_);
    //稀疏的块, 16*8
    // __shared__ at::Half sparse_[64];
    // half * sparse = reinterpret_cast<half *>(sparse_);
    // __shared__ int sparse_to_col[8];

    float sparse_fragment[1] = {0.0};
    at::Half dense_fragment1_[4] = {0.0, 0.0, 0.0, 0.0};
    half * sparse_fragment1 = reinterpret_cast<half *>(sparse_fragment);
    half * dense_fragment = reinterpret_cast<half *>(dense_fragment1_);
    uint32_t * sparse_fragment_ = reinterpret_cast<uint32_t*>(sparse_fragment);
    uint32_t * dense_fragment_ = reinterpret_cast<uint32_t*>(dense_fragment);
    
    const int * t_column_ = t_column + t_win_offset*8;
    // uint32_t * t_output_fragment_ = reinterpret_cast<uint32_t*>(t_output_fragment);
    //读取稠密矩阵的行偏移
    int col_offset = dimN_index + (warp_id<<4) + (warpin_id/4);
    const half * matrix_base_ = rhs_matrix + col_offset;
    //循环遍历每个block
    for(int i=0; i<tcu_blocks; i++)
    {
        // __syncthreads();
        //block内非零元的数量
        int value_offset = __ldg(t_block_offset + t_win_offset + i);
        // int nnz_block = __ldg(t_block_offset + t_win_offset + i + 1) - value_offset;
        long binary = __ldg(t_binary + t_win_offset + i);

        long temp = (binary >> (warpin_id*2));
        long a= 1;
        long mask = (a << (warpin_id*2));
        int fifthBit = ((temp) & 1);
        int block_offset = -1;
        if(fifthBit == 1){
            block_offset = __popcll(binary & (mask-1));
            sparse_fragment1[0] = __ldg(t_value + value_offset + block_offset);
        }else{
            sparse_fragment1[0]=__float2half(0.0);
        }
        fifthBit = ((temp>>1) & 1);
        if(fifthBit == 1){
            if(block_offset==-1)
            {
                mask = (a << ((warpin_id*2)+1));
                block_offset = __popcll(binary & (mask-1));
            }
            sparse_fragment1[1] = __ldg(t_value + value_offset + block_offset + 1);
        }else{
            sparse_fragment1[1]=__float2half(0.0);
        }
        //搬运稀疏数据
        // if(threadIdx.x<nnz_block)
        // {
        //     half v = __ldg(t_value + value_offset + threadIdx.x);
        //     int row = __ldg(t_row + value_offset + threadIdx.x);
        //     int col = __ldg(t_col + value_offset + threadIdx.x);
        //     *(sparse + row*8 + col) = v;
        // }
        //  __syncthreads();
        //搬运dense数据
        int col =  __ldg(t_column_ + (threadIdx.x%4)*2);
        int col1 =  __ldg(t_column_ + (threadIdx.x%4)*2 + 1);
        t_column_ += 8;
        for(int d=0;d<2;d++)
        {
            if((col_offset + d*8) < nOri)
            { 
                    if(col != -1)
                    {
                        *(dense_fragment + 2*d) = __ldg(matrix_base_ + (col*nOri) + d*8);
                    }

                    if(col1 != -1)
                    {
                        *(dense_fragment + 2*d + 1) = __ldg(matrix_base_ + (col1*nOri) + d*8);
                    }
            }

        } 
        
        //读取稀疏数据
        // float *sparse_ = reinterpret_cast<float *>(sparse);
        // *(sparse_fragment) = *(sparse_ + warpin_id);
        __syncwarp();

            //MMA计算
        asm("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 \t"
                "{%0,%1}, \t"
                "{%2,%3}, \t"
                "{%4}, \t"
                "{%0,%1}; ":
                "+r"(output_fragment_[0]), "+r"(output_fragment_[1]):
                "r"(dense_fragment_[0]),  "r"(dense_fragment_[1]),
                "r"(sparse_fragment_[0])
            );  
            
    }
        int cur_m_index_vec = __ldg(t_window_row + m_index_vec);
        int cur_t_atomic = __ldg(t_atomic + m_index_vec);
        int row=(cur_m_index_vec << 3)+  (warpin_id%4)*2;
        int col=dimN_index + warp_id*16 + + (warpin_id/4);

        if(row<mOri)
        {
            float * output_matrix_ = output_matrix +(row*nOri)+col;
            if(cur_t_atomic==0)
            {
                if(col<nOri)
                *(output_matrix_ ) = __half2float(output_fragment[0]);
                if((col+8)<nOri)
                *(output_matrix_+8) =  __half2float(output_fragment[2]);
                if((row+1)<mOri)
                {
                    output_matrix_ += nOri;
                    if(col<nOri)
                    *(output_matrix_) = __half2float(output_fragment[1]);
                    if((col+8)<nOri)
                    *(output_matrix_+8) = __half2float( output_fragment[3]);
                }
            }else{
                if(col<nOri)
                atomicAdd(output_matrix_ ,__half2float(output_fragment[0]));
                if((col+8)<nOri)
                atomicAdd(output_matrix_+8, __half2float(output_fragment[2]));
                if((row+1)<mOri)
                {
                    output_matrix_ += nOri;
                    if(col<nOri)
                    atomicAdd(output_matrix_ , __half2float(output_fragment[1]));
                    if((col+8)<nOri)
                    atomicAdd(output_matrix_+8 , __half2float(output_fragment[3]));
                }
            }
        }
}

template <int Tile_N>
__global__ void spmm_forward_fp16_csr_v2_kernel_tcu_csr(
    const int* __restrict__ t_window_offset,
    const int* __restrict__ t_block_offset,
    const half* __restrict__ t_value,
    const int* __restrict__ t_column,
    const int* __restrict__ t_row,
    const int* t_window_row,
    const int* t_atomic,
    const half* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix,
    int dimN,
    int windows,
    int nOri,
    int mOri,
    int splitk,
    int grid_x)
{
    int m_index_vec = (blockIdx.z*splitk)+blockIdx.y;
    if(m_index_vec>=windows)
    return;
    int dimN_index = blockIdx.x * Tile_N;
    //判断执行tcu还是cuda

    //tcu
    // 需要计算的TCU block个数tcu_blocks
     int t_win_offset = __ldg(t_window_offset + m_index_vec);
    int tcu_blocks = __ldg(t_window_offset + m_index_vec + 1) - t_win_offset; 
    if(tcu_blocks==0) return;

      int warp_id = threadIdx.x>>5;
    if((dimN_index+(((warp_id)+1)*16))>dimN)  return;
    int warpin_id = threadIdx.x%32;
    //用于TCU计算的结果
    uint32_t output_fragment_[2] = {0,0};
    half * output_fragment = reinterpret_cast<half *>(output_fragment_);
    //稀疏的块, 16*8
    __shared__ at::Half sparse_[64];
    half * sparse = reinterpret_cast<half *>(sparse_);
    // __shared__ int sparse_to_col[8];

    float sparse_fragment[1] = {0.0};
    at::Half dense_fragment1_[4] = {0.0, 0.0, 0.0, 0.0};
    half * dense_fragment = reinterpret_cast<half *>(dense_fragment1_);
    uint32_t * sparse_fragment_ = reinterpret_cast<uint32_t*>(sparse_fragment);
    uint32_t * dense_fragment_ = reinterpret_cast<uint32_t*>(dense_fragment);
    
    const int * t_column_ = t_column + t_win_offset*8;
    // uint32_t * t_output_fragment_ = reinterpret_cast<uint32_t*>(t_output_fragment);
    //读取稠密矩阵的行偏移
    int col_offset = dimN_index + (warp_id<<4) + (warpin_id/4);
    const half * matrix_base_ = rhs_matrix + col_offset;
    //循环遍历每个block
 for(int i=0; i<tcu_blocks; i++)
    {
        __syncthreads();
        //block内非零元的数量
        int value_offset = __ldg(t_block_offset + t_win_offset + i);
        int nnz_block = __ldg(t_block_offset + t_win_offset + i + 1) - value_offset;
        if(threadIdx.x < 64){
            sparse[threadIdx.x] = __float2half(0.0f);
        }
        __syncthreads();
        // 获取列索引
        // if(threadIdx.x < 8){
        //     sparse_to_col[threadIdx.x] = __ldg(t_column_ + threadIdx.x);
        // }
        // t_column_ += 8;
        //搬运稀疏数据
        if(threadIdx.x<nnz_block)
        {
            half v = __ldg(t_value + value_offset + threadIdx.x);
            int row = __ldg(t_row + value_offset + threadIdx.x);
            // int col = __ldg(t_col + value_offset + threadIdx.x);
            *(sparse + row) = v;
        }
         __syncthreads();
        //搬运dense数据
        // for(int d=0;d<2;d++)
        // {
        //     if((col_offset + d*8) < nOri)
        //     { 
        //             if(sparse_to_col[(threadIdx.x%4)*2] != -1)
        //             {
        //                 *(dense_fragment + 2*d) = __ldg(matrix_base_ + (sparse_to_col[(threadIdx.x%4)*2]*nOri) + d*8);
        //             }

        //             if(sparse_to_col[(threadIdx.x%4)*2 + 1] != -1)
        //             {
        //                 *(dense_fragment + 2*d + 1) = __ldg(matrix_base_ + (sparse_to_col[(threadIdx.x%4)*2 + 1]*nOri) + d*8);
        //             }
        //     }

        // } 
        int col =  __ldg(t_column_ + (threadIdx.x%4)*2);
        int col1 =  __ldg(t_column_ + (threadIdx.x%4)*2 + 1);
        t_column_ += 8;
        for(int d=0;d<2;d++)
        {
            if((col_offset + d*8) < nOri)
            { 
                    if(col != -1)
                    {
                        *(dense_fragment + 2*d) = __ldg(matrix_base_ + (col*nOri) + d*8);
                    }

                    if(col1 != -1)
                    {
                        *(dense_fragment + 2*d + 1) = __ldg(matrix_base_ + (col1*nOri) + d*8);
                    }
            }

        } 
        //读取稀疏数据
        float *sparse_ = reinterpret_cast<float *>(sparse);
        *(sparse_fragment) = *(sparse_ + warpin_id);
        __syncwarp();

        //MMA计算
        asm("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 \t"
                "{%0,%1}, \t"
                "{%2,%3}, \t"
                "{%4}, \t"
                "{%0,%1}; ":
                "+r"(output_fragment_[0]), "+r"(output_fragment_[1]):
                "r"(dense_fragment_[0]),  "r"(dense_fragment_[1]),
                "r"(sparse_fragment_[0])
            );  
        
    }

        int cur_m_index_vec = __ldg(t_window_row + m_index_vec);
        int cur_t_atomic = __ldg(t_atomic + m_index_vec);
        int row=(cur_m_index_vec << 3)+  (warpin_id%4)*2;
        int col=dimN_index + warp_id*16 + + (warpin_id/4);

        if(row<mOri)
        {
            float * output_matrix_ = output_matrix +(row*nOri)+col;
            if(cur_t_atomic==0)
            {
                if(col<nOri)
                *(output_matrix_ ) = __half2float(output_fragment[0]);
                if((col+8)<nOri)
                *(output_matrix_+8) =  __half2float(output_fragment[2]);
                if((row+1)<mOri)
                {
                    output_matrix_ += nOri;
                    if(col<nOri)
                    *(output_matrix_) = __half2float(output_fragment[1]);
                    if((col+8)<nOri)
                    *(output_matrix_+8) = __half2float( output_fragment[3]);
                }
            }else{
                if(col<nOri)
                atomicAdd(output_matrix_ ,__half2float(output_fragment[0]));
                if((col+8)<nOri)
                atomicAdd(output_matrix_+8, __half2float(output_fragment[2]));
                if((row+1)<mOri)
                {
                    output_matrix_ += nOri;
                    if(col<nOri)
                    atomicAdd(output_matrix_ , __half2float(output_fragment[1]));
                    if((col+8)<nOri)
                    atomicAdd(output_matrix_+8 , __half2float(output_fragment[3]));
                }
            }
        }
    
}
//tcu cuda
float spmm_forward_tf32_tcu_part_kernel(
    int * t_row_offset,
    int * t_blockNew_offset,
    float * t_value, 
    int * t_column, 
    int* t_window_row,
    int * t_atomic,
    int * d_t_binary,

    float * rhs_matrix,
    float * output_matrix,

    const int parts_t,
    const int dimN,
    const int mOri,
    int epoches)
{
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

    for(int iter=0; iter<10; ++iter){
  spmm_forward_tf32_csr_v2_kernel_tcu<64><<<grid_dim_t, block_dim_t>>>(
            t_row_offset, 
            t_blockNew_offset,
            t_value, 
            t_column, 
            d_t_binary, 
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
        spmm_forward_tf32_csr_v2_kernel_tcu<64><<<grid_dim_t, block_dim_t>>>(
            t_row_offset, 
            t_blockNew_offset,
            t_value, 
            t_column, 
            d_t_binary, 
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



//tcu cuda 
float spmm_forward_fp16_tcu_part_kernel(
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
    if((dimN%16)!=0) n1_t=((dimN/16)+1)*16;
    int grid_x_t = (n1_t/64)+1;
    if(n1_t%64==0) grid_x_t-=1;
    int splitk_t = 0;
    if(parts_t<500000) splitk_t=8;
    else splitk_t=((parts_t/1250000)+1)*20;
    // 4是每个block中的warp数量
    dim3 grid_dim_t(grid_x_t, splitk_t ,((parts_t/splitk_t)+1));
    dim3 block_dim_t(128, 1, 1);

    for(int iter=0; iter<10; ++iter){
        spmm_forward_fp16_csr_v2_kernel_tcu<64><<<grid_dim_t, block_dim_t>>>(
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
        spmm_forward_fp16_csr_v2_kernel_tcu<64><<<grid_dim_t, block_dim_t>>>(
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


//part + csr
//tcu cuda
float spmm_forward_tf32_tcu_part_kernel_csr(
    int * t_row_offset,
    int * t_blockNew_offset,
    float * t_value, 
    int * t_column, 
    int* t_window_row,
    int * t_atomic,
    int * d_t_binary,

    float * rhs_matrix,
    float * output_matrix,

    const int parts_t,
    const int dimN,
    const int mOri,
    int epoches)
{
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

    for(int iter=0; iter<10; ++iter){
  spmm_forward_tf32_csr_v2_kernel_tcu_csr<64><<<grid_dim_t, block_dim_t>>>(
            t_row_offset, 
            t_blockNew_offset,
            t_value, 
            t_column, 
            d_t_binary, 
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
        spmm_forward_tf32_csr_v2_kernel_tcu_csr<64><<<grid_dim_t, block_dim_t>>>(
            t_row_offset, 
            t_blockNew_offset,
            t_value, 
            t_column, 
            d_t_binary, 
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



//tcu cuda 
float spmm_forward_fp16_tcu_part_kernel_csr(
    int * t_row_offset,
    int * t_blockNew_offset,
    half * t_value, 
    int * t_column, 
    int* t_window_row,
    int * t_atomic,
    int *  t_binary,

    half * rhs_matrix,
    float * output_matrix,

    const int parts_t,
    const int dimN,
    const int mOri,
    int epoches)
{
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

    for(int iter=0; iter<10; ++iter){
        spmm_forward_fp16_csr_v2_kernel_tcu_csr<64><<<grid_dim_t, block_dim_t>>>(
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
        spmm_forward_fp16_csr_v2_kernel_tcu_csr<64><<<grid_dim_t, block_dim_t>>>(
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

template <int Tile_N>
__global__ void spmm_forward_tf32_csr_v2_kernel_cuda(
    const int* __restrict__ c_row_offset,
    const int* __restrict__ c_row,
    const int* __restrict__ c_atomic,
    const int* __restrict__ c_column,
    const float* __restrict__ c_value,
    const float2* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix,
    int dimN,
    int windows,
    int nOri,
    int mOri,
    int splitk,
    int parts,
    int grid_x,
    int partsize)
{
    int c_part_offset_vec = (blockIdx.z*splitk)+blockIdx.y;
    if(c_part_offset_vec>=parts) return;
    //判断执行tcu还是cuda

    //cuda
    int warpin_id = threadIdx.x%32;
    int warp_id = threadIdx.x/32;
    int dimN_index = blockIdx.x * Tile_N + warp_id*64;
    if((warp_id+1)*64 > (nOri + 32))
    return;
    // int c_part_offset_vec = m_index_vec; 
    // if(c_part_offset_vec>=parts) return;
    //当前cuda需要计算的行
    int out_row_offset = __ldg(c_row + c_part_offset_vec);
    if(out_row_offset<mOri)
    {
        // int dimN_index = blockIdx.x * Tile_N;

        int c_row_offset_vec = __ldg(c_row_offset + c_part_offset_vec); 
        int nonzeros = __ldg(c_row_offset + c_part_offset_vec + 1) -  c_row_offset_vec;
        // __shared__ float c_part_result[dimN];
        extern __shared__ float sparse_[];
        float * sparse =sparse_;
        int * sparse_col = (int *) & sparse_[partsize];
        // *(c_part_result) = 0.0;
        //进行cuda计算
        if(nonzeros!=0) 
        {  
            // printf("11111");
            // __shared__ float sparse[16*4];      
            //32个线程把数据搬运到shared, 前提是partSize小于等于32
            if(threadIdx.x<nonzeros){
                *(sparse + threadIdx.x) = __ldg(c_value + c_row_offset_vec + threadIdx.x);
                *(sparse_col + threadIdx.x) = __ldg(c_column + c_row_offset_vec + threadIdx.x);               
            }
            __syncthreads();
            // if(threadIdx.x==0 && c_part_offset_vec==0)
            // {printf("%f, %f, %f\n",sparse[0], sparse[1] , sparse[2]);
            // printf("%d, %d, %d\n",sparse_col[0], sparse_col[1] , sparse_col[2]);}
            cudaComputeUtils_tf32_trans compute(
            rhs_matrix,
            sparse_col,
            // c_column + c_row_offset_vec,
            sparse,
            // c_value + c_row_offset_vec,
            output_matrix,
            nOri,
            warpin_id);           

            compute.cudaCompute(nonzeros, dimN_index/2,
            out_row_offset,
            __ldg(c_atomic + c_part_offset_vec));    

        }
    } 
    
    
}

template <int Tile_N>
__global__ void spmm_forward_tf32_csr_v2_kernel_cuda_short(
    const int* __restrict__ c_row_offset,
    const int* __restrict__ c_row,
    const int* __restrict__ c_atomic,
    const int* __restrict__ c_column,
    const float* __restrict__ c_value,
    const float2* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix,
    int dimN,
    int windows,
    int nOri,
    int mOri,
    int splitk,
    int parts,
    int grid_x)
{
    int m_index_vec = (blockIdx.z*splitk)+blockIdx.y;
    if(m_index_vec>=windows)
    return;

    //判断执行tcu还是cuda

    //cuda
    int warpin_id = threadIdx.x%32;
    int warp_id = threadIdx.x/32;
    int dimN_index = blockIdx.x * Tile_N + warp_id*64;
    if((warp_id+1)*64 > (nOri + 32))
    return;
    int c_part_offset_vec = m_index_vec; 
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
            cudaComputeUtils_tf32_trans_short compute(
            rhs_matrix,
            c_column + c_row_offset_vec,
            c_value + c_row_offset_vec,
            output_matrix,
            nOri,
            warpin_id);           

            compute.cudaCompute(nonzeros, dimN_index/2,
            out_row_offset,
            __ldg(c_atomic + c_part_offset_vec));    

        }
    } 
    
    
}


__global__ void spmm_forward_tf32_csr_v2_kernel_cuda_32(
    const int* __restrict__ c_row_offset,
    const int* __restrict__ c_row,
    const int* __restrict__ c_atomic,
    const int* __restrict__ c_column,
    const float* __restrict__ c_value,
    const float* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix,
    int dimN,
    int windows,
    int nOri,
    int mOri,
    int splitk,
    int parts,
    int grid_x,
    int partsize)
{
    int c_part_offset_vec = (blockIdx.z*splitk)+blockIdx.y;
    if(c_part_offset_vec>=parts) return;
    //判断执行tcu还是cuda

    //cuda
    int warpin_id = threadIdx.x%32;
    int warp_id = threadIdx.x/32;
    // int dimN_index = blockIdx.x * Tile_N + warp_id*64;
    // if((warp_id+1)*64 > (nOri + 32))
    // return;

    int out_row_offset = __ldg(c_row + c_part_offset_vec);
    if(out_row_offset<mOri)
    {
        // int dimN_index = blockIdx.x * Tile_N;

        int c_row_offset_vec = __ldg(c_row_offset + c_part_offset_vec); 
        int nonzeros = __ldg(c_row_offset + c_part_offset_vec + 1) -  c_row_offset_vec;
        // __shared__ float c_part_result[dimN];
        extern __shared__ float sparse_[];
        float * sparse =sparse_;
        int * sparse_col = (int *) & sparse_[partsize];
        // *(c_part_result) = 0.0;
        //进行cuda计算
        if(nonzeros!=0) 
        {  
            // printf("11111");
            // __shared__ float sparse[16*4];      
            //32个线程把数据搬运到shared, 前提是partSize小于等于32
            if(threadIdx.x<nonzeros){
                *(sparse + threadIdx.x) = __ldg(c_value + c_row_offset_vec + threadIdx.x);
                *(sparse_col + threadIdx.x) = __ldg(c_column + c_row_offset_vec + threadIdx.x);               
            }
            __syncthreads();
            // if(threadIdx.x==0 && c_part_offset_vec==0)
            // {printf("%f, %f, %f\n",sparse[0], sparse[1] , sparse[2]);
            // printf("%d, %d, %d\n",sparse_col[0], sparse_col[1] , sparse_col[2]);}
            cudaComputeUtils_tf32_trans_32 compute(
            rhs_matrix,
            sparse_col,
            // c_column + c_row_offset_vec,
            sparse,
            // c_value + c_row_offset_vec,
            output_matrix,
            nOri,
            warpin_id);           

            compute.cudaCompute(nonzeros,
            out_row_offset,
            __ldg(c_atomic + c_part_offset_vec));    

        }
    } 
    
    
}

__global__ void spmm_forward_tf32_csr_v2_kernel_cuda_short_32(
    const int* __restrict__ c_row_offset,
    const int* __restrict__ c_row,
    const int* __restrict__ c_atomic,
    const int* __restrict__ c_column,
    const float* __restrict__ c_value,
    const float* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix,
    int dimN,
    int windows,
    int nOri,
    int mOri,
    int splitk,
    int parts,
    int grid_x)
{
    int m_index_vec = (blockIdx.z*splitk)+blockIdx.y;
    if(m_index_vec>=windows)
    return;

    //判断执行tcu还是cuda

    //cuda
    int warpin_id = threadIdx.x%32;
    int warp_id = threadIdx.x/32;
    // int dimN_index = blockIdx.x * Tile_N + warp_id*64;
    // if((warp_id+1)*64 > (nOri + 32))
    // return;
    int c_part_offset_vec = m_index_vec; 
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
            cudaComputeUtils_tf32_trans_short_32 compute(
            rhs_matrix,
            c_column + c_row_offset_vec,
            c_value + c_row_offset_vec,
            output_matrix,
            nOri,
            warpin_id);           

            compute.cudaCompute(nonzeros,
            out_row_offset,
            __ldg(c_atomic + c_part_offset_vec));    

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
    const half2* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix,
    int dimN,
    int windows,
    int nOri,
    int mOri,
    int splitk,
    int parts,
    int grid_x,
    int partsize)
{
    int c_part_offset_vec = (blockIdx.z*splitk)+blockIdx.y;
    if(c_part_offset_vec>=parts) return;

    //判断执行tcu还是cuda

    //cuda
    int warpin_id = threadIdx.x%32;
    int warp_id = threadIdx.x/32;
    int dimN_index = blockIdx.x * Tile_N + warp_id*64;
    if((warp_id+1)*64 > (nOri + 32))
    return;
    // int c_part_offset_vec = m_index_vec; 
    // if(c_part_offset_vec>=parts) return;
    //当前cuda需要计算的行
    int out_row_offset = __ldg(c_row + c_part_offset_vec);
    if(out_row_offset<mOri)
    {
        // int dimN_index = blockIdx.x * Tile_N;

        int c_row_offset_vec = __ldg(c_row_offset + c_part_offset_vec); 
        int nonzeros = __ldg(c_row_offset + c_part_offset_vec + 1) -  c_row_offset_vec;
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
            //32个线程把数据搬运到shared, 前提是partSize小于等于32
            if(threadIdx.x<nonzeros){
                *(sparse + threadIdx.x) = __ldg(c_value + c_row_offset_vec + threadIdx.x);
                *(sparse_col + threadIdx.x) = __ldg(c_column + c_row_offset_vec + threadIdx.x);               
            }
            __syncthreads();
            // if(threadIdx.x==0 && c_part_offset_vec==0)
            // {printf("%f, %f, %f\n",sparse[0], sparse[1] , sparse[2]);
            // printf("%d, %d, %d\n",sparse_col[0], sparse_col[1] , sparse_col[2]);}
            cudaComputeUtils_fp16_trans compute(
            rhs_matrix,
            sparse_col,
            // c_column + c_row_offset_vec,
            sparse,
            // c_value + c_row_offset_vec,
            output_matrix,
            nOri,
            warpin_id);           

            compute.cudaCompute(nonzeros, dimN_index/2,
            out_row_offset,
            __ldg(c_atomic + c_part_offset_vec));    

        }
    } 
    
    
}

__global__ void spmm_forward_fp16_csr_v2_kernel_cuda_32(
    const int* __restrict__ c_row_offset,
    const int* __restrict__ c_row,
    const int* __restrict__ c_atomic,
    const int* __restrict__ c_column,
    const half* __restrict__ c_value,
    const half* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix,
    int dimN,
    int windows,
    int nOri,
    int mOri,
    int splitk,
    int parts,
    int grid_x,
    int partsize)
{
    int c_part_offset_vec = (blockIdx.z*splitk)+blockIdx.y;
    if(c_part_offset_vec>=parts) return;

    //判断执行tcu还是cuda

    //cuda
    int warpin_id = threadIdx.x%32;
    int warp_id = threadIdx.x/32;
    // int dimN_index = blockIdx.x * Tile_N + warp_id*64;
    // if((warp_id+1)*64 > (nOri + 32))
    // return;
    // int c_part_offset_vec = m_index_vec; 
    // if(c_part_offset_vec>=parts) return;
    //当前cuda需要计算的行
    int out_row_offset = __ldg(c_row + c_part_offset_vec);
    if(out_row_offset<mOri)
    {
        // int dimN_index = blockIdx.x * Tile_N;

        int c_row_offset_vec = __ldg(c_row_offset + c_part_offset_vec); 
        int nonzeros = __ldg(c_row_offset + c_part_offset_vec + 1) -  c_row_offset_vec;
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
            //32个线程把数据搬运到shared, 前提是partSize小于等于32
            if(threadIdx.x<nonzeros){
                *(sparse + threadIdx.x) = __ldg(c_value + c_row_offset_vec + threadIdx.x);
                *(sparse_col + threadIdx.x) = __ldg(c_column + c_row_offset_vec + threadIdx.x);               
            }
            __syncthreads();
            // if(threadIdx.x==0 && c_part_offset_vec==0)
            // {printf("%f, %f, %f\n",sparse[0], sparse[1] , sparse[2]);
            // printf("%d, %d, %d\n",sparse_col[0], sparse_col[1] , sparse_col[2]);}
            cudaComputeUtils_fp16_trans_32 compute(
            rhs_matrix,
            sparse_col,
            // c_column + c_row_offset_vec,
            sparse,
            // c_value + c_row_offset_vec,
            output_matrix,
            nOri,
            warpin_id);           

            compute.cudaCompute(nonzeros,
            out_row_offset,
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
    const half2* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix,
    int dimN,
    int windows,
    int nOri,
    int mOri,
    int splitk,
    int parts,
    int grid_x)
{
    int m_index_vec = (blockIdx.z*splitk)+blockIdx.y;
    if(m_index_vec>=windows)
    return;

    //判断执行tcu还是cuda

    //cuda
    int warpin_id = threadIdx.x%32;
    int warp_id = threadIdx.x/32;
    int dimN_index = blockIdx.x * Tile_N + warp_id*64;
    if((warp_id+1)*64 > (nOri + 32))
    return;
    int c_part_offset_vec = m_index_vec; 
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
            rhs_matrix,
            c_column + c_row_offset_vec,
            c_value + c_row_offset_vec,
            output_matrix,
            nOri,
            warpin_id);           

            compute.cudaCompute(nonzeros, dimN_index/2,
            out_row_offset,
            __ldg(c_atomic + c_part_offset_vec));    

        }
    } 
    
    
}

__global__ void spmm_forward_fp16_csr_v2_kernel_cuda_short_32(
    const int* __restrict__ c_row_offset,
    const int* __restrict__ c_row,
    const int* __restrict__ c_atomic,
    const int* __restrict__ c_column,
    const half* __restrict__ c_value,
    const half* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix,
    int dimN,
    int windows,
    int nOri,
    int mOri,
    int splitk,
    int parts,
    int grid_x)
{
    int m_index_vec = (blockIdx.z*splitk)+blockIdx.y;
    if(m_index_vec>=windows)
    return;

    //判断执行tcu还是cuda

    //cuda
    int warpin_id = threadIdx.x%32;
    int warp_id = threadIdx.x/32;
    int c_part_offset_vec = m_index_vec; 
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
            cudaComputeUtils_fp16_trans_short_32 compute(
            rhs_matrix,
            c_column + c_row_offset_vec,
            c_value + c_row_offset_vec,
            output_matrix,
            nOri,
            warpin_id);           

            compute.cudaCompute(nonzeros,
            out_row_offset,
            __ldg(c_atomic + c_part_offset_vec));    

        }
    } 
    
    
}

//cuda 长短行
float spmm_forward_tf32_cuda_kernel_split(

    int * c_row_offset,
    int * c_row,
    int * c_atomic,
    int * c_column, 
    float * c_value, 

    int * c_row_offset_short,
    int * c_row_short,
    int * c_atomic_short,
    int * c_column_short, 
    float * c_value_short, 

    float2 * rhs_matrix,
    float * output_matrix,

    int partsize,
    const int dimN,
    const int mOri,
    int epoches,
    int parts,
    int parts_short,
    bool swizzle)
{
    // int n1=dimN;
    // if((dimN%16)!=0) n1=((dimN/16)+1)*16;
    // int grid_x = (n1/64)+1;
    // if(n1%64==0) grid_x-=1;
    // int windows =  (parts/4) + 1;
    // if(parts%4==0) windows-=1;
    // int splitk = 0;
    // if(windows<500000) splitk=8;
    // else splitk=((windows/1250000)+1)*20;

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

    dim3 grid_dim(grid_x, splitk ,((windows/splitk)+1));
    dim3 grid_dim_short(grid_x, splitk_short ,((windows_short/splitk_short)+1));
    dim3 block_dim(64, 1, 1);
    // int warpSize = 32;
    int sharedmemory = partsize*(sizeof(float)+ sizeof(int));
    cudaStream_t stream1,stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    for(int iter=0; iter<10; ++iter){
        spmm_forward_tf32_csr_v2_kernel_cuda<128><<<grid_dim, block_dim, sharedmemory, stream1>>>(
            c_row_offset, 
            c_row,
            c_atomic,
            c_column,
            c_value, 
            rhs_matrix,  
            output_matrix,
            n1, windows, dimN, mOri, splitk, parts, grid_x, partsize);
        spmm_forward_tf32_csr_v2_kernel_cuda_short<128><<<grid_dim_short, block_dim, 0 ,stream2>>>(
            c_row_offset_short, 
            c_row_short,
            c_atomic_short,
            c_column_short,
            c_value_short, 
            rhs_matrix,  
            output_matrix,
            n1, windows_short, dimN, mOri, splitk_short, parts_short, grid_x);
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
        spmm_forward_tf32_csr_v2_kernel_cuda<128><<<grid_dim, block_dim, sharedmemory, stream1>>>(
            c_row_offset, 
            c_row,
            c_atomic,
            c_column,
            c_value, 
            rhs_matrix,  
            output_matrix,
            n1, windows, dimN, mOri, splitk, parts, grid_x, partsize);
        spmm_forward_tf32_csr_v2_kernel_cuda_short<128><<<grid_dim_short, block_dim, 0, stream2>>>(
            c_row_offset_short, 
            c_row_short,
            c_atomic_short,
            c_column_short,
            c_value_short, 
            rhs_matrix,  
            output_matrix,
            n1, windows_short, dimN, mOri, splitk_short, parts_short, grid_x);
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

float spmm_forward_tf32_cuda_kernel_split_32(

    int * c_row_offset,
    int * c_row,
    int * c_atomic,
    int * c_column, 
    float * c_value, 

    int * c_row_offset_short,
    int * c_row_short,
    int * c_atomic_short,
    int * c_column_short, 
    float * c_value_short, 

    float * rhs_matrix,
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

    dim3 grid_dim(grid_x, splitk ,((windows/splitk)+1));
    dim3 grid_dim_short(grid_x, splitk_short ,((windows_short/splitk_short)+1));
    dim3 block_dim(32, 1, 1);
    // int warpSize = 32;
    int sharedmemory = partsize*(sizeof(float)+ sizeof(int));
    cudaStream_t stream1,stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    for(int iter=0; iter<10; ++iter){
        spmm_forward_tf32_csr_v2_kernel_cuda_32<<<grid_dim, block_dim, sharedmemory, stream1>>>(
            c_row_offset, 
            c_row,
            c_atomic,
            c_column,
            c_value, 
            rhs_matrix,  
            output_matrix,
            n1, windows, dimN, mOri, splitk, parts, grid_x, partsize);
        spmm_forward_tf32_csr_v2_kernel_cuda_short_32<<<grid_dim_short, block_dim, 0, stream2>>>(
            c_row_offset_short, 
            c_row_short,
            c_atomic_short,
            c_column_short,
            c_value_short, 
            rhs_matrix,  
            output_matrix,
            n1, windows_short, dimN, mOri, splitk_short, parts_short, grid_x);
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
        spmm_forward_tf32_csr_v2_kernel_cuda_32<<<grid_dim, block_dim, sharedmemory, stream1>>>(
            c_row_offset, 
            c_row,
            c_atomic,
            c_column,
            c_value, 
            rhs_matrix,  
            output_matrix,
            n1, windows, dimN, mOri, splitk, parts, grid_x, partsize);
        spmm_forward_tf32_csr_v2_kernel_cuda_short_32<<<grid_dim_short, block_dim, 0, stream2>>>(
            c_row_offset_short, 
            c_row_short,
            c_atomic_short,
            c_column_short,
            c_value_short, 
            rhs_matrix,  
            output_matrix,
            n1, windows_short, dimN, mOri, splitk_short, parts_short, grid_x);
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

    half2 * rhs_matrix,
    float * output_matrix,

    int partsize,
    const int dimN,
    const int mOri,
    int epoches,
    int parts,
    int parts_short,
    bool swizzle)
{
    // int n1=dimN;
    // if((dimN%16)!=0) n1=((dimN/16)+1)*16;
    // int grid_x = (n1/64)+1;
    // if(n1%64==0) grid_x-=1;
    // int windows =  (parts/4) + 1;
    // if(parts%4==0) windows-=1;
    // int splitk = 0;
    // if(windows<500000) splitk=8;
    // else splitk=((windows/1250000)+1)*20;

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

    dim3 grid_dim(grid_x, splitk ,((windows/splitk)+1));
    dim3 grid_dim_short(grid_x, splitk_short ,((windows_short/splitk_short)+1));
    dim3 block_dim(64, 1, 1);
    // int warpSize = 32;
    int sharedmemory = partsize*(sizeof(float)+ sizeof(int));
    cudaStream_t stream1,stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    for(int iter=0; iter<10; ++iter){
        spmm_forward_fp16_csr_v2_kernel_cuda<128><<<grid_dim, block_dim, sharedmemory, stream1>>>(
            c_row_offset, 
            c_row,
            c_atomic,
            c_column,
            c_value, 
            rhs_matrix,  
            output_matrix,
            n1, windows, dimN, mOri, splitk, parts, grid_x, partsize);
        spmm_forward_fp16_csr_v2_kernel_cuda_short<128><<<grid_dim_short, block_dim, 0, stream2>>>(
            c_row_offset_short, 
            c_row_short,
            c_atomic_short,
            c_column_short,
            c_value_short, 
            rhs_matrix,  
            output_matrix,
            n1, windows_short, dimN, mOri, splitk_short, parts_short, grid_x);
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
        spmm_forward_fp16_csr_v2_kernel_cuda<128><<<grid_dim, block_dim, sharedmemory,stream1>>>(
            c_row_offset, 
            c_row,
            c_atomic,
            c_column,
            c_value, 
            rhs_matrix,  
            output_matrix,
            n1, windows, dimN, mOri, splitk, parts, grid_x, partsize);
        spmm_forward_fp16_csr_v2_kernel_cuda_short<128><<<grid_dim_short, block_dim, 0, stream2>>>(
            c_row_offset_short, 
            c_row_short,
            c_atomic_short,
            c_column_short,
            c_value_short, 
            rhs_matrix,  
            output_matrix,
            n1, windows_short, dimN, mOri, splitk_short, parts_short, grid_x);
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

float spmm_forward_fp16_cuda_kernel_split_32(

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
    // int n1=dimN;
    // if((dimN%16)!=0) n1=((dimN/16)+1)*16;
    // int grid_x = (n1/64)+1;
    // if(n1%64==0) grid_x-=1;
    // int windows =  (parts/4) + 1;
    // if(parts%4==0) windows-=1;
    // int splitk = 0;
    // if(windows<500000) splitk=8;
    // else splitk=((windows/1250000)+1)*20;

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

    dim3 grid_dim(grid_x, splitk ,((windows/splitk)+1));
    dim3 grid_dim_short(grid_x, splitk_short ,((windows_short/splitk_short)+1));
    dim3 block_dim(32, 1, 1);
    cudaStream_t stream1,stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    // int warpSize = 32;
    int sharedmemory = partsize*(sizeof(float)+ sizeof(int));
    for(int iter=0; iter<10; ++iter){
        spmm_forward_fp16_csr_v2_kernel_cuda_32<<<grid_dim, block_dim, sharedmemory, stream1>>>(
            c_row_offset, 
            c_row,
            c_atomic,
            c_column,
            c_value, 
            rhs_matrix,  
            output_matrix,
            n1, windows, dimN, mOri, splitk, parts, grid_x, partsize);
        spmm_forward_fp16_csr_v2_kernel_cuda_short_32<<<grid_dim_short, block_dim, 0, stream2>>>(
            c_row_offset_short, 
            c_row_short,
            c_atomic_short,
            c_column_short,
            c_value_short, 
            rhs_matrix,  
            output_matrix,
            n1, windows_short, dimN, mOri, splitk_short, parts_short, grid_x);
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
        spmm_forward_fp16_csr_v2_kernel_cuda_32<<<grid_dim, block_dim, sharedmemory, stream1>>>(
            c_row_offset, 
            c_row,
            c_atomic,
            c_column,
            c_value, 
            rhs_matrix,  
            output_matrix,
            n1, windows, dimN, mOri, splitk, parts, grid_x, partsize);
        spmm_forward_fp16_csr_v2_kernel_cuda_short_32<<<grid_dim_short, block_dim, 0, stream2>>>(
            c_row_offset_short, 
            c_row_short,
            c_atomic_short,
            c_column_short,
            c_value_short, 
            rhs_matrix,  
            output_matrix,
            n1, windows_short, dimN, mOri, splitk_short, parts_short, grid_x);
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
//tcu cuda
float spmm_forward_tf32_tcu_cuda_kernel(
    int * t_row_offset,
    int * t_blockNew_offset,
    float * t_value, 
    int * t_column, 
    int* t_window_row,
    int * t_atomic,
    int *  t_binary,
    
    int * c_row_offset,
    int * c_row,
    int * c_atomic,
    int * c_column, 
    float * c_value, 

    int * c_row_offset_short,
    int * c_row_short,
    int * c_atomic_short,
    int * c_column_short, 
    float * c_value_short, 

    float * rhs_matrix,
    float * output_matrix,

    const int parts_t,
    const int parts_c,
    const int partsize_c,
    const int parts_c_short,
    const int dimN,
    const int mOri,
    int epoches)
{
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
    dim3 block_dim(64, 1, 1);
    int sharedmemory = partsize_c*(sizeof(float)+ sizeof(int));
    float2 * rhs_matrix_c = reinterpret_cast<float2 *>(rhs_matrix);
    cudaStream_t stream1,stream2,stream3;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);
    for(int iter=0; iter<10; ++iter){
        spmm_forward_tf32_csr_v2_kernel_tcu<64><<<grid_dim_t, block_dim_t, 0, stream1>>>(
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

        spmm_forward_tf32_csr_v2_kernel_cuda<128><<<grid_dim_c, block_dim, sharedmemory, stream2>>>(
            c_row_offset, 
            c_row,
            c_atomic,
            c_column,
            c_value, 
            rhs_matrix_c,  
            output_matrix,
            n1_c, windows, dimN, mOri, splitk_c, parts_c, grid_x_c,partsize_c);
        
        spmm_forward_tf32_csr_v2_kernel_cuda_short<128><<<grid_dim_c_short, block_dim, 0, stream3>>>(
            c_row_offset_short, 
            c_row_short,
            c_atomic_short,
            c_column_short,
            c_value_short, 
            rhs_matrix_c,  
            output_matrix,
            n1_c, windows_short, dimN, mOri, splitk_c_short, parts_c_short, grid_x_c);
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
    cudaDeviceSynchronize();
    for(int iter=0; iter<epoches; ++iter){
        spmm_forward_tf32_csr_v2_kernel_tcu<64><<<grid_dim_t, block_dim_t, 0, stream1>>>(
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

        spmm_forward_tf32_csr_v2_kernel_cuda<128><<<grid_dim_c, block_dim, sharedmemory, stream2>>>(
            c_row_offset, 
            c_row,
            c_atomic,
            c_column,
            c_value, 
            rhs_matrix_c,  
            output_matrix,
            n1_c, windows, dimN, mOri, splitk_c, parts_c, grid_x_c,partsize_c);
        
        spmm_forward_tf32_csr_v2_kernel_cuda_short<128><<<grid_dim_c_short, block_dim, 0, stream3>>>(
            c_row_offset_short, 
            c_row_short,
            c_atomic_short,
            c_column_short,
            c_value_short, 
            rhs_matrix_c,  
            output_matrix,
            n1_c, windows_short, dimN, mOri, splitk_c_short, parts_c_short, grid_x_c);
    }
    cudaEventRecord(spmm_end);
    cudaEventSynchronize(spmm_end);
    cudaEventElapsedTime(&spmm_ms, spmm_start, spmm_end);

    cudaEventDestroy(spmm_start);
    cudaEventDestroy(spmm_end);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaStreamDestroy(stream3);

    //计算时间 ms
    spmm_ms_avg = spmm_ms/(float)epoches;


    return spmm_ms_avg;
}
float spmm_forward_tf32_tcu_cuda_kernel_32(
    int * t_row_offset,
    int * t_blockNew_offset,
    float * t_value, 
    int * t_column, 
    int* t_window_row,
    int * t_atomic,
    int *  t_binary,
    
    int * c_row_offset,
    int * c_row,
    int * c_atomic,
    int * c_column, 
    float * c_value, 

    int * c_row_offset_short,
    int * c_row_short,
    int * c_atomic_short,
    int * c_column_short, 
    float * c_value_short, 

    float * rhs_matrix,
    float * output_matrix,

    const int parts_t,
    const int parts_c,
    const int partsize_c,
    const int parts_c_short,
    const int dimN,
    const int mOri,
    int epoches)
{
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
    dim3 block_dim(32, 1, 1);
    int sharedmemory = partsize_c*(sizeof(float)+ sizeof(int));
    cudaStream_t stream1,stream2,stream3;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);
    for(int iter=0; iter<10; ++iter){
        spmm_forward_tf32_csr_v2_kernel_tcu<64><<<grid_dim_t, block_dim_t, 0, stream1>>>(
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

        spmm_forward_tf32_csr_v2_kernel_cuda_32<<<grid_dim_c, block_dim, sharedmemory, stream2>>>(
            c_row_offset, 
            c_row,
            c_atomic,
            c_column,
            c_value, 
            rhs_matrix,  
            output_matrix,
            n1_c, windows, dimN, mOri, splitk_c, parts_c, grid_x_c,partsize_c);
        
        spmm_forward_tf32_csr_v2_kernel_cuda_short_32<<<grid_dim_c_short, block_dim, 0, stream3>>>(
            c_row_offset_short, 
            c_row_short,
            c_atomic_short,
            c_column_short,
            c_value_short, 
            rhs_matrix,  
            output_matrix,
            n1_c, windows_short, dimN, mOri, splitk_c_short, parts_c_short, grid_x_c);
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
    cudaDeviceSynchronize();
    for(int iter=0; iter<epoches; ++iter){
        spmm_forward_tf32_csr_v2_kernel_tcu<64><<<grid_dim_t, block_dim_t, 0, stream1>>>(
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

        spmm_forward_tf32_csr_v2_kernel_cuda_32<<<grid_dim_c, block_dim, sharedmemory, stream2>>>(
            c_row_offset, 
            c_row,
            c_atomic,
            c_column,
            c_value, 
            rhs_matrix,  
            output_matrix,
            n1_c, windows, dimN, mOri, splitk_c, parts_c, grid_x_c,partsize_c);
        
        spmm_forward_tf32_csr_v2_kernel_cuda_short_32<<<grid_dim_c_short, block_dim, 0, stream3>>>(
            c_row_offset_short, 
            c_row_short,
            c_atomic_short,
            c_column_short,
            c_value_short, 
            rhs_matrix,  
            output_matrix,
            n1_c, windows_short, dimN, mOri, splitk_c_short, parts_c_short, grid_x_c);
    }
    cudaEventRecord(spmm_end);
    cudaEventSynchronize(spmm_end);
    cudaEventElapsedTime(&spmm_ms, spmm_start, spmm_end);

    cudaEventDestroy(spmm_start);
    cudaEventDestroy(spmm_end);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaStreamDestroy(stream3);

    //计算时间 ms
    spmm_ms_avg = spmm_ms/(float)epoches;


    return spmm_ms_avg;
}


//tcu cuda 
float spmm_forward_fp16_tcu_cuda_kernel(
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
    dim3 block_dim(64, 1, 1);
    int sharedmemory = partsize_c*(sizeof(half)+ sizeof(int));
    half2 * rhs_matrix_c = reinterpret_cast<half2 *>(rhs_matrix);
    cudaStream_t stream1,stream2,stream3;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);
    for(int iter=0; iter<10; ++iter){
        spmm_forward_fp16_csr_v2_kernel_tcu<64><<<grid_dim_t, block_dim_t, 0, stream1>>>(
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

        spmm_forward_fp16_csr_v2_kernel_cuda<128><<<grid_dim_c, block_dim, sharedmemory, stream2>>>(
            c_row_offset, 
            c_row,
            c_atomic,
            c_column,
            c_value, 
            rhs_matrix_c,  
            output_matrix,
            n1_c, windows, dimN, mOri, splitk_c, parts_c, grid_x_c,partsize_c);
        
        spmm_forward_fp16_csr_v2_kernel_cuda_short<128><<<grid_dim_c_short, block_dim, 0, stream3>>>(
            c_row_offset_short, 
            c_row_short,
            c_atomic_short,
            c_column_short,
            c_value_short, 
            rhs_matrix_c,  
            output_matrix,
            n1_c, windows_short, dimN, mOri, splitk_c_short, parts_c_short, grid_x_c);
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
        spmm_forward_fp16_csr_v2_kernel_tcu<64><<<grid_dim_t, block_dim_t, 0, stream1>>>(
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

        spmm_forward_fp16_csr_v2_kernel_cuda<128><<<grid_dim_c, block_dim, sharedmemory,stream2>>>(
            c_row_offset, 
            c_row,
            c_atomic,
            c_column,
            c_value, 
            rhs_matrix_c,  
            output_matrix,
            n1_c, windows, dimN, mOri, splitk_c, parts_c, grid_x_c,partsize_c);
        
        spmm_forward_fp16_csr_v2_kernel_cuda_short<128><<<grid_dim_c_short, block_dim,0,stream3>>>(
            c_row_offset_short, 
            c_row_short,
            c_atomic_short,
            c_column_short,
            c_value_short, 
            rhs_matrix_c,  
            output_matrix,
            n1_c, windows_short, dimN, mOri, splitk_c_short, parts_c_short, grid_x_c);
    }
    cudaEventRecord(spmm_end);
    cudaEventSynchronize(spmm_end);
    cudaEventElapsedTime(&spmm_ms, spmm_start, spmm_end);
    cudaEventDestroy(spmm_start);
    cudaEventDestroy(spmm_end);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaStreamDestroy(stream3);

    //计算时间 ms
    spmm_ms_avg = spmm_ms/(float)epoches;


    return spmm_ms_avg;
}

float spmm_forward_fp16_tcu_cuda_kernel_32(
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
    dim3 block_dim(32, 1, 1);
    int sharedmemory = partsize_c*(sizeof(half)+ sizeof(int));
    cudaStream_t stream1,stream2,stream3;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);
    for(int iter=0; iter<10; ++iter){
        spmm_forward_fp16_csr_v2_kernel_tcu<64><<<grid_dim_t, block_dim_t, 0, stream1>>>(
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

        spmm_forward_fp16_csr_v2_kernel_cuda_32<<<grid_dim_c, block_dim, sharedmemory, stream2>>>(
            c_row_offset, 
            c_row,
            c_atomic,
            c_column,
            c_value, 
            rhs_matrix,  
            output_matrix,
            n1_c, windows, dimN, mOri, splitk_c, parts_c, grid_x_c,partsize_c);
        
        spmm_forward_fp16_csr_v2_kernel_cuda_short_32<<<grid_dim_c_short, block_dim, 0, stream3>>>(
            c_row_offset_short, 
            c_row_short,
            c_atomic_short,
            c_column_short,
            c_value_short, 
            rhs_matrix,  
            output_matrix,
            n1_c, windows_short, dimN, mOri, splitk_c_short, parts_c_short, grid_x_c);
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
        spmm_forward_fp16_csr_v2_kernel_tcu<64><<<grid_dim_t, block_dim_t, 0, stream1>>>(
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

        spmm_forward_fp16_csr_v2_kernel_cuda_32<<<grid_dim_c, block_dim, sharedmemory,stream2>>>(
            c_row_offset, 
            c_row,
            c_atomic,
            c_column,
            c_value, 
            rhs_matrix,  
            output_matrix,
            n1_c, windows, dimN, mOri, splitk_c, parts_c, grid_x_c,partsize_c);
        
        spmm_forward_fp16_csr_v2_kernel_cuda_short_32<<<grid_dim_c_short, block_dim,0,stream3>>>(
            c_row_offset_short, 
            c_row_short,
            c_atomic_short,
            c_column_short,
            c_value_short, 
            rhs_matrix,  
            output_matrix,
            n1_c, windows_short, dimN, mOri, splitk_c_short, parts_c_short, grid_x_c);
    }
    cudaEventRecord(spmm_end);
    cudaEventSynchronize(spmm_end);
    cudaEventElapsedTime(&spmm_ms, spmm_start, spmm_end);
    cudaEventDestroy(spmm_start);
    cudaEventDestroy(spmm_end);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaStreamDestroy(stream3);

    //计算时间 ms
    spmm_ms_avg = spmm_ms/(float)epoches;


    return spmm_ms_avg;
}