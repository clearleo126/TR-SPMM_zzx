#include <stdio.h>
#include <mma.h>
#include <cstdint>
#include <iostream>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include "./sddmm_utils/dense_tile.h"
#include "./sddmm_utils/compute.h"
#include "./sddmm_utils/output_tile.h"
#include <cuda_runtime.h>

//spmm_forward_tf32_csr_kernel, two kernel
__global__ void sddmm_forward_tf32_csr_v2_kernel_cuda(
    const int* __restrict__ c_row_offset,
    const int* __restrict__ c_row,
    const int* __restrict__ c_column,
    const float* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix,
    int parts,
    int nOri,
    int mOri,
    int splitk)
{
    //假如每个block中1个warp, 每个warp负责计算稀疏矩阵的一行中的非零元
    int c_part_offset_vec = ((blockIdx.z*splitk)+blockIdx.y);
    if(c_part_offset_vec>=parts) return;
    int warpin_id = threadIdx.x%32;
    int warp_id = threadIdx.x/32;
    //当前cuda需要计算的行
    int out_row_offset = __ldg(c_row + c_part_offset_vec);
    if(out_row_offset<mOri)
    {
        int c_row_offset_vec = __ldg(c_row_offset + c_part_offset_vec); 
        int nonzeros = __ldg(c_row_offset + c_part_offset_vec + 1) -  c_row_offset_vec;
        extern __shared__  float res_share[];
        float * res_share_ = res_share + warpin_id*nonzeros;
        int * sparse_col = (int *) & res_share[32*nonzeros];
        const float * rhs_matrix_a = rhs_matrix + out_row_offset*nOri;

        int iters = (nOri/32);
        // if((nOri%32)==0) iters-=1;
        //进行cuda计算
        if(nonzeros!=0) 
        {  
            //32个线程一起搬运
            if(warpin_id<nonzeros){
                *(sparse_col + threadIdx.x) = __ldg(c_column + c_row_offset_vec + threadIdx.x);               
            }
            for(int j=0;j<nonzeros;j++)
            *(res_share_ + j) = 0.0;
            
            __syncthreads();

            for(int i=0; i<iters; i++){
                // if((i*32+warpin_id) < nOri){
                    float a = __ldg(rhs_matrix_a + warpin_id*iters + i);
                    for(int j=0;j<nonzeros;j++){
                        *(res_share_ + j) +=  a * __ldg(rhs_matrix + sparse_col[j]*nOri + warpin_id*iters + i);
                    }
                // }
            }
            __syncthreads();
            if(warpin_id<nonzeros){
                //每个线程负责处理一个nnz的结果
                float res = 0.0;
                float * temp = res_share+warpin_id;
                for(int i=0; i<32; i++)
                {
                    res+= *(temp);
                    temp+=nonzeros;
                }
                *(output_matrix + c_row_offset_vec + warpin_id) = res;
                // if(c_part_offset_vec==0)
                // {
                //     printf("%d %f\n", nonzeros,res);
                // }
            }
        }
    } 
}

//32
__global__ void sddmm_forward_tf32_csr_v2_kernel_cuda_32(
    const int* __restrict__ c_row_offset,
    const int* __restrict__ c_row,
    const int* __restrict__ c_column,
    const float4* __restrict__ lhs_matrix,
    const float4* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix,
    int parts,
    int nOri,
    int mOri,
    int splitk)
{
    //假如每个block中1个warp, 每个warp负责计算稀疏矩阵的一行中的非零元
    int c_part_offset_vec = ((blockIdx.z*splitk)+blockIdx.y);
    if(c_part_offset_vec>=parts) return;
    int warpin_id = threadIdx.x%32;
    int warp_id = threadIdx.x/32;
    //当前cuda需要计算的行
    int out_row_offset = __ldg(c_row + c_part_offset_vec);
    if(out_row_offset<mOri)
    {
        int c_row_offset_vec = __ldg(c_row_offset + c_part_offset_vec); 
        int nonzeros = __ldg(c_row_offset + c_part_offset_vec + 1) -  c_row_offset_vec;
        __shared__  float4 lhs_share[8];
        const float4 * lhs_matrix_ = lhs_matrix + out_row_offset*nOri/4;
        //加载对应的行
        if(warpin_id<8){
            *(lhs_share + warpin_id) = __ldg(lhs_matrix_ + warpin_id);
        }
        __syncthreads();
        //进行cuda计算
        if(nonzeros!=0) 
        {  
            //32个线程一起搬运
            for(int i=warpin_id; i<nonzeros; i+=32){
                int col = __ldg(c_column + c_row_offset_vec + i);  
                const float4 * rhs_matrix_ = rhs_matrix + col*nOri/4;
                float res = 0.0;
                //每次fetch float4
                float4 a= make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                float4 b= make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                for(int j=0; j<8; j++){
                    b = __ldg(rhs_matrix_ + j);
                    a = lhs_share[j];
                    res += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
                }
                //写回global
                *(output_matrix + c_row_offset_vec + i) = res;
            }
        }
    } 
}

//128
__global__ void sddmm_forward_tf32_csr_v2_kernel_cuda_128(
    const int* __restrict__ c_row_offset,
    const int* __restrict__ c_row,
    const int* __restrict__ c_column,
    const float* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix,
    int parts,
    int nOri,
    int mOri,
    int splitk)
{
    const float4 * rhs_matrix_ = reinterpret_cast< const float4 *>(rhs_matrix);
    //假如每个block中1个warp, 每个warp负责计算稀疏矩阵的一行中的非零元
    int c_part_offset_vec = ((blockIdx.z*splitk)+blockIdx.y);
    if(c_part_offset_vec>=parts) return;
    int warpin_id = threadIdx.x%32;
    int warp_id = threadIdx.x/32;
    //当前cuda需要计算的行
    int out_row_offset = __ldg(c_row + c_part_offset_vec);
    if(out_row_offset<mOri)
    {
        int c_row_offset_vec = __ldg(c_row_offset + c_part_offset_vec); 
        int nonzeros = __ldg(c_row_offset + c_part_offset_vec + 1) -  c_row_offset_vec;
        extern __shared__  float res_share[];
        float * res_share_ = res_share + warpin_id*nonzeros;
        int * sparse_col = (int *) & res_share[32*nonzeros];
        const float * rhs_matrix_a = rhs_matrix + out_row_offset*nOri;

        int iters = 4;
        // if((nOri%32)==0) iters-=1;
        //进行cuda计算
        if(nonzeros!=0) 
        {  
            //32个线程一起搬运
            if(warpin_id<nonzeros){
                *(sparse_col + threadIdx.x) = __ldg(c_column + c_row_offset_vec + threadIdx.x);               
            }
            for(int j=0;j<nonzeros;j++)
            *(res_share_ + j) = 0.0;
            
            __syncthreads();

            float4 lhs = __ldg(rhs_matrix_ + warpin_id);
            for(int j=0;j<nonzeros;j++){
                float4 rhs = __ldg(rhs_matrix_ + sparse_col[j]*(nOri/4) + warpin_id);
                *(res_share_ + j) =  (lhs.x * rhs.x) + (lhs.y *  rhs.y) + (lhs.z *  rhs.z) + (lhs.w *  rhs.w);
            }
            __syncthreads();
            if(warpin_id<nonzeros){
                //每个线程负责处理一个nnz的结果
                float res = 0.0;
                float * temp = res_share+warpin_id;
                for(int i=0; i<32; i++)
                {
                    res+= *(temp);
                    temp+=nonzeros;
                }
                *(output_matrix + c_row_offset_vec + warpin_id) = res;
            }
        }
    } 
}
__global__ void sddmm_forward_tf32_csr_v2_kernel_cuda_v2(
    const int* __restrict__ c_row_offset,
    const int* __restrict__ c_row,
    const int* __restrict__ c_column,
    const float* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix,
    int parts,
    int nOri,
    int mOri,
    int splitk)
{
    //假如每个block中1个warp, 每个warp负责计算稀疏矩阵的一行中的非零元
    int c_part_offset_vec = ((blockIdx.z*splitk)+blockIdx.y);
    if(c_part_offset_vec>=parts) return;
    int warpin_id = threadIdx.x%32;
    int warp_id = threadIdx.x/32;
    //当前cuda需要计算的行
    int out_row_offset = __ldg(c_row + c_part_offset_vec);
    if(out_row_offset<mOri)
    {
        int c_row_offset_vec = __ldg(c_row_offset + c_part_offset_vec); 
        int nonzeros = __ldg(c_row_offset + c_part_offset_vec + 1) -  c_row_offset_vec;
        const float * rhs_matrix_a = rhs_matrix + out_row_offset*nOri;
        int col = __ldg(c_column + c_row_offset_vec + warpin_id);               
        const float * rhs_matrix_b = rhs_matrix + col*nOri;
        //进行cuda计算
        if(nonzeros!=0) 
        {  
            if(warpin_id<nonzeros){
                float res = 0.0;
                for(int i=0;i<nOri;i++)
                {
                    res+= __ldg(rhs_matrix_a+i) * __ldg(rhs_matrix_b+i);
                }
                *(output_matrix + c_row_offset_vec + warpin_id) = res;
            }
        }
    } 
}

__global__ void sddmm_forward_fp16_csr_v2_kernel_cuda_v2(
    const int* __restrict__ c_row_offset,
    const int* __restrict__ c_row,
    const int* __restrict__ c_column,
    const half* __restrict__ rhs_matrix,
    half* __restrict__ output_matrix,
    int parts,
    int nOri,
    int mOri,
    int splitk)
{
    //假如每个block中1个warp, 每个warp负责计算稀疏矩阵的一行中的非零元
    int c_part_offset_vec = ((blockIdx.z*splitk)+blockIdx.y);
    if(c_part_offset_vec>=parts) return;
    int warpin_id = threadIdx.x%32;
    int warp_id = threadIdx.x/32;
    //当前cuda需要计算的行
    int out_row_offset = __ldg(c_row + c_part_offset_vec);
    if(out_row_offset<mOri)
    {
        int c_row_offset_vec = __ldg(c_row_offset + c_part_offset_vec); 
        int nonzeros = __ldg(c_row_offset + c_part_offset_vec + 1) -  c_row_offset_vec;
        const half * rhs_matrix_a = rhs_matrix + out_row_offset*nOri;
        int col = __ldg(c_column + c_row_offset_vec + warpin_id);               
        const half * rhs_matrix_b = rhs_matrix + col*nOri;
        //进行cuda计算
        if(nonzeros!=0) 
        {  
            if(warpin_id<nonzeros){
                half res = __float2half(0.0);
                for(int i=0;i<nOri;i++)
                {
                    res =  __hadd(res, __hmul(__ldg(rhs_matrix_a+i) , __ldg(rhs_matrix_b+i)));
                }
                *(output_matrix + c_row_offset_vec + warpin_id) = res;
            }
        }
    } 
}

// fp16
__global__ void sddmm_forward_fp16_csr_v2_kernel_cuda(
    const int* __restrict__ c_row_offset,
    const int* __restrict__ c_row,
    const int* __restrict__ c_column,
    const double* __restrict__ lhs_matrix,
    const double* __restrict__ rhs_matrix,
    half* __restrict__ output_matrix,
    int parts,
    int nOri,
    int mOri,
    int splitk)
{
    //假如每个block中1个warp, 每个warp负责计算稀疏矩阵的一行中的非零元
    int c_part_offset_vec = ((blockIdx.z*splitk)+blockIdx.y);
    if(c_part_offset_vec>=parts) return;
    int warpin_id = threadIdx.x%32;
    int warp_id = threadIdx.x/32;
    //当前cuda需要计算的行
    int out_row_offset = __ldg(c_row + c_part_offset_vec);
    if(out_row_offset<mOri)
    {
        int c_row_offset_vec = __ldg(c_row_offset + c_part_offset_vec); 
        int nonzeros = __ldg(c_row_offset + c_part_offset_vec + 1) -  c_row_offset_vec;
        __shared__ double lhs_share[8];
        const double * lhs_matrix_ = lhs_matrix + out_row_offset*nOri/4;
        //加载对应的行
        if(warpin_id<8){
            *(lhs_share + warpin_id) = __ldg(lhs_matrix_ + warpin_id);
        }
        __syncthreads();
        //进行cuda计算
        if(nonzeros!=0) 
        {  
            //32个线程一起搬运
            for(int i=warpin_id; i<nonzeros; i+=32){
                int col = __ldg(c_column + c_row_offset_vec + i);  
                const double * rhs_matrix_ = rhs_matrix + col*nOri/4;
                float res = 0.0;
                //每次fetch double
                double a [1] = {0.0};
                double b [1] = {0.0};
                half * aa = reinterpret_cast<half *>(a);
                half * bb = reinterpret_cast<half *>(b);
                for(int j=0; j<8; j++){
                    b[0] = __ldg(rhs_matrix_ + j);
                    a[0] = lhs_share[j];
                    for(int k=0;k<4;k++)
                    {
                        res += (__half2float(aa[k]) * __half2float(bb[k]));
                    }
                }
                //写回global
                *(output_matrix + c_row_offset_vec + i) = __float2half(res);
            }
        }
    }
    
}

// __global__ void sddmm_forward_fp16_csr_v2_kernel_cuda_128(
//     const int* __restrict__ c_row_offset,
//     const int* __restrict__ c_row,
//     const int* __restrict__ c_column,
//     const half2* __restrict__ rhs_matrix,
//     half* __restrict__ output_matrix,
//     int parts,
//     int nOri,
//     int mOri,
//     int splitk)
// {
//     //假如每个block中1个warp, 每个warp负责计算稀疏矩阵的一行中的非零元
//     int c_part_offset_vec = ((blockIdx.z*splitk)+blockIdx.y);
//     if(c_part_offset_vec>=parts) return;
//     int warpin_id = threadIdx.x%32;
//     int warp_id = threadIdx.x/32;
//     //当前cuda需要计算的行
//     int out_row_offset = __ldg(c_row + c_part_offset_vec);
//     if(out_row_offset<mOri)
//     {
//         int c_row_offset_vec = __ldg(c_row_offset + c_part_offset_vec); 
//         int nonzeros = __ldg(c_row_offset + c_part_offset_vec + 1) -  c_row_offset_vec;
//         extern __shared__ half res_share_fp16[];
//         half * res_share_ = res_share_fp16 + warpin_id*nonzeros;
//         int * sparse_col = (int *) & res_share_fp16[32*nonzeros];
//         const half * rhs_matrix_a = rhs_matrix + out_row_offset*nOri;
//         int iters = (nOri/32) + 1;
//         if((nOri%32)>0) iters-=1;
//         //进行cuda计算
//         if(nonzeros!=0) 
//         {  
//             //32个线程一起搬运
//             if(warpin_id<nonzeros){
//                 *(sparse_col + threadIdx.x) = __ldg(c_column + c_row_offset_vec + threadIdx.x);               
//             }
//             for(int j=0;j<nonzeros;j++)
//             *(res_share_ + j) = __float2half(0.0);
            
//             __syncthreads();

//             for(int i=0; i<iters; i++){
//                 if((i*32+warpin_id) < nOri){
//                     half a = __ldg(rhs_matrix_a + i*32 + warpin_id);
//                     for(int j=0;j<nonzeros;j++){
//                         half b = __ldg(rhs_matrix + sparse_col[j]*nOri + i*32 + warpin_id);
//                         float c = (__half2float(a) * __half2float(b));
//                         *(res_share_ + j) =__hadd(*(res_share_ + j), __float2half(c));
//                     }
//                 }
//             }
//             __syncthreads();
            
//             if(warpin_id<nonzeros){
//                 //每个线程负责处理一个nnz的结果
//                 half res = __float2half(0.0);
//                 half * temp = res_share_fp16+warpin_id;
//                 for(int i=0; i<32; i++)
//                 {
//                     res = __hadd(res, *(temp));
//                     temp+=nonzeros;
//                 }
//                 *(output_matrix + c_row_offset_vec + warpin_id) = res;
//             }
//         }
//     } 
    
    
// }


//cuda 长短行
float sddmm_forward_tf32_cuda_kernel_split(

    int * c_row_offset,
    int * c_row,
    int * c_column, 

    float * rhs_matrix,
    float * output_matrix,

    const int nOri,
    const int mOri,
    int epoches,
    int parts)
{
    int splitk = 0;
    if(parts<500000) splitk=8;
    else splitk=((parts/1250000)+1)*20;

    dim3 grid_dim(1, splitk ,((parts/splitk)+1));
    dim3 block_dim(32, 1, 1);

    int sharedmemory = 32*32*sizeof(float)+ 32*sizeof(int);
    for(int iter=0; iter<10; ++iter){
        sddmm_forward_tf32_csr_v2_kernel_cuda<<<grid_dim, block_dim, sharedmemory>>>(
            c_row_offset, 
            c_row,
            c_column,
            rhs_matrix,  
            output_matrix,
            parts, nOri, mOri, splitk);
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
        sddmm_forward_tf32_csr_v2_kernel_cuda<<<grid_dim, block_dim, sharedmemory>>>(
            c_row_offset, 
            c_row,
            c_column,
            rhs_matrix,  
            output_matrix,
            parts, nOri, mOri, splitk);
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


//32
float sddmm_forward_tf32_cuda_kernel_split_32(

    int * c_row_offset,
    int * c_row,
    int * c_column, 

    float * rhs_matrix,
    float * output_matrix,

    const int nOri,
    const int mOri,
    int epoches,
    int parts)
{
    int splitk = 0;
    if(parts<500000) splitk=8;
    else splitk=((parts/1250000)+1)*20;

    dim3 grid_dim(1, splitk ,((parts/splitk)+1));
    dim3 block_dim(32, 1, 1);

    // int sharedmemory = 32*32*sizeof(float)+ 32*sizeof(int);
    for(int iter=0; iter<10; ++iter){
        sddmm_forward_tf32_csr_v2_kernel_cuda_32<<<grid_dim, block_dim>>>(
            c_row_offset, 
            c_row,
            c_column,
            reinterpret_cast<float4 *>(rhs_matrix), 
            reinterpret_cast<float4 *>(rhs_matrix),
            output_matrix,
            parts, nOri, mOri, splitk);
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
        sddmm_forward_tf32_csr_v2_kernel_cuda_32<<<grid_dim, block_dim>>>(
            c_row_offset, 
            c_row,
            c_column,
            reinterpret_cast<float4 *>(rhs_matrix), 
            reinterpret_cast<float4 *>(rhs_matrix),
            output_matrix,
            parts, nOri, mOri, splitk);
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

//128
float sddmm_forward_tf32_cuda_kernel_split_128(

    int * c_row_offset,
    int * c_row,
    int * c_column, 

    float * rhs_matrix,
    float * output_matrix,

    const int nOri,
    const int mOri,
    int epoches,
    int parts)
{
    int splitk = 0;
    if(parts<500000) splitk=8;
    else splitk=((parts/1250000)+1)*20;

    dim3 grid_dim(1, splitk ,((parts/splitk)+1));
    dim3 block_dim(32, 1, 1);

    int sharedmemory = 32*32*sizeof(float)+ 32*sizeof(int);
    for(int iter=0; iter<10; ++iter){
        sddmm_forward_tf32_csr_v2_kernel_cuda_128<<<grid_dim, block_dim, sharedmemory>>>(
            c_row_offset, 
            c_row,
            c_column,
            rhs_matrix,  
            output_matrix,
            parts, nOri, mOri, splitk);
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
        sddmm_forward_tf32_csr_v2_kernel_cuda_128<<<grid_dim, block_dim, sharedmemory>>>(
            c_row_offset, 
            c_row,
            c_column,
            rhs_matrix,  
            output_matrix,
            parts, nOri, mOri, splitk);
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


float sddmm_forward_tf32_cuda_kernel_split_navie(

    int * c_row_offset,
    int * c_row,
    int * c_column, 

    float * rhs_matrix,
    float * output_matrix,

    const int nOri,
    const int mOri,
    int epoches,
    int parts)
{
    int splitk = 0;
    if(parts<500000) splitk=8;
    else splitk=((parts/1250000)+1)*20;

    dim3 grid_dim(1, splitk ,((parts/splitk)+1));
    dim3 block_dim(32, 1, 1);

    int sharedmemory = 32*32*sizeof(float)+ 32*sizeof(int);
    for(int iter=0; iter<10; ++iter){
        sddmm_forward_tf32_csr_v2_kernel_cuda_v2<<<grid_dim, block_dim>>>(
            c_row_offset, 
            c_row,
            c_column,
            rhs_matrix,  
            output_matrix,
            parts, nOri, mOri, splitk);
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
        sddmm_forward_tf32_csr_v2_kernel_cuda_v2<<<grid_dim, block_dim>>>(
            c_row_offset, 
            c_row,
            c_column,
            rhs_matrix,  
            output_matrix,
            parts, nOri, mOri, splitk);
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

float sddmm_forward_fp16_cuda_kernel_split_navie(

    int * c_row_offset,
    int * c_row,
    int * c_column, 

    half * rhs_matrix,
    half * output_matrix,

    const int nOri,
    const int mOri,
    int epoches,
    int parts)
{
    int splitk = 0;
    if(parts<500000) splitk=8;
    else splitk=((parts/1250000)+1)*20;

    dim3 grid_dim(1, splitk ,((parts/splitk)+1));
    dim3 block_dim(32, 1, 1);

    int sharedmemory = 32*32*sizeof(half)+ 32*sizeof(int);
    for(int iter=0; iter<10; ++iter){
        sddmm_forward_fp16_csr_v2_kernel_cuda_v2<<<grid_dim, block_dim>>>(
            c_row_offset, 
            c_row,
            c_column,
            rhs_matrix,  
            output_matrix,
            parts, nOri, mOri, splitk);
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
        sddmm_forward_fp16_csr_v2_kernel_cuda_v2<<<grid_dim, block_dim>>>(
            c_row_offset, 
            c_row,
            c_column,
            rhs_matrix,  
            output_matrix,
            parts, nOri, mOri, splitk);
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
//cuda 长短行 fp16
float sddmm_forward_fp16_cuda_kernel_split(

    int * c_row_offset,
    int * c_row,
    int * c_column, 

    half * rhs_matrix,
    half * output_matrix,

    const int nOri,
    const int mOri,
    int epoches,
    int parts)
{
    int splitk = 0;
    if(parts<500000) splitk=8;
    else splitk=((parts/1250000)+1)*20;

    dim3 grid_dim(1, splitk ,((parts/splitk)+1));
    dim3 block_dim(32, 1, 1);

    int sharedmemory = 32*32*sizeof(half)+ 32*sizeof(int);
    for(int iter=0; iter<10; ++iter){
        sddmm_forward_fp16_csr_v2_kernel_cuda<<<grid_dim, block_dim, sharedmemory>>>(
            c_row_offset, 
            c_row,
            c_column,
            reinterpret_cast<double *>(rhs_matrix), 
            reinterpret_cast<double *>(rhs_matrix),
            output_matrix,
            parts, nOri, mOri, splitk);
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
        sddmm_forward_fp16_csr_v2_kernel_cuda<<<grid_dim, block_dim, sharedmemory>>>(
            c_row_offset, 
            c_row,
            c_column,
            reinterpret_cast<double *>(rhs_matrix), 
            reinterpret_cast<double *>(rhs_matrix),
            output_matrix,
            parts, nOri, mOri, splitk);
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

__global__ void sddmm_forward_tf32_binary_v2_kernel_tcu(
    const int* __restrict__ t_window_offset,
    const int* __restrict__ t_block_offset,
    const int* __restrict__ t_column,
    const long* __restrict__ t_binary,
    const int* t_window_row,
    const float* __restrict__ lhs_matrix,
    const float4* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix,
    int windows,
    int nOri,
    int mOri,
    int splitk,
    int warps)
{
    int m_index_vec = (blockIdx.z*splitk)+blockIdx.y;
    if(m_index_vec>=windows)
    return;

    int t_win_offset = __ldg(t_window_offset + m_index_vec);
    int tcu_blocks = __ldg(t_window_offset + m_index_vec + 1) - t_win_offset; 
    if(tcu_blocks==0) return;
    int id=blockIdx.x*warps+(threadIdx.x/32);
    if(id>= tcu_blocks) return;
    int warp_id = threadIdx.x>>5;
    int warpin_id = threadIdx.x%32;
    float output_fragment[4] = {0,0,0,0};
    int col=__ldg(t_column + t_win_offset*16 + (id*16) + (warpin_id/2));
    // int col1=*(t_column + t_win_offset*16 + (id*16) + (warpin_id/4));
    // int col2=*(t_column + t_win_offset*16 + (id*16) + (warpin_id/4) +8);
    int cur_m_index_vec = __ldg(t_window_row + m_index_vec);
    int row=cur_m_index_vec*8 + (warpin_id/4);
    const float4 *r_matrix = rhs_matrix + col*nOri/4 + (warpin_id%2);
    const float *l_matrix = lhs_matrix + row*nOri;
    int steps = nOri/8;
    // int residue = nOri%8;
    //warps = 4
    __shared__ float4 block_a[32*4]; 
    // __shared__ float2 block_b[32];

    // float4 * block_a_cur = block_a + 32*warp_id;
    mmaComputeUtils_tf32_gen_v2 computer(
        l_matrix,
        r_matrix, 
        block_a + 32*warp_id,
        output_fragment, 
        threadIdx.x);

    if(steps>0)
    for(int i = 0; i < steps; i++){
        computer.TileMAC(nOri, col, mOri, row);
        // __syncthreads();
        __syncwarp();
    }
    // if(residue>0)
    //     computer.TileMACResidue(residue, nOri, col, col1, mOri, row);

    // if(row<mOri)
    // {
        int offset = ((warpin_id%4)*2*8) + (warpin_id/4);
        int offset1 = offset+8;
        long binary_1 = __ldg(t_binary + t_win_offset*2 + id*2);
        long binary_2 = __ldg(t_binary + t_win_offset*2 + id*2 + 1);
        long temp_0 = (binary_1 >> offset);
        long temp_1 = (binary_1 >> offset1);
        long temp_2 = (binary_2 >> offset);
        long temp_3 = (binary_2 >> offset1);
        long a= 1;
        long mask = (a << offset);
        long mask1 = (a << offset1);

        int fifthBit_0 = ((temp_0) & 1);
        int fifthBit_1 = ((temp_1) & 1);
        int fifthBit_2 = ((temp_2) & 1);
        int fifthBit_3 = ((temp_3) & 1);


        int value_offset = __ldg(t_block_offset + t_win_offset + id);
        float * output_matrix_1 = output_matrix + value_offset;
        float * output_matrix_2 = output_matrix_1 + __popcll(binary_1);
        if(fifthBit_0 == 1)
        {
            int block_offset_1 = __popcll(binary_1 & (mask-1));
            *(output_matrix_1 + block_offset_1) = output_fragment[0];
        }
        if(fifthBit_1 == 1)
        {
            int block_offset_1 = __popcll(binary_1 & (mask1-1));
            *(output_matrix_1 + block_offset_1 ) = output_fragment[1];
        }
        if(fifthBit_2 == 1)
        {
            int block_offset_2 = __popcll(binary_2 & (mask-1));
            *(output_matrix_2 + block_offset_2) = output_fragment[2];
        }
        if(fifthBit_3 == 1)
        {
            int block_offset_2 = __popcll(binary_2 & (mask1-1));
            *(output_matrix_2 + block_offset_2 ) = output_fragment[3];
        }
}


// sddmm tcu tf32
__global__ void sddmm_forward_tf32_tcf_v2_kernel_tcu(
    const int* __restrict__ row_offsets,
    const int* __restrict__ t_block_offset,
    const int* __restrict__ t_tcl,
    const int* __restrict__ t_column,
    const int* t_window_row,
    const float* __restrict__ lhs_matrix,
    const float4* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix,
    int windows,
    int nOri,
    int mOri,
    int splitk,
    int warps)
{
    int m_index_vec = (blockIdx.z*splitk)+blockIdx.y;
    if(m_index_vec>=windows)
    return;

    int t_win_offset = __ldg(row_offsets + (m_index_vec));
    int tcu_blocks = __ldg(row_offsets + (m_index_vec) + 1) - t_win_offset; 
    if(tcu_blocks==0) return;
    int id=blockIdx.x*warps+(threadIdx.x/32);
    if(id>= tcu_blocks) return;

    //稀疏块
    int warp_id = threadIdx.x>>5;
    int warpin_id = threadIdx.x%32;
    float output_fragment[4] = {0,0,0,0};
    int col=__ldg(t_column + t_win_offset*16 + (id*16) + (warpin_id/2));
    // int col1=*(t_column + t_win_offset*16 + (id*16) + (warpin_id/4));
    // int col2=*(t_column + t_win_offset*16 + (id*16) + (warpin_id/4) +8);
    int cur_m_index_vec = __ldg(t_window_row + m_index_vec);
    int row=cur_m_index_vec*8 + (warpin_id/4);
    const float4 *r_matrix = rhs_matrix + col*nOri/4 + (warpin_id%2);
    const float *l_matrix = lhs_matrix + row*nOri;
    int steps = nOri/8;
    // int residue = nOri%8;
    //warps = 4
    __shared__ float4 block_a[32*4]; 
    // __shared__ float2 block_b[32];

    // float4 * block_a_cur = block_a + 32*warp_id;
    mmaComputeUtils_tf32_gen_v2 computer(
        l_matrix,
        r_matrix, 
        block_a + 32*warp_id,
        output_fragment, 
        threadIdx.x);

    if(steps>0)
    for(int i = 0; i < steps; i++){
        computer.TileMAC(nOri, col, mOri, row);
        // __syncthreads();
        __syncwarp();
    }
    // if(residue>0)
    //     computer.TileMACResidue(residue, nOri, col, col1, mOri, row);

    // if(row<mOri)
    // {
        //一个warp算8x16, 一个block-4个warp, 即512
        // __shared__ float sparse[512];
        // for(int i=0; i<4; i++){
        //     sparse[threadIdx.x*4 + i] = 0.0;
        // }
        // __syncthreads();
        // float *sparse_ = sparse + warp_id*128;
        // 获取列索引
        //搬运稀疏数据
        int value_offset = __ldg(t_block_offset + t_win_offset + id);
        int nnz_block = __ldg(t_block_offset + t_win_offset + id + 1) - value_offset;

        int flag0 = (warpin_id%4)*2*16 + (warpin_id/4);
        int flag1 = (warpin_id%4)*2*16 + (warpin_id/4) + 16;
        int flag2 = (warpin_id%4)*2*16 + (warpin_id/4) + 8;
        int flag3 = (warpin_id%4)*2*16 + (warpin_id/4) + 24;

        float * output_matrix_ = output_matrix + value_offset;
        for(int i=0; i<nnz_block; i++)
        {
           int pos = __ldg(t_tcl + value_offset + i);
           if(pos == flag0){
            *(output_matrix_ + i) = output_fragment[0];
           }else if(pos == flag1){
            *(output_matrix_ + i) = output_fragment[1];
           }else if(pos == flag2){
            *(output_matrix_ + i) = output_fragment[2];
           }else if(pos == flag3){
            *(output_matrix_ + i) = output_fragment[3];
           }
        }
}



//tcu cuda
float sddmm_forward_tf32_tcu_part_kernel_tcf(
    int * t_row_offset,
    int * t_blockNew_offset,
    int * t_tcl,
    int * t_column, 
    int* t_window_row,

    float * rhs_matrix,
    float * output_matrix,

    const int parts_t,
    const int maxPart,
    const int dimN,
    const int mOri,
    int epoches)
{
    int splitk = 0;
    //一个block中4个warp
    int warps = 4;
    if(parts_t<500000) splitk=8;
    else splitk=((parts_t/1250000)+1)*20;
    dim3 grid_dim(((maxPart/warps)+1), splitk ,(parts_t/splitk+1));
    dim3 block_dim(warps*32, 1, 1);

    for(int iter=0; iter<0; ++iter){
        sddmm_forward_tf32_tcf_v2_kernel_tcu<<<grid_dim, block_dim>>>(
            t_row_offset, 
            t_blockNew_offset,
            t_tcl,
            t_column, 
            t_window_row,
            rhs_matrix,  
            reinterpret_cast<float4 *>(rhs_matrix),  
            output_matrix,
            parts_t, dimN, mOri, splitk, warps);
    }
    //测试kernel
    float spmm_ms_avg = 0.0f;
    float spmm_ms = 0.0f;
    cudaEvent_t spmm_start;
    cudaEvent_t spmm_end;
    cudaEventCreate(&spmm_start);
    cudaEventCreate(&spmm_end); 
    cudaEventRecord(spmm_start);
    for(int iter=0; iter<epoches; ++iter){
        sddmm_forward_tf32_tcf_v2_kernel_tcu<<<grid_dim, block_dim>>>(
            t_row_offset, 
            t_blockNew_offset,
            t_tcl,
            t_column, 
            t_window_row,
            rhs_matrix,  
            reinterpret_cast<float4 *>(rhs_matrix),  
            output_matrix,
            parts_t, dimN, mOri, splitk, warps);
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


// sddmm tcu fp16
__global__ void sddmm_forward_fp16_binary_v2_kernel_tcu(
    const int* __restrict__ t_window_offset,
    const int* __restrict__ t_block_offset,
    const int* __restrict__ t_column,
    const long* __restrict__ t_binary,
    const int* t_window_row,
    const half* __restrict__ rhs_matrix,
    half* __restrict__ output_matrix,
    int windows,
    int nOri,
    int mOri,
    int splitk,
    int warps)
{
    int m_index_vec = (blockIdx.z*splitk)+blockIdx.y;
    if(m_index_vec>=windows)
    return;

    int t_win_offset = __ldg(t_window_offset + m_index_vec);
    int tcu_blocks = __ldg(t_window_offset + m_index_vec + 1) - t_win_offset; 
    if(tcu_blocks==0) return;
    int id=blockIdx.x*warps+(threadIdx.x/32);
    if(id>= tcu_blocks) return;
    int warp_id = threadIdx.x>>5;
    int warpin_id = threadIdx.x%32;
    at::Half output_fragment_[4] = {0,0,0,0};
    half * output_fragment = reinterpret_cast< half *>(output_fragment_);
    long col=*(t_column + t_win_offset*16 + (id*16) + (warpin_id/4));
    long col1=*(t_column + t_win_offset*16 + (id*16) + (warpin_id/4) +8);
    int cur_m_index_vec = __ldg(t_window_row + m_index_vec);
    int row=cur_m_index_vec*8 + (warpin_id/4);
    const half *r_matrix;
    if(col>=0) r_matrix = rhs_matrix + col*nOri;
    const half *l_matrix = rhs_matrix + row*nOri;
    int steps = nOri/16;
    // int residue = nOri%16;
    mmaComputeUtils_fp16_gen computer(
        reinterpret_cast<const half2 *>(l_matrix),
        reinterpret_cast<const half2 *>(r_matrix), 
        reinterpret_cast<uint32_t *>(output_fragment), 
        threadIdx.x);

    if(steps>0)
    for(int i = 0; i < steps; i++){
        computer.TileMAC(nOri, col, col1, mOri, row);
    }
    // if(residue>0)
    //     computer.TileMACResidue(residue, nOri, col, col1, mOri, row);


        int offset = ((warpin_id%4)*2*8) + (warpin_id/4);
        int offset1 = offset+8;
        long binary_1 = __ldg(t_binary + t_win_offset*2 + id*2);
        long binary_2 = __ldg(t_binary + t_win_offset*2 + id*2 + 1);
        long temp_0 = (binary_1 >> offset);
        long temp_1 = (binary_1 >> offset1);
        long temp_2 = (binary_2 >> offset);
        long temp_3 = (binary_2 >> offset1);
        long a= 1;
        long mask = (a << offset);
        long mask1 = (a << offset1);

        int fifthBit_0 = ((temp_0) & 1);
        int fifthBit_1 = ((temp_1) & 1);
        int fifthBit_2 = ((temp_2) & 1);
        int fifthBit_3 = ((temp_3) & 1);


        int value_offset = __ldg(t_block_offset + t_win_offset + id);
        half * output_matrix_1 = output_matrix + value_offset;
        half * output_matrix_2 = output_matrix_1 + __popcll(binary_1);
        if(fifthBit_0 == 1)
        {
            int block_offset_1 = __popcll(binary_1 & (mask-1));
            *(output_matrix_1 + block_offset_1) = output_fragment[0];
        }
        if(fifthBit_1 == 1)
        {
            int block_offset_1 = __popcll(binary_1 & (mask1-1));
            *(output_matrix_1 + block_offset_1 ) = output_fragment[1];
        }
        if(fifthBit_2 == 1)
        {
            int block_offset_2 = __popcll(binary_2 & (mask-1));
            *(output_matrix_2 + block_offset_2) = output_fragment[2];
        }
        if(fifthBit_3 == 1)
        {
            int block_offset_2 = __popcll(binary_2 & (mask1-1));
            *(output_matrix_2 + block_offset_2 ) = output_fragment[3];
        }
    // if(row<mOri)
    // {
    //     long binary_1 = __ldg(t_binary + t_win_offset*2 + id*2);
    //     long binary_2 = __ldg(t_binary + t_win_offset*2 + id*2 + 1);
    //     long temp_1 = (binary_1 >> (warpin_id*2));
    //     long temp_2 = (binary_2 >> (warpin_id*2));
    //     long a= 1;
    //     long mask = (a << (warpin_id*2));
    //     long mask1 = (a << ((warpin_id*2)+1));
    //     int fifthBit_1 = ((temp_1) & 1);
    //     int fifthBit_2 = ((temp_2) & 1);
    //     int block_offset_1 = -1;
    //     int block_offset_2 = -1;
    //     int value_offset = __ldg(t_block_offset + t_win_offset + id);
    //     half * output_matrix_1 = output_matrix + value_offset;
    //     half * output_matrix_2 = output_matrix_1 + __popcll(binary_1);
    //     if(fifthBit_1 == 1)
    //     {
    //         block_offset_1 = __popcll(binary_1 & (mask-1));
    //         *(output_matrix_1 + block_offset_1) = output_fragment[0];
    //     }
    //     if(fifthBit_2 == 1)
    //     {
    //         block_offset_2 = __popcll(binary_2 & (mask-1));
    //         *(output_matrix_2 + block_offset_2 ) = output_fragment[2];
    //     }

    //     if((row+1)<mOri)
    //     {
    //         fifthBit_1 = ((temp_1>>1) & 1);
    //         fifthBit_2 = ((temp_2>>1) & 1);
    //         if(fifthBit_1 == 1)
    //         {
    //             if(block_offset_1==-1)
    //             {
    //                 block_offset_1 = __popcll(binary_1 & (mask1-1));
    //                 *(output_matrix_1 + block_offset_1) = output_fragment[1];
    //             }
    //             else
    //             *(output_matrix_1 + block_offset_1 + 1) = output_fragment[1];
    //         }
    //         if(fifthBit_2 == 1)
    //         {
    //             if(block_offset_2==-1)
    //             {
    //                 block_offset_2 = __popcll(binary_2 & (mask1-1));
    //                 *(output_matrix_2 + block_offset_2) = output_fragment[3];
    //             }
    //             else *(output_matrix_2 + block_offset_2 + 1) = output_fragment[3];
    //         }
    //     }
    // }
    /*
    if(row<mOri)
    {
        long binary_1 = __ldg(t_binary + t_win_offset*2 + id*2 + warpin_id/16);
        long temp_1 = (binary_1 >> ((warpin_id%16)*4));
        long a= 1;
        long mask = (a << ((warpin_id%16)*4));
        int fifthBit_1 = ((temp_1) & 1);
        int block_offset_1 = -1;
        int value_offset = __ldg(t_block_offset + t_win_offset + id);
        float * output_matrix_1 = output_matrix + value_offset;
        if(warpin_id>15){
            long binary_2 = __ldg(t_binary + t_win_offset*2 + id*2);
            output_matrix_1 += __popcll(binary_2);
        }
        if(fifthBit_1 == 1)
        {
            block_offset_1 = __popcll(binary_1 & (mask-1));
            *(output_matrix_1 + block_offset_1) = output_fragment[0];
        }
        for(int q=1;q<4;q++){
            temp_1 = temp_1>>1;
            fifthBit_1 = (temp_1 & 1);
            mask = mask<<1;
            if(fifthBit_1 == 1)
            {
                if(block_offset_1==-1)
                {
                    block_offset_1 = __popcll(binary_1 & (mask-1));
                    *(output_matrix_1 + block_offset_1) = output_fragment[q];
                }
                else{
                *(output_matrix_1 + block_offset_1 + 1) = output_fragment[q];
                block_offset_1++;}
            }
        }
    }
    */
}



__global__ void sddmm_forward_fp16_binary_v2_kernel_tcu_cc_only(
    const int* __restrict__ t_window_offset,
    const int* __restrict__ t_block_offset,
    const int* __restrict__ t_column,
    const long* __restrict__ t_binary,
    const int* t_window_row,
    const half* __restrict__ rhs_matrix,
    half* __restrict__ output_matrix,
    int windows,
    int nOri,
    int mOri,
    int splitk,
    int warps)
{
    int m_index_vec = (blockIdx.z*splitk)+blockIdx.y;
    if(m_index_vec>=windows)
    return;

constexpr int WARPS_PER_BLOCK = 4;
    //判断执行tcu还是cuda
__shared__ half smem_A[WARPS_PER_BLOCK][32 * 4];
__shared__ half smem_B[WARPS_PER_BLOCK][32 * 8];
__shared__ half smem_ACC[WARPS_PER_BLOCK][32 * 4];

    int t_win_offset = __ldg(t_window_offset + m_index_vec);
    int tcu_blocks = __ldg(t_window_offset + m_index_vec + 1) - t_win_offset; 
    if(tcu_blocks==0) return;
    int id=blockIdx.x*warps+(threadIdx.x/32);
    if(id>= tcu_blocks) return;
    int warp_id = threadIdx.x>>5;
    int warpin_id = threadIdx.x%32;
    at::Half output_fragment_[4] = {0,0,0,0};
    half * output_fragment = reinterpret_cast< half *>(output_fragment_);
    long col=*(t_column + t_win_offset*16 + (id*16) + (warpin_id/4));
    long col1=*(t_column + t_win_offset*16 + (id*16) + (warpin_id/4) +8);
    int cur_m_index_vec = __ldg(t_window_row + m_index_vec);
    int row=cur_m_index_vec*8 + (warpin_id/4);
    const half *r_matrix;
    if(col>=0) r_matrix = rhs_matrix + col*nOri;
    const half *l_matrix = rhs_matrix + row*nOri;
    int steps = nOri/16;
    // int residue = nOri%16;
    mmaComputeUtils_fp16_gen_cc_only computer(
        reinterpret_cast<const half2 *>(l_matrix),
        reinterpret_cast<const half2 *>(r_matrix), 
        reinterpret_cast<uint32_t *>(output_fragment), 
        threadIdx.x);

    if(steps>0)
    for(int i = 0; i < steps; i++){
        computer.TileMAC(nOri, col, col1, mOri, row, 
            &smem_A[warp_id][0],
            &smem_B[warp_id][0],
            &smem_ACC[warp_id][0]);
    }
    // if(residue>0)
    //     computer.TileMACResidue(residue, nOri, col, col1, mOri, row);


        int offset = ((warpin_id%4)*2*8) + (warpin_id/4);
        int offset1 = offset+8;
        long binary_1 = __ldg(t_binary + t_win_offset*2 + id*2);
        long binary_2 = __ldg(t_binary + t_win_offset*2 + id*2 + 1);
        long temp_0 = (binary_1 >> offset);
        long temp_1 = (binary_1 >> offset1);
        long temp_2 = (binary_2 >> offset);
        long temp_3 = (binary_2 >> offset1);
        long a= 1;
        long mask = (a << offset);
        long mask1 = (a << offset1);

        int fifthBit_0 = ((temp_0) & 1);
        int fifthBit_1 = ((temp_1) & 1);
        int fifthBit_2 = ((temp_2) & 1);
        int fifthBit_3 = ((temp_3) & 1);


        int value_offset = __ldg(t_block_offset + t_win_offset + id);
        half * output_matrix_1 = output_matrix + value_offset;
        half * output_matrix_2 = output_matrix_1 + __popcll(binary_1);
        if(fifthBit_0 == 1)
        {
            int block_offset_1 = __popcll(binary_1 & (mask-1));
            *(output_matrix_1 + block_offset_1) = output_fragment[0];
        }
        if(fifthBit_1 == 1)
        {
            int block_offset_1 = __popcll(binary_1 & (mask1-1));
            *(output_matrix_1 + block_offset_1 ) = output_fragment[1];
        }
        if(fifthBit_2 == 1)
        {
            int block_offset_2 = __popcll(binary_2 & (mask-1));
            *(output_matrix_2 + block_offset_2) = output_fragment[2];
        }
        if(fifthBit_3 == 1)
        {
            int block_offset_2 = __popcll(binary_2 & (mask1-1));
            *(output_matrix_2 + block_offset_2 ) = output_fragment[3];
        }
}



//tcu cuda
float sddmm_forward_fp16_tcu_part_kernel(
    int * t_row_offset,
    int * t_blockNew_offset,
    int * t_column, 
    int* t_window_row,
    long * d_t_binary,

    half * rhs_matrix,
    half * output_matrix,

    const int parts_t,
    const int maxPart,
    const int dimN,
    const int mOri,
    int epoches)
{
    int splitk = 0;
    //一个block中4个warp
    int warps = 4;
    if(parts_t<500000) splitk=8;
    else splitk=((parts_t/1250000)+1)*20;
    dim3 grid_dim(((maxPart/warps)+1), splitk ,(parts_t/splitk+1));
    dim3 block_dim(warps*32, 1, 1);

    for(int iter=0; iter<0; ++iter){
    sddmm_forward_fp16_binary_v2_kernel_tcu<<<grid_dim, block_dim>>>(
            t_row_offset, 
            t_blockNew_offset,
            t_column, 
            d_t_binary, 
            t_window_row,
            rhs_matrix,  
            output_matrix,
            parts_t, dimN, mOri, splitk, warps);
    }
    //测试kernel
    float spmm_ms_avg = 0.0f;
    float spmm_ms = 0.0f;
    cudaEvent_t spmm_start;
    cudaEvent_t spmm_end;
    cudaEventCreate(&spmm_start);
    cudaEventCreate(&spmm_end); 
    cudaEventRecord(spmm_start);
    for(int iter=0; iter<epoches; ++iter){
        sddmm_forward_fp16_binary_v2_kernel_tcu<<<grid_dim, block_dim>>>(
            t_row_offset, 
            t_blockNew_offset,
            t_column, 
            d_t_binary, 
            t_window_row,
            rhs_matrix,  
            output_matrix,
            parts_t, dimN, mOri, splitk, warps);
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
// sddmm tcu tf32 csr
__global__ void sddmm_forward_tf32_csr_v2_kernel_tcu(
    const int* __restrict__ t_window_offset,
    const int* __restrict__ t_block_offset,
    const int* __restrict__ t_column,
    const int* __restrict__ t_row,
    const int* t_window_row,
    const float* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix,
    int windows,
    int nOri,
    int mOri,
    int splitk,
    int warps)
{
    int m_index_vec = (blockIdx.z*splitk)+blockIdx.y;
    if(m_index_vec>=windows)
    return;

    int t_win_offset = __ldg(t_window_offset + m_index_vec);
    int tcu_blocks = __ldg(t_window_offset + m_index_vec + 1) - t_win_offset; 
    if(tcu_blocks==0) return;
    int id=blockIdx.x*warps+(threadIdx.x/32);
    if(id>= tcu_blocks) return;
    int warp_id = threadIdx.x>>5;
    int warpin_id = threadIdx.x%32;
    float output_fragment[4] = {0,0,0,0};
    long col=*(t_column + t_win_offset*16 + (id*16) + (warpin_id/4));
    long col1=*(t_column + t_win_offset*16 + (id*16) + (warpin_id/4) +8);
    int cur_m_index_vec = __ldg(t_window_row + m_index_vec);
    long row=cur_m_index_vec*8 + (warpin_id/4);
    const float *r_matrix;
    if(col>=0) r_matrix = rhs_matrix + col*nOri;
    const float *l_matrix = rhs_matrix + row*nOri;
    int steps = nOri/8;
    int residue = nOri%8;
    mmaComputeUtils_tf32_gen computer(
        l_matrix,
        r_matrix, 
        output_fragment, 
        threadIdx.x);

    if(steps>0)
    for(int i = 0; i < steps; i++){
        computer.TileMAC(nOri, col, col1, mOri, row);
    }
    if(residue>0)
        computer.TileMACResidue(residue, nOri, col, col1, mOri, row);

    //output
    __shared__ float sparse[128*4];
    float * sparse_ = sparse + warp_id*128;
    //把结果转置写入sparse
    float* sparse_1 = sparse_ + ((warpin_id%4)*2*16) + (warpin_id/4);
    *(sparse_1) = output_fragment[0];
    *(sparse_1+8) = output_fragment[2];
    *(sparse_1+16) = output_fragment[1];
    *(sparse_1+24) = output_fragment[3];
    __syncwarp();
    // if(m_index_vec==0 && threadIdx.x==0)
    // {
    //     for(int q=0;q<128;q++)
    //     printf("%f ", sparse_[q]);
    //     printf("\n");
    // }
    // for(int q=0;q<4;q++)
    // sparse_[warpin_id*4+q] = -1;
    //__syncwarp();
    int value_offset = __ldg(t_block_offset + t_win_offset + id);
    int nnz_block = __ldg(t_block_offset + t_win_offset + id + 1) - value_offset;
    for(int q=0;q<4;q++)
    {
        if((warpin_id*4 + q) < nnz_block){
            int temp = __ldg(t_row + value_offset + warpin_id*4 + q);
            *(output_matrix + value_offset + warpin_id*4 + q) = *(sparse_ + temp);
        }
    }
    // __syncwarp();
    // if(m_index_vec==0 && threadIdx.x==0)
    // {
    //     for(int q=0;q<128;q++)
    //     printf("%d ", sparse_[q]);
    //     printf("\n");
    // }
    // if(row<mOri)
    // {
    //     float* sparse_1 = sparse_ + ((warpin_id%4)*2*16) + (warpin_id/4);
    //     if( *(sparse_1) > 0){
    //         *(output_matrix + value_offset + *(sparse_1)) = output_fragment[0];
    //     }
    //     if( *(sparse_1+8) > 0){
    //         *(output_matrix + value_offset + *(sparse_1+8)) = output_fragment[2];
    //     }

    //     if((row+1)<mOri)
    //     {
    //         if( *(sparse_1+16) > 0){
    //             *(output_matrix + value_offset + *(sparse_1+16)) = output_fragment[1];
    //         }
    //         if( *(sparse_1+24) > 0){
    //             *(output_matrix + value_offset + *(sparse_1+24)) = output_fragment[3];
    //         }
    //     }
    // }
}

//tcu cuda
float sddmm_forward_tf32_tcu_part_kernel_csr(
    int * t_row_offset,
    int * t_blockNew_offset,
    int * t_column, 
    int* t_window_row,
    int * d_t_row,

    float * rhs_matrix,
    float * output_matrix,

    const int parts_t,
    const int maxPart,
    const int dimN,
    const int mOri,
    int epoches)
{
    int splitk = 0;
    //一个block中4个warp
    int warps = 4;
    if(parts_t<500000) splitk=8;
    else splitk=((parts_t/1250000)+1)*20;
    dim3 grid_dim(((maxPart/warps)+1), splitk ,(parts_t/splitk+1));
    dim3 block_dim(warps*32, 1, 1);

    for(int iter=0; iter<0; ++iter){
    sddmm_forward_tf32_csr_v2_kernel_tcu<<<grid_dim, block_dim>>>(
            t_row_offset, 
            t_blockNew_offset,
            t_column, 
            d_t_row, 
            t_window_row,
            rhs_matrix,  
            output_matrix,
            parts_t, dimN, mOri, splitk, warps);
    }
    //测试kernel
    float spmm_ms_avg = 0.0f;
    float spmm_ms = 0.0f;
    cudaEvent_t spmm_start;
    cudaEvent_t spmm_end;
    cudaEventCreate(&spmm_start);
    cudaEventCreate(&spmm_end); 
    cudaEventRecord(spmm_start);
    for(int iter=0; iter<epoches; ++iter){
        sddmm_forward_tf32_csr_v2_kernel_tcu<<<grid_dim, block_dim>>>(
            t_row_offset, 
            t_blockNew_offset,
            t_column, 
            d_t_row, 
            t_window_row,
            rhs_matrix,  
            output_matrix,
            parts_t, dimN, mOri, splitk, warps);
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


// sddmm tcu tf32 csr
__global__ void sddmm_forward_fp16_csr_v2_kernel_tcu(
    const int* __restrict__ t_window_offset,
    const int* __restrict__ t_block_offset,
    const int* __restrict__ t_column,
    const int* __restrict__ t_row,
    const int* t_window_row,
    const half* __restrict__ rhs_matrix,
    half* __restrict__ output_matrix,
    int windows,
    int nOri,
    int mOri,
    int splitk,
    int warps)
{
    int m_index_vec = (blockIdx.z*splitk)+blockIdx.y;
    if(m_index_vec>=windows)
    return;

    int t_win_offset = __ldg(t_window_offset + m_index_vec);
    int tcu_blocks = __ldg(t_window_offset + m_index_vec + 1) - t_win_offset; 
    if(tcu_blocks==0) return;
    int id=blockIdx.x*warps+(threadIdx.x/32);
    if(id>= tcu_blocks) return;
    int warp_id = threadIdx.x>>5;
    int warpin_id = threadIdx.x%32;
    at::Half output_fragment_[4] = {0,0,0,0};
    half * output_fragment = reinterpret_cast< half *>(output_fragment_);
    long col=*(t_column + t_win_offset*16 + (id*16) + (warpin_id/4));
    long col1=*(t_column + t_win_offset*16 + (id*16) + (warpin_id/4) +8);
    int cur_m_index_vec = __ldg(t_window_row + m_index_vec);
    long row=cur_m_index_vec*8 + (warpin_id/4);
    const half *r_matrix;
    if(col>=0) r_matrix = rhs_matrix + col*nOri;
    const half *l_matrix = rhs_matrix + row*nOri;
    int steps = nOri/16;
    // int residue = nOri%16;
    mmaComputeUtils_fp16_gen computer(
        reinterpret_cast<const half2 *>(l_matrix),
        reinterpret_cast<const half2 *>(r_matrix), 
        reinterpret_cast<uint32_t *>(output_fragment), 
        threadIdx.x);

    if(steps>0)
    for(int i = 0; i < steps; i++){
        computer.TileMAC(nOri, col, col1, mOri, row);
    }
    // if(residue>0)
    //     computer.TileMACResidue(residue, nOri, col, col1, mOri, row);

    //output
    __shared__ half sparse[128*4];
    half * sparse_ = sparse + warp_id*128;
    //把结果转置写入sparse
    half* sparse_1 = sparse_ + ((warpin_id%4)*2*16) + (warpin_id/4);
    *(sparse_1) = output_fragment[0];
    *(sparse_1+8) = output_fragment[2];
    *(sparse_1+16) = output_fragment[1];
    *(sparse_1+24) = output_fragment[3];
    __syncwarp();
    // if(m_index_vec==0 && threadIdx.x==0)
    // {
    //     for(int q=0;q<128;q++)
    //     printf("%f ", sparse_[q]);
    //     printf("\n");
    // }
    // for(int q=0;q<4;q++)
    // sparse_[warpin_id*4+q] = -1;
    //__syncwarp();
    int value_offset = __ldg(t_block_offset + t_win_offset + id);
    int nnz_block = __ldg(t_block_offset + t_win_offset + id + 1) - value_offset;
    for(int q=0;q<4;q++)
    {
        if((warpin_id*4 + q) < nnz_block){
            int temp = __ldg(t_row + value_offset + warpin_id*4 + q);
            *(output_matrix + value_offset + warpin_id*4 + q) = *(sparse_ + temp);
        }
    }
    // __syncwarp();
    // if(m_index_vec==0 && threadIdx.x==0)
    // {
    //     for(int q=0;q<128;q++)
    //     printf("%d ", sparse_[q]);
    //     printf("\n");
    // }
    // if(row<mOri)
    // {
    //     float* sparse_1 = sparse_ + ((warpin_id%4)*2*16) + (warpin_id/4);
    //     if( *(sparse_1) > 0){
    //         *(output_matrix + value_offset + *(sparse_1)) = output_fragment[0];
    //     }
    //     if( *(sparse_1+8) > 0){
    //         *(output_matrix + value_offset + *(sparse_1+8)) = output_fragment[2];
    //     }

    //     if((row+1)<mOri)
    //     {
    //         if( *(sparse_1+16) > 0){
    //             *(output_matrix + value_offset + *(sparse_1+16)) = output_fragment[1];
    //         }
    //         if( *(sparse_1+24) > 0){
    //             *(output_matrix + value_offset + *(sparse_1+24)) = output_fragment[3];
    //         }
    //     }
    // }
}

//tcu cuda
float sddmm_forward_fp16_tcu_part_kernel_csr(
    int * t_row_offset,
    int * t_blockNew_offset,
    int * t_column, 
    int* t_window_row,
    int * d_t_row,

    half * rhs_matrix,
    half * output_matrix,

    const int parts_t,
    const int maxPart,
    const int dimN,
    const int mOri,
    int epoches)
{
    int splitk = 0;
    //一个block中4个warp
    int warps = 4;
    if(parts_t<500000) splitk=8;
    else splitk=((parts_t/1250000)+1)*20;
    dim3 grid_dim(((maxPart/warps)+1), splitk ,(parts_t/splitk+1));
    dim3 block_dim(warps*32, 1, 1);

    for(int iter=0; iter<10; ++iter){
    sddmm_forward_fp16_csr_v2_kernel_tcu<<<grid_dim, block_dim>>>(
            t_row_offset, 
            t_blockNew_offset,
            t_column, 
            d_t_row, 
            t_window_row,
            rhs_matrix,  
            output_matrix,
            parts_t, dimN, mOri, splitk, warps);
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
        sddmm_forward_fp16_csr_v2_kernel_tcu<<<grid_dim, block_dim>>>(
            t_row_offset, 
            t_blockNew_offset,
            t_column, 
            d_t_row, 
            t_window_row,
            rhs_matrix,  
            output_matrix,
            parts_t, dimN, mOri, splitk, warps);
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

//tcu+cuda
//tcu cuda
float spmm_forward_tf32_tcu_cuda_kernel(
    int * t_row_offset,
    int * t_blockNew_offset,
    int * t_column, 
    int* t_window_row,
    long * d_t_binary,

    int * c_row_offset,
    int * c_row,
    int * c_column, 

    float * rhs_matrix,
    float * output_matrix_t,
    float * output_matrix_c,

    const int parts_t,
    const int maxPart,
    const int nOri,
    const int mOri,
    int epoches,
    int parts)
{
    //tcu
    int splitk_t = 0;
    //一个block中4个warp
    int warps = 4;
    if(parts_t<500000) splitk_t=8;
    else splitk_t=((parts_t/1250000)+1)*20;
    dim3 grid_dim_t(((maxPart/warps)+1), splitk_t ,(parts_t/splitk_t+1));
    dim3 block_dim_t(warps*32, 1, 1);

    //cuda
    int splitk_c = 0;
    if(parts<500000) splitk_c=8;
    else splitk_c=((parts/1250000)+1)*20;
    dim3 grid_dim_c(1, splitk_c ,((parts/splitk_c)+1));
    dim3 block_dim_c(32, 1, 1);
    int sharedmemory = 32*32*sizeof(float)+ 32*sizeof(int);
    cudaStream_t stream1,stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    for(int iter=0; iter<10; ++iter){
    sddmm_forward_tf32_binary_v2_kernel_tcu<<<grid_dim_t, block_dim_t, 0, stream1>>>(
            t_row_offset, 
            t_blockNew_offset,
            t_column, 
            d_t_binary, 
            t_window_row,
            rhs_matrix,  
            reinterpret_cast<float4 *>(rhs_matrix),  
            output_matrix_t,
            parts_t, nOri, mOri, splitk_t, warps);

    sddmm_forward_tf32_csr_v2_kernel_cuda<<<grid_dim_c, block_dim_c, sharedmemory, stream2>>>(
        c_row_offset, 
        c_row,
        c_column,
        rhs_matrix,  
        output_matrix_c,
        parts, nOri, mOri, splitk_c);
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
        sddmm_forward_tf32_binary_v2_kernel_tcu<<<grid_dim_t, block_dim_t, 0, stream1>>>(
                t_row_offset, 
                t_blockNew_offset,
                t_column, 
                d_t_binary, 
                t_window_row,
            rhs_matrix,  
            reinterpret_cast<float4 *>(rhs_matrix),  
                output_matrix_t,
                parts_t, nOri, mOri, splitk_t, warps);

        sddmm_forward_tf32_csr_v2_kernel_cuda<<<grid_dim_c, block_dim_c, sharedmemory, stream2>>>(
            c_row_offset, 
            c_row,
            c_column,
            rhs_matrix,  
            output_matrix_c,
            parts, nOri, mOri, splitk_c);
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

float spmm_forward_tf32_tcu_cuda_kernel_32(
    int * t_row_offset,
    int * t_blockNew_offset,
    int * t_column, 
    int* t_window_row,
    long * d_t_binary,

    int * c_row_offset,
    int * c_row,
    int * c_column, 

    float * rhs_matrix,
    float * output_matrix_t,
    float * output_matrix_c,

    const int parts_t,
    const int maxPart,
    const int nOri,
    const int mOri,
    int epoches,
    int parts)
{
    //tcu
    int splitk_t = 0;
    //一个block中4个warp
    int warps = 4;
    if(parts_t<500000) splitk_t=8;
    else splitk_t=((parts_t/1250000)+1)*20;
    dim3 grid_dim_t(((maxPart/warps)+1), splitk_t ,(parts_t/splitk_t+1));
    dim3 block_dim_t(warps*32, 1, 1);

    //cuda
    int splitk_c = 0;
    if(parts<500000) splitk_c=8;
    else splitk_c=((parts/1250000)+1)*20;
    dim3 grid_dim_c(1, splitk_c ,((parts/splitk_c)+1));
    dim3 block_dim_c(32, 1, 1);
    int sharedmemory = 32*32*sizeof(float)+ 32*sizeof(int);
    cudaStream_t stream1,stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    for(int iter=0; iter<10; ++iter){
    sddmm_forward_tf32_binary_v2_kernel_tcu<<<grid_dim_t, block_dim_t, 0, stream1>>>(
            t_row_offset, 
            t_blockNew_offset,
            t_column, 
            d_t_binary, 
            t_window_row,
            rhs_matrix,  
            reinterpret_cast<float4 *>(rhs_matrix),  
            output_matrix_t,
            parts_t, nOri, mOri, splitk_t, warps);

    sddmm_forward_tf32_csr_v2_kernel_cuda_32<<<grid_dim_c, block_dim_c, sharedmemory, stream2>>>(
        c_row_offset, 
        c_row,
        c_column,
        reinterpret_cast<float4 *>(rhs_matrix), 
        reinterpret_cast<float4 *>(rhs_matrix),
        output_matrix_c,
        parts, nOri, mOri, splitk_c);
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
        sddmm_forward_tf32_binary_v2_kernel_tcu<<<grid_dim_t, block_dim_t, 0, stream1>>>(
                t_row_offset, 
                t_blockNew_offset,
                t_column, 
                d_t_binary, 
                t_window_row,
            rhs_matrix,  
            reinterpret_cast<float4 *>(rhs_matrix),  
                output_matrix_t,
                parts_t, nOri, mOri, splitk_t, warps);

        sddmm_forward_tf32_csr_v2_kernel_cuda_32<<<grid_dim_c, block_dim_c, sharedmemory, stream2>>>(
            c_row_offset, 
            c_row,
            c_column,
            reinterpret_cast<float4 *>(rhs_matrix), 
            reinterpret_cast<float4 *>(rhs_matrix),
            output_matrix_c,
            parts, nOri, mOri, splitk_c);
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

float spmm_forward_tf32_tcu_cuda_kernel_128(
    int * t_row_offset,
    int * t_blockNew_offset,
    int * t_column, 
    int* t_window_row,
    long * d_t_binary,

    int * c_row_offset,
    int * c_row,
    int * c_column, 

    float * rhs_matrix,
    float * output_matrix_t,
    float * output_matrix_c,

    const int parts_t,
    const int maxPart,
    const int nOri,
    const int mOri,
    int epoches,
    int parts)
{
    //tcu
    int splitk_t = 0;
    //一个block中4个warp
    int warps = 4;
    if(parts_t<500000) splitk_t=8;
    else splitk_t=((parts_t/1250000)+1)*20;
    dim3 grid_dim_t(((maxPart/warps)+1), splitk_t ,(parts_t/splitk_t+1));
    dim3 block_dim_t(warps*32, 1, 1);

    //cuda
    int splitk_c = 0;
    if(parts<500000) splitk_c=8;
    else splitk_c=((parts/1250000)+1)*20;
    dim3 grid_dim_c(1, splitk_c ,((parts/splitk_c)+1));
    dim3 block_dim_c(32, 1, 1);
    int sharedmemory = 32*32*sizeof(float)+ 32*sizeof(int);
    cudaStream_t stream1,stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    for(int iter=0; iter<10; ++iter){
    sddmm_forward_tf32_binary_v2_kernel_tcu<<<grid_dim_t, block_dim_t, 0, stream1>>>(
            t_row_offset, 
            t_blockNew_offset,
            t_column, 
            d_t_binary, 
            t_window_row,
            rhs_matrix,  
            reinterpret_cast<float4 *>(rhs_matrix),  
            output_matrix_t,
            parts_t, nOri, mOri, splitk_t, warps);

    sddmm_forward_tf32_csr_v2_kernel_cuda_128<<<grid_dim_c, block_dim_c, sharedmemory, stream2>>>(
        c_row_offset, 
        c_row,
        c_column,
        rhs_matrix,  
        output_matrix_c,
        parts, nOri, mOri, splitk_c);
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
        sddmm_forward_tf32_binary_v2_kernel_tcu<<<grid_dim_t, block_dim_t, 0, stream1>>>(
                t_row_offset, 
                t_blockNew_offset,
                t_column, 
                d_t_binary, 
                t_window_row,
             rhs_matrix,  
            reinterpret_cast<float4 *>(rhs_matrix),  
                output_matrix_t,
                parts_t, nOri, mOri, splitk_t, warps);

        sddmm_forward_tf32_csr_v2_kernel_cuda_128<<<grid_dim_c, block_dim_c, sharedmemory, stream2>>>(
            c_row_offset, 
            c_row,
            c_column,
            rhs_matrix,  
            output_matrix_c,
            parts, nOri, mOri, splitk_c);
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


float spmm_forward_tf32_tcu_cuda_kernel_navie(
    int * t_row_offset,
    int * t_blockNew_offset,
    int * t_column, 
    int* t_window_row,
    long * d_t_binary,

    int * c_row_offset,
    int * c_row,
    int * c_column, 

    float * rhs_matrix,
    float * output_matrix_t,
    float * output_matrix_c,

    const int parts_t,
    const int maxPart,
    const int nOri,
    const int mOri,
    int epoches,
    int parts)
{
    //tcu
    int splitk_t = 0;
    //一个block中4个warp
    int warps = 4;
    if(parts_t<500000) splitk_t=8;
    else splitk_t=((parts_t/1250000)+1)*20;
    dim3 grid_dim_t(((maxPart/warps)+1), splitk_t ,(parts_t/splitk_t+1));
    dim3 block_dim_t(warps*32, 1, 1);

    //cuda
    int splitk_c = 0;
    if(parts<500000) splitk_c=8;
    else splitk_c=((parts/1250000)+1)*20;
    dim3 grid_dim_c(1, splitk_c ,((parts/splitk_c)+1));
    dim3 block_dim_c(32, 1, 1);
    int sharedmemory = 32*32*sizeof(float)+ 32*sizeof(int);
    cudaStream_t stream1,stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    for(int iter=0; iter<10; ++iter){
    sddmm_forward_tf32_binary_v2_kernel_tcu<<<grid_dim_t, block_dim_t, 0, stream1>>>(
            t_row_offset, 
            t_blockNew_offset,
            t_column, 
            d_t_binary, 
            t_window_row,
            rhs_matrix,  
            reinterpret_cast<float4 *>(rhs_matrix),  
            output_matrix_t,
            parts_t, nOri, mOri, splitk_t, warps);

    sddmm_forward_tf32_csr_v2_kernel_cuda_v2<<<grid_dim_c, block_dim_c, 0 , stream2>>>(
        c_row_offset, 
        c_row,
        c_column,
        rhs_matrix,  
        output_matrix_c,
        parts, nOri, mOri, splitk_c);
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
        sddmm_forward_tf32_binary_v2_kernel_tcu<<<grid_dim_t, block_dim_t, 0, stream1>>>(
                t_row_offset, 
                t_blockNew_offset,
                t_column, 
                d_t_binary, 
                t_window_row,
            rhs_matrix,  
            reinterpret_cast<float4 *>(rhs_matrix),  
                output_matrix_t,
                parts_t, nOri, mOri, splitk_t, warps);

        sddmm_forward_tf32_csr_v2_kernel_cuda_v2<<<grid_dim_c, block_dim_c, 0, stream2>>>(
            c_row_offset, 
            c_row,
            c_column,
            rhs_matrix,  
            output_matrix_c,
            parts, nOri, mOri, splitk_c);
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

float spmm_forward_fp16_tcu_cuda_kernel(
    int * t_row_offset,
    int * t_blockNew_offset,
    int * t_column, 
    int* t_window_row,
    long * d_t_binary,

    int * c_row_offset,
    int * c_row,
    int * c_column, 

    half * rhs_matrix,
    half * output_matrix_t,
    half * output_matrix_c,

    const int parts_t,
    const int maxPart,
    const int nOri,
    const int mOri,
    int epoches,
    int parts)
{
    //tcu
    int splitk_t = 0;
    //一个block中4个warp
    int warps = 4;
    if(parts_t<500000) splitk_t=8;
    else splitk_t=((parts_t/1250000)+1)*20;
    dim3 grid_dim_t(((maxPart/warps)+1), splitk_t ,(parts_t/splitk_t+1));
    dim3 block_dim_t(warps*32, 1, 1);

    //cuda
    int splitk_c = 0;
    if(parts<500000) splitk_c=8;
    else splitk_c=((parts/1250000)+1)*20;
    dim3 grid_dim_c(1, splitk_c ,((parts/splitk_c)+1));
    dim3 block_dim_c(32, 1, 1);
    int sharedmemory = 32*32*sizeof(float)+ 32*sizeof(int);
    cudaStream_t stream1,stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    for(int iter=0; iter<10; ++iter){
    sddmm_forward_fp16_binary_v2_kernel_tcu<<<grid_dim_t, block_dim_t, 0, stream1>>>(
            t_row_offset, 
            t_blockNew_offset,
            t_column, 
            d_t_binary, 
            t_window_row,
            rhs_matrix,  
            output_matrix_t,
            parts_t, nOri, mOri, splitk_t, warps);

    sddmm_forward_fp16_csr_v2_kernel_cuda<<<grid_dim_c, block_dim_c, sharedmemory, stream2>>>(
        c_row_offset, 
        c_row,
        c_column,
        reinterpret_cast<double *>(rhs_matrix), 
        reinterpret_cast<double *>(rhs_matrix),
        output_matrix_c,
        parts, nOri, mOri, splitk_c);
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
        sddmm_forward_fp16_binary_v2_kernel_tcu<<<grid_dim_t, block_dim_t, 0, stream1>>>(
                t_row_offset, 
                t_blockNew_offset,
                t_column, 
                d_t_binary, 
                t_window_row,
                rhs_matrix,  
                output_matrix_t,
                parts_t, nOri, mOri, splitk_t, warps);

        sddmm_forward_fp16_csr_v2_kernel_cuda<<<grid_dim_c, block_dim_c, sharedmemory, stream2>>>(
            c_row_offset, 
            c_row,
            c_column,
            reinterpret_cast<double *>(rhs_matrix), 
            reinterpret_cast<double *>(rhs_matrix), 
            output_matrix_c,
            parts, nOri, mOri, splitk_c);
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

float spmm_forward_fp16_tcu_cuda_kernel_cc_only(
    int * t_row_offset,
    int * t_blockNew_offset,
    int * t_column, 
    int* t_window_row,
    long * d_t_binary,

    int * c_row_offset,
    int * c_row,
    int * c_column, 

    half * rhs_matrix,
    half * output_matrix_t,
    half * output_matrix_c,

    const int parts_t,
    const int maxPart,
    const int nOri,
    const int mOri,
    int epoches,
    int parts)
{
    //tcu
    int splitk_t = 0;
    //一个block中4个warp
    int warps = 4;
    if(parts_t<500000) splitk_t=8;
    else splitk_t=((parts_t/1250000)+1)*20;
    dim3 grid_dim_t(((maxPart/warps)+1), splitk_t ,(parts_t/splitk_t+1));
    dim3 block_dim_t(warps*32, 1, 1);

    //cuda
    int splitk_c = 0;
    if(parts<500000) splitk_c=8;
    else splitk_c=((parts/1250000)+1)*20;
    dim3 grid_dim_c(1, splitk_c ,((parts/splitk_c)+1));
    dim3 block_dim_c(32, 1, 1);
    int sharedmemory = 32*32*sizeof(float)+ 32*sizeof(int);
    cudaStream_t stream1,stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    for(int iter=0; iter<10; ++iter){
    sddmm_forward_fp16_binary_v2_kernel_tcu_cc_only<<<grid_dim_t, block_dim_t, 0, stream1>>>(
            t_row_offset, 
            t_blockNew_offset,
            t_column, 
            d_t_binary, 
            t_window_row,
            rhs_matrix,  
            output_matrix_t,
            parts_t, nOri, mOri, splitk_t, warps);

    sddmm_forward_fp16_csr_v2_kernel_cuda<<<grid_dim_c, block_dim_c, sharedmemory, stream2>>>(
        c_row_offset, 
        c_row,
        c_column,
        reinterpret_cast<double *>(rhs_matrix), 
        reinterpret_cast<double *>(rhs_matrix),
        output_matrix_c,
        parts, nOri, mOri, splitk_c);
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
        sddmm_forward_fp16_binary_v2_kernel_tcu_cc_only<<<grid_dim_t, block_dim_t, 0, stream1>>>(
                t_row_offset, 
                t_blockNew_offset,
                t_column, 
                d_t_binary, 
                t_window_row,
                rhs_matrix,  
                output_matrix_t,
                parts_t, nOri, mOri, splitk_t, warps);

        sddmm_forward_fp16_csr_v2_kernel_cuda<<<grid_dim_c, block_dim_c, sharedmemory, stream2>>>(
            c_row_offset, 
            c_row,
            c_column,
            reinterpret_cast<double *>(rhs_matrix), 
            reinterpret_cast<double *>(rhs_matrix), 
            output_matrix_c,
            parts, nOri, mOri, splitk_c);
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

float spmm_forward_fp16_tcu_cuda_kernel_navie(
    int * t_row_offset,
    int * t_blockNew_offset,
    int * t_column, 
    int* t_window_row,
    long * d_t_binary,

    int * c_row_offset,
    int * c_row,
    int * c_column, 

    half * rhs_matrix,
    half * output_matrix_t,
    half * output_matrix_c,

    const int parts_t,
    const int maxPart,
    const int nOri,
    const int mOri,
    int epoches,
    int parts)
{
    //tcu
    int splitk_t = 0;
    //一个block中4个warp
    int warps = 4;
    if(parts_t<500000) splitk_t=8;
    else splitk_t=((parts_t/1250000)+1)*20;
    dim3 grid_dim_t(((maxPart/warps)+1), splitk_t ,(parts_t/splitk_t+1));
    dim3 block_dim_t(warps*32, 1, 1);

    //cuda
    int splitk_c = 0;
    if(parts<500000) splitk_c=8;
    else splitk_c=((parts/1250000)+1)*20;
    dim3 grid_dim_c(1, splitk_c ,((parts/splitk_c)+1));
    dim3 block_dim_c(32, 1, 1);
    int sharedmemory = 32*32*sizeof(float)+ 32*sizeof(int);
    cudaStream_t stream1,stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    for(int iter=0; iter<10; ++iter){
    sddmm_forward_fp16_binary_v2_kernel_tcu<<<grid_dim_t, block_dim_t, 0, stream1>>>(
            t_row_offset, 
            t_blockNew_offset,
            t_column, 
            d_t_binary, 
            t_window_row,
            rhs_matrix,  
            output_matrix_t,
            parts_t, nOri, mOri, splitk_t, warps);

    sddmm_forward_fp16_csr_v2_kernel_cuda_v2<<<grid_dim_c, block_dim_c, 0 , stream2>>>(
        c_row_offset, 
        c_row,
        c_column,
        rhs_matrix,  
        output_matrix_c,
        parts, nOri, mOri, splitk_c);
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
        sddmm_forward_fp16_binary_v2_kernel_tcu<<<grid_dim_t, block_dim_t, 0, stream1>>>(
                t_row_offset, 
                t_blockNew_offset,
                t_column, 
                d_t_binary, 
                t_window_row,
                rhs_matrix,  
                output_matrix_t,
                parts_t, nOri, mOri, splitk_t, warps);

        sddmm_forward_fp16_csr_v2_kernel_cuda_v2<<<grid_dim_c, block_dim_c, 0, stream2>>>(
            c_row_offset, 
            c_row,
            c_column,
            rhs_matrix,  
            output_matrix_c,
            parts, nOri, mOri, splitk_c);
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