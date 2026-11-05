#include <iostream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <ctime>
#include <cuda_fp16.h>
#include <cuda.h>
#include <mma.h>
#include <cuda_runtime.h>
// #include "cutlass/uint128.h"

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
// #include "./utils.h"
#include "config.h"
using namespace nvcuda;
// using uint128_t = cutlass::uint128_t;

__global__ void warmup() {}


__global__ void sparse_mma_kernel_base(
  int M, int N, int K,
  const half *__restrict__ opd_A, 
  const uint32_t *__restrict__ GPU_metadata,
  const int *__restrict__ tcblocks_offset,
  const int *__restrict__ old_col_all,
  const half *__restrict__ input_B, 
  float *output_C)
{
  // thread block index : row window index
  // each thread block handel each row window
  // one dimension thread grid
  const unsigned bid = blockIdx.x;
  unsigned block_start = tcblocks_offset[bid];
  unsigned block_end = tcblocks_offset[bid + 1];
  if(block_start==block_end) return;

  // two dimension thread block
  const unsigned wid = threadIdx.y;
  const unsigned laneid = threadIdx.x;
  const unsigned wid_tile_N = wid * tile_N;

  const unsigned groupID = laneid >> 2;
  const unsigned threadID_in_group = laneid % 4;

  uint32_t RA[2];
  uint32_t RB[2];
  float RC[4] = {0.0f};
  uint32_t RM;
  
  // A index
  unsigned sparse_A_idx_0 = groupID * tile_K_sp + threadID_in_group * 2;
  unsigned sparse_A_idx_1 = (groupID + 8) * tile_K_sp + threadID_in_group * 2;

  // B index
  unsigned row_b0=(threadID_in_group * 2) + (0 & 0x1);
  unsigned row_b1=(threadID_in_group * 2) + (1 & 0x1);
  unsigned row_b2=(threadID_in_group * 2) + (2 & 0x1) + 8 ;
  unsigned row_b3=(threadID_in_group * 2) + (3 & 0x1) + 8 ;

  // C index
  unsigned col_c0=(threadID_in_group * 2) + (0 & 0x1);
  unsigned col_c1=(threadID_in_group * 2) + (1 & 0x1);
  unsigned col_c2=(threadID_in_group * 2) + (2 & 0x1);
  unsigned col_c3=(threadID_in_group * 2) + (3 & 0x1);

  unsigned A_offset;
  unsigned base_idx;
  unsigned dense_B_idx_0, dense_B_idx_1, dense_B_idx_2, dense_B_idx_3;

  for (unsigned i = block_start; i < block_end; i++){
    // load A tile
    A_offset = i*tile_M*tile_K_sp;
    RA[0] = __ldg(reinterpret_cast<const uint32_t*>(&opd_A[A_offset + sparse_A_idx_0]));
    RA[1] = __ldg(reinterpret_cast<const uint32_t*>(&opd_A[A_offset + sparse_A_idx_1]));

    // load B tile
    base_idx = i * tile_K;
    dense_B_idx_0 = old_col_all[base_idx + row_b0] * N + wid_tile_N + groupID;
    dense_B_idx_1 = old_col_all[base_idx + row_b1] * N + wid_tile_N + groupID;
    dense_B_idx_2 = old_col_all[base_idx + row_b2] * N + wid_tile_N + groupID;
    dense_B_idx_3 = old_col_all[base_idx + row_b3] * N + wid_tile_N + groupID;

    half* ptr_b = reinterpret_cast<half*>(RB);
    ptr_b[0] = __ldg(&input_B[dense_B_idx_0]);
    ptr_b[1] = __ldg(&input_B[dense_B_idx_1]);
    ptr_b[2] = __ldg(&input_B[dense_B_idx_2]);
    ptr_b[3] = __ldg(&input_B[dense_B_idx_3]);

    // load matadata
    RM = GPU_metadata[8 * i + laneid/4];

    __syncwarp();
    // __syncthreads();

    // sparse tensor core compute
    SPMMA16816(RC[0], RC[1], RC[2], RC[3], RA[0], RA[1], RB[0], RB[1], RC[0], RC[1], RC[2], RC[3], RM);

  }
  
  // write back
  unsigned C_offset = bid*tile_M*N + wid_tile_N;
  unsigned out_C_idx_0 = C_offset + groupID * N + col_c0;
  unsigned out_C_idx_1 = C_offset + groupID * N + col_c1;
  unsigned out_C_idx_2 = C_offset + (groupID+8) * N + col_c2;
  unsigned out_C_idx_3 = C_offset + (groupID+8) * N + col_c3;

  output_C[out_C_idx_0]=RC[0];
  output_C[out_C_idx_1]=RC[1];
  output_C[out_C_idx_2]=RC[2];
  output_C[out_C_idx_3]=RC[3];
}



__global__ void sparse_mma_kernel_base_Cfloat2(
  int M, int N, int K,
  const half *__restrict__ opd_A, 
  const uint32_t *__restrict__ GPU_metadata,
  const int *__restrict__ tcblocks_offset,
  const int *__restrict__ old_col_all,
  const half *__restrict__ input_B, 
  float *output_C)
{
  // thread block index : row window index
  // each thread block handel each row window
  // one dimension thread grid
  const unsigned bid = blockIdx.x;
  unsigned block_start = tcblocks_offset[bid];
  unsigned block_end = tcblocks_offset[bid + 1];
  if(block_start==block_end) return;

  // two dimension thread block
  const unsigned wid = threadIdx.y;
  const unsigned laneid = threadIdx.x;
  const unsigned wid_tile_N = wid * tile_N;

  const unsigned groupID = laneid >> 2;
  const unsigned threadID_in_group = laneid % 4;
  const unsigned threadID_in_group_x2 = threadID_in_group*2;

  uint32_t RA[2];
  uint32_t RB[2];
  float RC[4] = {0.0f};
  uint32_t RM;
  
  // A index
  unsigned sparse_A_idx_0 = groupID * tile_K_sp + threadID_in_group_x2;
  unsigned sparse_A_idx_1 = (groupID + 8) * tile_K_sp + threadID_in_group_x2;

  // B index
  unsigned row_b0=(threadID_in_group * 2) + (0 & 0x1);
  unsigned row_b1=(threadID_in_group * 2) + (1 & 0x1);
  unsigned row_b2=(threadID_in_group * 2) + (2 & 0x1) + 8 ;
  unsigned row_b3=(threadID_in_group * 2) + (3 & 0x1) + 8 ;

  
  unsigned A_offset;
  unsigned base_idx;
  unsigned dense_B_idx_0, dense_B_idx_1, dense_B_idx_2, dense_B_idx_3;

  for (unsigned i = block_start; i < block_end; i++){
    // load A tile
    A_offset = i*tile_M*tile_K_sp;
    RA[0] = __ldg(reinterpret_cast<const uint32_t*>(&opd_A[A_offset + sparse_A_idx_0]));
    RA[1] = __ldg(reinterpret_cast<const uint32_t*>(&opd_A[A_offset + sparse_A_idx_1]));

    // load B tile
    base_idx = i * tile_K;
    dense_B_idx_0 = old_col_all[base_idx + row_b0] * N + wid_tile_N + groupID;
    dense_B_idx_1 = old_col_all[base_idx + row_b1] * N + wid_tile_N + groupID;
    dense_B_idx_2 = old_col_all[base_idx + row_b2] * N + wid_tile_N + groupID;
    dense_B_idx_3 = old_col_all[base_idx + row_b3] * N + wid_tile_N + groupID;

    half* ptr_b = reinterpret_cast<half*>(RB);
    ptr_b[0] = __ldg(&input_B[dense_B_idx_0]);
    ptr_b[1] = __ldg(&input_B[dense_B_idx_1]);
    ptr_b[2] = __ldg(&input_B[dense_B_idx_2]);
    ptr_b[3] = __ldg(&input_B[dense_B_idx_3]);

    // load matadata
    RM = GPU_metadata[8 * i + laneid/4];

    __syncwarp();
    // __syncthreads();

    // sparse tensor core compute
    SPMMA16816(RC[0], RC[1], RC[2], RC[3], RA[0], RA[1], RB[0], RB[1], RC[0], RC[1], RC[2], RC[3], RM);

  }
  
  // write back

  // C index
  // unsigned col_c0=(threadID_in_group * 2);
  // unsigned col_c1=(threadID_in_group * 2) + (1 & 0x1);
  // unsigned col_c2=(threadID_in_group * 2);
  // unsigned col_c3=(threadID_in_group * 2) + (3 & 0x1);


  unsigned C_offset = bid*tile_M*N + wid_tile_N;
  unsigned out_C_idx_0 = C_offset + groupID * N + threadID_in_group_x2;
  // unsigned out_C_idx_1 = C_offset + groupID * N + col_c1;
  unsigned out_C_idx_2 = out_C_idx_0 + 8*N;
  // unsigned out_C_idx_3 = C_offset + (groupID+8) * N + col_c3;

  *((float2*)(output_C + out_C_idx_0)) = make_float2(RC[0], RC[1]);
  *((float2*)(output_C + out_C_idx_2)) = make_float2(RC[2], RC[3]);
  // output_C[out_C_idx_0]=RC[0];
  // output_C[out_C_idx_1]=RC[1];
  // output_C[out_C_idx_2]=RC[2];
  // output_C[out_C_idx_3]=RC[3];
}



__global__ void sparse_mma_kernel_base_Bhalf2_Cfloat2(
  int M, int N, int K,
  const half *__restrict__ opd_A, 
  const uint32_t *__restrict__ GPU_metadata,
  const int *__restrict__ tcblocks_offset,
  const int *__restrict__ old_col_all,
  half *input_B, 
  float *output_C)
{
  // thread block index : row window index
  // each thread block handel each row window
  // one dimension thread grid
  const unsigned bid = blockIdx.x;
  unsigned block_start = tcblocks_offset[bid];
  unsigned block_end = tcblocks_offset[bid + 1];
  if(block_start==block_end) return;


  const unsigned threadPerBlock = blockDim.x * blockDim.y;

  // two dimension thread block
  const unsigned wid = threadIdx.y;
  const unsigned laneid = threadIdx.x;
  const unsigned wid_tile_N = wid * tile_N;

  const unsigned tid = wid * blockDim.x + laneid;	

  const unsigned groupID = laneid >> 2;
  const unsigned threadID_in_group = laneid % 4;

  uint32_t RA[2];
  uint32_t RB[4];
  float RC[8] = {0.0f};
  uint32_t RM;

  const unsigned groupID_x2 = groupID << 1;
  const unsigned threadID_in_group_x2 = threadID_in_group << 1; // threadID_in_group * 2
  const unsigned threadID_in_group_x4 = threadID_in_group << 2; // threadID_in_group * 4

  const unsigned warp_offset = wid<<4; // wid * 16

  // A index
  // const unsigned sparse_A_idx_0 = groupID * tile_K_sp + threadID_in_group_x2;
  // const unsigned sparse_A_idx_1 = (groupID + 8) * tile_K_sp + threadID_in_group_x2;
  const unsigned sparse_A_idx_0 = groupID*8 + threadID_in_group_x2;
  const unsigned sparse_A_idx_1 = sparse_A_idx_0+64;

  // B index
  const unsigned row_b0=threadID_in_group_x2 + 0;
  const unsigned row_b1=threadID_in_group_x2 + 1;
  const unsigned row_b2=threadID_in_group_x2 + 8;
  const unsigned row_b3=threadID_in_group_x2 + 9;

  unsigned A_offset;
  unsigned base_idx;
  // unsigned dense_B_idx_0, dense_B_idx_1, dense_B_idx_2, dense_B_idx_3;

  const unsigned dense_B_idx_base = groupID_x2 + warp_offset;
  unsigned dense_B_idx[4];
  half2 t[4];

  // const unsigned dimTileNum = N / 16; 

  // main loop
  for (unsigned i = block_start; i < block_end; i++){
    // load A tile
    A_offset = i*tile_M*tile_K_sp;
    RA[0] = __ldg(reinterpret_cast<const uint32_t*>(&opd_A[A_offset + sparse_A_idx_0]));
    RA[1] = __ldg(reinterpret_cast<const uint32_t*>(&opd_A[A_offset + sparse_A_idx_1]));

    // load matadata
    RM = GPU_metadata[8*i + laneid/4];

    // load B tile
    base_idx = i * tile_K;

    // dense_B_idx[0]=dense_B_idx_base+old_col_all[row_b0]*N;
    // dense_B_idx[1]=dense_B_idx_base+old_col_all[row_b1]*N;
    // dense_B_idx[2]=dense_B_idx_base+old_col_all[row_b2]*N;
    // dense_B_idx[3]=dense_B_idx_base+old_col_all[row_b3]*N;

    dense_B_idx[0]=old_col_all[base_idx + row_b0] * N + dense_B_idx_base;
    dense_B_idx[1]=old_col_all[base_idx + row_b1] * N + dense_B_idx_base;
    dense_B_idx[2]=old_col_all[base_idx + row_b2] * N + dense_B_idx_base;
    dense_B_idx[3]=old_col_all[base_idx + row_b3] * N + dense_B_idx_base;

    // dense_B_idx_0 = old_col_all[base_idx + row_b0] * N + groupID_x2 + warp_offset;
    // dense_B_idx_1 = old_col_all[base_idx + row_b1] * N + groupID_x2 + warp_offset;
    // dense_B_idx_2 = old_col_all[base_idx + row_b2] * N + groupID_x2 + warp_offset;
    // dense_B_idx_3 = old_col_all[base_idx + row_b3] * N + groupID_x2 + warp_offset;

    // load B tile from global memory to register
    half* ptr_b = reinterpret_cast<half*>(RB);
    #pragma unroll
    for (unsigned i = 0; i < 4; i++)
    {
      t[i] = HALF2(input_B[dense_B_idx[i]]);
    }

    #pragma unroll
    for (unsigned i = 0; i < 4; i++)
    {
      ptr_b[i] = __low2half(t[i]);
      ptr_b[i+4] = __high2half(t[i]);
    }
    __syncwarp();

    // half2 t0 = HALF2(input_B[dense_B_idx_0]);
    // half2 t1 = HALF2(input_B[dense_B_idx_1]);
    // half2 t2 = HALF2(input_B[dense_B_idx_2]);
    // half2 t3 = HALF2(input_B[dense_B_idx_3]);
    // ptr_b[0] = __low2half(t0);
    // ptr_b[1] = __low2half(t1);
    // ptr_b[2] = __low2half(t2);
    // ptr_b[3] = __low2half(t3);
    // ptr_b[4] = __high2half(t0);
    // ptr_b[5] = __high2half(t1);
    // ptr_b[6] = __high2half(t2);
    // ptr_b[7] = __high2half(t3);
    // __syncwarp();

    // sparse tensor core compute
    SPMMA16816(RC[0], RC[1], RC[2], RC[3], 
               RA[0], RA[1], 
               RB[0], RB[1], 
               RC[0], RC[1], RC[2], RC[3], 
               RM);
    SPMMA16816(RC[4], RC[5], RC[6], RC[7], 
               RA[0], RA[1], 
               RB[2], RB[3], 
               RC[4], RC[5], RC[6], RC[7], 
               RM);
  }
  
  // write back
  unsigned C_offset = bid*tile_M*N + warp_offset;
  unsigned out_C_idx_0 = C_offset + groupID * N + threadID_in_group_x4;
  unsigned out_C_idx_1 =out_C_idx_0 + 2;
  // unsigned out_C_idx_1 = C_offset + groupID * N + threadID_in_group_x4 + 2;
  // unsigned out_C_idx_2 = C_offset + (groupID+8) * N + threadID_in_group_x4 + 0;
  // unsigned out_C_idx_3 = C_offset + (groupID+8) * N + threadID_in_group_x4 + 2;
  unsigned out_C_idx_2 = out_C_idx_0 + N*8;
  unsigned out_C_idx_3 = out_C_idx_1 + N*8;

  *((float2*)(output_C + out_C_idx_0)) = make_float2(RC[0], RC[4]);
  *((float2*)(output_C + out_C_idx_1)) = make_float2(RC[1], RC[5]);
  *((float2*)(output_C + out_C_idx_2)) = make_float2(RC[2], RC[6]);
  *((float2*)(output_C + out_C_idx_3)) = make_float2(RC[3], RC[7]);
}



__global__ void sparse_mma_kernel_base_Bhalf2_Cfloat4(
  int M, int N, int K,
  const half *__restrict__ opd_A, 
  const uint32_t *__restrict__ GPU_metadata,
  const int *__restrict__ tcblocks_offset,
  const int *__restrict__ old_col_all,
  half *input_B, 
  float *output_C)
{
  // thread block index : row window index
  // each thread block handel each row window
  // one dimension thread grid
  const unsigned bid = blockIdx.x;
  unsigned block_start = tcblocks_offset[bid];
  unsigned block_end = tcblocks_offset[bid + 1];
  if(block_start==block_end) return;


  const unsigned threadPerBlock = blockDim.x * blockDim.y;

  // two dimension thread block
  const unsigned wid = threadIdx.y;
  const unsigned laneid = threadIdx.x;
  const unsigned wid_tile_N = wid * tile_N;

  const unsigned tid = wid * blockDim.x + laneid;	

  const unsigned groupID = laneid >> 2;
  const unsigned threadID_in_group = laneid % 4;

  uint32_t RA[2];
  uint32_t RB[4];
  float RC[8] = {0.0f};
  uint32_t RM;

  const unsigned groupID_x2 = groupID << 1;
  const unsigned threadID_in_group_x2 = threadID_in_group << 1; // threadID_in_group * 2
  const unsigned threadID_in_group_x4 = threadID_in_group << 2; // threadID_in_group * 4

  const unsigned warp_offset = wid<<4; // wid * 16

  // A index
  // const unsigned sparse_A_idx_0 = groupID * tile_K_sp + threadID_in_group_x2;
  // const unsigned sparse_A_idx_1 = (groupID + 8) * tile_K_sp + threadID_in_group_x2;
  const unsigned sparse_A_idx_0 = groupID*8 + threadID_in_group_x2;
  const unsigned sparse_A_idx_1 = sparse_A_idx_0+64;

  // B index
  const unsigned row_b0=threadID_in_group_x2 + 0;
  const unsigned row_b1=threadID_in_group_x2 + 1;
  const unsigned row_b2=threadID_in_group_x2 + 8;
  const unsigned row_b3=threadID_in_group_x2 + 9;

  unsigned A_offset;
  unsigned base_idx;
  // unsigned dense_B_idx_0, dense_B_idx_1, dense_B_idx_2, dense_B_idx_3;

  const unsigned dense_B_idx_base = groupID_x2 + warp_offset;
  unsigned dense_B_idx[4];
  half2 t[4];

  // const unsigned dimTileNum = N / 16; 

  // main loop
  for (unsigned i = block_start; i < block_end; i++){
    // load A tile
    A_offset = i*tile_M*tile_K_sp;
    RA[0] = __ldg(reinterpret_cast<const uint32_t*>(&opd_A[A_offset + sparse_A_idx_0]));
    RA[1] = __ldg(reinterpret_cast<const uint32_t*>(&opd_A[A_offset + sparse_A_idx_1]));

    // load matadata
    RM = GPU_metadata[8*i + laneid/4];

    // load B tile
    base_idx = i * tile_K;

    // dense_B_idx[0]=dense_B_idx_base+old_col_all[row_b0]*N;
    // dense_B_idx[1]=dense_B_idx_base+old_col_all[row_b1]*N;
    // dense_B_idx[2]=dense_B_idx_base+old_col_all[row_b2]*N;
    // dense_B_idx[3]=dense_B_idx_base+old_col_all[row_b3]*N;

    dense_B_idx[0]=old_col_all[base_idx + row_b0] * N + dense_B_idx_base;
    dense_B_idx[1]=old_col_all[base_idx + row_b1] * N + dense_B_idx_base;
    dense_B_idx[2]=old_col_all[base_idx + row_b2] * N + dense_B_idx_base;
    dense_B_idx[3]=old_col_all[base_idx + row_b3] * N + dense_B_idx_base;

    // dense_B_idx_0 = old_col_all[base_idx + row_b0] * N + groupID_x2 + warp_offset;
    // dense_B_idx_1 = old_col_all[base_idx + row_b1] * N + groupID_x2 + warp_offset;
    // dense_B_idx_2 = old_col_all[base_idx + row_b2] * N + groupID_x2 + warp_offset;
    // dense_B_idx_3 = old_col_all[base_idx + row_b3] * N + groupID_x2 + warp_offset;

    // load B tile from global memory to register
    half* ptr_b = reinterpret_cast<half*>(RB);
    #pragma unroll
    for (unsigned i = 0; i < 4; i++)
    {
      t[i] = HALF2(input_B[dense_B_idx[i]]);
    }

    #pragma unroll
    for (unsigned i = 0; i < 4; i++)
    {
      ptr_b[i] = __low2half(t[i]);
      ptr_b[i+4] = __high2half(t[i]);
    }
    __syncwarp();

    // half2 t0 = HALF2(input_B[dense_B_idx_0]);
    // half2 t1 = HALF2(input_B[dense_B_idx_1]);
    // half2 t2 = HALF2(input_B[dense_B_idx_2]);
    // half2 t3 = HALF2(input_B[dense_B_idx_3]);
    // ptr_b[0] = __low2half(t0);
    // ptr_b[1] = __low2half(t1);
    // ptr_b[2] = __low2half(t2);
    // ptr_b[3] = __low2half(t3);
    // ptr_b[4] = __high2half(t0);
    // ptr_b[5] = __high2half(t1);
    // ptr_b[6] = __high2half(t2);
    // ptr_b[7] = __high2half(t3);
    // __syncwarp();

    // sparse tensor core compute
    SPMMA16816(RC[0], RC[1], RC[2], RC[3], 
               RA[0], RA[1], 
               RB[0], RB[1], 
               RC[0], RC[1], RC[2], RC[3], 
               RM);
    SPMMA16816(RC[4], RC[5], RC[6], RC[7], 
               RA[0], RA[1], 
               RB[2], RB[3], 
               RC[4], RC[5], RC[6], RC[7], 
               RM);
  }
  
  // write back
  unsigned C_offset = bid*tile_M*N + warp_offset;
  unsigned out_C_idx_0 = C_offset + groupID * N + threadID_in_group_x4;
  // unsigned out_C_idx_1 =out_C_idx_0 + 2;
  unsigned out_C_idx_2 = out_C_idx_0 + N*8;
  // unsigned out_C_idx_3 = out_C_idx_1 + N*8;

  *((float4*)(output_C + out_C_idx_0)) = make_float4(RC[0], RC[4], RC[1], RC[5]);
  // *((float2*)(output_C + out_C_idx_1)) = make_float2(RC[1], RC[5]);
  *((float4*)(output_C + out_C_idx_2)) = make_float4(RC[2], RC[6], RC[3], RC[7]);
  // *((float2*)(output_C + out_C_idx_3)) = make_float2(RC[3], RC[7]);
}



__global__ void sparse_mma_kernel_base_Apipeline_Bhalf2_Cfloat2(
  int M, int N, int K,
  const half *__restrict__ opd_A, 
  const uint32_t *__restrict__ GPU_metadata,
  const int *__restrict__ tcblocks_offset,
  const int *__restrict__ old_col_all,
  half *input_B, 
  float *output_C)
{
  // thread block index : row window index
  // each thread block handel each row window
  // one dimension thread grid
  const unsigned bid = blockIdx.x;

  const unsigned block_start = tcblocks_offset[bid];
  const unsigned block_end = tcblocks_offset[bid + 1];
  if(block_start==block_end) return;

  const unsigned threadPerBlock = blockDim.x * blockDim.y;

  // two dimension thread block
  const unsigned wid = threadIdx.y;
  const unsigned laneid = threadIdx.x;
  const unsigned wid_tile_N = wid * tile_N;

  const unsigned tid = wid * blockDim.x + laneid;	

  const unsigned groupID = laneid >> 2;
  const unsigned threadID_in_group = laneid % 4;

  uint32_t RA[4]; // double
  uint32_t RM[2]; // double
  uint32_t RB[4];
  float RC[8] = {0.0f};
  

  const unsigned groupID_x2 = groupID << 1;
  const unsigned threadID_in_group_x2 = threadID_in_group << 1; // threadID_in_group * 2
  const unsigned threadID_in_group_x4 = threadID_in_group << 2; // threadID_in_group * 4

  const unsigned warp_offset = wid<<4; // wid * 16

  // A index
  // const unsigned sparse_A_idx_0 = groupID * tile_K_sp + threadID_in_group_x2;
  // const unsigned sparse_A_idx_1 = (groupID + 8) * tile_K_sp + threadID_in_group_x2;
  const unsigned sparse_A_idx_0 = groupID*8 + threadID_in_group_x2;
  const unsigned sparse_A_idx_1 = sparse_A_idx_0+64;

  // B index
  const unsigned row_b0=threadID_in_group_x2 + 0;
  const unsigned row_b1=threadID_in_group_x2 + 1;
  const unsigned row_b2=threadID_in_group_x2 + 8;
  const unsigned row_b3=threadID_in_group_x2 + 9;

  

  unsigned A_offset;
  unsigned base_idx;
  // unsigned dense_B_idx_0, dense_B_idx_1, dense_B_idx_2, dense_B_idx_3;

  const unsigned dense_B_idx_base = groupID_x2 + warp_offset;
  unsigned dense_B_idx[4];
  half2 t[4];

  // const unsigned dimTileNum = N / 16; 

  // prefetch opd and metadata for A tile
  A_offset = block_start*tile_M*tile_K_sp;
  RA[0] = __ldg(reinterpret_cast<const uint32_t*>(&opd_A[A_offset + sparse_A_idx_0]));
  RA[1] = __ldg(reinterpret_cast<const uint32_t*>(&opd_A[A_offset + sparse_A_idx_1]));

  // load matadata
  RM[0] = GPU_metadata[8*block_start + laneid/4];

  __syncwarp();

  // main loop
  for (unsigned i = block_start+1; i < block_end; i++){

  
    // load current B tile
    base_idx = (i-1) * tile_K;

    // dense_B_idx[0]=dense_B_idx_base+old_col_all[row_b0]*N;
    // dense_B_idx[1]=dense_B_idx_base+old_col_all[row_b1]*N;
    // dense_B_idx[2]=dense_B_idx_base+old_col_all[row_b2]*N;
    // dense_B_idx[3]=dense_B_idx_base+old_col_all[row_b3]*N;

    dense_B_idx[0]=old_col_all[base_idx + row_b0] * N + dense_B_idx_base;
    dense_B_idx[1]=old_col_all[base_idx + row_b1] * N + dense_B_idx_base;
    dense_B_idx[2]=old_col_all[base_idx + row_b2] * N + dense_B_idx_base;
    dense_B_idx[3]=old_col_all[base_idx + row_b3] * N + dense_B_idx_base;

    // dense_B_idx_0 = old_col_all[base_idx + row_b0] * N + groupID_x2 + warp_offset;
    // dense_B_idx_1 = old_col_all[base_idx + row_b1] * N + groupID_x2 + warp_offset;
    // dense_B_idx_2 = old_col_all[base_idx + row_b2] * N + groupID_x2 + warp_offset;
    // dense_B_idx_3 = old_col_all[base_idx + row_b3] * N + groupID_x2 + warp_offset;

    // load B tile from global memory to register
    half* ptr_b = reinterpret_cast<half*>(RB);
    #pragma unroll
    for (unsigned j = 0; j < 4; j++)
    {
      t[j] = HALF2(input_B[dense_B_idx[j]]);
    }

    #pragma unroll
    for (unsigned j = 0; j < 4; j++)
    {
      ptr_b[j] = __low2half(t[j]);
      ptr_b[j+4] = __high2half(t[j]);
    }
    __syncwarp();

    // half2 t0 = HALF2(input_B[dense_B_idx_0]);
    // half2 t1 = HALF2(input_B[dense_B_idx_1]);
    // half2 t2 = HALF2(input_B[dense_B_idx_2]);
    // half2 t3 = HALF2(input_B[dense_B_idx_3]);
    // ptr_b[0] = __low2half(t0);
    // ptr_b[1] = __low2half(t1);
    // ptr_b[2] = __low2half(t2);
    // ptr_b[3] = __low2half(t3);
    // ptr_b[4] = __high2half(t0);
    // ptr_b[5] = __high2half(t1);
    // ptr_b[6] = __high2half(t2);
    // ptr_b[7] = __high2half(t3);
    // __syncwarp();


    // load next A tile

    unsigned cur = ((i - block_start) & 1) ^ 1;
    unsigned next = ((i - block_start - 1) & 1) ^ 1;

    // unsigned next_ptr = (i-block_start)&1;
    A_offset = i*tile_M*tile_K_sp;
    RA[next*2] = __ldg(reinterpret_cast<const uint32_t*>(&opd_A[A_offset + sparse_A_idx_0]));
    RA[next*2+1] = __ldg(reinterpret_cast<const uint32_t*>(&opd_A[A_offset + sparse_A_idx_1]));

    // load next matadata
    RM[next] = GPU_metadata[8*i + laneid/4];

    // unsigned cur_ptr = (i-block_start-1)&1;

    __syncwarp();
    // sparse tensor core compute
    SPMMA16816(RC[0], RC[1], RC[2], RC[3], 
               RA[cur*2], RA[cur*2+1], 
               RB[0], RB[1], 
               RC[0], RC[1], RC[2], RC[3], 
               RM[cur]);
    SPMMA16816(RC[4], RC[5], RC[6], RC[7], 
               RA[cur*2], RA[cur*2+1], 
               RB[2], RB[3], 
               RC[4], RC[5], RC[6], RC[7], 
               RM[cur]);
  }

  // process last block
  // load last B tile
  base_idx = (block_end-1)*tile_K;
  dense_B_idx[0]=old_col_all[base_idx + row_b0] * N + dense_B_idx_base;
  dense_B_idx[1]=old_col_all[base_idx + row_b1] * N + dense_B_idx_base;
  dense_B_idx[2]=old_col_all[base_idx + row_b2] * N + dense_B_idx_base;
  dense_B_idx[3]=old_col_all[base_idx + row_b3] * N + dense_B_idx_base;

  half* ptr_b = reinterpret_cast<half*>(RB);
  #pragma unroll
  for (unsigned j = 0; j < 4; j++)
  {
    t[j] = HALF2(input_B[dense_B_idx[j]]);
  }

  #pragma unroll
  for (unsigned j = 0; j < 4; j++)
  {
    ptr_b[j] = __low2half(t[j]);
    ptr_b[j+4] = __high2half(t[j]);
  }
  __syncwarp();

  unsigned cur = ((block_end-block_start) & 1) ^ 1;
  // sparse tensor core compute
  SPMMA16816(RC[0], RC[1], RC[2], RC[3], 
            RA[cur*2], RA[cur*2+1], 
            RB[0], RB[1], 
            RC[0], RC[1], RC[2], RC[3], 
            RM[cur]);

  SPMMA16816(RC[4], RC[5], RC[6], RC[7], 
            RA[cur*2], RA[cur*2+1], 
            RB[2], RB[3], 
            RC[4], RC[5], RC[6], RC[7], 
            RM[cur]);
  
  __syncwarp();

  // write back
  unsigned C_offset = bid*tile_M*N + warp_offset;
  unsigned out_C_idx_0 = C_offset + groupID * N + threadID_in_group_x4;
  unsigned out_C_idx_1 =out_C_idx_0 + 2;
  // unsigned out_C_idx_1 = C_offset + groupID * N + threadID_in_group_x4 + 2;
  // unsigned out_C_idx_2 = C_offset + (groupID+8) * N + threadID_in_group_x4 + 0;
  // unsigned out_C_idx_3 = C_offset + (groupID+8) * N + threadID_in_group_x4 + 2;
  unsigned out_C_idx_2 = out_C_idx_0 + N*8;
  unsigned out_C_idx_3 = out_C_idx_1 + N*8;

  *((float2*)(output_C + out_C_idx_0)) = make_float2(RC[0], RC[4]);
  *((float2*)(output_C + out_C_idx_1)) = make_float2(RC[1], RC[5]);
  *((float2*)(output_C + out_C_idx_2)) = make_float2(RC[2], RC[6]);
  *((float2*)(output_C + out_C_idx_3)) = make_float2(RC[3], RC[7]);
}



#define UINT64(pointer) (reinterpret_cast<uint64_t*>(&(pointer))[0])

__global__ void sparse_mma_kernel_base_Buint64_Cfloat4(
  int M, int N, int K,
  const half *__restrict__ opd_A, 
  const uint32_t *__restrict__ GPU_metadata,
  const int *__restrict__ tcblocks_offset,
  const int *__restrict__ old_col_all,
  half *input_B, 
  float *output_C)
{
  // thread block index : row window index
  // each thread block handel each row window
  // one dimension thread grid
  const unsigned bid = blockIdx.x;
  unsigned block_start = tcblocks_offset[bid];
  unsigned block_end = tcblocks_offset[bid + 1];
  if(block_start==block_end) return;

  const unsigned threadPerBlock = blockDim.x * blockDim.y;

  // two dimension thread block
  const unsigned wid = threadIdx.y;
  const unsigned laneid = threadIdx.x;
  const unsigned wid_tile_N = wid * tile_N;

  const unsigned tid = wid * blockDim.x + laneid;	

  const unsigned groupID = laneid >> 2;
  const unsigned threadID_in_group = laneid % 4;

  // __shared__  half tile_A_smem[A_tile_size];
  // __shared__ __align__(16) half tile_A_smem[A_tile_size];

  // uint32_t RA[2];
  // uint32_t RB[2];
  // float RC[4] = {0.0f};
  // uint32_t RM;

  uint32_t RA[2];
  uint32_t RB[8];
  float RC[16] = {0.0f};
  uint32_t RM;

  unsigned threadID_in_group_x2 = threadID_in_group << 1; // threadID_in_group * 2
  unsigned threadID_in_group_x4 = threadID_in_group << 2; // threadID_in_group * 4
  
  // A index
  unsigned sparse_A_idx_0 = groupID * tile_K_sp + threadID_in_group_x2;
  unsigned sparse_A_idx_1 = sparse_A_idx_0+64;

  // B index
  unsigned row_b0=threadID_in_group_x2;
  unsigned row_b1=threadID_in_group_x2+1;
  unsigned row_b2=threadID_in_group_x2+8;
  unsigned row_b3=threadID_in_group_x2+9;

  unsigned A_offset;
  unsigned base_idx;
  unsigned dense_B_idx_0, dense_B_idx_1, dense_B_idx_2, dense_B_idx_3;
  unsigned warp_offset = wid<<5;
  unsigned dense_B_idx_base = (groupID<<2) + warp_offset;

  // main loop
  for (unsigned i = block_start; i < block_end; i++){
    // load A tile
    A_offset = i*tile_M*tile_K_sp;
    RA[0] = __ldg(reinterpret_cast<const uint32_t*>(&opd_A[A_offset + sparse_A_idx_0]));
    RA[1] = __ldg(reinterpret_cast<const uint32_t*>(&opd_A[A_offset + sparse_A_idx_1]));

    // load matadata
    RM = GPU_metadata[8 * i + laneid/4];

    // load B tile
    base_idx = i * tile_K;
    dense_B_idx_0 = old_col_all[base_idx + row_b0] * N + dense_B_idx_base;
    dense_B_idx_1 = old_col_all[base_idx + row_b1] * N + dense_B_idx_base;
    dense_B_idx_2 = old_col_all[base_idx + row_b2] * N + dense_B_idx_base;
    dense_B_idx_3 = old_col_all[base_idx + row_b3] * N + dense_B_idx_base;

    // load B tile from global memory to register
    half* ptr_b = reinterpret_cast<half*>(RB);
    
    // const uint64_t* src64 = reinterpret_cast<const uint64_t*>(src);
    uint64_t t0 = UINT64(input_B[dense_B_idx_0]);
    uint64_t t1 = UINT64(input_B[dense_B_idx_1]);
    uint64_t t2 = UINT64(input_B[dense_B_idx_2]);
    uint64_t t3 = UINT64(input_B[dense_B_idx_3]);

    half* t0_ptr = reinterpret_cast<half*>(&t0);
    half* t1_ptr = reinterpret_cast<half*>(&t1);
    half* t2_ptr = reinterpret_cast<half*>(&t2);
    half* t3_ptr = reinterpret_cast<half*>(&t3);

    // for mma 0
    ptr_b[0] = t0_ptr[0];
    ptr_b[1] = t1_ptr[0];
    ptr_b[2] = t2_ptr[0];
    ptr_b[3] = t3_ptr[0];

    // for mma 1
    ptr_b[4] = t0_ptr[1];
    ptr_b[5] = t1_ptr[1];
    ptr_b[6] = t2_ptr[1];
    ptr_b[7] = t3_ptr[1];

    // for mma 2
    ptr_b[8] = t0_ptr[2];
    ptr_b[9] = t1_ptr[2];
    ptr_b[10] = t2_ptr[2];
    ptr_b[11] = t3_ptr[2];

    // for mma 3
    ptr_b[12] = t0_ptr[3];
    ptr_b[13] = t1_ptr[3];
    ptr_b[14] = t2_ptr[3];
    ptr_b[15] = t3_ptr[3];

    __syncwarp();

    // sparse tensor core compute
    SPMMA16816(RC[0], RC[1], RC[2], RC[3], 
               RA[0], RA[1], 
               RB[0], RB[1], 
               RC[0], RC[1], RC[2], RC[3], 
               RM);
    SPMMA16816(RC[4], RC[5], RC[6], RC[7], 
               RA[0], RA[1], 
               RB[2], RB[3], 
               RC[4], RC[5], RC[6], RC[7], 
               RM);
    SPMMA16816(RC[8], RC[9], RC[10], RC[11], 
               RA[0], RA[1], 
               RB[4], RB[5], 
               RC[8], RC[9], RC[10], RC[11], 
               RM);
    SPMMA16816(RC[12], RC[13], RC[14], RC[15], 
               RA[0], RA[1], 
               RB[6], RB[7], 
               RC[12], RC[13], RC[14], RC[15], 
               RM);

  }
  
  // write back
  unsigned C_offset = bid*tile_M*N + warp_offset;
  unsigned out_C_idx_0 = C_offset+groupID*N+threadID_in_group*8;
  unsigned out_C_idx_1 = out_C_idx_0+4;
  unsigned out_C_idx_2 = out_C_idx_0+N*8;
  unsigned out_C_idx_3 = out_C_idx_1+N*8;

  *((float4*)(output_C + out_C_idx_0)) = make_float4(RC[0], RC[4], RC[8], RC[12]);
  *((float4*)(output_C + out_C_idx_1)) = make_float4(RC[1], RC[5], RC[9], RC[13]);
  *((float4*)(output_C + out_C_idx_2)) = make_float4(RC[2], RC[6], RC[10], RC[14]);
  *((float4*)(output_C + out_C_idx_3)) = make_float4(RC[3], RC[7], RC[11], RC[15]);

}

