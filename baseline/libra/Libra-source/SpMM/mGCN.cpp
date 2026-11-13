#include <torch/extension.h>
#include <cuda_fp16.h>
#include <assert.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
inline
cudaError_t checkCuda(cudaError_t result){
    if (result != cudaSuccess){
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}

// cuda fp16

// cuda 长短行 fp16

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
    bool swizzle);
std::vector<torch::Tensor> spmm_forward_fp16_cuda_v2(
    
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

    torch::Tensor rhs_matrix,

    int partsize,
    const int dimN,
    const int mOri,
    const int kOri,
    int epoches,
    int parts,
    int parts_short,
    bool swizzle)
{
    // int dimM=dimM1/16;
    auto output_matrix = torch::zeros({mOri, dimN}, torch::kFloat32).to(torch::kCPU);
    
    //把CPU端的tensor转成C++的数据结构
    int * c_row_offsets_ = c_row_offsets.data<int>();
    int * c_row_ = c_row.data<int>();
    int * c_atomic_ = c_atomic.data<int>();
    int * c_column_ = c_column.data<int>();
    half * c_value_ = reinterpret_cast<half *>(c_value.data<at::Half>()); 

    int * c_row_offsets_short_ = c_row_offsets_short.data<int>();
    int * c_row_short_ = c_row_short.data<int>();
    int * c_atomic_short_ = c_atomic_short.data<int>();
    int * c_column_short_ = c_column_short.data<int>();
    half * c_value_short_ = reinterpret_cast<half *>(c_value_short.data<at::Half>()); 

    half * rhs_matrix_ = reinterpret_cast<half *>(rhs_matrix.data<at::Half>()); 
    float * output_matrix_ = output_matrix.data<float>();

    // Device
    int *d_c_row_offsets, *d_c_row, *d_c_atomic, *d_c_column;
    int *d_c_row_offsets_short, *d_c_row_short, *d_c_atomic_short, *d_c_column_short;
    half *d_c_value, *d_c_value_short; 
	half *d_rhs_matrix;
    float *d_output_matrix;

    checkCuda(cudaMalloc(&d_c_row_offsets, (c_row_offsets.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_row, (c_row.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_atomic, (c_atomic.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_column, (c_column.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_value, (c_value.size(0)) * sizeof(half)));

    checkCuda(cudaMalloc(&d_c_row_offsets_short, (c_row_offsets_short.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_row_short, (c_row_short.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_atomic_short, (c_atomic_short.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_column_short, (c_column_short.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_value_short, (c_value_short.size(0)) * sizeof(half)));

    checkCuda(cudaMalloc(&d_rhs_matrix, (kOri*dimN) * sizeof(half)));
    checkCuda(cudaMalloc(&d_output_matrix, (mOri*dimN) * sizeof(float)));

    //把CPU数据放到GPU上
    checkCuda(cudaMemcpy(d_c_row_offsets, c_row_offsets_ , (c_row_offsets.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_row, c_row_, (c_row.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_atomic, c_atomic_, (c_atomic.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_column, c_column_, (c_column.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_value, c_value_, (c_value.size(0)) * sizeof(half), cudaMemcpyHostToDevice));

    checkCuda(cudaMemcpy(d_c_row_offsets_short, c_row_offsets_short_ , (c_row_offsets_short.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_row_short, c_row_short_, (c_row_short.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_atomic_short, c_atomic_short_, (c_atomic_short.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_column_short, c_column_short_, (c_column_short.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_value_short, c_value_short_, (c_value_short.size(0)) * sizeof(half), cudaMemcpyHostToDevice));
    
    checkCuda(cudaMemcpy(d_rhs_matrix, rhs_matrix_,(kOri*dimN) * sizeof(half), cudaMemcpyHostToDevice));
    float spmm_ms_avg = 0.0;

     spmm_ms_avg =  spmm_forward_fp16_cuda_kernel_split(

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

      d_rhs_matrix,
      d_output_matrix,

      partsize,
      dimN,
      mOri,
      epoches,
      parts,
      parts_short,
      swizzle);
      
    checkCuda(cudaMemcpy(output_matrix_, d_output_matrix, mOri * dimN * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_c_row_offsets);
    cudaFree(d_c_row);
    cudaFree(d_c_atomic);
    cudaFree(d_c_column);
    cudaFree(d_c_value);

    cudaFree(d_c_row_offsets_short);
    cudaFree(d_c_row_short);
    cudaFree(d_c_atomic_short);
    cudaFree(d_c_column_short);
    cudaFree(d_c_value_short);

    cudaFree(d_rhs_matrix);
    cudaFree(d_output_matrix);
    cudaDeviceSynchronize(); 
    return {output_matrix, torch::tensor(spmm_ms_avg)};
}


float spmm_forward_fp16_cuda_kernel_split_v1(

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
    bool swizzle);
std::vector<torch::Tensor> spmm_forward_fp16_cuda_v1(
    
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

    torch::Tensor rhs_matrix,

    int partsize,
    const int dimN,
    const int mOri,
    const int kOri,
    int epoches,
    int parts,
    int parts_short,
    bool swizzle)
{
    // int dimM=dimM1/16;
    auto output_matrix = torch::zeros({mOri, dimN}, torch::kFloat32).to(torch::kCPU);
    
    //把CPU端的tensor转成C++的数据结构
    int * c_row_offsets_ = c_row_offsets.data<int>();
    int * c_row_ = c_row.data<int>();
    int * c_atomic_ = c_atomic.data<int>();
    int * c_column_ = c_column.data<int>();
    half * c_value_ = reinterpret_cast<half *>(c_value.data<at::Half>()); 

    int * c_row_offsets_short_ = c_row_offsets_short.data<int>();
    int * c_row_short_ = c_row_short.data<int>();
    int * c_atomic_short_ = c_atomic_short.data<int>();
    int * c_column_short_ = c_column_short.data<int>();
    half * c_value_short_ = reinterpret_cast<half *>(c_value_short.data<at::Half>()); 

    half * rhs_matrix_ = reinterpret_cast<half *>(rhs_matrix.data<at::Half>()); 
    float * output_matrix_ = output_matrix.data<float>();

    // Device
    int *d_c_row_offsets, *d_c_row, *d_c_atomic, *d_c_column;
    int *d_c_row_offsets_short, *d_c_row_short, *d_c_atomic_short, *d_c_column_short;
    half *d_c_value, *d_c_value_short; 
	half *d_rhs_matrix;
    float *d_output_matrix;

    checkCuda(cudaMalloc(&d_c_row_offsets, (c_row_offsets.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_row, (c_row.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_atomic, (c_atomic.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_column, (c_column.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_value, (c_value.size(0)) * sizeof(half)));

    checkCuda(cudaMalloc(&d_c_row_offsets_short, (c_row_offsets_short.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_row_short, (c_row_short.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_atomic_short, (c_atomic_short.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_column_short, (c_column_short.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_value_short, (c_value_short.size(0)) * sizeof(half)));

    checkCuda(cudaMalloc(&d_rhs_matrix, (kOri*dimN) * sizeof(half)));
    checkCuda(cudaMalloc(&d_output_matrix, (mOri*dimN) * sizeof(float)));

    //把CPU数据放到GPU上
    checkCuda(cudaMemcpy(d_c_row_offsets, c_row_offsets_ , (c_row_offsets.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_row, c_row_, (c_row.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_atomic, c_atomic_, (c_atomic.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_column, c_column_, (c_column.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_value, c_value_, (c_value.size(0)) * sizeof(half), cudaMemcpyHostToDevice));

    checkCuda(cudaMemcpy(d_c_row_offsets_short, c_row_offsets_short_ , (c_row_offsets_short.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_row_short, c_row_short_, (c_row_short.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_atomic_short, c_atomic_short_, (c_atomic_short.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_column_short, c_column_short_, (c_column_short.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_value_short, c_value_short_, (c_value_short.size(0)) * sizeof(half), cudaMemcpyHostToDevice));
    
    checkCuda(cudaMemcpy(d_rhs_matrix, rhs_matrix_,(kOri*dimN) * sizeof(half), cudaMemcpyHostToDevice));
    float spmm_ms_avg = 0.0;

     spmm_ms_avg =  spmm_forward_fp16_cuda_kernel_split_v1(

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

      d_rhs_matrix,
      d_output_matrix,

      partsize,
      dimN,
      mOri,
      epoches,
      parts,
      parts_short,
      swizzle);
      
    checkCuda(cudaMemcpy(output_matrix_, d_output_matrix, mOri * dimN * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_c_row_offsets);
    cudaFree(d_c_row);
    cudaFree(d_c_atomic);
    cudaFree(d_c_column);
    cudaFree(d_c_value);

    cudaFree(d_c_row_offsets_short);
    cudaFree(d_c_row_short);
    cudaFree(d_c_atomic_short);
    cudaFree(d_c_column_short);
    cudaFree(d_c_value_short);

    cudaFree(d_rhs_matrix);
    cudaFree(d_output_matrix);
    cudaDeviceSynchronize(); 
    return {output_matrix, torch::tensor(spmm_ms_avg)};
}


// tcu cuda fp16
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
    int epoches);

    // tcu cuda fp16


std::vector<torch::Tensor> spmm_forward_fp16_tcu_cuda(
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

    torch::Tensor rhs_matrix,

    const int parts_t,
    const int parts_c,
    const int partsize_c,
    const int parts_c_short,
    const int dimN,
    const int mOri,
    const int kOri,
    int epoches)
{
    auto output_matrix = torch::zeros({mOri, dimN}, torch::kFloat32).to(torch::kCPU);
    
    //把CPU端的tensor转成C++的数据结构
    int * t_row_offset_ = t_row_offset.data<int>();
    int * t_blockNew_offset_ = t_blockNew_offset.data<int>();
    half * t_value_ = reinterpret_cast<half *>(t_value.data<at::Half>()); 
    int * t_column_ = t_column.data<int>();
    int * t_window_row_ = t_window_row.data<int>();
    int * t_atomic_ = t_atomic.data<int>();
    long * t_binary_ = t_binary.data<long>();
    
    int * c_row_offsets_ = c_row_offsets.data<int>();
    int * c_row_ = c_row.data<int>();
    int * c_atomic_ = c_atomic.data<int>();
    int * c_column_ = c_column.data<int>();
    half * c_value_ = reinterpret_cast<half *>(c_value.data<at::Half>()); 

    int * c_row_offsets_short_ = c_row_offsets_short.data<int>();
    int * c_row_short_ = c_row_short.data<int>();
    int * c_atomic_short_ = c_atomic_short.data<int>();
    int * c_column_short_ = c_column_short.data<int>();
    half * c_value_short_ = reinterpret_cast<half *>(c_value_short.data<at::Half>()); 

    half * rhs_matrix_ = reinterpret_cast<half *>(rhs_matrix.data<at::Half>()); 
    float * output_matrix_ = output_matrix.data<float>();

    // Device
    int *d_t_row_offset, *d_t_blockNew_offset, *d_t_column;
    int *d_t_window_row, *d_t_atomic;
    long *d_t_binary;
    half *d_t_value; 
    int *d_c_row_offsets, *d_c_row, *d_c_atomic, *d_c_column;
    int *d_c_row_offsets_short, *d_c_row_short, *d_c_atomic_short, *d_c_column_short;
    half *d_c_value, *d_c_value_short; 
	half *d_rhs_matrix;
    float *d_output_matrix;

    checkCuda(cudaMalloc(&d_t_row_offset, (t_row_offset.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_blockNew_offset, (t_blockNew_offset.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_value, (t_value.size(0)) * sizeof(half)));
    checkCuda(cudaMalloc(&d_t_column, (t_column.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_window_row, (t_window_row.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_atomic, (t_atomic.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_binary, (t_binary.size(0)) * sizeof(long)));

    checkCuda(cudaMalloc(&d_c_row_offsets, (c_row_offsets.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_row, (c_row.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_atomic, (c_atomic.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_column, (c_column.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_value, (c_value.size(0)) * sizeof(half)));

    checkCuda(cudaMalloc(&d_c_row_offsets_short, (c_row_offsets_short.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_row_short, (c_row_short.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_atomic_short, (c_atomic_short.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_column_short, (c_column_short.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_value_short, (c_value_short.size(0)) * sizeof(half)));

    checkCuda(cudaMalloc(&d_rhs_matrix, (kOri*dimN) * sizeof(half)));
    checkCuda(cudaMalloc(&d_output_matrix, (mOri*dimN) * sizeof(float)));

    //把CPU数据放到GPU上
    checkCuda(cudaMemcpy(d_t_row_offset, t_row_offset_ , (t_row_offset.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_blockNew_offset, t_blockNew_offset_ , (t_blockNew_offset.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_value, t_value_, (t_value.size(0)) * sizeof(half), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_column, t_column_, (t_column.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_window_row, t_window_row_, (t_window_row.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_atomic, t_atomic_, (t_atomic.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_binary, t_binary_, (t_binary.size(0)) * sizeof(long), cudaMemcpyHostToDevice));

    checkCuda(cudaMemcpy(d_c_row_offsets, c_row_offsets_ , (c_row_offsets.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_row, c_row_, (c_row.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_atomic, c_atomic_, (c_atomic.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_column, c_column_, (c_column.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_value, c_value_, (c_value.size(0)) * sizeof(half), cudaMemcpyHostToDevice));

    checkCuda(cudaMemcpy(d_c_row_offsets_short, c_row_offsets_short_ , (c_row_offsets_short.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_row_short, c_row_short_, (c_row_short.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_atomic_short, c_atomic_short_, (c_atomic_short.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_column_short, c_column_short_, (c_column_short.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_value_short, c_value_short_, (c_value_short.size(0)) * sizeof(half), cudaMemcpyHostToDevice));

    checkCuda(cudaMemcpy(d_rhs_matrix, rhs_matrix_,(kOri*dimN) * sizeof(half), cudaMemcpyHostToDevice));
    float spmm_ms_avg = 0.0;
    
      spmm_ms_avg =  spmm_forward_fp16_tcu_cuda_kernel(
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

      d_rhs_matrix,
      d_output_matrix,

      parts_t,
      parts_c,
      partsize_c,
      parts_c_short,
      dimN,
      mOri,
      epoches);
      
    checkCuda(cudaMemcpy(output_matrix_, d_output_matrix, mOri * dimN * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_t_row_offset);
    cudaFree(d_t_blockNew_offset);
    cudaFree(d_t_value);
    cudaFree(d_t_column);
    cudaFree(d_t_window_row);
    cudaFree(d_t_atomic);
    cudaFree(d_t_binary);

    cudaFree(d_c_row_offsets);
    cudaFree(d_c_row);
    cudaFree(d_c_atomic);
    cudaFree(d_c_column);
    cudaFree(d_c_value);

    cudaFree(d_c_row_offsets_short);
    cudaFree(d_c_row_short);
    cudaFree(d_c_atomic_short);
    cudaFree(d_c_column_short);
    cudaFree(d_c_value_short);

    cudaFree(d_rhs_matrix);
    cudaFree(d_output_matrix);
    cudaDeviceSynchronize(); 
    return {output_matrix, torch::tensor(spmm_ms_avg)};
}


float spmm_forward_fp16_tcu_cuda_kernel_cc_only(
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
    int epoches);

    // tcu cuda fp16


std::vector<torch::Tensor> spmm_forward_fp16_tcu_cuda_cc_only(
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

    torch::Tensor rhs_matrix,

    const int parts_t,
    const int parts_c,
    const int partsize_c,
    const int parts_c_short,
    const int dimN,
    const int mOri,
    const int kOri,
    int epoches)
{
    auto output_matrix = torch::zeros({mOri, dimN}, torch::kFloat32).to(torch::kCPU);
    
    //把CPU端的tensor转成C++的数据结构
    int * t_row_offset_ = t_row_offset.data<int>();
    int * t_blockNew_offset_ = t_blockNew_offset.data<int>();
    half * t_value_ = reinterpret_cast<half *>(t_value.data<at::Half>()); 
    int * t_column_ = t_column.data<int>();
    int * t_window_row_ = t_window_row.data<int>();
    int * t_atomic_ = t_atomic.data<int>();
    long * t_binary_ = t_binary.data<long>();
    
    int * c_row_offsets_ = c_row_offsets.data<int>();
    int * c_row_ = c_row.data<int>();
    int * c_atomic_ = c_atomic.data<int>();
    int * c_column_ = c_column.data<int>();
    half * c_value_ = reinterpret_cast<half *>(c_value.data<at::Half>()); 

    int * c_row_offsets_short_ = c_row_offsets_short.data<int>();
    int * c_row_short_ = c_row_short.data<int>();
    int * c_atomic_short_ = c_atomic_short.data<int>();
    int * c_column_short_ = c_column_short.data<int>();
    half * c_value_short_ = reinterpret_cast<half *>(c_value_short.data<at::Half>()); 

    half * rhs_matrix_ = reinterpret_cast<half *>(rhs_matrix.data<at::Half>()); 
    float * output_matrix_ = output_matrix.data<float>();

    // Device
    int *d_t_row_offset, *d_t_blockNew_offset, *d_t_column;
    int *d_t_window_row, *d_t_atomic;
    long *d_t_binary;
    half *d_t_value; 
    int *d_c_row_offsets, *d_c_row, *d_c_atomic, *d_c_column;
    int *d_c_row_offsets_short, *d_c_row_short, *d_c_atomic_short, *d_c_column_short;
    half *d_c_value, *d_c_value_short; 
	half *d_rhs_matrix;
    float *d_output_matrix;

    checkCuda(cudaMalloc(&d_t_row_offset, (t_row_offset.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_blockNew_offset, (t_blockNew_offset.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_value, (t_value.size(0)) * sizeof(half)));
    checkCuda(cudaMalloc(&d_t_column, (t_column.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_window_row, (t_window_row.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_atomic, (t_atomic.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_binary, (t_binary.size(0)) * sizeof(long)));

    checkCuda(cudaMalloc(&d_c_row_offsets, (c_row_offsets.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_row, (c_row.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_atomic, (c_atomic.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_column, (c_column.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_value, (c_value.size(0)) * sizeof(half)));

    checkCuda(cudaMalloc(&d_c_row_offsets_short, (c_row_offsets_short.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_row_short, (c_row_short.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_atomic_short, (c_atomic_short.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_column_short, (c_column_short.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_value_short, (c_value_short.size(0)) * sizeof(half)));

    checkCuda(cudaMalloc(&d_rhs_matrix, (kOri*dimN) * sizeof(half)));
    checkCuda(cudaMalloc(&d_output_matrix, (mOri*dimN) * sizeof(float)));

    //把CPU数据放到GPU上
    checkCuda(cudaMemcpy(d_t_row_offset, t_row_offset_ , (t_row_offset.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_blockNew_offset, t_blockNew_offset_ , (t_blockNew_offset.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_value, t_value_, (t_value.size(0)) * sizeof(half), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_column, t_column_, (t_column.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_window_row, t_window_row_, (t_window_row.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_atomic, t_atomic_, (t_atomic.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_binary, t_binary_, (t_binary.size(0)) * sizeof(long), cudaMemcpyHostToDevice));

    checkCuda(cudaMemcpy(d_c_row_offsets, c_row_offsets_ , (c_row_offsets.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_row, c_row_, (c_row.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_atomic, c_atomic_, (c_atomic.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_column, c_column_, (c_column.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_value, c_value_, (c_value.size(0)) * sizeof(half), cudaMemcpyHostToDevice));

    checkCuda(cudaMemcpy(d_c_row_offsets_short, c_row_offsets_short_ , (c_row_offsets_short.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_row_short, c_row_short_, (c_row_short.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_atomic_short, c_atomic_short_, (c_atomic_short.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_column_short, c_column_short_, (c_column_short.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_value_short, c_value_short_, (c_value_short.size(0)) * sizeof(half), cudaMemcpyHostToDevice));

    checkCuda(cudaMemcpy(d_rhs_matrix, rhs_matrix_,(kOri*dimN) * sizeof(half), cudaMemcpyHostToDevice));
    float spmm_ms_avg = 0.0;
    
      spmm_ms_avg =  spmm_forward_fp16_tcu_cuda_kernel_cc_only(
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

      d_rhs_matrix,
      d_output_matrix,

      parts_t,
      parts_c,
      partsize_c,
      parts_c_short,
      dimN,
      mOri,
      epoches);
      
    checkCuda(cudaMemcpy(output_matrix_, d_output_matrix, mOri * dimN * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_t_row_offset);
    cudaFree(d_t_blockNew_offset);
    cudaFree(d_t_value);
    cudaFree(d_t_column);
    cudaFree(d_t_window_row);
    cudaFree(d_t_atomic);
    cudaFree(d_t_binary);

    cudaFree(d_c_row_offsets);
    cudaFree(d_c_row);
    cudaFree(d_c_atomic);
    cudaFree(d_c_column);
    cudaFree(d_c_value);

    cudaFree(d_c_row_offsets_short);
    cudaFree(d_c_row_short);
    cudaFree(d_c_atomic_short);
    cudaFree(d_c_column_short);
    cudaFree(d_c_value_short);

    cudaFree(d_rhs_matrix);
    cudaFree(d_output_matrix);
    cudaDeviceSynchronize(); 
    return {output_matrix, torch::tensor(spmm_ms_avg)};
}

float spmm_forward_fp16_tcu_cuda_kernel_seq(
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
    int epoches);

// tcu cuda fp16

std::vector<torch::Tensor> spmm_forward_fp16_tcu_cuda_seq(
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

    torch::Tensor rhs_matrix,

    const int parts_t,
    const int parts_c,
    const int partsize_c,
    const int parts_c_short,
    const int dimN,
    const int mOri,
    const int kOri,
    int epoches)
{
    auto output_matrix = torch::zeros({mOri, dimN}, torch::kFloat32).to(torch::kCPU);
    
    //把CPU端的tensor转成C++的数据结构
    int * t_row_offset_ = t_row_offset.data<int>();
    int * t_blockNew_offset_ = t_blockNew_offset.data<int>();
    half * t_value_ = reinterpret_cast<half *>(t_value.data<at::Half>()); 
    int * t_column_ = t_column.data<int>();
    int * t_window_row_ = t_window_row.data<int>();
    int * t_atomic_ = t_atomic.data<int>();
    long * t_binary_ = t_binary.data<long>();
    
    int * c_row_offsets_ = c_row_offsets.data<int>();
    int * c_row_ = c_row.data<int>();
    int * c_atomic_ = c_atomic.data<int>();
    int * c_column_ = c_column.data<int>();
    half * c_value_ = reinterpret_cast<half *>(c_value.data<at::Half>()); 

    int * c_row_offsets_short_ = c_row_offsets_short.data<int>();
    int * c_row_short_ = c_row_short.data<int>();
    int * c_atomic_short_ = c_atomic_short.data<int>();
    int * c_column_short_ = c_column_short.data<int>();
    half * c_value_short_ = reinterpret_cast<half *>(c_value_short.data<at::Half>()); 

    half * rhs_matrix_ = reinterpret_cast<half *>(rhs_matrix.data<at::Half>()); 
    float * output_matrix_ = output_matrix.data<float>();

    // Device
    int *d_t_row_offset, *d_t_blockNew_offset, *d_t_column;
    int *d_t_window_row, *d_t_atomic;
    long *d_t_binary;
    half *d_t_value; 
    int *d_c_row_offsets, *d_c_row, *d_c_atomic, *d_c_column;
    int *d_c_row_offsets_short, *d_c_row_short, *d_c_atomic_short, *d_c_column_short;
    half *d_c_value, *d_c_value_short; 
	half *d_rhs_matrix;
    float *d_output_matrix;

    checkCuda(cudaMalloc(&d_t_row_offset, (t_row_offset.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_blockNew_offset, (t_blockNew_offset.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_value, (t_value.size(0)) * sizeof(half)));
    checkCuda(cudaMalloc(&d_t_column, (t_column.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_window_row, (t_window_row.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_atomic, (t_atomic.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_binary, (t_binary.size(0)) * sizeof(long)));

    checkCuda(cudaMalloc(&d_c_row_offsets, (c_row_offsets.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_row, (c_row.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_atomic, (c_atomic.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_column, (c_column.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_value, (c_value.size(0)) * sizeof(half)));

    checkCuda(cudaMalloc(&d_c_row_offsets_short, (c_row_offsets_short.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_row_short, (c_row_short.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_atomic_short, (c_atomic_short.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_column_short, (c_column_short.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_value_short, (c_value_short.size(0)) * sizeof(half)));

    checkCuda(cudaMalloc(&d_rhs_matrix, (kOri*dimN) * sizeof(half)));
    checkCuda(cudaMalloc(&d_output_matrix, (mOri*dimN) * sizeof(float)));

    //把CPU数据放到GPU上
    checkCuda(cudaMemcpy(d_t_row_offset, t_row_offset_ , (t_row_offset.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_blockNew_offset, t_blockNew_offset_ , (t_blockNew_offset.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_value, t_value_, (t_value.size(0)) * sizeof(half), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_column, t_column_, (t_column.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_window_row, t_window_row_, (t_window_row.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_atomic, t_atomic_, (t_atomic.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_binary, t_binary_, (t_binary.size(0)) * sizeof(long), cudaMemcpyHostToDevice));

    checkCuda(cudaMemcpy(d_c_row_offsets, c_row_offsets_ , (c_row_offsets.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_row, c_row_, (c_row.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_atomic, c_atomic_, (c_atomic.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_column, c_column_, (c_column.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_value, c_value_, (c_value.size(0)) * sizeof(half), cudaMemcpyHostToDevice));

    checkCuda(cudaMemcpy(d_c_row_offsets_short, c_row_offsets_short_ , (c_row_offsets_short.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_row_short, c_row_short_, (c_row_short.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_atomic_short, c_atomic_short_, (c_atomic_short.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_column_short, c_column_short_, (c_column_short.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_value_short, c_value_short_, (c_value_short.size(0)) * sizeof(half), cudaMemcpyHostToDevice));

    checkCuda(cudaMemcpy(d_rhs_matrix, rhs_matrix_,(kOri*dimN) * sizeof(half), cudaMemcpyHostToDevice));
    float spmm_ms_avg = 0.0;
    
      spmm_ms_avg =  spmm_forward_fp16_tcu_cuda_kernel_seq(
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

      d_rhs_matrix,
      d_output_matrix,

      parts_t,
      parts_c,
      partsize_c,
      parts_c_short,
      dimN,
      mOri,
      epoches);
      
    checkCuda(cudaMemcpy(output_matrix_, d_output_matrix, mOri * dimN * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_t_row_offset);
    cudaFree(d_t_blockNew_offset);
    cudaFree(d_t_value);
    cudaFree(d_t_column);
    cudaFree(d_t_window_row);
    cudaFree(d_t_atomic);
    cudaFree(d_t_binary);

    cudaFree(d_c_row_offsets);
    cudaFree(d_c_row);
    cudaFree(d_c_atomic);
    cudaFree(d_c_column);
    cudaFree(d_c_value);

    cudaFree(d_c_row_offsets_short);
    cudaFree(d_c_row_short);
    cudaFree(d_c_atomic_short);
    cudaFree(d_c_column_short);
    cudaFree(d_c_value_short);

    cudaFree(d_rhs_matrix);
    cudaFree(d_output_matrix);
    cudaDeviceSynchronize(); 
    return {output_matrix, torch::tensor(spmm_ms_avg)};
}

float spmm_forward_fp16_tcu_cuda_hybrid_kernel(
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
    int epoches);


std::vector<torch::Tensor> spmm_forward_fp16_tcu_cuda_hybrid(
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

    torch::Tensor rhs_matrix,

    const int parts_t,
    const int parts_c,
    const int partsize_c,
    const int parts_c_short,
    const int dimN,
    const int mOri,
    const int kOri,
    int epoches)
{
    auto output_matrix = torch::zeros({mOri, dimN}, torch::kFloat32).to(torch::kCPU);
    
    //把CPU端的tensor转成C++的数据结构
    int * t_row_offset_ = t_row_offset.data<int>();
    int * t_blockNew_offset_ = t_blockNew_offset.data<int>();
    half * t_value_ = reinterpret_cast<half *>(t_value.data<at::Half>()); 
    int * t_column_ = t_column.data<int>();
    int * t_window_row_ = t_window_row.data<int>();
    int * t_atomic_ = t_atomic.data<int>();
    long * t_binary_ = t_binary.data<long>();
    
    int * c_row_offsets_ = c_row_offsets.data<int>();
    int * c_row_ = c_row.data<int>();
    int * c_atomic_ = c_atomic.data<int>();
    int * c_column_ = c_column.data<int>();
    half * c_value_ = reinterpret_cast<half *>(c_value.data<at::Half>()); 

    int * c_row_offsets_short_ = c_row_offsets_short.data<int>();
    int * c_row_short_ = c_row_short.data<int>();
    int * c_atomic_short_ = c_atomic_short.data<int>();
    int * c_column_short_ = c_column_short.data<int>();
    half * c_value_short_ = reinterpret_cast<half *>(c_value_short.data<at::Half>()); 

    half * rhs_matrix_ = reinterpret_cast<half *>(rhs_matrix.data<at::Half>()); 
    float * output_matrix_ = output_matrix.data<float>();

    // Device
    int *d_t_row_offset, *d_t_blockNew_offset, *d_t_column;
    int *d_t_window_row, *d_t_atomic;
    long *d_t_binary;
    half *d_t_value; 
    int *d_c_row_offsets, *d_c_row, *d_c_atomic, *d_c_column;
    int *d_c_row_offsets_short, *d_c_row_short, *d_c_atomic_short, *d_c_column_short;
    half *d_c_value, *d_c_value_short; 
	half *d_rhs_matrix;
    float *d_output_matrix;

    checkCuda(cudaMalloc(&d_t_row_offset, (t_row_offset.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_blockNew_offset, (t_blockNew_offset.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_value, (t_value.size(0)) * sizeof(half)));
    checkCuda(cudaMalloc(&d_t_column, (t_column.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_window_row, (t_window_row.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_atomic, (t_atomic.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_binary, (t_binary.size(0)) * sizeof(long)));

    checkCuda(cudaMalloc(&d_c_row_offsets, (c_row_offsets.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_row, (c_row.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_atomic, (c_atomic.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_column, (c_column.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_value, (c_value.size(0)) * sizeof(half)));

    checkCuda(cudaMalloc(&d_c_row_offsets_short, (c_row_offsets_short.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_row_short, (c_row_short.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_atomic_short, (c_atomic_short.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_column_short, (c_column_short.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_value_short, (c_value_short.size(0)) * sizeof(half)));

    checkCuda(cudaMalloc(&d_rhs_matrix, (kOri*dimN) * sizeof(half)));
    checkCuda(cudaMalloc(&d_output_matrix, (mOri*dimN) * sizeof(float)));

    //把CPU数据放到GPU上
    checkCuda(cudaMemcpy(d_t_row_offset, t_row_offset_ , (t_row_offset.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_blockNew_offset, t_blockNew_offset_ , (t_blockNew_offset.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_value, t_value_, (t_value.size(0)) * sizeof(half), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_column, t_column_, (t_column.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_window_row, t_window_row_, (t_window_row.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_atomic, t_atomic_, (t_atomic.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_binary, t_binary_, (t_binary.size(0)) * sizeof(long), cudaMemcpyHostToDevice));

    checkCuda(cudaMemcpy(d_c_row_offsets, c_row_offsets_ , (c_row_offsets.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_row, c_row_, (c_row.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_atomic, c_atomic_, (c_atomic.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_column, c_column_, (c_column.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_value, c_value_, (c_value.size(0)) * sizeof(half), cudaMemcpyHostToDevice));

    checkCuda(cudaMemcpy(d_c_row_offsets_short, c_row_offsets_short_ , (c_row_offsets_short.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_row_short, c_row_short_, (c_row_short.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_atomic_short, c_atomic_short_, (c_atomic_short.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_column_short, c_column_short_, (c_column_short.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_value_short, c_value_short_, (c_value_short.size(0)) * sizeof(half), cudaMemcpyHostToDevice));

    checkCuda(cudaMemcpy(d_rhs_matrix, rhs_matrix_,(kOri*dimN) * sizeof(half), cudaMemcpyHostToDevice));
    float spmm_ms_avg = 0.0;
    
      spmm_ms_avg =  spmm_forward_fp16_tcu_cuda_hybrid_kernel(
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

      d_rhs_matrix,
      d_output_matrix,

      parts_t,
      parts_c,
      partsize_c,
      parts_c_short,
      dimN,
      mOri,
      epoches);
      
    checkCuda(cudaMemcpy(output_matrix_, d_output_matrix, mOri * dimN * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_t_row_offset);
    cudaFree(d_t_blockNew_offset);
    cudaFree(d_t_value);
    cudaFree(d_t_column);
    cudaFree(d_t_window_row);
    cudaFree(d_t_atomic);
    cudaFree(d_t_binary);

    cudaFree(d_c_row_offsets);
    cudaFree(d_c_row);
    cudaFree(d_c_atomic);
    cudaFree(d_c_column);
    cudaFree(d_c_value);

    cudaFree(d_c_row_offsets_short);
    cudaFree(d_c_row_short);
    cudaFree(d_c_atomic_short);
    cudaFree(d_c_column_short);
    cudaFree(d_c_value_short);

    cudaFree(d_rhs_matrix);
    cudaFree(d_output_matrix);
    cudaDeviceSynchronize(); 
    return {output_matrix, torch::tensor(spmm_ms_avg)};
}


float spmm_forward_fp16_tcu_cuda_hybrid_kernel_seq(
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
    int epoches);

// tcu cuda fp16

std::vector<torch::Tensor> spmm_forward_fp16_tcu_cuda_hybrid_seq(
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

    torch::Tensor rhs_matrix,

    const int parts_t,
    const int parts_c,
    const int partsize_c,
    const int parts_c_short,
    const int dimN,
    const int mOri,
    const int kOri,
    int epoches)
{
    auto output_matrix = torch::zeros({mOri, dimN}, torch::kFloat32).to(torch::kCPU);
    
    //把CPU端的tensor转成C++的数据结构
    int * t_row_offset_ = t_row_offset.data<int>();
    int * t_blockNew_offset_ = t_blockNew_offset.data<int>();
    half * t_value_ = reinterpret_cast<half *>(t_value.data<at::Half>()); 
    int * t_column_ = t_column.data<int>();
    int * t_window_row_ = t_window_row.data<int>();
    int * t_atomic_ = t_atomic.data<int>();
    long * t_binary_ = t_binary.data<long>();
    
    int * c_row_offsets_ = c_row_offsets.data<int>();
    int * c_row_ = c_row.data<int>();
    int * c_atomic_ = c_atomic.data<int>();
    int * c_column_ = c_column.data<int>();
    half * c_value_ = reinterpret_cast<half *>(c_value.data<at::Half>()); 

    int * c_row_offsets_short_ = c_row_offsets_short.data<int>();
    int * c_row_short_ = c_row_short.data<int>();
    int * c_atomic_short_ = c_atomic_short.data<int>();
    int * c_column_short_ = c_column_short.data<int>();
    half * c_value_short_ = reinterpret_cast<half *>(c_value_short.data<at::Half>()); 

    half * rhs_matrix_ = reinterpret_cast<half *>(rhs_matrix.data<at::Half>()); 
    float * output_matrix_ = output_matrix.data<float>();

    // Device
    int *d_t_row_offset, *d_t_blockNew_offset, *d_t_column;
    int *d_t_window_row, *d_t_atomic;
    long *d_t_binary;
    half *d_t_value; 
    int *d_c_row_offsets, *d_c_row, *d_c_atomic, *d_c_column;
    int *d_c_row_offsets_short, *d_c_row_short, *d_c_atomic_short, *d_c_column_short;
    half *d_c_value, *d_c_value_short; 
	half *d_rhs_matrix;
    float *d_output_matrix;

    checkCuda(cudaMalloc(&d_t_row_offset, (t_row_offset.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_blockNew_offset, (t_blockNew_offset.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_value, (t_value.size(0)) * sizeof(half)));
    checkCuda(cudaMalloc(&d_t_column, (t_column.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_window_row, (t_window_row.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_atomic, (t_atomic.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_binary, (t_binary.size(0)) * sizeof(long)));

    checkCuda(cudaMalloc(&d_c_row_offsets, (c_row_offsets.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_row, (c_row.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_atomic, (c_atomic.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_column, (c_column.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_value, (c_value.size(0)) * sizeof(half)));

    checkCuda(cudaMalloc(&d_c_row_offsets_short, (c_row_offsets_short.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_row_short, (c_row_short.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_atomic_short, (c_atomic_short.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_column_short, (c_column_short.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_value_short, (c_value_short.size(0)) * sizeof(half)));

    checkCuda(cudaMalloc(&d_rhs_matrix, (kOri*dimN) * sizeof(half)));
    checkCuda(cudaMalloc(&d_output_matrix, (mOri*dimN) * sizeof(float)));

    //把CPU数据放到GPU上
    checkCuda(cudaMemcpy(d_t_row_offset, t_row_offset_ , (t_row_offset.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_blockNew_offset, t_blockNew_offset_ , (t_blockNew_offset.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_value, t_value_, (t_value.size(0)) * sizeof(half), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_column, t_column_, (t_column.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_window_row, t_window_row_, (t_window_row.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_atomic, t_atomic_, (t_atomic.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_binary, t_binary_, (t_binary.size(0)) * sizeof(long), cudaMemcpyHostToDevice));

    checkCuda(cudaMemcpy(d_c_row_offsets, c_row_offsets_ , (c_row_offsets.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_row, c_row_, (c_row.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_atomic, c_atomic_, (c_atomic.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_column, c_column_, (c_column.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_value, c_value_, (c_value.size(0)) * sizeof(half), cudaMemcpyHostToDevice));

    checkCuda(cudaMemcpy(d_c_row_offsets_short, c_row_offsets_short_ , (c_row_offsets_short.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_row_short, c_row_short_, (c_row_short.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_atomic_short, c_atomic_short_, (c_atomic_short.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_column_short, c_column_short_, (c_column_short.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_value_short, c_value_short_, (c_value_short.size(0)) * sizeof(half), cudaMemcpyHostToDevice));

    checkCuda(cudaMemcpy(d_rhs_matrix, rhs_matrix_,(kOri*dimN) * sizeof(half), cudaMemcpyHostToDevice));
    float spmm_ms_avg = 0.0;
    
      spmm_ms_avg =  spmm_forward_fp16_tcu_cuda_hybrid_kernel_seq(
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

      d_rhs_matrix,
      d_output_matrix,

      parts_t,
      parts_c,
      partsize_c,
      parts_c_short,
      dimN,
      mOri,
      epoches);
      
    checkCuda(cudaMemcpy(output_matrix_, d_output_matrix, mOri * dimN * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_t_row_offset);
    cudaFree(d_t_blockNew_offset);
    cudaFree(d_t_value);
    cudaFree(d_t_column);
    cudaFree(d_t_window_row);
    cudaFree(d_t_atomic);
    cudaFree(d_t_binary);

    cudaFree(d_c_row_offsets);
    cudaFree(d_c_row);
    cudaFree(d_c_atomic);
    cudaFree(d_c_column);
    cudaFree(d_c_value);

    cudaFree(d_c_row_offsets_short);
    cudaFree(d_c_row_short);
    cudaFree(d_c_atomic_short);
    cudaFree(d_c_column_short);
    cudaFree(d_c_value_short);

    cudaFree(d_rhs_matrix);
    cudaFree(d_output_matrix);
    cudaDeviceSynchronize(); 
    return {output_matrix, torch::tensor(spmm_ms_avg)};
}

float spmm_forward_fp16_tcu_cuda_bcrs_kernel(
    int * t_row_offset,
    half * t_value, 
    int * t_column, 
    int* t_window_row,
    int * t_atomic,

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
    int epoches);


std::vector<torch::Tensor> spmm_forward_fp16_tcu_cuda_bcrs(
    torch::Tensor t_row_offset,
    torch::Tensor t_column, 
    torch::Tensor t_value, 
    torch::Tensor t_window_row,
    torch::Tensor t_atomic,

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

    torch::Tensor rhs_matrix,

    const int parts_t,
    const int parts_c,
    const int partsize_c,
    const int parts_c_short,
    const int dimN,
    const int mOri,
    const int kOri,
    int epoches)
{
    auto output_matrix = torch::zeros({mOri, dimN}, torch::kFloat32).to(torch::kCPU);
    
    //把CPU端的tensor转成C++的数据结构
    int * t_row_offset_ = t_row_offset.data<int>();
    half * t_value_ = reinterpret_cast<half *>(t_value.data<at::Half>()); 
    int * t_column_ = t_column.data<int>();
    int * t_window_row_ = t_window_row.data<int>();
    int * t_atomic_ = t_atomic.data<int>();
    
    int * c_row_offsets_ = c_row_offsets.data<int>();
    int * c_row_ = c_row.data<int>();
    int * c_atomic_ = c_atomic.data<int>();
    int * c_column_ = c_column.data<int>();
    half * c_value_ = reinterpret_cast<half *>(c_value.data<at::Half>()); 

    int * c_row_offsets_short_ = c_row_offsets_short.data<int>();
    int * c_row_short_ = c_row_short.data<int>();
    int * c_atomic_short_ = c_atomic_short.data<int>();
    int * c_column_short_ = c_column_short.data<int>();
    half * c_value_short_ = reinterpret_cast<half *>(c_value_short.data<at::Half>()); 

    half * rhs_matrix_ = reinterpret_cast<half *>(rhs_matrix.data<at::Half>()); 
    float * output_matrix_ = output_matrix.data<float>();

    // Device
    int *d_t_row_offset, *d_t_column;
    int *d_t_window_row, *d_t_atomic;
    half *d_t_value; 
    int *d_c_row_offsets, *d_c_row, *d_c_atomic, *d_c_column;
    int *d_c_row_offsets_short, *d_c_row_short, *d_c_atomic_short, *d_c_column_short;
    half *d_c_value, *d_c_value_short; 
	half *d_rhs_matrix;
    float *d_output_matrix;

    checkCuda(cudaMalloc(&d_t_row_offset, (t_row_offset.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_value, (t_value.size(0)) * sizeof(half)));
    checkCuda(cudaMalloc(&d_t_column, (t_column.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_window_row, (t_window_row.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_atomic, (t_atomic.size(0)) * sizeof(int)));

    checkCuda(cudaMalloc(&d_c_row_offsets, (c_row_offsets.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_row, (c_row.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_atomic, (c_atomic.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_column, (c_column.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_value, (c_value.size(0)) * sizeof(half)));

    checkCuda(cudaMalloc(&d_c_row_offsets_short, (c_row_offsets_short.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_row_short, (c_row_short.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_atomic_short, (c_atomic_short.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_column_short, (c_column_short.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_value_short, (c_value_short.size(0)) * sizeof(half)));

    checkCuda(cudaMalloc(&d_rhs_matrix, (kOri*dimN) * sizeof(half)));
    checkCuda(cudaMalloc(&d_output_matrix, (mOri*dimN) * sizeof(float)));

    //把CPU数据放到GPU上
    checkCuda(cudaMemcpy(d_t_row_offset, t_row_offset_ , (t_row_offset.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_value, t_value_, (t_value.size(0)) * sizeof(half), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_column, t_column_, (t_column.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_window_row, t_window_row_, (t_window_row.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_atomic, t_atomic_, (t_atomic.size(0)) * sizeof(int), cudaMemcpyHostToDevice));

    checkCuda(cudaMemcpy(d_c_row_offsets, c_row_offsets_ , (c_row_offsets.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_row, c_row_, (c_row.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_atomic, c_atomic_, (c_atomic.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_column, c_column_, (c_column.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_value, c_value_, (c_value.size(0)) * sizeof(half), cudaMemcpyHostToDevice));

    checkCuda(cudaMemcpy(d_c_row_offsets_short, c_row_offsets_short_ , (c_row_offsets_short.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_row_short, c_row_short_, (c_row_short.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_atomic_short, c_atomic_short_, (c_atomic_short.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_column_short, c_column_short_, (c_column_short.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_value_short, c_value_short_, (c_value_short.size(0)) * sizeof(half), cudaMemcpyHostToDevice));

    checkCuda(cudaMemcpy(d_rhs_matrix, rhs_matrix_,(kOri*dimN) * sizeof(half), cudaMemcpyHostToDevice));
    float spmm_ms_avg = 0.0;
    
      spmm_ms_avg =  spmm_forward_fp16_tcu_cuda_bcrs_kernel(
      d_t_row_offset,
      d_t_value, 
      d_t_column,
      d_t_window_row,
      d_t_atomic,
      
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

      d_rhs_matrix,
      d_output_matrix,

      parts_t,
      parts_c,
      partsize_c,
      parts_c_short,
      dimN,
      mOri,
      epoches);
      
    checkCuda(cudaMemcpy(output_matrix_, d_output_matrix, mOri * dimN * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_t_row_offset);
    cudaFree(d_t_value);
    cudaFree(d_t_column);
    cudaFree(d_t_window_row);
    cudaFree(d_t_atomic);

    cudaFree(d_c_row_offsets);
    cudaFree(d_c_row);
    cudaFree(d_c_atomic);
    cudaFree(d_c_column);
    cudaFree(d_c_value);

    cudaFree(d_c_row_offsets_short);
    cudaFree(d_c_row_short);
    cudaFree(d_c_atomic_short);
    cudaFree(d_c_column_short);
    cudaFree(d_c_value_short);

    cudaFree(d_rhs_matrix);
    cudaFree(d_output_matrix);
    cudaDeviceSynchronize(); 
    return {output_matrix, torch::tensor(spmm_ms_avg)};
}



float spmm_forward_fp16_tcu_cuda_bcrs_kernel_seq(
    int * t_row_offset,
    half * t_value, 
    int * t_column, 
    int* t_window_row,
    int * t_atomic,

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
    int epoches);


std::vector<torch::Tensor> spmm_forward_fp16_tcu_cuda_bcrs_seq(
    torch::Tensor t_row_offset,
    torch::Tensor t_column, 
    torch::Tensor t_value, 
    torch::Tensor t_window_row,
    torch::Tensor t_atomic,

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

    torch::Tensor rhs_matrix,

    const int parts_t,
    const int parts_c,
    const int partsize_c,
    const int parts_c_short,
    const int dimN,
    const int mOri,
    const int kOri,
    int epoches)
{
    auto output_matrix = torch::zeros({mOri, dimN}, torch::kFloat32).to(torch::kCPU);
    
    //把CPU端的tensor转成C++的数据结构
    int * t_row_offset_ = t_row_offset.data<int>();
    half * t_value_ = reinterpret_cast<half *>(t_value.data<at::Half>()); 
    int * t_column_ = t_column.data<int>();
    int * t_window_row_ = t_window_row.data<int>();
    int * t_atomic_ = t_atomic.data<int>();
    
    int * c_row_offsets_ = c_row_offsets.data<int>();
    int * c_row_ = c_row.data<int>();
    int * c_atomic_ = c_atomic.data<int>();
    int * c_column_ = c_column.data<int>();
    half * c_value_ = reinterpret_cast<half *>(c_value.data<at::Half>()); 

    int * c_row_offsets_short_ = c_row_offsets_short.data<int>();
    int * c_row_short_ = c_row_short.data<int>();
    int * c_atomic_short_ = c_atomic_short.data<int>();
    int * c_column_short_ = c_column_short.data<int>();
    half * c_value_short_ = reinterpret_cast<half *>(c_value_short.data<at::Half>()); 

    half * rhs_matrix_ = reinterpret_cast<half *>(rhs_matrix.data<at::Half>()); 
    float * output_matrix_ = output_matrix.data<float>();

    // Device
    int *d_t_row_offset, *d_t_column;
    int *d_t_window_row, *d_t_atomic;
    half *d_t_value; 
    int *d_c_row_offsets, *d_c_row, *d_c_atomic, *d_c_column;
    int *d_c_row_offsets_short, *d_c_row_short, *d_c_atomic_short, *d_c_column_short;
    half *d_c_value, *d_c_value_short; 
	half *d_rhs_matrix;
    float *d_output_matrix;

    checkCuda(cudaMalloc(&d_t_row_offset, (t_row_offset.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_value, (t_value.size(0)) * sizeof(half)));
    checkCuda(cudaMalloc(&d_t_column, (t_column.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_window_row, (t_window_row.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_atomic, (t_atomic.size(0)) * sizeof(int)));

    checkCuda(cudaMalloc(&d_c_row_offsets, (c_row_offsets.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_row, (c_row.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_atomic, (c_atomic.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_column, (c_column.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_value, (c_value.size(0)) * sizeof(half)));

    checkCuda(cudaMalloc(&d_c_row_offsets_short, (c_row_offsets_short.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_row_short, (c_row_short.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_atomic_short, (c_atomic_short.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_column_short, (c_column_short.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_value_short, (c_value_short.size(0)) * sizeof(half)));

    checkCuda(cudaMalloc(&d_rhs_matrix, (kOri*dimN) * sizeof(half)));
    checkCuda(cudaMalloc(&d_output_matrix, (mOri*dimN) * sizeof(float)));

    //把CPU数据放到GPU上
    checkCuda(cudaMemcpy(d_t_row_offset, t_row_offset_ , (t_row_offset.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_value, t_value_, (t_value.size(0)) * sizeof(half), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_column, t_column_, (t_column.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_window_row, t_window_row_, (t_window_row.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_atomic, t_atomic_, (t_atomic.size(0)) * sizeof(int), cudaMemcpyHostToDevice));

    checkCuda(cudaMemcpy(d_c_row_offsets, c_row_offsets_ , (c_row_offsets.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_row, c_row_, (c_row.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_atomic, c_atomic_, (c_atomic.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_column, c_column_, (c_column.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_value, c_value_, (c_value.size(0)) * sizeof(half), cudaMemcpyHostToDevice));

    checkCuda(cudaMemcpy(d_c_row_offsets_short, c_row_offsets_short_ , (c_row_offsets_short.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_row_short, c_row_short_, (c_row_short.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_atomic_short, c_atomic_short_, (c_atomic_short.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_column_short, c_column_short_, (c_column_short.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_value_short, c_value_short_, (c_value_short.size(0)) * sizeof(half), cudaMemcpyHostToDevice));

    checkCuda(cudaMemcpy(d_rhs_matrix, rhs_matrix_,(kOri*dimN) * sizeof(half), cudaMemcpyHostToDevice));
    float spmm_ms_avg = 0.0;
    
      spmm_ms_avg =  spmm_forward_fp16_tcu_cuda_bcrs_kernel_seq(
      d_t_row_offset,
      d_t_value, 
      d_t_column,
      d_t_window_row,
      d_t_atomic,
      
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

      d_rhs_matrix,
      d_output_matrix,

      parts_t,
      parts_c,
      partsize_c,
      parts_c_short,
      dimN,
      mOri,
      epoches);
      
    checkCuda(cudaMemcpy(output_matrix_, d_output_matrix, mOri * dimN * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_t_row_offset);
    cudaFree(d_t_value);
    cudaFree(d_t_column);
    cudaFree(d_t_window_row);
    cudaFree(d_t_atomic);

    cudaFree(d_c_row_offsets);
    cudaFree(d_c_row);
    cudaFree(d_c_atomic);
    cudaFree(d_c_column);
    cudaFree(d_c_value);

    cudaFree(d_c_row_offsets_short);
    cudaFree(d_c_row_short);
    cudaFree(d_c_atomic_short);
    cudaFree(d_c_column_short);
    cudaFree(d_c_value_short);

    cudaFree(d_rhs_matrix);
    cudaFree(d_output_matrix);
    cudaDeviceSynchronize(); 
    return {output_matrix, torch::tensor(spmm_ms_avg)};
}

/*
TF32
*/
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

    float * rhs_matrix,
    float * output_matrix,
    int partsize,
    const int dimN,
    const int mOri,
    int epoches,
    int parts,
    int parts_short,
    bool swizzle);

std::vector<torch::Tensor> spmm_forward_tf32_cuda_v2(
    
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

    torch::Tensor rhs_matrix,

    int partsize,
    const int dimN,
    const int mOri,
    const int kOri,
    int epoches,
    int parts,
    int parts_short,
    bool swizzle)
{
    // int dimM=dimM1/16;
    auto output_matrix = torch::zeros({mOri, dimN}, torch::kFloat32).to(torch::kCPU);
    
    //把CPU端的tensor转成C++的数据结构
    int * c_row_offsets_ = c_row_offsets.data<int>();
    int * c_row_ = c_row.data<int>();
    int * c_atomic_ = c_atomic.data<int>();
    int * c_column_ = c_column.data<int>();
    float * c_value_ = c_value.data<float>(); 

    int * c_row_offsets_short_ = c_row_offsets_short.data<int>();
    int * c_row_short_ = c_row_short.data<int>();
    int * c_atomic_short_ = c_atomic_short.data<int>();
    int * c_column_short_ = c_column_short.data<int>();
    float * c_value_short_ = c_value_short.data<float>(); 

    float * rhs_matrix_ = rhs_matrix.data<float>(); 
    float * output_matrix_ = output_matrix.data<float>();

    // Device
    int *d_c_row_offsets, *d_c_row, *d_c_atomic, *d_c_column;
    int *d_c_row_offsets_short, *d_c_row_short, *d_c_atomic_short, *d_c_column_short;
    float *d_c_value, *d_c_value_short; 
	float *d_rhs_matrix;
    float *d_output_matrix;

    checkCuda(cudaMalloc(&d_c_row_offsets, (c_row_offsets.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_row, (c_row.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_atomic, (c_atomic.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_column, (c_column.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_value, (c_value.size(0)) * sizeof(float)));

    checkCuda(cudaMalloc(&d_c_row_offsets_short, (c_row_offsets_short.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_row_short, (c_row_short.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_atomic_short, (c_atomic_short.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_column_short, (c_column_short.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_value_short, (c_value_short.size(0)) * sizeof(float)));

    checkCuda(cudaMalloc(&d_rhs_matrix, (kOri*dimN) * sizeof(float)));
    checkCuda(cudaMalloc(&d_output_matrix, (mOri*dimN) * sizeof(float)));

    //把CPU数据放到GPU上
    checkCuda(cudaMemcpy(d_c_row_offsets, c_row_offsets_ , (c_row_offsets.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_row, c_row_, (c_row.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_atomic, c_atomic_, (c_atomic.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_column, c_column_, (c_column.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_value, c_value_, (c_value.size(0)) * sizeof(float), cudaMemcpyHostToDevice));

    checkCuda(cudaMemcpy(d_c_row_offsets_short, c_row_offsets_short_ , (c_row_offsets_short.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_row_short, c_row_short_, (c_row_short.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_atomic_short, c_atomic_short_, (c_atomic_short.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_column_short, c_column_short_, (c_column_short.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_value_short, c_value_short_, (c_value_short.size(0)) * sizeof(float), cudaMemcpyHostToDevice));
    
    checkCuda(cudaMemcpy(d_rhs_matrix, rhs_matrix_,(kOri*dimN) * sizeof(float), cudaMemcpyHostToDevice));

    float spmm_ms_avg = 0.0;

    spmm_ms_avg = spmm_forward_tf32_cuda_kernel_split(

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

    d_rhs_matrix,
    d_output_matrix,

    partsize,
    dimN,
    mOri,
    epoches,
    parts,
    parts_short,
    swizzle); 
     
    checkCuda(cudaMemcpy(output_matrix_, d_output_matrix, mOri * dimN * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_c_row_offsets);
    cudaFree(d_c_row);
    cudaFree(d_c_atomic);
    cudaFree(d_c_column);
    cudaFree(d_c_value);

    cudaFree(d_c_row_offsets_short);
    cudaFree(d_c_row_short);
    cudaFree(d_c_atomic_short);
    cudaFree(d_c_column_short);
    cudaFree(d_c_value_short);

    cudaFree(d_rhs_matrix);
    cudaFree(d_output_matrix);
    cudaDeviceSynchronize(); 
    return {output_matrix, torch::tensor(spmm_ms_avg)};
}


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
    int epoches);

std::vector<torch::Tensor> spmm_forward_tf32_tcu_cuda(
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

    torch::Tensor rhs_matrix,

    const int parts_t,
    const int parts_c,
    const int partsize_c,
    const int parts_c_short,
    const int dimN,
    const int mOri,
    const int kOri,
    int epoches)
{
    auto output_matrix = torch::zeros({mOri, dimN}, torch::kFloat32).to(torch::kCPU);
    
    //把CPU端的tensor转成C++的数据结构
    int * t_row_offset_ = t_row_offset.data<int>();
    int * t_blockNew_offset_ = t_blockNew_offset.data<int>();
    float * t_value_ = t_value.data<float>(); 
    int * t_column_ = t_column.data<int>();
    int * t_window_row_ = t_window_row.data<int>();
    int * t_atomic_ = t_atomic.data<int>();
    int * t_binary_ = t_binary.data<int>();
    
    int * c_row_offsets_ = c_row_offsets.data<int>();
    int * c_row_ = c_row.data<int>();
    int * c_atomic_ = c_atomic.data<int>();
    int * c_column_ = c_column.data<int>();
    float * c_value_ = c_value.data<float>(); 

    int * c_row_offsets_short_ = c_row_offsets_short.data<int>();
    int * c_row_short_ = c_row_short.data<int>();
    int * c_atomic_short_ = c_atomic_short.data<int>();
    int * c_column_short_ = c_column_short.data<int>();
    float * c_value_short_ = c_value_short.data<float>(); 

    float * rhs_matrix_ = rhs_matrix.data<float>(); 
    float * output_matrix_ = output_matrix.data<float>();

    // Device
    int *d_t_row_offset, *d_t_blockNew_offset, *d_t_column;
    int *d_t_window_row, *d_t_atomic;
    int *d_t_binary;
    float *d_t_value; 
    int *d_c_row_offsets, *d_c_row, *d_c_atomic, *d_c_column;
    int *d_c_row_offsets_short, *d_c_row_short, *d_c_atomic_short, *d_c_column_short;
    float *d_c_value, *d_c_value_short; 
	float *d_rhs_matrix;
    float *d_output_matrix;

    checkCuda(cudaMalloc(&d_t_row_offset, (t_row_offset.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_blockNew_offset, (t_blockNew_offset.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_value, (t_value.size(0)) * sizeof(float)));
    checkCuda(cudaMalloc(&d_t_column, (t_column.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_window_row, (t_window_row.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_atomic, (t_atomic.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_binary, (t_binary.size(0)) * sizeof(int)));

    checkCuda(cudaMalloc(&d_c_row_offsets, (c_row_offsets.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_row, (c_row.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_atomic, (c_atomic.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_column, (c_column.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_value, (c_value.size(0)) * sizeof(float)));

    checkCuda(cudaMalloc(&d_c_row_offsets_short, (c_row_offsets_short.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_row_short, (c_row_short.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_atomic_short, (c_atomic_short.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_column_short, (c_column_short.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_value_short, (c_value_short.size(0)) * sizeof(float)));

    checkCuda(cudaMalloc(&d_rhs_matrix, (kOri*dimN) * sizeof(float)));
    checkCuda(cudaMalloc(&d_output_matrix, (mOri*dimN) * sizeof(float)));

    //把CPU数据放到GPU上
    checkCuda(cudaMemcpy(d_t_row_offset, t_row_offset_ , (t_row_offset.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_blockNew_offset, t_blockNew_offset_ , (t_blockNew_offset.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_value, t_value_, (t_value.size(0)) * sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_column, t_column_, (t_column.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_window_row, t_window_row_, (t_window_row.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_atomic, t_atomic_, (t_atomic.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_binary, t_binary_, (t_binary.size(0)) * sizeof(int), cudaMemcpyHostToDevice));

    checkCuda(cudaMemcpy(d_c_row_offsets, c_row_offsets_ , (c_row_offsets.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_row, c_row_, (c_row.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_atomic, c_atomic_, (c_atomic.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_column, c_column_, (c_column.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_value, c_value_, (c_value.size(0)) * sizeof(float), cudaMemcpyHostToDevice));

    checkCuda(cudaMemcpy(d_c_row_offsets_short, c_row_offsets_short_ , (c_row_offsets_short.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_row_short, c_row_short_, (c_row_short.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_atomic_short, c_atomic_short_, (c_atomic_short.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_column_short, c_column_short_, (c_column_short.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_value_short, c_value_short_, (c_value_short.size(0)) * sizeof(float), cudaMemcpyHostToDevice));

    checkCuda(cudaMemcpy(d_rhs_matrix, rhs_matrix_,(kOri*dimN) * sizeof(float), cudaMemcpyHostToDevice));
    float spmm_ms_avg = 0.0;

    spmm_ms_avg =  spmm_forward_tf32_tcu_cuda_kernel(
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

    d_rhs_matrix,
    d_output_matrix,

    parts_t,
    parts_c,
    partsize_c,
    parts_c_short,
    dimN,
    mOri,
    epoches);

    checkCuda(cudaMemcpy(output_matrix_, d_output_matrix, mOri * dimN * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_t_row_offset);
    cudaFree(d_t_blockNew_offset);
    cudaFree(d_t_value);
    cudaFree(d_t_column);
    cudaFree(d_t_window_row);
    cudaFree(d_t_atomic);
    cudaFree(d_t_binary);

    cudaFree(d_c_row_offsets);
    cudaFree(d_c_row);
    cudaFree(d_c_atomic);
    cudaFree(d_c_column);
    cudaFree(d_c_value);

    cudaFree(d_c_row_offsets_short);
    cudaFree(d_c_row_short);
    cudaFree(d_c_atomic_short);
    cudaFree(d_c_column_short);
    cudaFree(d_c_value_short);

    cudaFree(d_rhs_matrix);
    cudaFree(d_output_matrix);
    // cudaDeviceSynchronize(); 
    return {output_matrix, torch::tensor(spmm_ms_avg)};
}



float spmm_forward_tf32_tcu_cuda_kernel_seq(
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
    int epoches);

std::vector<torch::Tensor> spmm_forward_tf32_tcu_cuda_seq(
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

    torch::Tensor rhs_matrix,

    const int parts_t,
    const int parts_c,
    const int partsize_c,
    const int parts_c_short,
    const int dimN,
    const int mOri,
    const int kOri,
    int epoches)
{
    auto output_matrix = torch::zeros({mOri, dimN}, torch::kFloat32).to(torch::kCPU);
    
    //把CPU端的tensor转成C++的数据结构
    int * t_row_offset_ = t_row_offset.data<int>();
    int * t_blockNew_offset_ = t_blockNew_offset.data<int>();
    float * t_value_ = t_value.data<float>(); 
    int * t_column_ = t_column.data<int>();
    int * t_window_row_ = t_window_row.data<int>();
    int * t_atomic_ = t_atomic.data<int>();
    int * t_binary_ = t_binary.data<int>();
    
    int * c_row_offsets_ = c_row_offsets.data<int>();
    int * c_row_ = c_row.data<int>();
    int * c_atomic_ = c_atomic.data<int>();
    int * c_column_ = c_column.data<int>();
    float * c_value_ = c_value.data<float>(); 

    int * c_row_offsets_short_ = c_row_offsets_short.data<int>();
    int * c_row_short_ = c_row_short.data<int>();
    int * c_atomic_short_ = c_atomic_short.data<int>();
    int * c_column_short_ = c_column_short.data<int>();
    float * c_value_short_ = c_value_short.data<float>(); 

    float * rhs_matrix_ = rhs_matrix.data<float>(); 
    float * output_matrix_ = output_matrix.data<float>();

    // Device
    int *d_t_row_offset, *d_t_blockNew_offset, *d_t_column;
    int *d_t_window_row, *d_t_atomic;
    int *d_t_binary;
    float *d_t_value; 
    int *d_c_row_offsets, *d_c_row, *d_c_atomic, *d_c_column;
    int *d_c_row_offsets_short, *d_c_row_short, *d_c_atomic_short, *d_c_column_short;
    float *d_c_value, *d_c_value_short; 
	float *d_rhs_matrix;
    float *d_output_matrix;

    checkCuda(cudaMalloc(&d_t_row_offset, (t_row_offset.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_blockNew_offset, (t_blockNew_offset.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_value, (t_value.size(0)) * sizeof(float)));
    checkCuda(cudaMalloc(&d_t_column, (t_column.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_window_row, (t_window_row.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_atomic, (t_atomic.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_binary, (t_binary.size(0)) * sizeof(int)));

    checkCuda(cudaMalloc(&d_c_row_offsets, (c_row_offsets.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_row, (c_row.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_atomic, (c_atomic.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_column, (c_column.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_value, (c_value.size(0)) * sizeof(float)));

    checkCuda(cudaMalloc(&d_c_row_offsets_short, (c_row_offsets_short.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_row_short, (c_row_short.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_atomic_short, (c_atomic_short.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_column_short, (c_column_short.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_value_short, (c_value_short.size(0)) * sizeof(float)));

    checkCuda(cudaMalloc(&d_rhs_matrix, (kOri*dimN) * sizeof(float)));
    checkCuda(cudaMalloc(&d_output_matrix, (mOri*dimN) * sizeof(float)));

    //把CPU数据放到GPU上
    checkCuda(cudaMemcpy(d_t_row_offset, t_row_offset_ , (t_row_offset.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_blockNew_offset, t_blockNew_offset_ , (t_blockNew_offset.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_value, t_value_, (t_value.size(0)) * sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_column, t_column_, (t_column.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_window_row, t_window_row_, (t_window_row.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_atomic, t_atomic_, (t_atomic.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_binary, t_binary_, (t_binary.size(0)) * sizeof(int), cudaMemcpyHostToDevice));

    checkCuda(cudaMemcpy(d_c_row_offsets, c_row_offsets_ , (c_row_offsets.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_row, c_row_, (c_row.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_atomic, c_atomic_, (c_atomic.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_column, c_column_, (c_column.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_value, c_value_, (c_value.size(0)) * sizeof(float), cudaMemcpyHostToDevice));

    checkCuda(cudaMemcpy(d_c_row_offsets_short, c_row_offsets_short_ , (c_row_offsets_short.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_row_short, c_row_short_, (c_row_short.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_atomic_short, c_atomic_short_, (c_atomic_short.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_column_short, c_column_short_, (c_column_short.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_value_short, c_value_short_, (c_value_short.size(0)) * sizeof(float), cudaMemcpyHostToDevice));

    checkCuda(cudaMemcpy(d_rhs_matrix, rhs_matrix_,(kOri*dimN) * sizeof(float), cudaMemcpyHostToDevice));
    float spmm_ms_avg = 0.0;

    spmm_ms_avg =  spmm_forward_tf32_tcu_cuda_kernel_seq(
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

    d_rhs_matrix,
    d_output_matrix,

    parts_t,
    parts_c,
    partsize_c,
    parts_c_short,
    dimN,
    mOri,
    epoches);

    checkCuda(cudaMemcpy(output_matrix_, d_output_matrix, mOri * dimN * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_t_row_offset);
    cudaFree(d_t_blockNew_offset);
    cudaFree(d_t_value);
    cudaFree(d_t_column);
    cudaFree(d_t_window_row);
    cudaFree(d_t_atomic);
    cudaFree(d_t_binary);

    cudaFree(d_c_row_offsets);
    cudaFree(d_c_row);
    cudaFree(d_c_atomic);
    cudaFree(d_c_column);
    cudaFree(d_c_value);

    cudaFree(d_c_row_offsets_short);
    cudaFree(d_c_row_short);
    cudaFree(d_c_atomic_short);
    cudaFree(d_c_column_short);
    cudaFree(d_c_value_short);

    cudaFree(d_rhs_matrix);
    cudaFree(d_output_matrix);
    // cudaDeviceSynchronize(); 
    return {output_matrix, torch::tensor(spmm_ms_avg)};
}

//block
struct Value_tcu_bcrs {
    std::vector<at::Half> t_value;
    std::vector<int> t_column;
    std::vector<int> row;
    //记录window中每个block中的非零元个数
    std::vector<int> t_block_offset;
    bool tcu_flag = false;
    //记录每个元素在window内的行和列
    std::vector<int> t_row;
    std::vector<int> t_col;
    std::vector<long> t_binary;
    std::vector<int> window;
    std::vector<int> atomic;
    std::vector<int> atomic_seq;
};

struct Value_tcu_tcf {
    std::vector<float> t_value;
    std::vector<int> t_column;
    std::vector<int> row;
    //记录window中每个block中的非零元个数
    std::vector<int> t_block_offset;
    bool tcu_flag = false;
    //记录每个元素在window内的行和列
    std::vector<int> tcl;
    std::vector<int> t_col;
    std::vector<long> t_binary;
    std::vector<int> window;
    std::vector<int> atomic;
};

struct Value_cuda {
    int c_row_offset;
    int c_row;
    int c_atomic;
    int c_atomic_seq;
    std::vector<int> c_column;
    std::vector<float> c_value;
    bool cuda_flag = false;
    bool cuda_residue_flag = false;
};

// 初始化vector
std::vector<std::vector<float>> createVectorOfVectors_float(int numVectors) {
    std::vector<std::vector<float>> vectorOfVectors;

    // 在向量数组中放入指定数量的空向量
    for (int i = 0; i < numVectors; ++i) {
        std::vector<float> emptyVector;
        vectorOfVectors.push_back(emptyVector);
    }

    return vectorOfVectors;
}
// 初始化vector
std::vector<long> createVectorOfVectors_v2(int numVectors) {
    std::vector<long> vectorOfVectors;

    // 在向量数组中放入指定数量的空向量
    for (int i = 0; i < numVectors; ++i) {
        vectorOfVectors.push_back(0);
    }

    return vectorOfVectors;
}
// 合并vector
std::vector<float> mergeVectors_float(const std::vector<std::vector<float>>& vectorOfVectors) {
    std::vector<float> mergedVector;

    // 遍历向量数组中的每个向量，并将其元素添加到合并后的向量中
    for (const auto& vec : vectorOfVectors) {
        for (float num : vec) {
            mergedVector.push_back(num);
        }
    }

    return mergedVector;
}
//统计vector中每个子vector中的非零元数
std::vector<int> countVectors(const std::vector<std::vector<float>>& vectorOfVectors) {
    std::vector<int> res;
    // 在向量数组中放入指定数量的空向量
    for (const auto& vec : vectorOfVectors) {
        res.push_back(vec.size());
    }
    return res;
}
// m-tcu + cuda + partSize + binary
std::vector<torch::Tensor> block_fp16_tcu_cuda_bcrs_seq(torch::Tensor row1, torch::Tensor column1, torch::Tensor values1, 
    int partSize_t, int partSize_c, int shortSize, int density, int window, int wide)
    {
        auto *row=row1.data_ptr<int>();
        auto column=column1.accessor<int, 1>();
        auto values=values1.accessor<float, 1>();
    
        auto start = std::chrono::high_resolution_clock::now();
    
        int rows=row1.size(0)-1;
        // Result
        std::map<int, Value_tcu_bcrs> res_t;
        std::map<int, std::vector<Value_cuda>> res_c;
        std::map<int, std::vector<Value_cuda>> res_c_residue;
        //需要的window数量
        int num_windows = rows/window;
        #pragma omp parallel for
        for(int i=0;i<num_windows;i++)
        {
            /*
                1. 统计当前window中的vector, 并按照稠密度降序排列
            */
            //对column进行合并
            std::vector<int> mergedVector(&column[row[i*window]], &column[row[i*window+window]]);
            // 使用 unordered_map 统计 vector 中每个元素出现次数
            std::unordered_map<int, int> elementCounts;
            for (const auto& element : mergedVector) elementCounts[element]++;
            // 将统计结果放入 vector<pair> 中
            std::vector<std::pair<int, int>> countVector(elementCounts.begin(), elementCounts.end());
            // 按元素个数降序排序
            std::sort(countVector.begin(), countVector.end(), [](const auto& a, const auto& b) {
                return a.second > b.second;
            });
    
            /*
                2. 确定需要进行tcu计算的vector数量, t_percen
            */
            int size = countVector.size();
            int residue = size%wide;
            //统计每个block中的非零元数量，计算稠密度
            int t_percen = 0; //确定前多少个vector会被用作tcu进行稠密计算
            t_percen = std::count_if(countVector.begin(), countVector.end(), [density](const auto& pair) {
                return pair.second >= density;
            });
            int t_percen_ori = t_percen;
            if((t_percen%wide) != 0)
            {
                // if((t_percen + (wide - t_percen%wide)) <= size) 
                t_percen = (t_percen + (wide - t_percen%wide));
                // else
                // t_percen = (t_percen - (t_percen%wide));
    
            }
            // if(t_percen<=size){
            // // 将前 k 个数按升序排列
            // std::sort(countVector.begin(), countVector.begin() + t_percen);
            // // 将剩下的数按升序排列
            // std::sort(countVector.begin() + t_percen, countVector.end());
            // }else{
            //     std::sort(countVector.begin(), countVector.end());
            // }
            std::sort(countVector.begin(), countVector.begin()+t_percen_ori);
            std::sort(countVector.begin() + t_percen_ori, countVector.end());
            /*
                3. 根据t_percen分别划分TCU和CUDA的负载
            */
    
            // TCU 划分
            // 3.1 根据partSize_t划分t_part
            //当前window中有几个block
            int blocks = t_percen/wide; 
            Value_tcu_bcrs v_tcu;
            if(blocks>0) {v_tcu.tcu_flag = true; }
            std::map<int, int> colmap;
            int c=0;
            for(auto col:countVector){
                if (c<t_percen_ori)
                {
                    v_tcu.t_column.push_back(col.first);
                }
                colmap[col.first]=c++;
            }
            if((t_percen_ori%wide)>0){
                for(int tt=0;tt<(wide-(t_percen_ori%wide));tt++)
                v_tcu.t_column.push_back(-1);
            }
            //存放拆分的cuda长短行数据
            std::vector<Value_cuda> large;
            std::vector<Value_cuda> small;
    
            // std::vector<std::vector<float>> t_value_temp = createVectorOfVectors_float(blocks);
            // std::vector<long> t_binary_temp = createVectorOfVectors_v2(blocks);
            std::vector<at::Half> demo(t_percen*window);
            // std::vector<std::vector<int>> t_row_temp = createVectorOfVectors(blocks);
            // std::vector<std::vector<int>> t_col_temp = createVectorOfVectors(blocks);
            // 指示快内的行偏移
            int p=0;
            for(int j=i*window;j<(i+1)*window;j++)
            {
                //用来存储每行的cuda数据
                std::queue<int> c_col;
                std::queue<int> c_value;
                int cuda_nnz=0;
                for(int m=row[j];m<row[j+1];m++)
                {
                    int newCol = colmap[column[m]];
                    if(newCol<t_percen_ori)
                    {
                        int block = newCol/wide;
                        int bInId= newCol%wide;
                        demo[block*window*wide + p*wide +bInId]=values[m];
                    }else{
                        //划分cuda部分
                        c_col.push(column[m]);
                        c_value.push(values[m]);
                        cuda_nnz++;
                    }
                }
                //处理当前行剩余的cuda元素
                if(cuda_nnz>0)
                {
                    if(cuda_nnz <= shortSize)
                    {
                        Value_cuda temp;
                        temp.cuda_residue_flag = true;
                        temp.c_row_offset = cuda_nnz;
                        temp.c_row = i*window + p;
                        if(t_percen_ori>0) 
                        {
                            temp.c_atomic = 1;
                            temp.c_atomic_seq = 0;
                        }
                        else{
                             temp.c_atomic = 0;
                             temp.c_atomic_seq = 0;
                        }
                        for(int m=0; m<cuda_nnz; m++)
                        {
                            temp.c_column.push_back(c_col.front());
                            temp.c_value.push_back(c_value.front());
                            c_col.pop();
                            c_value.pop();
                        }
                        small.push_back(temp);
                    }else{
                        if(cuda_nnz <= partSize_c)
                        {
                            Value_cuda temp;
                            temp.cuda_flag = true;
                            temp.c_row_offset = (cuda_nnz);
                            temp.c_row = i*window + p;
                            if(t_percen_ori>0) 
                            {
                                temp.c_atomic = 1;
                                temp.c_atomic_seq = 0;
                            }
                            else{
                                temp.c_atomic = 0;
                                temp.c_atomic_seq = 0;
                            }
                            for(int m=0; m<cuda_nnz; m++)
                            {
                                temp.c_column.push_back(c_col.front());
                                temp.c_value.push_back(c_value.front());
                                c_col.pop();
                                c_value.pop();
                            }
                            large.push_back(temp);
                        }else{
                            // 当前行的非零元被分成多个part，并需要原子操作
                            int parts_row = cuda_nnz/partSize_c;
                            // int cur_value_indice = row[i];
                            for(int cur_i = 0; cur_i<parts_row; cur_i++)
                            {
                                Value_cuda temp;
                                temp.cuda_flag = true;      
                                temp.c_row_offset = partSize_c;
                                temp.c_row = i*window + p;
                                temp.c_atomic = 1;
                                temp.c_atomic_seq = 1;
                                for(int m=0; m<partSize_c; m++)
                                {
                                    temp.c_column.push_back(c_col.front());
                                    temp.c_value.push_back(c_value.front());
                                    c_col.pop();
                                    c_value.pop();
                                    // cur_value_indice++;
                                }
                                large.push_back(temp);
                            }
                            int residue  = (cuda_nnz%partSize_c);
                            if(residue>0){
                                Value_cuda temp;
                                temp.c_row_offset = residue;
                                temp.c_row = i*window + p;
                                temp.c_atomic = 1;
                                temp.c_atomic_seq = 1;
                                for(int m=0; m<residue; m++)
                                {
                                    temp.c_column.push_back(c_col.front());
                                    temp.c_value.push_back(c_value.front());
                                    c_col.pop();
                                    c_value.pop();
                                    // cur_value_indice++;
                                }
                                if(residue <= shortSize)
                                {
                                    temp.cuda_residue_flag = true;
                                    small.push_back(temp);
                                }else{
                                    temp.cuda_flag = true;
                                    large.push_back(temp);
                                }
                            }
                        }
                    }
                }
                p++;
            }
            //整合vector
            v_tcu.t_value = demo;
            // v_tcu.t_row = mergeVectors(t_row_temp);
            // v_tcu.t_col = mergeVectors(t_col_temp);
        
            if(blocks <= partSize_t)
            {
                v_tcu.row.push_back(t_percen_ori);
                v_tcu.row.push_back(t_percen-t_percen_ori);
                if(t_percen_ori==size) 
                {
                    v_tcu.atomic.push_back(0);
                    v_tcu.atomic_seq.push_back(0);
                }
                else {
                    v_tcu.atomic.push_back(1);
                    v_tcu.atomic_seq.push_back(0);
                }
                v_tcu.window.push_back(i);
            }else{
                // 对block太多的块进行分割
                int part_number = blocks / partSize_t;
                int block_residue = blocks % partSize_t;
                if(block_residue > 0)
                {
                    for(int j=0; j<part_number; j++)
                    {
                        v_tcu.row.push_back(partSize_t*wide);
                        v_tcu.row.push_back(0);
                        v_tcu.window.push_back(i);
                        v_tcu.atomic.push_back(1);
                        v_tcu.atomic_seq.push_back(1);
                    }
                    v_tcu.row.push_back(block_residue*wide - (t_percen-t_percen_ori));
                    v_tcu.row.push_back(t_percen-t_percen_ori);
                    v_tcu.window.push_back(i);
                    v_tcu.atomic.push_back(1);
                    v_tcu.atomic_seq.push_back(1);
                }else{
                    for(int j=0; j<part_number; j++)
                    {
                        //如果是最后一组block
                        if(j == (part_number-1)){
                            v_tcu.row.push_back((partSize_t*wide) - (t_percen-t_percen_ori));
                            v_tcu.row.push_back(t_percen-t_percen_ori);
                        }else{
                            v_tcu.row.push_back(partSize_t*wide);
                            v_tcu.row.push_back(0);
                        }
                        v_tcu.window.push_back(i);
                        v_tcu.atomic.push_back(1);
                        v_tcu.atomic_seq.push_back(1);
                    }
                }
            }
            /*
                4. 汇总三部分的结果
            */
            #pragma omp critical  
            res_t[i]=v_tcu;
    
            #pragma omp critical  
            res_c[i]=large;
    
            #pragma omp critical  
            res_c_residue[i]=small;
            
        
        }
        /*
            5. 整合所有window的数据
        */
        //tcu
        std::vector<int> t_rowkNew_offset;
        t_rowkNew_offset.push_back(0);
        std::vector<at::Half> t_valueNew;
        std::vector<int> t_columnNew;
        std::vector<int> t_window_rowNew;
        std::vector<int> t_atomicNew;
        std::vector<int> t_atomicNew_seq;
    
        //cuda 长行
        std::vector<int> c_rowNew_offset;
        c_rowNew_offset.push_back(0);
        std::vector<int> c_rowNew;
        std::vector<int> c_atomicNew;
        std::vector<int> c_atomicNew_seq;
        std::vector<int> c_columnNew;
        std::vector<float> c_valueNew;
    
        //cuda 短行
        std::vector<int> c_rowNew_offset_residue;
        c_rowNew_offset_residue.push_back(0);
        std::vector<int> c_rowNew_residue;
        std::vector<int> c_atomicNew_residue;
        std::vector<int> c_atomicNew_residue_seq;
        std::vector<int> c_columnNew_residue;
        std::vector<float> c_valueNew_residue;
    
        //按顺序整合res
        //tcu
        for (const auto& pair : res_t) {
            if (pair.second.tcu_flag == true )
            {
                for(int sub : pair.second.row)
                {
                    t_rowkNew_offset.push_back(t_rowkNew_offset.back()+sub);
                }   
                t_valueNew.insert(t_valueNew.end(),pair.second.t_value.begin(),pair.second.t_value.end());
                t_columnNew.insert(t_columnNew.end(),pair.second.t_column.begin(),pair.second.t_column.end());
                t_window_rowNew.insert(t_window_rowNew.end(),pair.second.window.begin(),pair.second.window.end());
                t_atomicNew.insert(t_atomicNew.end(),pair.second.atomic.begin(),pair.second.atomic.end());
                t_atomicNew_seq.insert(t_atomicNew_seq.end(),pair.second.atomic_seq.begin(),pair.second.atomic_seq.end());

            }   
        }
    
        //cuda long
        for (const auto& pair : res_c) {
            for( Value_cuda sub : pair.second){
                if (sub.cuda_flag == true )
                {c_rowNew_offset.push_back(c_rowNew_offset.back()+sub.c_row_offset);
                c_rowNew.push_back(sub.c_row);
                c_atomicNew.push_back(sub.c_atomic);
                c_atomicNew_seq.push_back(sub.c_atomic_seq);
                c_columnNew.insert(c_columnNew.end(),sub.c_column.begin(),sub.c_column.end());
                c_valueNew.insert(c_valueNew.end(),sub.c_value.begin(), sub.c_value.end());}
            }
        }
    
        //cuda short
        for (const auto& pair : res_c_residue) {
            for( Value_cuda sub : pair.second){
                if (sub.cuda_residue_flag == true )
                {c_rowNew_offset_residue.push_back(c_rowNew_offset_residue.back()+sub.c_row_offset);
                c_rowNew_residue.push_back(sub.c_row);
                c_atomicNew_residue.push_back(sub.c_atomic);
                c_atomicNew_residue_seq.push_back(sub.c_atomic_seq);
                c_columnNew_residue.insert(c_columnNew_residue.end(),sub.c_column.begin(),sub.c_column.end());
                c_valueNew_residue.insert(c_valueNew_residue.end(),sub.c_value.begin(), sub.c_value.end());}
            }
        }
    
        auto end = std::chrono::high_resolution_clock::now();
    
        // 计算耗时
        std::chrono::duration<double> elapsed = end - start;
    
        std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;
    
        auto t_rowNew_offsetTensor1 = torch::from_blob(t_rowkNew_offset.data(), t_rowkNew_offset.size(), torch::kInt32);
        auto t_valueTensor1 = torch::from_blob(t_valueNew.data(), t_valueNew.size(), torch::kFloat16);
        auto t_columnTensor1 = torch::from_blob(t_columnNew.data(), t_columnNew.size(), torch::kInt32);
        auto t_window_rowTensor1 = torch::from_blob(t_window_rowNew.data(), t_window_rowNew.size(), torch::kInt32);
        auto t_atomicTensor1 = torch::from_blob(t_atomicNew.data(), t_atomicNew.size(), torch::kInt32);
        auto t_atomicTensor1_seq = torch::from_blob(t_atomicNew_seq.data(), t_atomicNew_seq.size(), torch::kInt32);


        torch::Tensor t_rowNew_offsetTensor = torch::empty_like(t_rowNew_offsetTensor1);
        t_rowNew_offsetTensor.copy_(t_rowNew_offsetTensor1);
        torch::Tensor t_valueTensor = torch::empty_like(t_valueTensor1);
        t_valueTensor.copy_(t_valueTensor1);
        torch::Tensor t_columnTensor = torch::empty_like(t_columnTensor1);
        t_columnTensor.copy_(t_columnTensor1);
        torch::Tensor t_window_rowTensor = torch::empty_like(t_window_rowTensor1);
        t_window_rowTensor.copy_(t_window_rowTensor1);
        torch::Tensor t_atomicTensor = torch::empty_like(t_atomicTensor1);
        t_atomicTensor.copy_(t_atomicTensor1);
        torch::Tensor t_atomicTensor_seq = torch::empty_like(t_atomicTensor1_seq);
        t_atomicTensor_seq.copy_(t_atomicTensor1_seq);

    
        auto c_row_offsetTensor1 = torch::from_blob(c_rowNew_offset.data(), c_rowNew_offset.size(), torch::kInt32);
        auto c_rowTensor1 = torch::from_blob(c_rowNew.data(), c_rowNew.size(), torch::kInt32);
        auto c_atomicTensor1 = torch::from_blob(c_atomicNew.data(), c_atomicNew.size(), torch::kInt32);
        auto c_atomicTensor1_seq = torch::from_blob(c_atomicNew_seq.data(), c_atomicNew_seq.size(), torch::kInt32);   
        auto c_colTensor1 = torch::from_blob(c_columnNew.data(), c_columnNew.size(), torch::kInt32);
        auto c_valueTensor1 = torch::from_blob(c_valueNew.data(), c_valueNew.size(), torch::kFloat32);
        
        torch::Tensor c_row_offsetTensor = torch::empty_like(c_row_offsetTensor1);
        c_row_offsetTensor.copy_(c_row_offsetTensor1);
        torch::Tensor c_rowTensor = torch::empty_like(c_rowTensor1);
        c_rowTensor.copy_(c_rowTensor1);
        torch::Tensor c_atomicTensor = torch::empty_like(c_atomicTensor1);
        c_atomicTensor.copy_(c_atomicTensor1);
        torch::Tensor c_atomicTensor_seq = torch::empty_like(c_atomicTensor1_seq);
        c_atomicTensor_seq.copy_(c_atomicTensor1_seq);
        torch::Tensor c_colTensor = torch::empty_like(c_colTensor1);
        c_colTensor.copy_(c_colTensor1);
        torch::Tensor c_valueTensor = torch::empty_like(c_valueTensor1);
        c_valueTensor.copy_(c_valueTensor1);
        
    
        auto c_row_offsetTensor1_residue = torch::from_blob(c_rowNew_offset_residue.data(), c_rowNew_offset_residue.size(), torch::kInt32);
        auto c_rowTensor1_residue = torch::from_blob(c_rowNew_residue.data(), c_rowNew_residue.size(), torch::kInt32);
        auto c_atomicTensor1_residue = torch::from_blob(c_atomicNew_residue.data(), c_atomicNew_residue.size(), torch::kInt32);
        auto c_atomicTensor1_residue_seq = torch::from_blob(c_atomicNew_residue_seq.data(), c_atomicNew_residue_seq.size(), torch::kInt32);  
        auto c_colTensor1_residue = torch::from_blob(c_columnNew_residue.data(), c_columnNew_residue.size(), torch::kInt32);
        auto c_valueTensor1_residue = torch::from_blob(c_valueNew_residue.data(), c_valueNew_residue.size(), torch::kFloat32);
        
        torch::Tensor c_row_offsetTensor_residue = torch::empty_like(c_row_offsetTensor1_residue);
        c_row_offsetTensor_residue.copy_(c_row_offsetTensor1_residue);
        torch::Tensor c_rowTensor_residue = torch::empty_like(c_rowTensor1_residue);
        c_rowTensor_residue.copy_(c_rowTensor1_residue);
        torch::Tensor c_atomicTensor_residue = torch::empty_like(c_atomicTensor1_residue);
        c_atomicTensor_residue.copy_(c_atomicTensor1_residue);
        torch::Tensor c_atomicTensor_residue_seq = torch::empty_like(c_atomicTensor1_residue_seq);
        c_atomicTensor_residue_seq.copy_(c_atomicTensor1_residue_seq);
        torch::Tensor c_colTensor_residue = torch::empty_like(c_colTensor1_residue);
        c_colTensor_residue.copy_(c_colTensor1_residue);
        torch::Tensor c_valueTensor_residue = torch::empty_like(c_valueTensor1_residue);
        c_valueTensor_residue.copy_(c_valueTensor1_residue);
    
        return {t_rowNew_offsetTensor,t_columnTensor, t_valueTensor, t_window_rowTensor, t_atomicTensor, t_atomicTensor_seq,
                c_row_offsetTensor, c_rowTensor, c_atomicTensor, c_atomicTensor_seq, c_colTensor, c_valueTensor,
                c_row_offsetTensor_residue, c_rowTensor_residue, c_atomicTensor_residue, c_atomicTensor_residue_seq, c_colTensor_residue, c_valueTensor_residue, torch::tensor(elapsed.count())};
    
    }


std::vector<torch::Tensor> block_fp16_tcu_cuda_bcrs(torch::Tensor row1, torch::Tensor column1, torch::Tensor values1, 
    int partSize_t, int partSize_c, int shortSize, int density, int window, int wide)
    {
        auto *row=row1.data_ptr<int>();
        auto column=column1.accessor<int, 1>();
        auto values=values1.accessor<float, 1>();
    
        auto start = std::chrono::high_resolution_clock::now();
    
        int rows=row1.size(0)-1;
        // Result
        std::map<int, Value_tcu_bcrs> res_t;
        std::map<int, std::vector<Value_cuda>> res_c;
        std::map<int, std::vector<Value_cuda>> res_c_residue;
        //需要的window数量
        int num_windows = rows/window;
        #pragma omp parallel for
        for(int i=0;i<num_windows;i++)
        {
            /*
                1. 统计当前window中的vector, 并按照稠密度降序排列
            */
            //对column进行合并
            std::vector<int> mergedVector(&column[row[i*window]], &column[row[i*window+window]]);
            // 使用 unordered_map 统计 vector 中每个元素出现次数
            std::unordered_map<int, int> elementCounts;
            for (const auto& element : mergedVector) elementCounts[element]++;
            // 将统计结果放入 vector<pair> 中
            std::vector<std::pair<int, int>> countVector(elementCounts.begin(), elementCounts.end());
            // 按元素个数降序排序
            std::sort(countVector.begin(), countVector.end(), [](const auto& a, const auto& b) {
                return a.second > b.second;
            });
    
            /*
                2. 确定需要进行tcu计算的vector数量, t_percen
            */
            int size = countVector.size();
            int residue = size%wide;
            //统计每个block中的非零元数量，计算稠密度
            int t_percen = 0; //确定前多少个vector会被用作tcu进行稠密计算
            t_percen = std::count_if(countVector.begin(), countVector.end(), [density](const auto& pair) {
                return pair.second >= density;
            });
            int t_percen_ori = t_percen;
            if((t_percen%wide) != 0)
            {
                // if((t_percen + (wide - t_percen%wide)) <= size) 
                t_percen = (t_percen + (wide - t_percen%wide));
                // else
                // t_percen = (t_percen - (t_percen%wide));
    
            }
            // if(t_percen<=size){
            // // 将前 k 个数按升序排列
            // std::sort(countVector.begin(), countVector.begin() + t_percen);
            // // 将剩下的数按升序排列
            // std::sort(countVector.begin() + t_percen, countVector.end());
            // }else{
            //     std::sort(countVector.begin(), countVector.end());
            // }
            std::sort(countVector.begin(), countVector.begin()+t_percen_ori);
            std::sort(countVector.begin() + t_percen_ori, countVector.end());
            /*
                3. 根据t_percen分别划分TCU和CUDA的负载
            */
    
            // TCU 划分
            // 3.1 根据partSize_t划分t_part
            //当前window中有几个block
            int blocks = t_percen/wide; 
            Value_tcu_bcrs v_tcu;
            if(blocks>0) {v_tcu.tcu_flag = true; }
            std::map<int, int> colmap;
            int c=0;
            for(auto col:countVector){
                if (c<t_percen_ori)
                {
                    v_tcu.t_column.push_back(col.first);
                }
                colmap[col.first]=c++;
            }
            if((t_percen_ori%wide)>0){
                for(int tt=0;tt<(wide-(t_percen_ori%wide));tt++)
                v_tcu.t_column.push_back(-1);
            }
            //存放拆分的cuda长短行数据
            std::vector<Value_cuda> large;
            std::vector<Value_cuda> small;
    
            // std::vector<std::vector<float>> t_value_temp = createVectorOfVectors_float(blocks);
            // std::vector<long> t_binary_temp = createVectorOfVectors_v2(blocks);
            std::vector<at::Half> demo(t_percen*window);
            // std::vector<std::vector<int>> t_row_temp = createVectorOfVectors(blocks);
            // std::vector<std::vector<int>> t_col_temp = createVectorOfVectors(blocks);
            // 指示快内的行偏移
            int p=0;
            for(int j=i*window;j<(i+1)*window;j++)
            {
                //用来存储每行的cuda数据
                std::queue<int> c_col;
                std::queue<int> c_value;
                int cuda_nnz=0;
                for(int m=row[j];m<row[j+1];m++)
                {
                    int newCol = colmap[column[m]];
                    if(newCol<t_percen_ori)
                    {
                        int block = newCol/wide;
                        int bInId= newCol%wide;
                        demo[block*window*wide + p*wide +bInId]=values[m];
                    }else{
                        //划分cuda部分
                        c_col.push(column[m]);
                        c_value.push(values[m]);
                        cuda_nnz++;
                    }
                }
                //处理当前行剩余的cuda元素
                if(cuda_nnz>0)
                {
                    if(cuda_nnz <= shortSize)
                    {
                        Value_cuda temp;
                        temp.cuda_residue_flag = true;
                        temp.c_row_offset = cuda_nnz;
                        temp.c_row = i*window + p;
                        if(t_percen_ori>0) temp.c_atomic = 1;
                        else temp.c_atomic = 0;
                        for(int m=0; m<cuda_nnz; m++)
                        {
                            temp.c_column.push_back(c_col.front());
                            temp.c_value.push_back(c_value.front());
                            c_col.pop();
                            c_value.pop();
                        }
                        small.push_back(temp);
                    }else{
                        if(cuda_nnz <= partSize_c)
                        {
                            Value_cuda temp;
                            temp.cuda_flag = true;
                            temp.c_row_offset = (cuda_nnz);
                            temp.c_row = i*window + p;
                            if(t_percen_ori>0) temp.c_atomic = 1;
                            else temp.c_atomic = 0;
                            for(int m=0; m<cuda_nnz; m++)
                            {
                                temp.c_column.push_back(c_col.front());
                                temp.c_value.push_back(c_value.front());
                                c_col.pop();
                                c_value.pop();
                            }
                            large.push_back(temp);
                        }else{
                            // 当前行的非零元被分成多个part，并需要原子操作
                            int parts_row = cuda_nnz/partSize_c;
                            // int cur_value_indice = row[i];
                            for(int cur_i = 0; cur_i<parts_row; cur_i++)
                            {
                                Value_cuda temp;
                                temp.cuda_flag = true;      
                                temp.c_row_offset = partSize_c;
                                temp.c_row = i*window + p;
                                temp.c_atomic = 1;
                                for(int m=0; m<partSize_c; m++)
                                {
                                    temp.c_column.push_back(c_col.front());
                                    temp.c_value.push_back(c_value.front());
                                    c_col.pop();
                                    c_value.pop();
                                    // cur_value_indice++;
                                }
                                large.push_back(temp);
                            }
                            int residue  = (cuda_nnz%partSize_c);
                            if(residue>0){
                                Value_cuda temp;
                                temp.c_row_offset = residue;
                                temp.c_row = i*window + p;
                                temp.c_atomic = 1;
                                for(int m=0; m<residue; m++)
                                {
                                    temp.c_column.push_back(c_col.front());
                                    temp.c_value.push_back(c_value.front());
                                    c_col.pop();
                                    c_value.pop();
                                    // cur_value_indice++;
                                }
                                if(residue <= shortSize)
                                {
                                    temp.cuda_residue_flag = true;
                                    small.push_back(temp);
                                }else{
                                    temp.cuda_flag = true;
                                    large.push_back(temp);
                                }
                            }
                        }
                    }
                }
                p++;
            }
            //整合vector
            v_tcu.t_value = demo;
            // v_tcu.t_row = mergeVectors(t_row_temp);
            // v_tcu.t_col = mergeVectors(t_col_temp);
        
            if(blocks <= partSize_t)
            {
                v_tcu.row.push_back(t_percen_ori);
                v_tcu.row.push_back(t_percen-t_percen_ori);
                if(t_percen_ori==size) v_tcu.atomic.push_back(0);
                else v_tcu.atomic.push_back(1);
                v_tcu.window.push_back(i);
            }else{
                // 对block太多的块进行分割
                int part_number = blocks / partSize_t;
                int block_residue = blocks % partSize_t;
                if(block_residue > 0)
                {
                    for(int j=0; j<part_number; j++)
                    {
                        v_tcu.row.push_back(partSize_t*wide);
                        v_tcu.row.push_back(0);
                        v_tcu.window.push_back(i);
                        v_tcu.atomic.push_back(1);
                    }
                    v_tcu.row.push_back(block_residue*wide - (t_percen-t_percen_ori));
                    v_tcu.row.push_back(t_percen-t_percen_ori);
                    v_tcu.window.push_back(i);
                    v_tcu.atomic.push_back(1);
                }else{
                    for(int j=0; j<part_number; j++)
                    {
                        //如果是最后一组block
                        if(j == (part_number-1)){
                            v_tcu.row.push_back((partSize_t*wide) - (t_percen-t_percen_ori));
                            v_tcu.row.push_back(t_percen-t_percen_ori);
                        }else{
                            v_tcu.row.push_back(partSize_t*wide);
                            v_tcu.row.push_back(0);
                        }
                        v_tcu.window.push_back(i);
                        v_tcu.atomic.push_back(1);
                    }
                }
            }
            /*
                4. 汇总三部分的结果
            */
            #pragma omp critical  
            res_t[i]=v_tcu;
    
            #pragma omp critical  
            res_c[i]=large;
    
            #pragma omp critical  
            res_c_residue[i]=small;
            
        
        }
        /*
            5. 整合所有window的数据
        */
        //tcu
        std::vector<int> t_rowkNew_offset;
        t_rowkNew_offset.push_back(0);
        std::vector<at::Half> t_valueNew;
        std::vector<int> t_columnNew;
        std::vector<int> t_window_rowNew;
        std::vector<int> t_atomicNew;
    
        //cuda 长行
        std::vector<int> c_rowNew_offset;
        c_rowNew_offset.push_back(0);
        std::vector<int> c_rowNew;
        std::vector<int> c_atomicNew;
        std::vector<int> c_columnNew;
        std::vector<float> c_valueNew;
    
        //cuda 短行
        std::vector<int> c_rowNew_offset_residue;
        c_rowNew_offset_residue.push_back(0);
        std::vector<int> c_rowNew_residue;
        std::vector<int> c_atomicNew_residue;
        std::vector<int> c_columnNew_residue;
        std::vector<float> c_valueNew_residue;
    
        //按顺序整合res
        //tcu
        for (const auto& pair : res_t) {
            if (pair.second.tcu_flag == true )
            {
                for(int sub : pair.second.row)
                {
                    t_rowkNew_offset.push_back(t_rowkNew_offset.back()+sub);
                }   
                t_valueNew.insert(t_valueNew.end(),pair.second.t_value.begin(),pair.second.t_value.end());
                t_columnNew.insert(t_columnNew.end(),pair.second.t_column.begin(),pair.second.t_column.end());
                t_window_rowNew.insert(t_window_rowNew.end(),pair.second.window.begin(),pair.second.window.end());
                t_atomicNew.insert(t_atomicNew.end(),pair.second.atomic.begin(),pair.second.atomic.end());
            }   
        }
    
        //cuda long
        for (const auto& pair : res_c) {
            for( Value_cuda sub : pair.second){
                if (sub.cuda_flag == true )
                {c_rowNew_offset.push_back(c_rowNew_offset.back()+sub.c_row_offset);
                c_rowNew.push_back(sub.c_row);
                c_atomicNew.push_back(sub.c_atomic);
                c_columnNew.insert(c_columnNew.end(),sub.c_column.begin(),sub.c_column.end());
                c_valueNew.insert(c_valueNew.end(),sub.c_value.begin(), sub.c_value.end());}
            }
        }
    
        //cuda short
        for (const auto& pair : res_c_residue) {
            for( Value_cuda sub : pair.second){
                if (sub.cuda_residue_flag == true )
                {c_rowNew_offset_residue.push_back(c_rowNew_offset_residue.back()+sub.c_row_offset);
                c_rowNew_residue.push_back(sub.c_row);
                c_atomicNew_residue.push_back(sub.c_atomic);
                c_columnNew_residue.insert(c_columnNew_residue.end(),sub.c_column.begin(),sub.c_column.end());
                c_valueNew_residue.insert(c_valueNew_residue.end(),sub.c_value.begin(), sub.c_value.end());}
            }
        }
    
        auto end = std::chrono::high_resolution_clock::now();
    
        // 计算耗时
        std::chrono::duration<double> elapsed = end - start;
    
        std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;
    
        auto t_rowNew_offsetTensor1 = torch::from_blob(t_rowkNew_offset.data(), t_rowkNew_offset.size(), torch::kInt32);
        auto t_valueTensor1 = torch::from_blob(t_valueNew.data(), t_valueNew.size(), torch::kFloat16);
        auto t_columnTensor1 = torch::from_blob(t_columnNew.data(), t_columnNew.size(), torch::kInt32);
        auto t_window_rowTensor1 = torch::from_blob(t_window_rowNew.data(), t_window_rowNew.size(), torch::kInt32);
        auto t_atomicTensor1 = torch::from_blob(t_atomicNew.data(), t_atomicNew.size(), torch::kInt32);
    
        torch::Tensor t_rowNew_offsetTensor = torch::empty_like(t_rowNew_offsetTensor1);
        t_rowNew_offsetTensor.copy_(t_rowNew_offsetTensor1);
        torch::Tensor t_valueTensor = torch::empty_like(t_valueTensor1);
        t_valueTensor.copy_(t_valueTensor1);
        torch::Tensor t_columnTensor = torch::empty_like(t_columnTensor1);
        t_columnTensor.copy_(t_columnTensor1);
        torch::Tensor t_window_rowTensor = torch::empty_like(t_window_rowTensor1);
        t_window_rowTensor.copy_(t_window_rowTensor1);
        torch::Tensor t_atomicTensor = torch::empty_like(t_atomicTensor1);
        t_atomicTensor.copy_(t_atomicTensor1);
    
    
        auto c_row_offsetTensor1 = torch::from_blob(c_rowNew_offset.data(), c_rowNew_offset.size(), torch::kInt32);
        auto c_rowTensor1 = torch::from_blob(c_rowNew.data(), c_rowNew.size(), torch::kInt32);
        auto c_atomicTensor1 = torch::from_blob(c_atomicNew.data(), c_atomicNew.size(), torch::kInt32);
        auto c_colTensor1 = torch::from_blob(c_columnNew.data(), c_columnNew.size(), torch::kInt32);
        auto c_valueTensor1 = torch::from_blob(c_valueNew.data(), c_valueNew.size(), torch::kFloat32);
        
        torch::Tensor c_row_offsetTensor = torch::empty_like(c_row_offsetTensor1);
        c_row_offsetTensor.copy_(c_row_offsetTensor1);
        torch::Tensor c_rowTensor = torch::empty_like(c_rowTensor1);
        c_rowTensor.copy_(c_rowTensor1);
        torch::Tensor c_atomicTensor = torch::empty_like(c_atomicTensor1);
        c_atomicTensor.copy_(c_atomicTensor1);
        torch::Tensor c_colTensor = torch::empty_like(c_colTensor1);
        c_colTensor.copy_(c_colTensor1);
        torch::Tensor c_valueTensor = torch::empty_like(c_valueTensor1);
        c_valueTensor.copy_(c_valueTensor1);
        
    
        auto c_row_offsetTensor1_residue = torch::from_blob(c_rowNew_offset_residue.data(), c_rowNew_offset_residue.size(), torch::kInt32);
        auto c_rowTensor1_residue = torch::from_blob(c_rowNew_residue.data(), c_rowNew_residue.size(), torch::kInt32);
        auto c_atomicTensor1_residue = torch::from_blob(c_atomicNew_residue.data(), c_atomicNew_residue.size(), torch::kInt32);
        auto c_colTensor1_residue = torch::from_blob(c_columnNew_residue.data(), c_columnNew_residue.size(), torch::kInt32);
        auto c_valueTensor1_residue = torch::from_blob(c_valueNew_residue.data(), c_valueNew_residue.size(), torch::kFloat32);
        
        torch::Tensor c_row_offsetTensor_residue = torch::empty_like(c_row_offsetTensor1_residue);
        c_row_offsetTensor_residue.copy_(c_row_offsetTensor1_residue);
        torch::Tensor c_rowTensor_residue = torch::empty_like(c_rowTensor1_residue);
        c_rowTensor_residue.copy_(c_rowTensor1_residue);
        torch::Tensor c_atomicTensor_residue = torch::empty_like(c_atomicTensor1_residue);
        c_atomicTensor_residue.copy_(c_atomicTensor1_residue);
        torch::Tensor c_colTensor_residue = torch::empty_like(c_colTensor1_residue);
        c_colTensor_residue.copy_(c_colTensor1_residue);
        torch::Tensor c_valueTensor_residue = torch::empty_like(c_valueTensor1_residue);
        c_valueTensor_residue.copy_(c_valueTensor1_residue);
    
        return {t_rowNew_offsetTensor,t_columnTensor, t_valueTensor, t_window_rowTensor, t_atomicTensor,
                c_row_offsetTensor, c_rowTensor, c_atomicTensor, c_colTensor, c_valueTensor,
                c_row_offsetTensor_residue, c_rowTensor_residue, c_atomicTensor_residue, c_colTensor_residue, c_valueTensor_residue, torch::tensor(elapsed.count())};
    
    }


//     // m-tcu + cuda + partSize + binary
// std::vector<torch::Tensor> block_fp16_tcu_cuda_bcrs_seq(torch::Tensor row1, torch::Tensor column1, torch::Tensor values1, 
//     int partSize_t, int partSize_c, int shortSize, int density, int window, int wide)
//     {
//         auto *row=row1.data_ptr<int>();
//         auto column=column1.accessor<int, 1>();
//         auto values=values1.accessor<float, 1>();
    
//         auto start = std::chrono::high_resolution_clock::now();
    
//         int rows=row1.size(0)-1;
//         // Result
//         std::map<int, Value_tcu_bcrs> res_t;
//         std::map<int, std::vector<Value_cuda>> res_c;
//         std::map<int, std::vector<Value_cuda>> res_c_residue;
//         //需要的window数量
//         int num_windows = rows/window;
//         #pragma omp parallel for
//         for(int i=0;i<num_windows;i++)
//         {
//             /*
//                 1. 统计当前window中的vector, 并按照稠密度降序排列
//             */
//             //对column进行合并
//             std::vector<int> mergedVector(&column[row[i*window]], &column[row[i*window+window]]);
//             // 使用 unordered_map 统计 vector 中每个元素出现次数
//             std::unordered_map<int, int> elementCounts;
//             for (const auto& element : mergedVector) elementCounts[element]++;
//             // 将统计结果放入 vector<pair> 中
//             std::vector<std::pair<int, int>> countVector(elementCounts.begin(), elementCounts.end());
//             // 按元素个数降序排序
//             std::sort(countVector.begin(), countVector.end(), [](const auto& a, const auto& b) {
//                 return a.second > b.second;
//             });
    
//             /*
//                 2. 确定需要进行tcu计算的vector数量, t_percen
//             */
//             int size = countVector.size();
//             int residue = size%wide;
//             //统计每个block中的非零元数量，计算稠密度
//             int t_percen = 0; //确定前多少个vector会被用作tcu进行稠密计算
//             t_percen = std::count_if(countVector.begin(), countVector.end(), [density](const auto& pair) {
//                 return pair.second >= density;
//             });
//             int t_percen_ori = t_percen;
//             if((t_percen%wide) != 0)
//             {
//                 // if((t_percen + (wide - t_percen%wide)) <= size) 
//                 t_percen = (t_percen + (wide - t_percen%wide));
//                 // else
//                 // t_percen = (t_percen - (t_percen%wide));
    
//             }
//             // if(t_percen<=size){
//             // // 将前 k 个数按升序排列
//             // std::sort(countVector.begin(), countVector.begin() + t_percen);
//             // // 将剩下的数按升序排列
//             // std::sort(countVector.begin() + t_percen, countVector.end());
//             // }else{
//             //     std::sort(countVector.begin(), countVector.end());
//             // }
//             std::sort(countVector.begin(), countVector.begin()+t_percen_ori);
//             std::sort(countVector.begin() + t_percen_ori, countVector.end());
//             /*
//                 3. 根据t_percen分别划分TCU和CUDA的负载
//             */
    
//             // TCU 划分
//             // 3.1 根据partSize_t划分t_part
//             //当前window中有几个block
//             int blocks = t_percen/wide; 
//             Value_tcu_bcrs v_tcu;
//             if(blocks>0) {v_tcu.tcu_flag = true; }
//             std::map<int, int> colmap;
//             int c=0;
//             for(auto col:countVector){
//                 if (c<t_percen_ori)
//                 {
//                     v_tcu.t_column.push_back(col.first);
//                 }
//                 colmap[col.first]=c++;
//             }
//             if((t_percen_ori%wide)>0){
//                 for(int tt=0;tt<(wide-(t_percen_ori%wide));tt++)
//                 v_tcu.t_column.push_back(-1);
//             }
//             //存放拆分的cuda长短行数据
//             std::vector<Value_cuda> large;
//             std::vector<Value_cuda> small;
    
//             // std::vector<std::vector<float>> t_value_temp = createVectorOfVectors_float(blocks);
//             // std::vector<long> t_binary_temp = createVectorOfVectors_v2(blocks);
//             std::vector<at::Half> demo(t_percen*window);
//             // std::vector<std::vector<int>> t_row_temp = createVectorOfVectors(blocks);
//             // std::vector<std::vector<int>> t_col_temp = createVectorOfVectors(blocks);
//             // 指示快内的行偏移
//             int p=0;
//             for(int j=i*window;j<(i+1)*window;j++)
//             {
//                 //用来存储每行的cuda数据
//                 std::queue<int> c_col;
//                 std::queue<int> c_value;
//                 int cuda_nnz=0;
//                 for(int m=row[j];m<row[j+1];m++)
//                 {
//                     int newCol = colmap[column[m]];
//                     if(newCol<t_percen_ori)
//                     {
//                         int block = newCol/wide;
//                         int bInId= newCol%wide;
//                         demo[block*window*wide + p*wide +bInId]=values[m];
//                     }else{
//                         //划分cuda部分
//                         c_col.push(column[m]);
//                         c_value.push(values[m]);
//                         cuda_nnz++;
//                     }
//                 }
//                 //处理当前行剩余的cuda元素
//                 if(cuda_nnz>0)
//                 {
//                     if(cuda_nnz <= shortSize)
//                     {
//                         Value_cuda temp;
//                         temp.cuda_residue_flag = true;
//                         temp.c_row_offset = cuda_nnz;
//                         temp.c_row = i*window + p;
//                         // if(t_percen_ori>0) temp.c_atomic = 1;
//                         // else temp.c_atomic = 0;
//                         temp.c_atomic = 0;
//                         for(int m=0; m<cuda_nnz; m++)
//                         {
//                             temp.c_column.push_back(c_col.front());
//                             temp.c_value.push_back(c_value.front());
//                             c_col.pop();
//                             c_value.pop();
//                         }
//                         small.push_back(temp);
//                     }else{
//                         if(cuda_nnz <= partSize_c)
//                         {
//                             Value_cuda temp;
//                             temp.cuda_flag = true;
//                             temp.c_row_offset = (cuda_nnz);
//                             temp.c_row = i*window + p;
//                             // if(t_percen_ori>0) temp.c_atomic = 1;
//                             // else temp.c_atomic = 0;
//                             temp.c_atomic = 0;
//                             for(int m=0; m<cuda_nnz; m++)
//                             {
//                                 temp.c_column.push_back(c_col.front());
//                                 temp.c_value.push_back(c_value.front());
//                                 c_col.pop();
//                                 c_value.pop();
//                             }
//                             large.push_back(temp);
//                         }else{
//                             // 当前行的非零元被分成多个part，并需要原子操作
//                             int parts_row = cuda_nnz/partSize_c;
//                             // int cur_value_indice = row[i];
//                             for(int cur_i = 0; cur_i<parts_row; cur_i++)
//                             {
//                                 Value_cuda temp;
//                                 temp.cuda_flag = true;      
//                                 temp.c_row_offset = partSize_c;
//                                 temp.c_row = i*window + p;
//                                 temp.c_atomic = 1;
//                                 for(int m=0; m<partSize_c; m++)
//                                 {
//                                     temp.c_column.push_back(c_col.front());
//                                     temp.c_value.push_back(c_value.front());
//                                     c_col.pop();
//                                     c_value.pop();
//                                     // cur_value_indice++;
//                                 }
//                                 large.push_back(temp);
//                             }
//                             int residue  = (cuda_nnz%partSize_c);
//                             if(residue>0){
//                                 Value_cuda temp;
//                                 temp.c_row_offset = residue;
//                                 temp.c_row = i*window + p;
//                                 temp.c_atomic = 1;
//                                 for(int m=0; m<residue; m++)
//                                 {
//                                     temp.c_column.push_back(c_col.front());
//                                     temp.c_value.push_back(c_value.front());
//                                     c_col.pop();
//                                     c_value.pop();
//                                     // cur_value_indice++;
//                                 }
//                                 if(residue <= shortSize)
//                                 {
//                                     temp.cuda_residue_flag = true;
//                                     small.push_back(temp);
//                                 }else{
//                                     temp.cuda_flag = true;
//                                     large.push_back(temp);
//                                 }
//                             }
//                         }
//                     }
//                 }
//                 p++;
//             }
//             //整合vector
//             v_tcu.t_value = demo;
//             // v_tcu.t_row = mergeVectors(t_row_temp);
//             // v_tcu.t_col = mergeVectors(t_col_temp);
        
//             if(blocks <= partSize_t)
//             {
//                 v_tcu.row.push_back(t_percen_ori);
//                 v_tcu.row.push_back(t_percen-t_percen_ori);
//                 // if(t_percen_ori==size) v_tcu.atomic.push_back(0);
//                 // else v_tcu.atomic.push_back(1);
//                 v_tcu.atomic.push_back(0);
//                 v_tcu.window.push_back(i);
//             }else{
//                 // 对block太多的块进行分割
//                 int part_number = blocks / partSize_t;
//                 int block_residue = blocks % partSize_t;
//                 if(block_residue > 0)
//                 {
//                     for(int j=0; j<part_number; j++)
//                     {
//                         v_tcu.row.push_back(partSize_t*wide);
//                         v_tcu.row.push_back(0);
//                         v_tcu.window.push_back(i);
//                         v_tcu.atomic.push_back(1);
//                     }
//                     v_tcu.row.push_back(block_residue*wide - (t_percen-t_percen_ori));
//                     v_tcu.row.push_back(t_percen-t_percen_ori);
//                     v_tcu.window.push_back(i);
//                     v_tcu.atomic.push_back(1);
//                 }else{
//                     for(int j=0; j<part_number; j++)
//                     {
//                         //如果是最后一组block
//                         if(j == (part_number-1)){
//                             v_tcu.row.push_back((partSize_t*wide) - (t_percen-t_percen_ori));
//                             v_tcu.row.push_back(t_percen-t_percen_ori);
//                         }else{
//                             v_tcu.row.push_back(partSize_t*wide);
//                             v_tcu.row.push_back(0);
//                         }
//                         v_tcu.window.push_back(i);
//                         v_tcu.atomic.push_back(1);
//                     }
//                 }
//             }
//             /*
//                 4. 汇总三部分的结果
//             */
//             #pragma omp critical  
//             res_t[i]=v_tcu;
    
//             #pragma omp critical  
//             res_c[i]=large;
    
//             #pragma omp critical  
//             res_c_residue[i]=small;
            
        
//         }
//         /*
//             5. 整合所有window的数据
//         */
//         //tcu
//         std::vector<int> t_rowkNew_offset;
//         t_rowkNew_offset.push_back(0);
//         std::vector<at::Half> t_valueNew;
//         std::vector<int> t_columnNew;
//         std::vector<int> t_window_rowNew;
//         std::vector<int> t_atomicNew;
    
//         //cuda 长行
//         std::vector<int> c_rowNew_offset;
//         c_rowNew_offset.push_back(0);
//         std::vector<int> c_rowNew;
//         std::vector<int> c_atomicNew;
//         std::vector<int> c_columnNew;
//         std::vector<float> c_valueNew;
    
//         //cuda 短行
//         std::vector<int> c_rowNew_offset_residue;
//         c_rowNew_offset_residue.push_back(0);
//         std::vector<int> c_rowNew_residue;
//         std::vector<int> c_atomicNew_residue;
//         std::vector<int> c_columnNew_residue;
//         std::vector<float> c_valueNew_residue;
    
//         //按顺序整合res
//         //tcu
//         for (const auto& pair : res_t) {
//             if (pair.second.tcu_flag == true )
//             {
//                 for(int sub : pair.second.row)
//                 {
//                     t_rowkNew_offset.push_back(t_rowkNew_offset.back()+sub);
//                 }   
//                 t_valueNew.insert(t_valueNew.end(),pair.second.t_value.begin(),pair.second.t_value.end());
//                 t_columnNew.insert(t_columnNew.end(),pair.second.t_column.begin(),pair.second.t_column.end());
//                 t_window_rowNew.insert(t_window_rowNew.end(),pair.second.window.begin(),pair.second.window.end());
//                 t_atomicNew.insert(t_atomicNew.end(),pair.second.atomic.begin(),pair.second.atomic.end());
//             }   
//         }
    
//         //cuda long
//         for (const auto& pair : res_c) {
//             for( Value_cuda sub : pair.second){
//                 if (sub.cuda_flag == true )
//                 {c_rowNew_offset.push_back(c_rowNew_offset.back()+sub.c_row_offset);
//                 c_rowNew.push_back(sub.c_row);
//                 c_atomicNew.push_back(sub.c_atomic);
//                 c_columnNew.insert(c_columnNew.end(),sub.c_column.begin(),sub.c_column.end());
//                 c_valueNew.insert(c_valueNew.end(),sub.c_value.begin(), sub.c_value.end());}
//             }
//         }
    
//         //cuda short
//         for (const auto& pair : res_c_residue) {
//             for( Value_cuda sub : pair.second){
//                 if (sub.cuda_residue_flag == true )
//                 {c_rowNew_offset_residue.push_back(c_rowNew_offset_residue.back()+sub.c_row_offset);
//                 c_rowNew_residue.push_back(sub.c_row);
//                 c_atomicNew_residue.push_back(sub.c_atomic);
//                 c_columnNew_residue.insert(c_columnNew_residue.end(),sub.c_column.begin(),sub.c_column.end());
//                 c_valueNew_residue.insert(c_valueNew_residue.end(),sub.c_value.begin(), sub.c_value.end());}
//             }
//         }
    
//         auto end = std::chrono::high_resolution_clock::now();
    
//         // 计算耗时
//         std::chrono::duration<double> elapsed = end - start;
    
//         std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;
    
//         auto t_rowNew_offsetTensor1 = torch::from_blob(t_rowkNew_offset.data(), t_rowkNew_offset.size(), torch::kInt32);
//         auto t_valueTensor1 = torch::from_blob(t_valueNew.data(), t_valueNew.size(), torch::kFloat16);
//         auto t_columnTensor1 = torch::from_blob(t_columnNew.data(), t_columnNew.size(), torch::kInt32);
//         auto t_window_rowTensor1 = torch::from_blob(t_window_rowNew.data(), t_window_rowNew.size(), torch::kInt32);
//         auto t_atomicTensor1 = torch::from_blob(t_atomicNew.data(), t_atomicNew.size(), torch::kInt32);
    
//         torch::Tensor t_rowNew_offsetTensor = torch::empty_like(t_rowNew_offsetTensor1);
//         t_rowNew_offsetTensor.copy_(t_rowNew_offsetTensor1);
//         torch::Tensor t_valueTensor = torch::empty_like(t_valueTensor1);
//         t_valueTensor.copy_(t_valueTensor1);
//         torch::Tensor t_columnTensor = torch::empty_like(t_columnTensor1);
//         t_columnTensor.copy_(t_columnTensor1);
//         torch::Tensor t_window_rowTensor = torch::empty_like(t_window_rowTensor1);
//         t_window_rowTensor.copy_(t_window_rowTensor1);
//         torch::Tensor t_atomicTensor = torch::empty_like(t_atomicTensor1);
//         t_atomicTensor.copy_(t_atomicTensor1);
    
    
//         auto c_row_offsetTensor1 = torch::from_blob(c_rowNew_offset.data(), c_rowNew_offset.size(), torch::kInt32);
//         auto c_rowTensor1 = torch::from_blob(c_rowNew.data(), c_rowNew.size(), torch::kInt32);
//         auto c_atomicTensor1 = torch::from_blob(c_atomicNew.data(), c_atomicNew.size(), torch::kInt32);
//         auto c_colTensor1 = torch::from_blob(c_columnNew.data(), c_columnNew.size(), torch::kInt32);
//         auto c_valueTensor1 = torch::from_blob(c_valueNew.data(), c_valueNew.size(), torch::kFloat32);
        
//         torch::Tensor c_row_offsetTensor = torch::empty_like(c_row_offsetTensor1);
//         c_row_offsetTensor.copy_(c_row_offsetTensor1);
//         torch::Tensor c_rowTensor = torch::empty_like(c_rowTensor1);
//         c_rowTensor.copy_(c_rowTensor1);
//         torch::Tensor c_atomicTensor = torch::empty_like(c_atomicTensor1);
//         c_atomicTensor.copy_(c_atomicTensor1);
//         torch::Tensor c_colTensor = torch::empty_like(c_colTensor1);
//         c_colTensor.copy_(c_colTensor1);
//         torch::Tensor c_valueTensor = torch::empty_like(c_valueTensor1);
//         c_valueTensor.copy_(c_valueTensor1);
        
    
//         auto c_row_offsetTensor1_residue = torch::from_blob(c_rowNew_offset_residue.data(), c_rowNew_offset_residue.size(), torch::kInt32);
//         auto c_rowTensor1_residue = torch::from_blob(c_rowNew_residue.data(), c_rowNew_residue.size(), torch::kInt32);
//         auto c_atomicTensor1_residue = torch::from_blob(c_atomicNew_residue.data(), c_atomicNew_residue.size(), torch::kInt32);
//         auto c_colTensor1_residue = torch::from_blob(c_columnNew_residue.data(), c_columnNew_residue.size(), torch::kInt32);
//         auto c_valueTensor1_residue = torch::from_blob(c_valueNew_residue.data(), c_valueNew_residue.size(), torch::kFloat32);
        
//         torch::Tensor c_row_offsetTensor_residue = torch::empty_like(c_row_offsetTensor1_residue);
//         c_row_offsetTensor_residue.copy_(c_row_offsetTensor1_residue);
//         torch::Tensor c_rowTensor_residue = torch::empty_like(c_rowTensor1_residue);
//         c_rowTensor_residue.copy_(c_rowTensor1_residue);
//         torch::Tensor c_atomicTensor_residue = torch::empty_like(c_atomicTensor1_residue);
//         c_atomicTensor_residue.copy_(c_atomicTensor1_residue);
//         torch::Tensor c_colTensor_residue = torch::empty_like(c_colTensor1_residue);
//         c_colTensor_residue.copy_(c_colTensor1_residue);
//         torch::Tensor c_valueTensor_residue = torch::empty_like(c_valueTensor1_residue);
//         c_valueTensor_residue.copy_(c_valueTensor1_residue);
    
//         return {t_rowNew_offsetTensor,t_columnTensor, t_valueTensor, t_window_rowTensor, t_atomicTensor,
//                 c_row_offsetTensor, c_rowTensor, c_atomicTensor, c_colTensor, c_valueTensor,
//                 c_row_offsetTensor_residue, c_rowTensor_residue, c_atomicTensor_residue, c_colTensor_residue, c_valueTensor_residue, torch::tensor(elapsed.count())};
    
//     }



    //only TCU
// tcu cuda
float spmm_forward_fp16_tcu_bcrs_part_kernel(
    int * t_row_offset,
    half * t_value, 
    int * t_column, 
    int* t_window_row,
    int * t_atomic,

    half * rhs_matrix,
    float * output_matrix,

    const int parts_t,
    const int dimN,
    const int mOri,
    int epoches);

std::vector<torch::Tensor> spmm_forward_fp16_tcu_bcrs_part(
    torch::Tensor t_row_offset,
    torch::Tensor t_column, 
    torch::Tensor t_value, 
    torch::Tensor t_window_row,
    torch::Tensor t_atomic,

    torch::Tensor rhs_matrix,

    const int parts_t,

    const int dimN,
    const int mOri,
    const int kOri,
    int epoches)
{
    auto output_matrix = torch::zeros({mOri, dimN}, torch::kFloat32).to(torch::kCPU);
    
    //把CPU端的tensor转成C++的数据结构
    int * t_row_offset_ = t_row_offset.data<int>();
    half * t_value_ = reinterpret_cast<half *>(t_value.data<at::Half>()); 
    int * t_column_ = t_column.data<int>();
    int * t_window_row_ = t_window_row.data<int>();
    int * t_atomic_ = t_atomic.data<int>();

    half * rhs_matrix_ = reinterpret_cast<half *>(rhs_matrix.data<at::Half>()); 
    float * output_matrix_ = output_matrix.data<float>();

    // Device
    int *d_t_row_offset, *d_t_column;
    int *d_t_window_row, *d_t_atomic;
    half *d_t_value; 
    // int *d_c_row_offsets, *d_c_row, *d_c_atomic, *d_c_column;
    // int *d_c_row_offsets_short, *d_c_row_short, *d_c_atomic_short, *d_c_column_short;
    // half *d_c_value, *d_c_value_short; 
	half *d_rhs_matrix;
    float *d_output_matrix;

    checkCuda(cudaMalloc(&d_t_row_offset, (t_row_offset.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_value, (t_value.size(0)) * sizeof(half)));
    checkCuda(cudaMalloc(&d_t_column, (t_column.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_window_row, (t_window_row.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_atomic, (t_atomic.size(0)) * sizeof(int)));

    checkCuda(cudaMalloc(&d_rhs_matrix, (kOri*dimN) * sizeof(half)));
    checkCuda(cudaMalloc(&d_output_matrix, (mOri*dimN) * sizeof(float)));

    //把CPU数据放到GPU上
    checkCuda(cudaMemcpy(d_t_row_offset, t_row_offset_ , (t_row_offset.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_value, t_value_, (t_value.size(0)) * sizeof(half), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_column, t_column_, (t_column.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_window_row, t_window_row_, (t_window_row.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_atomic, t_atomic_, (t_atomic.size(0)) * sizeof(int), cudaMemcpyHostToDevice));


    checkCuda(cudaMemcpy(d_rhs_matrix, rhs_matrix_,(kOri*dimN) * sizeof(half), cudaMemcpyHostToDevice));

    float spmm_ms_avg =  spmm_forward_fp16_tcu_bcrs_part_kernel(
      d_t_row_offset,
      d_t_value, 
      d_t_column,
      d_t_window_row,
      d_t_atomic,

      d_rhs_matrix,
      d_output_matrix,

      parts_t,

      dimN,
      mOri,
      epoches);    
    checkCuda(cudaMemcpy(output_matrix_, d_output_matrix, mOri * dimN * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_t_row_offset);
    cudaFree(d_t_value);
    cudaFree(d_t_column);
    cudaFree(d_t_window_row);
    cudaFree(d_t_atomic);

    cudaFree(d_rhs_matrix);
    cudaFree(d_output_matrix);
    cudaDeviceSynchronize(); 
    return {output_matrix, torch::tensor(spmm_ms_avg)};
}


float spmm_forward_tf32_tcu_bcrs_part_kernel(
    int * t_row_offset,
    float * t_value, 
    int * t_column, 
    int* t_window_row,
    int * t_atomic,

    float * rhs_matrix,
    float * output_matrix,

    const int parts_t,
    const int dimN,
    const int mOri,
    int epoches);

std::vector<torch::Tensor> spmm_forward_tf32_tcu_bcrs_part(
    torch::Tensor t_row_offset,
    torch::Tensor t_column, 
    torch::Tensor t_value, 
    torch::Tensor t_window_row,
    torch::Tensor t_atomic,

    torch::Tensor rhs_matrix,

    const int parts_t,

    const int dimN,
    const int mOri,
    const int kOri,
    int epoches)
{
    auto output_matrix = torch::zeros({mOri, dimN}, torch::kFloat32).to(torch::kCPU);
    
    //把CPU端的tensor转成C++的数据结构
    int * t_row_offset_ = t_row_offset.data<int>();
    float * t_value_ = t_value.data<float>(); 
    int * t_column_ = t_column.data<int>();
    int * t_window_row_ = t_window_row.data<int>();
    int * t_atomic_ = t_atomic.data<int>();

    float * rhs_matrix_ = rhs_matrix.data<float>(); 
    float * output_matrix_ = output_matrix.data<float>();

    // Device
    int *d_t_row_offset, *d_t_column;
    int *d_t_window_row, *d_t_atomic;
    float *d_t_value; 
    // int *d_c_row_offsets, *d_c_row, *d_c_atomic, *d_c_column;
    // int *d_c_row_offsets_short, *d_c_row_short, *d_c_atomic_short, *d_c_column_short;
    // half *d_c_value, *d_c_value_short; 
	float *d_rhs_matrix;
    float *d_output_matrix;

    checkCuda(cudaMalloc(&d_t_row_offset, (t_row_offset.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_value, (t_value.size(0)) * sizeof(float)));
    checkCuda(cudaMalloc(&d_t_column, (t_column.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_window_row, (t_window_row.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_atomic, (t_atomic.size(0)) * sizeof(int)));

    checkCuda(cudaMalloc(&d_rhs_matrix, (kOri*dimN) * sizeof(float)));
    checkCuda(cudaMalloc(&d_output_matrix, (mOri*dimN) * sizeof(float)));

    //把CPU数据放到GPU上
    checkCuda(cudaMemcpy(d_t_row_offset, t_row_offset_ , (t_row_offset.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_value, t_value_, (t_value.size(0)) * sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_column, t_column_, (t_column.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_window_row, t_window_row_, (t_window_row.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_atomic, t_atomic_, (t_atomic.size(0)) * sizeof(int), cudaMemcpyHostToDevice));


    checkCuda(cudaMemcpy(d_rhs_matrix, rhs_matrix_,(kOri*dimN) * sizeof(float), cudaMemcpyHostToDevice));

    float spmm_ms_avg =  spmm_forward_tf32_tcu_bcrs_part_kernel(
      d_t_row_offset,
      d_t_value, 
      d_t_column,
      d_t_window_row,
      d_t_atomic,

      d_rhs_matrix,
      d_output_matrix,

      parts_t,

      dimN,
      mOri,
      epoches);    
    checkCuda(cudaMemcpy(output_matrix_, d_output_matrix, mOri * dimN * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_t_row_offset);
    cudaFree(d_t_value);
    cudaFree(d_t_column);
    cudaFree(d_t_window_row);
    cudaFree(d_t_atomic);

    cudaFree(d_rhs_matrix);
    cudaFree(d_output_matrix);
    cudaDeviceSynchronize(); 
    return {output_matrix, torch::tensor(spmm_ms_avg)};
}
// m-tcu + cuda + partSize + binary
std::vector<torch::Tensor> block_fp16_tcu_bcrs(torch::Tensor row1, torch::Tensor column1, torch::Tensor values1, 
    int partSize_t, int window, int wide)
    {
        auto *row=row1.data_ptr<int>();
        auto column=column1.accessor<int, 1>();
        auto values=values1.accessor<float, 1>();
    
        auto start = std::chrono::high_resolution_clock::now();
    
        int rows=row1.size(0)-1;
        // Result
        std::map<int, Value_tcu_bcrs> res_t;

        //需要的window数量
        int num_windows = rows/window;
        #pragma omp parallel for
        for(int i=0;i<num_windows;i++)
        {
            /*
                1. 统计当前window中的vector, 并按照稠密度降序排列
            */
            //对column进行合并
            std::vector<int> mergedVector(&column[row[i*window]], &column[row[i*window+window]]);
            // 使用 unordered_map 统计 vector 中每个元素出现次数
            std::unordered_map<int, int> elementCounts;
            for (const auto& element : mergedVector) elementCounts[element]++;
            // 将统计结果放入 vector<pair> 中
            std::vector<std::pair<int, int>> countVector(elementCounts.begin(), elementCounts.end());
            std::sort(countVector.begin(), countVector.end(), [](const auto& a, const auto& b) {
                return a.first < b.first;
            });
            /*
                2. 确定需要进行tcu计算的vector数量, t_percen
            */
            int size = countVector.size();
            int residue = size%wide;
            //统计每个block中的非零元数量，计算稠密度
            int t_percen = size; 
            int t_percen_ori = size;
            if((t_percen%wide) != 0)
            {
                // if((t_percen + (wide - t_percen%wide)) <= size) 
                t_percen = (t_percen + (wide - t_percen%wide));
                // else
                // t_percen = (t_percen - (t_percen%wide));
    
            }

            int blocks = t_percen/wide; 
            Value_tcu_bcrs v_tcu;
            if(blocks>0) {v_tcu.tcu_flag = true; }
            std::map<int, int> colmap;
            int c=0;
            for(auto col:countVector){
                v_tcu.t_column.push_back(col.first);
                colmap[col.first]=c++;
            }
            if((t_percen_ori%wide)>0){
                for(int tt=0;tt<(wide-(t_percen_ori%wide));tt++)
                v_tcu.t_column.push_back(-1);
            }
            
            std::vector<at::Half> demo(t_percen*window);
           
            int p=0;
            for(int j=i*window;j<(i+1)*window;j++)
            {
                //用来存储每行的cuda数据
                std::queue<int> c_col;
                std::queue<int> c_value;
                int cuda_nnz=0;
                for(int m=row[j];m<row[j+1];m++)
                {
                    int newCol = colmap[column[m]];
                    int block = newCol/wide;
                    int bInId= newCol%wide;
                    demo[block*window*wide + p*wide +bInId]=values[m];
  
                }
                p++;
            }
            //整合vector
            v_tcu.t_value = demo;
        
            if(blocks <= partSize_t)
            {
                v_tcu.row.push_back(t_percen_ori);
                v_tcu.row.push_back(t_percen-t_percen_ori);
                v_tcu.atomic.push_back(0);
                v_tcu.window.push_back(i);
            }else{
                // 对block太多的块进行分割
                int part_number = blocks / partSize_t;
                int block_residue = blocks % partSize_t;
                if(block_residue > 0)
                {
                    for(int j=0; j<part_number; j++)
                    {
                        v_tcu.row.push_back(partSize_t*wide);
                        v_tcu.row.push_back(0);
                        v_tcu.window.push_back(i);
                        v_tcu.atomic.push_back(1);
                    }
                    v_tcu.row.push_back(block_residue*wide - (t_percen-t_percen_ori));
                    v_tcu.row.push_back(t_percen-t_percen_ori);
                    v_tcu.window.push_back(i);
                    v_tcu.atomic.push_back(1);
                }else{
                    for(int j=0; j<part_number; j++)
                    {
                        //如果是最后一组block
                        if(j == (part_number-1)){
                            v_tcu.row.push_back((partSize_t*wide) - (t_percen-t_percen_ori));
                            v_tcu.row.push_back(t_percen-t_percen_ori);
                        }else{
                            v_tcu.row.push_back(partSize_t*wide);
                            v_tcu.row.push_back(0);
                        }
                        v_tcu.window.push_back(i);
                        v_tcu.atomic.push_back(1);
                    }
                }
            }
            /*
                4. 汇总三部分的结果
            */
            #pragma omp critical  
            res_t[i]=v_tcu;
        
        }
        /*
            5. 整合所有window的数据
        */
        //tcu
        std::vector<int> t_rowkNew_offset;
        t_rowkNew_offset.push_back(0);
        std::vector<at::Half> t_valueNew;
        std::vector<int> t_columnNew;
        std::vector<int> t_window_rowNew;
        std::vector<int> t_atomicNew;
    
        //按顺序整合res
        //tcu
        for (const auto& pair : res_t) {
            if (pair.second.tcu_flag == true )
            {
                for(int sub : pair.second.row)
                {
                    t_rowkNew_offset.push_back(t_rowkNew_offset.back()+sub);
                }   
                t_valueNew.insert(t_valueNew.end(),pair.second.t_value.begin(),pair.second.t_value.end());
                t_columnNew.insert(t_columnNew.end(),pair.second.t_column.begin(),pair.second.t_column.end());
                t_window_rowNew.insert(t_window_rowNew.end(),pair.second.window.begin(),pair.second.window.end());
                t_atomicNew.insert(t_atomicNew.end(),pair.second.atomic.begin(),pair.second.atomic.end());
            }   
        }
    
    
        auto end = std::chrono::high_resolution_clock::now();
    
        // 计算耗时
        std::chrono::duration<double> elapsed = end - start;
    
        std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;
    
        auto t_rowNew_offsetTensor1 = torch::from_blob(t_rowkNew_offset.data(), t_rowkNew_offset.size(), torch::kInt32);
        auto t_valueTensor1 = torch::from_blob(t_valueNew.data(), t_valueNew.size(), torch::kFloat16);
        auto t_columnTensor1 = torch::from_blob(t_columnNew.data(), t_columnNew.size(), torch::kInt32);
        auto t_window_rowTensor1 = torch::from_blob(t_window_rowNew.data(), t_window_rowNew.size(), torch::kInt32);
        auto t_atomicTensor1 = torch::from_blob(t_atomicNew.data(), t_atomicNew.size(), torch::kInt32);
    
        torch::Tensor t_rowNew_offsetTensor = torch::empty_like(t_rowNew_offsetTensor1);
        t_rowNew_offsetTensor.copy_(t_rowNew_offsetTensor1);
        torch::Tensor t_valueTensor = torch::empty_like(t_valueTensor1);
        t_valueTensor.copy_(t_valueTensor1);
        torch::Tensor t_columnTensor = torch::empty_like(t_columnTensor1);
        t_columnTensor.copy_(t_columnTensor1);
        torch::Tensor t_window_rowTensor = torch::empty_like(t_window_rowTensor1);
        t_window_rowTensor.copy_(t_window_rowTensor1);
        torch::Tensor t_atomicTensor = torch::empty_like(t_atomicTensor1);
        t_atomicTensor.copy_(t_atomicTensor1);
    
    
        return {t_rowNew_offsetTensor,t_columnTensor, t_valueTensor, t_window_rowTensor, t_atomicTensor};
    
    }



    void gcn_forward_fp16_tcu_cuda_bcrs_kernel(
        int * t_row_offset,
        half * t_value, 
        int * t_column, 
        int* t_window_row,
        int * t_atomic,
    
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
        const int mOri);
    
    
    std::vector<torch::Tensor> gcn_forward_fp16_tcu_cuda_bcrs(
        torch::Tensor t_row_offset,
        torch::Tensor t_column, 
        torch::Tensor t_value, 
        torch::Tensor t_window_row,
        torch::Tensor t_atomic,
    
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
    
        torch::Tensor rhs_matrix,
    
        const int parts_t,
        const int parts_c,
        const int partsize_c,
        const int parts_c_short,
        const int dimN,
        const int mOri,
        const int kOri)
    {
        auto output_matrix = torch::zeros({mOri,dimN}, torch::kCUDA).to(torch::kFloat32);
        
        //把CPU端的tensor转成C++的数据结构
        int * t_row_offset_ = t_row_offset.data<int>();
        half * t_value_ = reinterpret_cast<half *>(t_value.data<at::Half>()); 
        int * t_column_ = t_column.data<int>();
        int * t_window_row_ = t_window_row.data<int>();
        int * t_atomic_ = t_atomic.data<int>();
        
        int * c_row_offsets_ = c_row_offsets.data<int>();
        int * c_row_ = c_row.data<int>();
        int * c_atomic_ = c_atomic.data<int>();
        int * c_column_ = c_column.data<int>();
        half * c_value_ = reinterpret_cast<half *>(c_value.data<at::Half>()); 
    
        int * c_row_offsets_short_ = c_row_offsets_short.data<int>();
        int * c_row_short_ = c_row_short.data<int>();
        int * c_atomic_short_ = c_atomic_short.data<int>();
        int * c_column_short_ = c_column_short.data<int>();
        half * c_value_short_ = reinterpret_cast<half *>(c_value_short.data<at::Half>()); 
    
        half * rhs_matrix_ = reinterpret_cast<half *>(rhs_matrix.data<at::Half>()); 
        float * output_matrix_ = output_matrix.data<float>();
    
        
        gcn_forward_fp16_tcu_cuda_bcrs_kernel(
        t_row_offset_,
        t_value_, 
        t_column_,
        t_window_row_,
        t_atomic_,
        
        c_row_offsets_, 
        c_row_, 
        c_atomic_,
        c_column_,
        c_value_,

        c_row_offsets_short_, 
        c_row_short_, 
        c_atomic_short_,
        c_column_short_,
        c_value_short_,

        rhs_matrix_,
        output_matrix_,
    
        parts_t,
        parts_c,
        partsize_c,
        parts_c_short,
        dimN,
        mOri);
          
        return {output_matrix};
    }

    float spmm_forward_tf32_tcu_tcf_part_kernel(
        int * t_row_offset,
        int * t_tco, 
        int * t_tcl,
        float * t_value, 
        int * t_column, 
        int* t_window_row,
        int * t_atomic,
    
        float * rhs_matrix,
        float * output_matrix,
    
        const int parts_t,
        const int dimN,
        const int mOri,
        int epoches);
    
    std::vector<torch::Tensor> spmm_forward_tf32_tcu_tcf_part(
        torch::Tensor t_row_offset,
        torch::Tensor t_tco, 
        torch::Tensor t_tcl, 
        torch::Tensor t_column, 
        torch::Tensor t_value, 
        torch::Tensor t_window_row,
        torch::Tensor t_atomic,
    
        torch::Tensor rhs_matrix,
    
        const int parts_t,
    
        const int dimN,
        const int mOri,
        const int kOri,
        int epoches)
    {
        auto output_matrix = torch::zeros({mOri, dimN}, torch::kFloat32).to(torch::kCPU);
        
        //把CPU端的tensor转成C++的数据结构
        int * t_row_offset_ = t_row_offset.data<int>();
        int * t_tco_ = t_tco.data<int>();
        int * t_tcl_ = t_tcl.data<int>();
        float * t_value_ = t_value.data<float>(); 
        int * t_column_ = t_column.data<int>();
        int * t_window_row_ = t_window_row.data<int>();
        int * t_atomic_ = t_atomic.data<int>();
    
        float * rhs_matrix_ = rhs_matrix.data<float>(); 
        float * output_matrix_ = output_matrix.data<float>();
    
        // Device
        int *d_t_row_offset, *d_t_column;
        int *d_t_tco, *d_t_tcl;
        int *d_t_window_row, *d_t_atomic;
        float *d_t_value; 

        float *d_rhs_matrix;
        float *d_output_matrix;
    
        checkCuda(cudaMalloc(&d_t_row_offset, (t_row_offset.size(0)) * sizeof(int)));
        checkCuda(cudaMalloc(&d_t_tco, (t_tco.size(0)) * sizeof(int)));
        checkCuda(cudaMalloc(&d_t_tcl, (t_tcl.size(0)) * sizeof(int)));
        checkCuda(cudaMalloc(&d_t_value, (t_value.size(0)) * sizeof(float)));
        checkCuda(cudaMalloc(&d_t_column, (t_column.size(0)) * sizeof(int)));
        checkCuda(cudaMalloc(&d_t_window_row, (t_window_row.size(0)) * sizeof(int)));
        checkCuda(cudaMalloc(&d_t_atomic, (t_atomic.size(0)) * sizeof(int)));
    
        checkCuda(cudaMalloc(&d_rhs_matrix, (kOri*dimN) * sizeof(float)));
        checkCuda(cudaMalloc(&d_output_matrix, (mOri*dimN) * sizeof(float)));
    
        //把CPU数据放到GPU上
        checkCuda(cudaMemcpy(d_t_row_offset, t_row_offset_ , (t_row_offset.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(d_t_tco, t_tco_, (t_tco.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(d_t_tcl, t_tcl_, (t_tcl.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(d_t_value, t_value_, (t_value.size(0)) * sizeof(float), cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(d_t_column, t_column_, (t_column.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(d_t_window_row, t_window_row_, (t_window_row.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(d_t_atomic, t_atomic_, (t_atomic.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    
    
        checkCuda(cudaMemcpy(d_rhs_matrix, rhs_matrix_,(kOri*dimN) * sizeof(float), cudaMemcpyHostToDevice));
    
        float spmm_ms_avg =  spmm_forward_tf32_tcu_tcf_part_kernel(
          d_t_row_offset,
          d_t_tco,
          d_t_tcl,
          d_t_value, 
          d_t_column,
          d_t_window_row,
          d_t_atomic,
    
          d_rhs_matrix,
          d_output_matrix,
    
          parts_t,
    
          dimN,
          mOri,
          epoches);    
        checkCuda(cudaMemcpy(output_matrix_, d_output_matrix, mOri * dimN * sizeof(float), cudaMemcpyDeviceToHost));
        cudaFree(d_t_row_offset);
        cudaFree(d_t_tco);
        cudaFree(d_t_tcl);
        cudaFree(d_t_value);
        cudaFree(d_t_column);
        cudaFree(d_t_window_row);
        cudaFree(d_t_atomic);
    
        cudaFree(d_rhs_matrix);
        cudaFree(d_output_matrix);
        cudaDeviceSynchronize(); 
        return {output_matrix, torch::tensor(spmm_ms_avg)};
    }


std::vector<std::vector<int>> createVectorOfVectors(int numVectors) {
    std::vector<std::vector<int>> vectorOfVectors;

    // 在向量数组中放入指定数量的空向量
    for (int i = 0; i < numVectors; ++i) {
        std::vector<int> emptyVector;
        vectorOfVectors.push_back(emptyVector);
    }

    return vectorOfVectors;
}

    // 合并vector
std::vector<int> mergeVectors(const std::vector<std::vector<int>>& vectorOfVectors) {
    std::vector<int> mergedVector;

    // 遍历向量数组中的每个向量，并将其元素添加到合并后的向量中
    for (const auto& vec : vectorOfVectors) {
        for (int num : vec) {
            mergedVector.push_back(num);
        }
    }

    return mergedVector;
}
std::vector<torch::Tensor> block_tf32_tcu_tcf(torch::Tensor row1, torch::Tensor column1, torch::Tensor values1, 
    int partSize_t, int window, int wide)
    {
        auto *row=row1.data_ptr<int>();
        auto column=column1.accessor<int, 1>();
        auto values=values1.accessor<float, 1>();
    
        auto start = std::chrono::high_resolution_clock::now();
    
        int rows=row1.size(0)-1;
        // Result
        std::map<int, Value_tcu_tcf> res_t;

        //需要的window数量
        int num_windows = rows/window;
        #pragma omp parallel for
        for(int i=0;i<num_windows;i++)
        {
            /*
                1. 统计当前window中的vector, 并按照稠密度降序排列
            */
            //对column进行合并
            std::vector<int> mergedVector(&column[row[i*window]], &column[row[i*window+window]]);
            // 使用 unordered_map 统计 vector 中每个元素出现次数
            std::unordered_map<int, int> elementCounts;
            for (const auto& element : mergedVector) elementCounts[element]++;
            // 将统计结果放入 vector<pair> 中
            std::vector<std::pair<int, int>> countVector(elementCounts.begin(), elementCounts.end());
            std::sort(countVector.begin(), countVector.end(), [](const auto& a, const auto& b) {
                return a.first < b.first;
            });
            /*
                2. 确定需要进行tcu计算的vector数量, t_percen
            */
            int size = countVector.size();
            int residue = size%wide;
            //统计每个block中的非零元数量，计算稠密度
            int t_percen = size; 
            int t_percen_ori = size;
            if((t_percen%wide) != 0)
            {
                // if((t_percen + (wide - t_percen%wide)) <= size) 
                t_percen = (t_percen + (wide - t_percen%wide));
                // else
                // t_percen = (t_percen - (t_percen%wide));
    
            }

            int blocks = t_percen/wide; 
            std::vector<std::vector<float>> t_value_temp = createVectorOfVectors_float(blocks);
            std::vector<std::vector<int>> t_row_temp = createVectorOfVectors(blocks);
            Value_tcu_tcf v_tcu;
            if(blocks>0) {v_tcu.tcu_flag = true; }
            std::map<int, int> colmap;
            int c=0;
            for(auto col:countVector){
                v_tcu.t_column.push_back(col.first);
                colmap[col.first]=c++;
            }
            if((t_percen_ori%wide)>0){
                for(int tt=0;tt<(wide-(t_percen_ori%wide));tt++)
                v_tcu.t_column.push_back(-1);
            }
           
            int p=0;
            for(int j=i*window;j<(i+1)*window;j++)
            {
                //用来存储每行的cuda数据
                std::queue<int> c_col;
                std::queue<int> c_value;
                int cuda_nnz=0;
                for(int m=row[j];m<row[j+1];m++)
                {
                    int newCol = colmap[column[m]];
                    int block = newCol/wide;
                    t_value_temp[block].push_back(values[m]);
                    t_row_temp[block].push_back((p*wide) + (newCol%wide));
                }
                p++;
            }
            //整合vector
            v_tcu.t_value = mergeVectors_float(t_value_temp);
            v_tcu.tcl = mergeVectors(t_row_temp);
            v_tcu.t_block_offset = countVectors(t_value_temp);
            if(blocks <= partSize_t)
            {
                v_tcu.row.push_back(blocks);
                v_tcu.atomic.push_back(0);
                v_tcu.window.push_back(i);
            }else{
                // 对block太多的块进行分割
                int part_number = blocks / partSize_t;
                int block_residue = blocks % partSize_t;
                if(block_residue > 0)
                {
                    for(int j=0; j<part_number; j++)
                    {
                        v_tcu.row.push_back(partSize_t);
                        v_tcu.window.push_back(i);
                        v_tcu.atomic.push_back(1);
                    }
                    v_tcu.row.push_back(block_residue);
                    v_tcu.window.push_back(i);
                    v_tcu.atomic.push_back(1);
                }else{
                    for(int j=0; j<part_number; j++)
                    {
                        //如果是最后一组block
                        v_tcu.row.push_back(partSize_t);
                        v_tcu.window.push_back(i);
                        v_tcu.atomic.push_back(1);
                    }
                }
            }
            /*
                4. 汇总三部分的结果
            */
            #pragma omp critical  
            res_t[i]=v_tcu;
        
        }
        /*
            5. 整合所有window的数据
        */
        //tcu
        std::vector<int> t_rowkNew_offset;
        t_rowkNew_offset.push_back(0);
        std::vector<int> t_tco; //TCO
        t_tco.push_back(0);
        std::vector<int> t_tcl; //TCL
        std::vector<float> t_valueNew;
        std::vector<int> t_columnNew;
        std::vector<int> t_window_rowNew;
        std::vector<int> t_atomicNew;
    
        //按顺序整合res
        //tcu
        for (const auto& pair : res_t) {
            if (pair.second.tcu_flag == true )
            {
                for(int sub : pair.second.row)
                {
                    t_rowkNew_offset.push_back(t_rowkNew_offset.back()+sub);
                }  
                for(int sub : pair.second.t_block_offset)
                { 
                    t_tco.push_back(t_tco.back()+sub);
                }
                t_tcl.insert(t_tcl.end(),pair.second.tcl.begin(),pair.second.tcl.end());
                t_valueNew.insert(t_valueNew.end(),pair.second.t_value.begin(),pair.second.t_value.end());
                t_columnNew.insert(t_columnNew.end(),pair.second.t_column.begin(),pair.second.t_column.end());
                t_window_rowNew.insert(t_window_rowNew.end(),pair.second.window.begin(),pair.second.window.end());
                t_atomicNew.insert(t_atomicNew.end(),pair.second.atomic.begin(),pair.second.atomic.end());
            }   
        }
    
    
        auto end = std::chrono::high_resolution_clock::now();
    
        // 计算耗时
        std::chrono::duration<double> elapsed = end - start;
    
        std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;
    
        auto t_rowNew_offsetTensor1 = torch::from_blob(t_rowkNew_offset.data(), t_rowkNew_offset.size(), torch::kInt32);
        auto t_tcoNewTensor1 = torch::from_blob(t_tco.data(), t_tco.size(), torch::kInt32);       
        auto t_tclNewTensor1 = torch::from_blob(t_tcl.data(), t_tcl.size(), torch::kInt32);     
        auto t_valueTensor1 = torch::from_blob(t_valueNew.data(), t_valueNew.size(), torch::kFloat32);
        auto t_columnTensor1 = torch::from_blob(t_columnNew.data(), t_columnNew.size(), torch::kInt32);
        auto t_window_rowTensor1 = torch::from_blob(t_window_rowNew.data(), t_window_rowNew.size(), torch::kInt32);
        auto t_atomicTensor1 = torch::from_blob(t_atomicNew.data(), t_atomicNew.size(), torch::kInt32);
    
        torch::Tensor t_rowNew_offsetTensor = torch::empty_like(t_rowNew_offsetTensor1);
        t_rowNew_offsetTensor.copy_(t_rowNew_offsetTensor1);
        torch::Tensor t_tcoNewTensor = torch::empty_like(t_tcoNewTensor1);
        t_tcoNewTensor.copy_(t_tcoNewTensor1);
        torch::Tensor t_tclNewTensor = torch::empty_like(t_tclNewTensor1);
        t_tclNewTensor.copy_(t_tclNewTensor1);
        torch::Tensor t_valueTensor = torch::empty_like(t_valueTensor1);
        t_valueTensor.copy_(t_valueTensor1);
        torch::Tensor t_columnTensor = torch::empty_like(t_columnTensor1);
        t_columnTensor.copy_(t_columnTensor1);
        torch::Tensor t_window_rowTensor = torch::empty_like(t_window_rowTensor1);
        t_window_rowTensor.copy_(t_window_rowTensor1);
        torch::Tensor t_atomicTensor = torch::empty_like(t_atomicTensor1);
        t_atomicTensor.copy_(t_atomicTensor1);
    
    
        return {t_rowNew_offsetTensor, t_tcoNewTensor, t_tclNewTensor, t_columnTensor, t_valueTensor, t_window_rowTensor, t_atomicTensor};
    
    }



float spmm_forward_fp16_tcu_cuda_nomap_kernel(
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
    int epoches);


std::vector<torch::Tensor> spmm_forward_fp16_tcu_cuda_nomap(
    torch::Tensor t_row_offset,
    torch::Tensor t_blockNew_offset, 
    torch::Tensor t_column, 
    torch::Tensor t_value, 
    torch::Tensor t_window_row,
    torch::Tensor t_atomic,
    torch::Tensor t_binary, 

    torch::Tensor rhs_matrix,

    const int parts_t,

    const int dimN,
    const int mOri,
    const int kOri,
    int epoches)
{
    auto output_matrix = torch::zeros({mOri, dimN}, torch::kFloat32).to(torch::kCPU);
    
    //把CPU端的tensor转成C++的数据结构
    int * t_row_offset_ = t_row_offset.data<int>();
    int * t_blockNew_offset_ = t_blockNew_offset.data<int>();
    half * t_value_ = reinterpret_cast<half *>(t_value.data<at::Half>()); 
    int * t_column_ = t_column.data<int>();
    int * t_window_row_ = t_window_row.data<int>();
    int * t_atomic_ = t_atomic.data<int>();
    long * t_binary_ = t_binary.data<long>();

    half * rhs_matrix_ = reinterpret_cast<half *>(rhs_matrix.data<at::Half>()); 
    float * output_matrix_ = output_matrix.data<float>();

    // Device
    int *d_t_row_offset, *d_t_blockNew_offset, *d_t_column;
    int *d_t_window_row, *d_t_atomic;
    long *d_t_binary;
    half *d_t_value; 
	half *d_rhs_matrix;
    float *d_output_matrix;

    checkCuda(cudaMalloc(&d_t_row_offset, (t_row_offset.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_blockNew_offset, (t_blockNew_offset.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_value, (t_value.size(0)) * sizeof(half)));
    checkCuda(cudaMalloc(&d_t_column, (t_column.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_window_row, (t_window_row.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_atomic, (t_atomic.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_binary, (t_binary.size(0)) * sizeof(long)));

    checkCuda(cudaMalloc(&d_rhs_matrix, (kOri*dimN) * sizeof(half)));
    checkCuda(cudaMalloc(&d_output_matrix, (mOri*dimN) * sizeof(float)));

    //把CPU数据放到GPU上
    checkCuda(cudaMemcpy(d_t_row_offset, t_row_offset_ , (t_row_offset.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_blockNew_offset, t_blockNew_offset_ , (t_blockNew_offset.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_value, t_value_, (t_value.size(0)) * sizeof(half), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_column, t_column_, (t_column.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_window_row, t_window_row_, (t_window_row.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_atomic, t_atomic_, (t_atomic.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_binary, t_binary_, (t_binary.size(0)) * sizeof(long), cudaMemcpyHostToDevice));

    checkCuda(cudaMemcpy(d_rhs_matrix, rhs_matrix_,(kOri*dimN) * sizeof(half), cudaMemcpyHostToDevice));
    float spmm_ms_avg = 0.0;
    
      spmm_ms_avg =  spmm_forward_fp16_tcu_cuda_nomap_kernel(
      d_t_row_offset,
      d_t_blockNew_offset,
      d_t_value, 
      d_t_column,
      d_t_window_row,
      d_t_atomic,
      d_t_binary,

      d_rhs_matrix,
      d_output_matrix,

      parts_t,

      dimN,
      mOri,
      epoches);
      
    checkCuda(cudaMemcpy(output_matrix_, d_output_matrix, mOri * dimN * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_t_row_offset);
    cudaFree(d_t_blockNew_offset);
    cudaFree(d_t_value);
    cudaFree(d_t_column);
    cudaFree(d_t_window_row);
    cudaFree(d_t_atomic);
    cudaFree(d_t_binary);

    cudaFree(d_rhs_matrix);
    cudaFree(d_output_matrix);
    cudaDeviceSynchronize(); 
    return {output_matrix, torch::tensor(spmm_ms_avg)};
}




float spmm_forward_tf32_tcu_cuda_bcrs_kernel(
    int * t_row_offset,
    float * t_value, 
    int * t_column, 
    int* t_window_row,
    int * t_atomic,

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
    int epoches);

std::vector<torch::Tensor> spmm_forward_tf32_tcu_cuda_bcrs(
    torch::Tensor t_row_offset,
    torch::Tensor t_column, 
    torch::Tensor t_value, 
    torch::Tensor t_window_row,
    torch::Tensor t_atomic,


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

    torch::Tensor rhs_matrix,

    const int parts_t,
    const int parts_c,
    const int partsize_c,
    const int parts_c_short,
    const int dimN,
    const int mOri,
    const int kOri,
    int epoches)
{
    auto output_matrix = torch::zeros({mOri, dimN}, torch::kFloat32).to(torch::kCPU);
    
    //把CPU端的tensor转成C++的数据结构
    int * t_row_offset_ = t_row_offset.data<int>();
    float * t_value_ = t_value.data<float>(); 
    int * t_column_ = t_column.data<int>();
    int * t_window_row_ = t_window_row.data<int>();
    int * t_atomic_ = t_atomic.data<int>();
    
    int * c_row_offsets_ = c_row_offsets.data<int>();
    int * c_row_ = c_row.data<int>();
    int * c_atomic_ = c_atomic.data<int>();
    int * c_column_ = c_column.data<int>();
    float * c_value_ = c_value.data<float>(); 

    int * c_row_offsets_short_ = c_row_offsets_short.data<int>();
    int * c_row_short_ = c_row_short.data<int>();
    int * c_atomic_short_ = c_atomic_short.data<int>();
    int * c_column_short_ = c_column_short.data<int>();
    float * c_value_short_ = c_value_short.data<float>(); 

    float * rhs_matrix_ = rhs_matrix.data<float>(); 
    float * output_matrix_ = output_matrix.data<float>();

    // Device
    int *d_t_row_offset, *d_t_column;
    int *d_t_window_row, *d_t_atomic;
    float *d_t_value; 
    int *d_c_row_offsets, *d_c_row, *d_c_atomic, *d_c_column;
    int *d_c_row_offsets_short, *d_c_row_short, *d_c_atomic_short, *d_c_column_short;
    float *d_c_value, *d_c_value_short; 
	float *d_rhs_matrix;
    float *d_output_matrix;

    checkCuda(cudaMalloc(&d_t_row_offset, (t_row_offset.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_value, (t_value.size(0)) * sizeof(float)));
    checkCuda(cudaMalloc(&d_t_column, (t_column.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_window_row, (t_window_row.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_atomic, (t_atomic.size(0)) * sizeof(int)));

    checkCuda(cudaMalloc(&d_c_row_offsets, (c_row_offsets.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_row, (c_row.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_atomic, (c_atomic.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_column, (c_column.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_value, (c_value.size(0)) * sizeof(float)));

    checkCuda(cudaMalloc(&d_c_row_offsets_short, (c_row_offsets_short.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_row_short, (c_row_short.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_atomic_short, (c_atomic_short.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_column_short, (c_column_short.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_value_short, (c_value_short.size(0)) * sizeof(float)));

    checkCuda(cudaMalloc(&d_rhs_matrix, (kOri*dimN) * sizeof(float)));
    checkCuda(cudaMalloc(&d_output_matrix, (mOri*dimN) * sizeof(float)));

    //把CPU数据放到GPU上
    checkCuda(cudaMemcpy(d_t_row_offset, t_row_offset_ , (t_row_offset.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_value, t_value_, (t_value.size(0)) * sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_column, t_column_, (t_column.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_window_row, t_window_row_, (t_window_row.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_atomic, t_atomic_, (t_atomic.size(0)) * sizeof(int), cudaMemcpyHostToDevice));

    checkCuda(cudaMemcpy(d_c_row_offsets, c_row_offsets_ , (c_row_offsets.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_row, c_row_, (c_row.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_atomic, c_atomic_, (c_atomic.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_column, c_column_, (c_column.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_value, c_value_, (c_value.size(0)) * sizeof(float), cudaMemcpyHostToDevice));

    checkCuda(cudaMemcpy(d_c_row_offsets_short, c_row_offsets_short_ , (c_row_offsets_short.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_row_short, c_row_short_, (c_row_short.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_atomic_short, c_atomic_short_, (c_atomic_short.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_column_short, c_column_short_, (c_column_short.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_value_short, c_value_short_, (c_value_short.size(0)) * sizeof(float), cudaMemcpyHostToDevice));

    checkCuda(cudaMemcpy(d_rhs_matrix, rhs_matrix_,(kOri*dimN) * sizeof(float), cudaMemcpyHostToDevice));
    float spmm_ms_avg = 0.0;

    spmm_ms_avg =  spmm_forward_tf32_tcu_cuda_bcrs_kernel(
    d_t_row_offset,
    d_t_value, 
    d_t_column,
    d_t_window_row,
    d_t_atomic,
    
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

    d_rhs_matrix,
    d_output_matrix,

    parts_t,
    parts_c,
    partsize_c,
    parts_c_short,
    dimN,
    mOri,
    epoches);

    checkCuda(cudaMemcpy(output_matrix_, d_output_matrix, mOri * dimN * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_t_row_offset);
    cudaFree(d_t_value);
    cudaFree(d_t_column);
    cudaFree(d_t_window_row);
    cudaFree(d_t_atomic);

    cudaFree(d_c_row_offsets);
    cudaFree(d_c_row);
    cudaFree(d_c_atomic);
    cudaFree(d_c_column);
    cudaFree(d_c_value);

    cudaFree(d_c_row_offsets_short);
    cudaFree(d_c_row_short);
    cudaFree(d_c_atomic_short);
    cudaFree(d_c_column_short);
    cudaFree(d_c_value_short);

    cudaFree(d_rhs_matrix);
    cudaFree(d_output_matrix);
    // cudaDeviceSynchronize(); 
    return {output_matrix, torch::tensor(spmm_ms_avg)};
}


float spmm_forward_tf32_tcu_cuda_bcrs_kernel_seq(
    int * t_row_offset,
    float * t_value, 
    int * t_column, 
    int* t_window_row,
    int * t_atomic,
    
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
    int epoches);

std::vector<torch::Tensor> spmm_forward_tf32_tcu_cuda_bcrs_seq(
    torch::Tensor t_row_offset,
    torch::Tensor t_column, 
    torch::Tensor t_value, 
    torch::Tensor t_window_row,
    torch::Tensor t_atomic,


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

    torch::Tensor rhs_matrix,

    const int parts_t,
    const int parts_c,
    const int partsize_c,
    const int parts_c_short,
    const int dimN,
    const int mOri,
    const int kOri,
    int epoches)
{
    auto output_matrix = torch::zeros({mOri, dimN}, torch::kFloat32).to(torch::kCPU);
    
    //把CPU端的tensor转成C++的数据结构
    int * t_row_offset_ = t_row_offset.data<int>();
    float * t_value_ = t_value.data<float>(); 
    int * t_column_ = t_column.data<int>();
    int * t_window_row_ = t_window_row.data<int>();
    int * t_atomic_ = t_atomic.data<int>();
    
    int * c_row_offsets_ = c_row_offsets.data<int>();
    int * c_row_ = c_row.data<int>();
    int * c_atomic_ = c_atomic.data<int>();
    int * c_column_ = c_column.data<int>();
    float * c_value_ = c_value.data<float>(); 

    int * c_row_offsets_short_ = c_row_offsets_short.data<int>();
    int * c_row_short_ = c_row_short.data<int>();
    int * c_atomic_short_ = c_atomic_short.data<int>();
    int * c_column_short_ = c_column_short.data<int>();
    float * c_value_short_ = c_value_short.data<float>(); 

    float * rhs_matrix_ = rhs_matrix.data<float>(); 
    float * output_matrix_ = output_matrix.data<float>();

    // Device
    int *d_t_row_offset, *d_t_column;
    int *d_t_window_row, *d_t_atomic;
    float *d_t_value; 
    int *d_c_row_offsets, *d_c_row, *d_c_atomic, *d_c_column;
    int *d_c_row_offsets_short, *d_c_row_short, *d_c_atomic_short, *d_c_column_short;
    float *d_c_value, *d_c_value_short; 
	float *d_rhs_matrix;
    float *d_output_matrix;

    checkCuda(cudaMalloc(&d_t_row_offset, (t_row_offset.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_value, (t_value.size(0)) * sizeof(float)));
    checkCuda(cudaMalloc(&d_t_column, (t_column.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_window_row, (t_window_row.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_atomic, (t_atomic.size(0)) * sizeof(int)));

    checkCuda(cudaMalloc(&d_c_row_offsets, (c_row_offsets.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_row, (c_row.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_atomic, (c_atomic.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_column, (c_column.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_value, (c_value.size(0)) * sizeof(float)));

    checkCuda(cudaMalloc(&d_c_row_offsets_short, (c_row_offsets_short.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_row_short, (c_row_short.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_atomic_short, (c_atomic_short.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_column_short, (c_column_short.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_value_short, (c_value_short.size(0)) * sizeof(float)));

    checkCuda(cudaMalloc(&d_rhs_matrix, (kOri*dimN) * sizeof(float)));
    checkCuda(cudaMalloc(&d_output_matrix, (mOri*dimN) * sizeof(float)));

    //把CPU数据放到GPU上
    checkCuda(cudaMemcpy(d_t_row_offset, t_row_offset_ , (t_row_offset.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_value, t_value_, (t_value.size(0)) * sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_column, t_column_, (t_column.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_window_row, t_window_row_, (t_window_row.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_atomic, t_atomic_, (t_atomic.size(0)) * sizeof(int), cudaMemcpyHostToDevice));

    checkCuda(cudaMemcpy(d_c_row_offsets, c_row_offsets_ , (c_row_offsets.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_row, c_row_, (c_row.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_atomic, c_atomic_, (c_atomic.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_column, c_column_, (c_column.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_value, c_value_, (c_value.size(0)) * sizeof(float), cudaMemcpyHostToDevice));

    checkCuda(cudaMemcpy(d_c_row_offsets_short, c_row_offsets_short_ , (c_row_offsets_short.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_row_short, c_row_short_, (c_row_short.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_atomic_short, c_atomic_short_, (c_atomic_short.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_column_short, c_column_short_, (c_column_short.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_value_short, c_value_short_, (c_value_short.size(0)) * sizeof(float), cudaMemcpyHostToDevice));

    checkCuda(cudaMemcpy(d_rhs_matrix, rhs_matrix_,(kOri*dimN) * sizeof(float), cudaMemcpyHostToDevice));
    float spmm_ms_avg = 0.0;

    spmm_ms_avg =  spmm_forward_tf32_tcu_cuda_bcrs_kernel_seq(
    d_t_row_offset,
    d_t_value, 
    d_t_column,
    d_t_window_row,
    d_t_atomic,
    
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

    d_rhs_matrix,
    d_output_matrix,

    parts_t,
    parts_c,
    partsize_c,
    parts_c_short,
    dimN,
    mOri,
    epoches);

    checkCuda(cudaMemcpy(output_matrix_, d_output_matrix, mOri * dimN * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_t_row_offset);
    cudaFree(d_t_value);
    cudaFree(d_t_column);
    cudaFree(d_t_window_row);
    cudaFree(d_t_atomic);

    cudaFree(d_c_row_offsets);
    cudaFree(d_c_row);
    cudaFree(d_c_atomic);
    cudaFree(d_c_column);
    cudaFree(d_c_value);

    cudaFree(d_c_row_offsets_short);
    cudaFree(d_c_row_short);
    cudaFree(d_c_atomic_short);
    cudaFree(d_c_column_short);
    cudaFree(d_c_value_short);

    cudaFree(d_rhs_matrix);
    cudaFree(d_output_matrix);
    // cudaDeviceSynchronize(); 
    return {output_matrix, torch::tensor(spmm_ms_avg)};
}

std::vector<torch::Tensor> block_fp16_tcu_cuda_bcrs_hcspmm(torch::Tensor row1, torch::Tensor column1, torch::Tensor values1, 
    int partSize_t, int partSize_c, int shortSize, int density, int window, int wide)
    {
        auto *row=row1.data_ptr<int>();
        auto column=column1.accessor<int, 1>();
        auto values=values1.accessor<float, 1>();
    
        auto start = std::chrono::high_resolution_clock::now();
    
        int rows=row1.size(0)-1;
        // Result
        std::map<int, Value_tcu_bcrs> res_t;
        std::map<int, std::vector<Value_cuda>> res_c;
        std::map<int, std::vector<Value_cuda>> res_c_residue;
        //需要的window数量
        int num_windows = rows/window;
        #pragma omp parallel for
        for(int i=0;i<num_windows;i++)
        {
            /*
                1. 统计当前window中的vector, 并按照稠密度降序排列
            */
            //对column进行合并
            std::vector<int> mergedVector(&column[row[i*window]], &column[row[i*window+window]]);
            // 使用 unordered_map 统计 vector 中每个元素出现次数
            std::unordered_map<int, int> elementCounts;
            for (const auto& element : mergedVector) elementCounts[element]++;
            // 将统计结果放入 vector<pair> 中
            std::vector<std::pair<int, int>> countVector(elementCounts.begin(), elementCounts.end());
            // 按元素个数降序排序
            std::sort(countVector.begin(), countVector.end(), [](const auto& a, const auto& b) {
                return a.second > b.second;
            });
    
            /*
                2. 确定需要进行tcu计算的vector数量, t_percen
            */
            //统计每个block中的非零元数量，计算稠密度
            int size = countVector.size();
            int residue = size%wide;
            int t_percen = 0; //确定前多少个vector会被用作tcu进行稠密计算
            int num_window_edges = row[i*window+window] - row[i*window];
            int num = (size + (wide - size%wide)) / wide;
            // 0:cuda 1:tcu
            bool if_tcu = (float)size * 0.19854024 - ((float)num_window_edges / (num * 16 * 8)) * 6.578043 - 3.14922857 ? 0:1;
            if(if_tcu){
                t_percen = size;
            }else{
                t_percen = 0;
            }
            int t_percen_ori = t_percen;
            if((t_percen%wide) != 0)
            {
                // if((t_percen + (wide - t_percen%wide)) <= size) 
                t_percen = (t_percen + (wide - t_percen%wide));
                // else
                // t_percen = (t_percen - (t_percen%wide));
    
            }

            // if(t_percen<=size){
            // // 将前 k 个数按升序排列
            // std::sort(countVector.begin(), countVector.begin() + t_percen);
            // // 将剩下的数按升序排列
            // std::sort(countVector.begin() + t_percen, countVector.end());
            // }else{
            //     std::sort(countVector.begin(), countVector.end());
            // }
            std::sort(countVector.begin(), countVector.begin()+t_percen_ori);
            std::sort(countVector.begin() + t_percen_ori, countVector.end());
            /*
                3. 根据t_percen分别划分TCU和CUDA的负载
            */
    
            // TCU 划分
            // 3.1 根据partSize_t划分t_part
            //当前window中有几个block
            int blocks = t_percen/wide; 
            Value_tcu_bcrs v_tcu;
            if(blocks>0) {v_tcu.tcu_flag = true; }
            std::map<int, int> colmap;
            int c=0;
            for(auto col:countVector){
                if (c<t_percen_ori)
                {
                    v_tcu.t_column.push_back(col.first);
                }
                colmap[col.first]=c++;
            }
            if((t_percen_ori%wide)>0){
                for(int tt=0;tt<(wide-(t_percen_ori%wide));tt++)
                v_tcu.t_column.push_back(-1);
            }
            //存放拆分的cuda长短行数据
            std::vector<Value_cuda> large;
            std::vector<Value_cuda> small;
    
            // std::vector<std::vector<float>> t_value_temp = createVectorOfVectors_float(blocks);
            // std::vector<long> t_binary_temp = createVectorOfVectors_v2(blocks);
            std::vector<at::Half> demo(t_percen*window);
            // std::vector<std::vector<int>> t_row_temp = createVectorOfVectors(blocks);
            // std::vector<std::vector<int>> t_col_temp = createVectorOfVectors(blocks);
            // 指示快内的行偏移
            int p=0;
            for(int j=i*window;j<(i+1)*window;j++)
            {
                //用来存储每行的cuda数据
                std::queue<int> c_col;
                std::queue<int> c_value;
                int cuda_nnz=0;
                for(int m=row[j];m<row[j+1];m++)
                {
                    int newCol = colmap[column[m]];
                    if(newCol<t_percen_ori)
                    {
                        int block = newCol/wide;
                        int bInId= newCol%wide;
                        demo[block*window*wide + p*wide +bInId]=values[m];
                    }else{
                        //划分cuda部分
                        c_col.push(column[m]);
                        c_value.push(values[m]);
                        cuda_nnz++;
                    }
                }
                //处理当前行剩余的cuda元素
                if(cuda_nnz>0)
                {
                    if(cuda_nnz <= shortSize)
                    {
                        Value_cuda temp;
                        temp.cuda_residue_flag = true;
                        temp.c_row_offset = cuda_nnz;
                        temp.c_row = i*window + p;
                        if(t_percen_ori>0) temp.c_atomic = 1;
                        else temp.c_atomic = 0;
                        for(int m=0; m<cuda_nnz; m++)
                        {
                            temp.c_column.push_back(c_col.front());
                            temp.c_value.push_back(c_value.front());
                            c_col.pop();
                            c_value.pop();
                        }
                        small.push_back(temp);
                    }else{
                        if(cuda_nnz <= partSize_c)
                        {
                            Value_cuda temp;
                            temp.cuda_flag = true;
                            temp.c_row_offset = (cuda_nnz);
                            temp.c_row = i*window + p;
                            if(t_percen_ori>0) temp.c_atomic = 1;
                            else temp.c_atomic = 0;
                            for(int m=0; m<cuda_nnz; m++)
                            {
                                temp.c_column.push_back(c_col.front());
                                temp.c_value.push_back(c_value.front());
                                c_col.pop();
                                c_value.pop();
                            }
                            large.push_back(temp);
                        }else{
                            // 当前行的非零元被分成多个part，并需要原子操作
                            int parts_row = cuda_nnz/partSize_c;
                            // int cur_value_indice = row[i];
                            for(int cur_i = 0; cur_i<parts_row; cur_i++)
                            {
                                Value_cuda temp;
                                temp.cuda_flag = true;      
                                temp.c_row_offset = partSize_c;
                                temp.c_row = i*window + p;
                                temp.c_atomic = 1;
                                for(int m=0; m<partSize_c; m++)
                                {
                                    temp.c_column.push_back(c_col.front());
                                    temp.c_value.push_back(c_value.front());
                                    c_col.pop();
                                    c_value.pop();
                                    // cur_value_indice++;
                                }
                                large.push_back(temp);
                            }
                            int residue  = (cuda_nnz%partSize_c);
                            if(residue>0){
                                Value_cuda temp;
                                temp.c_row_offset = residue;
                                temp.c_row = i*window + p;
                                temp.c_atomic = 1;
                                for(int m=0; m<residue; m++)
                                {
                                    temp.c_column.push_back(c_col.front());
                                    temp.c_value.push_back(c_value.front());
                                    c_col.pop();
                                    c_value.pop();
                                    // cur_value_indice++;
                                }
                                if(residue <= shortSize)
                                {
                                    temp.cuda_residue_flag = true;
                                    small.push_back(temp);
                                }else{
                                    temp.cuda_flag = true;
                                    large.push_back(temp);
                                }
                            }
                        }
                    }
                }
                p++;
            }
            //整合vector
            v_tcu.t_value = demo;
            // v_tcu.t_row = mergeVectors(t_row_temp);
            // v_tcu.t_col = mergeVectors(t_col_temp);
        
            if(blocks <= partSize_t)
            {
                v_tcu.row.push_back(t_percen_ori);
                v_tcu.row.push_back(t_percen-t_percen_ori);
                if(t_percen_ori==size) v_tcu.atomic.push_back(0);
                else v_tcu.atomic.push_back(1);
                v_tcu.window.push_back(i);
            }else{
                // 对block太多的块进行分割
                int part_number = blocks / partSize_t;
                int block_residue = blocks % partSize_t;
                if(block_residue > 0)
                {
                    for(int j=0; j<part_number; j++)
                    {
                        v_tcu.row.push_back(partSize_t*wide);
                        v_tcu.row.push_back(0);
                        v_tcu.window.push_back(i);
                        v_tcu.atomic.push_back(1);
                    }
                    v_tcu.row.push_back(block_residue*wide - (t_percen-t_percen_ori));
                    v_tcu.row.push_back(t_percen-t_percen_ori);
                    v_tcu.window.push_back(i);
                    v_tcu.atomic.push_back(1);
                }else{
                    for(int j=0; j<part_number; j++)
                    {
                        //如果是最后一组block
                        if(j == (part_number-1)){
                            v_tcu.row.push_back((partSize_t*wide) - (t_percen-t_percen_ori));
                            v_tcu.row.push_back(t_percen-t_percen_ori);
                        }else{
                            v_tcu.row.push_back(partSize_t*wide);
                            v_tcu.row.push_back(0);
                        }
                        v_tcu.window.push_back(i);
                        v_tcu.atomic.push_back(1);
                    }
                }
            }
            /*
                4. 汇总三部分的结果
            */
            #pragma omp critical  
            res_t[i]=v_tcu;
    
            #pragma omp critical  
            res_c[i]=large;
    
            #pragma omp critical  
            res_c_residue[i]=small;
            
        
        }
        /*
            5. 整合所有window的数据
        */
        //tcu
        std::vector<int> t_rowkNew_offset;
        t_rowkNew_offset.push_back(0);
        std::vector<at::Half> t_valueNew;
        std::vector<int> t_columnNew;
        std::vector<int> t_window_rowNew;
        std::vector<int> t_atomicNew;
    
        //cuda 长行
        std::vector<int> c_rowNew_offset;
        c_rowNew_offset.push_back(0);
        std::vector<int> c_rowNew;
        std::vector<int> c_atomicNew;
        std::vector<int> c_columnNew;
        std::vector<float> c_valueNew;
    
        //cuda 短行
        std::vector<int> c_rowNew_offset_residue;
        c_rowNew_offset_residue.push_back(0);
        std::vector<int> c_rowNew_residue;
        std::vector<int> c_atomicNew_residue;
        std::vector<int> c_columnNew_residue;
        std::vector<float> c_valueNew_residue;
    
        //按顺序整合res
        //tcu
        for (const auto& pair : res_t) {
            if (pair.second.tcu_flag == true )
            {
                for(int sub : pair.second.row)
                {
                    t_rowkNew_offset.push_back(t_rowkNew_offset.back()+sub);
                }   
                t_valueNew.insert(t_valueNew.end(),pair.second.t_value.begin(),pair.second.t_value.end());
                t_columnNew.insert(t_columnNew.end(),pair.second.t_column.begin(),pair.second.t_column.end());
                t_window_rowNew.insert(t_window_rowNew.end(),pair.second.window.begin(),pair.second.window.end());
                t_atomicNew.insert(t_atomicNew.end(),pair.second.atomic.begin(),pair.second.atomic.end());
            }   
        }
    
        //cuda long
        for (const auto& pair : res_c) {
            for( Value_cuda sub : pair.second){
                if (sub.cuda_flag == true )
                {c_rowNew_offset.push_back(c_rowNew_offset.back()+sub.c_row_offset);
                c_rowNew.push_back(sub.c_row);
                c_atomicNew.push_back(sub.c_atomic);
                c_columnNew.insert(c_columnNew.end(),sub.c_column.begin(),sub.c_column.end());
                c_valueNew.insert(c_valueNew.end(),sub.c_value.begin(), sub.c_value.end());}
            }
        }
    
        //cuda short
        for (const auto& pair : res_c_residue) {
            for( Value_cuda sub : pair.second){
                if (sub.cuda_residue_flag == true )
                {c_rowNew_offset_residue.push_back(c_rowNew_offset_residue.back()+sub.c_row_offset);
                c_rowNew_residue.push_back(sub.c_row);
                c_atomicNew_residue.push_back(sub.c_atomic);
                c_columnNew_residue.insert(c_columnNew_residue.end(),sub.c_column.begin(),sub.c_column.end());
                c_valueNew_residue.insert(c_valueNew_residue.end(),sub.c_value.begin(), sub.c_value.end());}
            }
        }
    
        auto end = std::chrono::high_resolution_clock::now();
    
        // 计算耗时
        std::chrono::duration<double> elapsed = end - start;
    
        std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;
    
        auto t_rowNew_offsetTensor1 = torch::from_blob(t_rowkNew_offset.data(), t_rowkNew_offset.size(), torch::kInt32);
        auto t_valueTensor1 = torch::from_blob(t_valueNew.data(), t_valueNew.size(), torch::kFloat16);
        auto t_columnTensor1 = torch::from_blob(t_columnNew.data(), t_columnNew.size(), torch::kInt32);
        auto t_window_rowTensor1 = torch::from_blob(t_window_rowNew.data(), t_window_rowNew.size(), torch::kInt32);
        auto t_atomicTensor1 = torch::from_blob(t_atomicNew.data(), t_atomicNew.size(), torch::kInt32);
    
        torch::Tensor t_rowNew_offsetTensor = torch::empty_like(t_rowNew_offsetTensor1);
        t_rowNew_offsetTensor.copy_(t_rowNew_offsetTensor1);
        torch::Tensor t_valueTensor = torch::empty_like(t_valueTensor1);
        t_valueTensor.copy_(t_valueTensor1);
        torch::Tensor t_columnTensor = torch::empty_like(t_columnTensor1);
        t_columnTensor.copy_(t_columnTensor1);
        torch::Tensor t_window_rowTensor = torch::empty_like(t_window_rowTensor1);
        t_window_rowTensor.copy_(t_window_rowTensor1);
        torch::Tensor t_atomicTensor = torch::empty_like(t_atomicTensor1);
        t_atomicTensor.copy_(t_atomicTensor1);
    
    
        auto c_row_offsetTensor1 = torch::from_blob(c_rowNew_offset.data(), c_rowNew_offset.size(), torch::kInt32);
        auto c_rowTensor1 = torch::from_blob(c_rowNew.data(), c_rowNew.size(), torch::kInt32);
        auto c_atomicTensor1 = torch::from_blob(c_atomicNew.data(), c_atomicNew.size(), torch::kInt32);
        auto c_colTensor1 = torch::from_blob(c_columnNew.data(), c_columnNew.size(), torch::kInt32);
        auto c_valueTensor1 = torch::from_blob(c_valueNew.data(), c_valueNew.size(), torch::kFloat32);
        
        torch::Tensor c_row_offsetTensor = torch::empty_like(c_row_offsetTensor1);
        c_row_offsetTensor.copy_(c_row_offsetTensor1);
        torch::Tensor c_rowTensor = torch::empty_like(c_rowTensor1);
        c_rowTensor.copy_(c_rowTensor1);
        torch::Tensor c_atomicTensor = torch::empty_like(c_atomicTensor1);
        c_atomicTensor.copy_(c_atomicTensor1);
        torch::Tensor c_colTensor = torch::empty_like(c_colTensor1);
        c_colTensor.copy_(c_colTensor1);
        torch::Tensor c_valueTensor = torch::empty_like(c_valueTensor1);
        c_valueTensor.copy_(c_valueTensor1);
        
    
        auto c_row_offsetTensor1_residue = torch::from_blob(c_rowNew_offset_residue.data(), c_rowNew_offset_residue.size(), torch::kInt32);
        auto c_rowTensor1_residue = torch::from_blob(c_rowNew_residue.data(), c_rowNew_residue.size(), torch::kInt32);
        auto c_atomicTensor1_residue = torch::from_blob(c_atomicNew_residue.data(), c_atomicNew_residue.size(), torch::kInt32);
        auto c_colTensor1_residue = torch::from_blob(c_columnNew_residue.data(), c_columnNew_residue.size(), torch::kInt32);
        auto c_valueTensor1_residue = torch::from_blob(c_valueNew_residue.data(), c_valueNew_residue.size(), torch::kFloat32);
        
        torch::Tensor c_row_offsetTensor_residue = torch::empty_like(c_row_offsetTensor1_residue);
        c_row_offsetTensor_residue.copy_(c_row_offsetTensor1_residue);
        torch::Tensor c_rowTensor_residue = torch::empty_like(c_rowTensor1_residue);
        c_rowTensor_residue.copy_(c_rowTensor1_residue);
        torch::Tensor c_atomicTensor_residue = torch::empty_like(c_atomicTensor1_residue);
        c_atomicTensor_residue.copy_(c_atomicTensor1_residue);
        torch::Tensor c_colTensor_residue = torch::empty_like(c_colTensor1_residue);
        c_colTensor_residue.copy_(c_colTensor1_residue);
        torch::Tensor c_valueTensor_residue = torch::empty_like(c_valueTensor1_residue);
        c_valueTensor_residue.copy_(c_valueTensor1_residue);
    
        return {t_rowNew_offsetTensor,t_columnTensor, t_valueTensor, t_window_rowTensor, t_atomicTensor,
                c_row_offsetTensor, c_rowTensor, c_atomicTensor, c_colTensor, c_valueTensor,
                c_row_offsetTensor_residue, c_rowTensor_residue, c_atomicTensor_residue, c_colTensor_residue, c_valueTensor_residue, torch::tensor(elapsed.count())};
    
    }


    // m-tcu + cuda + partSize + binary
std::vector<torch::Tensor> block_fp16_tcu_cuda_bcrs_pcgcn(torch::Tensor row1, torch::Tensor column1, torch::Tensor values1, 
    int partSize_t, int partSize_c, int shortSize, int density, int window, int wide)
    {
        auto *row=row1.data_ptr<int>();
        auto column=column1.accessor<int, 1>();
        auto values=values1.accessor<float, 1>();
    
        auto start = std::chrono::high_resolution_clock::now();
    
        int rows=row1.size(0)-1;
        // Result
        std::map<int, Value_tcu_bcrs> res_t;
        std::map<int, std::vector<Value_cuda>> res_c;
        std::map<int, std::vector<Value_cuda>> res_c_residue;
        //需要的window数量
        int num_windows = rows/window;
        #pragma omp parallel for
        for(int i=0;i<num_windows;i++)
        {
            /*
                1. 统计当前window中的vector, 并按照稠密度降序排列
            */
            //对column进行合并
            std::vector<int> mergedVector(&column[row[i*window]], &column[row[i*window+window]]);
            // 使用 unordered_map 统计 vector 中每个元素出现次数
            std::unordered_map<int, int> elementCounts;
            for (const auto& element : mergedVector) elementCounts[element]++;
            // 将统计结果放入 vector<pair> 中
            std::vector<std::pair<int, int>> countVector(elementCounts.begin(), elementCounts.end());
            // 按元素个数降序排序
            std::sort(countVector.begin(), countVector.end(), [](const auto& a, const auto& b) {
                return a.second > b.second;
            });
    
            /*
                2. 确定需要进行tcu计算的vector数量, t_percen
            */
            int size = countVector.size();
            int residue = size%wide;
            //统计每个block中的非零元数量，计算稠密度
            int t_percen = 0; //确定前多少个vector会被用作tcu进行稠密计算
            t_percen = std::count_if(countVector.begin(), countVector.end(), [density](const auto& pair) {
                return pair.second >= density;
            });
            int t_percen_ori = t_percen;
            int t_percen_ori_pc = t_percen*wide;
            // if((t_percen%wide) != 0)
            // {
            //     // if((t_percen + (wide - t_percen%wide)) <= size) 
            //     t_percen = (t_percen + (wide - t_percen%wide));
            //     // else
            //     // t_percen = (t_percen - (t_percen%wide));
    
            // }
            // if(t_percen<=size){
            // // 将前 k 个数按升序排列
            // std::sort(countVector.begin(), countVector.begin() + t_percen);
            // // 将剩下的数按升序排列
            // std::sort(countVector.begin() + t_percen, countVector.end());
            // }else{
            //     std::sort(countVector.begin(), countVector.end());
            // }
            std::sort(countVector.begin(), countVector.begin()+t_percen_ori);
            std::sort(countVector.begin() + t_percen_ori, countVector.end());
            /*
                3. 根据t_percen分别划分TCU和CUDA的负载
            */
    
            // TCU 划分
            // 3.1 根据partSize_t划分t_part
            //当前window中有几个block
            int blocks = t_percen; 
            Value_tcu_bcrs v_tcu;
            if(blocks>0) {v_tcu.tcu_flag = true; }
            std::map<int, int> colmap;
            int c=0;
            for(auto col:countVector){
                if (c<t_percen_ori)
                {
                    v_tcu.t_column.push_back(col.first);
                    for(int tt=0; tt<wide-1;tt++)
                        v_tcu.t_column.push_back(0);
                }
                colmap[col.first]=c*wide;
            }
            //存放拆分的cuda长短行数据
            std::vector<Value_cuda> large;
            std::vector<Value_cuda> small;
    
            // std::vector<std::vector<float>> t_value_temp = createVectorOfVectors_float(blocks);
            // std::vector<long> t_binary_temp = createVectorOfVectors_v2(blocks);
            std::vector<at::Half> demo(t_percen_ori_pc*window);
            // std::vector<std::vector<int>> t_row_temp = createVectorOfVectors(blocks);
            // std::vector<std::vector<int>> t_col_temp = createVectorOfVectors(blocks);
            // 指示快内的行偏移
            int p=0;
            for(int j=i*window;j<(i+1)*window;j++)
            {
                //用来存储每行的cuda数据
                std::queue<int> c_col;
                std::queue<int> c_value;
                int cuda_nnz=0;
                for(int m=row[j];m<row[j+1];m++)
                {
                    int newCol = colmap[column[m]];
                    if(newCol<t_percen_ori_pc)
                    {
                        int block = newCol/wide;
                        int bInId= newCol%wide;
                        demo[block*window*wide + p*wide +bInId]=values[m];
                    }else{
                        //划分cuda部分
                        c_col.push(column[m]);
                        c_value.push(values[m]);
                        cuda_nnz++;
                    }
                }
                //处理当前行剩余的cuda元素
                if(cuda_nnz>0)
                {
                    if(cuda_nnz <= shortSize)
                    {
                        Value_cuda temp;
                        temp.cuda_residue_flag = true;
                        temp.c_row_offset = cuda_nnz;
                        temp.c_row = i*window + p;
                        if(t_percen_ori>0) 
                        {
                            temp.c_atomic = 1;
                            temp.c_atomic_seq = 0;
                        }
                        else{
                             temp.c_atomic = 0;
                             temp.c_atomic_seq = 0;
                        }
                        for(int m=0; m<cuda_nnz; m++)
                        {
                            temp.c_column.push_back(c_col.front());
                            temp.c_value.push_back(c_value.front());
                            c_col.pop();
                            c_value.pop();
                        }
                        small.push_back(temp);
                    }else{
                        if(cuda_nnz <= partSize_c)
                        {
                            Value_cuda temp;
                            temp.cuda_flag = true;
                            temp.c_row_offset = (cuda_nnz);
                            temp.c_row = i*window + p;
                            if(t_percen_ori>0) 
                            {
                                temp.c_atomic = 1;
                                temp.c_atomic_seq = 0;
                            }
                            else{
                                temp.c_atomic = 0;
                                temp.c_atomic_seq = 0;
                            }
                            for(int m=0; m<cuda_nnz; m++)
                            {
                                temp.c_column.push_back(c_col.front());
                                temp.c_value.push_back(c_value.front());
                                c_col.pop();
                                c_value.pop();
                            }
                            large.push_back(temp);
                        }else{
                            // 当前行的非零元被分成多个part，并需要原子操作
                            int parts_row = cuda_nnz/partSize_c;
                            // int cur_value_indice = row[i];
                            for(int cur_i = 0; cur_i<parts_row; cur_i++)
                            {
                                Value_cuda temp;
                                temp.cuda_flag = true;      
                                temp.c_row_offset = partSize_c;
                                temp.c_row = i*window + p;
                                temp.c_atomic = 1;
                                temp.c_atomic_seq = 1;
                                for(int m=0; m<partSize_c; m++)
                                {
                                    temp.c_column.push_back(c_col.front());
                                    temp.c_value.push_back(c_value.front());
                                    c_col.pop();
                                    c_value.pop();
                                    // cur_value_indice++;
                                }
                                large.push_back(temp);
                            }
                            int residue  = (cuda_nnz%partSize_c);
                            if(residue>0){
                                Value_cuda temp;
                                temp.c_row_offset = residue;
                                temp.c_row = i*window + p;
                                temp.c_atomic = 1;
                                temp.c_atomic_seq = 1;
                                for(int m=0; m<residue; m++)
                                {
                                    temp.c_column.push_back(c_col.front());
                                    temp.c_value.push_back(c_value.front());
                                    c_col.pop();
                                    c_value.pop();
                                    // cur_value_indice++;
                                }
                                if(residue <= shortSize)
                                {
                                    temp.cuda_residue_flag = true;
                                    small.push_back(temp);
                                }else{
                                    temp.cuda_flag = true;
                                    large.push_back(temp);
                                }
                            }
                        }
                    }
                }
                p++;
            }
            //整合vector
            v_tcu.t_value = demo;
            // v_tcu.t_row = mergeVectors(t_row_temp);
            // v_tcu.t_col = mergeVectors(t_col_temp);
        
            if(blocks <= partSize_t)
            {
                v_tcu.row.push_back(t_percen_ori_pc);
                v_tcu.row.push_back(0);
                if(t_percen_ori==size) 
                {
                    v_tcu.atomic.push_back(0);
                    v_tcu.atomic_seq.push_back(0);
                }
                else {
                    v_tcu.atomic.push_back(1);
                    v_tcu.atomic_seq.push_back(0);
                }
                v_tcu.window.push_back(i);
            }else{
                // 对block太多的块进行分割
                int part_number = blocks / partSize_t;
                int block_residue = blocks % partSize_t;
                if(block_residue > 0)
                {
                    for(int j=0; j<part_number; j++)
                    {
                        v_tcu.row.push_back(partSize_t*wide);
                        v_tcu.row.push_back(0);
                        v_tcu.window.push_back(i);
                        v_tcu.atomic.push_back(1);
                        v_tcu.atomic_seq.push_back(1);
                    }
                    v_tcu.row.push_back(block_residue*wide);
                    v_tcu.row.push_back(0);
                    v_tcu.window.push_back(i);
                    v_tcu.atomic.push_back(1);
                    v_tcu.atomic_seq.push_back(1);
                }else{
                    for(int j=0; j<part_number; j++)
                    {
                        //如果是最后一组block
                        if(j == (part_number-1)){
                            v_tcu.row.push_back((partSize_t*wide));
                            v_tcu.row.push_back(0);
                        }else{
                            v_tcu.row.push_back(partSize_t*wide);
                            v_tcu.row.push_back(0);
                        }
                        v_tcu.window.push_back(i);
                        v_tcu.atomic.push_back(1);
                        v_tcu.atomic_seq.push_back(1);
                    }
                }
            }
            /*
                4. 汇总三部分的结果
            */
            #pragma omp critical  
            res_t[i]=v_tcu;
    
            #pragma omp critical  
            res_c[i]=large;
    
            #pragma omp critical  
            res_c_residue[i]=small;
            
        
        }
        /*
            5. 整合所有window的数据
        */
        //tcu
        std::vector<int> t_rowkNew_offset;
        t_rowkNew_offset.push_back(0);
        std::vector<at::Half> t_valueNew;
        std::vector<int> t_columnNew;
        std::vector<int> t_window_rowNew;
        std::vector<int> t_atomicNew;
        std::vector<int> t_atomicNew_seq;
    
        //cuda 长行
        std::vector<int> c_rowNew_offset;
        c_rowNew_offset.push_back(0);
        std::vector<int> c_rowNew;
        std::vector<int> c_atomicNew;
        std::vector<int> c_atomicNew_seq;
        std::vector<int> c_columnNew;
        std::vector<float> c_valueNew;
    
        //cuda 短行
        std::vector<int> c_rowNew_offset_residue;
        c_rowNew_offset_residue.push_back(0);
        std::vector<int> c_rowNew_residue;
        std::vector<int> c_atomicNew_residue;
        std::vector<int> c_atomicNew_residue_seq;
        std::vector<int> c_columnNew_residue;
        std::vector<float> c_valueNew_residue;
    
        //按顺序整合res
        //tcu
        for (const auto& pair : res_t) {
            if (pair.second.tcu_flag == true )
            {
                for(int sub : pair.second.row)
                {
                    t_rowkNew_offset.push_back(t_rowkNew_offset.back()+sub);
                }   
                t_valueNew.insert(t_valueNew.end(),pair.second.t_value.begin(),pair.second.t_value.end());
                t_columnNew.insert(t_columnNew.end(),pair.second.t_column.begin(),pair.second.t_column.end());
                t_window_rowNew.insert(t_window_rowNew.end(),pair.second.window.begin(),pair.second.window.end());
                t_atomicNew.insert(t_atomicNew.end(),pair.second.atomic.begin(),pair.second.atomic.end());
                t_atomicNew_seq.insert(t_atomicNew_seq.end(),pair.second.atomic_seq.begin(),pair.second.atomic_seq.end());

            }   
        }
    
        //cuda long
        for (const auto& pair : res_c) {
            for( Value_cuda sub : pair.second){
                if (sub.cuda_flag == true )
                {c_rowNew_offset.push_back(c_rowNew_offset.back()+sub.c_row_offset);
                c_rowNew.push_back(sub.c_row);
                c_atomicNew.push_back(sub.c_atomic);
                c_atomicNew_seq.push_back(sub.c_atomic_seq);
                c_columnNew.insert(c_columnNew.end(),sub.c_column.begin(),sub.c_column.end());
                c_valueNew.insert(c_valueNew.end(),sub.c_value.begin(), sub.c_value.end());}
            }
        }
    
        //cuda short
        for (const auto& pair : res_c_residue) {
            for( Value_cuda sub : pair.second){
                if (sub.cuda_residue_flag == true )
                {c_rowNew_offset_residue.push_back(c_rowNew_offset_residue.back()+sub.c_row_offset);
                c_rowNew_residue.push_back(sub.c_row);
                c_atomicNew_residue.push_back(sub.c_atomic);
                c_atomicNew_residue_seq.push_back(sub.c_atomic_seq);
                c_columnNew_residue.insert(c_columnNew_residue.end(),sub.c_column.begin(),sub.c_column.end());
                c_valueNew_residue.insert(c_valueNew_residue.end(),sub.c_value.begin(), sub.c_value.end());}
            }
        }
    
        auto end = std::chrono::high_resolution_clock::now();
    
        // 计算耗时
        std::chrono::duration<double> elapsed = end - start;
    
        std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;
    
        auto t_rowNew_offsetTensor1 = torch::from_blob(t_rowkNew_offset.data(), t_rowkNew_offset.size(), torch::kInt32);
        auto t_valueTensor1 = torch::from_blob(t_valueNew.data(), t_valueNew.size(), torch::kFloat16);
        auto t_columnTensor1 = torch::from_blob(t_columnNew.data(), t_columnNew.size(), torch::kInt32);
        auto t_window_rowTensor1 = torch::from_blob(t_window_rowNew.data(), t_window_rowNew.size(), torch::kInt32);
        auto t_atomicTensor1 = torch::from_blob(t_atomicNew.data(), t_atomicNew.size(), torch::kInt32);
        auto t_atomicTensor1_seq = torch::from_blob(t_atomicNew_seq.data(), t_atomicNew_seq.size(), torch::kInt32);


        torch::Tensor t_rowNew_offsetTensor = torch::empty_like(t_rowNew_offsetTensor1);
        t_rowNew_offsetTensor.copy_(t_rowNew_offsetTensor1);
        torch::Tensor t_valueTensor = torch::empty_like(t_valueTensor1);
        t_valueTensor.copy_(t_valueTensor1);
        torch::Tensor t_columnTensor = torch::empty_like(t_columnTensor1);
        t_columnTensor.copy_(t_columnTensor1);
        torch::Tensor t_window_rowTensor = torch::empty_like(t_window_rowTensor1);
        t_window_rowTensor.copy_(t_window_rowTensor1);
        torch::Tensor t_atomicTensor = torch::empty_like(t_atomicTensor1);
        t_atomicTensor.copy_(t_atomicTensor1);
        torch::Tensor t_atomicTensor_seq = torch::empty_like(t_atomicTensor1_seq);
        t_atomicTensor_seq.copy_(t_atomicTensor1_seq);

    
        auto c_row_offsetTensor1 = torch::from_blob(c_rowNew_offset.data(), c_rowNew_offset.size(), torch::kInt32);
        auto c_rowTensor1 = torch::from_blob(c_rowNew.data(), c_rowNew.size(), torch::kInt32);
        auto c_atomicTensor1 = torch::from_blob(c_atomicNew.data(), c_atomicNew.size(), torch::kInt32);
        auto c_atomicTensor1_seq = torch::from_blob(c_atomicNew_seq.data(), c_atomicNew_seq.size(), torch::kInt32);   
        auto c_colTensor1 = torch::from_blob(c_columnNew.data(), c_columnNew.size(), torch::kInt32);
        auto c_valueTensor1 = torch::from_blob(c_valueNew.data(), c_valueNew.size(), torch::kFloat32);
        
        torch::Tensor c_row_offsetTensor = torch::empty_like(c_row_offsetTensor1);
        c_row_offsetTensor.copy_(c_row_offsetTensor1);
        torch::Tensor c_rowTensor = torch::empty_like(c_rowTensor1);
        c_rowTensor.copy_(c_rowTensor1);
        torch::Tensor c_atomicTensor = torch::empty_like(c_atomicTensor1);
        c_atomicTensor.copy_(c_atomicTensor1);
        torch::Tensor c_atomicTensor_seq = torch::empty_like(c_atomicTensor1_seq);
        c_atomicTensor_seq.copy_(c_atomicTensor1_seq);
        torch::Tensor c_colTensor = torch::empty_like(c_colTensor1);
        c_colTensor.copy_(c_colTensor1);
        torch::Tensor c_valueTensor = torch::empty_like(c_valueTensor1);
        c_valueTensor.copy_(c_valueTensor1);
        
    
        auto c_row_offsetTensor1_residue = torch::from_blob(c_rowNew_offset_residue.data(), c_rowNew_offset_residue.size(), torch::kInt32);
        auto c_rowTensor1_residue = torch::from_blob(c_rowNew_residue.data(), c_rowNew_residue.size(), torch::kInt32);
        auto c_atomicTensor1_residue = torch::from_blob(c_atomicNew_residue.data(), c_atomicNew_residue.size(), torch::kInt32);
        auto c_atomicTensor1_residue_seq = torch::from_blob(c_atomicNew_residue_seq.data(), c_atomicNew_residue_seq.size(), torch::kInt32);  
        auto c_colTensor1_residue = torch::from_blob(c_columnNew_residue.data(), c_columnNew_residue.size(), torch::kInt32);
        auto c_valueTensor1_residue = torch::from_blob(c_valueNew_residue.data(), c_valueNew_residue.size(), torch::kFloat32);
        
        torch::Tensor c_row_offsetTensor_residue = torch::empty_like(c_row_offsetTensor1_residue);
        c_row_offsetTensor_residue.copy_(c_row_offsetTensor1_residue);
        torch::Tensor c_rowTensor_residue = torch::empty_like(c_rowTensor1_residue);
        c_rowTensor_residue.copy_(c_rowTensor1_residue);
        torch::Tensor c_atomicTensor_residue = torch::empty_like(c_atomicTensor1_residue);
        c_atomicTensor_residue.copy_(c_atomicTensor1_residue);
        torch::Tensor c_atomicTensor_residue_seq = torch::empty_like(c_atomicTensor1_residue_seq);
        c_atomicTensor_residue_seq.copy_(c_atomicTensor1_residue_seq);
        torch::Tensor c_colTensor_residue = torch::empty_like(c_colTensor1_residue);
        c_colTensor_residue.copy_(c_colTensor1_residue);
        torch::Tensor c_valueTensor_residue = torch::empty_like(c_valueTensor1_residue);
        c_valueTensor_residue.copy_(c_valueTensor1_residue);
    
        return {t_rowNew_offsetTensor,t_columnTensor, t_valueTensor, t_window_rowTensor, t_atomicTensor, t_atomicTensor_seq,
                c_row_offsetTensor, c_rowTensor, c_atomicTensor, c_atomicTensor_seq, c_colTensor, c_valueTensor,
                c_row_offsetTensor_residue, c_rowTensor_residue, c_atomicTensor_residue, c_atomicTensor_residue_seq, c_colTensor_residue, c_valueTensor_residue, torch::tensor(elapsed.count())};
    
    }



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

  m.def("forward_fp16_cuda_v2", &spmm_forward_fp16_cuda_v2, "CUDA-v2");
  m.def("forward_fp16_cuda_v1", &spmm_forward_fp16_cuda_v1, "CUDA-v2");
  m.def("forward_fp16_tcu_nomap", &spmm_forward_fp16_tcu_cuda_nomap, "bitmap without threadmapping");
  m.def("forward_fp16_tcu_bcrs_part", &spmm_forward_fp16_tcu_bcrs_part, "TCU-BCRS");
  m.def("forward_tf32_tcu_bcrs_part", &spmm_forward_tf32_tcu_bcrs_part, "TCU-BCRS");

  m.def("forward_fp16_tcu_cuda", &spmm_forward_fp16_tcu_cuda, "CUDA-v2 + TCU-bitmap");
  m.def("forward_fp16_tcu_cuda_seq", &spmm_forward_fp16_tcu_cuda_seq, "CUDA-v2 + TCU-bitmap");
  m.def("forward_fp16_tcu_cuda_cc_only", &spmm_forward_fp16_tcu_cuda_cc_only, "CUDA-v2 + TCU-bitmap");
  m.def("forward_fp16_tcu_cuda_bcrs", &spmm_forward_fp16_tcu_cuda_bcrs, "CUDA-v2 + TCU-BCRS");
  m.def("forward_fp16_tcu_cuda_bcrs_seq", &spmm_forward_fp16_tcu_cuda_bcrs_seq, "CUDA-v2 + TCU-BCRS");
  m.def("forward_fp16_tcu_cuda_hybrid", &spmm_forward_fp16_tcu_cuda_hybrid, " cuda-v1 + bitmap without threadmapping");
  m.def("forward_fp16_tcu_cuda_hybrid_seq", &spmm_forward_fp16_tcu_cuda_hybrid_seq, "CUDA-v2 + TCU-bitmap cuda-v1 + bitmap without threadmapping");


  m.def("forward_tf32_cuda_v2", &spmm_forward_tf32_cuda_v2, "CUDA-v2");
  m.def("forward_tf32_tcu_cuda", &spmm_forward_tf32_tcu_cuda, "CUDA-v2 + TCU-bitmap");
  m.def("forward_tf32_tcu_cuda_seq", &spmm_forward_tf32_tcu_cuda_seq, "CUDA-v2 + TCU-bitmap");
  m.def("forward_tf32_tcu_cuda_bcrs", &spmm_forward_tf32_tcu_cuda_bcrs, "CUDA-v2 + TCU-bitmap");
  m.def("forward_tf32_tcu_cuda_bcrs_seq", &spmm_forward_tf32_tcu_cuda_bcrs_seq, "CUDA-v2 + TCU-bitmap");


  m.def("forward_tf32_tcu_tcf_part", &spmm_forward_tf32_tcu_tcf_part, "FlashSparse");

  m.def("block_fp16_tcu_cuda_bcrs", &block_fp16_tcu_cuda_bcrs, "bcsr for hybrid");
  m.def("block_fp16_tcu_cuda_bcrs_seq", &block_fp16_tcu_cuda_bcrs_seq, "bcsr for hybrid");
  m.def("block_fp16_tcu_cuda_bcrs_hcspmm", &block_fp16_tcu_cuda_bcrs_hcspmm, "bcsr for hybrid");
  m.def("block_fp16_tcu_cuda_bcrs_pcgcn", &block_fp16_tcu_cuda_bcrs_pcgcn, "bcsr for hybrid");

  m.def("block_fp16_tcu_bcrs", &block_fp16_tcu_bcrs, "bcsr fot tcu");
  m.def("block_tf32_tcu_tcf", &block_tf32_tcu_tcf, "tcf fot tcu");

  m.def("forward_fp16", &gcn_forward_fp16_tcu_cuda_bcrs, "gnn");
  }