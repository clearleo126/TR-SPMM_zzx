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



// cuda 长短行
float sddmm_forward_tf32_cuda_kernel_split(

    int * c_row_offset,
    int * c_row,
    int * c_column, 

    float * rhs_matrix,
    float * output_matrix,
    const int nOri,
    const int mOri,
    int epoches,
    int parts);
float sddmm_forward_tf32_cuda_kernel_split_32(

    int * c_row_offset,
    int * c_row,
    int * c_column, 

    float * rhs_matrix,
    float * output_matrix,
    const int nOri,
    const int mOri,
    int epoches,
    int parts);
float sddmm_forward_tf32_cuda_kernel_split_128(

    int * c_row_offset,
    int * c_row,
    int * c_column, 

    float * rhs_matrix,
    float * output_matrix,
    const int nOri,
    const int mOri,
    int epoches,
    int parts);
std::vector<torch::Tensor> sddmm_forward_tf32_cuda_v2(
    
    torch::Tensor c_row_offsets,
    torch::Tensor c_row, 
    torch::Tensor c_column, 

    torch::Tensor rhs_matrix,

    const int nOri,
    const int mOri,
    const int kOri,
    int epoches,
    int parts)
{
    // int dimM=dimM1/16;
    auto output_matrix = torch::zeros({c_column.size(0)}, torch::kFloat32).to(torch::kCPU);
    
    //把CPU端的tensor转成C++的数据结构
    int * c_row_offsets_ = c_row_offsets.data<int>();
    int * c_row_ = c_row.data<int>();
    int * c_column_ = c_column.data<int>();

    float * rhs_matrix_ = rhs_matrix.data<float>(); 
    float * output_matrix_ = output_matrix.data<float>();

    // Device
    int *d_c_row_offsets, *d_c_row, *d_c_column;
	float *d_rhs_matrix;
    float *d_output_matrix;

    checkCuda(cudaMalloc(&d_c_row_offsets, (c_row_offsets.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_row, (c_row.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_column, (c_column.size(0)) * sizeof(int)));

    checkCuda(cudaMalloc(&d_rhs_matrix, (kOri*nOri) * sizeof(float)));
    checkCuda(cudaMalloc(&d_output_matrix, (c_column.size(0)) * sizeof(float)));

    //把CPU数据放到GPU上
    checkCuda(cudaMemcpy(d_c_row_offsets, c_row_offsets_ , (c_row_offsets.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_row, c_row_, (c_row.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_column, c_column_, (c_column.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    
    checkCuda(cudaMemcpy(d_rhs_matrix, rhs_matrix_,(kOri*nOri) * sizeof(float), cudaMemcpyHostToDevice));

    float spmm_ms_avg = 0;
    if(nOri==32){
        spmm_ms_avg = sddmm_forward_tf32_cuda_kernel_split_32(

      d_c_row_offsets, 
      d_c_row, 
      d_c_column,

      d_rhs_matrix,
      d_output_matrix,

      nOri,
      mOri,
      epoches,
      parts);    
    }else if(nOri ==128){
        spmm_ms_avg =sddmm_forward_tf32_cuda_kernel_split_128(
        d_c_row_offsets, 
        d_c_row, 
        d_c_column,

        d_rhs_matrix,
        d_output_matrix,

        nOri,
        mOri,
        epoches,
        parts);    
    }else{
       spmm_ms_avg =  sddmm_forward_tf32_cuda_kernel_split(

      d_c_row_offsets, 
      d_c_row, 
      d_c_column,

      d_rhs_matrix,
      d_output_matrix,

      nOri,
      mOri,
      epoches,
      parts);    
    }
    checkCuda(cudaMemcpy(output_matrix_, d_output_matrix, c_column.size(0) * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_c_row_offsets);
    cudaFree(d_c_row);
    cudaFree(d_c_column);

    cudaFree(d_rhs_matrix);
    cudaFree(d_output_matrix);
    cudaDeviceSynchronize(); 
    return {output_matrix, torch::tensor(spmm_ms_avg)};
}

//navie
float sddmm_forward_tf32_cuda_kernel_split_navie(

    int * c_row_offset,
    int * c_row,
    int * c_column, 

    float * rhs_matrix,
    float * output_matrix,
    const int nOri,
    const int mOri,
    int epoches,
    int parts);

std::vector<torch::Tensor> sddmm_forward_tf32_cuda_v2_navie(
    
    torch::Tensor c_row_offsets,
    torch::Tensor c_row, 
    torch::Tensor c_column, 

    torch::Tensor rhs_matrix,

    const int nOri,
    const int mOri,
    int epoches,
    int parts)
{
    // int dimM=dimM1/16;
    auto output_matrix = torch::zeros({c_column.size(0)}, torch::kFloat32).to(torch::kCPU);
    
    //把CPU端的tensor转成C++的数据结构
    int * c_row_offsets_ = c_row_offsets.data<int>();
    int * c_row_ = c_row.data<int>();
    int * c_column_ = c_column.data<int>();

    float * rhs_matrix_ = rhs_matrix.data<float>(); 
    float * output_matrix_ = output_matrix.data<float>();

    // Device
    int *d_c_row_offsets, *d_c_row, *d_c_column;
	float *d_rhs_matrix;
    float *d_output_matrix;

    checkCuda(cudaMalloc(&d_c_row_offsets, (c_row_offsets.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_row, (c_row.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_column, (c_column.size(0)) * sizeof(int)));

    checkCuda(cudaMalloc(&d_rhs_matrix, (mOri*nOri) * sizeof(float)));
    checkCuda(cudaMalloc(&d_output_matrix, (c_column.size(0)) * sizeof(float)));

    //把CPU数据放到GPU上
    checkCuda(cudaMemcpy(d_c_row_offsets, c_row_offsets_ , (c_row_offsets.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_row, c_row_, (c_row.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_column, c_column_, (c_column.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    
    checkCuda(cudaMemcpy(d_rhs_matrix, rhs_matrix_,(mOri*nOri) * sizeof(float), cudaMemcpyHostToDevice));

    float spmm_ms_avg =  sddmm_forward_tf32_cuda_kernel_split_navie(

      d_c_row_offsets, 
      d_c_row, 
      d_c_column,

      d_rhs_matrix,
      d_output_matrix,

      nOri,
      mOri,
      epoches,
      parts);    
    checkCuda(cudaMemcpy(output_matrix_, d_output_matrix, c_column.size(0) * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_c_row_offsets);
    cudaFree(d_c_row);
    cudaFree(d_c_column);

    cudaFree(d_rhs_matrix);
    cudaFree(d_output_matrix);
    cudaDeviceSynchronize(); 
    return {output_matrix, torch::tensor(spmm_ms_avg)};
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
    int parts);

std::vector<torch::Tensor> sddmm_forward_fp16_cuda_v2_navie(
    
    torch::Tensor c_row_offsets,
    torch::Tensor c_row, 
    torch::Tensor c_column, 

    torch::Tensor rhs_matrix,

    const int nOri,
    const int mOri,
    const int kOri,
    int epoches,
    int parts)
{
    // int dimM=dimM1/16;
    auto output_matrix = torch::zeros({c_column.size(0)}, torch::kFloat16).to(torch::kCPU);
    
    //把CPU端的tensor转成C++的数据结构
    int * c_row_offsets_ = c_row_offsets.data<int>();
    int * c_row_ = c_row.data<int>();
    int * c_column_ = c_column.data<int>();

    half * rhs_matrix_ = reinterpret_cast<half *>(rhs_matrix.data<at::Half>()); 
    half * output_matrix_ = reinterpret_cast<half *>(output_matrix.data<at::Half>());

    // Device
    int *d_c_row_offsets, *d_c_row, *d_c_column;
	half *d_rhs_matrix;
    half *d_output_matrix;

    checkCuda(cudaMalloc(&d_c_row_offsets, (c_row_offsets.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_row, (c_row.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_column, (c_column.size(0)) * sizeof(int)));

    checkCuda(cudaMalloc(&d_rhs_matrix, (kOri*nOri) * sizeof(half)));
    checkCuda(cudaMalloc(&d_output_matrix, (c_column.size(0)) * sizeof(half)));

    //把CPU数据放到GPU上
    checkCuda(cudaMemcpy(d_c_row_offsets, c_row_offsets_ , (c_row_offsets.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_row, c_row_, (c_row.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_column, c_column_, (c_column.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    
    checkCuda(cudaMemcpy(d_rhs_matrix, rhs_matrix_,(kOri*nOri) * sizeof(half), cudaMemcpyHostToDevice));

    float spmm_ms_avg =  sddmm_forward_fp16_cuda_kernel_split_navie(

      d_c_row_offsets, 
      d_c_row, 
      d_c_column,

      d_rhs_matrix,
      d_output_matrix,

      nOri,
      mOri,
      epoches,
      parts);    
    checkCuda(cudaMemcpy(output_matrix_, d_output_matrix, c_column.size(0) * sizeof(half), cudaMemcpyDeviceToHost));

    cudaFree(d_c_row_offsets);
    cudaFree(d_c_row);
    cudaFree(d_c_column);

    cudaFree(d_rhs_matrix);
    cudaFree(d_output_matrix);
    cudaDeviceSynchronize(); 
    return {output_matrix, torch::tensor(spmm_ms_avg)};
}

float sddmm_forward_fp16_cuda_kernel_split(

    int * c_row_offset,
    int * c_row,
    int * c_column, 

    half * rhs_matrix,
    half * output_matrix,
    const int nOri,
    const int mOri,
    int epoches,
    int parts);

std::vector<torch::Tensor> sddmm_forward_fp16_cuda_v2(
    
    torch::Tensor c_row_offsets,
    torch::Tensor c_row, 
    torch::Tensor c_column, 

    torch::Tensor rhs_matrix,

    const int nOri,
    const int mOri,
    const int kOri,
    int epoches,
    int parts)
{
    // int dimM=dimM1/16;
    auto output_matrix = torch::zeros({c_column.size(0)}, torch::kFloat16).to(torch::kCPU);
    
    //把CPU端的tensor转成C++的数据结构
    int * c_row_offsets_ = c_row_offsets.data<int>();
    int * c_row_ = c_row.data<int>();
    int * c_column_ = c_column.data<int>();

    half * rhs_matrix_ = reinterpret_cast<half *>(rhs_matrix.data<at::Half>()); 
    half * output_matrix_ = reinterpret_cast<half *>(output_matrix.data<at::Half>()); 

    // Device
    int *d_c_row_offsets, *d_c_row, *d_c_column;
	half *d_rhs_matrix;
    half *d_output_matrix;

    checkCuda(cudaMalloc(&d_c_row_offsets, (c_row_offsets.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_row, (c_row.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_column, (c_column.size(0)) * sizeof(int)));

    checkCuda(cudaMalloc(&d_rhs_matrix, (kOri*nOri) * sizeof(half)));
    checkCuda(cudaMalloc(&d_output_matrix, c_column.size(0) * sizeof(half)));

    //把CPU数据放到GPU上
    checkCuda(cudaMemcpy(d_c_row_offsets, c_row_offsets_ , (c_row_offsets.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_row, c_row_, (c_row.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_column, c_column_, (c_column.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    
    checkCuda(cudaMemcpy(d_rhs_matrix, rhs_matrix_,(kOri*nOri) * sizeof(half), cudaMemcpyHostToDevice));

    float spmm_ms_avg =  sddmm_forward_fp16_cuda_kernel_split(

      d_c_row_offsets, 
      d_c_row, 
      d_c_column,

      d_rhs_matrix,
      d_output_matrix,

      nOri,
      mOri,
      epoches,
      parts);    
    checkCuda(cudaMemcpy(output_matrix_, d_output_matrix, c_column.size(0) * sizeof(half), cudaMemcpyDeviceToHost));

    cudaFree(d_c_row_offsets);
    cudaFree(d_c_row);
    cudaFree(d_c_column);

    cudaFree(d_rhs_matrix);
    cudaFree(d_output_matrix);
    cudaDeviceSynchronize(); 
    return {output_matrix, torch::tensor(spmm_ms_avg)};
}


// // tcu cuda
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
    int parts
    );
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
    int parts
    );
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
    int parts
    );
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
    int parts
    );
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
    int parts
    );
std::vector<torch::Tensor> spmm_forward_tf32_tcu_cuda(
    torch::Tensor t_row_offset,
    torch::Tensor t_blockNew_offset, 
    torch::Tensor t_column, 
    torch::Tensor t_window_row,
    torch::Tensor t_binary, 

    torch::Tensor c_row_offsets,
    torch::Tensor c_row, 
    torch::Tensor c_column, 

    torch::Tensor rhs_matrix,

    const int nnz,
    const int maxPart,
    const int parts_t,
    const int nOri,
    const int mOri,
    const int kOri,

    const int parts,
    int epoches,
    int type)
{
    auto output_matrix = torch::zeros({c_column.size(0)+nnz}, torch::kFloat32).to(torch::kCPU);
    // std::cout << c_column.size(0)+nnz <<std::endl;

    //把CPU端的tensor转成C++的数据结构
    int * t_row_offset_ = t_row_offset.data<int>();
    int * t_blockNew_offset_ = t_blockNew_offset.data<int>();
    int * t_column_ = t_column.data<int>();
    int * t_window_row_ = t_window_row.data<int>();
    long * t_binary_ = t_binary.data<long>();
    
    int * c_row_offsets_ = c_row_offsets.data<int>();
    int * c_row_ = c_row.data<int>();
    int * c_column_ = c_column.data<int>();


    float * rhs_matrix_ = rhs_matrix.data<float>(); 
    float * output_matrix_ = output_matrix.data<float>();

    // Device
    int *d_t_row_offset, *d_t_blockNew_offset, *d_t_column;
    int *d_t_window_row;
    long *d_t_binary;
    int *d_c_row_offsets, *d_c_row, *d_c_column;
	float *d_rhs_matrix;
    float *d_output_matrix;

    checkCuda(cudaMalloc(&d_t_row_offset, (t_row_offset.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_blockNew_offset, (t_blockNew_offset.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_column, (t_column.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_window_row, (t_window_row.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_binary, (t_binary.size(0)) * sizeof(long)));

    checkCuda(cudaMalloc(&d_c_row_offsets, (c_row_offsets.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_row, (c_row.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_column, (c_column.size(0)) * sizeof(int)));

    checkCuda(cudaMalloc(&d_rhs_matrix, (kOri*nOri) * sizeof(float)));
    checkCuda(cudaMalloc(&d_output_matrix, (c_column.size(0)+nnz) * sizeof(float)));

    //把CPU数据放到GPU上
    checkCuda(cudaMemcpy(d_t_row_offset, t_row_offset_ , (t_row_offset.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_blockNew_offset, t_blockNew_offset_ , (t_blockNew_offset.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_column, t_column_, (t_column.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_window_row, t_window_row_, (t_window_row.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_binary, t_binary_, (t_binary.size(0)) * sizeof(long), cudaMemcpyHostToDevice));

    checkCuda(cudaMemcpy(d_c_row_offsets, c_row_offsets_ , (c_row_offsets.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_row, c_row_, (c_row.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_column, c_column_, (c_column.size(0)) * sizeof(int), cudaMemcpyHostToDevice));

    checkCuda(cudaMemcpy(d_rhs_matrix, rhs_matrix_,(kOri*nOri) * sizeof(float), cudaMemcpyHostToDevice));
    float spmm_ms_avg = 0.0;
    if(nOri==32){
    spmm_ms_avg =  spmm_forward_tf32_tcu_cuda_kernel_32(
      d_t_row_offset,
      d_t_blockNew_offset,
      d_t_column,
      d_t_window_row,
      d_t_binary,
      
      d_c_row_offsets, 
      d_c_row, 
      d_c_column,

      d_rhs_matrix,
      d_output_matrix,
      d_output_matrix + nnz,

      parts_t,
      maxPart,
      nOri,
      mOri,
      epoches,
      parts); }
      else if(nOri==128){
        spmm_ms_avg =  spmm_forward_tf32_tcu_cuda_kernel_128(
        d_t_row_offset,
        d_t_blockNew_offset,
        d_t_column,
        d_t_window_row,
        d_t_binary,
        
        d_c_row_offsets, 
        d_c_row, 
        d_c_column,

        d_rhs_matrix,
        d_output_matrix,
        d_output_matrix + nnz,

        parts_t,
        maxPart,
        nOri,
        mOri,
        epoches,
        parts);
      }else{
        spmm_ms_avg =  spmm_forward_tf32_tcu_cuda_kernel(
        d_t_row_offset,
        d_t_blockNew_offset,
        d_t_column,
        d_t_window_row,
        d_t_binary,
        
        d_c_row_offsets, 
        d_c_row, 
        d_c_column,

        d_rhs_matrix,
        d_output_matrix,
        d_output_matrix + nnz,

        parts_t,
        maxPart,
        nOri,
        mOri,
        epoches,
        parts);

      }
    checkCuda(cudaMemcpy(output_matrix_, d_output_matrix, (c_column.size(0)+nnz) * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_t_row_offset);
    cudaFree(d_t_blockNew_offset);
    cudaFree(d_t_column),
    cudaFree(d_t_window_row);
    cudaFree(d_t_binary);

    cudaFree(d_c_row_offsets);
    cudaFree(d_c_row);
    cudaFree(d_c_column);

    cudaFree(d_rhs_matrix);
    cudaFree(d_output_matrix);
    // cudaDeviceSynchronize(); 
    return {output_matrix, torch::tensor(spmm_ms_avg)};
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
    int parts
    );
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
    int parts
    );
std::vector<torch::Tensor> spmm_forward_fp16_tcu_cuda(
    torch::Tensor t_row_offset,
    torch::Tensor t_blockNew_offset, 
    torch::Tensor t_column, 
    torch::Tensor t_window_row,
    torch::Tensor t_binary, 

    torch::Tensor c_row_offsets,
    torch::Tensor c_row, 
    torch::Tensor c_column, 

    torch::Tensor rhs_matrix,

    const int nnz,
    const int maxPart,
    const int parts_t,
    const int nOri,
    const int mOri,
    const int kOri,

    const int parts,
    int epoches,
    int type)
{
    auto output_matrix = torch::zeros({c_column.size(0)+nnz}, torch::kFloat16).to(torch::kCPU);
    // std::cout << c_column.size(0)+nnz <<std::endl;

    //把CPU端的tensor转成C++的数据结构
    int * t_row_offset_ = t_row_offset.data<int>();
    int * t_blockNew_offset_ = t_blockNew_offset.data<int>();
    int * t_column_ = t_column.data<int>();
    int * t_window_row_ = t_window_row.data<int>();
    long * t_binary_ = t_binary.data<long>();
    
    int * c_row_offsets_ = c_row_offsets.data<int>();
    int * c_row_ = c_row.data<int>();
    int * c_column_ = c_column.data<int>();


    half * rhs_matrix_ = reinterpret_cast<half *>(rhs_matrix.data<at::Half>()); 
    half * output_matrix_ = reinterpret_cast<half *>(output_matrix.data<at::Half>());

    // Device
    int *d_t_row_offset, *d_t_blockNew_offset, *d_t_column;
    int *d_t_window_row;
    long *d_t_binary;
    int *d_c_row_offsets, *d_c_row, *d_c_column;
	half *d_rhs_matrix;
    half *d_output_matrix;

    checkCuda(cudaMalloc(&d_t_row_offset, (t_row_offset.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_blockNew_offset, (t_blockNew_offset.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_column, (t_column.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_window_row, (t_window_row.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_binary, (t_binary.size(0)) * sizeof(long)));

    checkCuda(cudaMalloc(&d_c_row_offsets, (c_row_offsets.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_row, (c_row.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_column, (c_column.size(0)) * sizeof(int)));

    checkCuda(cudaMalloc(&d_rhs_matrix, (kOri*nOri) * sizeof(half)));
    checkCuda(cudaMalloc(&d_output_matrix, (c_column.size(0)+nnz) * sizeof(half)));

    //把CPU数据放到GPU上
    checkCuda(cudaMemcpy(d_t_row_offset, t_row_offset_ , (t_row_offset.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_blockNew_offset, t_blockNew_offset_ , (t_blockNew_offset.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_column, t_column_, (t_column.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_window_row, t_window_row_, (t_window_row.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_binary, t_binary_, (t_binary.size(0)) * sizeof(long), cudaMemcpyHostToDevice));

    checkCuda(cudaMemcpy(d_c_row_offsets, c_row_offsets_ , (c_row_offsets.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_row, c_row_, (c_row.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_column, c_column_, (c_column.size(0)) * sizeof(int), cudaMemcpyHostToDevice));

    checkCuda(cudaMemcpy(d_rhs_matrix, rhs_matrix_,(kOri*nOri) * sizeof(half), cudaMemcpyHostToDevice));
    float spmm_ms_avg = 0.0;
    if(type==0){
    spmm_ms_avg =  spmm_forward_fp16_tcu_cuda_kernel(
      d_t_row_offset,
      d_t_blockNew_offset,
      d_t_column,
      d_t_window_row,
      d_t_binary,
      
      d_c_row_offsets, 
      d_c_row, 
      d_c_column,

      d_rhs_matrix,
      d_output_matrix,
      d_output_matrix + nnz,

      parts_t,
      maxPart,
      nOri,
      mOri,
      epoches,
      parts); }else{
    spmm_ms_avg =  spmm_forward_fp16_tcu_cuda_kernel_navie(
      d_t_row_offset,
      d_t_blockNew_offset,
      d_t_column,
      d_t_window_row,
      d_t_binary,
      
      d_c_row_offsets, 
      d_c_row, 
      d_c_column,

      d_rhs_matrix,
      d_output_matrix,
      d_output_matrix + nnz,

      parts_t,
      maxPart,
      nOri,
      mOri,
      epoches,
      parts);
      }
    checkCuda(cudaMemcpy(output_matrix_, d_output_matrix, (c_column.size(0)+nnz) * sizeof(half), cudaMemcpyDeviceToHost));
    cudaFree(d_t_row_offset);
    cudaFree(d_t_blockNew_offset);
    cudaFree(d_t_column),
    cudaFree(d_t_window_row);
    cudaFree(d_t_binary);

    cudaFree(d_c_row_offsets);
    cudaFree(d_c_row);
    cudaFree(d_c_column);

    cudaFree(d_rhs_matrix);
    cudaFree(d_output_matrix);
    // cudaDeviceSynchronize(); 
    return {output_matrix, torch::tensor(spmm_ms_avg)};
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
    int parts
    );

std::vector<torch::Tensor> spmm_forward_fp16_tcu_cuda_cc_only(
    torch::Tensor t_row_offset,
    torch::Tensor t_blockNew_offset, 
    torch::Tensor t_column, 
    torch::Tensor t_window_row,
    torch::Tensor t_binary, 

    torch::Tensor c_row_offsets,
    torch::Tensor c_row, 
    torch::Tensor c_column, 

    torch::Tensor rhs_matrix,

    const int nnz,
    const int maxPart,
    const int parts_t,
    const int nOri,
    const int mOri,
    const int kOri,

    const int parts,
    int epoches,
    int type)
{
    auto output_matrix = torch::zeros({c_column.size(0)+nnz}, torch::kFloat16).to(torch::kCPU);
    // std::cout << c_column.size(0)+nnz <<std::endl;

    //把CPU端的tensor转成C++的数据结构
    int * t_row_offset_ = t_row_offset.data<int>();
    int * t_blockNew_offset_ = t_blockNew_offset.data<int>();
    int * t_column_ = t_column.data<int>();
    int * t_window_row_ = t_window_row.data<int>();
    long * t_binary_ = t_binary.data<long>();
    
    int * c_row_offsets_ = c_row_offsets.data<int>();
    int * c_row_ = c_row.data<int>();
    int * c_column_ = c_column.data<int>();


    half * rhs_matrix_ = reinterpret_cast<half *>(rhs_matrix.data<at::Half>()); 
    half * output_matrix_ = reinterpret_cast<half *>(output_matrix.data<at::Half>());

    // Device
    int *d_t_row_offset, *d_t_blockNew_offset, *d_t_column;
    int *d_t_window_row;
    long *d_t_binary;
    int *d_c_row_offsets, *d_c_row, *d_c_column;
	half *d_rhs_matrix;
    half *d_output_matrix;

    checkCuda(cudaMalloc(&d_t_row_offset, (t_row_offset.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_blockNew_offset, (t_blockNew_offset.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_column, (t_column.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_window_row, (t_window_row.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_binary, (t_binary.size(0)) * sizeof(long)));

    checkCuda(cudaMalloc(&d_c_row_offsets, (c_row_offsets.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_row, (c_row.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_column, (c_column.size(0)) * sizeof(int)));

    checkCuda(cudaMalloc(&d_rhs_matrix, (kOri*nOri) * sizeof(half)));
    checkCuda(cudaMalloc(&d_output_matrix, (c_column.size(0)+nnz) * sizeof(half)));

    //把CPU数据放到GPU上
    checkCuda(cudaMemcpy(d_t_row_offset, t_row_offset_ , (t_row_offset.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_blockNew_offset, t_blockNew_offset_ , (t_blockNew_offset.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_column, t_column_, (t_column.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_window_row, t_window_row_, (t_window_row.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_binary, t_binary_, (t_binary.size(0)) * sizeof(long), cudaMemcpyHostToDevice));

    checkCuda(cudaMemcpy(d_c_row_offsets, c_row_offsets_ , (c_row_offsets.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_row, c_row_, (c_row.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_column, c_column_, (c_column.size(0)) * sizeof(int), cudaMemcpyHostToDevice));

    checkCuda(cudaMemcpy(d_rhs_matrix, rhs_matrix_,(kOri*nOri) * sizeof(half), cudaMemcpyHostToDevice));
    float spmm_ms_avg = 0.0;
    if(type==0){
    spmm_ms_avg =  spmm_forward_fp16_tcu_cuda_kernel_cc_only(
      d_t_row_offset,
      d_t_blockNew_offset,
      d_t_column,
      d_t_window_row,
      d_t_binary,
      
      d_c_row_offsets, 
      d_c_row, 
      d_c_column,

      d_rhs_matrix,
      d_output_matrix,
      d_output_matrix + nnz,

      parts_t,
      maxPart,
      nOri,
      mOri,
      epoches,
      parts); }else{
    spmm_ms_avg =  spmm_forward_fp16_tcu_cuda_kernel_navie(
      d_t_row_offset,
      d_t_blockNew_offset,
      d_t_column,
      d_t_window_row,
      d_t_binary,
      
      d_c_row_offsets, 
      d_c_row, 
      d_c_column,

      d_rhs_matrix,
      d_output_matrix,
      d_output_matrix + nnz,

      parts_t,
      maxPart,
      nOri,
      mOri,
      epoches,
      parts);
      }
    checkCuda(cudaMemcpy(output_matrix_, d_output_matrix, (c_column.size(0)+nnz) * sizeof(half), cudaMemcpyDeviceToHost));
    cudaFree(d_t_row_offset);
    cudaFree(d_t_blockNew_offset);
    cudaFree(d_t_column),
    cudaFree(d_t_window_row);
    cudaFree(d_t_binary);

    cudaFree(d_c_row_offsets);
    cudaFree(d_c_row);
    cudaFree(d_c_column);

    cudaFree(d_rhs_matrix);
    cudaFree(d_output_matrix);
    // cudaDeviceSynchronize(); 
    return {output_matrix, torch::tensor(spmm_ms_avg)};
}

//csr
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
    int epoches);

std::vector<torch::Tensor> sddmm_forward_tf32_tcu_csr_part(
    torch::Tensor t_row_offset,
    torch::Tensor t_blockNew_offset, 
    torch::Tensor t_column, 
    torch::Tensor t_window_row,
    torch::Tensor t_row, 

    torch::Tensor rhs_matrix,

    const int nnz,
    const int maxPart,
    const int parts_t,
    const int dimN,
    const int mOri,
    int epoches)
{
    auto output_matrix = torch::zeros({nnz}, torch::kFloat32).to(torch::kCPU);
    
    //把CPU端的tensor转成C++的数据结构
    int * t_row_offset_ = t_row_offset.data<int>();
    int * t_blockNew_offset_ = t_blockNew_offset.data<int>();
    int * t_column_ = t_column.data<int>();
    int * t_window_row_ = t_window_row.data<int>();
    int * t_row_ = t_row.data<int>();
    

    float * rhs_matrix_ = rhs_matrix.data<float>(); 
    float * output_matrix_ = output_matrix.data<float>();

    // Device
    int *d_t_row_offset, *d_t_blockNew_offset, *d_t_column;
    int *d_t_window_row;
    int *d_t_row;
	float *d_rhs_matrix;
    float *d_output_matrix;

    checkCuda(cudaMalloc(&d_t_row_offset, (t_row_offset.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_blockNew_offset, (t_blockNew_offset.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_column, (t_column.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_window_row, (t_window_row.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_row, (t_row.size(0)) * sizeof(int)));


    checkCuda(cudaMalloc(&d_rhs_matrix, (mOri*dimN) * sizeof(float)));
    checkCuda(cudaMalloc(&d_output_matrix, nnz * sizeof(float)));

    //把CPU数据放到GPU上
    checkCuda(cudaMemcpy(d_t_row_offset, t_row_offset_ , (t_row_offset.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_blockNew_offset, t_blockNew_offset_ , (t_blockNew_offset.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_column, t_column_, (t_column.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_window_row, t_window_row_, (t_window_row.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_row, t_row_, (t_row.size(0)) * sizeof(int), cudaMemcpyHostToDevice));

    checkCuda(cudaMemcpy(d_rhs_matrix, rhs_matrix_,(mOri*dimN) * sizeof(float), cudaMemcpyHostToDevice));

    float spmm_ms_avg =  sddmm_forward_tf32_tcu_part_kernel_csr(
      d_t_row_offset,
      d_t_blockNew_offset,
      d_t_column,
      d_t_window_row,
      d_t_row,

      d_rhs_matrix,
      d_output_matrix,

      parts_t,
      maxPart,
      dimN,
      mOri,
      epoches);    
    checkCuda(cudaMemcpy(output_matrix_, d_output_matrix, nnz * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_t_row_offset);
    cudaFree(d_t_blockNew_offset);
    cudaFree(d_t_column),
    cudaFree(d_t_window_row);
    cudaFree(d_t_row);

    cudaFree(d_rhs_matrix);
    cudaFree(d_output_matrix);
    cudaDeviceSynchronize(); 
    return {output_matrix, torch::tensor(spmm_ms_avg)};
}


// fp16
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
    int epoches);

std::vector<torch::Tensor> sddmm_forward_fp16_tcu_csr_part(
    torch::Tensor t_row_offset,
    torch::Tensor t_blockNew_offset, 
    torch::Tensor t_column, 
    torch::Tensor t_window_row,
    torch::Tensor t_row, 

    torch::Tensor rhs_matrix,

    const int nnz,
    const int maxPart,
    const int parts_t,
    const int dimN,
    const int mOri,
    int epoches)
{
    auto output_matrix = torch::zeros({nnz}, torch::kFloat16).to(torch::kCPU);
    
    //把CPU端的tensor转成C++的数据结构
    int * t_row_offset_ = t_row_offset.data<int>();
    int * t_blockNew_offset_ = t_blockNew_offset.data<int>();
    int * t_column_ = t_column.data<int>();
    int * t_window_row_ = t_window_row.data<int>();
    int * t_row_ = t_row.data<int>();
    

    half * rhs_matrix_ = reinterpret_cast<half *>(rhs_matrix.data<at::Half>()); 
    half * output_matrix_ = reinterpret_cast<half *>(output_matrix.data<at::Half>());

    // Device
    int *d_t_row_offset, *d_t_blockNew_offset, *d_t_column;
    int *d_t_window_row;
    int *d_t_row;
	half *d_rhs_matrix;
    half *d_output_matrix;

    checkCuda(cudaMalloc(&d_t_row_offset, (t_row_offset.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_blockNew_offset, (t_blockNew_offset.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_column, (t_column.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_window_row, (t_window_row.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_row, (t_row.size(0)) * sizeof(int)));


    checkCuda(cudaMalloc(&d_rhs_matrix, (mOri*dimN) * sizeof(half)));
    checkCuda(cudaMalloc(&d_output_matrix, nnz * sizeof(half)));

    //把CPU数据放到GPU上
    checkCuda(cudaMemcpy(d_t_row_offset, t_row_offset_ , (t_row_offset.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_blockNew_offset, t_blockNew_offset_ , (t_blockNew_offset.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_column, t_column_, (t_column.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_window_row, t_window_row_, (t_window_row.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_row, t_row_, (t_row.size(0)) * sizeof(int), cudaMemcpyHostToDevice));

    checkCuda(cudaMemcpy(d_rhs_matrix, rhs_matrix_,(mOri*dimN) * sizeof(half), cudaMemcpyHostToDevice));

    float spmm_ms_avg =  sddmm_forward_fp16_tcu_part_kernel_csr(
      d_t_row_offset,
      d_t_blockNew_offset,
      d_t_column,
      d_t_window_row,
      d_t_row,

      d_rhs_matrix,
      d_output_matrix,

      parts_t,
      maxPart,
      dimN,
      mOri,
      epoches);    
    checkCuda(cudaMemcpy(output_matrix_, d_output_matrix, nnz * sizeof(half), cudaMemcpyDeviceToHost));
    cudaFree(d_t_row_offset);
    cudaFree(d_t_blockNew_offset);
    cudaFree(d_t_column),
    cudaFree(d_t_window_row);
    cudaFree(d_t_row);

    cudaFree(d_rhs_matrix);
    cudaFree(d_output_matrix);
    cudaDeviceSynchronize(); 
    return {output_matrix, torch::tensor(spmm_ms_avg)};
}

// tcu cuda
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
    int epoches);

std::vector<torch::Tensor> sddmm_forward_tf32_tcu_tcf_part(
    torch::Tensor t_row_offset,
    torch::Tensor t_blockNew_offset, 
    torch::Tensor t_tcl, 
    torch::Tensor t_column, 
    torch::Tensor t_window_row,

    torch::Tensor rhs_matrix,

    const int nnz,
    const int maxPart,
    const int parts_t,
    const int dimN,
    const int mOri,
    const int kOri,
    int epoches)
{
    auto output_matrix = torch::zeros({nnz}, torch::kFloat32).to(torch::kCPU);
    
    //把CPU端的tensor转成C++的数据结构
    int * t_row_offset_ = t_row_offset.data<int>();
    int * t_blockNew_offset_ = t_blockNew_offset.data<int>();
    int * t_tcl_ = t_tcl.data<int>();
    int * t_column_ = t_column.data<int>();
    int * t_window_row_ = t_window_row.data<int>();
    

    float * rhs_matrix_ = rhs_matrix.data<float>(); 
    float * output_matrix_ = output_matrix.data<float>();

    // Device
    int *d_t_row_offset, *d_t_blockNew_offset, *d_t_column;
    int *d_t_window_row;
    int *d_t_tcl;
	float *d_rhs_matrix;
    float *d_output_matrix;

    checkCuda(cudaMalloc(&d_t_row_offset, (t_row_offset.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_blockNew_offset, (t_blockNew_offset.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_column, (t_column.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_window_row, (t_window_row.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_tcl, (t_tcl.size(0)) * sizeof(int)));


    checkCuda(cudaMalloc(&d_rhs_matrix, (kOri*dimN) * sizeof(float)));
    checkCuda(cudaMalloc(&d_output_matrix, nnz * sizeof(float)));

    //把CPU数据放到GPU上
    checkCuda(cudaMemcpy(d_t_row_offset, t_row_offset_ , (t_row_offset.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_blockNew_offset, t_blockNew_offset_ , (t_blockNew_offset.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_column, t_column_, (t_column.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_window_row, t_window_row_, (t_window_row.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_tcl, t_tcl_, (t_tcl.size(0)) * sizeof(int), cudaMemcpyHostToDevice));

    checkCuda(cudaMemcpy(d_rhs_matrix, rhs_matrix_,(kOri*dimN) * sizeof(float), cudaMemcpyHostToDevice));

    float spmm_ms_avg =  sddmm_forward_tf32_tcu_part_kernel_tcf(
      d_t_row_offset,
      d_t_blockNew_offset,
      d_t_tcl,
      d_t_column,
      d_t_window_row,

      d_rhs_matrix,
      d_output_matrix,

      parts_t,
      maxPart,
      dimN,
      mOri,
      epoches);    
    checkCuda(cudaMemcpy(output_matrix_, d_output_matrix, nnz * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_t_row_offset);
    cudaFree(d_t_blockNew_offset);
    cudaFree(d_t_column),
    cudaFree(d_t_window_row);
    cudaFree(d_t_tcl);

    cudaFree(d_rhs_matrix);
    cudaFree(d_output_matrix);
    cudaDeviceSynchronize(); 
    return {output_matrix, torch::tensor(spmm_ms_avg)};
}

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
    int epoches);

std::vector<torch::Tensor> spmm_forward_fp16_tcu_binary_part(
    torch::Tensor t_row_offset,
    torch::Tensor t_blockNew_offset, 
    torch::Tensor t_column, 
    torch::Tensor t_window_row,
    torch::Tensor t_binary, 

    torch::Tensor rhs_matrix,

    const int nnz,
    const int maxPart,
    const int parts_t,
    const int dimN,
    const int mOri,
    const int kOri,
    int epoches)
{
    auto output_matrix = torch::zeros({nnz}, torch::kFloat16).to(torch::kCPU);
    
    //把CPU端的tensor转成C++的数据结构
    int * t_row_offset_ = t_row_offset.data<int>();
    int * t_blockNew_offset_ = t_blockNew_offset.data<int>();
    int * t_column_ = t_column.data<int>();
    int * t_window_row_ = t_window_row.data<int>();
    long * t_binary_ = t_binary.data<long>();
    

    half * rhs_matrix_ = reinterpret_cast<half *>(rhs_matrix.data<at::Half>()); 
    half * output_matrix_ = reinterpret_cast<half *>(output_matrix.data<at::Half>());

    // Device
    int *d_t_row_offset, *d_t_blockNew_offset, *d_t_column;
    int *d_t_window_row;
    long *d_t_binary;
	half *d_rhs_matrix;
    half *d_output_matrix;

    checkCuda(cudaMalloc(&d_t_row_offset, (t_row_offset.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_blockNew_offset, (t_blockNew_offset.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_column, (t_column.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_window_row, (t_window_row.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_binary, (t_binary.size(0)) * sizeof(long)));


    checkCuda(cudaMalloc(&d_rhs_matrix, (kOri*dimN) * sizeof(half)));
    checkCuda(cudaMalloc(&d_output_matrix, nnz * sizeof(half)));

    //把CPU数据放到GPU上
    checkCuda(cudaMemcpy(d_t_row_offset, t_row_offset_ , (t_row_offset.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_blockNew_offset, t_blockNew_offset_ , (t_blockNew_offset.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_column, t_column_, (t_column.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_window_row, t_window_row_, (t_window_row.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_binary, t_binary_, (t_binary.size(0)) * sizeof(long), cudaMemcpyHostToDevice));

    checkCuda(cudaMemcpy(d_rhs_matrix, rhs_matrix_,(kOri*dimN) * sizeof(half), cudaMemcpyHostToDevice));

    float spmm_ms_avg =  sddmm_forward_fp16_tcu_part_kernel(
      d_t_row_offset,
      d_t_blockNew_offset,
      d_t_column,
      d_t_window_row,
      d_t_binary,

      d_rhs_matrix,
      d_output_matrix,

      parts_t,
      maxPart,
      dimN,
      mOri,
      epoches);    
    checkCuda(cudaMemcpy(output_matrix_, d_output_matrix, nnz * sizeof(half), cudaMemcpyDeviceToHost));
    cudaFree(d_t_row_offset);
    cudaFree(d_t_blockNew_offset);
    cudaFree(d_t_column),
    cudaFree(d_t_window_row);
    cudaFree(d_t_binary);

    cudaFree(d_rhs_matrix);
    cudaFree(d_output_matrix);
    cudaDeviceSynchronize(); 
    return {output_matrix, torch::tensor(spmm_ms_avg)};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

    // tcu
    // m.def("forward_tf32_tcu_csr_part", &sddmm_forward_tf32_tcu_csr_part, "one kernel");
    // m.def("forward_fp16_tcu_csr_part", &sddmm_forward_fp16_tcu_csr_part, "one kernel");

    m.def("forward_tf32_tcu_tcf_part", &sddmm_forward_tf32_tcu_tcf_part, "one kernel");
    m.def("forward_fp16_tcu_binary_part", &spmm_forward_fp16_tcu_binary_part, "one kernel");
  
    // cuda
    m.def("forward_tf32_cuda_v2", &sddmm_forward_tf32_cuda_v2, "长短行");
    // m.def("forward_tf32_cuda_v2_navie", &sddmm_forward_tf32_cuda_v2_navie, "长短行");

    m.def("forward_fp16_cuda_v2", &sddmm_forward_fp16_cuda_v2, "长短行");
    // m.def("forward_fp16_cuda_v2_navie", &sddmm_forward_fp16_cuda_v2_navie, "长短行");

    // cuda + tcu 
    m.def("forward_tf32_tcu_cuda", &spmm_forward_tf32_tcu_cuda, "one kernel");
    m.def("forward_fp16_tcu_cuda", &spmm_forward_fp16_tcu_cuda, "one kernel");
    m.def("forward_fp16_tcu_cuda_cc_only", &spmm_forward_fp16_tcu_cuda_cc_only, "one kernel");
  }