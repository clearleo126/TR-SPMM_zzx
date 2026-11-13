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
//tcu_sgt
float spmm_forward_tf32_tcu_kernel_sgt(
    int * t_windowNew_offset,
    int * t_blockNew_offset,
    float * t_value, 
    int * t_column, 
    int*  t_row,
    int*  t_col,
    // int* t_window_row,

    float * rhs_matrix,
    float * output_matrix,

    const int dimM,
    const int dimN,
    const int mOri,
    int epoches);

std::vector<torch::Tensor> spmm_forward_tf32_tcu_sgt(
    torch::Tensor blockPartition,
    torch::Tensor row_pointer, 
    torch::Tensor t_column,
    torch::Tensor t_value, 
    torch::Tensor t_row, 
    torch::Tensor t_col, 
    // torch::Tensor t_window_row,
    
    torch::Tensor rhs_matrix,

    const int dimM1,
    const int dimN,
    const int mOri,
    const int kOri,
    int epoches)
{
    int dimM=dimM1/8;
    auto output_matrix = torch::zeros({mOri, dimN}, torch::kFloat32).to(torch::kCPU);
    
    //把CPU端的tensor转成C++的数据结构
    int * t_windowNew_offset_ = blockPartition.data<int>();
    int * t_blockNew_offset_ = row_pointer.data<int>();
    float * t_value_ = t_value.data<float>(); 
    int * t_column_ = t_column.data<int>();
    int * t_row_ = t_row.data<int>();
    int * t_col_ = t_col.data<int>();
    // int * t_window_row_ = t_window_row.data<int>();

    float * rhs_matrix_ = rhs_matrix.data<float>(); 
    float * output_matrix_ = output_matrix.data<float>();

    // Device
    int *d_t_windowNew_offset, *d_t_blockNew_offset, *d_t_row, *d_t_col, *d_t_column;
    float *d_t_value; 
	float *d_rhs_matrix;
    float *d_output_matrix;

    checkCuda(cudaMalloc(&d_t_windowNew_offset, (blockPartition.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_blockNew_offset, (row_pointer.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_value, (t_value.size(0)) * sizeof(float)));
    checkCuda(cudaMalloc(&d_t_column, (t_column.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_row, (t_row.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_col, (t_col.size(0)) * sizeof(int)));
    // checkCuda(cudaMalloc(&d_t_window_row, (t_window_row.size(0)) * sizeof(int)));

    checkCuda(cudaMalloc(&d_rhs_matrix, (kOri*dimN) * sizeof(float)));
    checkCuda(cudaMalloc(&d_output_matrix, (mOri*dimN) * sizeof(float)));

    //把CPU数据放到GPU上
    checkCuda(cudaMemcpy(d_t_windowNew_offset, t_windowNew_offset_ , (blockPartition.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_blockNew_offset, t_blockNew_offset_ , (row_pointer.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_value, t_value_, (t_value.size(0)) * sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_column, t_column_, (t_column.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_row, t_row_, (t_row.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_col, t_col_, (t_col.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    // checkCuda(cudaMemcpy(d_t_window_row, t_window_row_, (t_window_row.size(0)) * sizeof(int), cudaMemcpyHostToDevice));


    checkCuda(cudaMemcpy(d_rhs_matrix, rhs_matrix_,(kOri*dimN) * sizeof(float), cudaMemcpyHostToDevice));

    float spmm_ms_avg =  spmm_forward_tf32_tcu_kernel_sgt(
      d_t_windowNew_offset,
      d_t_blockNew_offset, 
      d_t_value, 
      d_t_column,
      d_t_row,
      d_t_col, 
    //   d_t_window_row,
      d_rhs_matrix,
      d_output_matrix,

      dimM,
      dimN,
      mOri,
      epoches);    
    checkCuda(cudaMemcpy(output_matrix_, d_output_matrix, mOri * dimN * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_t_windowNew_offset);
    cudaFree(d_t_blockNew_offset);
    cudaFree(d_t_value);
    cudaFree(d_t_column),
    cudaFree(d_t_row);
    cudaFree(d_t_col);
    // cudaFree(d_t_window_row);

    cudaFree(d_rhs_matrix);
    cudaFree(d_output_matrix);
    cudaDeviceSynchronize(); 
    return {output_matrix, torch::tensor(spmm_ms_avg)};
}

//tcu
float spmm_forward_fp16_tcu_kernel_sgt(
    int * t_windowNew_offset,
    int * t_blockNew_offset,
    int * t_column, 
    half * t_value, 
    int*  t_row,
    int*  t_col,
    // int* t_window_row,

    half * rhs_matrix,
    half * output_matrix,

    const int dimM,
    const int dimN,
    const int mOri,
    int epoches);

std::vector<torch::Tensor> spmm_forward_fp16_tcu_sgt(
    torch::Tensor blockPartition,
    torch::Tensor row_pointer, 
    torch::Tensor t_column,
    torch::Tensor t_value, 
    torch::Tensor t_row, 
    torch::Tensor t_col, 
    
    torch::Tensor rhs_matrix,

    const int dimM1,
    const int dimN,
    const int mOri,
    const int kOri,
    int epoches)
{
    int dimM=dimM1/8;
    auto output_matrix = torch::zeros({mOri, dimN}, torch::kFloat16).to(torch::kCPU);
    
    //把CPU端的tensor转成C++的数据结构
    int * t_windowNew_offset_ = blockPartition.data<int>();
    int * t_blockNew_offset_ = row_pointer.data<int>();
    half * t_value_ = reinterpret_cast<half *>(t_value.data<at::Half>()); 
    int * t_column_ = t_column.data<int>();
    int * t_row_ = t_row.data<int>();
    int * t_col_ = t_col.data<int>();
    // int * t_window_row_ = t_window_row.data<int>();

    half * rhs_matrix_ = reinterpret_cast<half *>(rhs_matrix.data<at::Half>()); 
    half * output_matrix_ = reinterpret_cast<half *>(output_matrix.data<at::Half>());

    // Device
    // int *d_t_windowNew_offset, *d_t_blockNew_offset, *d_t_column, *d_t_row, *d_t_col, *d_t_window_row;
    int *d_t_windowNew_offset, *d_t_blockNew_offset, *d_t_row, *d_t_col, *d_t_column;
    half *d_t_value; 
	half *d_rhs_matrix;
    half *d_output_matrix;

    checkCuda(cudaMalloc(&d_t_windowNew_offset, (blockPartition.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_blockNew_offset, (row_pointer.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_value, (t_value.size(0)) * sizeof(half)));
    checkCuda(cudaMalloc(&d_t_column, (t_column.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_row, (t_row.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_col, (t_col.size(0)) * sizeof(int)));
    // checkCuda(cudaMalloc(&d_t_window_row, (t_window_row.size(0)) * sizeof(int)));

    checkCuda(cudaMalloc(&d_rhs_matrix, (kOri*dimN) * sizeof(half)));
    checkCuda(cudaMalloc(&d_output_matrix, (mOri*dimN) * sizeof(half)));

    //把CPU数据放到GPU上
    checkCuda(cudaMemcpy(d_t_windowNew_offset, t_windowNew_offset_ , (blockPartition.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_blockNew_offset, t_blockNew_offset_ , (row_pointer.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_value, t_value_, (t_value.size(0)) * sizeof(half), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_column, t_column_, (t_column.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_row, t_row_, (t_row.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_col, t_col_, (t_col.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    // checkCuda(cudaMemcpy(d_t_window_row, t_window_row_, (t_window_row.size(0)) * sizeof(int), cudaMemcpyHostToDevice));


    checkCuda(cudaMemcpy(d_rhs_matrix, rhs_matrix_,(kOri*dimN) * sizeof(half), cudaMemcpyHostToDevice));

    float spmm_ms_avg =  spmm_forward_fp16_tcu_kernel_sgt(
      d_t_windowNew_offset,
      d_t_blockNew_offset, 
      d_t_column,
      d_t_value, 
      d_t_row,
      d_t_col, 
    //   d_t_window_row,
      d_rhs_matrix,
      d_output_matrix,

      dimM,
      dimN,
      mOri,
      epoches);    
    checkCuda(cudaMemcpy(output_matrix_, d_output_matrix, mOri * dimN * sizeof(half), cudaMemcpyDeviceToHost));
    cudaFree(d_t_windowNew_offset);
    cudaFree(d_t_blockNew_offset);
    cudaFree(d_t_value);
    cudaFree(d_t_column),
    cudaFree(d_t_row);
    cudaFree(d_t_col);
    // cudaFree(d_t_window_row);

    cudaFree(d_rhs_matrix);
    cudaFree(d_output_matrix);
    cudaDeviceSynchronize(); 
    return {output_matrix, torch::tensor(spmm_ms_avg)};
}




//tcu
float spmm_forward_tf32_tcu_kernel(
    int * t_windowNew_offset,
    int * t_blockNew_offset,
    float * t_value, 
    int * t_column, 
    int*  t_row,
    // int*  t_col,
    // int* t_window_row,

    float * rhs_matrix,
    float * output_matrix,

    const int dimM,
    const int dimN,
    const int mOri,
    int epoches);

std::vector<torch::Tensor> spmm_forward_tf32_tcu(
    torch::Tensor t_windowNew_offset,
    torch::Tensor t_blockNew_offset, 
    torch::Tensor t_column,
    torch::Tensor t_value, 
    torch::Tensor t_row, 
    // torch::Tensor t_col, 
    // torch::Tensor t_window_row,
    
    torch::Tensor rhs_matrix,

    const int dimM1,
    const int dimN,
    const int mOri,
    const int kOri,
    int epoches)
{
    int dimM=dimM1/8;
    auto output_matrix = torch::zeros({mOri, dimN}, torch::kFloat32).to(torch::kCPU);
    
    //把CPU端的tensor转成C++的数据结构
    int * t_windowNew_offset_ = t_windowNew_offset.data<int>();
    int * t_blockNew_offset_ = t_blockNew_offset.data<int>();
    float * t_value_ = t_value.data<float>(); 
    int * t_column_ = t_column.data<int>();
    int * t_row_ = t_row.data<int>();
    // int * t_col_ = t_col.data<int>();
    // int * t_window_row_ = t_window_row.data<int>();

    float * rhs_matrix_ = rhs_matrix.data<float>(); 
    float * output_matrix_ = output_matrix.data<float>();

    // Device
    int *d_t_windowNew_offset, *d_t_blockNew_offset, *d_t_column, *d_t_row;
    float *d_t_value; 
	float *d_rhs_matrix;
    float *d_output_matrix;

    checkCuda(cudaMalloc(&d_t_windowNew_offset, (t_windowNew_offset.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_blockNew_offset, (t_blockNew_offset.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_value, (t_value.size(0)) * sizeof(float)));
    checkCuda(cudaMalloc(&d_t_column, (t_column.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_row, (t_row.size(0)) * sizeof(int)));
    // checkCuda(cudaMalloc(&d_t_col, (t_col.size(0)) * sizeof(int)));
    // checkCuda(cudaMalloc(&d_t_window_row, (t_window_row.size(0)) * sizeof(int)));

    checkCuda(cudaMalloc(&d_rhs_matrix, (kOri*dimN) * sizeof(float)));
    checkCuda(cudaMalloc(&d_output_matrix, (mOri*dimN) * sizeof(float)));

    //把CPU数据放到GPU上
    checkCuda(cudaMemcpy(d_t_windowNew_offset, t_windowNew_offset_ , (t_windowNew_offset.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_blockNew_offset, t_blockNew_offset_ , (t_blockNew_offset.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_value, t_value_, (t_value.size(0)) * sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_column, t_column_, (t_column.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_row, t_row_, (t_row.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    //checkCuda(cudaMemcpy(d_t_col, t_col_, (t_col.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    // checkCuda(cudaMemcpy(d_t_window_row, t_window_row_, (t_window_row.size(0)) * sizeof(int), cudaMemcpyHostToDevice));


    checkCuda(cudaMemcpy(d_rhs_matrix, rhs_matrix_,(kOri*dimN) * sizeof(float), cudaMemcpyHostToDevice));

    float spmm_ms_avg =  spmm_forward_tf32_tcu_kernel(
      d_t_windowNew_offset,
      d_t_blockNew_offset, 
      d_t_value, 
      d_t_column,
      d_t_row,
    //   d_t_col, 
    //   d_t_window_row,
      d_rhs_matrix,
      d_output_matrix,

      dimM,
      dimN,
      mOri,
      epoches);    
    checkCuda(cudaMemcpy(output_matrix_, d_output_matrix, mOri * dimN * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_t_windowNew_offset);
    cudaFree(d_t_blockNew_offset);
    cudaFree(d_t_value);
    cudaFree(d_t_column),
    cudaFree(d_t_row);
    // cudaFree(d_t_col);
    // cudaFree(d_t_window_row);

    cudaFree(d_rhs_matrix);
    cudaFree(d_output_matrix);
    cudaDeviceSynchronize(); 
    return {output_matrix, torch::tensor(spmm_ms_avg)};
}


//tcu
float spmm_forward_fp16_tcu_kernel(
    int * t_windowNew_offset,
    int * t_blockNew_offset,
    int * t_column, 
    half * t_value, 
    int*  t_row,
    //int*  t_col,
    // int* t_window_row,

    half * rhs_matrix,
    half * output_matrix,

    const int dimM,
    const int dimN,
    const int mOri,
    int epoches);

std::vector<torch::Tensor> spmm_forward_fp16_tcu(
    torch::Tensor t_windowNew_offset,
    torch::Tensor t_blockNew_offset, 
    torch::Tensor t_column,
    torch::Tensor t_value, 
    torch::Tensor t_row, 
    
    // torch::Tensor t_col, 
    // torch::Tensor t_window_row,
    
    torch::Tensor rhs_matrix,

    const int dimM1,
    const int dimN,
    const int mOri,
    const int kOri,
    int epoches)
{
    int dimM=dimM1/8;
    auto output_matrix = torch::zeros({mOri, dimN}, torch::kFloat16).to(torch::kCPU);
    
    //把CPU端的tensor转成C++的数据结构
    int * t_windowNew_offset_ = t_windowNew_offset.data<int>();
    int * t_blockNew_offset_ = t_blockNew_offset.data<int>();
    half * t_value_ = reinterpret_cast<half *>(t_value.data<at::Half>()); 
    int * t_column_ = t_column.data<int>();
    int * t_row_ = t_row.data<int>();
    // int * t_col_ = t_col.data<int>();
    // int * t_window_row_ = t_window_row.data<int>();

    half * rhs_matrix_ = reinterpret_cast<half *>(rhs_matrix.data<at::Half>()); 
    half * output_matrix_ = reinterpret_cast<half *>(output_matrix.data<at::Half>());

    // Device
    // int *d_t_windowNew_offset, *d_t_blockNew_offset, *d_t_column, *d_t_row, *d_t_col, *d_t_window_row;
    int *d_t_windowNew_offset, *d_t_blockNew_offset, *d_t_column, *d_t_row;
    half *d_t_value; 
	half *d_rhs_matrix;
    half *d_output_matrix;

    checkCuda(cudaMalloc(&d_t_windowNew_offset, (t_windowNew_offset.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_blockNew_offset, (t_blockNew_offset.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_value, (t_value.size(0)) * sizeof(half)));
    checkCuda(cudaMalloc(&d_t_column, (t_column.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_row, (t_row.size(0)) * sizeof(int)));
    //checkCuda(cudaMalloc(&d_t_col, (t_col.size(0)) * sizeof(int)));
    // checkCuda(cudaMalloc(&d_t_window_row, (t_window_row.size(0)) * sizeof(int)));

    checkCuda(cudaMalloc(&d_rhs_matrix, (kOri*dimN) * sizeof(half)));
    checkCuda(cudaMalloc(&d_output_matrix, (mOri*dimN) * sizeof(half)));

    //把CPU数据放到GPU上
    checkCuda(cudaMemcpy(d_t_windowNew_offset, t_windowNew_offset_ , (t_windowNew_offset.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_blockNew_offset, t_blockNew_offset_ , (t_blockNew_offset.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_value, t_value_, (t_value.size(0)) * sizeof(half), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_column, t_column_, (t_column.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_row, t_row_, (t_row.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    //checkCuda(cudaMemcpy(d_t_col, t_col_, (t_col.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    // checkCuda(cudaMemcpy(d_t_window_row, t_window_row_, (t_window_row.size(0)) * sizeof(int), cudaMemcpyHostToDevice));


    checkCuda(cudaMemcpy(d_rhs_matrix, rhs_matrix_,(kOri*dimN) * sizeof(half), cudaMemcpyHostToDevice));

    float spmm_ms_avg =  spmm_forward_fp16_tcu_kernel(
      d_t_windowNew_offset,
      d_t_blockNew_offset, 
      d_t_column,
      d_t_value, 
      d_t_row,
     // d_t_col, 
    //   d_t_window_row,
      d_rhs_matrix,
      d_output_matrix,

      dimM,
      dimN,
      mOri,
      epoches);    
    checkCuda(cudaMemcpy(output_matrix_, d_output_matrix, mOri * dimN * sizeof(half), cudaMemcpyDeviceToHost));
    cudaFree(d_t_windowNew_offset);
    cudaFree(d_t_blockNew_offset);
    cudaFree(d_t_value);
    cudaFree(d_t_column),
    cudaFree(d_t_row);
    //cudaFree(d_t_col);
    // cudaFree(d_t_window_row);

    cudaFree(d_rhs_matrix);
    cudaFree(d_output_matrix);
    cudaDeviceSynchronize(); 
    return {output_matrix, torch::tensor(spmm_ms_avg)};
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
    int epoches);

std::vector<torch::Tensor> spmm_forward_fp16_tcu_stream(
    torch::Tensor t_windowNew_offset,
    torch::Tensor t_blockNew_offset, 
    torch::Tensor t_column,
    torch::Tensor t_value, 
    torch::Tensor t_row, 

    torch::Tensor t_windowNew_offset1,
    torch::Tensor t_blockNew_offset1, 
    torch::Tensor t_column1,
    torch::Tensor t_value1, 
    torch::Tensor t_row1, 
    
    // torch::Tensor t_col, 
    // torch::Tensor t_window_row,
    
    torch::Tensor rhs_matrix,

    const int dimM_,
    const int dimM1_,
    const int dimN,
    const int mOri,
    const int mOri1,
    const int kOri,
    int epoches)
{
    int dimM=dimM_/8;
    int dimM1=dimM1_/8;
    auto output_matrix = torch::zeros({mOri, dimN}, torch::kFloat16).to(torch::kCPU);
    auto output_matrix1 = torch::zeros({mOri1, dimN}, torch::kFloat16).to(torch::kCPU);
    //把CPU端的tensor转成C++的数据结构
    int * t_windowNew_offset_ = t_windowNew_offset.data<int>();
    int * t_blockNew_offset_ = t_blockNew_offset.data<int>();
    half * t_value_ = reinterpret_cast<half *>(t_value.data<at::Half>()); 
    int * t_column_ = t_column.data<int>();
    int * t_row_ = t_row.data<int>();

    int * t_windowNew_offset1_ = t_windowNew_offset1.data<int>();
    int * t_blockNew_offset1_ = t_blockNew_offset1.data<int>();
    half * t_value1_ = reinterpret_cast<half *>(t_value1.data<at::Half>()); 
    int * t_column1_ = t_column1.data<int>();
    int * t_row1_ = t_row1.data<int>();

    half * rhs_matrix_ = reinterpret_cast<half *>(rhs_matrix.data<at::Half>()); 
    half * output_matrix_ = reinterpret_cast<half *>(output_matrix.data<at::Half>());
    half * output_matrix1_ = reinterpret_cast<half *>(output_matrix1.data<at::Half>());
    // Device
    // int *d_t_windowNew_offset, *d_t_blockNew_offset, *d_t_column, *d_t_row, *d_t_col, *d_t_window_row;
    int *d_t_windowNew_offset, *d_t_blockNew_offset, *d_t_column, *d_t_row;
    int *d_t_windowNew_offset1, *d_t_blockNew_offset1, *d_t_column1, *d_t_row1;
    half *d_t_value, *d_t_value1; 
	half *d_rhs_matrix;
    half *d_output_matrix, *d_output_matrix1;

    checkCuda(cudaMalloc(&d_t_windowNew_offset, (t_windowNew_offset.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_blockNew_offset, (t_blockNew_offset.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_value, (t_value.size(0)) * sizeof(half)));
    checkCuda(cudaMalloc(&d_t_column, (t_column.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_row, (t_row.size(0)) * sizeof(int)));

    checkCuda(cudaMalloc(&d_t_windowNew_offset1, (t_windowNew_offset1.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_blockNew_offset1, (t_blockNew_offset1.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_value1, (t_value1.size(0)) * sizeof(half)));
    checkCuda(cudaMalloc(&d_t_column1, (t_column1.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_row1, (t_row1.size(0)) * sizeof(int)));

    checkCuda(cudaMalloc(&d_rhs_matrix, (kOri*dimN) * sizeof(half)));
    checkCuda(cudaMalloc(&d_output_matrix, (mOri*dimN) * sizeof(half)));
    checkCuda(cudaMalloc(&d_output_matrix1, (mOri1*dimN) * sizeof(half)));

    //把CPU数据放到GPU上
    checkCuda(cudaMemcpy(d_t_windowNew_offset, t_windowNew_offset_ , (t_windowNew_offset.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_blockNew_offset, t_blockNew_offset_ , (t_blockNew_offset.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_value, t_value_, (t_value.size(0)) * sizeof(half), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_column, t_column_, (t_column.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_row, t_row_, (t_row.size(0)) * sizeof(int), cudaMemcpyHostToDevice));

    checkCuda(cudaMemcpy(d_t_windowNew_offset1, t_windowNew_offset1_ , (t_windowNew_offset1.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_blockNew_offset1, t_blockNew_offset1_ , (t_blockNew_offset1.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_value1, t_value1_, (t_value1.size(0)) * sizeof(half), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_column1, t_column1_, (t_column1.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_row1, t_row1_, (t_row1.size(0)) * sizeof(int), cudaMemcpyHostToDevice));


    checkCuda(cudaMemcpy(d_rhs_matrix, rhs_matrix_,(kOri*dimN) * sizeof(half), cudaMemcpyHostToDevice));

    float spmm_ms_avg =  spmm_forward_fp16_tcu_kernel_stream(
      d_t_windowNew_offset,
      d_t_blockNew_offset, 
      d_t_column,
      d_t_value, 
      d_t_row,

      d_t_windowNew_offset1,
      d_t_blockNew_offset1, 
      d_t_column1,
      d_t_value1, 
      d_t_row1,

      d_rhs_matrix,
      d_output_matrix,
      d_output_matrix1,
      dimM,
      dimM1,
      dimN,
      mOri,      
      mOri1,
      epoches);    
    checkCuda(cudaMemcpy(output_matrix_, d_output_matrix, mOri * dimN * sizeof(half), cudaMemcpyDeviceToHost));
    cudaFree(d_t_windowNew_offset);
    cudaFree(d_t_blockNew_offset);
    cudaFree(d_t_value);
    cudaFree(d_t_column),
    cudaFree(d_t_row);
    checkCuda(cudaMemcpy(output_matrix1_, d_output_matrix1, mOri1 * dimN * sizeof(half), cudaMemcpyDeviceToHost));
    cudaFree(d_t_windowNew_offset1);
    cudaFree(d_t_blockNew_offset1);
    cudaFree(d_t_value1);
    cudaFree(d_t_column1),
    cudaFree(d_t_row1);

    cudaFree(d_rhs_matrix);
    cudaFree(d_output_matrix);
    cudaFree(d_output_matrix1);
    cudaDeviceSynchronize(); 
    return {output_matrix, output_matrix1, torch::tensor(spmm_ms_avg)};
}

//binary
//tcu
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
    int epoches);

std::vector<torch::Tensor> spmm_forward_tf32_tcu_binary(
    torch::Tensor t_windowNew_offset,
    torch::Tensor t_blockNew_offset, 
    torch::Tensor t_column,
    torch::Tensor t_value, 
    torch::Tensor t_binary, 

    torch::Tensor rhs_matrix,

    const int dimM1,
    const int dimN,
    const int mOri,
    const int kOri,
    int epoches)
{
    int dimM=dimM1/8;
    auto output_matrix = torch::zeros({mOri, dimN}, torch::kFloat32).to(torch::kCPU);
    
    //把CPU端的tensor转成C++的数据结构
    int * t_windowNew_offset_ = t_windowNew_offset.data<int>();
    int * t_blockNew_offset_ = t_blockNew_offset.data<int>();
    float * t_value_ = t_value.data<float>(); 
    int * t_column_ = t_column.data<int>();
    int * t_binary_ = t_binary.data<int>();

    // int * t_window_row_ = t_window_row.data<int>();

    float * rhs_matrix_ = rhs_matrix.data<float>(); 
    float * output_matrix_ = output_matrix.data<float>();

    // Device
    int *d_t_windowNew_offset, *d_t_blockNew_offset, *d_t_column, *d_t_binary;
    float *d_t_value; 
	float *d_rhs_matrix;
    float *d_output_matrix;

    checkCuda(cudaMalloc(&d_t_windowNew_offset, (t_windowNew_offset.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_blockNew_offset, (t_blockNew_offset.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_value, (t_value.size(0)) * sizeof(float)));
    checkCuda(cudaMalloc(&d_t_column, (t_column.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_binary, (t_binary.size(0)) * sizeof(int)));

    // checkCuda(cudaMalloc(&d_t_window_row, (t_window_row.size(0)) * sizeof(int)));

    checkCuda(cudaMalloc(&d_rhs_matrix, (kOri*dimN) * sizeof(float)));
    checkCuda(cudaMalloc(&d_output_matrix, (mOri*dimN) * sizeof(float)));

    //把CPU数据放到GPU上
    checkCuda(cudaMemcpy(d_t_windowNew_offset, t_windowNew_offset_ , (t_windowNew_offset.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_blockNew_offset, t_blockNew_offset_ , (t_blockNew_offset.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_value, t_value_, (t_value.size(0)) * sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_column, t_column_, (t_column.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_binary, t_binary_, (t_binary.size(0)) * sizeof(int), cudaMemcpyHostToDevice));



    checkCuda(cudaMemcpy(d_rhs_matrix, rhs_matrix_,(kOri*dimN) * sizeof(float), cudaMemcpyHostToDevice));

    float spmm_ms_avg =  spmm_forward_tf32_tcu_kernel_binary(
      d_t_windowNew_offset,
      d_t_blockNew_offset, 
      d_t_value, 
      d_t_column,
      d_t_binary,

    //   d_t_window_row,
      d_rhs_matrix,
      d_output_matrix,

      dimM,
      dimN,
      mOri,
      epoches);    
    checkCuda(cudaMemcpy(output_matrix_, d_output_matrix, mOri * dimN * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_t_windowNew_offset);
    cudaFree(d_t_blockNew_offset);
    cudaFree(d_t_value);
    cudaFree(d_t_column),
    cudaFree(d_t_binary);

    // cudaFree(d_t_window_row);

    cudaFree(d_rhs_matrix);
    cudaFree(d_output_matrix);
    cudaDeviceSynchronize(); 
    return {output_matrix, torch::tensor(spmm_ms_avg)};
}

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
    int epoches);

std::vector<torch::Tensor> spmm_forward_fp16_tcu_binary(
    torch::Tensor t_windowNew_offset,
    torch::Tensor t_blockNew_offset, 
    torch::Tensor t_column,
    torch::Tensor t_value, 
    torch::Tensor t_binary, 
    
    torch::Tensor rhs_matrix,

    const int dimM1,
    const int dimN,
    const int mOri,
    const int kOri,
    int epoches)
{
    int dimM=dimM1/8;
    auto output_matrix = torch::zeros({mOri, dimN}, torch::kFloat16).to(torch::kCPU);
    
    //把CPU端的tensor转成C++的数据结构
    int * t_windowNew_offset_ = t_windowNew_offset.data<int>();
    int * t_blockNew_offset_ = t_blockNew_offset.data<int>();
    half * t_value_ = reinterpret_cast<half *>(t_value.data<at::Half>()); 
    int * t_column_ = t_column.data<int>();
    long * t_binary_ = t_binary.data<long>();
    // int * t_window_row_ = t_window_row.data<int>();

    half * rhs_matrix_ = reinterpret_cast<half *>(rhs_matrix.data<at::Half>()); 
    half * output_matrix_ = reinterpret_cast<half *>(output_matrix.data<at::Half>());

    // Device
    // int *d_t_windowNew_offset, *d_t_blockNew_offset, *d_t_column, *d_t_row, *d_t_col, *d_t_window_row;
    int *d_t_windowNew_offset, *d_t_blockNew_offset, *d_t_column;
    long *d_t_binary;
    half *d_t_value; 
	half *d_rhs_matrix;
    half *d_output_matrix;

    checkCuda(cudaMalloc(&d_t_windowNew_offset, (t_windowNew_offset.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_blockNew_offset, (t_blockNew_offset.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_value, (t_value.size(0)) * sizeof(half)));
    checkCuda(cudaMalloc(&d_t_column, (t_column.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_binary, (t_binary.size(0)) * sizeof(long)));

    checkCuda(cudaMalloc(&d_rhs_matrix, (kOri*dimN) * sizeof(half)));
    checkCuda(cudaMalloc(&d_output_matrix, (mOri*dimN) * sizeof(half)));

    //把CPU数据放到GPU上
    checkCuda(cudaMemcpy(d_t_windowNew_offset, t_windowNew_offset_ , (t_windowNew_offset.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_blockNew_offset, t_blockNew_offset_ , (t_blockNew_offset.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_value, t_value_, (t_value.size(0)) * sizeof(half), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_column, t_column_, (t_column.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_binary, t_binary_, (t_binary.size(0)) * sizeof(long), cudaMemcpyHostToDevice));


    checkCuda(cudaMemcpy(d_rhs_matrix, rhs_matrix_,(kOri*dimN) * sizeof(half), cudaMemcpyHostToDevice));

    float spmm_ms_avg =  spmm_forward_fp16_tcu_kernel_binary(
      d_t_windowNew_offset,
      d_t_blockNew_offset, 
      d_t_column,
      d_t_value, 
      d_t_binary,
      d_rhs_matrix,
      d_output_matrix,

      dimM,
      dimN,
      mOri,
      epoches);    
    checkCuda(cudaMemcpy(output_matrix_, d_output_matrix, mOri * dimN * sizeof(half), cudaMemcpyDeviceToHost));
    cudaFree(d_t_windowNew_offset);
    cudaFree(d_t_blockNew_offset);
    cudaFree(d_t_value);
    cudaFree(d_t_column),
    cudaFree(d_t_binary);
    // cudaFree(d_t_window_row);

    cudaFree(d_rhs_matrix);
    cudaFree(d_output_matrix);
    cudaDeviceSynchronize(); 
    return {output_matrix, torch::tensor(spmm_ms_avg)};
}


// cuda 长短行
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
    bool swizzle);

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
    if(dimN==32){
    spmm_ms_avg = spmm_forward_tf32_cuda_kernel_split_32(

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
    }else{
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

        reinterpret_cast<float2 *>(d_rhs_matrix),
        d_output_matrix,

        partsize,
        dimN,
        mOri,
        epoches,
        parts,
        parts_short,
        swizzle); 
    } 
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


// cuda 长短行 fp16
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
    bool swizzle);

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
    if(dimN==32){
     spmm_ms_avg =  spmm_forward_fp16_cuda_kernel_split_32(

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
      swizzle); }else{
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

      reinterpret_cast<half2 *>(d_rhs_matrix),
      d_output_matrix,

      partsize,
      dimN,
      mOri,
      epoches,
      parts,
      parts_short,
      swizzle);
      } 
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


// tcu cuda
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
    if(dimN==32){
    spmm_ms_avg =  spmm_forward_tf32_tcu_cuda_kernel_32(
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
    }else{
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
    }    
    checkCuda(cudaMemcpy(output_matrix_, d_output_matrix, mOri * dimN * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_t_row_offset);
    cudaFree(d_t_blockNew_offset);
    cudaFree(d_t_value);
    cudaFree(d_t_column),
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

// tcu cuda
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
    int epoches);

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
    if(dimN==32){
     spmm_ms_avg =  spmm_forward_fp16_tcu_cuda_kernel_32(
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
      epoches); }else{
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
      }
    checkCuda(cudaMemcpy(output_matrix_, d_output_matrix, mOri * dimN * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_t_row_offset);
    cudaFree(d_t_blockNew_offset);
    cudaFree(d_t_value);
    cudaFree(d_t_column),
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




// tcu cuda
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
    int epoches);

std::vector<torch::Tensor> spmm_forward_tf32_tcu_binary_part(
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
    float * t_value_ = t_value.data<float>(); 
    int * t_column_ = t_column.data<int>();
    int * t_window_row_ = t_window_row.data<int>();
    int * t_atomic_ = t_atomic.data<int>();
    int * t_binary_ = t_binary.data<int>();
    

    float * rhs_matrix_ = rhs_matrix.data<float>(); 
    float * output_matrix_ = output_matrix.data<float>();

    // Device
    int *d_t_row_offset, *d_t_blockNew_offset, *d_t_column;
    int *d_t_window_row, *d_t_atomic;
    int *d_t_binary;
    float *d_t_value; 
    // int *d_c_row_offsets, *d_c_row, *d_c_atomic, *d_c_column;
    // int *d_c_row_offsets_short, *d_c_row_short, *d_c_atomic_short, *d_c_column_short;
    // float *d_c_value, *d_c_value_short; 
	float *d_rhs_matrix;
    float *d_output_matrix;

    checkCuda(cudaMalloc(&d_t_row_offset, (t_row_offset.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_blockNew_offset, (t_blockNew_offset.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_value, (t_value.size(0)) * sizeof(float)));
    checkCuda(cudaMalloc(&d_t_column, (t_column.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_window_row, (t_window_row.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_atomic, (t_atomic.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_binary, (t_binary.size(0)) * sizeof(int)));


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

    checkCuda(cudaMemcpy(d_rhs_matrix, rhs_matrix_,(kOri*dimN) * sizeof(float), cudaMemcpyHostToDevice));

    float spmm_ms_avg =  spmm_forward_tf32_tcu_part_kernel(
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
    cudaFree(d_t_column),
    cudaFree(d_t_window_row);
    cudaFree(d_t_atomic);
    cudaFree(d_t_binary);

    cudaFree(d_rhs_matrix);
    cudaFree(d_output_matrix);
    cudaDeviceSynchronize(); 
    return {output_matrix, torch::tensor(spmm_ms_avg)};
}

// tcu cuda
float spmm_forward_fp16_tcu_part_kernel(
    int * t_row_offset,
    int * t_blockNew_offset,
    half * t_value, 
    int * t_column, 
    int* t_window_row,
    int * t_atomic,
    long * t_binary,

    half * rhs_matrix,
    float * output_matrix,

    const int parts_t,
    const int dimN,
    const int mOri,
    int epoches);

std::vector<torch::Tensor> spmm_forward_fp16_tcu_binary_part(
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
    // int *d_c_row_offsets, *d_c_row, *d_c_atomic, *d_c_column;
    // int *d_c_row_offsets_short, *d_c_row_short, *d_c_atomic_short, *d_c_column_short;
    // half *d_c_value, *d_c_value_short; 
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

    float spmm_ms_avg =  spmm_forward_fp16_tcu_part_kernel(
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
    cudaFree(d_t_column),
    cudaFree(d_t_window_row);
    cudaFree(d_t_atomic);
    cudaFree(d_t_binary);

    cudaFree(d_rhs_matrix);
    cudaFree(d_output_matrix);
    cudaDeviceSynchronize(); 
    return {output_matrix, torch::tensor(spmm_ms_avg)};
}




// tcu cuda
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
    int epoches);

std::vector<torch::Tensor> spmm_forward_tf32_tcu_part(
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
    float * t_value_ = t_value.data<float>(); 
    int * t_column_ = t_column.data<int>();
    int * t_window_row_ = t_window_row.data<int>();
    int * t_atomic_ = t_atomic.data<int>();
    int * t_binary_ = t_binary.data<int>();
    

    float * rhs_matrix_ = rhs_matrix.data<float>(); 
    float * output_matrix_ = output_matrix.data<float>();

    // Device
    int *d_t_row_offset, *d_t_blockNew_offset, *d_t_column;
    int *d_t_window_row, *d_t_atomic;
    int *d_t_binary;
    float *d_t_value; 
    // int *d_c_row_offsets, *d_c_row, *d_c_atomic, *d_c_column;
    // int *d_c_row_offsets_short, *d_c_row_short, *d_c_atomic_short, *d_c_column_short;
    // float *d_c_value, *d_c_value_short; 
	float *d_rhs_matrix;
    float *d_output_matrix;

    checkCuda(cudaMalloc(&d_t_row_offset, (t_row_offset.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_blockNew_offset, (t_blockNew_offset.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_value, (t_value.size(0)) * sizeof(float)));
    checkCuda(cudaMalloc(&d_t_column, (t_column.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_window_row, (t_window_row.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_atomic, (t_atomic.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_binary, (t_binary.size(0)) * sizeof(int)));


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

    checkCuda(cudaMemcpy(d_rhs_matrix, rhs_matrix_,(kOri*dimN) * sizeof(float), cudaMemcpyHostToDevice));

    float spmm_ms_avg =  spmm_forward_tf32_tcu_part_kernel_csr(
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
    cudaFree(d_t_column),
    cudaFree(d_t_window_row);
    cudaFree(d_t_atomic);
    cudaFree(d_t_binary);

    cudaFree(d_rhs_matrix);
    cudaFree(d_output_matrix);
    cudaDeviceSynchronize(); 
    return {output_matrix, torch::tensor(spmm_ms_avg)};
}

// tcu cuda
float spmm_forward_fp16_tcu_part_kernel_csr(
    int * t_row_offset,
    int * t_blockNew_offset,
    half * t_value, 
    int * t_column, 
    int* t_window_row,
    int * t_atomic,
    int * t_binary,

    half * rhs_matrix,
    float * output_matrix,

    const int parts_t,
    const int dimN,
    const int mOri,
    int epoches);

std::vector<torch::Tensor> spmm_forward_fp16_tcu_part(
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
    int * t_binary_ = t_binary.data<int>();

    half * rhs_matrix_ = reinterpret_cast<half *>(rhs_matrix.data<at::Half>()); 
    float * output_matrix_ = output_matrix.data<float>();

    // Device
    int *d_t_row_offset, *d_t_blockNew_offset, *d_t_column;
    int *d_t_window_row, *d_t_atomic;
    int *d_t_binary;
    half *d_t_value; 
    // int *d_c_row_offsets, *d_c_row, *d_c_atomic, *d_c_column;
    // int *d_c_row_offsets_short, *d_c_row_short, *d_c_atomic_short, *d_c_column_short;
    // half *d_c_value, *d_c_value_short; 
	half *d_rhs_matrix;
    float *d_output_matrix;

    checkCuda(cudaMalloc(&d_t_row_offset, (t_row_offset.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_blockNew_offset, (t_blockNew_offset.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_value, (t_value.size(0)) * sizeof(half)));
    checkCuda(cudaMalloc(&d_t_column, (t_column.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_window_row, (t_window_row.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_atomic, (t_atomic.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_binary, (t_binary.size(0)) * sizeof(int)));

    checkCuda(cudaMalloc(&d_rhs_matrix, (kOri*dimN) * sizeof(half)));
    checkCuda(cudaMalloc(&d_output_matrix, (mOri*dimN) * sizeof(float)));

    //把CPU数据放到GPU上
    checkCuda(cudaMemcpy(d_t_row_offset, t_row_offset_ , (t_row_offset.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_blockNew_offset, t_blockNew_offset_ , (t_blockNew_offset.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_value, t_value_, (t_value.size(0)) * sizeof(half), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_column, t_column_, (t_column.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_window_row, t_window_row_, (t_window_row.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_atomic, t_atomic_, (t_atomic.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_binary, t_binary_, (t_binary.size(0)) * sizeof(int), cudaMemcpyHostToDevice));


    checkCuda(cudaMemcpy(d_rhs_matrix, rhs_matrix_,(kOri*dimN) * sizeof(half), cudaMemcpyHostToDevice));

    float spmm_ms_avg =  spmm_forward_fp16_tcu_part_kernel_csr(
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
    cudaFree(d_t_column),
    cudaFree(d_t_window_row);
    cudaFree(d_t_atomic);
    cudaFree(d_t_binary);

    cudaFree(d_rhs_matrix);
    cudaFree(d_output_matrix);
    cudaDeviceSynchronize(); 
    return {output_matrix, torch::tensor(spmm_ms_avg)};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

  m.def("forward_tf32_tcu_sgt", &spmm_forward_tf32_tcu_sgt, "tcu");
  m.def("forward_fp16_tcu_sgt", &spmm_forward_fp16_tcu_sgt, "tcu");

  m.def("forward_tf32_tcu", &spmm_forward_tf32_tcu, "tcu");
  m.def("forward_fp16_tcu", &spmm_forward_fp16_tcu, "tcu");
  m.def("forward_fp16_tcu_stream", &spmm_forward_fp16_tcu_stream, "tcu");

  m.def("forward_tf32_tcu_part", &spmm_forward_tf32_tcu_part, "tcu + part");
  m.def("forward_fp16_tcu_part", &spmm_forward_fp16_tcu_part, "tcu + part");

  m.def("forward_tf32_tcu_binary", &spmm_forward_tf32_tcu_binary, "tcu + binary");
  m.def("forward_fp16_tcu_binary", &spmm_forward_fp16_tcu_binary, "tcu + binary");

  m.def("forward_tf32_tcu_binary_part", &spmm_forward_tf32_tcu_binary_part, "tcu + part + binary");
  m.def("forward_fp16_tcu_binary_part", &spmm_forward_fp16_tcu_binary_part, "tcu + part + binary");
  
  m.def("forward_tf32_cuda_v2", &spmm_forward_tf32_cuda_v2, "长短行");
  m.def("forward_fp16_cuda_v2", &spmm_forward_fp16_cuda_v2, "长短行");

  m.def("forward_tf32_tcu_cuda", &spmm_forward_tf32_tcu_cuda, "one kernel");
  m.def("forward_fp16_tcu_cuda", &spmm_forward_fp16_tcu_cuda, "one kernel");

  }