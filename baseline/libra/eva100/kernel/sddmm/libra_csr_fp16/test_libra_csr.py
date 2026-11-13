import os
import sys
# sys.path.append('./eva100/kernel/gcn')
from libra_csr_fp16.mdataset2 import *
import Libra5BenchmarkGAT

import Libra6SDDMM


# 只cuda
def kernel_cuda(inputInfo, epoches, type):
    
    if type ==0 :
        X_prime, spmm_ms_avg = Libra6SDDMM.forward_fp16_cuda_v2(
            inputInfo.c_row_offsetTensor,
            inputInfo.c_rowTensor, 
            inputInfo.c_colTensor,
            inputInfo.x,    

            inputInfo.x.size(1), 
            inputInfo.num_nodes_ori, 
            inputInfo.num_nodes_dst, 
            epoches,
            inputInfo.parts)
    else : 
        X_prime, spmm_ms_avg = Libra5BenchmarkGAT.forward_fp16_cuda_v2_navie(
            inputInfo.c_row_offsetTensor,
            inputInfo.c_rowTensor, 
            inputInfo.c_colTensor,
            inputInfo.x,    

            inputInfo.x.size(1), 
            inputInfo.num_nodes_ori, 
            inputInfo.num_nodes_dst, 
            epoches,
            inputInfo.parts)
    return round(spmm_ms_avg.item(),4)

def test_cuda(data, epoches, dimN, density, partsize, data_path, type):

    
    inputInfo = GCN_dataset_cuda(data, dimN, density, partsize, data_path)
    
   
    execution_time = kernel_cuda(inputInfo, epoches, type)
    print(str(dimN) + '-' + data + ' only cuda-' + str(execution_time))
    return execution_time

# 只tcu 
def kernel_tcu(inputInfo, epoches):
    
    X_prime, spmm_ms_avg = Libra5BenchmarkGAT.forward_fp16_tcu_binary_part(
        inputInfo.t_rowNew_offsetTensor,
        inputInfo.t_blockTensor, 
        inputInfo.t_columnTensor,
        inputInfo.t_window_rowTensor,
        inputInfo.t_binaryTensor,
        
        inputInfo.x, 
        inputInfo.nnz,
        inputInfo.max,
        inputInfo.parts_t, 
        inputInfo.x.size(1), 
        inputInfo.num_nodes_ori, 
        inputInfo.num_nodes_dst, 
        epoches)
    return round(spmm_ms_avg.item(),4)

def test_tcu(data, epoches, dimN, density, partsize, data_path,  window, wide):

    inputInfo = GCN_dataset_tcu(data, dimN, density, partsize, data_path, window, wide)

    execution_time = kernel_tcu(inputInfo, epoches)
    print(str(dimN) + '-' + data + ' only tcu-' + str(execution_time))
    return execution_time



# tcu cuda
def kernel_tcu_cuda(inputInfo, epoches, type):
    
    X_prime, spmm_ms_avg = Libra6SDDMM.forward_fp16_tcu_cuda(
        inputInfo.t_rowNew_offsetTensor, 
        inputInfo.t_blockTensor,
        inputInfo.t_columnTensor, 
        inputInfo.t_window_rowTensor,
        inputInfo.t_binaryTensor,

        
        inputInfo.c_row_offsetTensor,
        inputInfo.c_rowTensor, 
        inputInfo.c_colTensor,
        
        
        inputInfo.x, 
        inputInfo.nnz,
        inputInfo.max,
        inputInfo.parts_t, 
        inputInfo.x.size(1), 
        inputInfo.num_nodes_ori, 
        inputInfo.num_nodes_dst, 
        inputInfo.parts_c,epoches, type)
    return round(spmm_ms_avg.item(),4)

def test_tcu_cuda(data, epoches, dimN, density, partsize_t, partsize_c, shortsize, data_path,  window, wide, type):

    
    inputInfo = GCN_dataset_tcu_cuda(data, dimN, density, partsize_t, partsize_c, shortsize, data_path, window, wide)
   
    execution_time = kernel_tcu_cuda(inputInfo, epoches, type)
    print(str(dimN) + '-' + data + ' mtcu-cuda-dense-' + str(density) + '-' +str(execution_time))
    return execution_time



# tcu cuda
def kernel_tcu_cuda_cc_only(inputInfo, epoches, type):
    
    X_prime, spmm_ms_avg = Libra6SDDMM.forward_fp16_tcu_cuda(
        inputInfo.t_rowNew_offsetTensor, 
        inputInfo.t_blockTensor,
        inputInfo.t_columnTensor, 
        inputInfo.t_window_rowTensor,
        inputInfo.t_binaryTensor,

        
        inputInfo.c_row_offsetTensor,
        inputInfo.c_rowTensor, 
        inputInfo.c_colTensor,
        
        
        inputInfo.x, 
        inputInfo.nnz,
        inputInfo.max,
        inputInfo.parts_t, 
        inputInfo.x.size(1), 
        inputInfo.num_nodes_ori, 
        inputInfo.num_nodes_dst, 
        inputInfo.parts_c,epoches, type)
    
    X_prime, spmm_ms_avg_cc_only = Libra6SDDMM.forward_fp16_tcu_cuda_cc_only(
        inputInfo.t_rowNew_offsetTensor, 
        inputInfo.t_blockTensor,
        inputInfo.t_columnTensor, 
        inputInfo.t_window_rowTensor,
        inputInfo.t_binaryTensor,

        
        inputInfo.c_row_offsetTensor,
        inputInfo.c_rowTensor, 
        inputInfo.c_colTensor,
        
        
        inputInfo.x, 
        inputInfo.nnz,
        inputInfo.max,
        inputInfo.parts_t, 
        inputInfo.x.size(1), 
        inputInfo.num_nodes_ori, 
        inputInfo.num_nodes_dst, 
        inputInfo.parts_c,epoches, type)
    return round(spmm_ms_avg.item(),4), round(spmm_ms_avg_cc_only.item(),4)

def test_tcu_cuda_cc_only(data, epoches, dimN, density, partsize_t, partsize_c, shortsize, data_path,  window, wide, type):

    
    inputInfo = GCN_dataset_tcu_cuda(data, dimN, density, partsize_t, partsize_c, shortsize, data_path, window, wide)
   
    execution_time, execution_time_cc_only = kernel_tcu_cuda_cc_only(inputInfo, epoches, type)
    print(str(dimN) + '-' + data + ' mtcu-cuda-dense-' + str(density) + '-' +str(execution_time))
    return execution_time, execution_time_cc_only