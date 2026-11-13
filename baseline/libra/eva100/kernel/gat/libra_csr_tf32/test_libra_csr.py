import os
import sys
# sys.path.append('./eva100/kernel/gcn')
from libra_csr_tf32.mdataset2 import *
import Libra5BenchmarkGAT

# tcu part csr
def kernel_tcu_csr(inputInfo, epoches):
    
    X_prime, spmm_ms_avg = Libra5BenchmarkGAT.forward_tf32_tcu_csr_part(
        inputInfo.t_rowNew_offsetTensor,
        inputInfo.t_blockTensor, 
        inputInfo.t_columnTensor,
        inputInfo.t_window_rowTensor, 
        inputInfo.t_rowTensor,

        inputInfo.x,    
        inputInfo.t_blockTensor[-1],
        inputInfo.maxPart, 
        inputInfo.parts_t,
        inputInfo.x.size(1), 
        inputInfo.num_nodes_ori, 
        epoches)
    return round(spmm_ms_avg.item(),4)

def libra_csr_tcu(data, epoches, dimN, density, partsize, data_path,  window, wide):


    inputInfo = GCN_dataset_tcu(data, dimN, density, partsize, data_path, window, wide)
   
    execution_time = kernel_tcu_csr(inputInfo, epoches)
    print(str(dimN) + '-' + data + ' only tcu csr-' + str(execution_time))
    return execution_time

# tcu part binary
def kernel_tcu_bianry(inputInfo, epoches):
    X_prime, spmm_ms_avg = Libra5BenchmarkGAT.forward_tf32_tcu_binary_part(
        inputInfo.t_rowNew_offsetTensor,
        inputInfo.t_blockTensor, 
        inputInfo.t_columnTensor,
        inputInfo.t_window_rowTensor, 
        inputInfo.t_rowTensor,

        inputInfo.x,    
        inputInfo.t_blockTensor[-1],
        inputInfo.maxPart, 
        inputInfo.parts_t,
        inputInfo.x.size(1), 
        inputInfo.num_nodes_ori, 
        epoches)
    return round(spmm_ms_avg.item(),4)

def libra_binary_tcu(data, epoches, dimN, density, partsize, data_path,  window, wide):


    inputInfo = GCN_dataset_tcu1(data, dimN, density, partsize, data_path, window, wide)
    
   
    execution_time = kernel_tcu_bianry(inputInfo, epoches)
    print(str(dimN) + '-' + data + ' only tcu binary-' + str(execution_time))
    return execution_time


# cuda
def kernel_cuda(inputInfo, epoches):
    X_prime, spmm_ms_avg = Libra5BenchmarkGAT.forward_tf32_cuda_v2(
        inputInfo.c_row_offsetTensor,
        inputInfo.c_rowTensor, 
        inputInfo.c_colTensor,

        inputInfo.x,    
        inputInfo.x.size(1), 
        inputInfo.num_nodes_ori, 
        epoches,
        inputInfo.c_rowTensor.shape[0])
    return round(spmm_ms_avg.item(),4)

def libra_cuda(data, epoches, dimN, density, partsize, data_path,  window, wide):


    inputInfo = GCN_dataset_cuda(data, dimN, density, partsize, data_path, window, wide)
    
   
    execution_time = kernel_cuda(inputInfo, epoches)
    print(str(dimN) + '-' + data + ' only cuda-' + str(execution_time))
    return execution_time


def kernel_cuda_v2(inputInfo, epoches):
    X_prime, spmm_ms_avg = Libra5BenchmarkGAT.forward_tf32_cuda_v2_navie(
        inputInfo.c_row_offsetTensor,
        inputInfo.c_rowTensor, 
        inputInfo.c_colTensor,

        inputInfo.x,    
        inputInfo.x.size(1), 
        inputInfo.num_nodes_ori, 
        epoches,
        inputInfo.c_rowTensor.shape[0])
    return round(spmm_ms_avg.item(),4)

def libra_cuda_v2(data, epoches, dimN, density, partsize, data_path,  window, wide):


    inputInfo = GCN_dataset_cuda(data, dimN, density, partsize, data_path, window, wide)
    
   
    execution_time = kernel_cuda_v2(inputInfo, epoches)
    print(str(dimN) + '-' + data + ' only cuda-' + str(execution_time))
    return execution_time

# cuda + tcu part binary
def kernel_cuda_tcu_bianry(inputInfo, epoches, type):
    X_prime, spmm_ms_avg = Libra5BenchmarkGAT.forward_tf32_tcu_cuda(
        inputInfo.t_rowNew_offsetTensor,
        inputInfo.t_blockTensor, 
        inputInfo.t_columnTensor,
        inputInfo.t_window_rowTensor, 
        inputInfo.t_binaryTensor,
        
        inputInfo.c_row_offsetTensor,
        inputInfo.c_rowTensor, 
        inputInfo.c_colTensor,

        inputInfo.x,    
        inputInfo.t_blockTensor[-1],
        inputInfo.maxPart, 
        inputInfo.parts_t,
        inputInfo.x.size(1), 
        inputInfo.num_nodes_ori, 
        inputInfo.c_rowTensor.shape[0],
        epoches,
        type)
    return round(spmm_ms_avg.item(),4)

def libra_binary_cuda_tcu(data, epoches, dimN, density, partsize_t, partsize_c, data_path,  window, wide, type):


    inputInfo = GCN_dataset_cuda_tcu_binary(data, dimN, density,  partsize_t, partsize_c, data_path, window, wide)
    
   
    execution_time = kernel_cuda_tcu_bianry(inputInfo, epoches,type)
    print(str(dimN) + '-' + data + ' tcu+cuda-' + str(density) + '-' + str(execution_time))
    return execution_time