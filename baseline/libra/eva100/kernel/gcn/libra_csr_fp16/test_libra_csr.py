import os
import sys
# sys.path.append('./eva100/kernel/gcn')
from libra_csr_fp16.mdataset2 import *
import Libra5BenchmarkGCN

import MagicsphereGCN_kernel
# 混合
def kernel(inputInfo, epoches, kernel_type):
    spmm_ms_avg = None
    if kernel_type == 1:
        X_prime, spmm_ms_avg = Libra5BenchmarkGCN.forward_tf32_csr_trans(
            inputInfo.t_windowNew_offsetTensor,
            inputInfo.t_blockNew_offsetTensor, 
            inputInfo.t_valueTensor, 
            inputInfo.t_columnTensor,
            inputInfo.t_rowTensor,
            inputInfo.t_colTensor,
            inputInfo.t_window_rowTensor,
            
            inputInfo.c_part_offsetTensor,
            inputInfo.c_row_offsetTensor,
            inputInfo.c_rowTensor, 
            inputInfo.c_atomicTensor,
            inputInfo.c_colTensor,
            inputInfo.c_valueTensor, 
            inputInfo.x,    
            inputInfo.num_nodes, 
            inputInfo.x.size(1), 
            inputInfo.num_nodes_ori, 
            epoches,
            inputInfo.boundary,
            inputInfo.parts)
    else:
        X_prime, spmm_ms_avg = Libra5BenchmarkGCN.forward_tf32_csr_v2(
            inputInfo.t_windowNew_offsetTensor,
            inputInfo.t_blockNew_offsetTensor, 
            inputInfo.t_valueTensor, 
            inputInfo.t_columnTensor,
            inputInfo.t_rowTensor,
            inputInfo.t_colTensor,
            inputInfo.t_window_rowTensor,

            inputInfo.c_part_offsetTensor,
            inputInfo.c_row_offsetTensor,
            inputInfo.c_rowTensor, 
            inputInfo.c_atomicTensor,
            inputInfo.c_colTensor,
            inputInfo.c_valueTensor, 
            # inputInfo.c_row_offsetTensor_new,
            inputInfo.x,    
            inputInfo.num_nodes, 
            inputInfo.x.size(1), 
            inputInfo.num_nodes_ori, 
            epoches,
            inputInfo.boundary,
            inputInfo.parts)
    
    return round(spmm_ms_avg.item(),4)

def test(data, epoches, dimN, density, partsize, data_path, window,  wide, kernel_type):

    
    inputInfo = GCN_dataset(data, dimN, density, partsize, data_path,  window, wide, kernel_type)
    
   
    execution_time = kernel(inputInfo, epoches, kernel_type)
    print(str(dimN) + '-' + data + ' libra-' + str(density) + '-' + str(execution_time))
    return execution_time

# 混合,整行
def kernel_v2(inputInfo, epoches, kernel_type):
    spmm_ms_avg = None
    if kernel_type == 1:
        X_prime, spmm_ms_avg = Libra5BenchmarkGCN.forward_tf32_csr_trans(
            inputInfo.t_windowNew_offsetTensor,
            inputInfo.t_blockNew_offsetTensor, 
            inputInfo.t_valueTensor, 
            inputInfo.t_columnTensor,
            inputInfo.t_rowTensor,
            inputInfo.t_colTensor,
            inputInfo.t_window_rowTensor,
            
            inputInfo.c_part_offsetTensor,
            inputInfo.c_row_offsetTensor,
            inputInfo.c_rowTensor, 
            inputInfo.c_atomicTensor,
            inputInfo.c_colTensor,
            inputInfo.c_valueTensor, 
            inputInfo.x,    
            inputInfo.num_nodes, 
            inputInfo.x.size(1), 
            inputInfo.num_nodes_ori, 
            epoches,
            inputInfo.boundary,
            inputInfo.parts)
        return round(spmm_ms_avg.item(),4)
    else :
        X_prime, spmm_ms_avg = Libra5BenchmarkGCN.forward_tf32_csr_v2(
            inputInfo.t_windowNew_offsetTensor,
            inputInfo.t_blockNew_offsetTensor, 
            inputInfo.t_valueTensor, 
            inputInfo.t_columnTensor,
            inputInfo.t_rowTensor,
            inputInfo.t_colTensor,
            inputInfo.t_window_rowTensor,
            
            inputInfo.c_part_offsetTensor,
            inputInfo.c_row_offsetTensor,
            inputInfo.c_rowTensor, 
            inputInfo.c_atomicTensor,
            inputInfo.c_colTensor,
            inputInfo.c_valueTensor, 
            inputInfo.x,    
            inputInfo.num_nodes, 
            inputInfo.x.size(1), 
            inputInfo.num_nodes_ori, 
            epoches,
            inputInfo.boundary,
            inputInfo.parts)
        return round(spmm_ms_avg.item(),4)

def test_v2(data, epoches, dimN, density, partsize, data_path,  window, wide, kernel_type):

    
    inputInfo = GCN_dataset_v2(data, dimN, density, partsize, data_path,  window, wide, kernel_type)
    
   
    execution_time = kernel_v2(inputInfo, epoches, kernel_type)
    print(str(dimN) + '-' + data + ' libra-' + str(density) + '-' + str(execution_time))
    return execution_time

# 只cuda
def kernel_cuda(inputInfo, epoches, swizzle):
    
    X_prime, spmm_ms_avg = Libra5BenchmarkGCN.forward_tf32_cuda(
        inputInfo.c_row_offsetTensor,
        inputInfo.c_rowTensor, 
        inputInfo.c_atomicTensor,
        inputInfo.c_colTensor,
        inputInfo.c_valueTensor, 
        inputInfo.x,    
        inputInfo.partSize, 
        inputInfo.x.size(1), 
        inputInfo.num_nodes_ori, 
        epoches,
        inputInfo.parts,
        swizzle)
    return round(spmm_ms_avg.item(),4)

def test_cuda(data, epoches, dimN, density, partsize, data_path, swizzle):

    
    inputInfo = GCN_dataset_cuda(data, dimN, density, partsize, data_path)
    
   
    execution_time = kernel_cuda(inputInfo, epoches, swizzle)
    print(str(dimN) + '-' + data + ' only cuda-' + str(execution_time))
    return execution_time

# 只tcu 
def kernel_tcu_v2(inputInfo, epoches):
    
    X_prime, spmm_ms_avg = Libra5BenchmarkGCN.forward_fp16_tcu_binary_part(
        inputInfo.t_windowNew_offsetTensor,
        inputInfo.t_blockNew_offsetTensor, 
        inputInfo.t_columnTensor,
        inputInfo.t_valueTensor, 
        inputInfo.t_window_rowTensor,
        inputInfo.t_atomicTensor,
        inputInfo.t_binaryTensor,
        
        inputInfo.x,    
        inputInfo.parts_t, 
        inputInfo.x.size(1), 
        inputInfo.num_nodes_ori, 
        inputInfo.num_nodes_dst, 
        epoches)
    return round(spmm_ms_avg.item(),4)

def test_tcu_v2(data, epoches, dimN, density, partsize, data_path,  window, wide):

    
    inputInfo = GCN_dataset_tcu_v2(data, dimN, density, partsize, data_path, window, wide)

    execution_time = kernel_tcu_v2(inputInfo, epoches)
    print(str(dimN) + '-' + data + ' only tcu-' + str(execution_time))
    return execution_time

# 只tcu without part
def kernel_tcu(inputInfo, epoches):
    
    X_prime, spmm_ms_avg = Libra5BenchmarkGCN.forward_tf32_tcu(
        inputInfo.t_rowNew_offsetTensor,
        inputInfo.t_blockNew_offsetTensor, 
        inputInfo.t_columnTensor,
        inputInfo.t_valueTensor, 
        inputInfo.t_rowTensor,
        inputInfo.t_colTensor,
        # inputInfo.t_window_rowTensor,
        
        inputInfo.x,    
        inputInfo.num_nodes, 
        inputInfo.x.size(1), 
        inputInfo.num_nodes_ori, 
        epoches)
    return round(spmm_ms_avg.item(),4)

def test_tcu(data, epoches, dimN, density, partsize, data_path,  window, wide):

    
    inputInfo = GCN_dataset_tcu(data, dimN, density, partsize, data_path, window, wide)
    
   
    execution_time = kernel_tcu(inputInfo, epoches)
    print(str(dimN) + '-' + data + ' only tcu-' + str(execution_time))
    return execution_time

# 只tcu without part binary
def kernel_tcu_binary(inputInfo, epoches):
    
    X_prime, spmm_ms_avg = Libra5BenchmarkGCN.forward_fp16_tcu_binary(
        inputInfo.t_rowNew_offsetTensor,
        inputInfo.t_blockNew_offsetTensor, 
        inputInfo.t_columnTensor,
        inputInfo.t_valueTensor, 
        inputInfo.t_binaryTensor,
        # inputInfo.t_colTensor,
        # inputInfo.t_window_rowTensor,
        
        inputInfo.x,    
        inputInfo.num_nodes, 
        inputInfo.x.size(1), 
        inputInfo.num_nodes_ori, 
        inputInfo.num_nodes_ori, 
        epoches)
    return round(spmm_ms_avg.item(),4)

def test_tcu_binary(data, epoches, dimN, density, partsize, data_path,  window, wide):

    
    inputInfo = GCN_dataset_tcu_binary(data, dimN, density, partsize, data_path, window, wide)
    
   
    execution_time = kernel_tcu_binary(inputInfo, epoches)
    print(str(dimN) + '-' + data + ' only tcu-' + str(execution_time))
    return execution_time


# 长短行
def kernel_cuda_v2(inputInfo, epoches, swizzle):
    
    X_prime, spmm_ms_avg = Libra5BenchmarkGCN.forward_fp16_cuda_v2(
        
        inputInfo.c_row_offsetTensor,
        inputInfo.c_rowTensor, 
        inputInfo.c_atomicTensor,
        inputInfo.c_colTensor,
        inputInfo.c_valueTensor, 
        
        inputInfo.c_row_offsetTensor_short,
        inputInfo.c_rowTensor_short, 
        inputInfo.c_atomicTensor_short,
        inputInfo.c_colTensor_short,
        inputInfo.c_valueTensor_short, 

        inputInfo.x,    
        inputInfo.partSize, 
        inputInfo.x.size(1), 
        inputInfo.num_nodes_ori, 
        inputInfo.num_nodes_dst, 
        epoches,
        inputInfo.parts,
        inputInfo.parts_short,
        swizzle)
    return round(spmm_ms_avg.item(),4)

def test_cuda_v2(data, epoches, dimN, density, partsize, data_path, swizzle, shortSize):

    
    inputInfo = GCN_dataset_cuda_v2(data, dimN, density, partsize, data_path, shortSize)
   
    execution_time = kernel_cuda_v2(inputInfo, epoches, swizzle)
    print(str(dimN) + '-' + data + ' only cuda-' + str(execution_time))
    return execution_time



# tcu cuda
def kernel_tcu_cuda(inputInfo, epoches):
    
    X_prime, spmm_ms_avg = Libra5BenchmarkGCN.forward_fp16_tcu_cuda(
        inputInfo.t_rowNew_offsetTensor, 
        inputInfo.t_blockTensor,
        inputInfo.t_columnTensor, 
        inputInfo.t_valueTensor, 
        inputInfo.t_window_rowTensor,
        inputInfo.t_atomicTensor,
        inputInfo.t_binaryTensor,

        
        inputInfo.c_row_offsetTensor,
        inputInfo.c_rowTensor, 
        inputInfo.c_atomicTensor,
        inputInfo.c_colTensor,
        inputInfo.c_valueTensor, 
        
        inputInfo.c_row_offsetTensor_short,
        inputInfo.c_rowTensor_short, 
        inputInfo.c_atomicTensor_short,
        inputInfo.c_colTensor_short,
        inputInfo.c_valueTensor_short, 
        
        inputInfo.x, 
        inputInfo.parts_t, 
        inputInfo.parts_c, 
        inputInfo.partsize_c,
        inputInfo.parts_c_short, 
        inputInfo.x.size(1), 
        inputInfo.num_nodes_ori, 
        inputInfo.num_nodes_dst, 
        epoches)
    return round(spmm_ms_avg.item(),4)

def test_tcu_cuda(data, epoches, dimN, density, partsize_t, partsize_c, shortsize, data_path,  window, wide):

    
    inputInfo = GCN_dataset_tcu_cuda_tf32(data, dimN, density, partsize_t, partsize_c, shortsize, data_path, window, wide)
    # 
   
    execution_time = kernel_tcu_cuda(inputInfo, epoches)
    print(str(dimN) + '-' + data + ' mtcu-cuda-dense-' + str(density) + '-' +str(execution_time))
    return execution_time