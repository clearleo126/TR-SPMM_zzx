import os
import sys
# sys.path.append('./eva100/kernel/gcn')
from libra_csr_tf32.mdataset2 import *
import Libra5BenchmarkGCN

# 只tcu 
def kernel_tcu_v2(inputInfo, epoches):
    
    X_prime, spmm_ms_avg = Libra5BenchmarkGCN.forward_tf32_tcu_binary_part(
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
    # 
   
    execution_time = kernel_tcu_v2(inputInfo, epoches)
    print(str(dimN) + '-' + data + ' only tcu + binary + part-' + str(execution_time))
    return execution_time

#只tcu
def kernel_tcu(inputInfo, epoches):
    
    X_prime, spmm_ms_avg = Libra5BenchmarkGCN.forward_tf32_tcu(
        inputInfo.t_rowNew_offsetTensor,
        inputInfo.t_blockNew_offsetTensor, 
        inputInfo.t_columnTensor,
        inputInfo.t_valueTensor, 
        inputInfo.t_rowTensor,
        
        inputInfo.x,    
        inputInfo.num_nodes, 
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

# 只tcu + part
def kernel_tcu_part(inputInfo, epoches):
    
    X_prime, spmm_ms_avg = Libra5BenchmarkGCN.forward_tf32_tcu_part(
        inputInfo.t_rowNew_offsetTensor,
        inputInfo.t_blockNew_offsetTensor, 
        inputInfo.t_columnTensor,
        inputInfo.t_valueTensor, 
        inputInfo.t_window_rowTensor,
        inputInfo.t_atomicTensor,
        inputInfo.t_rowTensor,
        
        inputInfo.x,    
        inputInfo.parts_t,
        inputInfo.x.size(1), 
        inputInfo.num_nodes_ori, 
        inputInfo.num_nodes_dst, 
        epoches)
    return round(spmm_ms_avg.item(),4)

def test_tcu_part(data, epoches, dimN, density, partsize, data_path,  window, wide):

    
    inputInfo = GCN_dataset_tcu_part(data, dimN, density, partsize, data_path, window, wide)
    
   
    execution_time = kernel_tcu_part(inputInfo, epoches)
    print(str(dimN) + '-' + data + ' only tcu + part-' + str(execution_time))
    return execution_time

# 只tcu + binary
def kernel_tcu_binary(inputInfo, epoches):
    
    X_prime, spmm_ms_avg = Libra5BenchmarkGCN.forward_tf32_tcu_binary(
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
        inputInfo.num_nodes_dst, 
        epoches)
    return round(spmm_ms_avg.item(),4)

def test_tcu_binary(data, epoches, dimN, density, partsize, data_path,  window, wide):

    
    inputInfo = GCN_dataset_tcu_binary(data, dimN, density, partsize, data_path, window, wide)
    
   
    execution_time = kernel_tcu_binary(inputInfo, epoches)
    print(str(dimN) + '-' + data + ' only tcu + bianry-' + str(execution_time))
    return execution_time


#测试SGT
def kernel_sgt(inputInfo, epoches):

    X_prime, spmm_ms_avg = Libra5BenchmarkGCN.forward_tf32_tcu_sgt(
        inputInfo.t_blockNew_offsetTensor, 
        inputInfo.row_pointers,
        inputInfo.column_index,
        inputInfo.t_valueTensor, 
        inputInfo.t_rowTensor,
        inputInfo.t_colTensor,
        
        inputInfo.x, 
        
        inputInfo.num_nodes, 
        inputInfo.x.size(1), 
        inputInfo.num_nodes_ori, 
        inputInfo.num_nodes_dst, 
        epoches)
    return round(spmm_ms_avg.item(),4)

def test_tcu_sgt(data, epoches, dimN, density, partsize, data_path,  window, wide):

    
    inputInfo = GCN_dataset_tcu_sgt(data, dimN, density, partsize, data_path, window, wide)
    
   
    execution_time = kernel_sgt(inputInfo, epoches)
    print(str(dimN) + '-' + data + ' only tcu-' + str(execution_time))
    return execution_time


#测试BCRS
def kernel_bcrs(inputInfo, epoches):

    X_prime, spmm_ms_avg = Libra5BenchmarkGCN.forward_tf32_tcu_bcrs(
        inputInfo.t_blockNew_offsetTensor, 
        inputInfo.t_colTensor,
        inputInfo.t_valueTensor, 
        
        inputInfo.x, 
        
        inputInfo.num_nodes, 
        inputInfo.x.size(1), 
        inputInfo.num_nodes_ori, 
        inputInfo.num_nodes_dst, 
        epoches)
    return round(spmm_ms_avg.item(),4)

def test_tcu_bcrs(data, epoches, dimN, density, partsize, data_path,  window, wide):

    
    inputInfo = GCN_dataset_tcu_bcrs(data, dimN, density, partsize, data_path, window, wide)
    
   
    execution_time = kernel_bcrs(inputInfo, epoches)
    print(str(dimN) + '-' + data + 'BCRS-' + str(execution_time))
    return execution_time