import os
import sys
# sys.path.append('./eva100/kernel/gcn')
from libra_csr_fp16.mdataset2 import *
import Libra5BenchmarkGCN

# 只tcu 
def kernel_tcu_v2(inputInfo, epoches):
    
    X_prime, spmm_ms_avg = Libra5BenchmarkGCN.forward_fp16_tcu_binary_part(
        inputInfo.t_windowNew_offsetTensor,
        inputInfo.t_blockNew_offsetTensor, 
        inputInfo.t_columnTensor,
        inputInfo.t_valueTensor.half(), 
        inputInfo.t_window_rowTensor,
        inputInfo.t_atomicTensor,
        inputInfo.t_binaryTensor,
        
        inputInfo.x.half(),    
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
    
    X_prime, spmm_ms_avg = Libra5BenchmarkGCN.forward_fp16_tcu(
        inputInfo.t_rowNew_offsetTensor,
        inputInfo.t_blockNew_offsetTensor, 
        inputInfo.t_columnTensor,
        inputInfo.t_valueTensor.half(), 
        inputInfo.t_rowTensor,
        
        inputInfo.x.half(),    
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

#测试stream
def kernel_tcu2(inputInfo, epoches):
    dimM = (inputInfo.t_rowNew_offsetTensor.shape[0]-1)*8
    dimM1 = (inputInfo.t_rowNew_offsetTensor.shape[0]-1)*8
    X_prime, X_prime1, spmm_ms_avg = Libra5BenchmarkGCN.forward_fp16_tcu_stream(
        inputInfo.t_rowNew_offsetTensor,
        inputInfo.t_blockNew_offsetTensor, 
        inputInfo.t_columnTensor,
        inputInfo.t_valueTensor.half(), 
        inputInfo.t_rowTensor,

        inputInfo.t_rowNew_offsetTensor1,
        inputInfo.t_blockNew_offsetTensor1, 
        inputInfo.t_columnTensor1,
        inputInfo.t_valueTensor1.half(), 
        inputInfo.t_rowTensor1,
        
        inputInfo.x.half(), 
        dimM,
        dimM1,
        inputInfo.x.size(1), 
        dimM,
        inputInfo.num_nodes_ori-dimM, 
        inputInfo.num_nodes_dst, 
        epoches)
    return round(spmm_ms_avg.item(),4)

def test_tcu_stream(data, epoches, dimN, density, partsize, data_path,  window, wide):

    
    inputInfo = GCN_dataset_tcu_stream(data, dimN, density, partsize, data_path, window, wide)
    
   
    execution_time = kernel_tcu2(inputInfo, epoches)
    print(str(dimN) + '-' + data + ' only tcu-' + str(execution_time))
    return execution_time

#测试SGT
def kernel_sgt(inputInfo, epoches):

    X_prime, spmm_ms_avg = Libra5BenchmarkGCN.forward_fp16_tcu_csr(
        inputInfo.t_windowkNew_offsetTensor,
        inputInfo.t_blockNew_offsetTensor, 
        inputInfo.t_columnTensor,
        inputInfo.t_valueTensor.half(), 
        inputInfo.t_rowTensor,
        
        inputInfo.x.half(), 
        
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



# 只tcu + part
def kernel_tcu_part(inputInfo, epoches):
    
    X_prime, spmm_ms_avg = Libra5BenchmarkGCN.forward_fp16_tcu_part(
        inputInfo.t_rowNew_offsetTensor,
        inputInfo.t_blockNew_offsetTensor, 
        inputInfo.t_columnTensor,
        inputInfo.t_valueTensor.half(), 
        inputInfo.t_window_rowTensor,
        inputInfo.t_atomicTensor,
        inputInfo.t_rowTensor,
        
        inputInfo.x.half(),    
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
    
    X_prime, spmm_ms_avg = Libra5BenchmarkGCN.forward_fp16_tcu_binary(
        inputInfo.t_rowNew_offsetTensor,
        inputInfo.t_blockNew_offsetTensor, 
        inputInfo.t_columnTensor,
        inputInfo.t_valueTensor.half(), 
        inputInfo.t_binaryTensor,
        # inputInfo.t_colTensor,
        # inputInfo.t_window_rowTensor,
        
        inputInfo.x.half(),    
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


