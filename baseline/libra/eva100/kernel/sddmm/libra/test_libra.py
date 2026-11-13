import os
import sys
# sys.path.append('./eva100/kernel/gcn')
from libra.mdataset2 import *
import Libra5BenchmarkGCN

def kernel(inputInfo, epoches):
    
        # X_prime, spmm_ms_avg = Libra5BenchmarkGCN.forward_tf32(
        # inputInfo.t_row_offsetTensor_,
        # inputInfo.t_colTensor_, 
        # inputInfo.t_valueTensor_, 
        # inputInfo.t_rowTensor_,
        # inputInfo.t_c_partTensor_,
        # inputInfo.t_c_row_offsetTensor_,
        # inputInfo.t_c_rowTensor_,
        # inputInfo.t_c_atomicTensor_,   
        # inputInfo.t_c_colTensor_,
        # inputInfo.t_c_valueTensor_, 
        # inputInfo.c_row_offsetTensor_,
        # inputInfo.c_rowTensor_,
        # inputInfo.c_atomicTensor_,   
        # inputInfo.c_colTensor_,
        # inputInfo.c_valueTensor_, 
        # inputInfo.x,    
        # inputInfo.num_nodes, 
        # inputInfo.x.size(1), 
        # inputInfo.num_nodes_ori, 
        # epoches,
        # inputInfo.boundary,
        # inputInfo.parts)
    X_prime, spmm_ms_avg = Libra5BenchmarkGCN.forward_tf32_v4(
        inputInfo.t_row_offsetTensor_,
        inputInfo.t_colTensor_, 
        inputInfo.t_valueTensor_, 
        inputInfo.t_rowTensor_,
        inputInfo.c_row_offsetTensor_,
        inputInfo.c_colTensor_,
        inputInfo.c_valueTensor_, 
        inputInfo.x,    
        inputInfo.c_partTensor_,
        inputInfo.c_rowTensor_,
        inputInfo.c_atomicTensor_,   
        inputInfo.num_nodes, 
        inputInfo.x.size(1), 
        inputInfo.num_nodes_ori, 
        epoches,
        inputInfo.boundary,
        inputInfo.parts)
    return round(spmm_ms_avg.item(),4)

def test(data, epoches, dimN, density, partsize):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    inputInfo = GCN_dataset(data, dimN, density, partsize)
    inputInfo.to(device)
   
    execution_time = kernel(inputInfo, epoches)
    print(str(dimN) + '-' + data + ' libra-' + str(density) + '-' + str(execution_time))
    return execution_time
