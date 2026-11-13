import os
import sys
# sys.path.append('./eva100/kernel/gcn')
from mdataset2 import *

import Libra6SDDMM




# 只tcu 


def test_tcu(data, epoches, dimN, density, partsize, data_path,  window, wide):

    inputInfo = GCN_dataset_tcu(data, dimN, density, partsize, data_path, window, wide)

    ratio = (inputInfo.num_edges  / (inputInfo.t_columnTensor.shape[0]*8)) * 100
    return ratio



# tcu cuda


def test_tcu_cuda(data, epoches, dimN, density, partsize_t, partsize_c, shortsize, data_path,  window, wide, type):

    
    inputInfo = GCN_dataset_tcu_cuda(data, dimN, density, partsize_t, partsize_c, shortsize, data_path, window, wide)
   
    if inputInfo.t_rowNew_offsetTensor.shape[0] == 1:
        ratio = 100
    else:
        num_tcu = inputInfo.num_edges - inputInfo.c_colTensor.shape[0]
        ratio = (num_tcu  / (inputInfo.t_columnTensor.shape[0]*8)) * 100
    return ratio