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
    access_cuda = inputInfo.num_edges * 2 * 128 * 2
    
    mask = inputInfo.t_columnTensor >= 0
    # 2. 把 True 当作 1，False 当作 0，求和就是>=0的元素个数
    count = int(mask.sum().item())
    count1 = inputInfo.t_rowNew_offsetTensor[-1].item()
    access_hybrid = ((count1*8 + count)*128 + inputInfo.c_colTensor.shape[0]*2*128) * 2
   
    return access_cuda, access_hybrid