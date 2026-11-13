import torch
from scipy.sparse import *
import sys
# sys.path.append('eva100/kernel/spmm')
from libra_csr_tf32 import test_libra_csr
import csv
import pandas as pd
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


#ME-TCF
def libra_csr_test_tcu(dataset, hidden, epoches, density, partsize,data_path,  window, wide) : 
    for dimN in hidden:        
        spmm = test_libra_csr.test_tcu(dataset, epoches, dimN, density, partsize, data_path, window,wide)
        return spmm


#SGT
def libra_csr_test_tcu_sgt(dataset, hidden, epoches, density, partsize,data_path,  window, wide) : 
    for dimN in hidden:        
        spmm = test_libra_csr.test_tcu_sgt(dataset, epoches, dimN, density, partsize, data_path, window,wide)
        return spmm   
    
#BCRS
def libra_csr_test_tcu_bcrs(dataset, hidden, epoches, density, partsize,data_path,  window, wide) : 
    for dimN in hidden:        
        spmm = test_libra_csr.test_tcu_bcrs(dataset, epoches, dimN, density, partsize, data_path, window,wide)
        return spmm   
'''
只tcu + part
'''
def libra_csr_test_tcu_part(dataset, hidden, epoches, density, partsize,data_path,  window, wide) : 
    for dimN in hidden:        
        spmm = test_libra_csr.test_tcu_part(dataset, epoches, dimN, density, partsize, data_path, window,wide)
        return spmm

'''
只tcu + binary
'''
def libra_binary_test_tcu(dataset, hidden, epoches, density, partsize,data_path,  window, wide) : 
    for dimN in hidden:        
        spmm = test_libra_csr.test_tcu_binary(dataset, epoches, dimN, density, partsize, data_path, window,wide)
        return spmm
           
''' 
只tcu + part + binary
'''
def libra_csr_test_tcu_v2(dataset, hidden, epoches, density, partsize,data_path,  window, wide) : 
    for dimN in hidden:        
        spmm = test_libra_csr.test_tcu_v2(dataset, epoches, dimN, density, partsize, data_path, window,wide)
        return spmm
             
if __name__ == "__main__":
    # 获取第一个可用的 GPU 设备
    gpu_device = torch.cuda.current_device()
    
    # 打印 GPU 设备的名称
    gpu = torch.cuda.get_device_name(gpu_device)
    print(gpu)


    # dimN = int(sys.argv[1])
    dimN=128
    print('dimN: ' + str(dimN))
    hidden = []
    hidden.append(dimN)
    epoches = 10
    window = 8
    wide = 4
    
    density = [1, 2, 3, 4, 5, 6, 7, 8]
    
    partsize_t = 32
    partsize_c = 32
    shortsize = 3

    data = 'reddit'

    spmm_tcu_sgt = libra_csr_test_tcu_sgt(data, hidden, epoches, 1, partsize_t, 'data_path',  window, wide)
    print()
    spmm_tcu_sgt = libra_csr_test_tcu(data, hidden, epoches, 1, partsize_t, 'data_path',  window, wide)
    print()
    spmm_tcu_sgt = libra_csr_test_tcu_bcrs(data, hidden, epoches, 1, partsize_t, 'data_path',  window, wide)
    print()
    spmm_tcu_binary = libra_binary_test_tcu(data, hidden, epoches, 1, partsize_t, 'data_path',  window, wide)


    