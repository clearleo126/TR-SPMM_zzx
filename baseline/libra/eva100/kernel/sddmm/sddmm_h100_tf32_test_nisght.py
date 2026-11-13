import torch
from scipy.sparse import *
import sys
sys.path.append('eva100/kernel/sddmm')
from libra_csr_tf32 import test_libra_csr
import csv
import pandas as pd
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


                
'''
只cuda, 长短行
'''
def libra_csr_test_cuda(dataset, hidden, epoches, density, partsize,data_path,type) : 
    for dimN in hidden:        
        sddmm = test_libra_csr.test_cuda(dataset, epoches, dimN, density, partsize, data_path, type)
        return sddmm    

'''
tcu cuda
'''
def libra_csr_test_tcu_cuda(dataset, hidden, epoches, density, partsize_t, partsize_c, shortsize, data_path,  window, wide, type) : 
    # for data in dataset:
    for dimN in hidden:        
        sddmm = test_libra_csr.test_tcu_cuda(dataset, epoches, dimN, density, partsize_t, partsize_c, shortsize, data_path, window,wide, type)   
        return sddmm
    
'''
只tcu
'''
def libra_csr_test_tcu(dataset, hidden, epoches, density, partsize,data_path,  window, wide) : 
    for dimN in hidden:        
        sddmm = test_libra_csr.test_tcu(dataset, epoches, dimN, density, partsize, data_path, window,wide)
        return sddmm

             
if __name__ == "__main__":
    # 获取第一个可用的 GPU 设备
    gpu_device = torch.cuda.current_device()
    
    # 打印 GPU 设备的名称
    gpu = torch.cuda.get_device_name(gpu_device)
    print(gpu)


    # dimN = int(sys.argv[1])
    dimN=32
    print('dimN: ' + str(dimN))
    hidden = []
    hidden.append(dimN)
    epoches = 10
    window = 8
    wide = 16
    
    density = [4]
    
    partsize_t = 32
    partsize_c = 32
    shortsize = 3
    # sp_matrix    gcn_matrix
    data_path = 'sp_matrix'
    # 定义目录路径和目标移动路径
    source_dir = '/home/shijinliang/module/Libra/dgl_dataset/sp_matrix'  # 要解压的目录路径
    type = 0
    #用于存储结果
    file_name = '/home/shijinliang/module/Libra/res/h100/sddmm/tf32/new_filter_h100_sddmm_tf32_result_' + str(dimN) + '.csv'
    head = ['dataSet', 'num_nodes', 'num_edges', 'sddmm_tcu', 'sddmm_cuda', '4', '8', '14', '18', '22', '30', '38', '50', 'density', 'speedup']
    data = sys.argv[1]
        
        # #纯tcu
    sddmm_tcu = libra_csr_test_tcu(data, hidden, epoches, 1, partsize_t, data_path,  window, wide)
       