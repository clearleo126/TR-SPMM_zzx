import torch
from scipy.sparse import *
import sys
sys.path.append('eva100/kernel/sddmm')
from magicsphere.mtest import *
from magicsphere.mdataset import *
from libra_csr_fp16 import test_libra_csr
import csv
import pandas as pd
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


                
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
def libra_csr_test_tcu_cuda_cc_only(dataset, hidden, epoches, density, partsize_t, partsize_c, shortsize, data_path,  window, wide, type) : 
    # for data in dataset:
    for dimN in hidden:        
        sddmm, sddmm_cc_only = test_libra_csr.test_tcu_cuda_cc_only(dataset, epoches, dimN, density, partsize_t, partsize_c, shortsize, data_path, window,wide, type)   
        return sddmm, sddmm_cc_only
    
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
    
    # density = [4, 8, 14, 18, 22, 30, 38, 50]
    density = [8]
    partsize_t = 32
    partsize_c = 32
    shortsize = 3
    # sp_matrix    gcn_matrix
    data_path = 'sp_matrix'
    # 定义目录路径和目标移动路径
    source_dir = '/home/shijinliang/module/Libra/dgl_dataset/sp_matrix'  # 要解压的目录路径
    type = 0
    #用于存储结果
    file_name = '/home/shijinliang/module/Libra-sc25/res/h100/sddmm/fp16/new_filter_h100_sddmm_fp16_result_cc_only_v1_' + str(dimN) + '.csv'
    head = ['dataSet', 'num_nodes', 'num_edges', 'sddmm_tcu_cuda_cc_only', 'sddmm_tcu_cuda', 'speedup']    # 写入数据到 CSV 文件
    with open(file_name, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(head)

    #读取csv遍历每个数据集
    df = pd.read_csv('/home/shijinliang/module/Libra/data_filter.csv')

    
    # filter = pd.read_csv('/home/shijinliang/module/Libra-sc25/res/h100/sddmm/fp16/new_filter_h100_sddmm_fp16_result_new0220_32.csv')
    # dataset_list = filter['dataSet'].tolist()

    for index, row in df.iterrows():
        res_temp = []
        res_temp.append(row.iloc[0])
        res_temp.append(row.iloc[1])
        res_temp.append(row.iloc[2])

        for dense in density:
            sddmm, sddmm_cc_only = libra_csr_test_tcu_cuda_cc_only(row.iloc[0], hidden, epoches, dense,partsize_t, partsize_c, shortsize, data_path, window, wide, 0)
            res_temp.append(sddmm)
            res_temp.append(sddmm_cc_only)
            res_temp.append(sddmm_cc_only/sddmm)
            print()
       
        with open(file_name, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(res_temp)

        print(row.iloc[0] + 'is success')
        print()