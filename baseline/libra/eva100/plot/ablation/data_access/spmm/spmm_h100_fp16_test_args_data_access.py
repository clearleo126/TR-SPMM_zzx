import torch
from scipy.sparse import *
import sys
sys.path.append('/home/shijinliang/module/ppopp26-Libra/eva100/plot/ablation/tcu_utilization/')
from test_libra_csr import *
import csv
import pandas as pd
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import math
# -*- coding: utf-8 -*-



def libra_csr_test_cuda_v2(dataset, hidden, epoches, density, partsize,data_path,  swizzle, shortSize) : 
    for dimN in hidden:        
        ratio = test_cuda_v2(dataset, epoches, dimN, density, partsize, data_path, swizzle, shortSize)
        return ratio    
    
    
'''
tcu cuda
'''
def libra_csr_test_tcu_cuda(dataset, hidden, epoches, density, partsize_t, partsize_c, shortsize, data_path,  window, wide) : 
    # for data in dataset:
    for dimN in hidden:        
        access_cuda, access_hybrid = test_tcu_cuda(dataset, epoches, dimN, density, partsize_t, partsize_c, shortsize, data_path, window,wide)   
        return access_cuda, access_hybrid

           
''' 
只tcu with part
'''
def libra_csr_test_tcu_v2(dataset, hidden, epoches, density, partsize,data_path,  window, wide) : 
    for dimN in hidden:        
        ratio = test_tcu_v2(dataset, epoches, dimN, density, partsize, data_path, window,wide)
        return ratio
             
if __name__ == "__main__":
   
    dimN=128
    # print('dimN: ' + str(dimN))
    hidden = []
    hidden.append(dimN)
    epoches = 10
    window = 8
    wide = 8
   
    partsize_t = 32
    partsize_c = 32
    shortsize = 3
    # sp_matrix    gcn_matrix
    data_path = 'sp_matrix'
    # 定义目录路径和目标移动路径
    source_dir = '/home/shijinliang/module/Libra-sc25/dgl_dataset/sp_matrix'  # 要解压的目录路径
    
    #用于存储结果
    file_name = '/home/shijinliang/module/ppopp26-Libra/eva100/plot/ablation/data_access/spmm/result1.csv'

    head = ['dataSet', 'num_nodes', 'num_edges', 'cuda', 'hybrid']    # 写入数据到 CSV 文件
    with open(file_name, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(head)



    #读取csv遍历每个数据集
    df = pd.read_csv('/home/shijinliang/module/Libra/res/h100/spmm/fp16/filter_h100_spmm_fp16_result_new_0216_128.csv')
    for index, row in df.iterrows():
        
        
        res_temp = []
        res_temp.append(row.iloc[0])
        res_temp.append(row.iloc[1])
        res_temp.append(row.iloc[2])

        #Bitmap
        access_cuda, access_hybrid = libra_csr_test_tcu_cuda(row.iloc[0], hidden, epoches, row['density'],partsize_t, partsize_c, shortsize, data_path, window, wide)
        res_temp.append(access_cuda)
        res_temp.append(access_hybrid)
        print()


        with open(file_name, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(res_temp)
        print(row.iloc[0] + 'is success')
        print()