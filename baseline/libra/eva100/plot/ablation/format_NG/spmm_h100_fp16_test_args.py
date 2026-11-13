import torch
from scipy.sparse import *
import sys
# sys.path.append('eva100/kernel/spmm')
from libra_csr_fp16 import test_libra_csr
import csv
import pandas as pd
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def libra_csr_test_tcu(dataset, hidden, epoches, density, partsize,data_path,  window, wide) : 
    for dimN in hidden:        
        spmm = test_libra_csr.test_tcu(dataset, epoches, dimN, density, partsize, data_path, window,wide)
        return spmm


def libra_csr_test_tcu_csr(dataset, hidden, epoches, density, partsize,data_path,  window, wide) : 
    for dimN in hidden:        
        spmm = test_libra_csr.test_tcu_sgt(dataset, epoches, dimN, density, partsize, data_path, window,wide)
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
    wide = 8
    
    density = [1, 2, 3, 4, 5, 6, 7, 8]
    
    partsize_t = 32
    partsize_c = 32
    shortsize = 3
    # sp_matrix    gcn_matrix
    data_path = 'sp_matrix'
    # 定义目录路径和目标移动路径
    source_dir = '/home/shijinliang/module/Libra/dgl_dataset/sp_matrix'  # 要解压的目录路径
    
    #用于存储结果
    file_name = '/home/shijinliang/module/Libra/eva100/plot/ablation/format_NG/result/fp16_test_all.csv'
    file_name2 = '/home/shijinliang/module/Libra/eva100/plot/ablation/format_NG/result/fp16_test_all_v2.csv'
    head = ['dataSet', 'num_nodes', 'num_edges', 'spmm_tcu_dtc', 'spmm_tcu_binary', 'spmm_tcu']
    # 写入数据到 CSV 文件
    with open(file_name2, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(head)

    #读取csv遍历每个数据集
    df = pd.read_csv('/home/shijinliang/module/Libra/data_filter.csv')
    df1 = pd.read_csv(file_name)
    df_res = pd.merge(df1, df, on='dataSet', how='inner')  # 使用内连接（inner join）
    for index, row in df_res.iterrows():
        res_temp = []
        res_temp.append(row.iloc[0])
        res_temp.append(row.iloc[1])
        res_temp.append(row.iloc[2])

        # spmm_tcu_ori = libra_csr_test_tcu(row.iloc[0], hidden, epoches, 1, partsize_t, data_path,  window, wide)
        # res_temp.append(spmm_tcu_ori)     
        
        # #tcu 
        spmm_tcu_csr = libra_csr_test_tcu_csr(row.iloc[0], hidden, epoches, 1, partsize_t, data_path,  window, wide)
        res_temp.append(spmm_tcu_csr)
        
        res_temp.append(row.iloc[3])
        res_temp.append(row.iloc[4])
        
        #tcu + part 
        # spmm_tcu_part = libra_csr_test_tcu_part(row.iloc[0], hidden, epoches, 1, partsize_t, data_path,  window, wide)
        # res_temp.append(spmm_tcu_part)
        
        # #tcu +binary
        # spmm_tcu_binary = libra_binary_test_tcu(row.iloc[0], hidden, epoches, 1, partsize_t, data_path,  window, wide)
        # res_temp.append(spmm_tcu_binary)
    
        # #tcu + part + binary
        # spmm_tcu = libra_csr_test_tcu_v2(row.iloc[0], hidden, epoches, 1, partsize_t, data_path,  window, wide)
        # res_temp.append(spmm_tcu)
                    

        with open(file_name2, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(res_temp)
        print(row.iloc[0] + 'is success')
        print()
    
