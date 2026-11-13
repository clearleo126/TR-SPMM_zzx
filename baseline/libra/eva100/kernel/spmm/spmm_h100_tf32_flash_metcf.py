import torch
from scipy.sparse import *
import sys
sys.path.append('eva100/kernel/spmm')
from cusparse import test_cusparse
from tcgnn import test_tcgnn
from libra_csr_tf32_flash import test_libra_csr
import csv
import pandas as pd
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

            
'''
Cusparse
'''
def cusparse_test(dataset, hidden, epoches ,data_path) : 
    for dimN in hidden:
        spmm = test_cusparse.test(dataset, epoches, dimN, data_path)
        return spmm
          
          
'''
只cuda, 长短行
'''
def libra_csr_test_cuda_v2(dataset, hidden, epoches, density, partsize,data_path,  swizzle, shortSize) : 
    for dimN in hidden:        
        spmm = test_libra_csr.test_cuda_v2(dataset, epoches, dimN, density, partsize, data_path, swizzle, shortSize)
        return spmm    
            
'''
TCGNN
'''
def tcgnn_test(dataset, hidden, epoches ,data_path) : 
    for data in dataset:
        for dimN in hidden:
            spmm = test_tcgnn.test(data, epoches, dimN, data_path)


'''
tcu cuda
'''
def libra_csr_test_tcu_cuda(dataset, hidden, epoches, density, partsize_t, partsize_c, shortsize, data_path,  window, wide) : 
    # for data in dataset:
    for dimN in hidden:        
        spmm, duration = test_libra_csr.test_tcu_cuda(dataset, epoches, dimN, density, partsize_t, partsize_c, shortsize, data_path, window,wide)   
        return spmm, duration
    
'''
只tcu without part
'''
def libra_csr_test_tcu(dataset, hidden, epoches, density, partsize,data_path,  window, wide) : 
    for dimN in hidden:        
        spmm = test_libra_csr.test_tcu(dataset, epoches, dimN, density, partsize, data_path, window,wide)
        return spmm

'''
只tcu without part binary
'''
def libra_csr_binary_test_tcu(dataset, hidden, epoches, density, partsize,data_path,  window, wide) : 
    for dimN in hidden:        
        spmm = test_libra_csr.test_tcu_binary(dataset, epoches, dimN, density, partsize, data_path, window,wide)
        return spmm
           
''' 
只tcu with part
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


    dimN = int(sys.argv[1])
    # dimN=128
    # print('dimN: ' + str(dimN))
    hidden = []
    hidden.append(dimN)
    epoches = 10
    window = 8
    wide = 4
    
    density = [1, 2, 3, 4, 5, 6, 7, 8]
    
    partsize_t = 32
    partsize_c = 32
    shortsize = 3
    # sp_matrix    gcn_matrix
    data_path = 'sp_matrix'
    # 定义目录路径和目标移动路径
    source_dir = '/home/shijinliang/module/Libra-sc25/dgl_dataset/sp_matrix'  # 要解压的目录路径
    
    #用于存储结果
    file_name = '/home/shijinliang/module/Libra-sc25/res/h100/spmm/tf32/h100_spmm_tf32_result_flash_' + str(dimN) + '.csv'
    head = ['dataSet', 'num_nodes', 'num_edges', 'flash']
    # 写入数据到 CSV 文件
    with open(file_name, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(head)

    #读取csv遍历每个数据集
    if dimN == 128:
        df = pd.read_csv('/home/shijinliang/module/Libra-sc25/data_filter.csv')
    else :
        df = pd.read_csv('/home/shijinliang/module/Libra-sc25/data_filter_256.csv')
    for index, row in df.iterrows():
        res_temp = []
        res_temp.append(row.iloc[0])
        res_temp.append(row.iloc[1])
        res_temp.append(row.iloc[2])
        
        # #tcu + part +binary
        spmm_tcu = libra_csr_test_tcu_v2(row.iloc[0], hidden, epoches, 1, partsize_t, data_path,  window, wide)
        res_temp.append(spmm_tcu)
        
        with open(file_name, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(res_temp)
        print(row.iloc[0] + 'is success')
        print()
    # # concat
    # df1 = pd.read_csv(file_name)
    # df2 = pd.read_csv('/home/shijinliang/module/Libra/data_concat.csv')
    # merged_df = pd.merge(df1, df2, on='dataSet', how='inner')  # 使用内连接（inner join）
    # file_name_concat = '/home/shijinliang/module/Libra/res/h100/spmm/fp16/h100_spmm_fp16_result_concat_' + str(dimN) + '.csv'
    # merged_df.to_csv(file_name_concat, index=False) 