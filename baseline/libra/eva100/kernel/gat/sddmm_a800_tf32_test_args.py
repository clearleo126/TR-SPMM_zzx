import torch
from scipy.sparse import *
import sys
sys.path.append('eva100/kernel/gcn')
from advisor import test_advisor
from cusparse import test_cusparse
from magicsphere.mtest import *
from magicsphere.mdataset import *
from tcgnn import test_tcgnn
from libra_csr_tf32 import test_libra_csr
import csv
import pandas as pd
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

            
    

'''
只tcu with part binary
'''
def libra_binary_test_tcu(dataset, hidden, epoches, density, partsize,data_path,  window, wide) : 
    for dimN in hidden:        
        spmm = test_libra_csr.libra_binary_tcu(dataset, epoches, dimN, density, partsize, data_path, window,wide)
        return spmm
           
''' 
只tcu with part
'''
def libra_csr_test_tcu(dataset, hidden, epoches, density, partsize,data_path,  window, wide) : 
    for dimN in hidden:        
        spmm = test_libra_csr.libra_csr_tcu(dataset, epoches, dimN, density, partsize, data_path, window,wide)
        return spmm
   
''' 
只cuda
'''
def libra_csr_test_cuda(dataset, hidden, epoches, density, partsize_c, data_path,  window, wide) : 
    for dimN in hidden:        
        spmm = test_libra_csr.libra_cuda(dataset, epoches, dimN, density, partsize_c, data_path, window,wide)
        return spmm
    
''' 
只cuda
'''
def libra_csr_test_cuda_v2(dataset, hidden, epoches, density, partsize_c, data_path,  window, wide) : 
    for dimN in hidden:        
        spmm = test_libra_csr.libra_cuda_v2(dataset, epoches, dimN, density, partsize_c, data_path, window,wide)
        return spmm
    
''' 
cuda + tcu_binary
'''
def libra_csr_test_tcu_cuda(dataset, hidden, epoches, density, partsize_t, partsize_c,data_path,  window, wide,type) : 
    for dimN in hidden:        
        spmm = test_libra_csr.libra_binary_cuda_tcu(dataset, epoches, dimN, density, partsize_t, partsize_c, data_path, window,wide,type)
        return spmm
              
if __name__ == "__main__":
    # 获取第一个可用的 GPU 设备
    gpu_device = torch.cuda.current_device()
    
    # 打印 GPU 设备的名称
    gpu = torch.cuda.get_device_name(gpu_device)
    print(gpu)
    
    #sys.argv[0] dimN
    dimN = int(sys.argv[1])
    print('dimN: ' + str(dimN))
    hidden = []
    hidden.append(dimN)
    epoches = 100
    window = 8
    wide = 16
    
    density = [10, 16, 18, 22, 26, 30, 35]
    
    partsize_t = 32
    partsize_c = 32
    shortsize = 3
    # sp_matrix    gcn_matrix
    data_path = 'sp_matrix'
    # 定义目录路径和目标移动路径
    source_dir = '/home/shijinliang/module/Libra/dgl_dataset/sp_matrix'  # 要解压的目录路径
    
    #用于存储结果
    file_name = '/home/shijinliang/module/Libra/res/a800/sddmm/tf32/a800_sddmm_tf32_result_' + str(dimN) + '.csv'
    head = ['dataSet', 'num_nodes', 'num_edges', 'tcu_csr', 'tcu_binary', 'cuda', '10', '16', '18', '22', '26', '30', '35', 'density', 'speedup']
    # head = ['dataSet', 'num_nodes', 'num_edges', 'tcu', 'tcu_binary', 'cuda', 'libra']
    # 写入数据到 CSV 文件
    with open(file_name, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(head)

    #读取csv遍历每个数据集
    df = pd.read_csv('/home/shijinliang/module/Libra/data.csv')
    
    for index, row in df.iterrows():
        res_temp = []
        res_temp.append(row.iloc[0])
        res_temp.append(row.iloc[1])
        res_temp.append(row.iloc[2])
        
        #执行纯tcu with part
        spmm_tcu_csr = libra_csr_test_tcu(row.iloc[0], hidden, epoches, 1, partsize_t, data_path, window, wide)
        
        #执行纯tcu with part_binary
        spmm_tcu_binary = libra_binary_test_tcu(row.iloc[0], hidden, epoches, 1, partsize_t, data_path, window, wide)
        

        #执行纯cuda with part    
        spmm_cuda = libra_csr_test_cuda(row.iloc[0], hidden, epoches, 1, partsize_c, data_path, True, shortsize)
        spmm_cuda_navie = libra_csr_test_cuda_v2(row.iloc[0], hidden, epoches, 1, partsize_c, data_path, True, shortsize)
        type = 0
        if spmm_cuda_navie < spmm_cuda :
            type = 1

        res_temp.append(spmm_tcu_csr)
        res_temp.append(spmm_tcu_binary)
        res_temp.append(min(spmm_cuda,spmm_cuda_navie))
                    
        for dense in density:
            spmm = libra_csr_test_tcu_cuda(row.iloc[0], hidden, epoches, dense,partsize_t, partsize_c, data_path, window, wide, type)
            res_temp.append(spmm)
            print()
            
        sub_list = res_temp[6:15]
        min_value = min(sub_list)
        min_density = density[sub_list.index(min(sub_list))]
        tcu = spmm_tcu_binary
        cuda = min(spmm_cuda,spmm_cuda_navie)
        if min_value< min(tcu, cuda):
            res_temp.append(min_density)
            res_temp.append(round((min(tcu,cuda)/min_value),2))
        elif tcu < cuda :
            res_temp.append(0)
            res_temp.append(round((tcu/min_value),2))
        else :
            res_temp.append(9)
            res_temp.append(round((cuda/min_value),2))
        with open(file_name, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(res_temp)
        print(row.iloc[0] + 'is success')
        print()
    # concat
    df1 = pd.read_csv(file_name)
    df2 = pd.read_csv('/home/shijinliang/module/Libra/data_concat.csv')
    merged_df = pd.merge(df1, df2, on='dataSet', how='inner')  # 使用内连接（inner join）
    file_name_concat = '/home/shijinliang/module/Libra/res/a800/sddmm/tf32/a800_sddmm_tf32_result_concat_' + str(dimN) + '.csv'
    merged_df.to_csv(file_name_concat, index=False) 