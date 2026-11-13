import torch
from scipy.sparse import *
import sys
sys.path.append('eva100/kernel/sddmm')
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


    dimN = int(sys.argv[1])
    # dimN=64
    print('dimN: ' + str(dimN))
    hidden = []
    hidden.append(dimN)
    epoches = 10
    window = 8
    wide = 16
    
    density = [4, 8, 14, 18, 22, 30, 38, 50]
    
    partsize_t = 32
    partsize_c = 32
    shortsize = 3
    # sp_matrix    gcn_matrix
    data_path = 'sp_matrix'
    # 定义目录路径和目标移动路径
    source_dir = '/home/shijinliang/module/Libra/dgl_dataset/sp_matrix'  # 要解压的目录路径
    type = 0
    #用于存储结果
    file_name = '/home/shijinliang/module/Libra/res/a800/sddmm/tf32/filter_a800_sddmm_tf32_result_' + str(dimN) + '.csv'
    head = ['dataSet', 'num_nodes', 'num_edges', 'sddmm_tcu', 'sddmm_cuda', '4', '8', '14', '18', '22', '30', '38', '50', 'density', 'speedup']
    # 写入数据到 CSV 文件
    with open(file_name, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(head)

    #读取csv遍历每个数据集
    if dimN == 128:
        df = pd.read_csv('/home/shijinliang/module/Libra/data_filter.csv')
    else :
        df = pd.read_csv('/home/shijinliang/module/Libra/data_filter_256.csv')
    for index, row in df.iterrows():
        res_temp = []
        res_temp.append(row.iloc[0])
        res_temp.append(row.iloc[1])
        res_temp.append(row.iloc[2])
        
        #纯tcu
        sddmm_tcu = libra_csr_test_tcu(row.iloc[0], hidden, epoches, 1, partsize_t, data_path,  window, wide)
        res_temp.append(sddmm_tcu)
        
        #纯cuda
        sddmm_cuda = libra_csr_test_cuda(row.iloc[0], hidden, epoches, 1, partsize_c, data_path, type)
        res_temp.append(sddmm_cuda)
                    
        for dense in density:
            sddmm = libra_csr_test_tcu_cuda(row.iloc[0], hidden, epoches, dense,partsize_t, partsize_c, shortsize, data_path, window, wide, type)
            res_temp.append(sddmm)
            print()
            
        sub_list = res_temp[5:13]
        min_value = min(sub_list)
        min_density = density[sub_list.index(min(sub_list))]

        if min_value< min(sddmm_tcu,sddmm_cuda):
            res_temp.append(min_density)
            res_temp.append(round((min(sddmm_tcu,sddmm_cuda)/min_value),2))
        elif sddmm_tcu < sddmm_cuda :
            res_temp.append(0)
            res_temp.append(round((sddmm_tcu/min_value),2))
        else :
            res_temp.append(128)
            res_temp.append(round((sddmm_cuda/min_value),2))
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