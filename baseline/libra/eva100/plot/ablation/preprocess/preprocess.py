import torch
from scipy.sparse import *
import sys
import csv
import pandas as pd
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
sys.path.append('/home/shijinliang/module/Libra/eva100/plot/ablation/preprocess')
from libra_csr_fp16.mdataset2 import *

if __name__ == "__main__":
    # 获取第一个可用的 GPU 设备
    gpu_device = torch.cuda.current_device()
    
    # 打印 GPU 设备的名称
    gpu = torch.cuda.get_device_name(gpu_device)
    print(gpu)

    dimN=128
    window = 8
    wide = 8
    
    density = 3
    partsize_t = 32
    partsize_c = 32
    shortsize = 3
  
    data_path = 'sp_matrix'
    # 定义目录路径和目标移动路径
    source_dir = '/home/shijinliang/module/Libra/dgl_dataset/sp_matrix'  # 要解压的目录路径
    
    #用于存储结果
    file_name = '/home/shijinliang/module/Libra/eva100/plot/ablation/format_NG/result/fp16_test_all.csv'
    file_name2 = '/home/shijinliang/module/Libra/eva100/plot/ablation/preprocess/result/pre.csv'
    head = ['dataSet', 'num_nodes', 'num_edges', 'openMP', 'GPU']
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

        
        # OpenMP
        inputInfo = GCN_dataset_tcu_cuda_tf32(row.iloc[0], dimN, density, partsize_t, partsize_c, shortsize, data_path, window, wide)
    
        #GPU
        inputInfo1 = GCN_dataset_tcu_cuda_tf32_gpu(row.iloc[0], dimN, density, partsize_t, partsize_c, shortsize, data_path, window, wide)
        res_temp.append(round((inputInfo.duration.item()*1000),2))
        res_temp.append(round((inputInfo1.duration*1000),2))         

        with open(file_name2, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(res_temp)
        print(row.iloc[0] + 'is success')
        print()
    
