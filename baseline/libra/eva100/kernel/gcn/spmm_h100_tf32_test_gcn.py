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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

            
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
        spmm = test_libra_csr.test_tcu_cuda(dataset, epoches, dimN, density, partsize_t, partsize_c, shortsize, data_path, window,wide)   
        return spmm
    
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
    # sp_matrix    gcn_matrix
    data_path = 'mythroughput'
    # 定义目录路径和目标移动路径
    source_dir = '/home/shijinliang/module/Libra/dgl_dataset/gcn_matrix'  # 要解压的目录路径
    
    #用于存储结果
    file_name = '/home/shijinliang/module/Libra/res_gnn/h100/spmm/tf32/h100_spmm_tf32_result_' + str(dimN) + '.csv'
    head = ['dataSet', 'num_nodes', 'num_edges', 'spmm_tcu', 'spmm_cuda', '1', '2', '3', '4', '5', '6', '7', '8', 'density', 'speedup']
    # 写入数据到 CSV 文件
    # with open(file_name, 'w', newline='') as csvfile:
    #     csv_writer = csv.writer(csvfile)
    #     csv_writer.writerow(head)

    #读取csv遍历每个数据集
    df = pd.read_csv('/home/shijinliang/module/Libra/data.csv')
    
    dataset_I = ['Coauthor_Physics',  'FacebookPagePage', 'email-Enron', 'loc-Brightkite', 'soc-Epinions1', 
                    'HR_NO', 'HU_NO', 'GitHub', 'artist', 'blog']

    dataset_II = ['ell',  'com-DBLP', 'Reddit2', 'amazon', 'amazon0505', 
                    'dd', 'yelp', 'comamazon', 'roadNet-CA', 'roadNet-PA', 
                    'roadNet-TX', 'yeast', 'DGraphFin', ]
    dataset_III = ['reddit', 'ogb', 'AmazonProducts', 'IGB_medium']
    dataset = dataset_I + dataset_II + dataset_III
    dataset = ['AmazonProducts', 'ogb', 'IGB_medium']
    dataset = ['IGB_medium', 'AmazonProducts', 'ogb', 'DGraphFin']
    for data in dataset:
        graph = np.load('/home/shijinliang/module/mnt/libra_suite/sp_matrix/' + data +'.npz')
        src_li=graph['src_li']
        num_nodes_ori  = graph['num_nodes_src']
        num_edges = len(src_li)
        # res_temp = []
        # res_temp.append(data)
        # res_temp.append(num_nodes_ori)
        # res_temp.append(num_edges)
        
        #tcu + part +binary
        spmm_tcu = libra_csr_test_tcu_v2(data, hidden, epoches, 1, partsize_t, data_path,  window, wide)
        print(num_edges*2*128 / (spmm_tcu * 1000000))
        # res_temp.append(spmm_tcu)
        
        # #纯cuda
        spmm_cuda = libra_csr_test_cuda_v2(data, hidden, epoches, 1, partsize_c, data_path, True, shortsize)
        print('cuda')
        print(num_edges*2*128 / (spmm_cuda * 1000000))
                    
        # for dense in density:
        #     spmm = libra_csr_test_tcu_cuda(data, hidden, epoches, dense,partsize_t, partsize_c, shortsize, data_path, window, wide)
        #     res_temp.append(spmm)
        #     print()
            
        # sub_list = res_temp[5:13]
        # min_value = min(sub_list)
        # min_density = density[sub_list.index(min(sub_list))]

        # if min_value< min(spmm_tcu,spmm_cuda):
        #     res_temp.append(min_density)
        #     res_temp.append(round((min(spmm_tcu,spmm_cuda)/min_value),2))
        # elif spmm_tcu < spmm_cuda :
        #     res_temp.append(0)
        #     res_temp.append(round((spmm_tcu/min_value),2))
        # else :
        #     res_temp.append(9)
        #     res_temp.append(round((spmm_cuda/min_value),2))
        # with open(file_name, 'a', newline='') as csvfile:
        #     csv_writer = csv.writer(csvfile)
        #     csv_writer.writerow(res_temp)
        print(data + 'is success')
        print()
    
    # # concat
    # df1 = pd.read_csv(file_name)
    # df2 = pd.read_csv('/home/shijinliang/module/Libra/data_concat.csv')
    # merged_df = pd.merge(df1, df2, on='dataSet', how='inner')  # 使用内连接（inner join）
    # file_name_concat = '/home/shijinliang/module/Libra/res_gnn/h100/spmm/tf32/h100_spmm_tf32_result_concat_' + str(dimN) + '.csv'
    # merged_df.to_csv(file_name_concat, index=False) 