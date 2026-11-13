#绘制Libra，cusaprse, sputnik, Rode的图
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import coo_matrix
import scipy.sparse as sp
import torch
from collections import Counter
import pandas as pd
import seaborn as sns
import Libra5Block

window = 8
density = [1, 2, 3, 4, 5, 6, 7, 8]
graph = np.load('/home/shijinliang/module/Libra/dgl_dataset/sp_matrix/' + 'pkustk01' +'.npz')
num_nodes_ori =  graph['num_nodes_src']-0
num_nodes_dst =  graph['num_nodes_dst']-0
num_nodes = num_nodes_ori
if num_nodes_ori%window !=0 :
    num_nodes = num_nodes_ori + window - num_nodes_ori%window 
num_edges = graph['num_edges']-0
src_li = graph['src_li']
dst_li = graph['dst_li']
# edge_index = graph['edge_index_new']
edge_index = np.stack([src_li, dst_li])
avg_degree = num_edges / num_nodes
#print("Num_nodes, Num_edges: " + str(num_nodes) + ' , ' + str(num_edges))
val = [1] * num_edges
scipy_coo = coo_matrix((val, edge_index), shape=(num_nodes, num_nodes_dst))
adj = scipy_coo.tocsr()
total =len(src_li)
column_index = torch.IntTensor(adj.indices)
row_pointers = torch.IntTensor(adj.indptr)
degrees = torch.randn(num_edges)

for den in density:
    t_rowNew_offsetTensor, \
    t_blockTensor, \
    t_columnTensor, \
    t_valueTensor, \
    t_window_rowTensor, \
    t_atomicTensor, \
    t_binaryTensor, \
    c_row_offsetTensor, \
    c_rowTensor, \
    c_atomicTensor, \
    c_colTensor, \
    c_valueTensor, \
    c_row_offsetTensor_short, \
    c_rowTensor_short, \
    c_atomicTensor_short, \
    c_colTensor_short, \
    c_valueTensor_short = Libra5Block.block_tf32_tcu_cuda_binary(row_pointers, column_index, degrees, 32, 32, 3, den, window, 8)
    
    tcu = t_valueTensor.shape[0]
    cuda = c_valueTensor.shape[0] + c_colTensor_short.shape[0]
    print('density-', den, ' tcu: ', (tcu/total)*100)

