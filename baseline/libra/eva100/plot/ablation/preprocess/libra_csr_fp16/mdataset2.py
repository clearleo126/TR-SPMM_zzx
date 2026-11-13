#!/usr/bin/env python3
import torch
import numpy as np
import torch.nn.functional as F
import scipy.sparse as sp
from scipy.sparse import coo_matrix
import Libra5Block
from scipy.sparse import *
import DTCSpMM_pre

class GCN_dataset_tcu_cuda_tf32(torch.nn.Module):
    """
    data loading for more graphs
    """
    def __init__(self, data, dimN, density, partsize_t, partsize_c, shortsize, data_path,  window, wide):
        super(GCN_dataset_tcu_cuda_tf32, self).__init__()

        self.graph = np.load('/home/shijinliang/module/mnt/libra_suite/' + data_path + '/' + data +'.npz')
        self.num_features = dimN
        self.init_edges(density,  partsize_t, partsize_c, shortsize, window, wide)


        
        
    def init_edges(self, density, partsize_t, partsize_c, shortsize,  window, wide):
        # loading from a .npz graph file
        self.num_nodes_ori =  self.graph['num_nodes_src']-0
        self.num_nodes_dst =  self.graph['num_nodes_dst']-0
        self.num_nodes = self.num_nodes_ori
        if self.num_nodes_ori%window !=0 :
            self.num_nodes = self.num_nodes_ori + window - self.num_nodes_ori%window 
        self.num_edges = self.graph['num_edges']-0
        src_li = self.graph['src_li']
        dst_li = self.graph['dst_li']
        # self.edge_index = self.graph['edge_index_new']
        self.edge_index = np.stack([src_li, dst_li])
        self.avg_degree = self.num_edges / self.num_nodes
        #print("Num_nodes, Num_edges: " + str(self.num_nodes) + ' , ' + str(self.num_edges))
        val = [1] * self.num_edges
        scipy_coo = coo_matrix((val, self.edge_index), shape=(self.num_nodes, self.num_nodes_dst))
        adj = scipy_coo.tocsr()
        
        self.column_index = torch.IntTensor(adj.indices)
        self.row_pointers = torch.IntTensor(adj.indptr)
        self.degrees = torch.randn(self.num_edges)
        
        # #统计
        # rows_per_group = 8
        # num_rows = self.num_nodes
        # num_groups = (num_rows + rows_per_group - 1) // rows_per_group  # 向上取整
        # res_nnz = []
        # res_col = []
        # sparse = []
        # # 遍历每个组
        # for group_index in range(num_groups):
        #     # 计算组的起始和结束行
        #     start_row = group_index * rows_per_group
        #     end_row = min((group_index + 1) * rows_per_group, num_rows)

        #     # 获取该组的起始和结束索引在 indptr 中的位置
        #     start_index = self.row_pointers[start_row]
        #     end_index = self.row_pointers[end_row]

        #     # 计算元素个数和列索引数
        #     num_elements = end_index - start_index
        #     num_columns = len(np.unique(self.column_index[start_index:end_index]))
        #     res_nnz.append(num_elements.item())
        #     res_col.append(num_columns)            
        #     sparse.append(num_elements.item()/(num_columns*8))
        # print()  
        
        
        self.t_rowNew_offsetTensor, \
        self.t_blockTensor, \
        self.t_columnTensor, \
        self.t_valueTensor, \
        self.t_window_rowTensor, \
        self.t_atomicTensor, \
        self.t_binaryTensor, \
        self.c_row_offsetTensor, \
        self.c_rowTensor, \
        self.c_atomicTensor, \
        self.c_colTensor, \
        self.c_valueTensor, \
        self.c_row_offsetTensor_short, \
        self.c_rowTensor_short, \
        self.c_atomicTensor_short, \
        self.c_colTensor_short, \
        self.c_valueTensor_short, \
        self.duration = Libra5Block.block_tf32_tcu_cuda_binary(self.row_pointers, self.column_index, self.degrees, partsize_t, partsize_c, shortsize, density, window, wide)
        
        # value_pad= torch.tensor([0])
        # if self.t_valueTensor.shape[0]%2==1 :
        #     self.t_valueTensor = torch.cat((self.t_valueTensor, value_pad), dim=0)
        # if self.c_valueTensor.shape[0]%2==1 :
        #     self.c_valueTensor = torch.cat((self.c_valueTensor, value_pad), dim=0)
        # if self.c_valueTensor_short.shape[0]%2==1 :
        #     self.c_valueTensor_short = torch.cat((self.c_valueTensor_short, value_pad), dim=0)
        self.parts_t = self.t_atomicTensor.shape[0]
        self.parts_c = self.c_atomicTensor.shape[0]
        self.parts_c_short = self.c_atomicTensor_short.shape[0]
        self.partsize_c = partsize_c
        self.t_valueTensor = self.t_valueTensor.half()
        self.c_valueTensor = self.c_valueTensor.half()
        self.c_valueTensor_short = self.c_valueTensor_short.half()
    def init_embedding(self):
        '''
        Generate node embedding for nodes.
        Called from __init__.
        '''
        self.x = torch.randn(self.num_nodes_dst, self.num_features)
        self.x = self.x.half()




class GCN_dataset_tcu_cuda_tf32_gpu(torch.nn.Module):
    """
    data loading for more graphs
    """
    def __init__(self, data, dimN, density, partsize_t, partsize_c, shortsize, data_path,  window, wide):
        super(GCN_dataset_tcu_cuda_tf32_gpu, self).__init__()

        self.graph = np.load('/home/shijinliang/module/mnt/libra_suite/' + data_path + '/' + data +'.npz')
        self.num_features = dimN
        self.init_edges(density,  partsize_t, partsize_c, shortsize, window, wide)


        
        
    def init_edges(self, density, partsize_t, partsize_c, shortsize,  window, wide):
        # loading from a .npz graph file
        self.num_nodes_ori =  self.graph['num_nodes_src']-0
        self.num_nodes_dst =  self.graph['num_nodes_dst']-0
        self.num_nodes = self.num_nodes_ori
        if self.num_nodes_ori%window !=0 :
            self.num_nodes = self.num_nodes_ori + window - self.num_nodes_ori%window 
        self.num_edges = self.graph['num_edges']-0
        src_li = self.graph['src_li']
        dst_li = self.graph['dst_li']
        # self.edge_index = self.graph['edge_index_new']
        self.edge_index = np.stack([src_li, dst_li])
        self.avg_degree = self.num_edges / self.num_nodes
        #print("Num_nodes, Num_edges: " + str(self.num_nodes) + ' , ' + str(self.num_edges))
        val = [1] * self.num_edges
        scipy_coo = coo_matrix((val, self.edge_index), shape=(self.num_nodes, self.num_nodes_dst))
        adj = scipy_coo.tocsr()
        
        self.column_index = torch.IntTensor(adj.indices).cuda()
        self.row_pointers = torch.IntTensor(adj.indptr).cuda()
        
        
        self.WindowOffset_tensor, \
        self.Curwindow_tensor, \
        self.t_Atomic_tensor, \
        self.ColumnIndice_tensor, \
        self.BlockOffset_tensor, \
        self.Binary_tensor, \
        self.t_Value_tensor, \
        self.cuda_long_group_tensor, \
        self.cuda_long_row_tensor, \
        self.cuda_long_atomic_tensor, \
        self.cuda_long_column_tensor, \
        self.cuda_long_value_tensor, \
        self.cuda_short_group_tensor, \
        self.cuda_short_row_tensor, \
        self.cuda_short_atomic_tensor, \
        self.cuda_short_column_tensor, \
        self.cuda_short_value_tensor, \
        self.duration = DTCSpMM_pre.preprocess_gpu_libra_spmm(self.column_index , self.row_pointers, self.num_nodes_ori, window, wide, density, shortsize, partsize_t, partsize_c, int(self.num_nodes/window), self.num_edges)
        
        # value_pad= torch.tensor([0])
        # if self.t_valueTensor.shape[0]%2==1 :
        #     self.t_valueTensor = torch.cat((self.t_valueTensor, value_pad), dim=0)
        # if self.c_valueTensor.shape[0]%2==1 :
        #     self.c_valueTensor = torch.cat((self.c_valueTensor, value_pad), dim=0)
        # if self.c_valueTensor_short.shape[0]%2==1 :
        #     self.c_valueTensor_short = torch.cat((self.c_valueTensor_short, value_pad), dim=0)
       
    def init_embedding(self):
        '''
        Generate node embedding for nodes.
        Called from __init__.
        '''
        self.x = torch.randn(self.num_nodes_dst, self.num_features)
        self.x = self.x.half()