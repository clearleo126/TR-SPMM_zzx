#!/usr/bin/env python3
import torch
import numpy as np
import torch.nn.functional as F
import scipy.sparse as sp
from scipy.sparse import coo_matrix
import Libra5Block
import Libra6SpMM
from scipy.sparse import *
def func(x):
    '''
    node degrees function
    '''
    if x > 0:
        return x
    else:
        return 1
class GCN_dataset(torch.nn.Module):
    """
    data loading for more graphs
    """
    def __init__(self, data, dimN, density,partsize, data_path,  window, wide, kernel):
        super(GCN_dataset, self).__init__()

        self.graph = np.load('/home/shijinliang/module/mnt/libra_suite/' + data_path + '/' + data +'.npz')
        self.num_features = dimN
        self.init_edges(density, partsize,  window, wide, kernel)
        self.init_embedding()

        
        
    def init_edges(self, density, partSize,  window, wide, kernel):
        # loading from a .npz graph file
        self.num_nodes_ori =  self.graph['num_nodes_src']-0
        self.num_nodes_dst =  self.graph['num_nodes_dst']-0
        self.num_nodes = self.num_nodes_ori
        if self.num_nodes_ori%8 !=0 :
            self.num_nodes = self.num_nodes_ori + 8 - self.num_nodes_ori%8 
        self.num_edges = self.graph['num_edges']-0
        src_li = self.graph['src_li']
        dst_li = self.graph['dst_li']
        # self.edge_index = self.graph['edge_index_new']
        self.edge_index = np.stack([src_li, dst_li])
        self.avg_degree = self.num_edges / self.num_nodes
        print("Num_nodes, Num_edges: " + str(self.num_nodes) + ' , ' + str(self.num_edges))
        val = [1] * self.num_edges
        scipy_coo = coo_matrix((val, self.edge_index), shape=(self.num_nodes, self.num_nodes_dst))
        adj = scipy_coo.tocsr()
        
        self.column_index = torch.IntTensor(adj.indices)
        self.row_pointers = torch.IntTensor(adj.indptr)
        self.degrees = torch.randn(self.num_edges)
        

        self.t_windowNew_offsetTensor, \
        self.t_blockNew_offsetTensor, \
        self.t_valueTensor, \
        self.t_columnTensor, \
        self.t_rowTensor, \
        self.t_colTensor, \
        self.t_window_rowTensor, \
        self.t_row_offsetTensor, \
        self.c_part_offsetTensor, \
        self.c_row_offsetTensor, \
        self.c_rowTensor, \
        self.c_atomicTensor, \
        self.c_colTensor, \
        self.c_valueTensor = Libra5Block.block_csr(self.row_pointers, self.column_index,self.degrees, density, partSize,  window, wide, kernel)

        # profile nnz per row
        # 使用unique函数获取唯一值及其对应的计数
        unique_values, counts = torch.unique((self.c_row_offsetTensor[1:] - self.c_row_offsetTensor[:-1]), return_counts=True)

        print("total")
        # 将唯一值及其对应的计数打印出来
        print("行数:", self.c_row_offsetTensor.shape[0])
        for value, count in zip(unique_values, counts):
            print("值:", value.item(), "出现次数:", count.item())
        print()
        
        unique_values1, counts1 = torch.unique((self.t_row_offsetTensor), return_counts=True)
        print("tcu 所在的行")
        # 将唯一值及其对应的计数打印出来
        print("行数:", self.t_row_offsetTensor.shape[0])
        for value, count in zip(unique_values1, counts1):
            print("值:", value.item(), "出现次数:", count.item())
        
        self.boundary =  self.t_window_rowTensor.shape[0]
        self.parts = self.c_part_offsetTensor[-1].item()
        print("parts: " + str(self.parts))
        # print("automic and not automic parts: " + str(torch.sum(self.c_atomicTensor == 1).item()) + ", " + str(torch.sum(self.c_atomicTensor == 0).item()))
        print("tcu-nnz, cuda-nnz: " + str(self.num_edges-self.c_row_offsetTensor[-1].item()) + ", " + str(self.c_row_offsetTensor[-1].item()))
        # print("vectors: " + str(  self.t_windowNew_offsetTensor[-1].item()*4))
        
        
        # # 对 self.c_row_offsetTensor 进行分割
        # temp = self.c_row_offsetTensor[1:] - self.c_row_offsetTensor[:-1]
        # # 对张量进行降序排列，并获取排序后的索引
        # sorted_tensor, sorted_indices = torch.sort(temp, dim=0, descending=True)
        # zero_tensor = torch.tensor([0])
        # sorted_tensor = torch.cat((zero_tensor, sorted_tensor), dim=0)
        # self.c_row_offsetTensor = torch.cumsum(sorted_tensor, dim=0).int()
        # self.c_rowTensor=torch.index_select( self.c_rowTensor, dim=0, index=sorted_indices).int()
        # self.c_atomicTensor=torch.index_select( self.c_atomicTensor, dim=0, index=sorted_indices).int()
        # self.c_colTensor=torch.index_select( self.c_colTensor, dim=0, index=sorted_indices).int()
        # self.c_valueTensor=torch.index_select( self.c_valueTensor, dim=0, index=sorted_indices)
    def init_embedding(self):
        '''
        Generate node embedding for nodes.
        Called from __init__.
        '''
        self.x = torch.randn(self.num_nodes_dst, self.num_features)

        
    def to(self, device):
        self.t_windowNew_offsetTensor_=self.t_windowNew_offsetTensor.to(device)
        self.t_blockNew_offsetTensor_=self.t_blockNew_offsetTensor.to(device)
        self.t_valueTensor_=self.t_valueTensor.to(device)
        self.t_columnTensor_=self.t_columnTensor.to(device)
        self.t_rowTensor_=self.t_rowTensor.to(device)
        self.t_colTensor_=self.t_colTensor.to(device)
        self.t_window_rowTensor_=self.t_window_rowTensor.to(device)

        self.c_part_offsetTensor_=self.c_part_offsetTensor.to(device)
        self.c_row_offsetTensor_=self.c_row_offsetTensor.to(device)
        self.c_rowTensor_=self.c_rowTensor.to(device)
        self.c_atomicTensor_=self.c_atomicTensor.to(device)
        self.c_colTensor_=self.c_colTensor.to(device)
        self.c_valueTensor_=self.c_valueTensor.to(device)
        # self.c_row_offsetTensor_new_ = self.c_row_offsetTensor_new.to(device)
        self.x =  self.x.to(device)
        return self



# cuda
class GCN_dataset_cuda(torch.nn.Module):
    """
    data loading for more graphs
    """
    def __init__(self, data, dimN, density,partsize, data_path):
        super(GCN_dataset_cuda, self).__init__()

        self.graph = np.load('/home/shijinliang/module/mnt/libra_suite/' + data_path + '/' + data +'.npz')
        self.num_features = dimN
        self.init_edges(density, partsize)
        self.init_embedding()

        
        
    def init_edges(self, density, partSize):
        # loading from a .npz graph file
        self.num_nodes_ori =  self.graph['num_nodes_src']-0
        self.num_nodes_dst =  self.graph['num_nodes_dst']-0
        self.num_nodes = self.num_nodes_ori
        if self.num_nodes_ori%8 !=0 :
            self.num_nodes = self.num_nodes_ori + 8 - self.num_nodes_ori%8 
        self.num_edges = self.graph['num_edges']-0
        src_li = self.graph['src_li']
        dst_li = self.graph['dst_li']
        # self.edge_index = self.graph['edge_index_new']
        self.edge_index = np.stack([src_li, dst_li])
        self.avg_degree = self.num_edges / self.num_nodes
        print("Num_nodes, Num_edges: " + str(self.num_nodes) + ' , ' + str(self.num_edges))
        val = [1] * self.num_edges
        scipy_coo = coo_matrix((val, self.edge_index), shape=(self.num_nodes, self.num_nodes_dst))
        adj = scipy_coo.tocsr()
        
        self.column_index = torch.IntTensor(adj.indices)
        self.row_pointers = torch.IntTensor(adj.indptr)
        self.degrees = torch.randn(self.num_edges)
        
        self.c_row_offsetTensor, \
        self.c_rowTensor, \
        self.c_atomicTensor, \
        self.c_colTensor, \
        self.c_valueTensor = Libra5Block.block_csr_cuda(self.row_pointers, self.column_index,self.degrees, partSize)
        
        self.parts = self.c_row_offsetTensor.shape[0]-1
        self.partSize = partSize
       
    def init_embedding(self):
        '''
        Generate node embedding for nodes.
        Called from __init__.
        '''
        self.x = torch.randn(self.num_nodes_dst, self.num_features)

        
    def to(self, device):

        self.c_row_offsetTensor_=self.c_row_offsetTensor.to(device)
        self.c_rowTensor_=self.c_rowTensor.to(device)
        self.c_atomicTensor_=self.c_atomicTensor.to(device)
        self.c_colTensor_=self.c_colTensor.to(device)
        self.c_valueTensor_=self.c_valueTensor.to(device)
        self.x =  self.x.to(device)
        return self



# cuda
class GCN_dataset_cuda(torch.nn.Module):
    """
    data loading for more graphs
    """
    def __init__(self, data, dimN, density,partsize, data_path):
        super(GCN_dataset_cuda, self).__init__()

        self.graph = np.load('/home/shijinliang/module/mnt/libra_suite/' + data_path + '/' + data +'.npz')
        self.num_features = dimN
        self.init_edges(density, partsize)
        self.init_embedding()

        
        
    def init_edges(self, density, partSize):
        # loading from a .npz graph file
        self.num_nodes_ori =  self.graph['num_nodes_src']-0
        self.num_nodes_dst =  self.graph['num_nodes_dst']-0
        self.num_nodes = self.num_nodes_ori
        if self.num_nodes_ori%8 !=0 :
            self.num_nodes = self.num_nodes_ori + 8 - self.num_nodes_ori%8 
        self.num_edges = self.graph['num_edges']-0
        src_li = self.graph['src_li']
        dst_li = self.graph['dst_li']
        # self.edge_index = self.graph['edge_index_new']
        self.edge_index = np.stack([src_li, dst_li])
        self.avg_degree = self.num_edges / self.num_nodes
        # print("Num_nodes, Num_edges: " + str(self.num_nodes) + ' , ' + str(self.num_edges))
        val = [1] * self.num_edges
        scipy_coo = coo_matrix((val, self.edge_index), shape=(self.num_nodes, self.num_nodes_dst))
        adj = scipy_coo.tocsr()
        
        self.column_index = torch.IntTensor(adj.indices)
        self.row_pointers = torch.IntTensor(adj.indptr)
        self.degrees = torch.randn(self.num_edges)
        
        self.c_row_offsetTensor, \
        self.c_rowTensor, \
        self.c_colTensor = Libra5Block.block_csr_sddmm_cuda_v2(self.row_pointers, self.column_index, partSize)
        self.parts = self.c_row_offsetTensor.shape[0]-1
        
    def init_embedding(self):
        '''
        Generate node embedding for nodes.
        Called from __init__.
        '''
        self.x = torch.randn(self.num_nodes_dst, self.num_features)
        self.x = self.x
        
    # def to(self, device):

    #     self.c_row_offsetTensor_=self.c_row_offsetTensor.to(device)
    #     self.c_rowTensor_=self.c_rowTensor.to(device)
    #     self.c_atomicTensor_=self.c_atomicTensor.to(device)
    #     self.c_colTensor_=self.c_colTensor.to(device)
    #     self.c_valueTensor_=self.c_valueTensor.to(device)
        
    #     self.c_row_offsetTensor_short_=self.c_row_offsetTensor_short.to(device)
    #     self.c_rowTensor_short_=self.c_rowTensor_short.to(device)
    #     self.c_atomicTensor_short_=self.c_atomicTensor_short.to(device)
    #     self.c_colTensor_short_=self.c_colTensor_short.to(device)
    #     self.c_valueTensor_short_=self.c_valueTensor_short.to(device)
      
    #     self.x =  self.x.to(device)
    #     return self

# tcu
class GCN_dataset_tcu_v2(torch.nn.Module):
    """
    data loading for more graphs
    """
    def __init__(self, data, dimN, density,partsize, data_path,  window, wide):
        super(GCN_dataset_tcu_v2, self).__init__()

        self.graph = np.load('/home/shijinliang/module/mnt/libra_suite/' + data_path + '/' + data +'.npz')
        self.num_features = dimN
        self.init_edges(density, partsize, window, wide)
        self.init_embedding()

        
        
    def init_edges(self, density, partSize,  window, wide):
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
        print("Num_nodes, Num_edges: " + str(self.num_nodes) + ' , ' + str(self.num_edges))
        val = [1] * self.num_edges
        scipy_coo = coo_matrix((val, self.edge_index), shape=(self.num_nodes, self.num_nodes_dst))
        adj = scipy_coo.tocsr()
        
        self.column_index = torch.IntTensor(adj.indices)
        self.row_pointers = torch.IntTensor(adj.indptr)
        self.degrees = torch.randn(self.num_edges)
        
        self.t_windowNew_offsetTensor, \
        self.t_blockNew_offsetTensor, \
        self.t_columnTensor, \
        self.t_valueTensor, \
        self.t_window_rowTensor, \
        self.t_atomicTensor, \
        self.t_binaryTensor = Libra5Block.block_csr_binary_tcu_part(self.row_pointers, self.column_index,self.degrees, partSize,  window, wide)
        
        self.parts_t = self.t_atomicTensor.shape[0]
        self.t_valueTensor = self.t_valueTensor


    def init_embedding(self):
        '''
        Generate node embedding for nodes.
        Called from __init__.
        '''
        self.x = torch.randn(self.num_nodes_dst, self.num_features)
        self.x = self.x
        
    # def to(self, device):

    #     self.t_windowNew_offsetTensor_=self.t_windowNew_offsetTensor.to(device)
    #     self.t_blockNew_offsetTensor_=self.t_blockNew_offsetTensor.to(device)
    #     self.t_columnTensor_=self.t_columnTensor.to(device)
    #     self.t_valueTensor_=self.t_valueTensor.to(device)
    #     self.t_window_rowTensor_=self.t_window_rowTensor.to(device)
    #     self.t_atomicTensor_=self.t_atomicTensor.to(device)
    #     self.t_binaryTensor_=self.t_binaryTensor.to(device)


    #     self.x =  self.x.to(device)
    #     return self

class GCN_dataset_tcu(torch.nn.Module):
    """
    data loading for more graphs
    """
    def __init__(self, data, dimN, density,partsize, data_path,  window, wide):
        super(GCN_dataset_tcu, self).__init__()

        self.graph = np.load('/home/shijinliang/module/mnt/libra_suite/' + data_path + '/' + data +'.npz')
        self.num_features = dimN
        self.init_edges(density, partsize, window, wide)
        self.init_embedding()

        
        
    def init_edges(self, density, partSize,  window, wide):
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
        print("Num_nodes, Num_edges: " + str(self.num_nodes) + ' , ' + str(self.num_edges))
        val = [1] * self.num_edges
        scipy_coo = coo_matrix((val, self.edge_index), shape=(self.num_nodes, self.num_nodes_dst))
        adj = scipy_coo.tocsr()
        
        self.column_index = torch.IntTensor(adj.indices)
        self.row_pointers = torch.IntTensor(adj.indptr)
        self.degrees = torch.randn(self.num_edges)
        

        self.t_rowNew_offsetTensor, \
        self.t_blockTensor, \
        self.t_tclTensor, \
        self.t_columnTensor, \
        self.t_valueTensor, \
        self.t_window_rowTensor, \
        self.t_atomicTensor = Libra6SpMM.block_tf32_tcu_tcf(self.row_pointers, self.column_index, self.degrees, partSize, window, wide)


        
        self.max = torch.max( self.t_rowNew_offsetTensor[1:]- self.t_rowNew_offsetTensor[:-1])
        self.parts_t = self.t_rowNew_offsetTensor.shape[0] - 1
        self.nnz = self.t_blockTensor[-1]

    def init_embedding(self):
        '''
        Generate node embedding for nodes.
        Called from __init__.
        '''
        self.x = torch.randn(self.num_nodes_dst, self.num_features)
        self.x = self.x
        
    
    
class GCN_dataset_tcu_cuda(torch.nn.Module):
    """
    data loading for more graphs
    """
    def __init__(self, data, dimN, density, partsize_t, partsize_c, shortsize, data_path,  window, wide):
        super(GCN_dataset_tcu_cuda, self).__init__()

        self.graph = np.load('/home/shijinliang/module/mnt/libra_suite/' + data_path + '/' + data +'.npz')
        self.num_features = dimN
        self.init_edges(density,  partsize_t, partsize_c, shortsize, window, wide)
        self.init_embedding()

        
        
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
        
        self.t_rowNew_offsetTensor, \
        self.t_blockTensor, \
        self.t_columnTensor, \
        self.t_window_rowTensor, \
        self.t_atomicTensor, \
        self.t_binaryTensor, \
        self.c_row_offsetTensor, \
        self.c_rowTensor, \
        self.c_colTensor, _ = Libra5Block.block_tf32_tcu_cuda_binary_sddmm(self.row_pointers, self.column_index, partsize_t, partsize_c, density, window, wide)
        
        if self.t_rowNew_offsetTensor.shape[0] == 1:
            self.max  = 0
        else:
            self.max = torch.max( self.t_rowNew_offsetTensor[1:]- self.t_rowNew_offsetTensor[:-1])
        self.parts_t = self.t_rowNew_offsetTensor.shape[0] - 1
        self.nnz = self.t_blockTensor[-1]
        self.parts_c = self.c_rowTensor.shape[0]

    def init_embedding(self):
        '''
        Generate node embedding for nodes.
        Called from __init__.
        '''
        self.x = torch.randn(self.num_nodes_dst, self.num_features)
        self.x = self.x
