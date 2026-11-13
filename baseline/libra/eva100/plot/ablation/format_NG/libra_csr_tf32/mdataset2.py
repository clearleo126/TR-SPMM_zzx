#!/usr/bin/env python3
import torch
import numpy as np
import torch.nn.functional as F
import scipy.sparse as sp
from scipy.sparse import coo_matrix
import Rabbit
import Libra5Block
from scipy.sparse import *
import MagicsphereBlock
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

        self.graph = np.load('./dgl_dataset/' + data_path + '/' + data +'.npz')
        self.num_features = dimN
        self.init_edges(density, partsize,  window, wide, kernel)
        self.init_embedding()

        
        
    def init_edges(self, density, partSize,  window, wide, kernel):
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

        self.graph = np.load('./dgl_dataset/' + data_path + '/' + data +'.npz')
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
        #print("Num_nodes, Num_edges: " + str(self.num_nodes) + ' , ' + str(self.num_edges))
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
class GCN_dataset_cuda_v2(torch.nn.Module):
    """
    data loading for more graphs
    """
    def __init__(self, data, dimN, density,partsize, data_path, shortSize):
        super(GCN_dataset_cuda_v2, self).__init__()

        self.graph = np.load('./dgl_dataset/' + data_path + '/' + data +'.npz')
        self.num_features = dimN
        self.init_edges(density, partsize, shortSize)
        self.init_embedding()

        
        
    def init_edges(self, density, partSize, shortSize):
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
        # #print("Num_nodes, Num_edges: " + str(self.num_nodes) + ' , ' + str(self.num_edges))
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
        self.c_valueTensor, \
        self.c_row_offsetTensor_short, \
        self.c_rowTensor_short, \
        self.c_atomicTensor_short, \
        self.c_colTensor_short, \
        self.c_valueTensor_short = Libra5Block.block_csr_cuda_v2(self.row_pointers, self.column_index,self.degrees, partSize, shortSize)
        self.parts = self.c_row_offsetTensor.shape[0]-1
        self.parts_short = self.c_row_offsetTensor_short.shape[0]-1
        # print("large and short parts: " + str(self.parts) + ', ' + str(self.parts_short) )
        # temp = self.c_row_offsetTensor[1:] - self.c_row_offsetTensor[:-1]
        # # 对张量进行降序排列，并获取排序后的索引
        # _, sorted_indices = torch.sort(temp, dim=0, descending=True)
        # self.sorted_indices = sorted_indices.int()
        # self.parts = self.c_part_offsetTensor[-1].item()
        self.partSize = partSize
    def init_embedding(self):
        '''
        Generate node embedding for nodes.
        Called from __init__.
        '''
        self.x = torch.randn(self.num_nodes_dst, self.num_features)

        
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

        self.graph = np.load('./dgl_dataset/' + data_path + '/' + data +'.npz')
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
        #print("Num_nodes, Num_edges: " + str(self.num_nodes) + ' , ' + str(self.num_edges))
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
        self.t_binaryTensor = self.t_binaryTensor.int()
        self.parts_t = self.t_atomicTensor.shape[0]
        # self.boundary =  self.t_window_rowTensor.shape[0]

    def init_embedding(self):
        '''
        Generate node embedding for nodes.
        Called from __init__.
        '''
        self.x = torch.randn(self.num_nodes_dst, self.num_features)

        
    # def to(self, device):

    #     self.t_windowNew_offsetTensor_=self.t_windowNew_offsetTensor.to(device)
    #     self.t_blockNew_offsetTensor_=self.t_blockNew_offsetTensor.to(device)
    #     self.t_columnTensor_=self.t_columnTensor.to(device)
    #     self.t_valueTensor_=self.t_valueTensor.to(device)
    #     self.t_window_rowTensor_=self.t_window_rowTensor.to(device)
    #     self.t_atomicTensor_=self.t_atomicTensor.to(device)
    #     self.t_binaryTensor_=self.t_binaryTensor.int().to(device)


    #     self.x =  self.x.to(device)
    #     return self

class GCN_dataset_tcu(torch.nn.Module):
    """
    data loading for more graphs
    """
    def __init__(self, data, dimN, density,partsize, data_path,  window, wide):
        super(GCN_dataset_tcu, self).__init__()

        self.graph = np.load('/home/shijinliang/module/mnt/libra_suite/sp_matrix/' + data +'.npz')
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
        #print("Num_nodes, Num_edges: " + str(self.num_nodes) + ' , ' + str(self.num_edges))
        val = [1] * self.num_edges
        scipy_coo = coo_matrix((val, self.edge_index), shape=(self.num_nodes, self.num_nodes_dst))
        adj = scipy_coo.tocsr()
        
        self.column_index = torch.IntTensor(adj.indices)
        self.row_pointers = torch.IntTensor(adj.indptr)
        self.degrees = torch.randn(self.num_edges)
        

        self.t_rowNew_offsetTensor, \
        self.t_blockNew_offsetTensor, \
        self.t_columnTensor, \
        self.t_valueTensor, \
        self.t_rowTensor = Libra5Block.block_csr_tcu(self.row_pointers, self.column_index,self.degrees, partSize,  window, wide)

        memory_byte = self.t_rowNew_offsetTensor.size(0) + self.t_blockNew_offsetTensor.size(0) + self.t_columnTensor.size(0) + self.t_valueTensor.size(0) +  self.t_rowTensor.size(0)
        memory_gb = ((memory_byte * 4)) / (1024 ** 3)
        print("BCRS : " + str(memory_gb) + " GB")
        # self.boundary =  self.t_window_rowTensor.shape[0]

    def init_embedding(self):
        '''
        Generate node embedding for nodes.
        Called from __init__.
        '''
        self.x = torch.randn(self.num_nodes_dst, self.num_features)



class GCN_dataset_tcu_part(torch.nn.Module):
    """
    data loading for more graphs
    """
    def __init__(self, data, dimN, density,partsize, data_path,  window, wide):
        super(GCN_dataset_tcu_part, self).__init__()

        self.graph = np.load('/home/shijinliang/module/mnt/libra_suite/sp_matrix/' + data +'.npz')
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
        #print("Num_nodes, Num_edges: " + str(self.num_nodes) + ' , ' + str(self.num_edges))
        val = [1] * self.num_edges
        scipy_coo = coo_matrix((val, self.edge_index), shape=(self.num_nodes, self.num_nodes_dst))
        adj = scipy_coo.tocsr()
        
        self.column_index = torch.IntTensor(adj.indices)
        self.row_pointers = torch.IntTensor(adj.indptr)
        self.degrees = torch.randn(self.num_edges)
        

        self.t_rowNew_offsetTensor, \
        self.t_blockNew_offsetTensor, \
        self.t_columnTensor, \
        self.t_valueTensor, \
        self.t_window_rowTensor, \
        self.t_atomicTensor, \
        self.t_rowTensor = Libra5Block.block_csr_tcu_part(self.row_pointers, self.column_index,self.degrees, partSize,  window, wide)
        self.parts_t = self.t_atomicTensor.shape[0]

        # self.boundary =  self.t_window_rowTensor.shape[0]

    def init_embedding(self):
        '''
        Generate node embedding for nodes.
        Called from __init__.
        '''
        self.x = torch.randn(self.num_nodes_dst, self.num_features)


#binary
class GCN_dataset_tcu_binary(torch.nn.Module):
    """
    data loading for more graphs
    """
    def __init__(self, data, dimN, density,partsize, data_path,  window, wide):
        super(GCN_dataset_tcu_binary, self).__init__()

        self.graph = np.load('/home/shijinliang/module/mnt/libra_suite/sp_matrix/' + data +'.npz')
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
        #print("Num_nodes, Num_edges: " + str(self.num_nodes) + ' , ' + str(self.num_edges))
        val = [1] * self.num_edges
        scipy_coo = coo_matrix((val, self.edge_index), shape=(self.num_nodes, self.num_nodes_dst))
        adj = scipy_coo.tocsr()
        
        self.column_index = torch.IntTensor(adj.indices)
        self.row_pointers = torch.IntTensor(adj.indptr)
        self.degrees = torch.randn(self.num_edges)
        

        self.t_rowNew_offsetTensor, \
        self.t_blockNew_offsetTensor, \
        self.t_columnTensor, \
        self.t_valueTensor, \
        self.t_binaryTensor  = Libra5Block.block_csr_binary_tcu(self.row_pointers, self.column_index,self.degrees, partSize,  window, wide)

        self.t_binaryTensor = self.t_binaryTensor.int()
        
        memory_byte = self.t_rowNew_offsetTensor.size(0) + self.t_blockNew_offsetTensor.size(0) + self.t_columnTensor.size(0) + self.t_valueTensor.size(0) + self.t_binaryTensor.size(0)
        memory_gb = (memory_byte * 4) / (1024 ** 3)
        print("BCRS : " + str(memory_gb) + " GB")
        # self.boundary =  self.t_window_rowTensor.shape[0]

    def init_embedding(self):
        '''
        Generate node embedding for nodes.
        Called from __init__.
        '''
        self.x = torch.randn(self.num_nodes_dst, self.num_features)

        
# tcu
class GCN_dataset_tcu_m(torch.nn.Module):
    """
    data loading for more graphs
    """
    def __init__(self, data, dimN, density,partsize, data_path,  window, wide):
        super(GCN_dataset_tcu_m, self).__init__()

        self.graph = np.load('./dgl_dataset/' + data_path + '/' + data +'.npz')
        self.num_features = dimN
        self.init_edges(density, partsize, window, wide)
        self.init_embedding()

        
        
    def init_edges(self, density, partSize,  window, wide):
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
        #print("Num_nodes, Num_edges: " + str(self.num_nodes) + ' , ' + str(self.num_edges))
        val = [1] * self.num_edges
        scipy_coo = coo_matrix((val, self.edge_index), shape=(self.num_nodes, self.num_nodes_dst))
        adj = scipy_coo.tocsr()
        
        self.column_index = torch.IntTensor(adj.indices)
        self.row_pointers = torch.IntTensor(adj.indptr)
        self.degrees = torch.randn(self.num_edges)
        

        # self.t_rowTensor, \
        # self.t_colTensor, \
        # self.t_valueTensor  = Libra5Block.block_tf32_tcu(self.row_pointers, self.column_index,self.degrees, window, wide)
        self.t_rowTensor, \
        self.t_colTensor, \
        self.t_valueTensor  = Libra5Block.block_m_tcu(self.row_pointers, self.column_index,self.degrees, window, wide)

        # tensor = self.t_rowTensor[1:] - self.t_rowTensor[:-1]
        # # 获取唯一元素和它们在张量中的索引
        # unique_elements, inverse_indices = torch.unique(tensor, return_inverse=True)

        # # 计算每个唯一元素在张量中出现的次数
        # counts = torch.bincount(inverse_indices)

        # # 将唯一元素和它们的计数打印出来
        # for element, count in zip(unique_elements, counts):
        #     print(f"元素 {element.item()} 出现了 {count.item()} 次")
    def init_embedding(self):
        '''
        Generate node embedding for nodes.
        Called from __init__.
        '''
        self.x = torch.randn(self.num_nodes_dst, self.num_features)

        
    def to(self, device):

        self.t_rowTensor_=self.t_rowTensor.to(device)
        self.t_colTensor_=self.t_colTensor.to(device)
        self.t_valueTensor_=self.t_valueTensor.to(device)

        self.x =  self.x.to(device)
        return self


# tcu_m_part
class GCN_dataset_tcu_m_part(torch.nn.Module):
    """
    data loading for more graphs
    """
    def __init__(self, data, dimN, density,partsize, data_path,  window, wide):
        super(GCN_dataset_tcu_m_part, self).__init__()

        self.graph = np.load('./dgl_dataset/' + data_path + '/' + data +'.npz')
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
        #print("Num_nodes, Num_edges: " + str(self.num_nodes) + ' , ' + str(self.num_edges))
        val = [1] * self.num_edges
        scipy_coo = coo_matrix((val, self.edge_index), shape=(self.num_nodes, self.num_nodes_dst))
        adj = scipy_coo.tocsr()
        
        self.column_index = torch.IntTensor(adj.indices)
        self.row_pointers = torch.IntTensor(adj.indptr)
        self.degrees = torch.randn(self.num_edges)
        

        # self.t_rowTensor, \
        # self.t_colTensor, \
        # self.t_valueTensor  = Libra5Block.block_tf32_tcu(self.row_pointers, self.column_index,self.degrees, window, wide)
        self.t_rowTensor, \
        self.t_colTensor, \
        self.t_valueTensor, \
        self.t_window_rowTensor, \
        self.t_atomicTensor = Libra5Block.block_m_tcu_part(self.row_pointers, self.column_index,self.degrees, partSize, window, wide)
        
        # tensor = self.t_rowTensor[1:] - self.t_rowTensor[:-1]
        # # 获取唯一元素和它们在张量中的索引
        # unique_elements, inverse_indices = torch.unique(tensor, return_inverse=True)

        # # 计算每个唯一元素在张量中出现的次数
        # counts = torch.bincount(inverse_indices)

        # # 将唯一元素和它们的计数打印出来
        # for element, count in zip(unique_elements, counts):
        #     print(f"元素 {element.item()} 出现了 {count.item()} 次")
    def init_embedding(self):
        '''
        Generate node embedding for nodes.
        Called from __init__.
        '''
        self.x = torch.randn(self.num_nodes_dst, self.num_features)

        
    def to(self, device):

        self.t_rowTensor_=self.t_rowTensor.to(device)
        self.t_colTensor_=self.t_colTensor.to(device)
        self.t_valueTensor_=self.t_valueTensor.to(device)
        self.t_window_rowTensor_=self.t_window_rowTensor.to(device)
        self.t_atomicTensor_=self.t_atomicTensor.to(device)
               
        self.x =  self.x.to(device)
        return self
# v2
class GCN_dataset_v2(torch.nn.Module):
    """
    data loading for more graphs
    """
    def __init__(self, data, dimN, density,partsize, data_path,  window, wide, kernel):
        super(GCN_dataset_v2, self).__init__()

        self.graph = np.load('./dgl_dataset/' + data_path + '/' + data +'.npz')
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
        #print("Num_nodes, Num_edges: " + str(self.num_nodes) + ' , ' + str(self.num_edges))
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
        self.c_valueTensor = Libra5Block.block_csr_v2(self.row_pointers, self.column_index,self.degrees, density, partSize,  window, wide, kernel)

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
        print("boundary, parts: " + str(self.boundary) + ", " + str(self.parts/4))
        # print("automic and not automic parts: " + str(torch.sum(self.c_atomicTensor == 1).item()) + ", " + str(torch.sum(self.c_atomicTensor == 0).item()))
        print("tcu-nnz, cuda-nnz: " + str(self.num_edges-self.c_row_offsetTensor[-1].item()) + ", " + str(self.c_row_offsetTensor[-1].item()))

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
        self.x =  self.x.to(device)
        return self
    
    
# tcu_m_part
class GCN_dataset_mtcu_cuda(torch.nn.Module):
    """
    data loading for more graphs
    """
    def __init__(self, data, dimN, density, partsize_t, partsize_c, shortsize, data_path,  window, wide):
        super(GCN_dataset_mtcu_cuda, self).__init__()

        self.graph = np.load('./dgl_dataset/' + data_path + '/' + data +'.npz')
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
        
        self.t_rowTensor, \
        self.t_colTensor, \
        self.t_valueTensor, \
        self.t_window_rowTensor, \
        self.t_atomicTensor, \
        self.c_row_offsetTensor, \
        self.c_rowTensor, \
        self.c_atomicTensor, \
        self.c_colTensor, \
        self.c_valueTensor, \
        self.c_row_offsetTensor_short, \
        self.c_rowTensor_short, \
        self.c_atomicTensor_short, \
        self.c_colTensor_short, \
        self.c_valueTensor_short = Libra5Block.block_tf32_mtcu_cuda_v3(self.row_pointers, self.column_index, self.degrees, partsize_t, partsize_c, shortsize, density, window, wide)
        
        self.parts_t = self.t_atomicTensor.shape[0]
        self.parts_c = self.c_atomicTensor.shape[0]
        self.parts_c_short = self.c_atomicTensor_short.shape[0]
        self.partsize_c = partsize_c
    def init_embedding(self):
        '''
        Generate node embedding for nodes.
        Called from __init__.
        '''
        self.x = torch.randn(self.num_nodes_dst, self.num_features)

        
    def to(self, device):

        self.t_rowTensor_=self.t_rowTensor.to(device)
        self.t_colTensor_=self.t_colTensor.to(device)
        self.t_valueTensor_=self.t_valueTensor.to(device)
        self.t_window_rowTensor_=self.t_window_rowTensor.to(device)
        self.t_atomicTensor_=self.t_atomicTensor.to(device)

        self.c_row_offsetTensor_=self.c_row_offsetTensor.to(device)
        self.c_rowTensor_=self.c_rowTensor.to(device)
        self.c_atomicTensor_=self.c_atomicTensor.to(device)
        self.c_colTensor_=self.c_colTensor.to(device)
        self.c_valueTensor_=self.c_valueTensor.to(device)
        
        self.c_row_offsetTensor_short_=self.c_row_offsetTensor_short.to(device)
        self.c_rowTensor_short_=self.c_rowTensor_short.to(device)
        self.c_atomicTensor_short_=self.c_atomicTensor_short.to(device)
        self.c_colTensor_short_=self.c_colTensor_short.to(device)
        self.c_valueTensor_short_=self.c_valueTensor_short.to(device)
               
        self.x =  self.x.to(device)
        return self
    
    
#tcu cuda
# tcu_m_part
class GCN_dataset_tcu_cuda(torch.nn.Module):
    """
    data loading for more graphs
    """
    def __init__(self, data, dimN, density, partsize_t, partsize_c, shortsize, data_path,  window, wide):
        super(GCN_dataset_tcu_cuda, self).__init__()

        self.graph = np.load('./dgl_dataset/' + data_path + '/' + data +'.npz')
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
        ##print("Num_nodes, Num_edges: " + str(self.num_nodes) + ' , ' + str(self.num_edges))
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
        self.t_rowTensor, \
        self.t_colTensor, \
        self.c_row_offsetTensor, \
        self.c_rowTensor, \
        self.c_atomicTensor, \
        self.c_colTensor, \
        self.c_valueTensor, \
        self.c_row_offsetTensor_short, \
        self.c_rowTensor_short, \
        self.c_atomicTensor_short, \
        self.c_colTensor_short, \
        self.c_valueTensor_short = Libra5Block.block_tf32_tcu_cuda_v4(self.row_pointers, self.column_index, self.degrees, partsize_t, partsize_c, shortsize, density, window, wide)
        
        self.parts_t = self.t_atomicTensor.shape[0]
        self.parts_c = self.c_atomicTensor.shape[0]
        self.parts_c_short = self.c_atomicTensor_short.shape[0]
        self.partsize_c = partsize_c
    def init_embedding(self):
        '''
        Generate node embedding for nodes.
        Called from __init__.
        '''
        self.x = torch.randn(self.num_nodes_dst, self.num_features)

        
    def to(self, device):

        self.t_rowNew_offsetTensor_=self.t_rowNew_offsetTensor.to(device)
        self.t_blockTensor_=self.t_blockTensor.to(device)
        self.t_columnTensor_=self.t_columnTensor.to(device)
        self.t_valueTensor_=self.t_valueTensor.to(device)
        self.t_window_rowTensor_=self.t_window_rowTensor.to(device)
        self.t_atomicTensor_=self.t_atomicTensor.to(device)
        self.t_rowTensor_=self.t_rowTensor.to(device)
        self.t_colTensor_=self.t_colTensor.to(device)
                
        self.c_row_offsetTensor_=self.c_row_offsetTensor.to(device)
        self.c_rowTensor_=self.c_rowTensor.to(device)
        self.c_atomicTensor_=self.c_atomicTensor.to(device)
        self.c_colTensor_=self.c_colTensor.to(device)
        self.c_valueTensor_=self.c_valueTensor.to(device)
        
        self.c_row_offsetTensor_short_=self.c_row_offsetTensor_short.to(device)
        self.c_rowTensor_short_=self.c_rowTensor_short.to(device)
        self.c_atomicTensor_short_=self.c_atomicTensor_short.to(device)
        self.c_colTensor_short_=self.c_colTensor_short.to(device)
        self.c_valueTensor_short_=self.c_valueTensor_short.to(device)
               
        self.x =  self.x.to(device)
        return self
    
    
class GCN_dataset_tcu_cuda_tf32(torch.nn.Module):
    """
    data loading for more graphs
    """
    def __init__(self, data, dimN, density, partsize_t, partsize_c, shortsize, data_path,  window, wide):
        super(GCN_dataset_tcu_cuda_tf32, self).__init__()

        self.graph = np.load('./dgl_dataset/' + data_path + '/' + data +'.npz')
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
        ##print("Num_nodes, Num_edges: " + str(self.num_nodes) + ' , ' + str(self.num_edges))
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
        self.c_valueTensor_short = Libra5Block.block_tf32_tcu_cuda_binary(self.row_pointers, self.column_index, self.degrees, partsize_t, partsize_c, shortsize, density, window, wide)
        
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
        self.t_binaryTensor = self.t_binaryTensor.int()
    def init_embedding(self):
        '''
        Generate node embedding for nodes.
        Called from __init__.
        '''
        self.x = torch.randn(self.num_nodes_dst, self.num_features)

        
    # def to(self, device):

    #     self.t_rowNew_offsetTensor_=self.t_rowNew_offsetTensor.to(device)
    #     self.t_blockTensor_=self.t_blockTensor.to(device)
    #     self.t_columnTensor_=self.t_columnTensor.to(device)
    #     self.t_valueTensor_=self.t_valueTensor.to(device)
    #     self.t_window_rowTensor_=self.t_window_rowTensor.to(device)
    #     self.t_atomicTensor_=self.t_atomicTensor.to(device)
    #     self.t_binaryTensor_=self.t_binaryTensor.int().to(device)
                
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
    
    
    
    
#SGT-TCGNN
class GCN_dataset_tcu_sgt(torch.nn.Module):
    """
    data loading for more graphs
    """
    def __init__(self, data, dimN, density,partsize, data_path,  window, wide):
        super(GCN_dataset_tcu_sgt, self).__init__()

        self.graph = np.load('/home/shijinliang/module/mnt/libra_suite/sp_matrix/' + data +'.npz')
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
        #print("Num_nodes, Num_edges: " + str(self.num_nodes) + ' , ' + str(self.num_edges))
        val = [1] * self.num_edges
        scipy_coo = coo_matrix((val, self.edge_index), shape=(self.num_nodes, self.num_nodes_dst))
        adj = scipy_coo.tocsr()
        
        self.column_index = torch.IntTensor(adj.indices)
        self.row_pointers = torch.IntTensor(adj.indptr)
        self.degrees = torch.randn(self.num_edges)
        

        self.t_blockNew_offsetTensor, \
        self.t_valueTensor, \
        self.t_rowTensor,\
        self.t_colTensor = Libra5Block.block_sgt_tcu(self.row_pointers, self.column_index,self.degrees, partSize,  window, wide)
        
        memory_byte = self.t_blockNew_offsetTensor.size(0) + self.t_colTensor.size(0) + self.t_valueTensor.size(0) + self.t_rowTensor.size(0)
        memory_gb = (memory_byte * 4) / (1024 ** 3)
        print("BCRS : " + str(memory_gb) + " GB")
        # self.boundary =  self.t_window_rowTensor.shape[0]

    def init_embedding(self):
        '''
        Generate node embedding for nodes.
        Called from __init__.
        '''
        self.x = torch.randn(self.num_nodes_dst, self.num_features)
        
        
        
#BCRS
class GCN_dataset_tcu_bcrs(torch.nn.Module):
    """
    data loading for more graphs
    """
    def __init__(self, data, dimN, density,partsize, data_path,  window, wide):
        super(GCN_dataset_tcu_bcrs, self).__init__()

        self.graph = np.load('/home/shijinliang/module/mnt/libra_suite/sp_matrix/' + data +'.npz')
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
        #print("Num_nodes, Num_edges: " + str(self.num_nodes) + ' , ' + str(self.num_edges))
        val = [1] * self.num_edges
        scipy_coo = coo_matrix((val, self.edge_index), shape=(self.num_nodes, self.num_nodes_dst))
        adj = scipy_coo.tocsr()
        
        self.column_index = torch.IntTensor(adj.indices)
        self.row_pointers = torch.IntTensor(adj.indptr)
        # self.degrees = torch.randn(self.num_edges)
        

        self.t_blockNew_offsetTensor, \
        self.t_colTensor, \
        self.t_valueTensor = Libra5Block.block_bcrs_tcu(self.row_pointers, self.column_index, partSize,  window, wide)

        # self.boundary =  self.t_window_rowTensor.shape[0]
        memory_byte = self.t_blockNew_offsetTensor.size(0) + self.t_colTensor.size(0) + self.t_valueTensor.size(0)
        memory_gb = (memory_byte * 4) / (1024 ** 3)
        print("BCRS : " + str(memory_gb) + " GB")

    def init_embedding(self):
        '''
        Generate node embedding for nodes.
        Called from __init__.
        '''
        self.x = torch.randn(self.num_nodes_dst, self.num_features)