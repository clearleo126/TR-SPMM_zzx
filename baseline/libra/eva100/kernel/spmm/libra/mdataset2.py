#!/usr/bin/env python3
import torch
import numpy as np
import torch.nn.functional as F
import scipy.sparse as sp
from scipy.sparse import coo_matrix
import Rabbit
import Libra5Block
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
    def __init__(self, data, dimN, density,partsize):
        super(GCN_dataset, self).__init__()

        self.graph = np.load('./dgl_dataset/sp_matrix/' + data +'.npz')
        self.num_features = dimN
        self.init_edges(density, partsize)
        self.init_embedding()

        
        
    def init_edges(self, density, partSize):
        # loading from a .npz graph file
        self.num_nodes_ori =  self.graph['num_nodes_src']-0
        self.num_nodes_dst =  self.graph['num_nodes_dst']-0
        self.num_nodes = self.num_nodes_ori
        if self.num_nodes_ori%16 !=0 :
            self.num_nodes = self.num_nodes_ori + 16 - self.num_nodes_ori%16 
        self.num_edges = self.graph['num_edges']-0
        src_li = self.graph['src_li']
        dst_li = self.graph['dst_li']
        # self.edge_index = self.graph['edge_index_new']
        self.edge_index = np.stack([src_li, dst_li])
        self.avg_degree = self.num_edges / self.num_nodes
        print("Num_nodes: " + str(self.num_nodes))
        val = [1] * self.num_edges
        scipy_coo = coo_matrix((val, self.edge_index), shape=(self.num_nodes, self.num_nodes))
        adj = scipy_coo.tocsr()
        
        self.column_index = torch.IntTensor(adj.indices)
        self.row_pointers = torch.IntTensor(adj.indptr)
        self.degrees = torch.randn(self.num_edges)
        
        # self.t_row_offsetTensor, self.t_colTensor, self.t_valueTensor,  self.t_rowTensor, self.c_partTensor, self.c_row_offsetTensor, self.c_rowTensor, self.c_atomicTensor, self.c_colTensor, self.c_valueTensor= Libra5Block.block_16_4_balance(self.row_pointers, self.column_index,self.degrees, density, partSize)
        self.t_row_offsetTensor, self.t_colTensor, self.t_valueTensor,  self.t_rowTensor, self.c_partTensor, self.c_row_offsetTensor, self.c_rowTensor, self.c_atomicTensor, self.c_colTensor, self.c_valueTensor= Libra5Block.block_16_4_split(self.row_pointers, self.column_index,self.degrees, density, partSize)


        self.boundary =  self.t_rowTensor.shape[0]
        self.parts = self.c_partTensor[-1].item()
        print("boundary, parts: " + str(self.boundary) + ", " + str(self.parts/4))
        print("tcu-nnz, cuda-nnz: " + str(self.num_edges-self.c_row_offsetTensor[-1].item()) + ", " + str(self.c_row_offsetTensor[-1].item()))
        print("vectors: " + str(  self.t_row_offsetTensor[-1].item()))
    def init_embedding(self):
        '''
        Generate node embedding for nodes.
        Called from __init__.
        '''
        self.x = torch.randn(self.num_nodes_dst, self.num_features)

        
    def to(self, device):
        self.t_row_offsetTensor_=self.t_row_offsetTensor.to(device)
        self.t_colTensor_=self.t_colTensor.to(device)
        self.t_valueTensor_=self.t_valueTensor.to(device)
        self.t_rowTensor_=self.t_rowTensor.to(device)
        self.c_row_offsetTensor_=self.c_row_offsetTensor.to(device)
        self.c_colTensor_=self.c_colTensor.to(device)
        self.c_valueTensor_=self.c_valueTensor.to(device)

        self.c_partTensor_=self.c_partTensor.to(device)
        self.c_rowTensor_=self.c_rowTensor.to(device)
        self.c_atomicTensor_=self.c_atomicTensor.to(device)
        self.x =  self.x.to(device)
        return self
