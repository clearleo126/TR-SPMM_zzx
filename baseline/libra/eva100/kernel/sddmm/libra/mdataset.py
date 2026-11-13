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
    def __init__(self, data, dimN, density):
        super(GCN_dataset, self).__init__()

        self.graph = np.load('./dgl_dataset/best/' + data +'.npz')
        self.num_features = dimN
        self.init_edges(density)
        self.init_embedding()

        
        
    def init_edges(self, density):
        # loading from a .npz graph file
        self.num_nodes = self.graph['num_nodes']-0
        self.num_nodes_ori = self.graph['num_nodes_ori']-0
        self.num_edges = self.graph['num_edges']-0
        self.edge_index = self.graph['edge_index_new']
        self.avg_degree = self.num_edges / self.num_nodes
        
        val = [1] * self.num_edges
        scipy_coo = coo_matrix((val, self.edge_index), shape=(self.num_nodes, self.num_nodes))
        adj = scipy_coo.tocsr()
        
        self.column_index = torch.IntTensor(adj.indices)
        self.row_pointers = torch.IntTensor(adj.indptr)
        self.degrees = torch.randn(self.num_edges)
        
        self.t_row_offsetTensor, self.t_colTensor, self.t_valueTensor, self.c_row_offsetTensor, self.c_colTensor, self.c_valueTensor= Libra5Block.block_8_4_split(self.row_pointers, self.column_index,self.degrees, density)
        # profile
        print("Density: " + str(density))
        temp = self.t_row_offsetTensor[1:] - self.t_row_offsetTensor[:-1]
        t_rows = (torch.nonzero(temp).size(0))*8
        temp = self.c_row_offsetTensor[1:] - self.c_row_offsetTensor[:-1]
        c_rows = (torch.nonzero(temp).size(0))
        print("tcu-rows, cuda-rows: " + str(t_rows) + ", " + str(c_rows))
        print("tcu-nnz, cuda-nnz: " + str(self.num_edges-self.c_row_offsetTensor[-1].item()) + ", " + str(self.c_row_offsetTensor[-1].item()))
    def init_embedding(self):
        '''
        Generate node embedding for nodes.
        Called from __init__.
        '''
        self.x = torch.randn(self.num_nodes, self.num_features)

        
    def to(self, device):
        self.t_rowTensor_=self.t_row_offsetTensor.to(device)
        self.t_colTensor_=self.t_colTensor.to(device)
        self.t_valueTensor_=self.t_valueTensor.to(device)
        self.c_rowTensor_=self.c_row_offsetTensor.to(device)
        self.c_colTensor_=self.c_colTensor.to(device)
        self.c_valueTensor_=self.c_valueTensor.to(device)
        self.x =  self.x.to(device)
        return self
