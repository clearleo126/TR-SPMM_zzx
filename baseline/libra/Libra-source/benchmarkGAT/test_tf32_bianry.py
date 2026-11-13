import numpy
import torch
import Libra5Block
import Libra5BenchmarkGAT
from scipy.sparse import *


def check(row_pointers1, column_index1, dd, rhs, n) :
    row_pointers1 = row_pointers1[:n+1]
    dd = dd.numpy()
    value = []
    for i in range(len(row_pointers1) - 1):
        for j in range(row_pointers1[i], row_pointers1[i+1]):
            value.append(dd[i]*dd[column_index1[j]])
    # n = row_pointers1.size(0)-1
    sparse_matrix = csr_matrix((value, column_index1.numpy(), row_pointers1.numpy()), shape=(n, n))
    result = sparse_matrix.dot(rhs.numpy())
    return result

row = torch.tensor([0, 6, 7,  8, 10, 10, 11, 13, 15, 18, 19, 19, 19, 20, 20, 20, 20, 24,
        25, 26, 28, 28, 29, 31, 33, 36, 37, 37, 37, 38, 38, 38, 38],dtype=torch.int32)
col = torch.tensor([1,2,3,11,16,21, 1,1,2,24,25,16,22,0,27,0,4,25,9,27,1,3,11,21,1,1,2,24,25,16,22,0,27,0,4,25,9,27],dtype=torch.int32)

t_rowNew_offsetTensor, \
t_blockTensor, \
t_columnTensor, \
t_window_rowTensor, \
t_atomicTensor, \
t_binaryTensor = Libra5Block.block_csr_binary_sddmm_tcu_part_c(row,col, 2, 8, 16)

print(t_rowNew_offsetTensor)
print(t_blockTensor)
print(t_columnTensor)
print(t_window_rowTensor)
print(t_atomicTensor)
print(t_binaryTensor)
print()
maxPart = torch.max(t_rowNew_offsetTensor[1:]-t_rowNew_offsetTensor[:-1])
parts_t = t_rowNew_offsetTensor.shape[0] - 1
rows = 30
dimN = 128
rhs = torch.ones((rows, dimN), dtype=torch.float32)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# t_rowNew_offsetTensor_=t_rowNew_offsetTensor.to(device)
# t_blockTensor_=t_blockTensor.to(device)
# t_columnTensor_=t_columnTensor.to(device)
# t_window_rowTensor_=t_window_rowTensor.to(device)
# t_atomicTensor_=t_atomicTensor.to(device)
# t_binaryTensor_=t_binaryTensor.to(device)
        

# rhs_ = rhs.to(device)

result, spmm_ms_avg = Libra5BenchmarkGAT.forward_tf32_tcu_binary_part(
t_rowNew_offsetTensor, 
t_blockTensor,
t_columnTensor, 
t_window_rowTensor,
t_binaryTensor,

rhs, 
t_blockTensor[-1], 
maxPart, 
parts_t,
rhs.size(1), 
rows, 
1)

print(result)
print()
        
