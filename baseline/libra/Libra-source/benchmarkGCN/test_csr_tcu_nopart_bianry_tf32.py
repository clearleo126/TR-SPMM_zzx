import numpy
import torch
import Libra5Block
import Libra5BenchmarkGCN
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

# dd=(row[1:] - row[:-1]).to(torch.float32)
# dd = dd
dd =  torch.ones_like(col).float()
t_rowNew_offsetTensor, \
t_blockTensor,\
t_columnTensor, \
t_valueTensor, \
t_binaryTensor  = Libra5Block.block_csr_binary_tcu(row,col,dd, 2, 8, 4)

print(t_rowNew_offsetTensor)
print(t_blockTensor)
print(t_valueTensor)
print(t_columnTensor)
print(t_binaryTensor)
print()

t_binaryTensor = t_binaryTensor.int()
# print(t_binaryTensor)
partsize_c = 4
rows = 32
dimN = 128
rhs = torch.ones((30, dimN), dtype=torch.float32)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# t_rowNew_offsetTensor_=t_rowNew_offsetTensor.to(device)
# t_blockTensor_=t_blockTensor.to(device)
# t_columnTensor_=t_columnTensor.to(device)
# t_valueTensor_=t_valueTensor.to(device)
# # t_valueTensor_=t_valueTensor.half().to(device)
# t_binaryTensor_=t_binaryTensor.to(device)
# # t_colTensor_=t_colTensor.to(device)


# rhs_ = rhs.to(device)
# rhs_ = rhs.half().to(device)

result, spmm_ms_avg = Libra5BenchmarkGCN.forward_tf32_tcu_binary(
t_rowNew_offsetTensor, 
t_blockTensor,
t_columnTensor, 
t_valueTensor, 
t_binaryTensor,
# t_rowTensor,
# t_colTensor,

rhs, 

32,
rhs.size(1), 
30, 
1)
# print(dd)
res = check(row,col,dd,rhs,30)
print(result)
print(res)


for i in range(30):
    if (result[i][0] - res[i][0]) != 0 :
            print("No")
            exit(0)
    if (result[i][dimN-1] - res[i][dimN-1]) != 0 :
            print("No")
            exit(0)
        
print("PASS")
    