import numpy
import torch
import Libra5Block
import Libra5BenchmarkGCN
from scipy.sparse import *
import Libra6SpMM

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

c_row_offsetTensor, \
c_rowTensor, \
c_atomicTensor, \
c_colTensor, \
c_valueTensor, \
c_row_offsetTensor_short, \
c_rowTensor_short, \
c_atomicTensor_short, \
c_colTensor_short, \
c_valueTensor_short= Libra5Block.block_csr_cuda_v2(row,col,dd, 4, 2)



print(c_row_offsetTensor)
print(c_rowTensor)
print(c_atomicTensor)
print(c_colTensor)
print(c_valueTensor)
print()

print(c_row_offsetTensor_short)
print(c_rowTensor_short)
print(c_atomicTensor_short)
print(c_colTensor_short)
print(c_valueTensor_short)
        

parts_c = c_atomicTensor.shape[0]
parts_c_short = c_atomicTensor_short.shape[0]
partsize_c = 4
rows = 32
dimN = 128
rhs = torch.ones((30, dimN), dtype=torch.float16)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


result, spmm_ms_avg = Libra6SpMM.forward_fp16_cuda_v2(

c_row_offsetTensor,
c_rowTensor, 
c_atomicTensor,
c_colTensor,
c_valueTensor.half(), 

c_row_offsetTensor_short,
c_rowTensor_short, 
c_atomicTensor_short,
c_colTensor_short,
c_valueTensor_short.half(), 

rhs, 

partsize_c,
rhs.size(1), 
rows, 
rows, 
1,
parts_c, 
parts_c_short, 
False)

print(dd)
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
    