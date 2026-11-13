import numpy
import torch
import Libra5Block
import Libra5BenchmarkGCN
from scipy.sparse import *
import numpy as np

def check(row_pointers1, column_index1, dd, rhs, m, n):
    # 将 row_pointers1 列表截取为前 n+1 个元素
    row_pointers1 = row_pointers1[:n+1]

    # 将 dd 转换为 NumPy 数组
    dd = dd.numpy()

    # 创建一个空列表用于存储乘积的结果
    value = []

    # 计算乘积并存储结果
    for i in range(len(row_pointers1) - 1):
        for j in range(row_pointers1[i], row_pointers1[i+1]):
            value.append(dd[i] * dd[column_index1[j]])

    # 创建稀疏矩阵
    sparse_matrix = csr_matrix((value, column_index1.numpy(), row_pointers1), shape=(m, n))

    # 计算稀疏矩阵和右手边向量 rhs 的乘积
    result = sparse_matrix.dot(rhs.numpy())

    return result

# 设置随机种子以确保结果可重复
torch.manual_seed(42)
window = 8
graph = np.load('/home/shijinliang/module/Libra/dgl_dataset/gcn_matrix/cora.npz')
num_nodes_ori =  graph['num_nodes_src']-0
num_nodes_dst =  graph['num_nodes_dst']-0
num_nodes = num_nodes_ori
if num_nodes_ori%window !=0 :
        num_nodes = num_nodes_ori + window - num_nodes_ori%window 
num_edges = graph['num_edges']-0
src_li = graph['src_li']
dst_li = graph['dst_li']
edge_index = np.stack([src_li, dst_li])
avg_degree = num_edges / num_nodes
print("Num_nodes, Num_edges: " + str(num_nodes) + ' , ' + str(num_edges))
val = [1] * num_edges
scipy_coo = coo_matrix((val, edge_index), shape=(num_nodes, num_nodes_dst))
adj = scipy_coo.tocsr()

col = torch.IntTensor(adj.indices)
row = torch.IntTensor(adj.indptr)
dd = torch.randint(low=1, high=3, size=(num_edges,)).float()


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
c_valueTensor_short= Libra5Block.block_tf32_tcu_cuda_binary(row,col,dd, 32, 32, 3, 3, 8, 4)

# print(t_rowNew_offsetTensor)
# print(t_blockTensor)
# print(t_columnTensor)
# print(t_valueTensor)
# print(t_window_rowTensor)
# print(t_atomicTensor)
# print(t_binaryTensor)
# print()

# print(c_row_offsetTensor)
# print(c_rowTensor)
# print(c_atomicTensor)
# print(c_colTensor)
# print(c_valueTensor)
# print()

# print(c_row_offsetTensor_short)
# print(c_rowTensor_short)
# print(c_atomicTensor_short)
# print(c_colTensor_short)
# print(c_valueTensor_short)
        
parts_t = t_atomicTensor.shape[0]
parts_c = c_atomicTensor.shape[0]
parts_c_short = c_atomicTensor_short.shape[0]
partsize_c = 32
dimN = 128

# 生成形状为 (30, dimN) 的张量，其中值为 1 或 2 的随机数
rhs = torch.randint(low=1, high=3, size=(num_nodes_dst, dimN)).float()

t_binaryTensor = t_binaryTensor.int()
result, spmm_ms_avg = Libra5BenchmarkGCN.forward_tf32_tcu_cuda(
t_rowNew_offsetTensor, 
t_blockTensor,
t_columnTensor, 
t_valueTensor, 
t_window_rowTensor,
t_atomicTensor,
t_binaryTensor,

c_row_offsetTensor,
c_rowTensor, 
c_atomicTensor,
c_colTensor,
c_valueTensor, 

c_row_offsetTensor_short,
c_rowTensor_short, 
c_atomicTensor_short,
c_colTensor_short,
c_valueTensor_short, 

rhs, 
parts_t, 
parts_c, 
partsize_c,
parts_c_short, 
rhs.size(1), 
num_nodes_ori, 
num_nodes_dst,
1)
print()
# print(result)


# #check 100行的第一个值和最后一个值
for i in range(100):
        front = 0
        end = 0
        for j in range(row[i], row[i+1]):
                front +=dd[j]*rhs[col[j]][0]
                end +=dd[j]*rhs[col[j]][-1]
                
        if (result[i][0] - front) != 0 :
                print("No")
                exit(0)
        if (result[i][-1] - end) != 0 :
                print("No")
                exit(0)
        
print("PASS")