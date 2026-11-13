import numpy as np
import torch
import Libra5Block
# row = torch.tensor([0,4,5,6,8,8,9,11,13,16,17,17,17,18,18,18,18,22,23,24,26,26,27,29,31,34,35,35,35,36,36,36,36],dtype=torch.int32)

row = torch.tensor([0, 6, 7,  8, 10, 10, 11, 13, 15, 18, 19, 19, 19, 20, 20, 20, 20, 24,
        25, 26, 28, 28, 29, 31, 33, 36, 37, 37, 37, 38, 38, 38, 38],dtype=torch.int32)
col = torch.tensor([1,2,3,11,16,21, 1,1,2,24,25,16,22,0,27,0,4,25,9,27,1,3,11,21,1,1,2,24,25,16,22,0,27,0,4,25,9,27],dtype=torch.int32)
value=col.float()
    
# t_row_offsetTensor, t_colTensor, t_valueTensor, c_row_offsetTensor, c_colTensor, c_valueTensor= Libra5Block.block_8_4(row,col,value, 0.15)

# print(t_row_offsetTensor)
# print(t_colTensor)
# print(t_valueTensor)
# print(c_row_offsetTensor)
# print(c_colTensor)
# print(c_valueTensor)


t_row_offsetTensor, t_colTensor, t_valueTensor, t_rowTensor, t_c_partTensor, t_c_row_offsetTensor, t_c_rowTensor, t_c_atomicTensor, t_c_colTensor, t_c_valueTensor, c_row_offsetTensor, c_rowTensor, c_atomicTensor, c_colTensor, c_valueTensor= Libra5Block.block_16_4_balance(row,col,value, 6, 4)

print(t_row_offsetTensor)
print(t_colTensor)
print(t_valueTensor)
print(t_rowTensor)
print()

print(t_c_partTensor)
print(t_c_row_offsetTensor)
print(t_c_rowTensor)
print(t_c_atomicTensor)
print(t_c_colTensor)
print(t_c_valueTensor)
print()

print(c_row_offsetTensor)
print(c_rowTensor)
print(c_atomicTensor)
print(c_colTensor)
print(c_valueTensor)