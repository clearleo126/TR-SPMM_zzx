import numpy as np
import torch
import Libra5Block
# row = torch.tensor([0,4,5,6,8,8,9,11,13,16,17,17,17,18,18,18,18,22,23,24,26,26,27,29,31,34,35,35,35,36,36,36,36],dtype=torch.int32)

row = torch.tensor([0, 6, 7,  8, 10, 10, 11, 13, 15, 18, 19, 19, 19, 20, 20, 20, 20, 24,
        25, 26, 28, 28, 29, 31, 33, 36, 37, 37, 37, 38, 38, 38, 38],dtype=torch.int32)
col = torch.tensor([1,2,3,11,16,21, 1,1,2,24,25,16,22,0,27,0,4,25,9,27,1,3,11,21,1,1,2,24,25,16,22,0,27,0,4,25,9,27],dtype=torch.int32)
value=col.float()
    

# t_rowNew_offsetTensor, t_valueTensor, t_columnTensor,  t_window_rowTensor, t_atomicTensor, \
# c_row_offsetTensor, c_rowTensor, c_atomicTensor, c_colTensor, c_valueTensor, \
# c_row_offsetTensor_residue, c_rowTensor_residue, c_atomicTensor_residue, c_colTensor_residue, c_valueTensor_residue= Libra5Block.block_tf32_mtcu_cuda_v2(row,col,value, 1, 4, 2, 0.13, 8, 4)
# print(t_rowNew_offsetTensor)
# print(t_valueTensor)
# print(t_columnTensor)
# print(t_window_rowTensor)
# print(t_atomicTensor)
# print()

# print(c_row_offsetTensor)
# print(c_rowTensor)
# print(c_atomicTensor)
# print(c_colTensor)
# print(c_valueTensor)
# print()

# print(c_row_offsetTensor_residue)
# print(c_rowTensor_residue)
# print(c_atomicTensor_residue)
# print(c_colTensor_residue)
# print(c_valueTensor_residue)

# t_rowNew_offsetTensor, t_blockTensor, t_columnTensor, t_valueTensor, t_binaryTensor = Libra5Block.block_csr_v2_tcu(row,col,value, 1,  8, 4)
# print(t_rowNew_offsetTensor)
# print(t_blockTensor)
# print(t_columnTensor)
# print(t_valueTensor)
# print(t_binaryTensor)

# # #sddmm
# t_rowNew_offsetTensor, \
# t_blockTensor, \
# t_columnTensor, \
# t_window_rowTensor, \
# t_atomicTensor, \
# t_binaryTensor, \
# c_row_offsetTensor, \
# c_rowTensor, \
# c_colTensor = Libra5Block.block_tf32_tcu_cuda_binary_sddmm(row,col, 2, 4, 6, 8, 16)

# print(t_rowNew_offsetTensor)
# print(t_blockTensor)
# print(t_columnTensor)
# print(t_window_rowTensor)
# print(t_atomicTensor)
# print(t_binaryTensor)
# print()

# print(c_row_offsetTensor)
# print(c_rowTensor)
# print(c_colTensor)
# print()


#sddmm part
t_rowNew_offsetTensor, \
t_blockTensor, \
t_columnTensor, \
t_window_rowTensor, \
t_atomicTensor, \
t_binaryTensor = Libra5Block.block_csr_binary_sddmm_tcu_part_c1(row,col, 2, 8, 16)

print(t_rowNew_offsetTensor)
print(t_blockTensor)
print(t_columnTensor)
print(t_window_rowTensor)
print(t_atomicTensor)
print(t_binaryTensor)
print()

# #sddmm_cuda_part
# c_row_offsetTensor, \
# c_rowTensor, \
# c_colTensor = Libra5Block.block_csr_sddmm_cuda_v2(row,col,value, 4)

# print(c_row_offsetTensor)
# print(c_rowTensor)
# print(c_colTensor)
# print()


# t_rowNew_offsetTensor, \
# t_blockTensor, \
# t_columnTensor, \
# _, \
# t_window_rowTensor, \
# t_atomicTensor, \
# t_rowTensor = Libra5Block.block_csr_tcu_part(row,col,value, 2, 8, 16)

# print(t_rowNew_offsetTensor)
# print(t_blockTensor)
# print(t_columnTensor)
# print(t_window_rowTensor)
# print(t_atomicTensor)
# print(t_rowTensor)
# print()



# c_row_offsetTensor, \
# c_rowTensor, \
# c_colTensor = Libra5Block.block_csr_sddmm_cuda_v2(row,col, 4)

# print(c_row_offsetTensor)
# print(c_rowTensor)
# print(c_colTensor)
# print()