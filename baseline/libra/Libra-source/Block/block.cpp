#include <torch/extension.h>
#include <omp.h>
#include <chrono>  // std::chrono 包含计时功能
struct Value_tf32 {
    std::vector<float> value;
    std::vector<float> value_templete;
    std::vector<int> colum;
    int row;
    int pad;
};

struct Value1 {
    int t_row_offset;
    std::vector<int> t_column;
    std::vector<float> t_value;
    int t_window;
    std::vector<int> t_c_part_offset;
    std::vector<int> t_c_row_offset;
    std::vector<int> t_c_row;
    std::vector<int> t_c_atomic;
    std::vector<int> t_c_column;
    std::vector<float> t_c_value;

    // std::vector<int> c_part_offset;
    std::vector<int> c_row_offset;
    std::vector<int> c_row;
    std::vector<int> c_atomic;
    std::vector<int> c_column;
    std::vector<float> c_value;
};

struct Value1_1 {
    int t_row_offset;
    std::vector<int> t_column;
    std::vector<float> t_value;

    std::vector<int> c_row_offset;
    std::vector<int> c_column;
    std::vector<float> c_value;
};


struct Value1_2 {
    int t_row_offset;
    std::vector<int> t_column;
    std::vector<float> t_value;
    int pad;

    std::vector<int> c_part_offset;
    std::vector<int> c_row_offset;
    std::vector<int> c_row;
    std::vector<int> c_atomic;
    std::vector<int> c_column;
    std::vector<float> c_value;
};

struct Value1_3 {
    int t_row_offset;
    std::vector<int> t_column;
    std::vector<float> t_value;
    int pad;
    int t_window;

    std::vector<int> c_part_offset;
    std::vector<int> c_row_offset;
    std::vector<int> c_row;
    std::vector<int> c_atomic;
    std::vector<int> c_column;
    std::vector<float> c_value;

    bool tcu_flag;
};
struct Value1_4 {
    int t_row_offset;
    std::vector<int> t_column;
    std::vector<float> t_value;
    int t_atomic;
    int t_window;

    std::vector<int> c_part_offset;
    std::vector<int> c_row_offset;
    std::vector<int> c_row;
    std::vector<int> c_atomic;
    std::vector<int> c_column;
    std::vector<float> c_value;

    bool tcu_flag = false;
    bool cuda_flag = false;
};
struct Value1_5 {
    //记录window中有几个block
    int t_window_offset;
    //记录window中每个block中的非零元个数
    std::vector<int> t_block_offset;
    //记录值
    std::vector<float> t_value;
    //记录window中的列索引
    std::vector<int> t_column;
    //记录每个元素在window内的行和列
    std::vector<int> t_row;
    std::vector<int> t_col;
    std::vector<long> t_binary;
    //当前window
    int t_window;
    bool tcu_flag = false;

    std::vector<int> c_part_offset;
    std::vector<int> c_row_offset;
    std::vector<int> c_row;
    std::vector<int> c_atomic;
    std::vector<int> c_column;
    std::vector<float> c_value;
    bool cuda_flag = false;
};
struct Value_cuda {
    int c_row_offset;
    int c_row;
    int c_atomic;
    int c_atomic_seq;
    std::vector<int> c_column;
    std::vector<float> c_value;
    bool cuda_flag = false;
    bool cuda_residue_flag = false;
};
struct Value_tcu {
    std::vector<float> t_value;
    std::vector<int> t_column;
    std::vector<int> row;
    bool tcu_flag = false;
    std::vector<int> window;
    std::vector<int> atomic;
};

struct Value_tcu_csr {
    std::vector<float> t_value;
    std::vector<int> t_column;
    std::vector<int> row;
    //记录window中每个block中的非零元个数
    std::vector<int> t_block_offset;
    bool tcu_flag = false;
    //记录每个元素在window内的行和列
    std::vector<int> t_row;
    std::vector<int> t_col;
    std::vector<int> window;
    std::vector<int> atomic;
    std::vector<long> t_binary;
};

struct Value_tcu_binary {
    std::vector<float> t_value;
    std::vector<int> t_column;
    std::vector<int> row;
    //记录window中每个block中的非零元个数
    std::vector<int> t_block_offset;
    bool tcu_flag = false;
    //记录每个元素在window内的行和列
    std::vector<int> t_row;
    std::vector<int> t_col;
    std::vector<long> t_binary;
    std::vector<int> window;
    std::vector<int> atomic;
    std::vector<int> atomic_seq;
};

struct Value1_6 {
    //记录window中有几个block
    int t_window_offset;
    //记录window中每个block中的非零元个数
    std::vector<int> t_block_offset;
    //记录值
    std::vector<float> t_value;
    //记录window中的列索引
    std::vector<int> t_column;
    //记录每个元素在window内的行和列
    std::vector<int> t_row;
    std::vector<int> t_col;
    //当前window
    int t_window;
    bool tcu_flag = false;

    std::vector<Value_cuda> cuda_info;
    bool cuda_flag = false;
};


// 比较函数，用于指定排序规则
bool compareByCRowOffsetDesc(const Value_cuda& a, const Value_cuda& b) {
    return a.c_row_offset > b.c_row_offset;
}

// 初始化vector
std::vector<std::vector<int>> createVectorOfVectors(int numVectors) {
    std::vector<std::vector<int>> vectorOfVectors;

    // 在向量数组中放入指定数量的空向量
    for (int i = 0; i < numVectors; ++i) {
        std::vector<int> emptyVector;
        vectorOfVectors.push_back(emptyVector);
    }

    return vectorOfVectors;
}

// 初始化vector
std::vector<long> createVectorOfVectors_v2(int numVectors) {
    std::vector<long> vectorOfVectors;

    // 在向量数组中放入指定数量的空向量
    for (int i = 0; i < numVectors; ++i) {
        vectorOfVectors.push_back(0);
    }

    return vectorOfVectors;
}
// 合并vector
std::vector<int> mergeVectors(const std::vector<std::vector<int>>& vectorOfVectors) {
    std::vector<int> mergedVector;

    // 遍历向量数组中的每个向量，并将其元素添加到合并后的向量中
    for (const auto& vec : vectorOfVectors) {
        for (int num : vec) {
            mergedVector.push_back(num);
        }
    }

    return mergedVector;
}
// 初始化vector
std::vector<std::vector<float>> createVectorOfVectors_float(int numVectors) {
    std::vector<std::vector<float>> vectorOfVectors;

    // 在向量数组中放入指定数量的空向量
    for (int i = 0; i < numVectors; ++i) {
        std::vector<float> emptyVector;
        vectorOfVectors.push_back(emptyVector);
    }

    return vectorOfVectors;
}
// 合并vector
std::vector<float> mergeVectors_float(const std::vector<std::vector<float>>& vectorOfVectors) {
    std::vector<float> mergedVector;

    // 遍历向量数组中的每个向量，并将其元素添加到合并后的向量中
    for (const auto& vec : vectorOfVectors) {
        for (float num : vec) {
            mergedVector.push_back(num);
        }
    }

    return mergedVector;
}
//统计vector中每个子vector中的非零元数
std::vector<int> countVectors(const std::vector<std::vector<float>>& vectorOfVectors) {
    std::vector<int> res;
    // 在向量数组中放入指定数量的空向量
    for (const auto& vec : vectorOfVectors) {
        res.push_back(vec.size());
    }
    return res;
}

// 初始化vector
std::vector<Value_tcu> createVectorTcu(int numVectors, int block_size, int partSize) {
    std::vector<Value_tcu> vectorOfVectors;

    // 在向量数组中放入指定数量的空向量
    for (int i = 0; i < numVectors; ++i) {
        Value_tcu v;
        std::vector<float> demo(block_size*partSize);
        v.t_value = demo;
        vectorOfVectors.push_back(v);
    }
    return vectorOfVectors;
}

// 长短行拆分
std::vector<torch::Tensor> block_csr_cuda_kernel(torch::Tensor row1, torch::Tensor column1, torch::Tensor values1, int partSize, int shortSize)
{
    auto *row=row1.data_ptr<int>();
    auto column=column1.accessor<int, 1>();
    auto values=values1.accessor<float, 1>();
    int rows=row1.size(0)-1;
    std::map<int, std::vector<Value_cuda>> res;
    std::map<int, std::vector<Value_cuda>> res_residue;
    int cuda_rows=0;
    //统计vector的稠密度，然后按照降序排列
    #pragma omp parallel for
    for(int i=0;i<rows;i++)
    {
        std::vector<Value_cuda> large;
        std::vector<Value_cuda> small;
        int cuda_nnz = row[i+1] - row[i];
        if(cuda_nnz>0)
        {
            if(cuda_nnz <= shortSize)
            {
                Value_cuda temp;
                temp.c_row_offset = cuda_nnz;
                temp.c_row = i;
                temp.c_atomic = 0;
                for(int m=row[i]; m<row[i+1]; m++)
                {
                    temp.c_column.push_back(column[m]);
                    temp.c_value.push_back(values[m]);
                }
                small.push_back(temp);
            }else{
                    if(cuda_nnz <= partSize)
                    {
                        Value_cuda temp;
                        temp.c_row_offset = (cuda_nnz);
                        temp.c_row = i;
                        temp.c_atomic = 0;
                        for(int m=row[i]; m<row[i+1]; m++)
                        {
                            temp.c_column.push_back(column[m]);
                            temp.c_value.push_back(values[m]);
                        }
                        large.push_back(temp);
                    }else{
                            // 当前行的非零元被分成多个part，并需要原子操作
                            int parts_row = cuda_nnz/partSize;
                            int cur_value_indice = row[i];
                            for(int cur_i = 0; cur_i<parts_row; cur_i++)
                            {
                                Value_cuda temp;
                                temp.c_row_offset = partSize;
                                temp.c_row = i;
                                temp.c_atomic = 1;
                                for(int m=0; m<partSize; m++)
                                {
                                    temp.c_column.push_back(column[cur_value_indice]);
                                    temp.c_value.push_back(values[cur_value_indice]);
                                    cur_value_indice++;
                                }
                                large.push_back(temp);
                            }
                            int residue  = (cuda_nnz%partSize);
                            if(residue>0){
                                Value_cuda temp;
                                temp.c_row_offset = (cuda_nnz%partSize);
                                temp.c_row = i;
                                temp.c_atomic = 1;
                                for(int m=0; m<(cuda_nnz%partSize); m++)
                                {
                                    temp.c_column.push_back(column[cur_value_indice]);
                                    temp.c_value.push_back(values[cur_value_indice]);
                                    cur_value_indice++;
                                }
                                if(residue <= shortSize)
                                {
                                    small.push_back(temp);
                                }else{
                                    large.push_back(temp);
                                }
                            }
                    }
            }
        }

        #pragma omp critical  
        { 
            res[i]=large;
            res_residue[i]=small;
        }
    }

    std::vector<int> c_rowNew_offset;
    c_rowNew_offset.push_back(0);
    std::vector<int> c_rowNew;
    std::vector<int> c_atomicNew;
    std::vector<int> c_columnNew;
    std::vector<float> c_valueNew;

    std::vector<int> c_rowNew_offset_residue;
    c_rowNew_offset_residue.push_back(0);
    std::vector<int> c_rowNew_residue;
    std::vector<int> c_atomicNew_residue;
    std::vector<int> c_columnNew_residue;
    std::vector<float> c_valueNew_residue;

    //按顺序整合res
    for (const auto& pair : res) {
        for( Value_cuda sub : pair.second){
            c_rowNew_offset.push_back(c_rowNew_offset.back()+sub.c_row_offset);
            c_rowNew.push_back(sub.c_row);
            c_atomicNew.push_back(sub.c_atomic);
            c_columnNew.insert(c_columnNew.end(),sub.c_column.begin(),sub.c_column.end());
            c_valueNew.insert(c_valueNew.end(),sub.c_value.begin(), sub.c_value.end());
        }
    }

    auto c_row_offsetTensor1 = torch::from_blob(c_rowNew_offset.data(), c_rowNew_offset.size(), torch::kInt32);
    auto c_rowTensor1 = torch::from_blob(c_rowNew.data(), c_rowNew.size(), torch::kInt32);
    auto c_atomicTensor1 = torch::from_blob(c_atomicNew.data(), c_atomicNew.size(), torch::kInt32);
    auto c_colTensor1 = torch::from_blob(c_columnNew.data(), c_columnNew.size(), torch::kInt32);
    auto c_valueTensor1 = torch::from_blob(c_valueNew.data(), c_valueNew.size(), torch::kFloat32);
    
    torch::Tensor c_row_offsetTensor = torch::empty_like(c_row_offsetTensor1);
    c_row_offsetTensor.copy_(c_row_offsetTensor1);
    torch::Tensor c_rowTensor = torch::empty_like(c_rowTensor1);
    c_rowTensor.copy_(c_rowTensor1);
    torch::Tensor c_atomicTensor = torch::empty_like(c_atomicTensor1);
    c_atomicTensor.copy_(c_atomicTensor1);
    torch::Tensor c_colTensor = torch::empty_like(c_colTensor1);
    c_colTensor.copy_(c_colTensor1);
    torch::Tensor c_valueTensor = torch::empty_like(c_valueTensor1);
    c_valueTensor.copy_(c_valueTensor1);



    //按顺序整合res_residue
    for (const auto& pair : res_residue) {
        for( Value_cuda sub : pair.second){
            c_rowNew_offset_residue.push_back(c_rowNew_offset_residue.back()+sub.c_row_offset);
            c_rowNew_residue.push_back(sub.c_row);
            c_atomicNew_residue.push_back(sub.c_atomic);
            c_columnNew_residue.insert(c_columnNew_residue.end(),sub.c_column.begin(),sub.c_column.end());
            c_valueNew_residue.insert(c_valueNew_residue.end(),sub.c_value.begin(), sub.c_value.end());
        }
    }

    auto c_row_offsetTensor1_residue = torch::from_blob(c_rowNew_offset_residue.data(), c_rowNew_offset_residue.size(), torch::kInt32);
    auto c_rowTensor1_residue = torch::from_blob(c_rowNew_residue.data(), c_rowNew_residue.size(), torch::kInt32);
    auto c_atomicTensor1_residue = torch::from_blob(c_atomicNew_residue.data(), c_atomicNew_residue.size(), torch::kInt32);
    auto c_colTensor1_residue = torch::from_blob(c_columnNew_residue.data(), c_columnNew_residue.size(), torch::kInt32);
    auto c_valueTensor1_residue = torch::from_blob(c_valueNew_residue.data(), c_valueNew_residue.size(), torch::kFloat32);
    
    torch::Tensor c_row_offsetTensor_residue = torch::empty_like(c_row_offsetTensor1_residue);
    c_row_offsetTensor_residue.copy_(c_row_offsetTensor1_residue);
    torch::Tensor c_rowTensor_residue = torch::empty_like(c_rowTensor1_residue);
    c_rowTensor_residue.copy_(c_rowTensor1_residue);
    torch::Tensor c_atomicTensor_residue = torch::empty_like(c_atomicTensor1_residue);
    c_atomicTensor_residue.copy_(c_atomicTensor1_residue);
    torch::Tensor c_colTensor_residue = torch::empty_like(c_colTensor1_residue);
    c_colTensor_residue.copy_(c_colTensor1_residue);
    torch::Tensor c_valueTensor_residue = torch::empty_like(c_valueTensor1_residue);
    c_valueTensor_residue.copy_(c_valueTensor1_residue);



    return {c_row_offsetTensor, c_rowTensor, c_atomicTensor, c_colTensor, c_valueTensor,
    c_row_offsetTensor_residue, c_rowTensor_residue, c_atomicTensor_residue, c_colTensor_residue, c_valueTensor_residue};

}

// 长短行拆分
std::vector<torch::Tensor> block_csr_sddmm_cuda_kernel(torch::Tensor row1, torch::Tensor column1, int partSize)
{
    auto *row=row1.data_ptr<int>();
    auto column=column1.accessor<int, 1>();
    int rows=row1.size(0)-1;
    std::map<int, std::vector<Value_cuda>> res;
    std::map<int, std::vector<Value_cuda>> res_residue;
    int cuda_rows=0;
    #pragma omp parallel for
    for(int i=0;i<rows;i++)
    {
        std::vector<Value_cuda> large;
        // std::vector<Value_cuda> small;
        int cuda_nnz = row[i+1] - row[i];
        if(cuda_nnz>0)
        {
            if(cuda_nnz <= partSize)
            {
                Value_cuda temp;
                temp.c_row_offset = (cuda_nnz);
                temp.c_row = i;
                // temp.c_atomic = 0;
                for(int m=row[i]; m<row[i+1]; m++)
                {
                    temp.c_column.push_back(column[m]);
                    // temp.c_value.push_back(values[m]);
                }
                large.push_back(temp);
            }else{
                    // 当前行的非零元被分成多个part，并需要原子操作
                    int parts_row = cuda_nnz/partSize;
                    int cur_value_indice = row[i];
                    for(int cur_i = 0; cur_i<parts_row; cur_i++)
                    {
                        Value_cuda temp;
                        temp.c_row_offset = partSize;
                        temp.c_row = i;
                        // temp.c_atomic = 1;
                        for(int m=0; m<partSize; m++)
                        {
                            temp.c_column.push_back(column[cur_value_indice]);
                            // temp.c_value.push_back(values[cur_value_indice]);
                            cur_value_indice++;
                        }
                        large.push_back(temp);
                    }
                    int residue  = (cuda_nnz%partSize);
                    if(residue>0){
                        Value_cuda temp;
                        temp.c_row_offset = (cuda_nnz%partSize);
                        temp.c_row = i;
                        // temp.c_atomic = 1;
                        for(int m=0; m<(cuda_nnz%partSize); m++)
                        {
                            temp.c_column.push_back(column[cur_value_indice]);
                            // temp.c_value.push_back(values[cur_value_indice]);
                            cur_value_indice++;
                        }
                        large.push_back(temp);
                    }
            }
        }

        #pragma omp critical  
        { 
            res[i]=large;
            // res_residue[i]=small;
        }
    }

    std::vector<int> c_rowNew_offset;
    c_rowNew_offset.push_back(0);
    std::vector<int> c_rowNew;
    std::vector<int> c_columnNew;

    //按顺序整合res
    for (const auto& pair : res) {
        for( Value_cuda sub : pair.second){
            c_rowNew_offset.push_back(c_rowNew_offset.back()+sub.c_row_offset);
            c_rowNew.push_back(sub.c_row);
            c_columnNew.insert(c_columnNew.end(),sub.c_column.begin(),sub.c_column.end());
        }
    }

    auto c_row_offsetTensor1 = torch::from_blob(c_rowNew_offset.data(), c_rowNew_offset.size(), torch::kInt32);
    auto c_rowTensor1 = torch::from_blob(c_rowNew.data(), c_rowNew.size(), torch::kInt32);
    auto c_colTensor1 = torch::from_blob(c_columnNew.data(), c_columnNew.size(), torch::kInt32);
    
    torch::Tensor c_row_offsetTensor = torch::empty_like(c_row_offsetTensor1);
    c_row_offsetTensor.copy_(c_row_offsetTensor1);
    torch::Tensor c_rowTensor = torch::empty_like(c_rowTensor1);
    c_rowTensor.copy_(c_rowTensor1);
    torch::Tensor c_colTensor = torch::empty_like(c_colTensor1);
    c_colTensor.copy_(c_colTensor1);

    return {c_row_offsetTensor, c_rowTensor, c_colTensor};

}
std::vector<torch::Tensor> block_csr_tcu_kernel(torch::Tensor row1, torch::Tensor column1, torch::Tensor values1, int partSize, int window1, int wide1)
{
    auto start = std::chrono::high_resolution_clock::now();
    auto *row=row1.data_ptr<int>();
    auto column=column1.accessor<int, 1>();
    auto values=values1.accessor<float, 1>();
    int rows=row1.size(0)-1;
    int window = window1;
    int wide = wide1;
    int warps_block = 4;
    //需要的window数量
    int num_windows = rows/window;
    //最终的map
    std::map<int, Value1_5> res;
    int tcu_rows=0;
    //统计vector的稠密度，然后按照降序排列
    #pragma omp parallel for
    for(int i=0;i<num_windows;i++)
    {
        Value1_5 v;
        //对column进行合并
        std::vector<int> mergedVector(&column[row[i*window]], &column[row[i*window+window]]); 
        // 使用 unordered_map 统计 vector 中每个元素出现次数
        std::unordered_map<int, int> elementCounts;
        for (const auto& element : mergedVector) {
            elementCounts[element]++;
        }
        // 将统计结果放入 vector<pair> 中
        std::vector<std::pair<int, int>> countVector(elementCounts.begin(), elementCounts.end());
        //获取countVector的长度
        int size = countVector.size();
        int residue = size%wide;
        //统计每个block中的非零元数量，计算稠密度
        int percen = size; 
        //当前window中有几个block
        v.t_window_offset = percen/wide;
        if(residue>0) v.t_window_offset+=1;
        if(v.t_window_offset>0)
        {   
            std::vector<std::vector<float>> t_value_temp = createVectorOfVectors_float(v.t_window_offset);
            std::vector<std::vector<int>> t_row_temp = createVectorOfVectors(v.t_window_offset);
            // std::vector<std::vector<int>> t_col_temp = createVectorOfVectors(v.t_window_offset);
            // 将前 k 个数按升序排列
            std::sort(countVector.begin(), countVector.end());

            // #pragma omp critical  
            // {tcu_rows+=window;}
            // 记录当前window
            v.t_window = i;
            // v.tcu_flag = true;

            std::map<int, int> colmap;
            int c=0;
            for(auto col:countVector){
                v.t_column.push_back(col.first);
                colmap[col.first]=c++;
            }
            //把额外的1放进去
            if((wide-residue)>0){
                for(int j=0; j<(wide-residue); j++)
                 v.t_column.push_back(-1);
            }

            int cur_row = 0;
            // 遍历window中的每一行
            for(int j=i*window; j<(i+1)*window; j++)
            {
                // 遍历行中的每个元素
                for(int m=row[j];m<row[j+1];m++)
                {
                    int newCol = colmap[column[m]];
                    int block = newCol/wide;

                    //TCU 计算的数据
                    //存储按wide列为一个block存储
                    t_value_temp[block].push_back(values[m]);
                    t_row_temp[block].push_back((cur_row*wide) + (newCol%wide));
                    // t_col_temp[block].push_back(newCol%wide);
                }
                cur_row++;
            }
            //整合vector
            v.t_value = mergeVectors_float(t_value_temp);
            v.t_row = mergeVectors(t_row_temp);
            // v.t_col = mergeVectors(t_col_temp);
            v.t_block_offset = countVectors(t_value_temp);
        }

        #pragma omp critical  
        {  res[i]=v;}
    }
    // std::cout << "tcu rows:" << tcu_rows << std::endl;
    //记录每个window中的vector数量
    std::vector<int> t_windowNew_offset;
    t_windowNew_offset.push_back(0);
    std::vector<int> t_blockNew_offset;
    t_blockNew_offset.push_back(0);
    std::vector<float> t_valueNew;
    std::vector<int> t_columnNew;
    std::vector<int> t_rowNew;
    // std::vector<int> t_colNew;
    // std::vector<int> t_window_rowNew;

    //按顺序整合res
    for (const auto& pair : res) {
        // if(pair.second.tcu_flag)
        // {
            //首先整合TCU的数据
            t_windowNew_offset.push_back(t_windowNew_offset.back() + pair.second.t_window_offset);  
            for(int sub : pair.second.t_block_offset)
            {
                t_blockNew_offset.push_back(t_blockNew_offset.back()+sub);
            }  
            t_valueNew.insert(t_valueNew.end(),pair.second.t_value.begin(),pair.second.t_value.end());
            t_columnNew.insert(t_columnNew.end(),pair.second.t_column.begin(),pair.second.t_column.end());
            t_rowNew.insert(t_rowNew.end(),pair.second.t_row.begin(),pair.second.t_row.end());
            // t_colNew.insert(t_colNew.end(),pair.second.t_col.begin(),pair.second.t_col.end());
            // t_window_rowNew.push_back(pair.second.t_window);
        // }
    }

    // 获取结束时间点
    auto end = std::chrono::high_resolution_clock::now();

    // 计算耗时
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;

    auto t_windowNew_offsetTensor1 = torch::from_blob(t_windowNew_offset.data(), t_windowNew_offset.size(), torch::kInt32);
    auto t_blockNew_offsetTensor1 = torch::from_blob(t_blockNew_offset.data(), t_blockNew_offset.size(), torch::kInt32);
    auto t_valueTensor1 = torch::from_blob(t_valueNew.data(), t_valueNew.size(), torch::kFloat32);
    auto t_columnTensor1 = torch::from_blob(t_columnNew.data(), t_columnNew.size(), torch::kInt32);
    auto t_rowTensor1 = torch::from_blob(t_rowNew.data(), t_rowNew.size(), torch::kInt32);
    //auto t_colTensor1 = torch::from_blob(t_colNew.data(), t_colNew.size(), torch::kInt32);
    // auto t_window_rowTensor1 = torch::from_blob(t_window_rowNew.data(), t_window_rowNew.size(), torch::kInt32);

    torch::Tensor t_windowNew_offsetTensor = torch::empty_like(t_windowNew_offsetTensor1);
    t_windowNew_offsetTensor.copy_(t_windowNew_offsetTensor1);
    torch::Tensor t_blockNew_offsetTensor = torch::empty_like(t_blockNew_offsetTensor1);
    t_blockNew_offsetTensor.copy_(t_blockNew_offsetTensor1);
    torch::Tensor t_valueTensor = torch::empty_like(t_valueTensor1);
    t_valueTensor.copy_(t_valueTensor1);
    torch::Tensor t_columnTensor = torch::empty_like(t_columnTensor1);
    t_columnTensor.copy_(t_columnTensor1);
    torch::Tensor t_rowTensor = torch::empty_like(t_rowTensor1);
    t_rowTensor.copy_(t_rowTensor1);
    // torch::Tensor t_colTensor = torch::empty_like(t_colTensor1);
    // t_colTensor.copy_(t_colTensor1);
    // torch::Tensor t_window_rowTensor = torch::empty_like(t_window_rowTensor1);
    // t_window_rowTensor.copy_(t_window_rowTensor1);

    return {t_windowNew_offsetTensor, t_blockNew_offsetTensor, t_columnTensor, t_valueTensor,t_rowTensor};

}

std::vector<torch::Tensor> block_sgt_tcu_kernel(torch::Tensor row1, torch::Tensor column1, torch::Tensor values1, int partSize, int window1, int wide1)
{
    auto *row=row1.data_ptr<int>();
    auto column=column1.accessor<int, 1>();
    auto values=values1.accessor<float, 1>();
    int rows=row1.size(0)-1;
    int window = window1;
    int wide = wide1;
    int warps_block = 4;
    //需要的window数量
    int num_windows = rows/window;
    //最终的map
    std::map<int, Value1_5> res;
    int tcu_rows=0;
    //统计vector的稠密度，然后按照降序排列
    #pragma omp parallel for
    for(int i=0;i<num_windows;i++)
    {
        Value1_5 v;
        //对column进行合并
        std::vector<int> mergedVector(&column[row[i*window]], &column[row[i*window+window]]);
        
        // 使用 unordered_map 统计 vector 中每个元素出现次数
        std::unordered_map<int, int> elementCounts;

        for (const auto& element : mergedVector) {
            elementCounts[element]++;
        }
        // 将统计结果放入 vector<pair> 中
        std::vector<std::pair<int, int>> countVector(elementCounts.begin(), elementCounts.end());

        // 按元素个数降序排序
        // std::sort(countVector.begin(), countVector.end(), [](const auto& a, const auto& b) {
        //     return a.second > b.second;
        // });
        //获取countVector的长度
        int size = countVector.size();
        int residue = size%wide;
        //统计每个block中的非零元数量，计算稠密度
        int percen = size; 
        //当前window中有几个block
        v.t_window_offset = percen/wide;
        if(residue>0) v.t_window_offset+=1;
        if(v.t_window_offset>0)
        {   
            std::vector<float> t_value_temp;
            std::vector<int> t_row_temp;
            std::vector<int> t_col_temp;
            // 将前 k 个数按升序排列
            std::sort(countVector.begin(), countVector.end());

            #pragma omp critical  
            {tcu_rows+=window;}
            // 记录当前window
            v.t_window = i;
            // v.tcu_flag = true;

            std::map<int, int> colmap;
            int c=0;
            for(auto col:countVector){
                v.t_column.push_back(col.first);
                colmap[col.first]=c++;
            }
            //把额外的1放进去
            if((wide-residue)>0){
                for(int j=0; j<(wide-residue); j++)
                 v.t_column.push_back(-1);
            }

            int cur_row = 0;
            // 遍历window中的每一行
            for(int j=i*window; j<(i+1)*window; j++)
            {
                // 遍历行中的每个元素
                for(int m=row[j];m<row[j+1];m++)
                {
                    int newCol = colmap[column[m]];
      
                    //存储按wide列为一个block存储
                    t_value_temp.push_back(values[m]);
                    t_row_temp.push_back(cur_row);
                    t_col_temp.push_back(newCol);
                }
                cur_row++;
            }
            //整合vector
            v.t_value = t_value_temp;
            v.t_row = t_row_temp;
            v.t_col = t_col_temp;
            // v.t_block_offset.push_back(t_value_temp.size());
        }

        #pragma omp critical  
        {  res[i]=v;}
    }
    // std::cout << "tcu rows:" << tcu_rows << std::endl;
    //记录每个window中的vector数量
    std::vector<int> t_windowNew_offset;
    // t_windowNew_offset.push_back(0);
    // std::vector<int> t_blockNew_offset;
    // t_blockNew_offset.push_back(0);
    std::vector<float> t_valueNew;
    // std::vector<int> t_columnNew;
    std::vector<int> t_rowNew;
    std::vector<int> t_colNew;
    // std::vector<int> t_window_rowNew;

    //按顺序整合res
    for (const auto& pair : res) {
        // if(pair.second.tcu_flag)
        // {
            //首先整合TCU的数据
            t_windowNew_offset.push_back(pair.second.t_window_offset);  
            // for(int sub : pair.second.t_block_offset)
            // {
            //     t_blockNew_offset.push_back(t_blockNew_offset.back()+sub);
            // }  
            t_valueNew.insert(t_valueNew.end(),pair.second.t_value.begin(),pair.second.t_value.end());
            // t_columnNew.insert(t_columnNew.end(),pair.second.t_column.begin(),pair.second.t_column.end());
            t_rowNew.insert(t_rowNew.end(),pair.second.t_row.begin(),pair.second.t_row.end());
            t_colNew.insert(t_colNew.end(),pair.second.t_col.begin(),pair.second.t_col.end());
            // t_window_rowNew.push_back(pair.second.t_window);
        // }
    }

    auto t_windowNew_offsetTensor1 = torch::from_blob(t_windowNew_offset.data(), t_windowNew_offset.size(), torch::kInt32);
   // auto t_blockNew_offsetTensor1 = torch::from_blob(t_blockNew_offset.data(), t_blockNew_offset.size(), torch::kInt32);
    auto t_valueTensor1 = torch::from_blob(t_valueNew.data(), t_valueNew.size(), torch::kFloat32);
    // auto t_columnTensor1 = torch::from_blob(t_columnNew.data(), t_columnNew.size(), torch::kInt32);
    auto t_rowTensor1 = torch::from_blob(t_rowNew.data(), t_rowNew.size(), torch::kInt32);
    auto t_colTensor1 = torch::from_blob(t_colNew.data(), t_colNew.size(), torch::kInt32);
    // auto t_window_rowTensor1 = torch::from_blob(t_window_rowNew.data(), t_window_rowNew.size(), torch::kInt32);

    torch::Tensor t_windowNew_offsetTensor = torch::empty_like(t_windowNew_offsetTensor1);
    t_windowNew_offsetTensor.copy_(t_windowNew_offsetTensor1);
    // torch::Tensor t_blockNew_offsetTensor = torch::empty_like(t_blockNew_offsetTensor1);
    // t_blockNew_offsetTensor.copy_(t_blockNew_offsetTensor1);
    torch::Tensor t_valueTensor = torch::empty_like(t_valueTensor1);
    t_valueTensor.copy_(t_valueTensor1);
    // torch::Tensor t_columnTensor = torch::empty_like(t_columnTensor1);
    // t_columnTensor.copy_(t_columnTensor1);
    torch::Tensor t_rowTensor = torch::empty_like(t_rowTensor1);
    t_rowTensor.copy_(t_rowTensor1);
    torch::Tensor t_colTensor = torch::empty_like(t_colTensor1);
    t_colTensor.copy_(t_colTensor1);
    // torch::Tensor t_window_rowTensor = torch::empty_like(t_window_rowTensor1);
    // t_window_rowTensor.copy_(t_window_rowTensor1);

    return {t_windowNew_offsetTensor, t_valueTensor,t_rowTensor, t_colTensor};

}


std::vector<torch::Tensor> block_csr_tcu_kernel_stream(torch::Tensor row1, torch::Tensor column1, torch::Tensor values1, int partSize, int window1, int wide1)
{
    auto *row=row1.data_ptr<int>();
    auto column=column1.accessor<int, 1>();
    auto values=values1.accessor<float, 1>();
    int rows=row1.size(0)-1;
    int window = window1;
    int wide = wide1;
    int warps_block = 4;
    //需要的window数量
    int num_windows = rows/window;
    //最终的map
    std::map<int, Value1_5> res;
    int tcu_rows=0;
    //统计vector的稠密度，然后按照降序排列
    #pragma omp parallel for
    for(int i=0;i<num_windows;i++)
    {
        Value1_5 v;
        //对column进行合并
        std::vector<int> mergedVector(&column[row[i*window]], &column[row[i*window+window]]);
        
        // 使用 unordered_map 统计 vector 中每个元素出现次数
        std::unordered_map<int, int> elementCounts;

        for (const auto& element : mergedVector) {
            elementCounts[element]++;
        }
        // 将统计结果放入 vector<pair> 中
        std::vector<std::pair<int, int>> countVector(elementCounts.begin(), elementCounts.end());

        //获取countVector的长度
        int size = countVector.size();
        int residue = size%wide;
        //统计每个block中的非零元数量，计算稠密度
        int percen = size; 
        //当前window中有几个block
        v.t_window_offset = percen/wide;
        if(residue>0) v.t_window_offset+=1;
        if(v.t_window_offset>0)
        {   
            std::vector<std::vector<float>> t_value_temp = createVectorOfVectors_float(v.t_window_offset);
            std::vector<std::vector<int>> t_row_temp = createVectorOfVectors(v.t_window_offset);
            // std::vector<std::vector<int>> t_col_temp = createVectorOfVectors(v.t_window_offset);
            // 将前 k 个数按升序排列
            std::sort(countVector.begin(), countVector.end());

            #pragma omp critical  
            {tcu_rows+=window;}
            // 记录当前window
            v.t_window = i;
            // v.tcu_flag = true;

            std::map<int, int> colmap;
            int c=0;
            for(auto col:countVector){
                v.t_column.push_back(col.first);
                colmap[col.first]=c++;
            }
            //把额外的1放进去
            if((wide-residue)>0){
                for(int j=0; j<(wide-residue); j++)
                 v.t_column.push_back(-1);
            }

            int cur_row = 0;
            // 遍历window中的每一行
            for(int j=i*window; j<(i+1)*window; j++)
            {
                // 遍历行中的每个元素
                for(int m=row[j];m<row[j+1];m++)
                {
                    int newCol = colmap[column[m]];
                    int block = newCol/wide;

                    //TCU 计算的数据
                    //存储按wide列为一个block存储
                    t_value_temp[block].push_back(values[m]);
                    t_row_temp[block].push_back((cur_row*wide) + (newCol%wide));
                    // t_col_temp[block].push_back(newCol%wide);
                }
                cur_row++;
            }
            //整合vector
            v.t_value = mergeVectors_float(t_value_temp);
            v.t_row = mergeVectors(t_row_temp);
            v.t_block_offset = countVectors(t_value_temp);
        }

        #pragma omp critical  
        {  res[i]=v;}
    }
    // std::cout << "tcu rows:" << tcu_rows << std::endl;
    //记录每个window中的vector数量
    std::vector<int> t_windowNew_offset;
    t_windowNew_offset.push_back(0);
    std::vector<int> t_blockNew_offset;
    t_blockNew_offset.push_back(0);
    std::vector<float> t_valueNew;
    std::vector<int> t_columnNew;
    std::vector<int> t_rowNew;

    std::vector<int> t_windowNew_offset1;
    t_windowNew_offset1.push_back(0);
    std::vector<int> t_blockNew_offset1;
    t_blockNew_offset1.push_back(0);
    std::vector<float> t_valueNew1;
    std::vector<int> t_columnNew1;
    std::vector<int> t_rowNew1;

    int bound = num_windows/2;

    //按顺序整合res
    int temp = 0;
    for (const auto& pair : res) {
            if(temp<bound){
            //首先整合TCU的数据
                t_windowNew_offset.push_back(t_windowNew_offset.back() + pair.second.t_window_offset);  
                for(int sub : pair.second.t_block_offset)
                {
                    t_blockNew_offset.push_back(t_blockNew_offset.back()+sub);
                }  
                t_valueNew.insert(t_valueNew.end(),pair.second.t_value.begin(),pair.second.t_value.end());
                t_columnNew.insert(t_columnNew.end(),pair.second.t_column.begin(),pair.second.t_column.end());
                t_rowNew.insert(t_rowNew.end(),pair.second.t_row.begin(),pair.second.t_row.end());
            }else{
                t_windowNew_offset1.push_back(t_windowNew_offset1.back() + pair.second.t_window_offset);  
                for(int sub : pair.second.t_block_offset)
                {
                    t_blockNew_offset1.push_back(t_blockNew_offset1.back()+sub);
                }  
                t_valueNew1.insert(t_valueNew1.end(),pair.second.t_value.begin(),pair.second.t_value.end());
                t_columnNew1.insert(t_columnNew1.end(),pair.second.t_column.begin(),pair.second.t_column.end());
                t_rowNew1.insert(t_rowNew1.end(),pair.second.t_row.begin(),pair.second.t_row.end());
                }
            temp++;
    }

    auto t_windowNew_offsetTensor1 = torch::from_blob(t_windowNew_offset.data(), t_windowNew_offset.size(), torch::kInt32);
    auto t_blockNew_offsetTensor1 = torch::from_blob(t_blockNew_offset.data(), t_blockNew_offset.size(), torch::kInt32);
    auto t_valueTensor1 = torch::from_blob(t_valueNew.data(), t_valueNew.size(), torch::kFloat32);
    auto t_columnTensor1 = torch::from_blob(t_columnNew.data(), t_columnNew.size(), torch::kInt32);
    auto t_rowTensor1 = torch::from_blob(t_rowNew.data(), t_rowNew.size(), torch::kInt32);


    torch::Tensor t_windowNew_offsetTensor = torch::empty_like(t_windowNew_offsetTensor1);
    t_windowNew_offsetTensor.copy_(t_windowNew_offsetTensor1);
    torch::Tensor t_blockNew_offsetTensor = torch::empty_like(t_blockNew_offsetTensor1);
    t_blockNew_offsetTensor.copy_(t_blockNew_offsetTensor1);
    torch::Tensor t_valueTensor = torch::empty_like(t_valueTensor1);
    t_valueTensor.copy_(t_valueTensor1);
    torch::Tensor t_columnTensor = torch::empty_like(t_columnTensor1);
    t_columnTensor.copy_(t_columnTensor1);
    torch::Tensor t_rowTensor = torch::empty_like(t_rowTensor1);
    t_rowTensor.copy_(t_rowTensor1);

    auto t_windowNew_offsetTensor2 = torch::from_blob(t_windowNew_offset1.data(), t_windowNew_offset1.size(), torch::kInt32);
    auto t_blockNew_offsetTensor2 = torch::from_blob(t_blockNew_offset1.data(), t_blockNew_offset1.size(), torch::kInt32);
    auto t_valueTensor2 = torch::from_blob(t_valueNew1.data(), t_valueNew1.size(), torch::kFloat32);
    auto t_columnTensor2 = torch::from_blob(t_columnNew1.data(), t_columnNew1.size(), torch::kInt32);
    auto t_rowTensor2 = torch::from_blob(t_rowNew1.data(), t_rowNew1.size(), torch::kInt32);


    torch::Tensor t_windowNew_offsetTensor_2 = torch::empty_like(t_windowNew_offsetTensor2);
    t_windowNew_offsetTensor_2.copy_(t_windowNew_offsetTensor2);
    torch::Tensor t_blockNew_offsetTensor_2 = torch::empty_like(t_blockNew_offsetTensor2);
    t_blockNew_offsetTensor_2.copy_(t_blockNew_offsetTensor2);
    torch::Tensor t_valueTensor_2 = torch::empty_like(t_valueTensor2);
    t_valueTensor_2.copy_(t_valueTensor2);
    torch::Tensor t_columnTensor_2 = torch::empty_like(t_columnTensor2);
    t_columnTensor_2.copy_(t_columnTensor2);
    torch::Tensor t_rowTensor_2 = torch::empty_like(t_rowTensor2);
    t_rowTensor_2.copy_(t_rowTensor2);
    return {t_windowNew_offsetTensor, t_blockNew_offsetTensor, t_columnTensor, t_valueTensor,t_rowTensor,
    t_windowNew_offsetTensor_2, t_blockNew_offsetTensor_2, t_columnTensor_2, t_valueTensor_2,t_rowTensor_2};

}
std::vector<torch::Tensor> block_csr_binary_tcu_kernel(torch::Tensor row1, torch::Tensor column1, torch::Tensor values1, int partSize, int window1, int wide1)
{
    auto *row=row1.data_ptr<int>();
    auto column=column1.accessor<int, 1>();
    auto values=values1.accessor<float, 1>();
    int rows=row1.size(0)-1;
    int window = window1;
    int wide = wide1;
    int warps_block = 4;
    //需要的window数量
    int num_windows = rows/window;
    //最终的map
    std::map<int, Value1_5> res;
    int tcu_rows=0;
    //统计vector的稠密度，然后按照降序排列
    #pragma omp parallel for
    for(int i=0;i<num_windows;i++)
    {
        Value1_5 v;
        //对column进行合并
        std::vector<int> mergedVector(&column[row[i*window]], &column[row[i*window+window]]);
        
        // 使用 unordered_map 统计 vector 中每个元素出现次数
        std::unordered_map<int, int> elementCounts;

        for (const auto& element : mergedVector) {
            elementCounts[element]++;
        }
        // 将统计结果放入 vector<pair> 中
        std::vector<std::pair<int, int>> countVector(elementCounts.begin(), elementCounts.end());

        // 按元素个数降序排序
        // std::sort(countVector.begin(), countVector.end(), [](const auto& a, const auto& b) {
        //     return a.second > b.second;
        // });
        //获取countVector的长度
        int size = countVector.size();
        int residue = size%wide;
        //统计每个block中的非零元数量，计算稠密度
        int percen = size; 
        //当前window中有几个block
        v.t_window_offset = percen/wide;
        if(residue>0) v.t_window_offset+=1;
        if(v.t_window_offset>0)
        {   
            std::vector<std::vector<float>> t_value_temp = createVectorOfVectors_float(v.t_window_offset);
            std::vector<long> t_binary_temp = createVectorOfVectors_v2(v.t_window_offset);

            // 将前 k 个数按升序排列
            std::sort(countVector.begin(), countVector.end());

            #pragma omp critical  
            {tcu_rows+=window;}
            // 记录当前window
            v.t_window = i;
            // v.tcu_flag = true;

            std::map<int, int> colmap;
            int c=0;
            for(auto col:countVector){
                v.t_column.push_back(col.first);
                colmap[col.first]=c++;
            }
            //把额外的1放进去
            if((wide-residue)>0){
                for(int j=0; j<(wide-residue); j++)
                 v.t_column.push_back(-1);
            }

            int cur_row = 0;
            // 遍历window中的每一行
            for(int j=i*window; j<(i+1)*window; j++)
            {
                // 遍历行中的每个元素
                for(int m=row[j];m<row[j+1];m++)
                {
                    int newCol = colmap[column[m]];
                    int block = newCol/wide;
                    int bInId= newCol%wide;
                    //TCU 计算的数据
                    //存储按wide列为一个block存储
                    t_value_temp[block].push_back(values[m]);
                    long offset = cur_row*wide + bInId;
                    long a=1;
                    t_binary_temp[block] |= a<<offset;
                    // t_binary_temp[block] += (1<<offset);
                }
                cur_row++;
            }
            //整合vector
            v.t_value = mergeVectors_float(t_value_temp);
            v.t_binary = t_binary_temp;
            v.t_block_offset = countVectors(t_value_temp);
        }

        #pragma omp critical  
        {  res[i]=v;}
    }
    // std::cout << "tcu rows:" << tcu_rows << std::endl;
    //记录每个window中的vector数量
    std::vector<int> t_windowNew_offset;
    t_windowNew_offset.push_back(0);
    std::vector<int> t_blockNew_offset;
    t_blockNew_offset.push_back(0);
    std::vector<float> t_valueNew;
    std::vector<int> t_columnNew;
    std::vector<long> t_binaryNew;


    //按顺序整合res
    for (const auto& pair : res) {
        // if(pair.second.tcu_flag)
        // {
            //首先整合TCU的数据
            t_windowNew_offset.push_back(t_windowNew_offset.back() + pair.second.t_window_offset);  
            for(int sub : pair.second.t_block_offset)
            {
                t_blockNew_offset.push_back(t_blockNew_offset.back()+sub);
            }  
            t_valueNew.insert(t_valueNew.end(),pair.second.t_value.begin(),pair.second.t_value.end());
            t_columnNew.insert(t_columnNew.end(),pair.second.t_column.begin(),pair.second.t_column.end());
            t_binaryNew.insert(t_binaryNew.end(),pair.second.t_binary.begin(),pair.second.t_binary.end());
        // }
    }

    auto t_windowNew_offsetTensor1 = torch::from_blob(t_windowNew_offset.data(), t_windowNew_offset.size(), torch::kInt32);
    auto t_blockNew_offsetTensor1 = torch::from_blob(t_blockNew_offset.data(), t_blockNew_offset.size(), torch::kInt32);
    auto t_valueTensor1 = torch::from_blob(t_valueNew.data(), t_valueNew.size(), torch::kFloat32);
    auto t_columnTensor1 = torch::from_blob(t_columnNew.data(), t_columnNew.size(), torch::kInt32);
    auto t_binaryTensor1 = torch::from_blob(t_binaryNew.data(), t_binaryNew.size(), torch::kInt64);
    // auto t_window_rowTensor1 = torch::from_blob(t_window_rowNew.data(), t_window_rowNew.size(), torch::kInt32);

    torch::Tensor t_windowNew_offsetTensor = torch::empty_like(t_windowNew_offsetTensor1);
    t_windowNew_offsetTensor.copy_(t_windowNew_offsetTensor1);
    torch::Tensor t_blockNew_offsetTensor = torch::empty_like(t_blockNew_offsetTensor1);
    t_blockNew_offsetTensor.copy_(t_blockNew_offsetTensor1);
    torch::Tensor t_valueTensor = torch::empty_like(t_valueTensor1);
    t_valueTensor.copy_(t_valueTensor1);
    torch::Tensor t_columnTensor = torch::empty_like(t_columnTensor1);
    t_columnTensor.copy_(t_columnTensor1);
    torch::Tensor t_binaryTensor = torch::empty_like(t_binaryTensor1);
    t_binaryTensor.copy_(t_binaryTensor1);

    // torch::Tensor t_window_rowTensor = torch::empty_like(t_window_rowTensor1);
    // t_window_rowTensor.copy_(t_window_rowTensor1);

    return {t_windowNew_offsetTensor, t_blockNew_offsetTensor, t_columnTensor, t_valueTensor,t_binaryTensor};

}

// m-tcu + cuda + partSize v3 
std::vector<torch::Tensor> block_tf32_tcu_cuda_csr_kernel(torch::Tensor row1, torch::Tensor column1, torch::Tensor values1, 
int partSize_t, int partSize_c, int shortSize, int density, int window, int wide)
{
    auto *row=row1.data_ptr<int>();
    auto column=column1.accessor<int, 1>();
    auto values=values1.accessor<float, 1>();
    int rows=row1.size(0)-1;
    // Result
    std::map<int, Value_tcu_csr> res_t;
    std::map<int, std::vector<Value_cuda>> res_c;
    std::map<int, std::vector<Value_cuda>> res_c_residue;
    //需要的window数量
    int num_windows = rows/window;
    #pragma omp parallel for
    for(int i=0;i<num_windows;i++)
    {
        /*
            1. 统计当前window中的vector, 并按照稠密度降序排列
        */
        //对column进行合并
        std::vector<int> mergedVector(&column[row[i*window]], &column[row[i*window+window]]);
        // 使用 unordered_map 统计 vector 中每个元素出现次数
        std::unordered_map<int, int> elementCounts;
        for (const auto& element : mergedVector) elementCounts[element]++;
        // 将统计结果放入 vector<pair> 中
        std::vector<std::pair<int, int>> countVector(elementCounts.begin(), elementCounts.end());
        // 按元素个数降序排序
        std::sort(countVector.begin(), countVector.end(), [](const auto& a, const auto& b) {
            return a.second > b.second;
        });

        /*
            2. 确定需要进行tcu计算的vector数量, t_percen
        */
        int size = countVector.size();
        int residue = size%wide;
        //统计每个block中的非零元数量，计算稠密度
        int t_percen = 0; //确定前多少个vector会被用作tcu进行稠密计算
        t_percen = std::count_if(countVector.begin(), countVector.end(), [density](const auto& pair) {
            return pair.second >= density;
        });
        if((t_percen%wide) != 0)
        {
            if((t_percen + (wide - t_percen%wide)) <= size) 
            t_percen = (t_percen + (wide - t_percen%wide));
            else
            t_percen = (t_percen - (t_percen%wide));

        }
        // 将前 k 个数按升序排列
        std::sort(countVector.begin(), countVector.begin() + t_percen);
        // 将剩下的数按升序排列
        std::sort(countVector.begin() + t_percen, countVector.end());
        /*
            3. 根据t_percen分别划分TCU和CUDA的负载
        */

        // TCU 划分
        // 3.1 根据partSize_t划分t_part
        //当前window中有几个block
        int blocks = t_percen/wide; 
        Value_tcu_csr v_tcu;
        if(blocks>0) {v_tcu.tcu_flag = true; }
        std::map<int, int> colmap;
        int c=0;
        for(auto col:countVector){
            if (c<t_percen)
            {
                v_tcu.t_column.push_back(col.first);
            }
            colmap[col.first]=c++;
        }
        //存放拆分的cuda长短行数据
        std::vector<Value_cuda> large;
        std::vector<Value_cuda> small;

        std::vector<std::vector<float>> t_value_temp = createVectorOfVectors_float(blocks);
        std::vector<std::vector<int>> t_row_temp = createVectorOfVectors(blocks);
        std::vector<std::vector<int>> t_col_temp = createVectorOfVectors(blocks);
        // 指示快内的行偏移
        int p=0;
        for(int j=i*window;j<(i+1)*window;j++)
        {
            //用来存储每行的cuda数据
            std::queue<int> c_col;
            std::queue<int> c_value;
            int cuda_nnz=0;
            for(int m=row[j];m<row[j+1];m++)
            {
                int newCol = colmap[column[m]];
                int block = newCol/wide;
                if(newCol<t_percen)
                {
                    // int bId=newCol/wide;
                    // int bInId=newCol%wide;
                    t_value_temp[block].push_back(values[m]);
                    t_row_temp[block].push_back(p);
                    t_col_temp[block].push_back(newCol%wide);
                }else{
                    //划分cuda部分
                    c_col.push(column[m]);
                    c_value.push(values[m]);
                    cuda_nnz++;
                }
            }
            //处理当前行剩余的cuda元素
            if(cuda_nnz>0)
            {
                if(cuda_nnz <= shortSize)
                {
                    Value_cuda temp;
                    temp.cuda_residue_flag = true;
                    temp.c_row_offset = cuda_nnz;
                    temp.c_row = i*window + p;
                    temp.c_atomic = 0;
                    for(int m=0; m<cuda_nnz; m++)
                    {
                        temp.c_column.push_back(c_col.front());
                        temp.c_value.push_back(c_value.front());
                        c_col.pop();
                        c_value.pop();
                    }
                    small.push_back(temp);
                }else{
                    if(cuda_nnz <= partSize_c)
                    {
                        Value_cuda temp;
                        temp.cuda_flag = true;
                        temp.c_row_offset = (cuda_nnz);
                        temp.c_row = i*window + p;
                        temp.c_atomic = 0;
                        for(int m=0; m<cuda_nnz; m++)
                        {
                            temp.c_column.push_back(c_col.front());
                            temp.c_value.push_back(c_value.front());
                            c_col.pop();
                            c_value.pop();
                        }
                        large.push_back(temp);
                    }else{
                        // 当前行的非零元被分成多个part，并需要原子操作
                        int parts_row = cuda_nnz/partSize_c;
                        // int cur_value_indice = row[i];
                        for(int cur_i = 0; cur_i<parts_row; cur_i++)
                        {
                            Value_cuda temp;
                            temp.cuda_flag = true;      
                            temp.c_row_offset = partSize_c;
                            temp.c_row = i*window + p;
                            temp.c_atomic = 1;
                            for(int m=0; m<partSize_c; m++)
                            {
                                temp.c_column.push_back(c_col.front());
                                temp.c_value.push_back(c_value.front());
                                c_col.pop();
                                c_value.pop();
                                // cur_value_indice++;
                            }
                            large.push_back(temp);
                        }
                        int residue  = (cuda_nnz%partSize_c);
                        if(residue>0){
                            Value_cuda temp;
                            temp.c_row_offset = residue;
                            temp.c_row = i*window + p;
                            temp.c_atomic = 1;
                            for(int m=0; m<residue; m++)
                            {
                                temp.c_column.push_back(c_col.front());
                                temp.c_value.push_back(c_value.front());
                                c_col.pop();
                                c_value.pop();
                                // cur_value_indice++;
                            }
                            if(residue <= shortSize)
                            {
                                temp.cuda_residue_flag = true;
                                small.push_back(temp);
                            }else{
                                temp.cuda_flag = true;
                                large.push_back(temp);
                            }
                        }
                    }
                }
            }
            p++;
        }
        //整合vector
        v_tcu.t_value = mergeVectors_float(t_value_temp);
        v_tcu.t_row = mergeVectors(t_row_temp);
        v_tcu.t_col = mergeVectors(t_col_temp);
        
        std::vector<int> count_tcu = countVectors(t_value_temp);
        if(blocks <= partSize_t)
        {
            v_tcu.row.push_back(blocks);
            //循放入每个block中的非零元数量
            for(int b=0; b<blocks; b++)
            {
                // int sum = std::accumulate(count_tcu.begin()+b, count_tcu.begin()+b+1, 0);
                v_tcu.t_block_offset.push_back(count_tcu[b]);
            }
            v_tcu.atomic.push_back(0);
            v_tcu.window.push_back(i);
        }else{
            // 对block太多的块进行分割
            int part_number = blocks / partSize_t;
            int block_residue = blocks % partSize_t;
            for(int j=0; j<part_number; j++)
            {
                v_tcu.row.push_back(partSize_t);
                for(int b=0; b<partSize_t; b++)
                {
                    int offset = j*partSize_t + b;
                    // int offset1 = offset + b;
                    // int sum = std::accumulate(count_tcu.begin()+offset, count_tcu.begin()+offset1, 0);
                    v_tcu.t_block_offset.push_back(count_tcu[offset]);
                }
                v_tcu.window.push_back(i);
                v_tcu.atomic.push_back(1);
            }
            if(block_residue > 0)
            {             
                v_tcu.row.push_back(block_residue);
                for(int b=0; b<block_residue; b++)
                {
                    int offset = part_number*partSize_t+b;
                    // int offset1 = offset + b;
                    // int sum = std::accumulate(count_tcu.begin()+offset, count_tcu.begin()+offset1, 0);
                    v_tcu.t_block_offset.push_back(count_tcu[offset]);
                }
                v_tcu.window.push_back(i);
                v_tcu.atomic.push_back(1);
            }
        }
        /*
            4. 汇总三部分的结果
        */
        #pragma omp critical  
        res_t[i]=v_tcu;

        #pragma omp critical  
        res_c[i]=large;

        #pragma omp critical  
        res_c_residue[i]=small;
        
    
    }
    /*
        5. 整合所有window的数据
    */
    //tcu
    std::vector<int> t_rowkNew_offset;
    t_rowkNew_offset.push_back(0);
    std::vector<int> t_blockNew_offset;
    t_blockNew_offset.push_back(0);
    std::vector<float> t_valueNew;
    std::vector<int> t_columnNew;
    std::vector<int> t_rowNew;
    std::vector<int> t_colNew;
    std::vector<int> t_window_rowNew;
    std::vector<int> t_atomicNew;

    //cuda 长行
    std::vector<int> c_rowNew_offset;
    c_rowNew_offset.push_back(0);
    std::vector<int> c_rowNew;
    std::vector<int> c_atomicNew;
    std::vector<int> c_columnNew;
    std::vector<float> c_valueNew;

    //cuda 短行
    std::vector<int> c_rowNew_offset_residue;
    c_rowNew_offset_residue.push_back(0);
    std::vector<int> c_rowNew_residue;
    std::vector<int> c_atomicNew_residue;
    std::vector<int> c_columnNew_residue;
    std::vector<float> c_valueNew_residue;

    //按顺序整合res
    //tcu
    for (const auto& pair : res_t) {
        if (pair.second.tcu_flag == true )
        {
            for(int sub : pair.second.row)
            {
                t_rowkNew_offset.push_back(t_rowkNew_offset.back()+sub);
            }  
            for(int sub : pair.second.t_block_offset)
            {
                t_blockNew_offset.push_back(t_blockNew_offset.back()+sub);
            }  
            t_valueNew.insert(t_valueNew.end(),pair.second.t_value.begin(),pair.second.t_value.end());
            t_columnNew.insert(t_columnNew.end(),pair.second.t_column.begin(),pair.second.t_column.end());
            t_window_rowNew.insert(t_window_rowNew.end(),pair.second.window.begin(),pair.second.window.end());
            t_atomicNew.insert(t_atomicNew.end(),pair.second.atomic.begin(),pair.second.atomic.end());
            t_rowNew.insert(t_rowNew.end(),pair.second.t_row.begin(),pair.second.t_row.end());
            t_colNew.insert(t_colNew.end(),pair.second.t_col.begin(),pair.second.t_col.end());
        }   
    }
    // std::cout << t_rowkNew_offset.size() <<std::endl;
    auto t_rowNew_offsetTensor1 = torch::from_blob(t_rowkNew_offset.data(), t_rowkNew_offset.size(), torch::kInt32);
    auto t_blockNew_offsetTensor1 = torch::from_blob(t_blockNew_offset.data(), t_blockNew_offset.size(), torch::kInt32);
    auto t_valueTensor1 = torch::from_blob(t_valueNew.data(), t_valueNew.size(), torch::kFloat32);
    auto t_columnTensor1 = torch::from_blob(t_columnNew.data(), t_columnNew.size(), torch::kInt32);
    auto t_window_rowTensor1 = torch::from_blob(t_window_rowNew.data(), t_window_rowNew.size(), torch::kInt32);
    auto t_atomicTensor1 = torch::from_blob(t_atomicNew.data(), t_atomicNew.size(), torch::kInt32);
    auto t_rowTensor1 = torch::from_blob(t_rowNew.data(), t_rowNew.size(), torch::kInt32);
    auto t_colTensor1 = torch::from_blob(t_colNew.data(), t_colNew.size(), torch::kInt32);

    torch::Tensor t_rowNew_offsetTensor = torch::empty_like(t_rowNew_offsetTensor1);
    t_rowNew_offsetTensor.copy_(t_rowNew_offsetTensor1);
    torch::Tensor t_blockNew_offsetTensor = torch::empty_like(t_blockNew_offsetTensor1);
    t_blockNew_offsetTensor.copy_(t_blockNew_offsetTensor1);
    torch::Tensor t_valueTensor = torch::empty_like(t_valueTensor1);
    t_valueTensor.copy_(t_valueTensor1);
    torch::Tensor t_columnTensor = torch::empty_like(t_columnTensor1);
    t_columnTensor.copy_(t_columnTensor1);
    torch::Tensor t_window_rowTensor = torch::empty_like(t_window_rowTensor1);
    t_window_rowTensor.copy_(t_window_rowTensor1);
    torch::Tensor t_atomicTensor = torch::empty_like(t_atomicTensor1);
    t_atomicTensor.copy_(t_atomicTensor1);
    torch::Tensor t_rowTensor = torch::empty_like(t_rowTensor1);
    t_rowTensor.copy_(t_rowTensor1);
    torch::Tensor t_colTensor = torch::empty_like(t_colTensor1);
    t_colTensor.copy_(t_colTensor1);

    //cuda long
    for (const auto& pair : res_c) {
        for( Value_cuda sub : pair.second){
            if (sub.cuda_flag == true )
            {c_rowNew_offset.push_back(c_rowNew_offset.back()+sub.c_row_offset);
            c_rowNew.push_back(sub.c_row);
            c_atomicNew.push_back(sub.c_atomic);
            c_columnNew.insert(c_columnNew.end(),sub.c_column.begin(),sub.c_column.end());
            c_valueNew.insert(c_valueNew.end(),sub.c_value.begin(), sub.c_value.end());}
        }
    }

    auto c_row_offsetTensor1 = torch::from_blob(c_rowNew_offset.data(), c_rowNew_offset.size(), torch::kInt32);
    auto c_rowTensor1 = torch::from_blob(c_rowNew.data(), c_rowNew.size(), torch::kInt32);
    auto c_atomicTensor1 = torch::from_blob(c_atomicNew.data(), c_atomicNew.size(), torch::kInt32);
    auto c_colTensor1 = torch::from_blob(c_columnNew.data(), c_columnNew.size(), torch::kInt32);
    auto c_valueTensor1 = torch::from_blob(c_valueNew.data(), c_valueNew.size(), torch::kFloat32);
    
    torch::Tensor c_row_offsetTensor = torch::empty_like(c_row_offsetTensor1);
    c_row_offsetTensor.copy_(c_row_offsetTensor1);
    torch::Tensor c_rowTensor = torch::empty_like(c_rowTensor1);
    c_rowTensor.copy_(c_rowTensor1);
    torch::Tensor c_atomicTensor = torch::empty_like(c_atomicTensor1);
    c_atomicTensor.copy_(c_atomicTensor1);
    torch::Tensor c_colTensor = torch::empty_like(c_colTensor1);
    c_colTensor.copy_(c_colTensor1);
    torch::Tensor c_valueTensor = torch::empty_like(c_valueTensor1);
    c_valueTensor.copy_(c_valueTensor1);
    
    //cuda short
    for (const auto& pair : res_c_residue) {
        for( Value_cuda sub : pair.second){
            if (sub.cuda_residue_flag == true )
            {c_rowNew_offset_residue.push_back(c_rowNew_offset_residue.back()+sub.c_row_offset);
            c_rowNew_residue.push_back(sub.c_row);
            c_atomicNew_residue.push_back(sub.c_atomic);
            c_columnNew_residue.insert(c_columnNew_residue.end(),sub.c_column.begin(),sub.c_column.end());
            c_valueNew_residue.insert(c_valueNew_residue.end(),sub.c_value.begin(), sub.c_value.end());}
        }
    }

    auto c_row_offsetTensor1_residue = torch::from_blob(c_rowNew_offset_residue.data(), c_rowNew_offset_residue.size(), torch::kInt32);
    auto c_rowTensor1_residue = torch::from_blob(c_rowNew_residue.data(), c_rowNew_residue.size(), torch::kInt32);
    auto c_atomicTensor1_residue = torch::from_blob(c_atomicNew_residue.data(), c_atomicNew_residue.size(), torch::kInt32);
    auto c_colTensor1_residue = torch::from_blob(c_columnNew_residue.data(), c_columnNew_residue.size(), torch::kInt32);
    auto c_valueTensor1_residue = torch::from_blob(c_valueNew_residue.data(), c_valueNew_residue.size(), torch::kFloat32);
    
    torch::Tensor c_row_offsetTensor_residue = torch::empty_like(c_row_offsetTensor1_residue);
    c_row_offsetTensor_residue.copy_(c_row_offsetTensor1_residue);
    torch::Tensor c_rowTensor_residue = torch::empty_like(c_rowTensor1_residue);
    c_rowTensor_residue.copy_(c_rowTensor1_residue);
    torch::Tensor c_atomicTensor_residue = torch::empty_like(c_atomicTensor1_residue);
    c_atomicTensor_residue.copy_(c_atomicTensor1_residue);
    torch::Tensor c_colTensor_residue = torch::empty_like(c_colTensor1_residue);
    c_colTensor_residue.copy_(c_colTensor1_residue);
    torch::Tensor c_valueTensor_residue = torch::empty_like(c_valueTensor1_residue);
    c_valueTensor_residue.copy_(c_valueTensor1_residue);

    return {t_rowNew_offsetTensor, t_blockNew_offsetTensor,t_columnTensor, t_valueTensor, t_window_rowTensor, t_atomicTensor, t_rowTensor, t_colTensor,
            c_row_offsetTensor, c_rowTensor, c_atomicTensor, c_colTensor, c_valueTensor,
            c_row_offsetTensor_residue, c_rowTensor_residue, c_atomicTensor_residue, c_colTensor_residue, c_valueTensor_residue};

}
// m-tcu + cuda + partSize + binary
std::vector<torch::Tensor> block_tf32_tcu_cuda_binary_kernel(torch::Tensor row1, torch::Tensor column1, torch::Tensor values1, 
int partSize_t, int partSize_c, int shortSize, int density, int window, int wide)
{
    auto *row=row1.data_ptr<int>();
    auto column=column1.accessor<int, 1>();
    auto values=values1.accessor<float, 1>();

    auto start = std::chrono::high_resolution_clock::now();

    int rows=row1.size(0)-1;
    // Result
    std::map<int, Value_tcu_binary> res_t;
    std::map<int, std::vector<Value_cuda>> res_c;
    std::map<int, std::vector<Value_cuda>> res_c_residue;
    //需要的window数量
    int num_windows = rows/window;
    #pragma omp parallel for
    for(int i=0;i<num_windows;i++)
    {
        /*
            1. 统计当前window中的vector, 并按照稠密度降序排列
        */
        //对column进行合并
        std::vector<int> mergedVector(&column[row[i*window]], &column[row[i*window+window]]);
        // 使用 unordered_map 统计 vector 中每个元素出现次数
        std::unordered_map<int, int> elementCounts;
        for (const auto& element : mergedVector) elementCounts[element]++;
        // 将统计结果放入 vector<pair> 中
        std::vector<std::pair<int, int>> countVector(elementCounts.begin(), elementCounts.end());
        // 按元素个数降序排序
        std::sort(countVector.begin(), countVector.end(), [](const auto& a, const auto& b) {
            return a.second > b.second;
        });

        /*
            2. 确定需要进行tcu计算的vector数量, t_percen
        */
        int size = countVector.size();
        int residue = size%wide;
        //统计每个block中的非零元数量，计算稠密度
        int t_percen = 0; //确定前多少个vector会被用作tcu进行稠密计算
        t_percen = std::count_if(countVector.begin(), countVector.end(), [density](const auto& pair) {
            return pair.second >= density;
        });
        int t_percen_ori = t_percen;
        if((t_percen%wide) != 0)
        {
            // if((t_percen + (wide - t_percen%wide)) <= size) 
            t_percen = (t_percen + (wide - t_percen%wide));
            // else
            // t_percen = (t_percen - (t_percen%wide));

        }
        // if(t_percen<=size){
        // // 将前 k 个数按升序排列
        // std::sort(countVector.begin(), countVector.begin() + t_percen);
        // // 将剩下的数按升序排列
        // std::sort(countVector.begin() + t_percen, countVector.end());
        // }else{
        //     std::sort(countVector.begin(), countVector.end());
        // }
        std::sort(countVector.begin(), countVector.begin()+t_percen_ori);
        std::sort(countVector.begin() + t_percen_ori, countVector.end());
        /*
            3. 根据t_percen分别划分TCU和CUDA的负载
        */

        // TCU 划分
        // 3.1 根据partSize_t划分t_part
        //当前window中有几个block
        int blocks = t_percen/wide; 
        Value_tcu_binary v_tcu;
        if(blocks>0) {v_tcu.tcu_flag = true; }
        std::map<int, int> colmap;
        int c=0;
        for(auto col:countVector){
            if (c<t_percen_ori)
            {
                v_tcu.t_column.push_back(col.first);
            }
            colmap[col.first]=c++;
        }
        if((t_percen_ori%wide)>0){
            for(int tt=0;tt<(wide-(t_percen_ori%wide));tt++)
            v_tcu.t_column.push_back(-1);
        }
        //存放拆分的cuda长短行数据
        std::vector<Value_cuda> large;
        std::vector<Value_cuda> small;

        std::vector<std::vector<float>> t_value_temp = createVectorOfVectors_float(blocks);
        std::vector<long> t_binary_temp = createVectorOfVectors_v2(blocks);
        // std::vector<std::vector<int>> t_row_temp = createVectorOfVectors(blocks);
        // std::vector<std::vector<int>> t_col_temp = createVectorOfVectors(blocks);
        // 指示快内的行偏移
        int p=0;
        for(int j=i*window;j<(i+1)*window;j++)
        {
            //用来存储每行的cuda数据
            std::queue<int> c_col;
            std::queue<int> c_value;
            int cuda_nnz=0;
            for(int m=row[j];m<row[j+1];m++)
            {
                int newCol = colmap[column[m]];
                if(newCol<t_percen_ori)
                {
                    int block = newCol/wide;
                    int bInId= newCol%wide;
                    t_value_temp[block].push_back(values[m]);
                    long offset = p*wide + bInId;
                    long a=1;
                    t_binary_temp[block] |= a<<offset;
                }else{
                    //划分cuda部分
                    c_col.push(column[m]);
                    c_value.push(values[m]);
                    cuda_nnz++;
                }
            }
            //处理当前行剩余的cuda元素
            if(cuda_nnz>0)
            {
                if(cuda_nnz <= shortSize)
                {
                    Value_cuda temp;
                    temp.cuda_residue_flag = true;
                    temp.c_row_offset = cuda_nnz;
                    temp.c_row = i*window + p;
                    if(t_percen_ori>0) temp.c_atomic = 1;
                    else temp.c_atomic = 0;
                    for(int m=0; m<cuda_nnz; m++)
                    {
                        temp.c_column.push_back(c_col.front());
                        temp.c_value.push_back(c_value.front());
                        c_col.pop();
                        c_value.pop();
                    }
                    small.push_back(temp);
                }else{
                    if(cuda_nnz <= partSize_c)
                    {
                        Value_cuda temp;
                        temp.cuda_flag = true;
                        temp.c_row_offset = (cuda_nnz);
                        temp.c_row = i*window + p;
                        if(t_percen_ori>0) temp.c_atomic = 1;
                        else temp.c_atomic = 0;
                        for(int m=0; m<cuda_nnz; m++)
                        {
                            temp.c_column.push_back(c_col.front());
                            temp.c_value.push_back(c_value.front());
                            c_col.pop();
                            c_value.pop();
                        }
                        large.push_back(temp);
                    }else{
                        // 当前行的非零元被分成多个part，并需要原子操作
                        int parts_row = cuda_nnz/partSize_c;
                        // int cur_value_indice = row[i];
                        for(int cur_i = 0; cur_i<parts_row; cur_i++)
                        {
                            Value_cuda temp;
                            temp.cuda_flag = true;      
                            temp.c_row_offset = partSize_c;
                            temp.c_row = i*window + p;
                            temp.c_atomic = 1;
                            for(int m=0; m<partSize_c; m++)
                            {
                                temp.c_column.push_back(c_col.front());
                                temp.c_value.push_back(c_value.front());
                                c_col.pop();
                                c_value.pop();
                                // cur_value_indice++;
                            }
                            large.push_back(temp);
                        }
                        int residue  = (cuda_nnz%partSize_c);
                        if(residue>0){
                            Value_cuda temp;
                            temp.c_row_offset = residue;
                            temp.c_row = i*window + p;
                            temp.c_atomic = 1;
                            for(int m=0; m<residue; m++)
                            {
                                temp.c_column.push_back(c_col.front());
                                temp.c_value.push_back(c_value.front());
                                c_col.pop();
                                c_value.pop();
                                // cur_value_indice++;
                            }
                            if(residue <= shortSize)
                            {
                                temp.cuda_residue_flag = true;
                                small.push_back(temp);
                            }else{
                                temp.cuda_flag = true;
                                large.push_back(temp);
                            }
                        }
                    }
                }
            }
            p++;
        }
        //整合vector
        v_tcu.t_value = mergeVectors_float(t_value_temp);
        v_tcu.t_binary = t_binary_temp;
        // v_tcu.t_row = mergeVectors(t_row_temp);
        // v_tcu.t_col = mergeVectors(t_col_temp);
        
        std::vector<int> count_tcu = countVectors(t_value_temp);
        if(blocks <= partSize_t)
        {
            v_tcu.row.push_back(blocks);
            //循放入每个block中的非零元数量
            for(int b=0; b<blocks; b++)
            {
                // int sum = std::accumulate(count_tcu.begin()+b, count_tcu.begin()+b+1, 0);
                v_tcu.t_block_offset.push_back(count_tcu[b]);
            }
            if(t_percen_ori==size) v_tcu.atomic.push_back(0);
            else v_tcu.atomic.push_back(1);
            v_tcu.window.push_back(i);
        }else{
            // 对block太多的块进行分割
            int part_number = blocks / partSize_t;
            int block_residue = blocks % partSize_t;
            for(int j=0; j<part_number; j++)
            {
                v_tcu.row.push_back(partSize_t);
                for(int b=0; b<partSize_t; b++)
                {
                    int offset = j*partSize_t + b;
                    // int offset1 = offset + b;
                    // int sum = std::accumulate(count_tcu.begin()+offset, count_tcu.begin()+offset1, 0);
                    v_tcu.t_block_offset.push_back(count_tcu[offset]);
                }
                v_tcu.window.push_back(i);
                v_tcu.atomic.push_back(1);
            }
            if(block_residue > 0)
            {             
                v_tcu.row.push_back(block_residue);
                for(int b=0; b<block_residue; b++)
                {
                    int offset = part_number*partSize_t+b;
                    // int offset1 = offset + b;
                    // int sum = std::accumulate(count_tcu.begin()+offset, count_tcu.begin()+offset1, 0);
                    v_tcu.t_block_offset.push_back(count_tcu[offset]);
                }
                v_tcu.window.push_back(i);
                v_tcu.atomic.push_back(1);
            }
        }
        /*
            4. 汇总三部分的结果
        */
        #pragma omp critical  
        res_t[i]=v_tcu;

        #pragma omp critical  
        res_c[i]=large;

        #pragma omp critical  
        res_c_residue[i]=small;
        
    
    }
    /*
        5. 整合所有window的数据
    */
    //tcu
    std::vector<int> t_rowkNew_offset;
    t_rowkNew_offset.push_back(0);
    std::vector<int> t_blockNew_offset;
    t_blockNew_offset.push_back(0);
    std::vector<float> t_valueNew;
    std::vector<int> t_columnNew;
    std::vector<long> t_binaryNew;
    std::vector<int> t_window_rowNew;
    std::vector<int> t_atomicNew;

    //cuda 长行
    std::vector<int> c_rowNew_offset;
    c_rowNew_offset.push_back(0);
    std::vector<int> c_rowNew;
    std::vector<int> c_atomicNew;
    std::vector<int> c_columnNew;
    std::vector<float> c_valueNew;

    //cuda 短行
    std::vector<int> c_rowNew_offset_residue;
    c_rowNew_offset_residue.push_back(0);
    std::vector<int> c_rowNew_residue;
    std::vector<int> c_atomicNew_residue;
    std::vector<int> c_columnNew_residue;
    std::vector<float> c_valueNew_residue;

    //按顺序整合res
    //tcu
    for (const auto& pair : res_t) {
        if (pair.second.tcu_flag == true )
        {
            for(int sub : pair.second.row)
            {
                t_rowkNew_offset.push_back(t_rowkNew_offset.back()+sub);
            }  
            for(int sub : pair.second.t_block_offset)
            {
                t_blockNew_offset.push_back(t_blockNew_offset.back()+sub);
            }  
            t_valueNew.insert(t_valueNew.end(),pair.second.t_value.begin(),pair.second.t_value.end());
            t_columnNew.insert(t_columnNew.end(),pair.second.t_column.begin(),pair.second.t_column.end());
            t_window_rowNew.insert(t_window_rowNew.end(),pair.second.window.begin(),pair.second.window.end());
            t_atomicNew.insert(t_atomicNew.end(),pair.second.atomic.begin(),pair.second.atomic.end());
            t_binaryNew.insert(t_binaryNew.end(),pair.second.t_binary.begin(),pair.second.t_binary.end());
        }   
    }

    //cuda long
    for (const auto& pair : res_c) {
        for( Value_cuda sub : pair.second){
            if (sub.cuda_flag == true )
            {c_rowNew_offset.push_back(c_rowNew_offset.back()+sub.c_row_offset);
            c_rowNew.push_back(sub.c_row);
            c_atomicNew.push_back(sub.c_atomic);
            c_columnNew.insert(c_columnNew.end(),sub.c_column.begin(),sub.c_column.end());
            c_valueNew.insert(c_valueNew.end(),sub.c_value.begin(), sub.c_value.end());}
        }
    }

    //cuda short
    for (const auto& pair : res_c_residue) {
        for( Value_cuda sub : pair.second){
            if (sub.cuda_residue_flag == true )
            {c_rowNew_offset_residue.push_back(c_rowNew_offset_residue.back()+sub.c_row_offset);
            c_rowNew_residue.push_back(sub.c_row);
            c_atomicNew_residue.push_back(sub.c_atomic);
            c_columnNew_residue.insert(c_columnNew_residue.end(),sub.c_column.begin(),sub.c_column.end());
            c_valueNew_residue.insert(c_valueNew_residue.end(),sub.c_value.begin(), sub.c_value.end());}
        }
    }

    auto end = std::chrono::high_resolution_clock::now();

    // 计算耗时
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;

    auto t_rowNew_offsetTensor1 = torch::from_blob(t_rowkNew_offset.data(), t_rowkNew_offset.size(), torch::kInt32);
    auto t_blockNew_offsetTensor1 = torch::from_blob(t_blockNew_offset.data(), t_blockNew_offset.size(), torch::kInt32);
    auto t_valueTensor1 = torch::from_blob(t_valueNew.data(), t_valueNew.size(), torch::kFloat32);
    auto t_columnTensor1 = torch::from_blob(t_columnNew.data(), t_columnNew.size(), torch::kInt32);
    auto t_window_rowTensor1 = torch::from_blob(t_window_rowNew.data(), t_window_rowNew.size(), torch::kInt32);
    auto t_atomicTensor1 = torch::from_blob(t_atomicNew.data(), t_atomicNew.size(), torch::kInt32);
    auto t_binaryTensor1 = torch::from_blob(t_binaryNew.data(), t_binaryNew.size(), torch::kInt64);

    torch::Tensor t_rowNew_offsetTensor = torch::empty_like(t_rowNew_offsetTensor1);
    t_rowNew_offsetTensor.copy_(t_rowNew_offsetTensor1);
    torch::Tensor t_blockNew_offsetTensor = torch::empty_like(t_blockNew_offsetTensor1);
    t_blockNew_offsetTensor.copy_(t_blockNew_offsetTensor1);
    torch::Tensor t_valueTensor = torch::empty_like(t_valueTensor1);
    t_valueTensor.copy_(t_valueTensor1);
    torch::Tensor t_columnTensor = torch::empty_like(t_columnTensor1);
    t_columnTensor.copy_(t_columnTensor1);
    torch::Tensor t_window_rowTensor = torch::empty_like(t_window_rowTensor1);
    t_window_rowTensor.copy_(t_window_rowTensor1);
    torch::Tensor t_atomicTensor = torch::empty_like(t_atomicTensor1);
    t_atomicTensor.copy_(t_atomicTensor1);
    torch::Tensor t_binaryTensor = torch::empty_like(t_binaryTensor1);
    t_binaryTensor.copy_(t_binaryTensor1);


    auto c_row_offsetTensor1 = torch::from_blob(c_rowNew_offset.data(), c_rowNew_offset.size(), torch::kInt32);
    auto c_rowTensor1 = torch::from_blob(c_rowNew.data(), c_rowNew.size(), torch::kInt32);
    auto c_atomicTensor1 = torch::from_blob(c_atomicNew.data(), c_atomicNew.size(), torch::kInt32);
    auto c_colTensor1 = torch::from_blob(c_columnNew.data(), c_columnNew.size(), torch::kInt32);
    auto c_valueTensor1 = torch::from_blob(c_valueNew.data(), c_valueNew.size(), torch::kFloat32);
    
    torch::Tensor c_row_offsetTensor = torch::empty_like(c_row_offsetTensor1);
    c_row_offsetTensor.copy_(c_row_offsetTensor1);
    torch::Tensor c_rowTensor = torch::empty_like(c_rowTensor1);
    c_rowTensor.copy_(c_rowTensor1);
    torch::Tensor c_atomicTensor = torch::empty_like(c_atomicTensor1);
    c_atomicTensor.copy_(c_atomicTensor1);
    torch::Tensor c_colTensor = torch::empty_like(c_colTensor1);
    c_colTensor.copy_(c_colTensor1);
    torch::Tensor c_valueTensor = torch::empty_like(c_valueTensor1);
    c_valueTensor.copy_(c_valueTensor1);
    

    auto c_row_offsetTensor1_residue = torch::from_blob(c_rowNew_offset_residue.data(), c_rowNew_offset_residue.size(), torch::kInt32);
    auto c_rowTensor1_residue = torch::from_blob(c_rowNew_residue.data(), c_rowNew_residue.size(), torch::kInt32);
    auto c_atomicTensor1_residue = torch::from_blob(c_atomicNew_residue.data(), c_atomicNew_residue.size(), torch::kInt32);
    auto c_colTensor1_residue = torch::from_blob(c_columnNew_residue.data(), c_columnNew_residue.size(), torch::kInt32);
    auto c_valueTensor1_residue = torch::from_blob(c_valueNew_residue.data(), c_valueNew_residue.size(), torch::kFloat32);
    
    torch::Tensor c_row_offsetTensor_residue = torch::empty_like(c_row_offsetTensor1_residue);
    c_row_offsetTensor_residue.copy_(c_row_offsetTensor1_residue);
    torch::Tensor c_rowTensor_residue = torch::empty_like(c_rowTensor1_residue);
    c_rowTensor_residue.copy_(c_rowTensor1_residue);
    torch::Tensor c_atomicTensor_residue = torch::empty_like(c_atomicTensor1_residue);
    c_atomicTensor_residue.copy_(c_atomicTensor1_residue);
    torch::Tensor c_colTensor_residue = torch::empty_like(c_colTensor1_residue);
    c_colTensor_residue.copy_(c_colTensor1_residue);
    torch::Tensor c_valueTensor_residue = torch::empty_like(c_valueTensor1_residue);
    c_valueTensor_residue.copy_(c_valueTensor1_residue);

    return {t_rowNew_offsetTensor, t_blockNew_offsetTensor,t_columnTensor, t_valueTensor, t_window_rowTensor, t_atomicTensor, t_binaryTensor,
            c_row_offsetTensor, c_rowTensor, c_atomicTensor, c_colTensor, c_valueTensor,
            c_row_offsetTensor_residue, c_rowTensor_residue, c_atomicTensor_residue, c_colTensor_residue, c_valueTensor_residue, torch::tensor(elapsed.count())};

}

// m-tcu + cuda + partSize + binary
std::vector<torch::Tensor> block_tf32_tcu_cuda_binary_kernel_new(torch::Tensor row1, torch::Tensor column1, torch::Tensor values1, 
int partSize_t, int partSize_c, int shortSize, int density, int window, int wide, float ratio)
{
    auto *row=row1.data_ptr<int>();
    auto column=column1.accessor<int, 1>();
    auto values=values1.accessor<float, 1>();

    auto start = std::chrono::high_resolution_clock::now();

    int rows=row1.size(0)-1;
    // Result
    std::map<int, Value_tcu_binary> res_t;
    std::map<int, std::vector<Value_cuda>> res_c;
    std::map<int, std::vector<Value_cuda>> res_c_residue;
    //需要的window数量
    int num_windows = rows/window;
    #pragma omp parallel for
    for(int i=0;i<num_windows;i++)
    {
        /*
            1. 统计当前window中的vector, 并按照稠密度降序排列
        */
        //对column进行合并
        std::vector<int> mergedVector(&column[row[i*window]], &column[row[i*window+window]]);
        // 使用 unordered_map 统计 vector 中每个元素出现次数
        std::unordered_map<int, int> elementCounts;
        for (const auto& element : mergedVector) elementCounts[element]++;
        // 将统计结果放入 vector<pair> 中
        std::vector<std::pair<int, int>> countVector(elementCounts.begin(), elementCounts.end());
        // 按元素个数降序排序
        std::sort(countVector.begin(), countVector.end(), [](const auto& a, const auto& b) {
            return a.second > b.second;
        });

        /*
            2. 确定需要进行tcu计算的vector数量, t_percen
        */
        int size = countVector.size();
        int residue = size%wide;
        //统计每个block中的非零元数量，计算稠密度
        int t_percen = 0; //确定前多少个vector会被用作tcu进行稠密计算
        t_percen = std::count_if(countVector.begin(), countVector.end(), [density](const auto& pair) {
            return pair.second >= density;
        });
        // 用于存储和
        int sum0 = 0; // 满足条件的second之和
        int sum1 = 0; // 不满足条件的second之和

        // 统计sum0和sum1
        for (const auto& pair : countVector) {
            if (pair.second >= density) {
                sum0 += pair.second;  // 满足条件
            } else {
                sum1 += pair.second;  // 不满足条件
            }
        }
        float t_ratio = sum0 / (sum0 + sum1);
        if (t_ratio < ratio) {
            // t_percen 等于 countVector 的总个数
            t_percen = 0;
        } else if(t_ratio > (1-ratio)){
            t_percen = countVector.size();
        }

        if((t_percen%wide) != 0)
        {
            // if((t_percen + (wide - t_percen%wide)) <= size) 
            t_percen = (t_percen + (wide - t_percen%wide));
            // else
            // t_percen = (t_percen - (t_percen%wide));

        }
        if(t_percen<=size){
        // 将前 k 个数按升序排列
        std::sort(countVector.begin(), countVector.begin() + t_percen);
        // 将剩下的数按升序排列
        std::sort(countVector.begin() + t_percen, countVector.end());
        }else{
            std::sort(countVector.begin(), countVector.end());
        }
        /*
            3. 根据t_percen分别划分TCU和CUDA的负载
        */

        // TCU 划分
        // 3.1 根据partSize_t划分t_part
        //当前window中有几个block
        int blocks = t_percen/wide; 
        Value_tcu_binary v_tcu;
        if(blocks>0) {v_tcu.tcu_flag = true; }
        std::map<int, int> colmap;
        int c=0;
        for(auto col:countVector){
            if (c<t_percen && c<size)
            {
                v_tcu.t_column.push_back(col.first);
            }
            colmap[col.first]=c++;
        }
        if(t_percen>size){
            for(int tt=0;tt<(t_percen-size);tt++)
            v_tcu.t_column.push_back(-1);
        }
        //存放拆分的cuda长短行数据
        std::vector<Value_cuda> large;
        std::vector<Value_cuda> small;

        std::vector<std::vector<float>> t_value_temp = createVectorOfVectors_float(blocks);
        std::vector<long> t_binary_temp = createVectorOfVectors_v2(blocks);
        // std::vector<std::vector<int>> t_row_temp = createVectorOfVectors(blocks);
        // std::vector<std::vector<int>> t_col_temp = createVectorOfVectors(blocks);
        // 指示快内的行偏移
        int p=0;
        for(int j=i*window;j<(i+1)*window;j++)
        {
            //用来存储每行的cuda数据
            std::queue<int> c_col;
            std::queue<int> c_value;
            int cuda_nnz=0;
            for(int m=row[j];m<row[j+1];m++)
            {
                int newCol = colmap[column[m]];
                if(newCol<t_percen)
                {
                    int block = newCol/wide;
                    int bInId= newCol%wide;
                    t_value_temp[block].push_back(values[m]);
                    long offset = p*wide + bInId;
                    long a=1;
                    t_binary_temp[block] |= a<<offset;
                }else{
                    //划分cuda部分
                    c_col.push(column[m]);
                    c_value.push(values[m]);
                    cuda_nnz++;
                }
            }
            //处理当前行剩余的cuda元素
            if(cuda_nnz>0)
            {
                if(cuda_nnz <= shortSize)
                {
                    Value_cuda temp;
                    temp.cuda_residue_flag = true;
                    temp.c_row_offset = cuda_nnz;
                    temp.c_row = i*window + p;
                    if(t_percen>0) temp.c_atomic = 1;
                    else temp.c_atomic = 0;
                    for(int m=0; m<cuda_nnz; m++)
                    {
                        temp.c_column.push_back(c_col.front());
                        temp.c_value.push_back(c_value.front());
                        c_col.pop();
                        c_value.pop();
                    }
                    small.push_back(temp);
                }else{
                    if(cuda_nnz <= partSize_c)
                    {
                        Value_cuda temp;
                        temp.cuda_flag = true;
                        temp.c_row_offset = (cuda_nnz);
                        temp.c_row = i*window + p;
                        if(t_percen>0) temp.c_atomic = 1;
                        else temp.c_atomic = 0;
                        for(int m=0; m<cuda_nnz; m++)
                        {
                            temp.c_column.push_back(c_col.front());
                            temp.c_value.push_back(c_value.front());
                            c_col.pop();
                            c_value.pop();
                        }
                        large.push_back(temp);
                    }else{
                        // 当前行的非零元被分成多个part，并需要原子操作
                        int parts_row = cuda_nnz/partSize_c;
                        // int cur_value_indice = row[i];
                        for(int cur_i = 0; cur_i<parts_row; cur_i++)
                        {
                            Value_cuda temp;
                            temp.cuda_flag = true;      
                            temp.c_row_offset = partSize_c;
                            temp.c_row = i*window + p;
                            temp.c_atomic = 1;
                            for(int m=0; m<partSize_c; m++)
                            {
                                temp.c_column.push_back(c_col.front());
                                temp.c_value.push_back(c_value.front());
                                c_col.pop();
                                c_value.pop();
                                // cur_value_indice++;
                            }
                            large.push_back(temp);
                        }
                        int residue  = (cuda_nnz%partSize_c);
                        if(residue>0){
                            Value_cuda temp;
                            temp.c_row_offset = residue;
                            temp.c_row = i*window + p;
                            temp.c_atomic = 1;
                            for(int m=0; m<residue; m++)
                            {
                                temp.c_column.push_back(c_col.front());
                                temp.c_value.push_back(c_value.front());
                                c_col.pop();
                                c_value.pop();
                                // cur_value_indice++;
                            }
                            if(residue <= shortSize)
                            {
                                temp.cuda_residue_flag = true;
                                small.push_back(temp);
                            }else{
                                temp.cuda_flag = true;
                                large.push_back(temp);
                            }
                        }
                    }
                }
            }
            p++;
        }
        //整合vector
        v_tcu.t_value = mergeVectors_float(t_value_temp);
        v_tcu.t_binary = t_binary_temp;
        // v_tcu.t_row = mergeVectors(t_row_temp);
        // v_tcu.t_col = mergeVectors(t_col_temp);
        
        std::vector<int> count_tcu = countVectors(t_value_temp);
        if(blocks <= partSize_t)
        {
            v_tcu.row.push_back(blocks);
            //循放入每个block中的非零元数量
            for(int b=0; b<blocks; b++)
            {
                // int sum = std::accumulate(count_tcu.begin()+b, count_tcu.begin()+b+1, 0);
                v_tcu.t_block_offset.push_back(count_tcu[b]);
            }
            if(t_percen>=size) v_tcu.atomic.push_back(0);
            else v_tcu.atomic.push_back(1);
            v_tcu.window.push_back(i);
        }else{
            // 对block太多的块进行分割
            int part_number = blocks / partSize_t;
            int block_residue = blocks % partSize_t;
            for(int j=0; j<part_number; j++)
            {
                v_tcu.row.push_back(partSize_t);
                for(int b=0; b<partSize_t; b++)
                {
                    int offset = j*partSize_t + b;
                    // int offset1 = offset + b;
                    // int sum = std::accumulate(count_tcu.begin()+offset, count_tcu.begin()+offset1, 0);
                    v_tcu.t_block_offset.push_back(count_tcu[offset]);
                }
                v_tcu.window.push_back(i);
                v_tcu.atomic.push_back(1);
            }
            if(block_residue > 0)
            {             
                v_tcu.row.push_back(block_residue);
                for(int b=0; b<block_residue; b++)
                {
                    int offset = part_number*partSize_t+b;
                    // int offset1 = offset + b;
                    // int sum = std::accumulate(count_tcu.begin()+offset, count_tcu.begin()+offset1, 0);
                    v_tcu.t_block_offset.push_back(count_tcu[offset]);
                }
                v_tcu.window.push_back(i);
                v_tcu.atomic.push_back(1);
            }
        }
        /*
            4. 汇总三部分的结果
        */
        #pragma omp critical  
        res_t[i]=v_tcu;

        #pragma omp critical  
        res_c[i]=large;

        #pragma omp critical  
        res_c_residue[i]=small;
        
    
    }
    /*
        5. 整合所有window的数据
    */
    //tcu
    std::vector<int> t_rowkNew_offset;
    t_rowkNew_offset.push_back(0);
    std::vector<int> t_blockNew_offset;
    t_blockNew_offset.push_back(0);
    std::vector<float> t_valueNew;
    std::vector<int> t_columnNew;
    std::vector<long> t_binaryNew;
    std::vector<int> t_window_rowNew;
    std::vector<int> t_atomicNew;

    //cuda 长行
    std::vector<int> c_rowNew_offset;
    c_rowNew_offset.push_back(0);
    std::vector<int> c_rowNew;
    std::vector<int> c_atomicNew;
    std::vector<int> c_columnNew;
    std::vector<float> c_valueNew;

    //cuda 短行
    std::vector<int> c_rowNew_offset_residue;
    c_rowNew_offset_residue.push_back(0);
    std::vector<int> c_rowNew_residue;
    std::vector<int> c_atomicNew_residue;
    std::vector<int> c_columnNew_residue;
    std::vector<float> c_valueNew_residue;

    //按顺序整合res
    //tcu
    for (const auto& pair : res_t) {
        if (pair.second.tcu_flag == true )
        {
            for(int sub : pair.second.row)
            {
                t_rowkNew_offset.push_back(t_rowkNew_offset.back()+sub);
            }  
            for(int sub : pair.second.t_block_offset)
            {
                t_blockNew_offset.push_back(t_blockNew_offset.back()+sub);
            }  
            t_valueNew.insert(t_valueNew.end(),pair.second.t_value.begin(),pair.second.t_value.end());
            t_columnNew.insert(t_columnNew.end(),pair.second.t_column.begin(),pair.second.t_column.end());
            t_window_rowNew.insert(t_window_rowNew.end(),pair.second.window.begin(),pair.second.window.end());
            t_atomicNew.insert(t_atomicNew.end(),pair.second.atomic.begin(),pair.second.atomic.end());
            t_binaryNew.insert(t_binaryNew.end(),pair.second.t_binary.begin(),pair.second.t_binary.end());
        }   
    }

    //cuda long
    for (const auto& pair : res_c) {
        for( Value_cuda sub : pair.second){
            if (sub.cuda_flag == true )
            {c_rowNew_offset.push_back(c_rowNew_offset.back()+sub.c_row_offset);
            c_rowNew.push_back(sub.c_row);
            c_atomicNew.push_back(sub.c_atomic);
            c_columnNew.insert(c_columnNew.end(),sub.c_column.begin(),sub.c_column.end());
            c_valueNew.insert(c_valueNew.end(),sub.c_value.begin(), sub.c_value.end());}
        }
    }

    //cuda short
    for (const auto& pair : res_c_residue) {
        for( Value_cuda sub : pair.second){
            if (sub.cuda_residue_flag == true )
            {c_rowNew_offset_residue.push_back(c_rowNew_offset_residue.back()+sub.c_row_offset);
            c_rowNew_residue.push_back(sub.c_row);
            c_atomicNew_residue.push_back(sub.c_atomic);
            c_columnNew_residue.insert(c_columnNew_residue.end(),sub.c_column.begin(),sub.c_column.end());
            c_valueNew_residue.insert(c_valueNew_residue.end(),sub.c_value.begin(), sub.c_value.end());}
        }
    }

    auto end = std::chrono::high_resolution_clock::now();

    // 计算时间差并转换为毫秒
    std::chrono::duration<double, std::milli> duration = end - start;

    // std::cout << t_rowkNew_offset.size() <<std::endl;
    auto t_rowNew_offsetTensor1 = torch::from_blob(t_rowkNew_offset.data(), t_rowkNew_offset.size(), torch::kInt32);
    auto t_blockNew_offsetTensor1 = torch::from_blob(t_blockNew_offset.data(), t_blockNew_offset.size(), torch::kInt32);
    auto t_valueTensor1 = torch::from_blob(t_valueNew.data(), t_valueNew.size(), torch::kFloat32);
    auto t_columnTensor1 = torch::from_blob(t_columnNew.data(), t_columnNew.size(), torch::kInt32);
    auto t_window_rowTensor1 = torch::from_blob(t_window_rowNew.data(), t_window_rowNew.size(), torch::kInt32);
    auto t_atomicTensor1 = torch::from_blob(t_atomicNew.data(), t_atomicNew.size(), torch::kInt32);
    auto t_binaryTensor1 = torch::from_blob(t_binaryNew.data(), t_binaryNew.size(), torch::kInt64);

    torch::Tensor t_rowNew_offsetTensor = torch::empty_like(t_rowNew_offsetTensor1);
    t_rowNew_offsetTensor.copy_(t_rowNew_offsetTensor1);
    torch::Tensor t_blockNew_offsetTensor = torch::empty_like(t_blockNew_offsetTensor1);
    t_blockNew_offsetTensor.copy_(t_blockNew_offsetTensor1);
    torch::Tensor t_valueTensor = torch::empty_like(t_valueTensor1);
    t_valueTensor.copy_(t_valueTensor1);
    torch::Tensor t_columnTensor = torch::empty_like(t_columnTensor1);
    t_columnTensor.copy_(t_columnTensor1);
    torch::Tensor t_window_rowTensor = torch::empty_like(t_window_rowTensor1);
    t_window_rowTensor.copy_(t_window_rowTensor1);
    torch::Tensor t_atomicTensor = torch::empty_like(t_atomicTensor1);
    t_atomicTensor.copy_(t_atomicTensor1);
    torch::Tensor t_binaryTensor = torch::empty_like(t_binaryTensor1);
    t_binaryTensor.copy_(t_binaryTensor1);


    auto c_row_offsetTensor1 = torch::from_blob(c_rowNew_offset.data(), c_rowNew_offset.size(), torch::kInt32);
    auto c_rowTensor1 = torch::from_blob(c_rowNew.data(), c_rowNew.size(), torch::kInt32);
    auto c_atomicTensor1 = torch::from_blob(c_atomicNew.data(), c_atomicNew.size(), torch::kInt32);
    auto c_colTensor1 = torch::from_blob(c_columnNew.data(), c_columnNew.size(), torch::kInt32);
    auto c_valueTensor1 = torch::from_blob(c_valueNew.data(), c_valueNew.size(), torch::kFloat32);
    
    torch::Tensor c_row_offsetTensor = torch::empty_like(c_row_offsetTensor1);
    c_row_offsetTensor.copy_(c_row_offsetTensor1);
    torch::Tensor c_rowTensor = torch::empty_like(c_rowTensor1);
    c_rowTensor.copy_(c_rowTensor1);
    torch::Tensor c_atomicTensor = torch::empty_like(c_atomicTensor1);
    c_atomicTensor.copy_(c_atomicTensor1);
    torch::Tensor c_colTensor = torch::empty_like(c_colTensor1);
    c_colTensor.copy_(c_colTensor1);
    torch::Tensor c_valueTensor = torch::empty_like(c_valueTensor1);
    c_valueTensor.copy_(c_valueTensor1);
    

    auto c_row_offsetTensor1_residue = torch::from_blob(c_rowNew_offset_residue.data(), c_rowNew_offset_residue.size(), torch::kInt32);
    auto c_rowTensor1_residue = torch::from_blob(c_rowNew_residue.data(), c_rowNew_residue.size(), torch::kInt32);
    auto c_atomicTensor1_residue = torch::from_blob(c_atomicNew_residue.data(), c_atomicNew_residue.size(), torch::kInt32);
    auto c_colTensor1_residue = torch::from_blob(c_columnNew_residue.data(), c_columnNew_residue.size(), torch::kInt32);
    auto c_valueTensor1_residue = torch::from_blob(c_valueNew_residue.data(), c_valueNew_residue.size(), torch::kFloat32);
    
    torch::Tensor c_row_offsetTensor_residue = torch::empty_like(c_row_offsetTensor1_residue);
    c_row_offsetTensor_residue.copy_(c_row_offsetTensor1_residue);
    torch::Tensor c_rowTensor_residue = torch::empty_like(c_rowTensor1_residue);
    c_rowTensor_residue.copy_(c_rowTensor1_residue);
    torch::Tensor c_atomicTensor_residue = torch::empty_like(c_atomicTensor1_residue);
    c_atomicTensor_residue.copy_(c_atomicTensor1_residue);
    torch::Tensor c_colTensor_residue = torch::empty_like(c_colTensor1_residue);
    c_colTensor_residue.copy_(c_colTensor1_residue);
    torch::Tensor c_valueTensor_residue = torch::empty_like(c_valueTensor1_residue);
    c_valueTensor_residue.copy_(c_valueTensor1_residue);

    return {t_rowNew_offsetTensor, t_blockNew_offsetTensor,t_columnTensor, t_valueTensor, t_window_rowTensor, t_atomicTensor, t_binaryTensor,
            c_row_offsetTensor, c_rowTensor, c_atomicTensor, c_colTensor, c_valueTensor,
            c_row_offsetTensor_residue, c_rowTensor_residue, c_atomicTensor_residue, c_colTensor_residue, c_valueTensor_residue, torch::tensor(duration.count())};

}
// without value
std::vector<torch::Tensor> block_tf32_tcu_cuda_binary_kernel2(torch::Tensor row1, torch::Tensor column1, torch::Tensor values1, 
int partSize_t, int partSize_c, int shortSize, int density, int window, int wide)
{
    auto *row=row1.data_ptr<int>();
    auto column=column1.accessor<int, 1>();

    auto start = std::chrono::high_resolution_clock::now();
    // auto values=values1.accessor<float, 1>();
    int rows=row1.size(0)-1;
    // Result
    std::map<int, Value_tcu_binary> res_t;
    std::map<int, std::vector<Value_cuda>> res_c;
    std::map<int, std::vector<Value_cuda>> res_c_residue;
    //需要的window数量
    int num_windows = rows/window;
    #pragma omp parallel for
    for(int i=0;i<num_windows;i++)
    {
        /*
            1. 统计当前window中的vector, 并按照稠密度降序排列
        */
        //对column进行合并
        std::vector<int> mergedVector(&column[row[i*window]], &column[row[i*window+window]]);
        // 使用 unordered_map 统计 vector 中每个元素出现次数
        std::unordered_map<int, int> elementCounts;
        for (const auto& element : mergedVector) elementCounts[element]++;
        // 将统计结果放入 vector<pair> 中
        std::vector<std::pair<int, int>> countVector(elementCounts.begin(), elementCounts.end());
        // 按元素个数降序排序
        std::sort(countVector.begin(), countVector.end(), [](const auto& a, const auto& b) {
            return a.second > b.second;
        });

        /*
            2. 确定需要进行tcu计算的vector数量, t_percen
        */
        int size = countVector.size();
        int residue = size%wide;
        //统计每个block中的非零元数量，计算稠密度
        int t_percen = 0; //确定前多少个vector会被用作tcu进行稠密计算
        t_percen = std::count_if(countVector.begin(), countVector.end(), [density](const auto& pair) {
            return pair.second >= density;
        });
        if((t_percen%wide) != 0)
        {
            // if((t_percen + (wide - t_percen%wide)) <= size) 
            t_percen = (t_percen + (wide - t_percen%wide));
            // else
            // t_percen = (t_percen - (t_percen%wide));

        }
        if(t_percen<=size){
        // 将前 k 个数按升序排列
        std::sort(countVector.begin(), countVector.begin() + t_percen);
        // 将剩下的数按升序排列
        std::sort(countVector.begin() + t_percen, countVector.end());
        }else{
            std::sort(countVector.begin(), countVector.end());
        }
        /*
            3. 根据t_percen分别划分TCU和CUDA的负载
        */

        // TCU 划分
        // 3.1 根据partSize_t划分t_part
        //当前window中有几个block
        int blocks = t_percen/wide; 
        Value_tcu_binary v_tcu;
        if(blocks>0) {v_tcu.tcu_flag = true; }
        std::map<int, int> colmap;
        int c=0;
        for(auto col:countVector){
            if (c<t_percen && c<size)
            {
                v_tcu.t_column.push_back(col.first);
            }
            colmap[col.first]=c++;
        }
        if(t_percen>size){
            for(int tt=0;tt<(t_percen-size);tt++)
            v_tcu.t_column.push_back(-1);
        }
        //存放拆分的cuda长短行数据
        std::vector<Value_cuda> large;
        std::vector<Value_cuda> small;

        std::vector<std::vector<float>> t_value_temp = createVectorOfVectors_float(blocks);
        std::vector<long> t_binary_temp = createVectorOfVectors_v2(blocks);
        // std::vector<std::vector<int>> t_row_temp = createVectorOfVectors(blocks);
        // std::vector<std::vector<int>> t_col_temp = createVectorOfVectors(blocks);
        // 指示快内的行偏移
        int p=0;
        for(int j=i*window;j<(i+1)*window;j++)
        {
            //用来存储每行的cuda数据
            std::queue<int> c_col;
            std::queue<int> c_value;
            int cuda_nnz=0;
            for(int m=row[j];m<row[j+1];m++)
            {
                int newCol = colmap[column[m]];
                if(newCol<t_percen)
                {
                    int block = newCol/wide;
                    int bInId= newCol%wide;
                    // t_value_temp[block].push_back(values[m]);
                    long offset = p*wide + bInId;
                    long a=1;
                    t_binary_temp[block] |= a<<offset;
                }else{
                    //划分cuda部分
                    c_col.push(column[m]);
                    // c_value.push(values[m]);
                    cuda_nnz++;
                }
            }
            //处理当前行剩余的cuda元素
            if(cuda_nnz>0)
            {
                if(cuda_nnz <= shortSize)
                {
                    Value_cuda temp;
                    temp.cuda_residue_flag = true;
                    temp.c_row_offset = cuda_nnz;
                    temp.c_row = i*window + p;
                    if(t_percen>0) temp.c_atomic = 1;
                    else temp.c_atomic = 0;
                    for(int m=0; m<cuda_nnz; m++)
                    {
                        temp.c_column.push_back(c_col.front());
                        temp.c_value.push_back(c_value.front());
                        c_col.pop();
                        c_value.pop();
                    }
                    small.push_back(temp);
                }else{
                    if(cuda_nnz <= partSize_c)
                    {
                        Value_cuda temp;
                        temp.cuda_flag = true;
                        temp.c_row_offset = (cuda_nnz);
                        temp.c_row = i*window + p;
                        if(t_percen>0) temp.c_atomic = 1;
                        else temp.c_atomic = 0;
                        for(int m=0; m<cuda_nnz; m++)
                        {
                            temp.c_column.push_back(c_col.front());
                            temp.c_value.push_back(c_value.front());
                            c_col.pop();
                            c_value.pop();
                        }
                        large.push_back(temp);
                    }else{
                        // 当前行的非零元被分成多个part，并需要原子操作
                        int parts_row = cuda_nnz/partSize_c;
                        // int cur_value_indice = row[i];
                        for(int cur_i = 0; cur_i<parts_row; cur_i++)
                        {
                            Value_cuda temp;
                            temp.cuda_flag = true;      
                            temp.c_row_offset = partSize_c;
                            temp.c_row = i*window + p;
                            temp.c_atomic = 1;
                            for(int m=0; m<partSize_c; m++)
                            {
                                temp.c_column.push_back(c_col.front());
                                temp.c_value.push_back(c_value.front());
                                c_col.pop();
                                c_value.pop();
                                // cur_value_indice++;
                            }
                            large.push_back(temp);
                        }
                        int residue  = (cuda_nnz%partSize_c);
                        if(residue>0){
                            Value_cuda temp;
                            temp.c_row_offset = residue;
                            temp.c_row = i*window + p;
                            temp.c_atomic = 1;
                            for(int m=0; m<residue; m++)
                            {
                                temp.c_column.push_back(c_col.front());
                                temp.c_value.push_back(c_value.front());
                                c_col.pop();
                                c_value.pop();
                                // cur_value_indice++;
                            }
                            if(residue <= shortSize)
                            {
                                temp.cuda_residue_flag = true;
                                small.push_back(temp);
                            }else{
                                temp.cuda_flag = true;
                                large.push_back(temp);
                            }
                        }
                    }
                }
            }
            p++;
        }
        //整合vector
        v_tcu.t_value = mergeVectors_float(t_value_temp);
        v_tcu.t_binary = t_binary_temp;
        // v_tcu.t_row = mergeVectors(t_row_temp);
        // v_tcu.t_col = mergeVectors(t_col_temp);
        
        std::vector<int> count_tcu = countVectors(t_value_temp);
        if(blocks <= partSize_t)
        {
            v_tcu.row.push_back(blocks);
            //循放入每个block中的非零元数量
            for(int b=0; b<blocks; b++)
            {
                // int sum = std::accumulate(count_tcu.begin()+b, count_tcu.begin()+b+1, 0);
                v_tcu.t_block_offset.push_back(count_tcu[b]);
            }
            if(t_percen>=size) v_tcu.atomic.push_back(0);
            else v_tcu.atomic.push_back(1);
            v_tcu.window.push_back(i);
        }else{
            // 对block太多的块进行分割
            int part_number = blocks / partSize_t;
            int block_residue = blocks % partSize_t;
            for(int j=0; j<part_number; j++)
            {
                v_tcu.row.push_back(partSize_t);
                for(int b=0; b<partSize_t; b++)
                {
                    int offset = j*partSize_t + b;
                    // int offset1 = offset + b;
                    // int sum = std::accumulate(count_tcu.begin()+offset, count_tcu.begin()+offset1, 0);
                    v_tcu.t_block_offset.push_back(count_tcu[offset]);
                }
                v_tcu.window.push_back(i);
                v_tcu.atomic.push_back(1);
            }
            if(block_residue > 0)
            {             
                v_tcu.row.push_back(block_residue);
                for(int b=0; b<block_residue; b++)
                {
                    int offset = part_number*partSize_t+b;
                    // int offset1 = offset + b;
                    // int sum = std::accumulate(count_tcu.begin()+offset, count_tcu.begin()+offset1, 0);
                    v_tcu.t_block_offset.push_back(count_tcu[offset]);
                }
                v_tcu.window.push_back(i);
                v_tcu.atomic.push_back(1);
            }
        }
        /*
            4. 汇总三部分的结果
        */
        #pragma omp critical  
        res_t[i]=v_tcu;

        #pragma omp critical  
        res_c[i]=large;

        #pragma omp critical  
        res_c_residue[i]=small;
        
    
    }
    /*
        5. 整合所有window的数据
    */
    //tcu
    std::vector<int> t_rowkNew_offset;
    t_rowkNew_offset.push_back(0);
    std::vector<int> t_blockNew_offset;
    t_blockNew_offset.push_back(0);
    //std::vector<float> t_valueNew;
    std::vector<int> t_columnNew;
    std::vector<long> t_binaryNew;
    std::vector<int> t_window_rowNew;
    std::vector<int> t_atomicNew;

    //cuda 长行
    std::vector<int> c_rowNew_offset;
    c_rowNew_offset.push_back(0);
    std::vector<int> c_rowNew;
    std::vector<int> c_atomicNew;
    std::vector<int> c_columnNew;
    //std::vector<float> c_valueNew;

    //cuda 短行
    std::vector<int> c_rowNew_offset_residue;
    c_rowNew_offset_residue.push_back(0);
    std::vector<int> c_rowNew_residue;
    std::vector<int> c_atomicNew_residue;
    std::vector<int> c_columnNew_residue;
    //std::vector<float> c_valueNew_residue;

    //按顺序整合res
    //tcu
    for (const auto& pair : res_t) {
        if (pair.second.tcu_flag == true )
        {
            for(int sub : pair.second.row)
            {
                t_rowkNew_offset.push_back(t_rowkNew_offset.back()+sub);
            }  
            for(int sub : pair.second.t_block_offset)
            {
                t_blockNew_offset.push_back(t_blockNew_offset.back()+sub);
            }  
            //t_valueNew.insert(t_valueNew.end(),pair.second.t_value.begin(),pair.second.t_value.end());
            t_columnNew.insert(t_columnNew.end(),pair.second.t_column.begin(),pair.second.t_column.end());
            t_window_rowNew.insert(t_window_rowNew.end(),pair.second.window.begin(),pair.second.window.end());
            t_atomicNew.insert(t_atomicNew.end(),pair.second.atomic.begin(),pair.second.atomic.end());
            t_binaryNew.insert(t_binaryNew.end(),pair.second.t_binary.begin(),pair.second.t_binary.end());
        }   
    }

    //cuda long
    for (const auto& pair : res_c) {
        for( Value_cuda sub : pair.second){
            if (sub.cuda_flag == true )
            {c_rowNew_offset.push_back(c_rowNew_offset.back()+sub.c_row_offset);
            c_rowNew.push_back(sub.c_row);
            c_atomicNew.push_back(sub.c_atomic);
            c_columnNew.insert(c_columnNew.end(),sub.c_column.begin(),sub.c_column.end());}
        }
    }

    //cuda short
    for (const auto& pair : res_c_residue) {
        for( Value_cuda sub : pair.second){
            if (sub.cuda_residue_flag == true )
            {c_rowNew_offset_residue.push_back(c_rowNew_offset_residue.back()+sub.c_row_offset);
            c_rowNew_residue.push_back(sub.c_row);
            c_atomicNew_residue.push_back(sub.c_atomic);
            c_columnNew_residue.insert(c_columnNew_residue.end(),sub.c_column.begin(),sub.c_column.end());}
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();

    // 计算时间差并转换为毫秒
    std::chrono::duration<double, std::milli> duration = end - start;

    // std::cout << t_rowkNew_offset.size() <<std::endl;
    auto t_rowNew_offsetTensor1 = torch::from_blob(t_rowkNew_offset.data(), t_rowkNew_offset.size(), torch::kInt32);
    auto t_blockNew_offsetTensor1 = torch::from_blob(t_blockNew_offset.data(), t_blockNew_offset.size(), torch::kInt32);
    //auto t_valueTensor1 = torch::from_blob(t_valueNew.data(), t_valueNew.size(), torch::kFloat32);
    auto t_columnTensor1 = torch::from_blob(t_columnNew.data(), t_columnNew.size(), torch::kInt32);
    auto t_window_rowTensor1 = torch::from_blob(t_window_rowNew.data(), t_window_rowNew.size(), torch::kInt32);
    auto t_atomicTensor1 = torch::from_blob(t_atomicNew.data(), t_atomicNew.size(), torch::kInt32);
    auto t_binaryTensor1 = torch::from_blob(t_binaryNew.data(), t_binaryNew.size(), torch::kInt64);

    torch::Tensor t_rowNew_offsetTensor = torch::empty_like(t_rowNew_offsetTensor1);
    t_rowNew_offsetTensor.copy_(t_rowNew_offsetTensor1);
    torch::Tensor t_blockNew_offsetTensor = torch::empty_like(t_blockNew_offsetTensor1);
    t_blockNew_offsetTensor.copy_(t_blockNew_offsetTensor1);
    //torch::Tensor t_valueTensor = torch::empty_like(t_valueTensor1);
    //t_valueTensor.copy_(t_valueTensor1);
    torch::Tensor t_columnTensor = torch::empty_like(t_columnTensor1);
    t_columnTensor.copy_(t_columnTensor1);
    torch::Tensor t_window_rowTensor = torch::empty_like(t_window_rowTensor1);
    t_window_rowTensor.copy_(t_window_rowTensor1);
    torch::Tensor t_atomicTensor = torch::empty_like(t_atomicTensor1);
    t_atomicTensor.copy_(t_atomicTensor1);
    torch::Tensor t_binaryTensor = torch::empty_like(t_binaryTensor1);
    t_binaryTensor.copy_(t_binaryTensor1);


    auto c_row_offsetTensor1 = torch::from_blob(c_rowNew_offset.data(), c_rowNew_offset.size(), torch::kInt32);
    auto c_rowTensor1 = torch::from_blob(c_rowNew.data(), c_rowNew.size(), torch::kInt32);
    auto c_atomicTensor1 = torch::from_blob(c_atomicNew.data(), c_atomicNew.size(), torch::kInt32);
    auto c_colTensor1 = torch::from_blob(c_columnNew.data(), c_columnNew.size(), torch::kInt32);
    //auto c_valueTensor1 = torch::from_blob(c_valueNew.data(), c_valueNew.size(), torch::kFloat32);
    
    torch::Tensor c_row_offsetTensor = torch::empty_like(c_row_offsetTensor1);
    c_row_offsetTensor.copy_(c_row_offsetTensor1);
    torch::Tensor c_rowTensor = torch::empty_like(c_rowTensor1);
    c_rowTensor.copy_(c_rowTensor1);
    torch::Tensor c_atomicTensor = torch::empty_like(c_atomicTensor1);
    c_atomicTensor.copy_(c_atomicTensor1);
    torch::Tensor c_colTensor = torch::empty_like(c_colTensor1);
    c_colTensor.copy_(c_colTensor1);
    //torch::Tensor c_valueTensor = torch::empty_like(c_valueTensor1);
    //c_valueTensor.copy_(c_valueTensor1);
    

    auto c_row_offsetTensor1_residue = torch::from_blob(c_rowNew_offset_residue.data(), c_rowNew_offset_residue.size(), torch::kInt32);
    auto c_rowTensor1_residue = torch::from_blob(c_rowNew_residue.data(), c_rowNew_residue.size(), torch::kInt32);
    auto c_atomicTensor1_residue = torch::from_blob(c_atomicNew_residue.data(), c_atomicNew_residue.size(), torch::kInt32);
    auto c_colTensor1_residue = torch::from_blob(c_columnNew_residue.data(), c_columnNew_residue.size(), torch::kInt32);
    //auto c_valueTensor1_residue = torch::from_blob(c_valueNew_residue.data(), c_valueNew_residue.size(), torch::kFloat32);
    
    torch::Tensor c_row_offsetTensor_residue = torch::empty_like(c_row_offsetTensor1_residue);
    c_row_offsetTensor_residue.copy_(c_row_offsetTensor1_residue);
    torch::Tensor c_rowTensor_residue = torch::empty_like(c_rowTensor1_residue);
    c_rowTensor_residue.copy_(c_rowTensor1_residue);
    torch::Tensor c_atomicTensor_residue = torch::empty_like(c_atomicTensor1_residue);
    c_atomicTensor_residue.copy_(c_atomicTensor1_residue);
    torch::Tensor c_colTensor_residue = torch::empty_like(c_colTensor1_residue);
    c_colTensor_residue.copy_(c_colTensor1_residue);
    // torch::Tensor c_valueTensor_residue = torch::empty_like(c_valueTensor1_residue);
    // c_valueTensor_residue.copy_(c_valueTensor1_residue);

    return {t_rowNew_offsetTensor, t_blockNew_offsetTensor,t_columnTensor, t_window_rowTensor, t_atomicTensor, t_binaryTensor,
            c_row_offsetTensor, c_rowTensor, c_atomicTensor, c_colTensor,
            c_row_offsetTensor_residue, c_rowTensor_residue, c_atomicTensor_residue, c_colTensor_residue, torch::tensor(duration.count())};

}


// m-tcu + cuda + partSize + binary + sddmm
std::vector<torch::Tensor> block_tf32_tcu_cuda_binary_kernel_sddmm(torch::Tensor row1, torch::Tensor column1, 
int partSize_t, int partSize_c,  int density, int window, int wide)
{
    auto *row=row1.data_ptr<int>();
    auto column=column1.accessor<int, 1>();
    // auto values=values1.accessor<float, 1>();
    int rows=row1.size(0)-1;
    // Result
    std::map<int, Value_tcu_binary> res_t;
    std::map<int, std::vector<Value_cuda>> res_c;
    std::map<int, std::vector<Value_cuda>> res_c_residue;
    //需要的window数量
    int num_windows = rows/window;
    #pragma omp parallel for
    for(int i=0;i<num_windows;i++)
    {
        /*
            1. 统计当前window中的vector, 并按照稠密度降序排列
        */
        //对column进行合并
        std::vector<int> mergedVector(&column[row[i*window]], &column[row[i*window+window]]);
        // 使用 unordered_map 统计 vector 中每个元素出现次数
        std::unordered_map<int, int> elementCounts;
        for (const auto& element : mergedVector) elementCounts[element]++;
        // 将统计结果放入 vector<pair> 中
        std::vector<std::pair<int, int>> countVector(elementCounts.begin(), elementCounts.end());
        // 按元素个数降序排序
        std::sort(countVector.begin(), countVector.end(), [](const auto& a, const auto& b) {
            return a.second > b.second;
        });

        /*
            2. 确定需要进行tcu计算的vector数量, t_percen
        */
        int size = countVector.size();
        int residue = size%wide;
        int pad = wide - residue;
        int size_pad = size + pad;
        int t_percen = 0;
        for (int cur_i = 0; cur_i < size_pad; cur_i += wide)
        {
            int localSum = 0;
            for (int cur_j = cur_i; cur_j < (cur_i + wide) && cur_j < size; cur_j++) {
                localSum += countVector[cur_j].second;
            }
            // if(localSum >= (density*temp))
            if(localSum >= density)  
            {t_percen +=wide;}
            // temp++;
        }
        if(t_percen<=size){
        // 将前 k 个数按升序排列
        std::sort(countVector.begin(), countVector.begin() + t_percen);
        // 将剩下的数按升序排列
        std::sort(countVector.begin() + t_percen, countVector.end());
        }else{
            std::sort(countVector.begin(), countVector.end());
        }
        /*
            3. 根据t_percen分别划分TCU和CUDA的负载
        */

        // TCU 划分
        // 3.1 根据partSize_t划分t_part
        //当前window中有几个block
        int blocks = t_percen/wide; 
        Value_tcu_binary v_tcu;
        if(blocks>0) {v_tcu.tcu_flag = true; }
        std::map<int, int> colmap;
        int c=0;
        for(auto col:countVector){
            if (c<t_percen && c<size)
            {
                v_tcu.t_column.push_back(col.first);
            }
            colmap[col.first]=c++;
        }
        if(t_percen>size){
            for(int tt=0;tt<(t_percen-size);tt++)
            v_tcu.t_column.push_back(-1);
        }
        //存放拆分的cuda长短行数据
        std::vector<Value_cuda> large;

        std::vector<std::vector<float>> t_value_temp = createVectorOfVectors_float(blocks);
        std::vector<long> t_binary_temp = createVectorOfVectors_v2(blocks*2);
        // 指示快内的行偏移
        int p=0;
        for(int j=i*window;j<(i+1)*window;j++)
        {
            //用来存储每行的cuda数据
            std::queue<int> c_col;
            std::queue<int> c_value;
            int cuda_nnz=0;
            for(int m=row[j];m<row[j+1];m++)
            {
                int newCol = colmap[column[m]];
                if(newCol<t_percen)
                {
                    int block = newCol/wide;
                    int bInId= newCol%wide;
                    t_value_temp[block].push_back(1);
                    long a=1;
                    if(bInId<8){
                        // long offset = bInId*window + p;
                        // t_binary_temp[block*2] |= a<<offset;
                        long offset = p*(wide/2) + bInId;
                        t_binary_temp[block*2] |= a<<offset;
                    }else{
                        bInId = bInId-8;
                        long offset = p*(wide/2) + bInId;
                        t_binary_temp[block*2 + 1] |= a<<offset;
                    }
                }else{
                    //划分cuda部分
                    c_col.push(column[m]);
                    c_value.push(1);
                    cuda_nnz++;
                }
            }
            //if(j==1) std::cout << cuda_nnz<<std::endl;
            //处理当前行剩余的cuda元素
            if(cuda_nnz>0)
            {
                if(cuda_nnz <= partSize_c)
                {
                    Value_cuda temp;
                    temp.cuda_flag = true;
                    temp.c_row_offset = (cuda_nnz);
                    temp.c_row = i*window + p;
                    if(t_percen>0) temp.c_atomic = 1;
                    else temp.c_atomic = 0;
                    for(int m=0; m<cuda_nnz; m++)
                    {
                        temp.c_column.push_back(c_col.front());
                        temp.c_value.push_back(c_value.front());
                        c_col.pop();
                        c_value.pop();
                    }
                    large.push_back(temp);
                }else{
                    // 当前行的非零元被分成多个part，并需要原子操作
                    int parts_row = cuda_nnz/partSize_c;
                    // int cur_value_indice = row[i];
                    for(int cur_i = 0; cur_i<parts_row; cur_i++)
                    {
                        Value_cuda temp;
                        temp.cuda_flag = true;      
                        temp.c_row_offset = partSize_c;
                        temp.c_row = i*window + p;
                        if(t_percen>0) temp.c_atomic = 1;
                        else temp.c_atomic = 0;
                        for(int m=0; m<partSize_c; m++)
                        {
                            temp.c_column.push_back(c_col.front());
                            temp.c_value.push_back(c_value.front());
                            c_col.pop();
                            c_value.pop();
                            // cur_value_indice++;
                        }
                        large.push_back(temp);
                    }
                    int residue  = (cuda_nnz%partSize_c);
                    if(residue>0){
                        Value_cuda temp;
                        temp.c_row_offset = residue;
                        temp.c_row = i*window + p;
                        temp.c_atomic = 1;
                        for(int m=0; m<residue; m++)
                        {
                            temp.c_column.push_back(c_col.front());
                            temp.c_value.push_back(c_value.front());
                            c_col.pop();
                            c_value.pop();
                            // cur_value_indice++;
                        }
                        temp.cuda_flag = true;
                        large.push_back(temp);
                    }
                }
            }
            p++;
        }
        //整合vector
        v_tcu.t_value = mergeVectors_float(t_value_temp);
        v_tcu.t_binary = t_binary_temp;
        
        std::vector<int> count_tcu = countVectors(t_value_temp);
        if(blocks <= partSize_t)
        {
            v_tcu.row.push_back(blocks);
            //循放入每个block中的非零元数量
            for(int b=0; b<blocks; b++)
            {
                // int sum = std::accumulate(count_tcu.begin()+b, count_tcu.begin()+b+1, 0);
                v_tcu.t_block_offset.push_back(count_tcu[b]);
            }
            if(t_percen>=size) v_tcu.atomic.push_back(0);
            else v_tcu.atomic.push_back(1);
            v_tcu.window.push_back(i);
        }else{
            // 对block太多的块进行分割
            int part_number = blocks / partSize_t;
            int block_residue = blocks % partSize_t;
            for(int j=0; j<part_number; j++)
            {
                v_tcu.row.push_back(partSize_t);
                for(int b=0; b<partSize_t; b++)
                {
                    int offset = j*partSize_t + b;
                    // int offset1 = offset + b;
                    // int sum = std::accumulate(count_tcu.begin()+offset, count_tcu.begin()+offset1, 0);
                    v_tcu.t_block_offset.push_back(count_tcu[offset]);
                }
                v_tcu.window.push_back(i);
                v_tcu.atomic.push_back(1);
            }
            if(block_residue > 0)
            {             
                v_tcu.row.push_back(block_residue);
                for(int b=0; b<block_residue; b++)
                {
                    int offset = part_number*partSize_t+b;
                    // int offset1 = offset + b;
                    // int sum = std::accumulate(count_tcu.begin()+offset, count_tcu.begin()+offset1, 0);
                    v_tcu.t_block_offset.push_back(count_tcu[offset]);
                }
                v_tcu.window.push_back(i);
                v_tcu.atomic.push_back(1);
            }
        }
        /*
            4. 汇总三部分的结果
        */
        #pragma omp critical  
        res_t[i]=v_tcu;

        #pragma omp critical  
        res_c[i]=large;
    }
    /*
        5. 整合所有window的数据
    */
    //tcu
    std::vector<int> t_rowkNew_offset;
    t_rowkNew_offset.push_back(0);
    std::vector<int> t_blockNew_offset;
    t_blockNew_offset.push_back(0);
    // std::vector<float> t_valueNew;
    std::vector<int> t_columnNew;
    std::vector<long> t_binaryNew;
    std::vector<int> t_window_rowNew;
    std::vector<int> t_atomicNew;

    //cuda 长行
    std::vector<int> c_rowNew_offset;
    c_rowNew_offset.push_back(0);
    std::vector<int> c_rowNew;
    std::vector<int> c_columnNew;
    std::vector<int> c_atomicNew;


    //按顺序整合res
    //tcu
    for (const auto& pair : res_t) {
        if (pair.second.tcu_flag == true )
        {
            for(int sub : pair.second.row)
            {
                t_rowkNew_offset.push_back(t_rowkNew_offset.back()+sub);
            }  
            for(int sub : pair.second.t_block_offset)
            {
                t_blockNew_offset.push_back(t_blockNew_offset.back()+sub);
            }  
            //t_valueNew.insert(t_valueNew.end(),pair.second.t_value.begin(),pair.second.t_value.end());
            t_columnNew.insert(t_columnNew.end(),pair.second.t_column.begin(),pair.second.t_column.end());
            t_window_rowNew.insert(t_window_rowNew.end(),pair.second.window.begin(),pair.second.window.end());
            t_atomicNew.insert(t_atomicNew.end(),pair.second.atomic.begin(),pair.second.atomic.end());
            t_binaryNew.insert(t_binaryNew.end(),pair.second.t_binary.begin(),pair.second.t_binary.end());
        }   
    }
    // std::cout << t_rowkNew_offset.size() <<std::endl;
    auto t_rowNew_offsetTensor1 = torch::from_blob(t_rowkNew_offset.data(), t_rowkNew_offset.size(), torch::kInt32);
    auto t_blockNew_offsetTensor1 = torch::from_blob(t_blockNew_offset.data(), t_blockNew_offset.size(), torch::kInt32);
    //auto t_valueTensor1 = torch::from_blob(t_valueNew.data(), t_valueNew.size(), torch::kFloat32);
    auto t_columnTensor1 = torch::from_blob(t_columnNew.data(), t_columnNew.size(), torch::kInt32);
    auto t_window_rowTensor1 = torch::from_blob(t_window_rowNew.data(), t_window_rowNew.size(), torch::kInt32);
    auto t_atomicTensor1 = torch::from_blob(t_atomicNew.data(), t_atomicNew.size(), torch::kInt32);
    auto t_binaryTensor1 = torch::from_blob(t_binaryNew.data(), t_binaryNew.size(), torch::kInt64);

    torch::Tensor t_rowNew_offsetTensor = torch::empty_like(t_rowNew_offsetTensor1);
    t_rowNew_offsetTensor.copy_(t_rowNew_offsetTensor1);
    torch::Tensor t_blockNew_offsetTensor = torch::empty_like(t_blockNew_offsetTensor1);
    t_blockNew_offsetTensor.copy_(t_blockNew_offsetTensor1);
    //torch::Tensor t_valueTensor = torch::empty_like(t_valueTensor1);
    //t_valueTensor.copy_(t_valueTensor1);
    torch::Tensor t_columnTensor = torch::empty_like(t_columnTensor1);
    t_columnTensor.copy_(t_columnTensor1);
    torch::Tensor t_window_rowTensor = torch::empty_like(t_window_rowTensor1);
    t_window_rowTensor.copy_(t_window_rowTensor1);
    torch::Tensor t_atomicTensor = torch::empty_like(t_atomicTensor1);
    t_atomicTensor.copy_(t_atomicTensor1);
    torch::Tensor t_binaryTensor = torch::empty_like(t_binaryTensor1);
    t_binaryTensor.copy_(t_binaryTensor1);

    //cuda long
    for (const auto& pair : res_c) {
        for( Value_cuda sub : pair.second){
            c_rowNew_offset.push_back(c_rowNew_offset.back()+sub.c_row_offset);
            c_rowNew.push_back(sub.c_row);
            c_columnNew.insert(c_columnNew.end(),sub.c_column.begin(),sub.c_column.end());
            c_atomicNew.push_back(sub.c_atomic);
        }
    }

    auto c_row_offsetTensor1 = torch::from_blob(c_rowNew_offset.data(), c_rowNew_offset.size(), torch::kInt32);
    auto c_rowTensor1 = torch::from_blob(c_rowNew.data(), c_rowNew.size(), torch::kInt32);
    auto c_colTensor1 = torch::from_blob(c_columnNew.data(), c_columnNew.size(), torch::kInt32);
    auto c_atomicTensor1 = torch::from_blob(c_atomicNew.data(), c_atomicNew.size(), torch::kInt32);

    torch::Tensor c_row_offsetTensor = torch::empty_like(c_row_offsetTensor1);
    c_row_offsetTensor.copy_(c_row_offsetTensor1);
    torch::Tensor c_rowTensor = torch::empty_like(c_rowTensor1);
    c_rowTensor.copy_(c_rowTensor1);
    torch::Tensor c_colTensor = torch::empty_like(c_colTensor1);
    c_colTensor.copy_(c_colTensor1);
    torch::Tensor c_atomicTensor = torch::empty_like(c_atomicTensor1);
    c_atomicTensor.copy_(c_atomicTensor1);


    return {t_rowNew_offsetTensor, t_blockNew_offsetTensor,t_columnTensor, t_window_rowTensor, t_atomicTensor, t_binaryTensor,
            c_row_offsetTensor, c_rowTensor, c_colTensor,c_atomicTensor};

}


//纯tcu part

// tcu+cuda v3
std::vector<torch::Tensor> block_csr_binary_tcu_part_kernel(torch::Tensor row1, torch::Tensor column1, torch::Tensor values1, 
int partSize_t, int window, int wide)
{
    auto *row=row1.data_ptr<int>();
    auto column=column1.accessor<int, 1>();
    auto values=values1.accessor<float, 1>();
    int rows=row1.size(0)-1;
    // Result
    std::map<int, Value_tcu_csr> res_t;
    //需要的window数量
    int num_windows = rows/window;
    #pragma omp parallel for
    for(int i=0;i<num_windows;i++)
    {
        /*
            1. 统计当前window中的vector, 并按照稠密度降序排列
        */
        //对column进行合并
        std::vector<int> mergedVector(&column[row[i*window]], &column[row[i*window+window]]);
        // 使用 unordered_map 统计 vector 中每个元素出现次数
        std::unordered_map<int, int> elementCounts;
        for (const auto& element : mergedVector) elementCounts[element]++;
        // 将统计结果放入 vector<pair> 中
        std::vector<std::pair<int, int>> countVector(elementCounts.begin(), elementCounts.end());
        // 按元素个数降序排序
        // std::sort(countVector.begin(), countVector.end(), [](const auto& a, const auto& b) {
        //     return a.second > b.second;
        // });

        /*
            2. 确定需要进行tcu计算的vector数量, t_percen
        */
        int size = countVector.size();
        int pad = wide - size%wide;
        //统计每个block中的非零元数量，计算稠密度
        int t_percen = size; //确定前多少个vector会被用作tcu进行稠密计算
        if(size%wide > 0)
        t_percen +=pad;

        std::sort(countVector.begin(), countVector.end());

        /*
            3. 根据t_percen分别划分TCU和CUDA的负载
        */

        // TCU 划分
        // 3.1 根据partSize_t划分t_part
        //当前window中有几个block
        int blocks = t_percen/wide; 
        Value_tcu_csr v_tcu;
        if(blocks>0) {v_tcu.tcu_flag = true; }
        std::map<int, int> colmap;
        int c=0;
        for(auto col:countVector){
            v_tcu.t_column.push_back(col.first);
            colmap[col.first]=c++;
        }
        for(int r=0;r<pad;r++)
        v_tcu.t_column.push_back(-1);

        std::vector<std::vector<float>> t_value_temp = createVectorOfVectors_float(blocks);
        std::vector<long> t_binary_temp = createVectorOfVectors_v2(blocks);
        // 指示快内的行偏移
        int p=0;
        for(int j=i*window;j<(i+1)*window;j++)
        {
            //用来存储每行的cuda数据
            for(int m=row[j];m<row[j+1];m++)
            {
                int newCol = colmap[column[m]];
                int block = newCol/wide;
                int bInId= newCol%wide;
                t_value_temp[block].push_back(values[m]);
                long offset = p*wide + bInId;
                long a=1;
                t_binary_temp[block] |= a<<offset;
            }
            p++;
        }
        //整合vector
        v_tcu.t_value = mergeVectors_float(t_value_temp);
        v_tcu.t_binary = t_binary_temp;
        
        std::vector<int> count_tcu = countVectors(t_value_temp);
        if(blocks <= partSize_t)
        {
            v_tcu.row.push_back(blocks);
            //循放入每个block中的非零元数量
            for(int b=0; b<blocks; b++)
            {
                // int sum = std::accumulate(count_tcu.begin()+b, count_tcu.begin()+b+1, 0);
                v_tcu.t_block_offset.push_back(count_tcu[b]);
            }
            v_tcu.atomic.push_back(0);
            v_tcu.window.push_back(i);
        }else{
            // 对block太多的块进行分割
            int part_number = blocks / partSize_t;
            int block_residue = blocks % partSize_t;
            for(int j=0; j<part_number; j++)
            {
                v_tcu.row.push_back(partSize_t);
                for(int b=0; b<partSize_t; b++)
                {
                    int offset = j*partSize_t + b;
                    // int offset1 = offset + b;
                    // int sum = std::accumulate(count_tcu.begin()+offset, count_tcu.begin()+offset1, 0);
                    v_tcu.t_block_offset.push_back(count_tcu[offset]);
                }
                v_tcu.window.push_back(i);
                v_tcu.atomic.push_back(1);
            }
            if(block_residue > 0)
            {             
                v_tcu.row.push_back(block_residue);
                for(int b=0; b<block_residue; b++)
                {
                    int offset = part_number*partSize_t+b;
                    // int offset1 = offset + b;
                    // int sum = std::accumulate(count_tcu.begin()+offset, count_tcu.begin()+offset1, 0);
                    v_tcu.t_block_offset.push_back(count_tcu[offset]);
                }
                v_tcu.window.push_back(i);
                v_tcu.atomic.push_back(1);
            }
        }
        /*
            4. 汇总三部分的结果
        */
        #pragma omp critical  
        res_t[i]=v_tcu;
        
    
    }
    /*
        5. 整合所有window的数据
    */
    //tcu
    std::vector<int> t_rowkNew_offset;
    t_rowkNew_offset.push_back(0);
    std::vector<int> t_blockNew_offset;
    t_blockNew_offset.push_back(0);
    std::vector<float> t_valueNew;
    std::vector<int> t_columnNew;
    std::vector<long> t_binaryNew;
    // std::vector<int> t_rowNew;
    // std::vector<int> t_colNew;
    std::vector<int> t_window_rowNew;
    std::vector<int> t_atomicNew;

    //按顺序整合res
    //tcu
    for (const auto& pair : res_t) {
        if (pair.second.tcu_flag == true )
        {
            for(int sub : pair.second.row)
            {
                t_rowkNew_offset.push_back(t_rowkNew_offset.back()+sub);
            }  
            for(int sub : pair.second.t_block_offset)
            {
                t_blockNew_offset.push_back(t_blockNew_offset.back()+sub);
            }  
            t_valueNew.insert(t_valueNew.end(),pair.second.t_value.begin(),pair.second.t_value.end());
            t_columnNew.insert(t_columnNew.end(),pair.second.t_column.begin(),pair.second.t_column.end());
            t_window_rowNew.insert(t_window_rowNew.end(),pair.second.window.begin(),pair.second.window.end());
            t_atomicNew.insert(t_atomicNew.end(),pair.second.atomic.begin(),pair.second.atomic.end());
            t_binaryNew.insert(t_binaryNew.end(),pair.second.t_binary.begin(),pair.second.t_binary.end());
        }   
    }
    // std::cout << t_rowkNew_offset.size() <<std::endl;
    auto t_rowNew_offsetTensor1 = torch::from_blob(t_rowkNew_offset.data(), t_rowkNew_offset.size(), torch::kInt32);
    auto t_blockNew_offsetTensor1 = torch::from_blob(t_blockNew_offset.data(), t_blockNew_offset.size(), torch::kInt32);
    auto t_valueTensor1 = torch::from_blob(t_valueNew.data(), t_valueNew.size(), torch::kFloat32);
    auto t_columnTensor1 = torch::from_blob(t_columnNew.data(), t_columnNew.size(), torch::kInt32);
    auto t_window_rowTensor1 = torch::from_blob(t_window_rowNew.data(), t_window_rowNew.size(), torch::kInt32);
    auto t_atomicTensor1 = torch::from_blob(t_atomicNew.data(), t_atomicNew.size(), torch::kInt32);
    auto t_binaryTensor1 = torch::from_blob(t_binaryNew.data(), t_binaryNew.size(), torch::kInt64);

    torch::Tensor t_rowNew_offsetTensor = torch::empty_like(t_rowNew_offsetTensor1);
    t_rowNew_offsetTensor.copy_(t_rowNew_offsetTensor1);
    torch::Tensor t_blockNew_offsetTensor = torch::empty_like(t_blockNew_offsetTensor1);
    t_blockNew_offsetTensor.copy_(t_blockNew_offsetTensor1);
    torch::Tensor t_valueTensor = torch::empty_like(t_valueTensor1);
    t_valueTensor.copy_(t_valueTensor1);
    torch::Tensor t_columnTensor = torch::empty_like(t_columnTensor1);
    t_columnTensor.copy_(t_columnTensor1);
    torch::Tensor t_window_rowTensor = torch::empty_like(t_window_rowTensor1);
    t_window_rowTensor.copy_(t_window_rowTensor1);
    torch::Tensor t_atomicTensor = torch::empty_like(t_atomicTensor1);
    t_atomicTensor.copy_(t_atomicTensor1);
    torch::Tensor t_binaryTensor = torch::empty_like(t_binaryTensor1);
    t_binaryTensor.copy_(t_binaryTensor1);

   
    return {t_rowNew_offsetTensor, t_blockNew_offsetTensor,t_columnTensor, t_valueTensor, t_window_rowTensor, t_atomicTensor, t_binaryTensor};

}

std::vector<torch::Tensor> block_csr_binary_sddmm_tcu_part_kernel(torch::Tensor row1, torch::Tensor column1, 
int partSize_t, int window, int wide)
{
    auto *row=row1.data_ptr<int>();
    auto column=column1.accessor<int, 1>();
    int rows=row1.size(0)-1;
    // Result
    std::map<int, Value_tcu_csr> res_t;
    //需要的window数量
    int num_windows = rows/window;
    #pragma omp parallel for
    for(int i=0;i<num_windows;i++)
    {
        /*
            1. 统计当前window中的vector, 并按照稠密度降序排列
        */
        //对column进行合并
        std::vector<int> mergedVector(&column[row[i*window]], &column[row[i*window+window]]);
        // 使用 unordered_map 统计 vector 中每个元素出现次数
        std::unordered_map<int, int> elementCounts;
        for (const auto& element : mergedVector) elementCounts[element]++;
        // 将统计结果放入 vector<pair> 中
        std::vector<std::pair<int, int>> countVector(elementCounts.begin(), elementCounts.end());
        // 按元素个数降序排序
        // std::sort(countVector.begin(), countVector.end(), [](const auto& a, const auto& b) {
        //     return a.second > b.second;
        // });

        /*
            2. 确定需要进行tcu计算的vector数量, t_percen
        */
        int size = countVector.size();
        int pad = wide - size%wide;
        //统计每个block中的非零元数量，计算稠密度
        int t_percen = size; //确定前多少个vector会被用作tcu进行稠密计算
        if(size%wide > 0)
        t_percen +=pad;

        std::sort(countVector.begin(), countVector.end());

        /*
            3. 根据t_percen分别划分TCU和CUDA的负载
        */

        // TCU 划分
        // 3.1 根据partSize_t划分t_part
        //当前window中有几个block
        int blocks = t_percen/wide; 
        Value_tcu_csr v_tcu;
        if(blocks>0) {v_tcu.tcu_flag = true; }
        std::map<int, int> colmap;
        int c=0;
        for(auto col:countVector){
            v_tcu.t_column.push_back(col.first);
            colmap[col.first]=c++;
        }
        for(int r=0;r<pad;r++)
        v_tcu.t_column.push_back(-1);

        std::vector<std::vector<float>> t_value_temp = createVectorOfVectors_float(blocks);
        std::vector<long> t_binary_temp = createVectorOfVectors_v2(blocks*2);
        // 指示快内的行偏移
        int p=0;
        for(int j=i*window;j<(i+1)*window;j++)
        {
            //用来存储每行的cuda数据
            std::queue<int> c_col;
            std::queue<int> c_value;
            for(int m=row[j];m<row[j+1];m++)
            {
                int newCol = colmap[column[m]];
                int block = newCol/wide;
                int bInId= newCol%wide;
                t_value_temp[block].push_back(1);
                long a=1;
                if(bInId<8){
                    long offset = p*(wide/2) + bInId;
                    t_binary_temp[block*2] |= a<<offset;
                }else{
                    bInId = bInId-8;
                    long offset = p*(wide/2) + bInId;
                    t_binary_temp[block*2 + 1] |= a<<offset;
                }
            }
            p++;
        }
        //整合vector
        v_tcu.t_value = mergeVectors_float(t_value_temp);
        v_tcu.t_binary = t_binary_temp;
        
        std::vector<int> count_tcu = countVectors(t_value_temp);
        if(blocks <= partSize_t)
        {
            v_tcu.row.push_back(blocks);
            //循放入每个block中的非零元数量
            for(int b=0; b<blocks; b++)
            {
                // int sum = std::accumulate(count_tcu.begin()+b, count_tcu.begin()+b+1, 0);
                v_tcu.t_block_offset.push_back(count_tcu[b]);
            }
            v_tcu.atomic.push_back(0);
            v_tcu.window.push_back(i);
        }else{
            // 对block太多的块进行分割
            int part_number = blocks / partSize_t;
            int block_residue = blocks % partSize_t;
            for(int j=0; j<part_number; j++)
            {
                v_tcu.row.push_back(partSize_t);
                for(int b=0; b<partSize_t; b++)
                {
                    int offset = j*partSize_t + b;
                    // int offset1 = offset + b;
                    // int sum = std::accumulate(count_tcu.begin()+offset, count_tcu.begin()+offset1, 0);
                    v_tcu.t_block_offset.push_back(count_tcu[offset]);
                }
                v_tcu.window.push_back(i);
                v_tcu.atomic.push_back(1);
            }
            if(block_residue > 0)
            {             
                v_tcu.row.push_back(block_residue);
                for(int b=0; b<block_residue; b++)
                {
                    int offset = part_number*partSize_t+b;
                    // int offset1 = offset + b;
                    // int sum = std::accumulate(count_tcu.begin()+offset, count_tcu.begin()+offset1, 0);
                    v_tcu.t_block_offset.push_back(count_tcu[offset]);
                }
                v_tcu.window.push_back(i);
                v_tcu.atomic.push_back(1);
            }
        }
        /*
            4. 汇总三部分的结果
        */
        #pragma omp critical  
        res_t[i]=v_tcu;
        
    
    }
    /*
        5. 整合所有window的数据
    */
    //tcu
    std::vector<int> t_rowkNew_offset;
    t_rowkNew_offset.push_back(0);
    std::vector<int> t_blockNew_offset;
    t_blockNew_offset.push_back(0);
    // std::vector<float> t_valueNew;
    std::vector<int> t_columnNew;
    std::vector<long> t_binaryNew;
    // std::vector<int> t_rowNew;
    // std::vector<int> t_colNew;
    std::vector<int> t_window_rowNew;
    std::vector<int> t_atomicNew;

    //按顺序整合res
    //tcu
    for (const auto& pair : res_t) {
        if (pair.second.tcu_flag == true )
        {
            for(int sub : pair.second.row)
            {
                t_rowkNew_offset.push_back(t_rowkNew_offset.back()+sub);
            }  
            for(int sub : pair.second.t_block_offset)
            {
                t_blockNew_offset.push_back(t_blockNew_offset.back()+sub);
            }  
            // t_valueNew.insert(t_valueNew.end(),pair.second.t_value.begin(),pair.second.t_value.end());
            t_columnNew.insert(t_columnNew.end(),pair.second.t_column.begin(),pair.second.t_column.end());
            t_window_rowNew.insert(t_window_rowNew.end(),pair.second.window.begin(),pair.second.window.end());
            t_atomicNew.insert(t_atomicNew.end(),pair.second.atomic.begin(),pair.second.atomic.end());
            t_binaryNew.insert(t_binaryNew.end(),pair.second.t_binary.begin(),pair.second.t_binary.end());
        }   
    }
    // std::cout << t_rowkNew_offset.size() <<std::endl;
    auto t_rowNew_offsetTensor1 = torch::from_blob(t_rowkNew_offset.data(), t_rowkNew_offset.size(), torch::kInt32);
    auto t_blockNew_offsetTensor1 = torch::from_blob(t_blockNew_offset.data(), t_blockNew_offset.size(), torch::kInt32);
    //auto t_valueTensor1 = torch::from_blob(t_valueNew.data(), t_valueNew.size(), torch::kFloat32);
    auto t_columnTensor1 = torch::from_blob(t_columnNew.data(), t_columnNew.size(), torch::kInt32);
    auto t_window_rowTensor1 = torch::from_blob(t_window_rowNew.data(), t_window_rowNew.size(), torch::kInt32);
    auto t_atomicTensor1 = torch::from_blob(t_atomicNew.data(), t_atomicNew.size(), torch::kInt32);
    auto t_binaryTensor1 = torch::from_blob(t_binaryNew.data(), t_binaryNew.size(), torch::kInt64);

    torch::Tensor t_rowNew_offsetTensor = torch::empty_like(t_rowNew_offsetTensor1);
    t_rowNew_offsetTensor.copy_(t_rowNew_offsetTensor1);
    torch::Tensor t_blockNew_offsetTensor = torch::empty_like(t_blockNew_offsetTensor1);
    t_blockNew_offsetTensor.copy_(t_blockNew_offsetTensor1);
    // torch::Tensor t_valueTensor = torch::empty_like(t_valueTensor1);
    // t_valueTensor.copy_(t_valueTensor1);
    torch::Tensor t_columnTensor = torch::empty_like(t_columnTensor1);
    t_columnTensor.copy_(t_columnTensor1);
    torch::Tensor t_window_rowTensor = torch::empty_like(t_window_rowTensor1);
    t_window_rowTensor.copy_(t_window_rowTensor1);
    torch::Tensor t_atomicTensor = torch::empty_like(t_atomicTensor1);
    t_atomicTensor.copy_(t_atomicTensor1);
    torch::Tensor t_binaryTensor = torch::empty_like(t_binaryTensor1);
    t_binaryTensor.copy_(t_binaryTensor1);

   
    return {t_rowNew_offsetTensor, t_blockNew_offsetTensor,t_columnTensor, t_window_rowTensor, t_atomicTensor, t_binaryTensor};

}

std::vector<torch::Tensor> block_csr_binary_sddmm_tcu_part_c_kernel(torch::Tensor row1, torch::Tensor column1, 
int partSize_t, int window, int wide)
{
    auto *row=row1.data_ptr<int>();
    auto column=column1.accessor<int, 1>();
    // auto values=values1.accessor<float, 1>();
    int rows=row1.size(0)-1;
    // Result
    std::map<int, Value_tcu_csr> res_t;
    //需要的window数量
    int num_windows = rows/window;
    #pragma omp parallel for
    for(int i=0;i<num_windows;i++)
    {
        /*
            1. 统计当前window中的vector, 并按照稠密度降序排列
        */
        //对column进行合并
        std::vector<int> mergedVector(&column[row[i*window]], &column[row[i*window+window]]);
        // 使用 unordered_map 统计 vector 中每个元素出现次数
        std::unordered_map<int, int> elementCounts;
        for (const auto& element : mergedVector) elementCounts[element]++;
        // 将统计结果放入 vector<pair> 中
        std::vector<std::pair<int, int>> countVector(elementCounts.begin(), elementCounts.end());
        // 按元素个数降序排序
        // std::sort(countVector.begin(), countVector.end(), [](const auto& a, const auto& b) {
        //     return a.second > b.second;
        // });

        /*
            2. 确定需要进行tcu计算的vector数量, t_percen
        */
        int size = countVector.size();
        int pad = wide - size%wide;
        //统计每个block中的非零元数量，计算稠密度
        int t_percen = size; //确定前多少个vector会被用作tcu进行稠密计算
        if(size%wide > 0)
        t_percen +=pad;

        std::sort(countVector.begin(), countVector.end());

        /*
            3. 根据t_percen分别划分TCU和CUDA的负载
        */

        // TCU 划分
        // 3.1 根据partSize_t划分t_part
        //当前window中有几个block
        int blocks = t_percen/wide; 
        Value_tcu_csr v_tcu;
        if(blocks>0) {v_tcu.tcu_flag = true; }
        std::map<int, int> colmap;
        int c=0;
        for(auto col:countVector){
            v_tcu.t_column.push_back(col.first);
            colmap[col.first]=c++;
        }
        for(int r=0;r<pad;r++)
        v_tcu.t_column.push_back(-1);

        std::vector<std::vector<float>> t_value_temp = createVectorOfVectors_float(blocks);
        std::vector<long> t_binary_temp = createVectorOfVectors_v2(blocks*2);
        // 指示快内的行偏移
        int p=0;
        for(int j=i*window;j<(i+1)*window;j++)
        {
            //用来存储每行的cuda数据
            std::queue<int> c_col;
            std::queue<int> c_value;
            for(int m=row[j];m<row[j+1];m++)
            {
                int newCol = colmap[column[m]];
                int block = newCol/wide;
                int bInId= newCol%wide;
                t_value_temp[block].push_back(1);
                long a=1;
                if(bInId<8){
                    // long offset = bInId*window + p;
                    // t_binary_temp[block*2] |= a<<offset;
                    long offset = p*(wide/2) + bInId;
                    t_binary_temp[block*2] |= a<<offset;
                }else{
                    bInId = bInId-8;
                    long offset = p*(wide/2) + bInId;
                    t_binary_temp[block*2 + 1] |= a<<offset;
                }
            }
            p++;
        }
        //整合vector
        v_tcu.t_value = mergeVectors_float(t_value_temp);
        v_tcu.t_binary = t_binary_temp;
        
        std::vector<int> count_tcu = countVectors(t_value_temp);
        if(blocks <= partSize_t)
        {
            v_tcu.row.push_back(blocks);
            //循放入每个block中的非零元数量
            for(int b=0; b<blocks; b++)
            {
                // int sum = std::accumulate(count_tcu.begin()+b, count_tcu.begin()+b+1, 0);
                v_tcu.t_block_offset.push_back(count_tcu[b]);
            }
            v_tcu.atomic.push_back(0);
            v_tcu.window.push_back(i);
        }else{
            // 对block太多的块进行分割
            int part_number = blocks / partSize_t;
            int block_residue = blocks % partSize_t;
            for(int j=0; j<part_number; j++)
            {
                v_tcu.row.push_back(partSize_t);
                for(int b=0; b<partSize_t; b++)
                {
                    int offset = j*partSize_t + b;
                    // int offset1 = offset + b;
                    // int sum = std::accumulate(count_tcu.begin()+offset, count_tcu.begin()+offset1, 0);
                    v_tcu.t_block_offset.push_back(count_tcu[offset]);
                }
                v_tcu.window.push_back(i);
                v_tcu.atomic.push_back(1);
            }
            if(block_residue > 0)
            {             
                v_tcu.row.push_back(block_residue);
                for(int b=0; b<block_residue; b++)
                {
                    int offset = part_number*partSize_t+b;
                    // int offset1 = offset + b;
                    // int sum = std::accumulate(count_tcu.begin()+offset, count_tcu.begin()+offset1, 0);
                    v_tcu.t_block_offset.push_back(count_tcu[offset]);
                }
                v_tcu.window.push_back(i);
                v_tcu.atomic.push_back(1);
            }
        }
        /*
            4. 汇总三部分的结果
        */
        #pragma omp critical  
        res_t[i]=v_tcu;
        
    
    }
    /*
        5. 整合所有window的数据
    */
    //tcu
    std::vector<int> t_rowkNew_offset;
    t_rowkNew_offset.push_back(0);
    std::vector<int> t_blockNew_offset;
    t_blockNew_offset.push_back(0);
    // std::vector<float> t_valueNew;
    std::vector<int> t_columnNew;
    std::vector<long> t_binaryNew;
    // std::vector<int> t_rowNew;
    // std::vector<int> t_colNew;
    std::vector<int> t_window_rowNew;
    std::vector<int> t_atomicNew;

    //按顺序整合res
    //tcu
    for (const auto& pair : res_t) {
        if (pair.second.tcu_flag == true )
        {
            for(int sub : pair.second.row)
            {
                t_rowkNew_offset.push_back(t_rowkNew_offset.back()+sub);
            }  
            for(int sub : pair.second.t_block_offset)
            {
                t_blockNew_offset.push_back(t_blockNew_offset.back()+sub);
            }  
            // t_valueNew.insert(t_valueNew.end(),pair.second.t_value.begin(),pair.second.t_value.end());
            t_columnNew.insert(t_columnNew.end(),pair.second.t_column.begin(),pair.second.t_column.end());
            t_window_rowNew.insert(t_window_rowNew.end(),pair.second.window.begin(),pair.second.window.end());
            t_atomicNew.insert(t_atomicNew.end(),pair.second.atomic.begin(),pair.second.atomic.end());
            t_binaryNew.insert(t_binaryNew.end(),pair.second.t_binary.begin(),pair.second.t_binary.end());
        }   
    }
    // std::cout << t_rowkNew_offset.size() <<std::endl;
    auto t_rowNew_offsetTensor1 = torch::from_blob(t_rowkNew_offset.data(), t_rowkNew_offset.size(), torch::kInt32);
    auto t_blockNew_offsetTensor1 = torch::from_blob(t_blockNew_offset.data(), t_blockNew_offset.size(), torch::kInt32);
    //auto t_valueTensor1 = torch::from_blob(t_valueNew.data(), t_valueNew.size(), torch::kFloat32);
    auto t_columnTensor1 = torch::from_blob(t_columnNew.data(), t_columnNew.size(), torch::kInt32);
    auto t_window_rowTensor1 = torch::from_blob(t_window_rowNew.data(), t_window_rowNew.size(), torch::kInt32);
    auto t_atomicTensor1 = torch::from_blob(t_atomicNew.data(), t_atomicNew.size(), torch::kInt32);
    auto t_binaryTensor1 = torch::from_blob(t_binaryNew.data(), t_binaryNew.size(), torch::kInt64);

    torch::Tensor t_rowNew_offsetTensor = torch::empty_like(t_rowNew_offsetTensor1);
    t_rowNew_offsetTensor.copy_(t_rowNew_offsetTensor1);
    torch::Tensor t_blockNew_offsetTensor = torch::empty_like(t_blockNew_offsetTensor1);
    t_blockNew_offsetTensor.copy_(t_blockNew_offsetTensor1);
    // torch::Tensor t_valueTensor = torch::empty_like(t_valueTensor1);
    // t_valueTensor.copy_(t_valueTensor1);
    torch::Tensor t_columnTensor = torch::empty_like(t_columnTensor1);
    t_columnTensor.copy_(t_columnTensor1);
    torch::Tensor t_window_rowTensor = torch::empty_like(t_window_rowTensor1);
    t_window_rowTensor.copy_(t_window_rowTensor1);
    torch::Tensor t_atomicTensor = torch::empty_like(t_atomicTensor1);
    t_atomicTensor.copy_(t_atomicTensor1);
    torch::Tensor t_binaryTensor = torch::empty_like(t_binaryTensor1);
    t_binaryTensor.copy_(t_binaryTensor1);

   
    return {t_rowNew_offsetTensor, t_blockNew_offsetTensor,t_columnTensor, t_window_rowTensor, t_atomicTensor, t_binaryTensor};

}

std::vector<torch::Tensor> block_csr_binary_sddmm_tcu_part_c1_kernel(torch::Tensor row1, torch::Tensor column1, 
int partSize_t, int window, int wide)
{
    auto *row=row1.data_ptr<int>();
    auto column=column1.accessor<int, 1>();
    // auto values=values1.accessor<float, 1>();
    int rows=row1.size(0)-1;
    // Result
    std::map<int, Value_tcu_csr> res_t;
    //需要的window数量
    int num_windows = rows/window;
    #pragma omp parallel for
    for(int i=0;i<num_windows;i++)
    {
        /*
            1. 统计当前window中的vector, 并按照稠密度降序排列
        */
        //对column进行合并
        std::vector<int> mergedVector(&column[row[i*window]], &column[row[i*window+window]]);
        // 使用 unordered_map 统计 vector 中每个元素出现次数
        std::unordered_map<int, int> elementCounts;
        for (const auto& element : mergedVector) elementCounts[element]++;
        // 将统计结果放入 vector<pair> 中
        std::vector<std::pair<int, int>> countVector(elementCounts.begin(), elementCounts.end());
        // 按元素个数降序排序
        // std::sort(countVector.begin(), countVector.end(), [](const auto& a, const auto& b) {
        //     return a.second > b.second;
        // });

        /*
            2. 确定需要进行tcu计算的vector数量, t_percen
        */
        int size = countVector.size();
        int pad = wide - size%wide;
        //统计每个block中的非零元数量，计算稠密度
        int t_percen = size; //确定前多少个vector会被用作tcu进行稠密计算
        if(size%wide > 0)
        t_percen +=pad;

        std::sort(countVector.begin(), countVector.end());

        /*
            3. 根据t_percen分别划分TCU和CUDA的负载
        */

        // TCU 划分
        // 3.1 根据partSize_t划分t_part
        //当前window中有几个block
        int blocks = t_percen/wide; 
        Value_tcu_csr v_tcu;
        if(blocks>0) {v_tcu.tcu_flag = true; }
        std::map<int, int> colmap;
        int c=0;
        for(auto col:countVector){
            v_tcu.t_column.push_back(col.first);
            colmap[col.first]=c++;
        }
        for(int r=0;r<pad;r++)
        v_tcu.t_column.push_back(-1);

        std::vector<std::vector<float>> t_value_temp = createVectorOfVectors_float(blocks);
        std::vector<long> t_binary_temp = createVectorOfVectors_v2(blocks*2);
        // 指示快内的行偏移
        int p=0;
        for(int j=i*window;j<(i+1)*window;j++)
        {
            //用来存储每行的cuda数据
            for(int m=row[j];m<row[j+1];m++)
            {
                int newCol = colmap[column[m]];
                int block = newCol/wide;
                int bInId= newCol%wide;
                t_value_temp[block].push_back(1);
                long a=1;
                //获取当前非零元在8x16 block内的偏移
                long offset = bInId*window + p;
                //获取当前值属于哪个线程
                int tid = (offset%64)/2;
                //获取当前值是当前线程中的第几个寄存器的值(c0,c1,c2,c3)
                int tid_inoffset =  (offset/64)*2 + (offset%64)%2;
                //块内偏移，t0-15在binary0， t16-31在15在binary1
                long local = (tid%16)*4 + tid_inoffset;
                if(tid<16){
                    t_binary_temp[block*2] |= a<<local;
                }else{
                    t_binary_temp[block*2 + 1] |= a<<local;
                }
            }
            p++;
        }
        //整合vector
        v_tcu.t_value = mergeVectors_float(t_value_temp);
        v_tcu.t_binary = t_binary_temp;
        
        std::vector<int> count_tcu = countVectors(t_value_temp);
        if(blocks <= partSize_t)
        {
            v_tcu.row.push_back(blocks);
            //循放入每个block中的非零元数量
            for(int b=0; b<blocks; b++)
            {
                // int sum = std::accumulate(count_tcu.begin()+b, count_tcu.begin()+b+1, 0);
                v_tcu.t_block_offset.push_back(count_tcu[b]);
            }
            v_tcu.atomic.push_back(0);
            v_tcu.window.push_back(i);
        }else{
            // 对block太多的块进行分割
            int part_number = blocks / partSize_t;
            int block_residue = blocks % partSize_t;
            for(int j=0; j<part_number; j++)
            {
                v_tcu.row.push_back(partSize_t);
                for(int b=0; b<partSize_t; b++)
                {
                    int offset = j*partSize_t + b;
                    // int offset1 = offset + b;
                    // int sum = std::accumulate(count_tcu.begin()+offset, count_tcu.begin()+offset1, 0);
                    v_tcu.t_block_offset.push_back(count_tcu[offset]);
                }
                v_tcu.window.push_back(i);
                v_tcu.atomic.push_back(1);
            }
            if(block_residue > 0)
            {             
                v_tcu.row.push_back(block_residue);
                for(int b=0; b<block_residue; b++)
                {
                    int offset = part_number*partSize_t+b;
                    // int offset1 = offset + b;
                    // int sum = std::accumulate(count_tcu.begin()+offset, count_tcu.begin()+offset1, 0);
                    v_tcu.t_block_offset.push_back(count_tcu[offset]);
                }
                v_tcu.window.push_back(i);
                v_tcu.atomic.push_back(1);
            }
        }
        /*
            4. 汇总三部分的结果
        */
        #pragma omp critical  
        res_t[i]=v_tcu;
        
    
    }
    /*
        5. 整合所有window的数据
    */
    //tcu
    std::vector<int> t_rowkNew_offset;
    t_rowkNew_offset.push_back(0);
    std::vector<int> t_blockNew_offset;
    t_blockNew_offset.push_back(0);
    // std::vector<float> t_valueNew;
    std::vector<int> t_columnNew;
    std::vector<long> t_binaryNew;
    // std::vector<int> t_rowNew;
    // std::vector<int> t_colNew;
    std::vector<int> t_window_rowNew;
    std::vector<int> t_atomicNew;

    //按顺序整合res
    //tcu
    for (const auto& pair : res_t) {
        if (pair.second.tcu_flag == true )
        {
            for(int sub : pair.second.row)
            {
                t_rowkNew_offset.push_back(t_rowkNew_offset.back()+sub);
            }  
            for(int sub : pair.second.t_block_offset)
            {
                t_blockNew_offset.push_back(t_blockNew_offset.back()+sub);
            }  
            // t_valueNew.insert(t_valueNew.end(),pair.second.t_value.begin(),pair.second.t_value.end());
            t_columnNew.insert(t_columnNew.end(),pair.second.t_column.begin(),pair.second.t_column.end());
            t_window_rowNew.insert(t_window_rowNew.end(),pair.second.window.begin(),pair.second.window.end());
            t_atomicNew.insert(t_atomicNew.end(),pair.second.atomic.begin(),pair.second.atomic.end());
            t_binaryNew.insert(t_binaryNew.end(),pair.second.t_binary.begin(),pair.second.t_binary.end());
        }   
    }
    // std::cout << t_rowkNew_offset.size() <<std::endl;
    auto t_rowNew_offsetTensor1 = torch::from_blob(t_rowkNew_offset.data(), t_rowkNew_offset.size(), torch::kInt32);
    auto t_blockNew_offsetTensor1 = torch::from_blob(t_blockNew_offset.data(), t_blockNew_offset.size(), torch::kInt32);
    //auto t_valueTensor1 = torch::from_blob(t_valueNew.data(), t_valueNew.size(), torch::kFloat32);
    auto t_columnTensor1 = torch::from_blob(t_columnNew.data(), t_columnNew.size(), torch::kInt32);
    auto t_window_rowTensor1 = torch::from_blob(t_window_rowNew.data(), t_window_rowNew.size(), torch::kInt32);
    auto t_atomicTensor1 = torch::from_blob(t_atomicNew.data(), t_atomicNew.size(), torch::kInt32);
    auto t_binaryTensor1 = torch::from_blob(t_binaryNew.data(), t_binaryNew.size(), torch::kInt64);

    torch::Tensor t_rowNew_offsetTensor = torch::empty_like(t_rowNew_offsetTensor1);
    t_rowNew_offsetTensor.copy_(t_rowNew_offsetTensor1);
    torch::Tensor t_blockNew_offsetTensor = torch::empty_like(t_blockNew_offsetTensor1);
    t_blockNew_offsetTensor.copy_(t_blockNew_offsetTensor1);
    // torch::Tensor t_valueTensor = torch::empty_like(t_valueTensor1);
    // t_valueTensor.copy_(t_valueTensor1);
    torch::Tensor t_columnTensor = torch::empty_like(t_columnTensor1);
    t_columnTensor.copy_(t_columnTensor1);
    torch::Tensor t_window_rowTensor = torch::empty_like(t_window_rowTensor1);
    t_window_rowTensor.copy_(t_window_rowTensor1);
    torch::Tensor t_atomicTensor = torch::empty_like(t_atomicTensor1);
    t_atomicTensor.copy_(t_atomicTensor1);
    torch::Tensor t_binaryTensor = torch::empty_like(t_binaryTensor1);
    t_binaryTensor.copy_(t_binaryTensor1);

   
    return {t_rowNew_offsetTensor, t_blockNew_offsetTensor,t_columnTensor, t_window_rowTensor, t_atomicTensor, t_binaryTensor};

}
//nnz profile
// tcu+cuda v3
float nnz_profile_kernel(torch::Tensor row1, torch::Tensor column1, int window, int density)
{
    auto *row=row1.data_ptr<int>();
    auto column=column1.accessor<int, 1>();
    int rows=row1.size(0)-1;
    //需要的window数量
    int num_windows = rows/window;
    int nnz_total;
    int rows_total;
    #pragma omp parallel for
    for(int i=0;i<num_windows;i++)
    {
        /*
            1. 统计当前window中的vector, 并按照稠密度降序排列
        */
        //对column进行合并
        std::vector<int> mergedVector(&column[row[i*window]], &column[row[i*window+window]]);
        // 使用 unordered_map 统计 vector 中每个元素出现次数
        std::unordered_map<int, int> elementCounts;
        for (const auto& element : mergedVector) elementCounts[element]++;
        // 将统计结果放入 vector<pair> 中
        std::vector<std::pair<int, int>> countVector(elementCounts.begin(), elementCounts.end());
        // 按元素个数升序排序
        std::sort(countVector.begin(), countVector.end(), [](const auto& a, const auto& b) {
            return a.second < b.second;
        });

        int size = countVector.size();
        //统计每个block中的非零元数量，计算稠密度
        int t_percen = 0; //确定前多少个vector会被用作tcu进行稠密计算
        t_percen = std::count_if(countVector.begin(), countVector.end(), [density](const auto& pair) {
            return pair.second >= density;
        });

        // 将前 k 个数按升序排列
        std::sort(countVector.begin(), countVector.begin() + t_percen);
        // 将剩下的数按升序排列
        std::sort(countVector.begin() + t_percen, countVector.end());
        /*
            3. 根据t_percen分别划分TCU和CUDA的负载
        */

        // TCU 划分
        // 3.1 根据partSize_t划分t_part
        //当前window中有几个block
        std::map<int, int> colmap;
        int c=0;
        for(auto col:countVector){
            colmap[col.first]=c++;
        }

        // 指示快内的行偏移
        int p=0;
        int nnz = 0;
        std::set<int> row_set;
        for(int j=i*window;j<(i+1)*window;j++)
        {
            for(int m=row[j];m<row[j+1];m++)
            {
                int newCol = colmap[column[m]];
                if(newCol<t_percen)
                {
                    row_set.insert(p);
                    nnz++;
                }
            }
            p++;
        }
    
        #pragma omp critical  
        {
            nnz_total += nnz;
            rows_total += row_set.size();
        }
    
    }

    return (float)nnz_total/rows_total;

}


std::vector<torch::Tensor> block_csr_tcu_part_kernel(torch::Tensor row1, torch::Tensor column1, torch::Tensor values1, 
int partSize_t, int window, int wide)
{
    auto *row=row1.data_ptr<int>();
    auto column=column1.accessor<int, 1>();
    auto values=values1.accessor<float, 1>();
    int rows=row1.size(0)-1;
    // Result
    std::map<int, Value_tcu_csr> res_t;
    //需要的window数量
    int num_windows = rows/window;
    #pragma omp parallel for
    for(int i=0;i<num_windows;i++)
    {
        /*
            1. 统计当前window中的vector, 并按照稠密度降序排列
        */
        //对column进行合并
        std::vector<int> mergedVector(&column[row[i*window]], &column[row[i*window+window]]);
        // 使用 unordered_map 统计 vector 中每个元素出现次数
        std::unordered_map<int, int> elementCounts;
        for (const auto& element : mergedVector) elementCounts[element]++;
        // 将统计结果放入 vector<pair> 中
        std::vector<std::pair<int, int>> countVector(elementCounts.begin(), elementCounts.end());
        // 按元素个数降序排序
        // std::sort(countVector.begin(), countVector.end(), [](const auto& a, const auto& b) {
        //     return a.second > b.second;
        // });

        /*
            2. 确定需要进行tcu计算的vector数量, t_percen
        */
        int size = countVector.size();
        int pad = wide - size%wide;
        //统计每个block中的非零元数量，计算稠密度
        int t_percen = size; //确定前多少个vector会被用作tcu进行稠密计算
        if(size%wide > 0)
        t_percen +=pad;

        std::sort(countVector.begin(), countVector.end());

        /*
            3. 根据t_percen分别划分TCU和CUDA的负载
        */

        // TCU 划分
        // 3.1 根据partSize_t划分t_part
        //当前window中有几个block
        int blocks = t_percen/wide; 
        Value_tcu_csr v_tcu;
        if(blocks>0) {v_tcu.tcu_flag = true; }
        std::map<int, int> colmap;
        int c=0;
        for(auto col:countVector){
            v_tcu.t_column.push_back(col.first);
            colmap[col.first]=c++;
        }
        for(int r=0;r<pad;r++)
        v_tcu.t_column.push_back(-1);

        std::vector<std::vector<float>> t_value_temp = createVectorOfVectors_float(blocks);
        std::vector<std::vector<int>> t_row_temp = createVectorOfVectors(blocks);
        // std::vector<std::vector<int>> t_col_temp = createVectorOfVectors(blocks);
        // 指示快内的行偏移
        int p=0;
        for(int j=i*window;j<(i+1)*window;j++)
        {
            //用来存储每行的cuda数据
            for(int m=row[j];m<row[j+1];m++)
            {
                int newCol = colmap[column[m]];
                int block = newCol/wide;

                // int bId=newCol/wide;
                // int bInId=newCol%wide;
                t_value_temp[block].push_back(values[m]);
                t_row_temp[block].push_back((p*wide) + (newCol%wide));
                // t_row_temp[block].push_back(p);
                // t_col_temp[block].push_back(newCol%wide);
            }
            p++;
        }
        //整合vector
        v_tcu.t_value = mergeVectors_float(t_value_temp);
        v_tcu.t_row = mergeVectors(t_row_temp);
        // v_tcu.t_col = mergeVectors(t_col_temp);
        
        std::vector<int> count_tcu = countVectors(t_value_temp);
        if(blocks <= partSize_t)
        {
            v_tcu.row.push_back(blocks);
            //循放入每个block中的非零元数量
            for(int b=0; b<blocks; b++)
            {
                // int sum = std::accumulate(count_tcu.begin()+b, count_tcu.begin()+b+1, 0);
                v_tcu.t_block_offset.push_back(count_tcu[b]);
            }
            v_tcu.atomic.push_back(0);
            v_tcu.window.push_back(i);
        }else{
            // 对block太多的块进行分割
            int part_number = blocks / partSize_t;
            int block_residue = blocks % partSize_t;
            for(int j=0; j<part_number; j++)
            {
                v_tcu.row.push_back(partSize_t);
                for(int b=0; b<partSize_t; b++)
                {
                    int offset = j*partSize_t + b;
                    // int offset1 = offset + b;
                    // int sum = std::accumulate(count_tcu.begin()+offset, count_tcu.begin()+offset1, 0);
                    v_tcu.t_block_offset.push_back(count_tcu[offset]);
                }
                v_tcu.window.push_back(i);
                v_tcu.atomic.push_back(1);
            }
            if(block_residue > 0)
            {             
                v_tcu.row.push_back(block_residue);
                for(int b=0; b<block_residue; b++)
                {
                    int offset = part_number*partSize_t+b;
                    // int offset1 = offset + b;
                    // int sum = std::accumulate(count_tcu.begin()+offset, count_tcu.begin()+offset1, 0);
                    v_tcu.t_block_offset.push_back(count_tcu[offset]);
                }
                v_tcu.window.push_back(i);
                v_tcu.atomic.push_back(1);
            }
        }
        /*
            4. 汇总三部分的结果
        */
        #pragma omp critical  
        res_t[i]=v_tcu;
        
    
    }
    /*
        5. 整合所有window的数据
    */
    //tcu
    std::vector<int> t_rowkNew_offset;
    t_rowkNew_offset.push_back(0);
    std::vector<int> t_blockNew_offset;
    t_blockNew_offset.push_back(0);
    std::vector<float> t_valueNew;
    std::vector<int> t_columnNew;
    std::vector<int> t_rowNew;
    // std::vector<int> t_colNew;
    std::vector<int> t_window_rowNew;
    std::vector<int> t_atomicNew;

    //按顺序整合res
    //tcu
    for (const auto& pair : res_t) {
        if (pair.second.tcu_flag == true )
        {
            for(int sub : pair.second.row)
            {
                t_rowkNew_offset.push_back(t_rowkNew_offset.back()+sub);
            }  
            for(int sub : pair.second.t_block_offset)
            {
                t_blockNew_offset.push_back(t_blockNew_offset.back()+sub);
            }  
            t_valueNew.insert(t_valueNew.end(),pair.second.t_value.begin(),pair.second.t_value.end());
            t_columnNew.insert(t_columnNew.end(),pair.second.t_column.begin(),pair.second.t_column.end());
            t_window_rowNew.insert(t_window_rowNew.end(),pair.second.window.begin(),pair.second.window.end());
            t_atomicNew.insert(t_atomicNew.end(),pair.second.atomic.begin(),pair.second.atomic.end());
            t_rowNew.insert(t_rowNew.end(),pair.second.t_row.begin(),pair.second.t_row.end());
            // t_colNew.insert(t_colNew.end(),pair.second.t_col.begin(),pair.second.t_col.end());
        }   
    }
    // std::cout << t_rowkNew_offset.size() <<std::endl;
    auto t_rowNew_offsetTensor1 = torch::from_blob(t_rowkNew_offset.data(), t_rowkNew_offset.size(), torch::kInt32);
    auto t_blockNew_offsetTensor1 = torch::from_blob(t_blockNew_offset.data(), t_blockNew_offset.size(), torch::kInt32);
    auto t_valueTensor1 = torch::from_blob(t_valueNew.data(), t_valueNew.size(), torch::kFloat32);
    auto t_columnTensor1 = torch::from_blob(t_columnNew.data(), t_columnNew.size(), torch::kInt32);
    auto t_window_rowTensor1 = torch::from_blob(t_window_rowNew.data(), t_window_rowNew.size(), torch::kInt32);
    auto t_atomicTensor1 = torch::from_blob(t_atomicNew.data(), t_atomicNew.size(), torch::kInt32);
    auto t_rowTensor1 = torch::from_blob(t_rowNew.data(), t_rowNew.size(), torch::kInt32);
    //auto t_colTensor1 = torch::from_blob(t_colNew.data(), t_colNew.size(), torch::kInt32);

    torch::Tensor t_rowNew_offsetTensor = torch::empty_like(t_rowNew_offsetTensor1);
    t_rowNew_offsetTensor.copy_(t_rowNew_offsetTensor1);
    torch::Tensor t_blockNew_offsetTensor = torch::empty_like(t_blockNew_offsetTensor1);
    t_blockNew_offsetTensor.copy_(t_blockNew_offsetTensor1);
    torch::Tensor t_valueTensor = torch::empty_like(t_valueTensor1);
    t_valueTensor.copy_(t_valueTensor1);
    torch::Tensor t_columnTensor = torch::empty_like(t_columnTensor1);
    t_columnTensor.copy_(t_columnTensor1);
    torch::Tensor t_window_rowTensor = torch::empty_like(t_window_rowTensor1);
    t_window_rowTensor.copy_(t_window_rowTensor1);
    torch::Tensor t_atomicTensor = torch::empty_like(t_atomicTensor1);
    t_atomicTensor.copy_(t_atomicTensor1);
    torch::Tensor t_rowTensor = torch::empty_like(t_rowTensor1);
    t_rowTensor.copy_(t_rowTensor1);
    //torch::Tensor t_colTensor = torch::empty_like(t_colTensor1);
    //t_colTensor.copy_(t_colTensor1);

   
    return {t_rowNew_offsetTensor, t_blockNew_offsetTensor,t_columnTensor, t_valueTensor, t_window_rowTensor, t_atomicTensor, t_rowTensor};

}


std::vector<torch::Tensor> block_bcrs_tcu_kernel(torch::Tensor row1, torch::Tensor column1, int partSize, int window1, int wide1)
{
    auto *row=row1.data_ptr<int>();
    auto column=column1.accessor<int, 1>();
    // auto values=values1.accessor<float, 1>();
    int rows=row1.size(0)-1;
    int window = window1;
    int wide = wide1;
    int warps_block = 4;
    //需要的window数量
    int num_windows = rows/window;
    //最终的map
    std::map<int, Value1_5> res;
    int tcu_rows=0;
    //统计vector的稠密度，然后按照降序排列
    #pragma omp parallel for
    for(int i=0;i<num_windows;i++)
    {
        Value1_5 v;
        //对column进行合并
        std::vector<int> mergedVector(&column[row[i*window]], &column[row[i*window+window]]);
        
        // 使用 unordered_map 统计 vector 中每个元素出现次数
        std::unordered_map<int, int> elementCounts;

        for (const auto& element : mergedVector) {
            elementCounts[element]++;
        }
        // 将统计结果放入 vector<pair> 中
        std::vector<std::pair<int, int>> countVector(elementCounts.begin(), elementCounts.end());

        // 按元素个数降序排序
        // std::sort(countVector.begin(), countVector.end(), [](const auto& a, const auto& b) {
        //     return a.second > b.second;
        // });
        //获取countVector的长度
        int size = countVector.size();
        int residue = size%wide;
        v.t_block_offset.push_back(size);
        v.t_block_offset.push_back(residue);
        std::vector<float> demo((size+wide-residue)*window, float(0.0));
        if(size>0)
        {   
            // 将前 k 个数按升序排列
            std::sort(countVector.begin(), countVector.end());

            std::map<int, int> colmap;
            int c=0;
            for(auto col:countVector){
                v.t_column.push_back(col.first);
                colmap[col.first]=c++;
            }
            //把额外的1放进去
            if((wide-residue)>0){
                for(int j=0; j<(wide-residue); j++)
                 v.t_column.push_back(-1);
            }

            int cur_row = 0;
            // 遍历window中的每一行
            for(int j=i*window; j<(i+1)*window; j++)
            {
                // 遍历行中的每个元素
                for(int m=row[j];m<row[j+1];m++)
                {
                    int newCol = colmap[column[m]];
                    int bId = newCol/wide;
                    int bInId= newCol%wide;
                    demo[bId*window*wide + cur_row*wide +bInId] = 1.0;
                }
                cur_row++;
            }
        }
        v.t_value = demo;
        #pragma omp critical  
        {  res[i]=v;}
    }
    // std::cout << "tcu rows:" << tcu_rows << std::endl;
    //记录每个window中的vector数量
    std::vector<int> t_blockNew_offset;
    t_blockNew_offset.push_back(0);
    std::vector<float> t_valueNew;
    std::vector<int> t_columnNew;


    //按顺序整合res
    for (const auto& pair : res) {
        // if(pair.second.tcu_flag)
        // {
            //首先整合TCU的数据
            for(int sub : pair.second.t_block_offset)
            {
                t_blockNew_offset.push_back(t_blockNew_offset.back()+sub);
            }  
            t_valueNew.insert(t_valueNew.end(),pair.second.t_value.begin(),pair.second.t_value.end());
            t_columnNew.insert(t_columnNew.end(),pair.second.t_column.begin(),pair.second.t_column.end());
        // }
    }
    auto t_blockNew_offsetTensor1 = torch::from_blob(t_blockNew_offset.data(), t_blockNew_offset.size(), torch::kInt32);
    auto t_valueTensor1 = torch::from_blob(t_valueNew.data(), t_valueNew.size(), torch::kFloat32);
    auto t_columnTensor1 = torch::from_blob(t_columnNew.data(), t_columnNew.size(), torch::kInt32);
    // auto t_window_rowTensor1 = torch::from_blob(t_window_rowNew.data(), t_window_rowNew.size(), torch::kInt32);

    torch::Tensor t_blockNew_offsetTensor = torch::empty_like(t_blockNew_offsetTensor1);
    t_blockNew_offsetTensor.copy_(t_blockNew_offsetTensor1);
    torch::Tensor t_valueTensor = torch::empty_like(t_valueTensor1);
    t_valueTensor.copy_(t_valueTensor1);
    torch::Tensor t_columnTensor = torch::empty_like(t_columnTensor1);
    t_columnTensor.copy_(t_columnTensor1);


    // torch::Tensor t_window_rowTensor = torch::empty_like(t_window_rowTensor1);
    // t_window_rowTensor.copy_(t_window_rowTensor1);

    return {t_blockNew_offsetTensor, t_columnTensor, t_valueTensor};

}




// m-tcu + cuda + partSize + binary
std::vector<torch::Tensor> block_tf32_tcu_cuda_binary_kernel_seq(torch::Tensor row1, torch::Tensor column1, torch::Tensor values1, 
int partSize_t, int partSize_c, int shortSize, int density, int window, int wide)
{
    auto *row=row1.data_ptr<int>();
    auto column=column1.accessor<int, 1>();
    auto values=values1.accessor<float, 1>();

    auto start = std::chrono::high_resolution_clock::now();

    int rows=row1.size(0)-1;
    // Result
    std::map<int, Value_tcu_binary> res_t;
    std::map<int, std::vector<Value_cuda>> res_c;
    std::map<int, std::vector<Value_cuda>> res_c_residue;
    //需要的window数量
    int num_windows = rows/window;
    #pragma omp parallel for
    for(int i=0;i<num_windows;i++)
    {
        /*
            1. 统计当前window中的vector, 并按照稠密度降序排列
        */
        //对column进行合并
        std::vector<int> mergedVector(&column[row[i*window]], &column[row[i*window+window]]);
        // 使用 unordered_map 统计 vector 中每个元素出现次数
        std::unordered_map<int, int> elementCounts;
        for (const auto& element : mergedVector) elementCounts[element]++;
        // 将统计结果放入 vector<pair> 中
        std::vector<std::pair<int, int>> countVector(elementCounts.begin(), elementCounts.end());
        // 按元素个数降序排序
        std::sort(countVector.begin(), countVector.end(), [](const auto& a, const auto& b) {
            return a.second > b.second;
        });

        /*
            2. 确定需要进行tcu计算的vector数量, t_percen
        */
        int size = countVector.size();
        int residue = size%wide;
        //统计每个block中的非零元数量，计算稠密度
        int t_percen = 0; //确定前多少个vector会被用作tcu进行稠密计算
        t_percen = std::count_if(countVector.begin(), countVector.end(), [density](const auto& pair) {
            return pair.second >= density;
        });
        int t_percen_ori = t_percen;
        if((t_percen%wide) != 0)
        {
            // if((t_percen + (wide - t_percen%wide)) <= size) 
            t_percen = (t_percen + (wide - t_percen%wide));
            // else
            // t_percen = (t_percen - (t_percen%wide));

        }
        // if(t_percen<=size){
        // // 将前 k 个数按升序排列
        // std::sort(countVector.begin(), countVector.begin() + t_percen);
        // // 将剩下的数按升序排列
        // std::sort(countVector.begin() + t_percen, countVector.end());
        // }else{
        //     std::sort(countVector.begin(), countVector.end());
        // }
        std::sort(countVector.begin(), countVector.begin()+t_percen_ori);
        std::sort(countVector.begin() + t_percen_ori, countVector.end());
        /*
            3. 根据t_percen分别划分TCU和CUDA的负载
        */

        // TCU 划分
        // 3.1 根据partSize_t划分t_part
        //当前window中有几个block
        int blocks = t_percen/wide; 
        Value_tcu_binary v_tcu;
        if(blocks>0) {v_tcu.tcu_flag = true; }
        std::map<int, int> colmap;
        int c=0;
        for(auto col:countVector){
            if (c<t_percen_ori)
            {
                v_tcu.t_column.push_back(col.first);
            }
            colmap[col.first]=c++;
        }
        if((t_percen_ori%wide)>0){
            for(int tt=0;tt<(wide-(t_percen_ori%wide));tt++)
            v_tcu.t_column.push_back(-1);
        }
        //存放拆分的cuda长短行数据
        std::vector<Value_cuda> large;
        std::vector<Value_cuda> small;

        std::vector<std::vector<float>> t_value_temp = createVectorOfVectors_float(blocks);
        std::vector<long> t_binary_temp = createVectorOfVectors_v2(blocks);
        // std::vector<std::vector<int>> t_row_temp = createVectorOfVectors(blocks);
        // std::vector<std::vector<int>> t_col_temp = createVectorOfVectors(blocks);
        // 指示快内的行偏移
        int p=0;
        for(int j=i*window;j<(i+1)*window;j++)
        {
            //用来存储每行的cuda数据
            std::queue<int> c_col;
            std::queue<int> c_value;
            int cuda_nnz=0;
            for(int m=row[j];m<row[j+1];m++)
            {
                int newCol = colmap[column[m]];
                if(newCol<t_percen_ori)
                {
                    int block = newCol/wide;
                    int bInId= newCol%wide;
                    t_value_temp[block].push_back(values[m]);
                    long offset = p*wide + bInId;
                    long a=1;
                    t_binary_temp[block] |= a<<offset;
                }else{
                    //划分cuda部分
                    c_col.push(column[m]);
                    c_value.push(values[m]);
                    cuda_nnz++;
                }
            }
            //处理当前行剩余的cuda元素
            if(cuda_nnz>0)
            {
                if(cuda_nnz <= shortSize)
                {
                    Value_cuda temp;
                    temp.cuda_residue_flag = true;
                    temp.c_row_offset = cuda_nnz;
                    temp.c_row = i*window + p;
                        if(t_percen_ori>0) 
                        {
                            temp.c_atomic = 1;
                            temp.c_atomic_seq = 0;
                        }
                        else{
                             temp.c_atomic = 0;
                             temp.c_atomic_seq = 0;
                        }
                    for(int m=0; m<cuda_nnz; m++)
                    {
                        temp.c_column.push_back(c_col.front());
                        temp.c_value.push_back(c_value.front());
                        c_col.pop();
                        c_value.pop();
                    }
                    small.push_back(temp);
                }else{
                    if(cuda_nnz <= partSize_c)
                    {
                        Value_cuda temp;
                        temp.cuda_flag = true;
                        temp.c_row_offset = (cuda_nnz);
                        temp.c_row = i*window + p;
                            if(t_percen_ori>0) 
                            {
                                temp.c_atomic = 1;
                                temp.c_atomic_seq = 0;
                            }
                            else{
                                temp.c_atomic = 0;
                                temp.c_atomic_seq = 0;
                            }
                        for(int m=0; m<cuda_nnz; m++)
                        {
                            temp.c_column.push_back(c_col.front());
                            temp.c_value.push_back(c_value.front());
                            c_col.pop();
                            c_value.pop();
                        }
                        large.push_back(temp);
                    }else{
                        // 当前行的非零元被分成多个part，并需要原子操作
                        int parts_row = cuda_nnz/partSize_c;
                        // int cur_value_indice = row[i];
                        for(int cur_i = 0; cur_i<parts_row; cur_i++)
                        {
                            Value_cuda temp;
                            temp.cuda_flag = true;      
                            temp.c_row_offset = partSize_c;
                            temp.c_row = i*window + p;
                            temp.c_atomic = 1;
                            temp.c_atomic_seq = 1;
                            for(int m=0; m<partSize_c; m++)
                            {
                                temp.c_column.push_back(c_col.front());
                                temp.c_value.push_back(c_value.front());
                                c_col.pop();
                                c_value.pop();
                                // cur_value_indice++;
                            }
                            large.push_back(temp);
                        }
                        int residue  = (cuda_nnz%partSize_c);
                        if(residue>0){
                            Value_cuda temp;
                            temp.c_row_offset = residue;
                            temp.c_row = i*window + p;
                            temp.c_atomic = 1;
                            temp.c_atomic_seq = 1;
                            for(int m=0; m<residue; m++)
                            {
                                temp.c_column.push_back(c_col.front());
                                temp.c_value.push_back(c_value.front());
                                c_col.pop();
                                c_value.pop();
                                // cur_value_indice++;
                            }
                            if(residue <= shortSize)
                            {
                                temp.cuda_residue_flag = true;
                                small.push_back(temp);
                            }else{
                                temp.cuda_flag = true;
                                large.push_back(temp);
                            }
                        }
                    }
                }
            }
            p++;
        }
        //整合vector
        v_tcu.t_value = mergeVectors_float(t_value_temp);
        v_tcu.t_binary = t_binary_temp;
        // v_tcu.t_row = mergeVectors(t_row_temp);
        // v_tcu.t_col = mergeVectors(t_col_temp);
        
        std::vector<int> count_tcu = countVectors(t_value_temp);
        if(blocks <= partSize_t)
        {
            v_tcu.row.push_back(blocks);
            //循放入每个block中的非零元数量
            for(int b=0; b<blocks; b++)
            {
                // int sum = std::accumulate(count_tcu.begin()+b, count_tcu.begin()+b+1, 0);
                v_tcu.t_block_offset.push_back(count_tcu[b]);
            }
                if(t_percen_ori==size) 
                {
                    v_tcu.atomic.push_back(0);
                    v_tcu.atomic_seq.push_back(0);
                }
                else {
                    v_tcu.atomic.push_back(1);
                    v_tcu.atomic_seq.push_back(0);
                }
            // if(t_percen_ori==size) v_tcu.atomic.push_back(0);
            // else v_tcu.atomic.push_back(1);
            v_tcu.window.push_back(i);
        }else{
            // 对block太多的块进行分割
            int part_number = blocks / partSize_t;
            int block_residue = blocks % partSize_t;
            for(int j=0; j<part_number; j++)
            {
                v_tcu.row.push_back(partSize_t);
                for(int b=0; b<partSize_t; b++)
                {
                    int offset = j*partSize_t + b;
                    // int offset1 = offset + b;
                    // int sum = std::accumulate(count_tcu.begin()+offset, count_tcu.begin()+offset1, 0);
                    v_tcu.t_block_offset.push_back(count_tcu[offset]);
                }
                v_tcu.window.push_back(i);
                v_tcu.atomic.push_back(1);
                v_tcu.atomic_seq.push_back(1);
            }
            if(block_residue > 0)
            {             
                v_tcu.row.push_back(block_residue);
                for(int b=0; b<block_residue; b++)
                {
                    int offset = part_number*partSize_t+b;
                    // int offset1 = offset + b;
                    // int sum = std::accumulate(count_tcu.begin()+offset, count_tcu.begin()+offset1, 0);
                    v_tcu.t_block_offset.push_back(count_tcu[offset]);
                }
                v_tcu.window.push_back(i);
                v_tcu.atomic.push_back(1);
                v_tcu.atomic_seq.push_back(1);
            }
        }
        /*
            4. 汇总三部分的结果
        */
        #pragma omp critical  
        res_t[i]=v_tcu;

        #pragma omp critical  
        res_c[i]=large;

        #pragma omp critical  
        res_c_residue[i]=small;
        
    
    }
    /*
        5. 整合所有window的数据
    */
    //tcu
    std::vector<int> t_rowkNew_offset;
    t_rowkNew_offset.push_back(0);
    std::vector<int> t_blockNew_offset;
    t_blockNew_offset.push_back(0);
    std::vector<float> t_valueNew;
    std::vector<int> t_columnNew;
    std::vector<long> t_binaryNew;
    std::vector<int> t_window_rowNew;
    std::vector<int> t_atomicNew;
    std::vector<int> t_atomicNew_seq;

    //cuda 长行
    std::vector<int> c_rowNew_offset;
    c_rowNew_offset.push_back(0);
    std::vector<int> c_rowNew;
    std::vector<int> c_atomicNew;
    std::vector<int> c_atomicNew_seq;
    std::vector<int> c_columnNew;
    std::vector<float> c_valueNew;

    //cuda 短行
    std::vector<int> c_rowNew_offset_residue;
    c_rowNew_offset_residue.push_back(0);
    std::vector<int> c_rowNew_residue;
    std::vector<int> c_atomicNew_residue;
    std::vector<int> c_atomicNew_residue_seq;
    std::vector<int> c_columnNew_residue;
    std::vector<float> c_valueNew_residue;

    //按顺序整合res
    //tcu
    for (const auto& pair : res_t) {
        if (pair.second.tcu_flag == true )
        {
            for(int sub : pair.second.row)
            {
                t_rowkNew_offset.push_back(t_rowkNew_offset.back()+sub);
            }  
            for(int sub : pair.second.t_block_offset)
            {
                t_blockNew_offset.push_back(t_blockNew_offset.back()+sub);
            }  
            t_valueNew.insert(t_valueNew.end(),pair.second.t_value.begin(),pair.second.t_value.end());
            t_columnNew.insert(t_columnNew.end(),pair.second.t_column.begin(),pair.second.t_column.end());
            t_window_rowNew.insert(t_window_rowNew.end(),pair.second.window.begin(),pair.second.window.end());
            t_atomicNew.insert(t_atomicNew.end(),pair.second.atomic.begin(),pair.second.atomic.end());
            t_atomicNew_seq.insert(t_atomicNew_seq.end(),pair.second.atomic_seq.begin(),pair.second.atomic_seq.end());
            t_binaryNew.insert(t_binaryNew.end(),pair.second.t_binary.begin(),pair.second.t_binary.end());
        }   
    }

    //cuda long
    for (const auto& pair : res_c) {
        for( Value_cuda sub : pair.second){
            if (sub.cuda_flag == true )
            {c_rowNew_offset.push_back(c_rowNew_offset.back()+sub.c_row_offset);
            c_rowNew.push_back(sub.c_row);
            c_atomicNew.push_back(sub.c_atomic);
            c_atomicNew_seq.push_back(sub.c_atomic_seq);
            c_columnNew.insert(c_columnNew.end(),sub.c_column.begin(),sub.c_column.end());
            c_valueNew.insert(c_valueNew.end(),sub.c_value.begin(), sub.c_value.end());}
        }
    }

    //cuda short
    for (const auto& pair : res_c_residue) {
        for( Value_cuda sub : pair.second){
            if (sub.cuda_residue_flag == true )
            {c_rowNew_offset_residue.push_back(c_rowNew_offset_residue.back()+sub.c_row_offset);
            c_rowNew_residue.push_back(sub.c_row);
            c_atomicNew_residue.push_back(sub.c_atomic);
            c_atomicNew_residue_seq.push_back(sub.c_atomic_seq);
            c_columnNew_residue.insert(c_columnNew_residue.end(),sub.c_column.begin(),sub.c_column.end());
            c_valueNew_residue.insert(c_valueNew_residue.end(),sub.c_value.begin(), sub.c_value.end());}
        }
    }

    auto end = std::chrono::high_resolution_clock::now();

    // 计算耗时
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;

    auto t_rowNew_offsetTensor1 = torch::from_blob(t_rowkNew_offset.data(), t_rowkNew_offset.size(), torch::kInt32);
    auto t_blockNew_offsetTensor1 = torch::from_blob(t_blockNew_offset.data(), t_blockNew_offset.size(), torch::kInt32);
    auto t_valueTensor1 = torch::from_blob(t_valueNew.data(), t_valueNew.size(), torch::kFloat32);
    auto t_columnTensor1 = torch::from_blob(t_columnNew.data(), t_columnNew.size(), torch::kInt32);
    auto t_window_rowTensor1 = torch::from_blob(t_window_rowNew.data(), t_window_rowNew.size(), torch::kInt32);
    auto t_atomicTensor1 = torch::from_blob(t_atomicNew.data(), t_atomicNew.size(), torch::kInt32);
    auto t_atomicTensor1_seq = torch::from_blob(t_atomicNew_seq.data(), t_atomicNew_seq.size(), torch::kInt32);
    auto t_binaryTensor1 = torch::from_blob(t_binaryNew.data(), t_binaryNew.size(), torch::kInt64);

    torch::Tensor t_rowNew_offsetTensor = torch::empty_like(t_rowNew_offsetTensor1);
    t_rowNew_offsetTensor.copy_(t_rowNew_offsetTensor1);
    torch::Tensor t_blockNew_offsetTensor = torch::empty_like(t_blockNew_offsetTensor1);
    t_blockNew_offsetTensor.copy_(t_blockNew_offsetTensor1);
    torch::Tensor t_valueTensor = torch::empty_like(t_valueTensor1);
    t_valueTensor.copy_(t_valueTensor1);
    torch::Tensor t_columnTensor = torch::empty_like(t_columnTensor1);
    t_columnTensor.copy_(t_columnTensor1);
    torch::Tensor t_window_rowTensor = torch::empty_like(t_window_rowTensor1);
    t_window_rowTensor.copy_(t_window_rowTensor1);
    torch::Tensor t_atomicTensor = torch::empty_like(t_atomicTensor1);
    t_atomicTensor.copy_(t_atomicTensor1);
    torch::Tensor t_binaryTensor = torch::empty_like(t_binaryTensor1);
    t_binaryTensor.copy_(t_binaryTensor1);
    torch::Tensor t_atomicTensor_seq = torch::empty_like(t_atomicTensor1_seq);
    t_atomicTensor_seq.copy_(t_atomicTensor1_seq);

    auto c_row_offsetTensor1 = torch::from_blob(c_rowNew_offset.data(), c_rowNew_offset.size(), torch::kInt32);
    auto c_rowTensor1 = torch::from_blob(c_rowNew.data(), c_rowNew.size(), torch::kInt32);
    auto c_atomicTensor1 = torch::from_blob(c_atomicNew.data(), c_atomicNew.size(), torch::kInt32);
    auto c_atomicTensor1_seq = torch::from_blob(c_atomicNew_seq.data(), c_atomicNew_seq.size(), torch::kInt32);
    auto c_colTensor1 = torch::from_blob(c_columnNew.data(), c_columnNew.size(), torch::kInt32);
    auto c_valueTensor1 = torch::from_blob(c_valueNew.data(), c_valueNew.size(), torch::kFloat32);
    
    torch::Tensor c_row_offsetTensor = torch::empty_like(c_row_offsetTensor1);
    c_row_offsetTensor.copy_(c_row_offsetTensor1);
    torch::Tensor c_rowTensor = torch::empty_like(c_rowTensor1);
    c_rowTensor.copy_(c_rowTensor1);
    torch::Tensor c_atomicTensor = torch::empty_like(c_atomicTensor1);
    c_atomicTensor.copy_(c_atomicTensor1);
    torch::Tensor c_atomicTensor_seq = torch::empty_like(c_atomicTensor1_seq);
    c_atomicTensor_seq.copy_(c_atomicTensor1_seq);
    torch::Tensor c_colTensor = torch::empty_like(c_colTensor1);
    c_colTensor.copy_(c_colTensor1);
    torch::Tensor c_valueTensor = torch::empty_like(c_valueTensor1);
    c_valueTensor.copy_(c_valueTensor1);
    

    auto c_row_offsetTensor1_residue = torch::from_blob(c_rowNew_offset_residue.data(), c_rowNew_offset_residue.size(), torch::kInt32);
    auto c_rowTensor1_residue = torch::from_blob(c_rowNew_residue.data(), c_rowNew_residue.size(), torch::kInt32);
    auto c_atomicTensor1_residue = torch::from_blob(c_atomicNew_residue.data(), c_atomicNew_residue.size(), torch::kInt32);
    auto c_atomicTensor1_residue_seq = torch::from_blob(c_atomicNew_residue_seq.data(), c_atomicNew_residue_seq.size(), torch::kInt32); 
    auto c_colTensor1_residue = torch::from_blob(c_columnNew_residue.data(), c_columnNew_residue.size(), torch::kInt32);
    auto c_valueTensor1_residue = torch::from_blob(c_valueNew_residue.data(), c_valueNew_residue.size(), torch::kFloat32);
    
    torch::Tensor c_row_offsetTensor_residue = torch::empty_like(c_row_offsetTensor1_residue);
    c_row_offsetTensor_residue.copy_(c_row_offsetTensor1_residue);
    torch::Tensor c_rowTensor_residue = torch::empty_like(c_rowTensor1_residue);
    c_rowTensor_residue.copy_(c_rowTensor1_residue);
    torch::Tensor c_atomicTensor_residue = torch::empty_like(c_atomicTensor1_residue);
    c_atomicTensor_residue.copy_(c_atomicTensor1_residue);
    torch::Tensor c_atomicTensor_residue_seq = torch::empty_like(c_atomicTensor1_residue_seq);
    c_atomicTensor_residue_seq.copy_(c_atomicTensor1_residue_seq);
    torch::Tensor c_colTensor_residue = torch::empty_like(c_colTensor1_residue);
    c_colTensor_residue.copy_(c_colTensor1_residue);
    torch::Tensor c_valueTensor_residue = torch::empty_like(c_valueTensor1_residue);
    c_valueTensor_residue.copy_(c_valueTensor1_residue);

    return {t_rowNew_offsetTensor, t_blockNew_offsetTensor,t_columnTensor, t_valueTensor, t_window_rowTensor, t_atomicTensor, t_atomicTensor_seq, t_binaryTensor,
            c_row_offsetTensor, c_rowTensor, c_atomicTensor, c_atomicTensor_seq, c_colTensor, c_valueTensor,
            c_row_offsetTensor_residue, c_rowTensor_residue, c_atomicTensor_residue, c_atomicTensor_residue, c_colTensor_residue, c_valueTensor_residue, torch::tensor(elapsed.count())};

}



// m-tcu + cuda + partSize + binary
std::vector<torch::Tensor> block_tf32_tcu_cuda_binary_kernel_seq_hcspmm(torch::Tensor row1, torch::Tensor column1, torch::Tensor values1, 
int partSize_t, int partSize_c, int shortSize, int density, int window, int wide)
{
    auto *row=row1.data_ptr<int>();
    auto column=column1.accessor<int, 1>();
    auto values=values1.accessor<float, 1>();

    auto start = std::chrono::high_resolution_clock::now();

    int rows=row1.size(0)-1;
    // Result
    std::map<int, Value_tcu_binary> res_t;
    std::map<int, std::vector<Value_cuda>> res_c;
    std::map<int, std::vector<Value_cuda>> res_c_residue;
    //需要的window数量
    int num_windows = rows/window;
    #pragma omp parallel for
    for(int i=0;i<num_windows;i++)
    {
        /*
            1. 统计当前window中的vector, 并按照稠密度降序排列
        */
        //对column进行合并
        std::vector<int> mergedVector(&column[row[i*window]], &column[row[i*window+window]]);
        // 使用 unordered_map 统计 vector 中每个元素出现次数
        std::unordered_map<int, int> elementCounts;
        for (const auto& element : mergedVector) elementCounts[element]++;
        // 将统计结果放入 vector<pair> 中
        std::vector<std::pair<int, int>> countVector(elementCounts.begin(), elementCounts.end());
        // 按元素个数降序排序
        std::sort(countVector.begin(), countVector.end(), [](const auto& a, const auto& b) {
            return a.second > b.second;
        });

        /*
            2. 确定需要进行tcu计算的vector数量, t_percen
        */
        int size = countVector.size();
        int residue = size%wide;
        //统计每个block中的非零元数量，计算稠密度
        int t_percen = 0; //确定前多少个vector会被用作tcu进行稠密计算
        int num_window_edges = row[i*window+window] - row[i*window];
        int num = (size + (wide - size%wide)) / wide;
        // 0:cuda 1:tcu
        bool if_tcu = (float)size * 0.19854024 - ((float)num_window_edges / (num * 16 * 8)) * 6.578043 - 3.14922857 ? 0:1;
        if(if_tcu){
            t_percen = size;
        }else{
            t_percen = 0;
        }
        int t_percen_ori = t_percen;
        if((t_percen%wide) != 0)
        {
            // if((t_percen + (wide - t_percen%wide)) <= size) 
            t_percen = (t_percen + (wide - t_percen%wide));
            // else
            // t_percen = (t_percen - (t_percen%wide));

        }
        // if(t_percen<=size){
        // // 将前 k 个数按升序排列
        // std::sort(countVector.begin(), countVector.begin() + t_percen);
        // // 将剩下的数按升序排列
        // std::sort(countVector.begin() + t_percen, countVector.end());
        // }else{
        //     std::sort(countVector.begin(), countVector.end());
        // }
        std::sort(countVector.begin(), countVector.begin()+t_percen_ori);
        std::sort(countVector.begin() + t_percen_ori, countVector.end());
        /*
            3. 根据t_percen分别划分TCU和CUDA的负载
        */

        // TCU 划分
        // 3.1 根据partSize_t划分t_part
        //当前window中有几个block
        int blocks = t_percen/wide; 
        Value_tcu_binary v_tcu;
        if(blocks>0) {v_tcu.tcu_flag = true; }
        std::map<int, int> colmap;
        int c=0;
        for(auto col:countVector){
            if (c<t_percen_ori)
            {
                v_tcu.t_column.push_back(col.first);
            }
            colmap[col.first]=c++;
        }
        if((t_percen_ori%wide)>0){
            for(int tt=0;tt<(wide-(t_percen_ori%wide));tt++)
            v_tcu.t_column.push_back(-1);
        }
        //存放拆分的cuda长短行数据
        std::vector<Value_cuda> large;
        std::vector<Value_cuda> small;

        std::vector<std::vector<float>> t_value_temp = createVectorOfVectors_float(blocks);
        std::vector<long> t_binary_temp = createVectorOfVectors_v2(blocks);
        // std::vector<std::vector<int>> t_row_temp = createVectorOfVectors(blocks);
        // std::vector<std::vector<int>> t_col_temp = createVectorOfVectors(blocks);
        // 指示快内的行偏移
        int p=0;
        for(int j=i*window;j<(i+1)*window;j++)
        {
            //用来存储每行的cuda数据
            std::queue<int> c_col;
            std::queue<int> c_value;
            int cuda_nnz=0;
            for(int m=row[j];m<row[j+1];m++)
            {
                int newCol = colmap[column[m]];
                if(newCol<t_percen_ori)
                {
                    int block = newCol/wide;
                    int bInId= newCol%wide;
                    t_value_temp[block].push_back(values[m]);
                    long offset = p*wide + bInId;
                    long a=1;
                    t_binary_temp[block] |= a<<offset;
                }else{
                    //划分cuda部分
                    c_col.push(column[m]);
                    c_value.push(values[m]);
                    cuda_nnz++;
                }
            }
            //处理当前行剩余的cuda元素
            if(cuda_nnz>0)
            {
                if(cuda_nnz <= shortSize)
                {
                    Value_cuda temp;
                    temp.cuda_residue_flag = true;
                    temp.c_row_offset = cuda_nnz;
                    temp.c_row = i*window + p;
                        if(t_percen_ori>0) 
                        {
                            temp.c_atomic = 1;
                            temp.c_atomic_seq = 0;
                        }
                        else{
                             temp.c_atomic = 0;
                             temp.c_atomic_seq = 0;
                        }
                    for(int m=0; m<cuda_nnz; m++)
                    {
                        temp.c_column.push_back(c_col.front());
                        temp.c_value.push_back(c_value.front());
                        c_col.pop();
                        c_value.pop();
                    }
                    small.push_back(temp);
                }else{
                    if(cuda_nnz <= partSize_c)
                    {
                        Value_cuda temp;
                        temp.cuda_flag = true;
                        temp.c_row_offset = (cuda_nnz);
                        temp.c_row = i*window + p;
                            if(t_percen_ori>0) 
                            {
                                temp.c_atomic = 1;
                                temp.c_atomic_seq = 0;
                            }
                            else{
                                temp.c_atomic = 0;
                                temp.c_atomic_seq = 0;
                            }
                        for(int m=0; m<cuda_nnz; m++)
                        {
                            temp.c_column.push_back(c_col.front());
                            temp.c_value.push_back(c_value.front());
                            c_col.pop();
                            c_value.pop();
                        }
                        large.push_back(temp);
                    }else{
                        // 当前行的非零元被分成多个part，并需要原子操作
                        int parts_row = cuda_nnz/partSize_c;
                        // int cur_value_indice = row[i];
                        for(int cur_i = 0; cur_i<parts_row; cur_i++)
                        {
                            Value_cuda temp;
                            temp.cuda_flag = true;      
                            temp.c_row_offset = partSize_c;
                            temp.c_row = i*window + p;
                            temp.c_atomic = 1;
                            temp.c_atomic_seq = 1;
                            for(int m=0; m<partSize_c; m++)
                            {
                                temp.c_column.push_back(c_col.front());
                                temp.c_value.push_back(c_value.front());
                                c_col.pop();
                                c_value.pop();
                                // cur_value_indice++;
                            }
                            large.push_back(temp);
                        }
                        int residue  = (cuda_nnz%partSize_c);
                        if(residue>0){
                            Value_cuda temp;
                            temp.c_row_offset = residue;
                            temp.c_row = i*window + p;
                            temp.c_atomic = 1;
                            temp.c_atomic_seq = 1;
                            for(int m=0; m<residue; m++)
                            {
                                temp.c_column.push_back(c_col.front());
                                temp.c_value.push_back(c_value.front());
                                c_col.pop();
                                c_value.pop();
                                // cur_value_indice++;
                            }
                            if(residue <= shortSize)
                            {
                                temp.cuda_residue_flag = true;
                                small.push_back(temp);
                            }else{
                                temp.cuda_flag = true;
                                large.push_back(temp);
                            }
                        }
                    }
                }
            }
            p++;
        }
        //整合vector
        v_tcu.t_value = mergeVectors_float(t_value_temp);
        v_tcu.t_binary = t_binary_temp;
        // v_tcu.t_row = mergeVectors(t_row_temp);
        // v_tcu.t_col = mergeVectors(t_col_temp);
        
        std::vector<int> count_tcu = countVectors(t_value_temp);
        if(blocks <= partSize_t)
        {
            v_tcu.row.push_back(blocks);
            //循放入每个block中的非零元数量
            for(int b=0; b<blocks; b++)
            {
                // int sum = std::accumulate(count_tcu.begin()+b, count_tcu.begin()+b+1, 0);
                v_tcu.t_block_offset.push_back(count_tcu[b]);
            }
                if(t_percen_ori==size) 
                {
                    v_tcu.atomic.push_back(0);
                    v_tcu.atomic_seq.push_back(0);
                }
                else {
                    v_tcu.atomic.push_back(1);
                    v_tcu.atomic_seq.push_back(0);
                }
            // if(t_percen_ori==size) v_tcu.atomic.push_back(0);
            // else v_tcu.atomic.push_back(1);
            v_tcu.window.push_back(i);
        }else{
            // 对block太多的块进行分割
            int part_number = blocks / partSize_t;
            int block_residue = blocks % partSize_t;
            for(int j=0; j<part_number; j++)
            {
                v_tcu.row.push_back(partSize_t);
                for(int b=0; b<partSize_t; b++)
                {
                    int offset = j*partSize_t + b;
                    // int offset1 = offset + b;
                    // int sum = std::accumulate(count_tcu.begin()+offset, count_tcu.begin()+offset1, 0);
                    v_tcu.t_block_offset.push_back(count_tcu[offset]);
                }
                v_tcu.window.push_back(i);
                v_tcu.atomic.push_back(1);
                v_tcu.atomic_seq.push_back(1);
            }
            if(block_residue > 0)
            {             
                v_tcu.row.push_back(block_residue);
                for(int b=0; b<block_residue; b++)
                {
                    int offset = part_number*partSize_t+b;
                    // int offset1 = offset + b;
                    // int sum = std::accumulate(count_tcu.begin()+offset, count_tcu.begin()+offset1, 0);
                    v_tcu.t_block_offset.push_back(count_tcu[offset]);
                }
                v_tcu.window.push_back(i);
                v_tcu.atomic.push_back(1);
                v_tcu.atomic_seq.push_back(1);
            }
        }
        /*
            4. 汇总三部分的结果
        */
        #pragma omp critical  
        res_t[i]=v_tcu;

        #pragma omp critical  
        res_c[i]=large;

        #pragma omp critical  
        res_c_residue[i]=small;
        
    
    }
    /*
        5. 整合所有window的数据
    */
    //tcu
    std::vector<int> t_rowkNew_offset;
    t_rowkNew_offset.push_back(0);
    std::vector<int> t_blockNew_offset;
    t_blockNew_offset.push_back(0);
    std::vector<float> t_valueNew;
    std::vector<int> t_columnNew;
    std::vector<long> t_binaryNew;
    std::vector<int> t_window_rowNew;
    std::vector<int> t_atomicNew;
    std::vector<int> t_atomicNew_seq;

    //cuda 长行
    std::vector<int> c_rowNew_offset;
    c_rowNew_offset.push_back(0);
    std::vector<int> c_rowNew;
    std::vector<int> c_atomicNew;
    std::vector<int> c_atomicNew_seq;
    std::vector<int> c_columnNew;
    std::vector<float> c_valueNew;

    //cuda 短行
    std::vector<int> c_rowNew_offset_residue;
    c_rowNew_offset_residue.push_back(0);
    std::vector<int> c_rowNew_residue;
    std::vector<int> c_atomicNew_residue;
    std::vector<int> c_atomicNew_residue_seq;
    std::vector<int> c_columnNew_residue;
    std::vector<float> c_valueNew_residue;

    //按顺序整合res
    //tcu
    for (const auto& pair : res_t) {
        if (pair.second.tcu_flag == true )
        {
            for(int sub : pair.second.row)
            {
                t_rowkNew_offset.push_back(t_rowkNew_offset.back()+sub);
            }  
            for(int sub : pair.second.t_block_offset)
            {
                t_blockNew_offset.push_back(t_blockNew_offset.back()+sub);
            }  
            t_valueNew.insert(t_valueNew.end(),pair.second.t_value.begin(),pair.second.t_value.end());
            t_columnNew.insert(t_columnNew.end(),pair.second.t_column.begin(),pair.second.t_column.end());
            t_window_rowNew.insert(t_window_rowNew.end(),pair.second.window.begin(),pair.second.window.end());
            t_atomicNew.insert(t_atomicNew.end(),pair.second.atomic.begin(),pair.second.atomic.end());
            t_atomicNew_seq.insert(t_atomicNew_seq.end(),pair.second.atomic_seq.begin(),pair.second.atomic_seq.end());
            t_binaryNew.insert(t_binaryNew.end(),pair.second.t_binary.begin(),pair.second.t_binary.end());
        }   
    }

    //cuda long
    for (const auto& pair : res_c) {
        for( Value_cuda sub : pair.second){
            if (sub.cuda_flag == true )
            {c_rowNew_offset.push_back(c_rowNew_offset.back()+sub.c_row_offset);
            c_rowNew.push_back(sub.c_row);
            c_atomicNew.push_back(sub.c_atomic);
            c_atomicNew_seq.push_back(sub.c_atomic_seq);
            c_columnNew.insert(c_columnNew.end(),sub.c_column.begin(),sub.c_column.end());
            c_valueNew.insert(c_valueNew.end(),sub.c_value.begin(), sub.c_value.end());}
        }
    }

    //cuda short
    for (const auto& pair : res_c_residue) {
        for( Value_cuda sub : pair.second){
            if (sub.cuda_residue_flag == true )
            {c_rowNew_offset_residue.push_back(c_rowNew_offset_residue.back()+sub.c_row_offset);
            c_rowNew_residue.push_back(sub.c_row);
            c_atomicNew_residue.push_back(sub.c_atomic);
            c_atomicNew_residue_seq.push_back(sub.c_atomic_seq);
            c_columnNew_residue.insert(c_columnNew_residue.end(),sub.c_column.begin(),sub.c_column.end());
            c_valueNew_residue.insert(c_valueNew_residue.end(),sub.c_value.begin(), sub.c_value.end());}
        }
    }

    auto end = std::chrono::high_resolution_clock::now();

    // 计算耗时
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;

    auto t_rowNew_offsetTensor1 = torch::from_blob(t_rowkNew_offset.data(), t_rowkNew_offset.size(), torch::kInt32);
    auto t_blockNew_offsetTensor1 = torch::from_blob(t_blockNew_offset.data(), t_blockNew_offset.size(), torch::kInt32);
    auto t_valueTensor1 = torch::from_blob(t_valueNew.data(), t_valueNew.size(), torch::kFloat32);
    auto t_columnTensor1 = torch::from_blob(t_columnNew.data(), t_columnNew.size(), torch::kInt32);
    auto t_window_rowTensor1 = torch::from_blob(t_window_rowNew.data(), t_window_rowNew.size(), torch::kInt32);
    auto t_atomicTensor1 = torch::from_blob(t_atomicNew.data(), t_atomicNew.size(), torch::kInt32);
    auto t_atomicTensor1_seq = torch::from_blob(t_atomicNew_seq.data(), t_atomicNew_seq.size(), torch::kInt32);
    auto t_binaryTensor1 = torch::from_blob(t_binaryNew.data(), t_binaryNew.size(), torch::kInt64);

    torch::Tensor t_rowNew_offsetTensor = torch::empty_like(t_rowNew_offsetTensor1);
    t_rowNew_offsetTensor.copy_(t_rowNew_offsetTensor1);
    torch::Tensor t_blockNew_offsetTensor = torch::empty_like(t_blockNew_offsetTensor1);
    t_blockNew_offsetTensor.copy_(t_blockNew_offsetTensor1);
    torch::Tensor t_valueTensor = torch::empty_like(t_valueTensor1);
    t_valueTensor.copy_(t_valueTensor1);
    torch::Tensor t_columnTensor = torch::empty_like(t_columnTensor1);
    t_columnTensor.copy_(t_columnTensor1);
    torch::Tensor t_window_rowTensor = torch::empty_like(t_window_rowTensor1);
    t_window_rowTensor.copy_(t_window_rowTensor1);
    torch::Tensor t_atomicTensor = torch::empty_like(t_atomicTensor1);
    t_atomicTensor.copy_(t_atomicTensor1);
    torch::Tensor t_binaryTensor = torch::empty_like(t_binaryTensor1);
    t_binaryTensor.copy_(t_binaryTensor1);
    torch::Tensor t_atomicTensor_seq = torch::empty_like(t_atomicTensor1_seq);
    t_atomicTensor_seq.copy_(t_atomicTensor1_seq);

    auto c_row_offsetTensor1 = torch::from_blob(c_rowNew_offset.data(), c_rowNew_offset.size(), torch::kInt32);
    auto c_rowTensor1 = torch::from_blob(c_rowNew.data(), c_rowNew.size(), torch::kInt32);
    auto c_atomicTensor1 = torch::from_blob(c_atomicNew.data(), c_atomicNew.size(), torch::kInt32);
    auto c_atomicTensor1_seq = torch::from_blob(c_atomicNew_seq.data(), c_atomicNew_seq.size(), torch::kInt32);
    auto c_colTensor1 = torch::from_blob(c_columnNew.data(), c_columnNew.size(), torch::kInt32);
    auto c_valueTensor1 = torch::from_blob(c_valueNew.data(), c_valueNew.size(), torch::kFloat32);
    
    torch::Tensor c_row_offsetTensor = torch::empty_like(c_row_offsetTensor1);
    c_row_offsetTensor.copy_(c_row_offsetTensor1);
    torch::Tensor c_rowTensor = torch::empty_like(c_rowTensor1);
    c_rowTensor.copy_(c_rowTensor1);
    torch::Tensor c_atomicTensor = torch::empty_like(c_atomicTensor1);
    c_atomicTensor.copy_(c_atomicTensor1);
    torch::Tensor c_atomicTensor_seq = torch::empty_like(c_atomicTensor1_seq);
    c_atomicTensor_seq.copy_(c_atomicTensor1_seq);
    torch::Tensor c_colTensor = torch::empty_like(c_colTensor1);
    c_colTensor.copy_(c_colTensor1);
    torch::Tensor c_valueTensor = torch::empty_like(c_valueTensor1);
    c_valueTensor.copy_(c_valueTensor1);
    

    auto c_row_offsetTensor1_residue = torch::from_blob(c_rowNew_offset_residue.data(), c_rowNew_offset_residue.size(), torch::kInt32);
    auto c_rowTensor1_residue = torch::from_blob(c_rowNew_residue.data(), c_rowNew_residue.size(), torch::kInt32);
    auto c_atomicTensor1_residue = torch::from_blob(c_atomicNew_residue.data(), c_atomicNew_residue.size(), torch::kInt32);
    auto c_atomicTensor1_residue_seq = torch::from_blob(c_atomicNew_residue_seq.data(), c_atomicNew_residue_seq.size(), torch::kInt32); 
    auto c_colTensor1_residue = torch::from_blob(c_columnNew_residue.data(), c_columnNew_residue.size(), torch::kInt32);
    auto c_valueTensor1_residue = torch::from_blob(c_valueNew_residue.data(), c_valueNew_residue.size(), torch::kFloat32);
    
    torch::Tensor c_row_offsetTensor_residue = torch::empty_like(c_row_offsetTensor1_residue);
    c_row_offsetTensor_residue.copy_(c_row_offsetTensor1_residue);
    torch::Tensor c_rowTensor_residue = torch::empty_like(c_rowTensor1_residue);
    c_rowTensor_residue.copy_(c_rowTensor1_residue);
    torch::Tensor c_atomicTensor_residue = torch::empty_like(c_atomicTensor1_residue);
    c_atomicTensor_residue.copy_(c_atomicTensor1_residue);
    torch::Tensor c_atomicTensor_residue_seq = torch::empty_like(c_atomicTensor1_residue_seq);
    c_atomicTensor_residue_seq.copy_(c_atomicTensor1_residue_seq);
    torch::Tensor c_colTensor_residue = torch::empty_like(c_colTensor1_residue);
    c_colTensor_residue.copy_(c_colTensor1_residue);
    torch::Tensor c_valueTensor_residue = torch::empty_like(c_valueTensor1_residue);
    c_valueTensor_residue.copy_(c_valueTensor1_residue);

    return {t_rowNew_offsetTensor, t_blockNew_offsetTensor,t_columnTensor, t_valueTensor, t_window_rowTensor, t_atomicTensor, t_atomicTensor_seq, t_binaryTensor,
            c_row_offsetTensor, c_rowTensor, c_atomicTensor, c_atomicTensor_seq, c_colTensor, c_valueTensor,
            c_row_offsetTensor_residue, c_rowTensor_residue, c_atomicTensor_residue, c_atomicTensor_residue, c_colTensor_residue, c_valueTensor_residue, torch::tensor(elapsed.count())};

}



// m-tcu + cuda + partSize + binary
std::vector<torch::Tensor> block_tf32_tcu_cuda_binary_kernel_seq_hcspmm1(torch::Tensor row1, torch::Tensor column1, torch::Tensor values1, 
int partSize_t, int partSize_c, int shortSize, int density, int window, int wide)
{
    auto *row=row1.data_ptr<int>();
    auto column=column1.accessor<int, 1>();
    auto values=values1.accessor<float, 1>();

    auto start = std::chrono::high_resolution_clock::now();

    int rows=row1.size(0)-1;
    // Result
    std::map<int, Value_tcu_binary> res_t;
    std::map<int, std::vector<Value_cuda>> res_c;
    std::map<int, std::vector<Value_cuda>> res_c_residue;
    //需要的window数量
    int num_windows = rows/window;
    #pragma omp parallel for
    for(int i=0;i<num_windows;i++)
    {
        /*
            1. 统计当前window中的vector, 并按照稠密度降序排列
        */
        //对column进行合并
        std::vector<int> mergedVector(&column[row[i*window]], &column[row[i*window+window]]);
        // 使用 unordered_map 统计 vector 中每个元素出现次数
        std::unordered_map<int, int> elementCounts;
        for (const auto& element : mergedVector) elementCounts[element]++;
        // 将统计结果放入 vector<pair> 中
        std::vector<std::pair<int, int>> countVector(elementCounts.begin(), elementCounts.end());
        // 按元素个数降序排序
        std::sort(countVector.begin(), countVector.end(), [](const auto& a, const auto& b) {
            return a.second > b.second;
        });

        /*
            2. 确定需要进行tcu计算的vector数量, t_percen
        */
        int size = countVector.size();
        int residue = size%wide;
        //统计每个block中的非零元数量，计算稠密度
        int t_percen = 0; //确定前多少个vector会被用作tcu进行稠密计算
        int num_window_edges = row[i*window+window] - row[i*window];
        int num = (size + (wide - size%wide)) / wide;
        // 0:cuda 1:tcu
        bool if_tcu = (float)size * 0.19854024 - ((float)num_window_edges / (num * 16 * 8)) * 6.578043 - 3.14922857 ? 0:1;
        if(if_tcu){
            t_percen = size;
        }else{
            t_percen = 0;
        }
        int t_percen_ori = t_percen;
        if((t_percen%wide) != 0)
        {
            // if((t_percen + (wide - t_percen%wide)) <= size) 
            t_percen = (t_percen + (wide - t_percen%wide));
            // else
            // t_percen = (t_percen - (t_percen%wide));

        }
        // if(t_percen<=size){
        // // 将前 k 个数按升序排列
        // std::sort(countVector.begin(), countVector.begin() + t_percen);
        // // 将剩下的数按升序排列
        // std::sort(countVector.begin() + t_percen, countVector.end());
        // }else{
        //     std::sort(countVector.begin(), countVector.end());
        // }
        std::sort(countVector.begin(), countVector.begin()+t_percen_ori);
        std::sort(countVector.begin() + t_percen_ori, countVector.end());
        /*
            3. 根据t_percen分别划分TCU和CUDA的负载
        */

        // TCU 划分
        // 3.1 根据partSize_t划分t_part
        //当前window中有几个block
        int blocks = t_percen/wide; 
        Value_tcu_binary v_tcu;
        if(blocks>0) {v_tcu.tcu_flag = true; }
        std::map<int, int> colmap;
        int c=0;
        for(auto col:countVector){
            if (c<t_percen_ori)
            {
                v_tcu.t_column.push_back(col.first);
            }
            colmap[col.first]=c++;
        }
        if((t_percen_ori%wide)>0){
            for(int tt=0;tt<(wide-(t_percen_ori%wide));tt++)
            v_tcu.t_column.push_back(-1);
        }
        //存放拆分的cuda长短行数据
        std::vector<Value_cuda> large;
        std::vector<Value_cuda> small;

        std::vector<std::vector<float>> t_value_temp = createVectorOfVectors_float(blocks);
        std::vector<long> t_binary_temp = createVectorOfVectors_v2(blocks);
        // std::vector<std::vector<int>> t_row_temp = createVectorOfVectors(blocks);
        // std::vector<std::vector<int>> t_col_temp = createVectorOfVectors(blocks);
        // 指示快内的行偏移
        int p=0;
        for(int j=i*window;j<(i+1)*window;j++)
        {
            //用来存储每行的cuda数据
            std::queue<int> c_col;
            std::queue<int> c_value;
            int cuda_nnz=0;
            for(int m=row[j];m<row[j+1];m++)
            {
                int newCol = colmap[column[m]];
                if(newCol<t_percen_ori)
                {
                    int block = newCol/wide;
                    int bInId= newCol%wide;
                    t_value_temp[block].push_back(values[m]);
                    long offset = p*wide + bInId;
                    long a=1;
                    t_binary_temp[block] |= a<<offset;
                }else{
                    //划分cuda部分
                    c_col.push(column[m]);
                    c_value.push(values[m]);
                    cuda_nnz++;
                }
            }
            //处理当前行剩余的cuda元素
            if(cuda_nnz>0)
            {
                if(cuda_nnz <= shortSize)
                {
                    Value_cuda temp;
                    temp.cuda_residue_flag = true;
                    temp.c_row_offset = cuda_nnz;
                    temp.c_row = i*window + p;
                        if(t_percen_ori>0) 
                        {
                            temp.c_atomic = 1;
                            temp.c_atomic_seq = 0;
                        }
                        else{
                             temp.c_atomic = 0;
                             temp.c_atomic_seq = 0;
                        }
                    for(int m=0; m<cuda_nnz; m++)
                    {
                        temp.c_column.push_back(c_col.front());
                        temp.c_value.push_back(c_value.front());
                        c_col.pop();
                        c_value.pop();
                    }
                    small.push_back(temp);
                }else{
                    if(cuda_nnz <= partSize_c)
                    {
                        Value_cuda temp;
                        temp.cuda_flag = true;
                        temp.c_row_offset = (cuda_nnz);
                        temp.c_row = i*window + p;
                            if(t_percen_ori>0) 
                            {
                                temp.c_atomic = 1;
                                temp.c_atomic_seq = 0;
                            }
                            else{
                                temp.c_atomic = 0;
                                temp.c_atomic_seq = 0;
                            }
                        for(int m=0; m<cuda_nnz; m++)
                        {
                            temp.c_column.push_back(c_col.front());
                            temp.c_value.push_back(c_value.front());
                            c_col.pop();
                            c_value.pop();
                        }
                        large.push_back(temp);
                    }else{
                        // 当前行的非零元被分成多个part，并需要原子操作
                        int parts_row = cuda_nnz/partSize_c;
                        // int cur_value_indice = row[i];
                        for(int cur_i = 0; cur_i<parts_row; cur_i++)
                        {
                            Value_cuda temp;
                            temp.cuda_flag = true;      
                            temp.c_row_offset = partSize_c;
                            temp.c_row = i*window + p;
                            temp.c_atomic = 1;
                            temp.c_atomic_seq = 1;
                            for(int m=0; m<partSize_c; m++)
                            {
                                temp.c_column.push_back(c_col.front());
                                temp.c_value.push_back(c_value.front());
                                c_col.pop();
                                c_value.pop();
                                // cur_value_indice++;
                            }
                            large.push_back(temp);
                        }
                        int residue  = (cuda_nnz%partSize_c);
                        if(residue>0){
                            Value_cuda temp;
                            temp.c_row_offset = residue;
                            temp.c_row = i*window + p;
                            temp.c_atomic = 1;
                            temp.c_atomic_seq = 1;
                            for(int m=0; m<residue; m++)
                            {
                                temp.c_column.push_back(c_col.front());
                                temp.c_value.push_back(c_value.front());
                                c_col.pop();
                                c_value.pop();
                                // cur_value_indice++;
                            }
                            if(residue <= shortSize)
                            {
                                temp.cuda_residue_flag = true;
                                small.push_back(temp);
                            }else{
                                temp.cuda_flag = true;
                                large.push_back(temp);
                            }
                        }
                    }
                }
            }
            p++;
        }
        //整合vector
        v_tcu.t_value = mergeVectors_float(t_value_temp);
        v_tcu.t_binary = t_binary_temp;
        // v_tcu.t_row = mergeVectors(t_row_temp);
        // v_tcu.t_col = mergeVectors(t_col_temp);
        
        std::vector<int> count_tcu = countVectors(t_value_temp);
        if(blocks <= partSize_t)
        {
            v_tcu.row.push_back(blocks);
            //循放入每个block中的非零元数量
            for(int b=0; b<blocks; b++)
            {
                // int sum = std::accumulate(count_tcu.begin()+b, count_tcu.begin()+b+1, 0);
                v_tcu.t_block_offset.push_back(count_tcu[b]);
            }
                if(t_percen_ori==size) 
                {
                    v_tcu.atomic.push_back(0);
                    v_tcu.atomic_seq.push_back(0);
                }
                else {
                    v_tcu.atomic.push_back(1);
                    v_tcu.atomic_seq.push_back(0);
                }
            // if(t_percen_ori==size) v_tcu.atomic.push_back(0);
            // else v_tcu.atomic.push_back(1);
            v_tcu.window.push_back(i);
        }else{
            // 对block太多的块进行分割
            int part_number = blocks / partSize_t;
            int block_residue = blocks % partSize_t;
            for(int j=0; j<part_number; j++)
            {
                v_tcu.row.push_back(partSize_t);
                for(int b=0; b<partSize_t; b++)
                {
                    int offset = j*partSize_t + b;
                    // int offset1 = offset + b;
                    // int sum = std::accumulate(count_tcu.begin()+offset, count_tcu.begin()+offset1, 0);
                    v_tcu.t_block_offset.push_back(count_tcu[offset]);
                }
                v_tcu.window.push_back(i);
                v_tcu.atomic.push_back(1);
                v_tcu.atomic_seq.push_back(1);
            }
            if(block_residue > 0)
            {             
                v_tcu.row.push_back(block_residue);
                for(int b=0; b<block_residue; b++)
                {
                    int offset = part_number*partSize_t+b;
                    // int offset1 = offset + b;
                    // int sum = std::accumulate(count_tcu.begin()+offset, count_tcu.begin()+offset1, 0);
                    v_tcu.t_block_offset.push_back(count_tcu[offset]);
                }
                v_tcu.window.push_back(i);
                v_tcu.atomic.push_back(1);
                v_tcu.atomic_seq.push_back(1);
            }
        }
        /*
            4. 汇总三部分的结果
        */
        #pragma omp critical  
        res_t[i]=v_tcu;

        #pragma omp critical  
        res_c[i]=large;

        #pragma omp critical  
        res_c_residue[i]=small;
        
    
    }
    /*
        5. 整合所有window的数据
    */
    //tcu
    std::vector<int> t_rowkNew_offset;
    t_rowkNew_offset.push_back(0);
    std::vector<int> t_blockNew_offset;
    t_blockNew_offset.push_back(0);
    std::vector<float> t_valueNew;
    std::vector<int> t_columnNew;
    std::vector<long> t_binaryNew;
    std::vector<int> t_window_rowNew;
    std::vector<int> t_atomicNew;
    std::vector<int> t_atomicNew_seq;

    //cuda 长行
    std::vector<int> c_rowNew_offset;
    c_rowNew_offset.push_back(0);
    std::vector<int> c_rowNew;
    std::vector<int> c_atomicNew;
    std::vector<int> c_atomicNew_seq;
    std::vector<int> c_columnNew;
    std::vector<float> c_valueNew;

    //cuda 短行
    std::vector<int> c_rowNew_offset_residue;
    c_rowNew_offset_residue.push_back(0);
    std::vector<int> c_rowNew_residue;
    std::vector<int> c_atomicNew_residue;
    std::vector<int> c_atomicNew_residue_seq;
    std::vector<int> c_columnNew_residue;
    std::vector<float> c_valueNew_residue;

    //按顺序整合res
    //tcu
    for (const auto& pair : res_t) {
        if (pair.second.tcu_flag == true )
        {
            for(int sub : pair.second.row)
            {
                t_rowkNew_offset.push_back(t_rowkNew_offset.back()+sub);
            }  
            for(int sub : pair.second.t_block_offset)
            {
                t_blockNew_offset.push_back(t_blockNew_offset.back()+sub);
            }  
            t_valueNew.insert(t_valueNew.end(),pair.second.t_value.begin(),pair.second.t_value.end());
            t_columnNew.insert(t_columnNew.end(),pair.second.t_column.begin(),pair.second.t_column.end());
            t_window_rowNew.insert(t_window_rowNew.end(),pair.second.window.begin(),pair.second.window.end());
            t_atomicNew.insert(t_atomicNew.end(),pair.second.atomic.begin(),pair.second.atomic.end());
            t_atomicNew_seq.insert(t_atomicNew_seq.end(),pair.second.atomic_seq.begin(),pair.second.atomic_seq.end());
            t_binaryNew.insert(t_binaryNew.end(),pair.second.t_binary.begin(),pair.second.t_binary.end());
        }   
    }

    //cuda long
    for (const auto& pair : res_c) {
        for( Value_cuda sub : pair.second){
            if (sub.cuda_flag == true )
            {c_rowNew_offset.push_back(c_rowNew_offset.back()+sub.c_row_offset);
            c_rowNew.push_back(sub.c_row);
            c_atomicNew.push_back(sub.c_atomic);
            c_atomicNew_seq.push_back(sub.c_atomic_seq);
            c_columnNew.insert(c_columnNew.end(),sub.c_column.begin(),sub.c_column.end());
            c_valueNew.insert(c_valueNew.end(),sub.c_value.begin(), sub.c_value.end());}
        }
    }

    //cuda short
    for (const auto& pair : res_c_residue) {
        for( Value_cuda sub : pair.second){
            if (sub.cuda_residue_flag == true )
            {c_rowNew_offset_residue.push_back(c_rowNew_offset_residue.back()+sub.c_row_offset);
            c_rowNew_residue.push_back(sub.c_row);
            c_atomicNew_residue.push_back(sub.c_atomic);
            c_atomicNew_residue_seq.push_back(sub.c_atomic_seq);
            c_columnNew_residue.insert(c_columnNew_residue.end(),sub.c_column.begin(),sub.c_column.end());
            c_valueNew_residue.insert(c_valueNew_residue.end(),sub.c_value.begin(), sub.c_value.end());}
        }
    }

    auto end = std::chrono::high_resolution_clock::now();

    // 计算耗时
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;

    auto t_rowNew_offsetTensor1 = torch::from_blob(t_rowkNew_offset.data(), t_rowkNew_offset.size(), torch::kInt32);
    auto t_blockNew_offsetTensor1 = torch::from_blob(t_blockNew_offset.data(), t_blockNew_offset.size(), torch::kInt32);
    auto t_valueTensor1 = torch::from_blob(t_valueNew.data(), t_valueNew.size(), torch::kFloat32);
    auto t_columnTensor1 = torch::from_blob(t_columnNew.data(), t_columnNew.size(), torch::kInt32);
    auto t_window_rowTensor1 = torch::from_blob(t_window_rowNew.data(), t_window_rowNew.size(), torch::kInt32);
    auto t_atomicTensor1 = torch::from_blob(t_atomicNew.data(), t_atomicNew.size(), torch::kInt32);
    auto t_atomicTensor1_seq = torch::from_blob(t_atomicNew_seq.data(), t_atomicNew_seq.size(), torch::kInt32);
    auto t_binaryTensor1 = torch::from_blob(t_binaryNew.data(), t_binaryNew.size(), torch::kInt64);

    torch::Tensor t_rowNew_offsetTensor = torch::empty_like(t_rowNew_offsetTensor1);
    t_rowNew_offsetTensor.copy_(t_rowNew_offsetTensor1);
    torch::Tensor t_blockNew_offsetTensor = torch::empty_like(t_blockNew_offsetTensor1);
    t_blockNew_offsetTensor.copy_(t_blockNew_offsetTensor1);
    torch::Tensor t_valueTensor = torch::empty_like(t_valueTensor1);
    t_valueTensor.copy_(t_valueTensor1);
    torch::Tensor t_columnTensor = torch::empty_like(t_columnTensor1);
    t_columnTensor.copy_(t_columnTensor1);
    torch::Tensor t_window_rowTensor = torch::empty_like(t_window_rowTensor1);
    t_window_rowTensor.copy_(t_window_rowTensor1);
    torch::Tensor t_atomicTensor = torch::empty_like(t_atomicTensor1);
    t_atomicTensor.copy_(t_atomicTensor1);
    torch::Tensor t_binaryTensor = torch::empty_like(t_binaryTensor1);
    t_binaryTensor.copy_(t_binaryTensor1);
    torch::Tensor t_atomicTensor_seq = torch::empty_like(t_atomicTensor1_seq);
    t_atomicTensor_seq.copy_(t_atomicTensor1_seq);

    auto c_row_offsetTensor1 = torch::from_blob(c_rowNew_offset.data(), c_rowNew_offset.size(), torch::kInt32);
    auto c_rowTensor1 = torch::from_blob(c_rowNew.data(), c_rowNew.size(), torch::kInt32);
    auto c_atomicTensor1 = torch::from_blob(c_atomicNew.data(), c_atomicNew.size(), torch::kInt32);
    auto c_atomicTensor1_seq = torch::from_blob(c_atomicNew_seq.data(), c_atomicNew_seq.size(), torch::kInt32);
    auto c_colTensor1 = torch::from_blob(c_columnNew.data(), c_columnNew.size(), torch::kInt32);
    auto c_valueTensor1 = torch::from_blob(c_valueNew.data(), c_valueNew.size(), torch::kFloat32);
    
    torch::Tensor c_row_offsetTensor = torch::empty_like(c_row_offsetTensor1);
    c_row_offsetTensor.copy_(c_row_offsetTensor1);
    torch::Tensor c_rowTensor = torch::empty_like(c_rowTensor1);
    c_rowTensor.copy_(c_rowTensor1);
    torch::Tensor c_atomicTensor = torch::empty_like(c_atomicTensor1);
    c_atomicTensor.copy_(c_atomicTensor1);
    torch::Tensor c_atomicTensor_seq = torch::empty_like(c_atomicTensor1_seq);
    c_atomicTensor_seq.copy_(c_atomicTensor1_seq);
    torch::Tensor c_colTensor = torch::empty_like(c_colTensor1);
    c_colTensor.copy_(c_colTensor1);
    torch::Tensor c_valueTensor = torch::empty_like(c_valueTensor1);
    c_valueTensor.copy_(c_valueTensor1);
    

    auto c_row_offsetTensor1_residue = torch::from_blob(c_rowNew_offset_residue.data(), c_rowNew_offset_residue.size(), torch::kInt32);
    auto c_rowTensor1_residue = torch::from_blob(c_rowNew_residue.data(), c_rowNew_residue.size(), torch::kInt32);
    auto c_atomicTensor1_residue = torch::from_blob(c_atomicNew_residue.data(), c_atomicNew_residue.size(), torch::kInt32);
    auto c_atomicTensor1_residue_seq = torch::from_blob(c_atomicNew_residue_seq.data(), c_atomicNew_residue_seq.size(), torch::kInt32); 
    auto c_colTensor1_residue = torch::from_blob(c_columnNew_residue.data(), c_columnNew_residue.size(), torch::kInt32);
    auto c_valueTensor1_residue = torch::from_blob(c_valueNew_residue.data(), c_valueNew_residue.size(), torch::kFloat32);
    
    torch::Tensor c_row_offsetTensor_residue = torch::empty_like(c_row_offsetTensor1_residue);
    c_row_offsetTensor_residue.copy_(c_row_offsetTensor1_residue);
    torch::Tensor c_rowTensor_residue = torch::empty_like(c_rowTensor1_residue);
    c_rowTensor_residue.copy_(c_rowTensor1_residue);
    torch::Tensor c_atomicTensor_residue = torch::empty_like(c_atomicTensor1_residue);
    c_atomicTensor_residue.copy_(c_atomicTensor1_residue);
    torch::Tensor c_atomicTensor_residue_seq = torch::empty_like(c_atomicTensor1_residue_seq);
    c_atomicTensor_residue_seq.copy_(c_atomicTensor1_residue_seq);
    torch::Tensor c_colTensor_residue = torch::empty_like(c_colTensor1_residue);
    c_colTensor_residue.copy_(c_colTensor1_residue);
    torch::Tensor c_valueTensor_residue = torch::empty_like(c_valueTensor1_residue);
    c_valueTensor_residue.copy_(c_valueTensor1_residue);

    return {t_rowNew_offsetTensor, t_blockNew_offsetTensor,t_columnTensor, t_valueTensor, t_window_rowTensor, t_atomicTensor, t_atomicTensor_seq, t_binaryTensor,
            c_row_offsetTensor, c_rowTensor, c_atomicTensor, c_atomicTensor_seq, c_colTensor, c_valueTensor,
            c_row_offsetTensor_residue, c_rowTensor_residue, c_atomicTensor_residue, c_atomicTensor_residue, c_colTensor_residue, c_valueTensor_residue, torch::tensor(elapsed.count())};

}



// m-tcu + cuda + partSize + binary
std::vector<torch::Tensor> block_tf32_tcu_cuda_binary_kernel_seq_pcgcn(torch::Tensor row1, torch::Tensor column1, torch::Tensor values1, 
int partSize_t, int partSize_c, int shortSize, int density, int window, int wide)
{
    auto *row=row1.data_ptr<int>();
    auto column=column1.accessor<int, 1>();
    auto values=values1.accessor<float, 1>();

    auto start = std::chrono::high_resolution_clock::now();

    int rows=row1.size(0)-1;
    // Result
    std::map<int, Value_tcu_binary> res_t;
    std::map<int, std::vector<Value_cuda>> res_c;
    std::map<int, std::vector<Value_cuda>> res_c_residue;
    //需要的window数量
    int num_windows = rows/window;
    #pragma omp parallel for
    for(int i=0;i<num_windows;i++)
    {
        /*
            1. 统计当前window中的vector, 并按照稠密度降序排列
        */
        //对column进行合并
        std::vector<int> mergedVector(&column[row[i*window]], &column[row[i*window+window]]);
        // 使用 unordered_map 统计 vector 中每个元素出现次数
        std::unordered_map<int, int> elementCounts;
        for (const auto& element : mergedVector) elementCounts[element]++;
        // 将统计结果放入 vector<pair> 中
        std::vector<std::pair<int, int>> countVector(elementCounts.begin(), elementCounts.end());
        // 按元素个数降序排序
        std::sort(countVector.begin(), countVector.end(), [](const auto& a, const auto& b) {
            return a.second > b.second;
        });

        /*
            2. 确定需要进行tcu计算的vector数量, t_percen
        */
        int size = countVector.size();
        int residue = size%wide;
        //统计每个block中的非零元数量，计算稠密度
        int t_percen = 0; //确定前多少个vector会被用作tcu进行稠密计算
        t_percen = std::count_if(countVector.begin(), countVector.end(), [density](const auto& pair) {
            return pair.second >= density;
        });
            int t_percen_ori = t_percen;
            int t_percen_ori_pc = t_percen*wide;
        
        

        std::sort(countVector.begin(), countVector.begin()+t_percen_ori);
        std::sort(countVector.begin() + t_percen_ori, countVector.end());
        /*
            3. 根据t_percen分别划分TCU和CUDA的负载
        */

        // TCU 划分
        // 3.1 根据partSize_t划分t_part
        //当前window中有几个block
        int blocks = t_percen; 
        Value_tcu_binary v_tcu;
        if(blocks>0) {v_tcu.tcu_flag = true; }
        std::map<int, int> colmap;
        int c=0;
            for(auto col:countVector){
                if (c<t_percen_ori)
                {
                    v_tcu.t_column.push_back(col.first);
                    for(int tt=0; tt<wide-1;tt++)
                        v_tcu.t_column.push_back(-1);
                }
                colmap[col.first]=c*wide;
            }
        //存放拆分的cuda长短行数据
        std::vector<Value_cuda> large;
        std::vector<Value_cuda> small;

        std::vector<std::vector<float>> t_value_temp = createVectorOfVectors_float(blocks);
        std::vector<long> t_binary_temp = createVectorOfVectors_v2(blocks);
        // std::vector<std::vector<int>> t_row_temp = createVectorOfVectors(blocks);
        // std::vector<std::vector<int>> t_col_temp = createVectorOfVectors(blocks);
        // 指示快内的行偏移
        int p=0;
        for(int j=i*window;j<(i+1)*window;j++)
        {
            //用来存储每行的cuda数据
            std::queue<int> c_col;
            std::queue<int> c_value;
            int cuda_nnz=0;
            for(int m=row[j];m<row[j+1];m++)
            {
                int newCol = colmap[column[m]];
                if(newCol<t_percen_ori_pc)
                {
                    int block = newCol/wide;
                    int bInId= newCol%wide;
                    t_value_temp[block].push_back(values[m]);
                    long offset = p*wide + bInId;
                    long a=1;
                    t_binary_temp[block] |= a<<offset;
                }else{
                    //划分cuda部分
                    c_col.push(column[m]);
                    c_value.push(values[m]);
                    cuda_nnz++;
                }
            }
            //处理当前行剩余的cuda元素
            if(cuda_nnz>0)
            {
                if(cuda_nnz <= shortSize)
                {
                    Value_cuda temp;
                    temp.cuda_residue_flag = true;
                    temp.c_row_offset = cuda_nnz;
                    temp.c_row = i*window + p;
                        if(t_percen_ori>0) 
                        {
                            temp.c_atomic = 1;
                            temp.c_atomic_seq = 0;
                        }
                        else{
                             temp.c_atomic = 0;
                             temp.c_atomic_seq = 0;
                        }
                    for(int m=0; m<cuda_nnz; m++)
                    {
                        temp.c_column.push_back(c_col.front());
                        temp.c_value.push_back(c_value.front());
                        c_col.pop();
                        c_value.pop();
                    }
                    small.push_back(temp);
                }else{
                    if(cuda_nnz <= partSize_c)
                    {
                        Value_cuda temp;
                        temp.cuda_flag = true;
                        temp.c_row_offset = (cuda_nnz);
                        temp.c_row = i*window + p;
                            if(t_percen_ori>0) 
                            {
                                temp.c_atomic = 1;
                                temp.c_atomic_seq = 0;
                            }
                            else{
                                temp.c_atomic = 0;
                                temp.c_atomic_seq = 0;
                            }
                        for(int m=0; m<cuda_nnz; m++)
                        {
                            temp.c_column.push_back(c_col.front());
                            temp.c_value.push_back(c_value.front());
                            c_col.pop();
                            c_value.pop();
                        }
                        large.push_back(temp);
                    }else{
                        // 当前行的非零元被分成多个part，并需要原子操作
                        int parts_row = cuda_nnz/partSize_c;
                        // int cur_value_indice = row[i];
                        for(int cur_i = 0; cur_i<parts_row; cur_i++)
                        {
                            Value_cuda temp;
                            temp.cuda_flag = true;      
                            temp.c_row_offset = partSize_c;
                            temp.c_row = i*window + p;
                            temp.c_atomic = 1;
                            temp.c_atomic_seq = 1;
                            for(int m=0; m<partSize_c; m++)
                            {
                                temp.c_column.push_back(c_col.front());
                                temp.c_value.push_back(c_value.front());
                                c_col.pop();
                                c_value.pop();
                                // cur_value_indice++;
                            }
                            large.push_back(temp);
                        }
                        int residue  = (cuda_nnz%partSize_c);
                        if(residue>0){
                            Value_cuda temp;
                            temp.c_row_offset = residue;
                            temp.c_row = i*window + p;
                            temp.c_atomic = 1;
                            temp.c_atomic_seq = 1;
                            for(int m=0; m<residue; m++)
                            {
                                temp.c_column.push_back(c_col.front());
                                temp.c_value.push_back(c_value.front());
                                c_col.pop();
                                c_value.pop();
                                // cur_value_indice++;
                            }
                            if(residue <= shortSize)
                            {
                                temp.cuda_residue_flag = true;
                                small.push_back(temp);
                            }else{
                                temp.cuda_flag = true;
                                large.push_back(temp);
                            }
                        }
                    }
                }
            }
            p++;
        }
        //整合vector
        v_tcu.t_value = mergeVectors_float(t_value_temp);
        v_tcu.t_binary = t_binary_temp;
        // v_tcu.t_row = mergeVectors(t_row_temp);
        // v_tcu.t_col = mergeVectors(t_col_temp);
        
        std::vector<int> count_tcu = countVectors(t_value_temp);
        if(blocks <= partSize_t)
        {
            v_tcu.row.push_back(blocks);
            //循放入每个block中的非零元数量
            for(int b=0; b<blocks; b++)
            {
                // int sum = std::accumulate(count_tcu.begin()+b, count_tcu.begin()+b+1, 0);
                v_tcu.t_block_offset.push_back(count_tcu[b]);
            }
                if(t_percen_ori==size) 
                {
                    v_tcu.atomic.push_back(0);
                    v_tcu.atomic_seq.push_back(0);
                }
                else {
                    v_tcu.atomic.push_back(1);
                    v_tcu.atomic_seq.push_back(0);
                }
            // if(t_percen_ori==size) v_tcu.atomic.push_back(0);
            // else v_tcu.atomic.push_back(1);
            v_tcu.window.push_back(i);
        }else{
            // 对block太多的块进行分割
            int part_number = blocks / partSize_t;
            int block_residue = blocks % partSize_t;
            for(int j=0; j<part_number; j++)
            {
                v_tcu.row.push_back(partSize_t);
                for(int b=0; b<partSize_t; b++)
                {
                    int offset = j*partSize_t + b;
                    // int offset1 = offset + b;
                    // int sum = std::accumulate(count_tcu.begin()+offset, count_tcu.begin()+offset1, 0);
                    v_tcu.t_block_offset.push_back(count_tcu[offset]);
                }
                v_tcu.window.push_back(i);
                v_tcu.atomic.push_back(1);
                v_tcu.atomic_seq.push_back(1);
            }
            if(block_residue > 0)
            {             
                v_tcu.row.push_back(block_residue);
                for(int b=0; b<block_residue; b++)
                {
                    int offset = part_number*partSize_t+b;
                    // int offset1 = offset + b;
                    // int sum = std::accumulate(count_tcu.begin()+offset, count_tcu.begin()+offset1, 0);
                    v_tcu.t_block_offset.push_back(count_tcu[offset]);
                }
                v_tcu.window.push_back(i);
                v_tcu.atomic.push_back(1);
                v_tcu.atomic_seq.push_back(1);
            }
        }
        /*
            4. 汇总三部分的结果
        */
        #pragma omp critical  
        res_t[i]=v_tcu;

        #pragma omp critical  
        res_c[i]=large;

        #pragma omp critical  
        res_c_residue[i]=small;
        
    
    }
    /*
        5. 整合所有window的数据
    */
    //tcu
    std::vector<int> t_rowkNew_offset;
    t_rowkNew_offset.push_back(0);
    std::vector<int> t_blockNew_offset;
    t_blockNew_offset.push_back(0);
    std::vector<float> t_valueNew;
    std::vector<int> t_columnNew;
    std::vector<long> t_binaryNew;
    std::vector<int> t_window_rowNew;
    std::vector<int> t_atomicNew;
    std::vector<int> t_atomicNew_seq;

    //cuda 长行
    std::vector<int> c_rowNew_offset;
    c_rowNew_offset.push_back(0);
    std::vector<int> c_rowNew;
    std::vector<int> c_atomicNew;
    std::vector<int> c_atomicNew_seq;
    std::vector<int> c_columnNew;
    std::vector<float> c_valueNew;

    //cuda 短行
    std::vector<int> c_rowNew_offset_residue;
    c_rowNew_offset_residue.push_back(0);
    std::vector<int> c_rowNew_residue;
    std::vector<int> c_atomicNew_residue;
    std::vector<int> c_atomicNew_residue_seq;
    std::vector<int> c_columnNew_residue;
    std::vector<float> c_valueNew_residue;

    //按顺序整合res
    //tcu
    for (const auto& pair : res_t) {
        if (pair.second.tcu_flag == true )
        {
            for(int sub : pair.second.row)
            {
                t_rowkNew_offset.push_back(t_rowkNew_offset.back()+sub);
            }  
            for(int sub : pair.second.t_block_offset)
            {
                t_blockNew_offset.push_back(t_blockNew_offset.back()+sub);
            }  
            t_valueNew.insert(t_valueNew.end(),pair.second.t_value.begin(),pair.second.t_value.end());
            t_columnNew.insert(t_columnNew.end(),pair.second.t_column.begin(),pair.second.t_column.end());
            t_window_rowNew.insert(t_window_rowNew.end(),pair.second.window.begin(),pair.second.window.end());
            t_atomicNew.insert(t_atomicNew.end(),pair.second.atomic.begin(),pair.second.atomic.end());
            t_atomicNew_seq.insert(t_atomicNew_seq.end(),pair.second.atomic_seq.begin(),pair.second.atomic_seq.end());
            t_binaryNew.insert(t_binaryNew.end(),pair.second.t_binary.begin(),pair.second.t_binary.end());
        }   
    }

    //cuda long
    for (const auto& pair : res_c) {
        for( Value_cuda sub : pair.second){
            if (sub.cuda_flag == true )
            {c_rowNew_offset.push_back(c_rowNew_offset.back()+sub.c_row_offset);
            c_rowNew.push_back(sub.c_row);
            c_atomicNew.push_back(sub.c_atomic);
            c_atomicNew_seq.push_back(sub.c_atomic_seq);
            c_columnNew.insert(c_columnNew.end(),sub.c_column.begin(),sub.c_column.end());
            c_valueNew.insert(c_valueNew.end(),sub.c_value.begin(), sub.c_value.end());}
        }
    }

    //cuda short
    for (const auto& pair : res_c_residue) {
        for( Value_cuda sub : pair.second){
            if (sub.cuda_residue_flag == true )
            {c_rowNew_offset_residue.push_back(c_rowNew_offset_residue.back()+sub.c_row_offset);
            c_rowNew_residue.push_back(sub.c_row);
            c_atomicNew_residue.push_back(sub.c_atomic);
            c_atomicNew_residue_seq.push_back(sub.c_atomic_seq);
            c_columnNew_residue.insert(c_columnNew_residue.end(),sub.c_column.begin(),sub.c_column.end());
            c_valueNew_residue.insert(c_valueNew_residue.end(),sub.c_value.begin(), sub.c_value.end());}
        }
    }

    auto end = std::chrono::high_resolution_clock::now();

    // 计算耗时
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;

    auto t_rowNew_offsetTensor1 = torch::from_blob(t_rowkNew_offset.data(), t_rowkNew_offset.size(), torch::kInt32);
    auto t_blockNew_offsetTensor1 = torch::from_blob(t_blockNew_offset.data(), t_blockNew_offset.size(), torch::kInt32);
    auto t_valueTensor1 = torch::from_blob(t_valueNew.data(), t_valueNew.size(), torch::kFloat32);
    auto t_columnTensor1 = torch::from_blob(t_columnNew.data(), t_columnNew.size(), torch::kInt32);
    auto t_window_rowTensor1 = torch::from_blob(t_window_rowNew.data(), t_window_rowNew.size(), torch::kInt32);
    auto t_atomicTensor1 = torch::from_blob(t_atomicNew.data(), t_atomicNew.size(), torch::kInt32);
    auto t_atomicTensor1_seq = torch::from_blob(t_atomicNew_seq.data(), t_atomicNew_seq.size(), torch::kInt32);
    auto t_binaryTensor1 = torch::from_blob(t_binaryNew.data(), t_binaryNew.size(), torch::kInt64);

    torch::Tensor t_rowNew_offsetTensor = torch::empty_like(t_rowNew_offsetTensor1);
    t_rowNew_offsetTensor.copy_(t_rowNew_offsetTensor1);
    torch::Tensor t_blockNew_offsetTensor = torch::empty_like(t_blockNew_offsetTensor1);
    t_blockNew_offsetTensor.copy_(t_blockNew_offsetTensor1);
    torch::Tensor t_valueTensor = torch::empty_like(t_valueTensor1);
    t_valueTensor.copy_(t_valueTensor1);
    torch::Tensor t_columnTensor = torch::empty_like(t_columnTensor1);
    t_columnTensor.copy_(t_columnTensor1);
    torch::Tensor t_window_rowTensor = torch::empty_like(t_window_rowTensor1);
    t_window_rowTensor.copy_(t_window_rowTensor1);
    torch::Tensor t_atomicTensor = torch::empty_like(t_atomicTensor1);
    t_atomicTensor.copy_(t_atomicTensor1);
    torch::Tensor t_binaryTensor = torch::empty_like(t_binaryTensor1);
    t_binaryTensor.copy_(t_binaryTensor1);
    torch::Tensor t_atomicTensor_seq = torch::empty_like(t_atomicTensor1_seq);
    t_atomicTensor_seq.copy_(t_atomicTensor1_seq);

    auto c_row_offsetTensor1 = torch::from_blob(c_rowNew_offset.data(), c_rowNew_offset.size(), torch::kInt32);
    auto c_rowTensor1 = torch::from_blob(c_rowNew.data(), c_rowNew.size(), torch::kInt32);
    auto c_atomicTensor1 = torch::from_blob(c_atomicNew.data(), c_atomicNew.size(), torch::kInt32);
    auto c_atomicTensor1_seq = torch::from_blob(c_atomicNew_seq.data(), c_atomicNew_seq.size(), torch::kInt32);
    auto c_colTensor1 = torch::from_blob(c_columnNew.data(), c_columnNew.size(), torch::kInt32);
    auto c_valueTensor1 = torch::from_blob(c_valueNew.data(), c_valueNew.size(), torch::kFloat32);
    
    torch::Tensor c_row_offsetTensor = torch::empty_like(c_row_offsetTensor1);
    c_row_offsetTensor.copy_(c_row_offsetTensor1);
    torch::Tensor c_rowTensor = torch::empty_like(c_rowTensor1);
    c_rowTensor.copy_(c_rowTensor1);
    torch::Tensor c_atomicTensor = torch::empty_like(c_atomicTensor1);
    c_atomicTensor.copy_(c_atomicTensor1);
    torch::Tensor c_atomicTensor_seq = torch::empty_like(c_atomicTensor1_seq);
    c_atomicTensor_seq.copy_(c_atomicTensor1_seq);
    torch::Tensor c_colTensor = torch::empty_like(c_colTensor1);
    c_colTensor.copy_(c_colTensor1);
    torch::Tensor c_valueTensor = torch::empty_like(c_valueTensor1);
    c_valueTensor.copy_(c_valueTensor1);
    

    auto c_row_offsetTensor1_residue = torch::from_blob(c_rowNew_offset_residue.data(), c_rowNew_offset_residue.size(), torch::kInt32);
    auto c_rowTensor1_residue = torch::from_blob(c_rowNew_residue.data(), c_rowNew_residue.size(), torch::kInt32);
    auto c_atomicTensor1_residue = torch::from_blob(c_atomicNew_residue.data(), c_atomicNew_residue.size(), torch::kInt32);
    auto c_atomicTensor1_residue_seq = torch::from_blob(c_atomicNew_residue_seq.data(), c_atomicNew_residue_seq.size(), torch::kInt32); 
    auto c_colTensor1_residue = torch::from_blob(c_columnNew_residue.data(), c_columnNew_residue.size(), torch::kInt32);
    auto c_valueTensor1_residue = torch::from_blob(c_valueNew_residue.data(), c_valueNew_residue.size(), torch::kFloat32);
    
    torch::Tensor c_row_offsetTensor_residue = torch::empty_like(c_row_offsetTensor1_residue);
    c_row_offsetTensor_residue.copy_(c_row_offsetTensor1_residue);
    torch::Tensor c_rowTensor_residue = torch::empty_like(c_rowTensor1_residue);
    c_rowTensor_residue.copy_(c_rowTensor1_residue);
    torch::Tensor c_atomicTensor_residue = torch::empty_like(c_atomicTensor1_residue);
    c_atomicTensor_residue.copy_(c_atomicTensor1_residue);
    torch::Tensor c_atomicTensor_residue_seq = torch::empty_like(c_atomicTensor1_residue_seq);
    c_atomicTensor_residue_seq.copy_(c_atomicTensor1_residue_seq);
    torch::Tensor c_colTensor_residue = torch::empty_like(c_colTensor1_residue);
    c_colTensor_residue.copy_(c_colTensor1_residue);
    torch::Tensor c_valueTensor_residue = torch::empty_like(c_valueTensor1_residue);
    c_valueTensor_residue.copy_(c_valueTensor1_residue);

    return {t_rowNew_offsetTensor, t_blockNew_offsetTensor,t_columnTensor, t_valueTensor, t_window_rowTensor, t_atomicTensor, t_atomicTensor_seq, t_binaryTensor,
            c_row_offsetTensor, c_rowTensor, c_atomicTensor, c_atomicTensor_seq, c_colTensor, c_valueTensor,
            c_row_offsetTensor_residue, c_rowTensor_residue, c_atomicTensor_residue, c_atomicTensor_residue, c_colTensor_residue, c_valueTensor_residue, torch::tensor(elapsed.count())};

}



// m-tcu + cuda + partSize + binary
std::vector<torch::Tensor> block_tf32_tcu_cuda_binary_kernel_seq_rtgnn(torch::Tensor row1, torch::Tensor column1, torch::Tensor values1, 
int partSize_t, int partSize_c, int shortSize, int density, int window, int wide)
{
    auto *row=row1.data_ptr<int>();
    auto column=column1.accessor<int, 1>();
    auto values=values1.accessor<float, 1>();

    auto start = std::chrono::high_resolution_clock::now();

    int rows=row1.size(0)-1;
    // Result
    std::map<int, Value_tcu_binary> res_t;
    std::map<int, std::vector<Value_cuda>> res_c;
    std::map<int, std::vector<Value_cuda>> res_c_residue;
    //需要的window数量
    int num_windows = rows/window;
    #pragma omp parallel for
    for(int i=0;i<num_windows;i++)
    {
        /*
            1. 统计当前window中的vector, 并按照稠密度降序排列
        */
        //对column进行合并
        std::vector<int> mergedVector(&column[row[i*window]], &column[row[i*window+window]]);
        // 使用 unordered_map 统计 vector 中每个元素出现次数
        std::unordered_map<int, int> elementCounts;
        for (const auto& element : mergedVector) elementCounts[element]++;
        // 将统计结果放入 vector<pair> 中
        std::vector<std::pair<int, int>> countVector(elementCounts.begin(), elementCounts.end());
        // 按元素个数降序排序
        // std::sort(countVector.begin(), countVector.end(), [](const auto& a, const auto& b) {
        //     return a.second > b.second;
        // });

        /*
            2. 确定需要进行tcu计算的vector数量, t_percen
        */
        int size = countVector.size();
        int residue = size%wide;
        //统计每个block中的非零元数量，计算稠密度
        int t_percen = size/3*2; //确定前多少个vector会被用作tcu进行稠密计算
        // t_percen = std::count_if(countVector.begin(), countVector.end(), [density](const auto& pair) {
        //     return pair.second >= density;
        // });
        int t_percen_ori = t_percen;
        if((t_percen%wide) != 0)
        {
            // if((t_percen + (wide - t_percen%wide)) <= size) 
            t_percen = (t_percen + (wide - t_percen%wide));
            // else
            // t_percen = (t_percen - (t_percen%wide));

        }
        // if(t_percen<=size){
        // // 将前 k 个数按升序排列
        // std::sort(countVector.begin(), countVector.begin() + t_percen);
        // // 将剩下的数按升序排列
        // std::sort(countVector.begin() + t_percen, countVector.end());
        // }else{
        //     std::sort(countVector.begin(), countVector.end());
        // }
        std::sort(countVector.begin(), countVector.begin()+t_percen_ori);
        std::sort(countVector.begin() + t_percen_ori, countVector.end());
        /*
            3. 根据t_percen分别划分TCU和CUDA的负载
        */

        // TCU 划分
        // 3.1 根据partSize_t划分t_part
        //当前window中有几个block
        int blocks = t_percen/wide; 
        Value_tcu_binary v_tcu;
        if(blocks>0) {v_tcu.tcu_flag = true; }
        std::map<int, int> colmap;
        int c=0;
        for(auto col:countVector){
            if (c<t_percen_ori)
            {
                v_tcu.t_column.push_back(col.first);
            }
            colmap[col.first]=c++;
        }
        if((t_percen_ori%wide)>0){
            for(int tt=0;tt<(wide-(t_percen_ori%wide));tt++)
            v_tcu.t_column.push_back(-1);
        }
        //存放拆分的cuda长短行数据
        std::vector<Value_cuda> large;
        std::vector<Value_cuda> small;

        std::vector<std::vector<float>> t_value_temp = createVectorOfVectors_float(blocks);
        std::vector<long> t_binary_temp = createVectorOfVectors_v2(blocks);
        // std::vector<std::vector<int>> t_row_temp = createVectorOfVectors(blocks);
        // std::vector<std::vector<int>> t_col_temp = createVectorOfVectors(blocks);
        // 指示快内的行偏移
        int p=0;
        for(int j=i*window;j<(i+1)*window;j++)
        {
            //用来存储每行的cuda数据
            std::queue<int> c_col;
            std::queue<int> c_value;
            int cuda_nnz=0;
            for(int m=row[j];m<row[j+1];m++)
            {
                int newCol = colmap[column[m]];
                if(newCol<t_percen_ori)
                {
                    int block = newCol/wide;
                    int bInId= newCol%wide;
                    t_value_temp[block].push_back(values[m]);
                    long offset = p*wide + bInId;
                    long a=1;
                    t_binary_temp[block] |= a<<offset;
                }else{
                    //划分cuda部分
                    c_col.push(column[m]);
                    c_value.push(values[m]);
                    cuda_nnz++;
                }
            }
            //处理当前行剩余的cuda元素
            if(cuda_nnz>0)
            {
                if(cuda_nnz <= shortSize)
                {
                    Value_cuda temp;
                    temp.cuda_residue_flag = true;
                    temp.c_row_offset = cuda_nnz;
                    temp.c_row = i*window + p;
                        if(t_percen_ori>0) 
                        {
                            temp.c_atomic = 1;
                            temp.c_atomic_seq = 0;
                        }
                        else{
                             temp.c_atomic = 0;
                             temp.c_atomic_seq = 0;
                        }
                    for(int m=0; m<cuda_nnz; m++)
                    {
                        temp.c_column.push_back(c_col.front());
                        temp.c_value.push_back(c_value.front());
                        c_col.pop();
                        c_value.pop();
                    }
                    small.push_back(temp);
                }else{
                    if(cuda_nnz <= partSize_c)
                    {
                        Value_cuda temp;
                        temp.cuda_flag = true;
                        temp.c_row_offset = (cuda_nnz);
                        temp.c_row = i*window + p;
                            if(t_percen_ori>0) 
                            {
                                temp.c_atomic = 1;
                                temp.c_atomic_seq = 0;
                            }
                            else{
                                temp.c_atomic = 0;
                                temp.c_atomic_seq = 0;
                            }
                        for(int m=0; m<cuda_nnz; m++)
                        {
                            temp.c_column.push_back(c_col.front());
                            temp.c_value.push_back(c_value.front());
                            c_col.pop();
                            c_value.pop();
                        }
                        large.push_back(temp);
                    }else{
                        // 当前行的非零元被分成多个part，并需要原子操作
                        int parts_row = cuda_nnz/partSize_c;
                        // int cur_value_indice = row[i];
                        for(int cur_i = 0; cur_i<parts_row; cur_i++)
                        {
                            Value_cuda temp;
                            temp.cuda_flag = true;      
                            temp.c_row_offset = partSize_c;
                            temp.c_row = i*window + p;
                            temp.c_atomic = 1;
                            temp.c_atomic_seq = 1;
                            for(int m=0; m<partSize_c; m++)
                            {
                                temp.c_column.push_back(c_col.front());
                                temp.c_value.push_back(c_value.front());
                                c_col.pop();
                                c_value.pop();
                                // cur_value_indice++;
                            }
                            large.push_back(temp);
                        }
                        int residue  = (cuda_nnz%partSize_c);
                        if(residue>0){
                            Value_cuda temp;
                            temp.c_row_offset = residue;
                            temp.c_row = i*window + p;
                            temp.c_atomic = 1;
                            temp.c_atomic_seq = 1;
                            for(int m=0; m<residue; m++)
                            {
                                temp.c_column.push_back(c_col.front());
                                temp.c_value.push_back(c_value.front());
                                c_col.pop();
                                c_value.pop();
                                // cur_value_indice++;
                            }
                            if(residue <= shortSize)
                            {
                                temp.cuda_residue_flag = true;
                                small.push_back(temp);
                            }else{
                                temp.cuda_flag = true;
                                large.push_back(temp);
                            }
                        }
                    }
                }
            }
            p++;
        }
        //整合vector
        v_tcu.t_value = mergeVectors_float(t_value_temp);
        v_tcu.t_binary = t_binary_temp;
        // v_tcu.t_row = mergeVectors(t_row_temp);
        // v_tcu.t_col = mergeVectors(t_col_temp);
        
        std::vector<int> count_tcu = countVectors(t_value_temp);
        if(blocks <= partSize_t)
        {
            v_tcu.row.push_back(blocks);
            //循放入每个block中的非零元数量
            for(int b=0; b<blocks; b++)
            {
                // int sum = std::accumulate(count_tcu.begin()+b, count_tcu.begin()+b+1, 0);
                v_tcu.t_block_offset.push_back(count_tcu[b]);
            }
                if(t_percen_ori==size) 
                {
                    v_tcu.atomic.push_back(0);
                    v_tcu.atomic_seq.push_back(0);
                }
                else {
                    v_tcu.atomic.push_back(1);
                    v_tcu.atomic_seq.push_back(0);
                }
            // if(t_percen_ori==size) v_tcu.atomic.push_back(0);
            // else v_tcu.atomic.push_back(1);
            v_tcu.window.push_back(i);
        }else{
            // 对block太多的块进行分割
            int part_number = blocks / partSize_t;
            int block_residue = blocks % partSize_t;
            for(int j=0; j<part_number; j++)
            {
                v_tcu.row.push_back(partSize_t);
                for(int b=0; b<partSize_t; b++)
                {
                    int offset = j*partSize_t + b;
                    // int offset1 = offset + b;
                    // int sum = std::accumulate(count_tcu.begin()+offset, count_tcu.begin()+offset1, 0);
                    v_tcu.t_block_offset.push_back(count_tcu[offset]);
                }
                v_tcu.window.push_back(i);
                v_tcu.atomic.push_back(1);
                v_tcu.atomic_seq.push_back(1);
            }
            if(block_residue > 0)
            {             
                v_tcu.row.push_back(block_residue);
                for(int b=0; b<block_residue; b++)
                {
                    int offset = part_number*partSize_t+b;
                    // int offset1 = offset + b;
                    // int sum = std::accumulate(count_tcu.begin()+offset, count_tcu.begin()+offset1, 0);
                    v_tcu.t_block_offset.push_back(count_tcu[offset]);
                }
                v_tcu.window.push_back(i);
                v_tcu.atomic.push_back(1);
                v_tcu.atomic_seq.push_back(1);
            }
        }
        /*
            4. 汇总三部分的结果
        */
        #pragma omp critical  
        res_t[i]=v_tcu;

        #pragma omp critical  
        res_c[i]=large;

        #pragma omp critical  
        res_c_residue[i]=small;
        
    
    }
    /*
        5. 整合所有window的数据
    */
    //tcu
    std::vector<int> t_rowkNew_offset;
    t_rowkNew_offset.push_back(0);
    std::vector<int> t_blockNew_offset;
    t_blockNew_offset.push_back(0);
    std::vector<float> t_valueNew;
    std::vector<int> t_columnNew;
    std::vector<long> t_binaryNew;
    std::vector<int> t_window_rowNew;
    std::vector<int> t_atomicNew;
    std::vector<int> t_atomicNew_seq;

    //cuda 长行
    std::vector<int> c_rowNew_offset;
    c_rowNew_offset.push_back(0);
    std::vector<int> c_rowNew;
    std::vector<int> c_atomicNew;
    std::vector<int> c_atomicNew_seq;
    std::vector<int> c_columnNew;
    std::vector<float> c_valueNew;

    //cuda 短行
    std::vector<int> c_rowNew_offset_residue;
    c_rowNew_offset_residue.push_back(0);
    std::vector<int> c_rowNew_residue;
    std::vector<int> c_atomicNew_residue;
    std::vector<int> c_atomicNew_residue_seq;
    std::vector<int> c_columnNew_residue;
    std::vector<float> c_valueNew_residue;

    //按顺序整合res
    //tcu
    for (const auto& pair : res_t) {
        if (pair.second.tcu_flag == true )
        {
            for(int sub : pair.second.row)
            {
                t_rowkNew_offset.push_back(t_rowkNew_offset.back()+sub);
            }  
            for(int sub : pair.second.t_block_offset)
            {
                t_blockNew_offset.push_back(t_blockNew_offset.back()+sub);
            }  
            t_valueNew.insert(t_valueNew.end(),pair.second.t_value.begin(),pair.second.t_value.end());
            t_columnNew.insert(t_columnNew.end(),pair.second.t_column.begin(),pair.second.t_column.end());
            t_window_rowNew.insert(t_window_rowNew.end(),pair.second.window.begin(),pair.second.window.end());
            t_atomicNew.insert(t_atomicNew.end(),pair.second.atomic.begin(),pair.second.atomic.end());
            t_atomicNew_seq.insert(t_atomicNew_seq.end(),pair.second.atomic_seq.begin(),pair.second.atomic_seq.end());
            t_binaryNew.insert(t_binaryNew.end(),pair.second.t_binary.begin(),pair.second.t_binary.end());
        }   
    }

    //cuda long
    for (const auto& pair : res_c) {
        for( Value_cuda sub : pair.second){
            if (sub.cuda_flag == true )
            {c_rowNew_offset.push_back(c_rowNew_offset.back()+sub.c_row_offset);
            c_rowNew.push_back(sub.c_row);
            c_atomicNew.push_back(sub.c_atomic);
            c_atomicNew_seq.push_back(sub.c_atomic_seq);
            c_columnNew.insert(c_columnNew.end(),sub.c_column.begin(),sub.c_column.end());
            c_valueNew.insert(c_valueNew.end(),sub.c_value.begin(), sub.c_value.end());}
        }
    }

    //cuda short
    for (const auto& pair : res_c_residue) {
        for( Value_cuda sub : pair.second){
            if (sub.cuda_residue_flag == true )
            {c_rowNew_offset_residue.push_back(c_rowNew_offset_residue.back()+sub.c_row_offset);
            c_rowNew_residue.push_back(sub.c_row);
            c_atomicNew_residue.push_back(sub.c_atomic);
            c_atomicNew_residue_seq.push_back(sub.c_atomic_seq);
            c_columnNew_residue.insert(c_columnNew_residue.end(),sub.c_column.begin(),sub.c_column.end());
            c_valueNew_residue.insert(c_valueNew_residue.end(),sub.c_value.begin(), sub.c_value.end());}
        }
    }

    auto end = std::chrono::high_resolution_clock::now();

    // 计算耗时
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;

    auto t_rowNew_offsetTensor1 = torch::from_blob(t_rowkNew_offset.data(), t_rowkNew_offset.size(), torch::kInt32);
    auto t_blockNew_offsetTensor1 = torch::from_blob(t_blockNew_offset.data(), t_blockNew_offset.size(), torch::kInt32);
    auto t_valueTensor1 = torch::from_blob(t_valueNew.data(), t_valueNew.size(), torch::kFloat32);
    auto t_columnTensor1 = torch::from_blob(t_columnNew.data(), t_columnNew.size(), torch::kInt32);
    auto t_window_rowTensor1 = torch::from_blob(t_window_rowNew.data(), t_window_rowNew.size(), torch::kInt32);
    auto t_atomicTensor1 = torch::from_blob(t_atomicNew.data(), t_atomicNew.size(), torch::kInt32);
    auto t_atomicTensor1_seq = torch::from_blob(t_atomicNew_seq.data(), t_atomicNew_seq.size(), torch::kInt32);
    auto t_binaryTensor1 = torch::from_blob(t_binaryNew.data(), t_binaryNew.size(), torch::kInt64);

    torch::Tensor t_rowNew_offsetTensor = torch::empty_like(t_rowNew_offsetTensor1);
    t_rowNew_offsetTensor.copy_(t_rowNew_offsetTensor1);
    torch::Tensor t_blockNew_offsetTensor = torch::empty_like(t_blockNew_offsetTensor1);
    t_blockNew_offsetTensor.copy_(t_blockNew_offsetTensor1);
    torch::Tensor t_valueTensor = torch::empty_like(t_valueTensor1);
    t_valueTensor.copy_(t_valueTensor1);
    torch::Tensor t_columnTensor = torch::empty_like(t_columnTensor1);
    t_columnTensor.copy_(t_columnTensor1);
    torch::Tensor t_window_rowTensor = torch::empty_like(t_window_rowTensor1);
    t_window_rowTensor.copy_(t_window_rowTensor1);
    torch::Tensor t_atomicTensor = torch::empty_like(t_atomicTensor1);
    t_atomicTensor.copy_(t_atomicTensor1);
    torch::Tensor t_binaryTensor = torch::empty_like(t_binaryTensor1);
    t_binaryTensor.copy_(t_binaryTensor1);
    torch::Tensor t_atomicTensor_seq = torch::empty_like(t_atomicTensor1_seq);
    t_atomicTensor_seq.copy_(t_atomicTensor1_seq);

    auto c_row_offsetTensor1 = torch::from_blob(c_rowNew_offset.data(), c_rowNew_offset.size(), torch::kInt32);
    auto c_rowTensor1 = torch::from_blob(c_rowNew.data(), c_rowNew.size(), torch::kInt32);
    auto c_atomicTensor1 = torch::from_blob(c_atomicNew.data(), c_atomicNew.size(), torch::kInt32);
    auto c_atomicTensor1_seq = torch::from_blob(c_atomicNew_seq.data(), c_atomicNew_seq.size(), torch::kInt32);
    auto c_colTensor1 = torch::from_blob(c_columnNew.data(), c_columnNew.size(), torch::kInt32);
    auto c_valueTensor1 = torch::from_blob(c_valueNew.data(), c_valueNew.size(), torch::kFloat32);
    
    torch::Tensor c_row_offsetTensor = torch::empty_like(c_row_offsetTensor1);
    c_row_offsetTensor.copy_(c_row_offsetTensor1);
    torch::Tensor c_rowTensor = torch::empty_like(c_rowTensor1);
    c_rowTensor.copy_(c_rowTensor1);
    torch::Tensor c_atomicTensor = torch::empty_like(c_atomicTensor1);
    c_atomicTensor.copy_(c_atomicTensor1);
    torch::Tensor c_atomicTensor_seq = torch::empty_like(c_atomicTensor1_seq);
    c_atomicTensor_seq.copy_(c_atomicTensor1_seq);
    torch::Tensor c_colTensor = torch::empty_like(c_colTensor1);
    c_colTensor.copy_(c_colTensor1);
    torch::Tensor c_valueTensor = torch::empty_like(c_valueTensor1);
    c_valueTensor.copy_(c_valueTensor1);
    

    auto c_row_offsetTensor1_residue = torch::from_blob(c_rowNew_offset_residue.data(), c_rowNew_offset_residue.size(), torch::kInt32);
    auto c_rowTensor1_residue = torch::from_blob(c_rowNew_residue.data(), c_rowNew_residue.size(), torch::kInt32);
    auto c_atomicTensor1_residue = torch::from_blob(c_atomicNew_residue.data(), c_atomicNew_residue.size(), torch::kInt32);
    auto c_atomicTensor1_residue_seq = torch::from_blob(c_atomicNew_residue_seq.data(), c_atomicNew_residue_seq.size(), torch::kInt32); 
    auto c_colTensor1_residue = torch::from_blob(c_columnNew_residue.data(), c_columnNew_residue.size(), torch::kInt32);
    auto c_valueTensor1_residue = torch::from_blob(c_valueNew_residue.data(), c_valueNew_residue.size(), torch::kFloat32);
    
    torch::Tensor c_row_offsetTensor_residue = torch::empty_like(c_row_offsetTensor1_residue);
    c_row_offsetTensor_residue.copy_(c_row_offsetTensor1_residue);
    torch::Tensor c_rowTensor_residue = torch::empty_like(c_rowTensor1_residue);
    c_rowTensor_residue.copy_(c_rowTensor1_residue);
    torch::Tensor c_atomicTensor_residue = torch::empty_like(c_atomicTensor1_residue);
    c_atomicTensor_residue.copy_(c_atomicTensor1_residue);
    torch::Tensor c_atomicTensor_residue_seq = torch::empty_like(c_atomicTensor1_residue_seq);
    c_atomicTensor_residue_seq.copy_(c_atomicTensor1_residue_seq);
    torch::Tensor c_colTensor_residue = torch::empty_like(c_colTensor1_residue);
    c_colTensor_residue.copy_(c_colTensor1_residue);
    torch::Tensor c_valueTensor_residue = torch::empty_like(c_valueTensor1_residue);
    c_valueTensor_residue.copy_(c_valueTensor1_residue);

    return {t_rowNew_offsetTensor, t_blockNew_offsetTensor,t_columnTensor, t_valueTensor, t_window_rowTensor, t_atomicTensor, t_atomicTensor_seq, t_binaryTensor,
            c_row_offsetTensor, c_rowTensor, c_atomicTensor, c_atomicTensor_seq, c_colTensor, c_valueTensor,
            c_row_offsetTensor_residue, c_rowTensor_residue, c_atomicTensor_residue, c_atomicTensor_residue, c_colTensor_residue, c_valueTensor_residue, torch::tensor(elapsed.count())};

}


// m-tcu + cuda + partSize + binary
std::vector<torch::Tensor> block_tf32_tcu_cuda_binary_kernel_seq_1(torch::Tensor row1, torch::Tensor column1, torch::Tensor values1, 
int partSize_t, int partSize_c, int shortSize, int density, int window, int wide)
{
    auto *row=row1.data_ptr<int>();
    auto column=column1.accessor<int, 1>();
    auto values=values1.accessor<float, 1>();

    auto start = std::chrono::high_resolution_clock::now();

    int rows=row1.size(0)-1;
    // Result
    std::map<int, Value_tcu_binary> res_t;
    std::map<int, std::vector<Value_cuda>> res_c;
    std::map<int, std::vector<Value_cuda>> res_c_residue;
    //需要的window数量
    int num_windows = rows/window;
    #pragma omp parallel for
    for(int i=0;i<num_windows;i++)
    {
        /*
            1. 统计当前window中的vector, 并按照稠密度降序排列
        */
        //对column进行合并
        std::vector<int> mergedVector(&column[row[i*window]], &column[row[i*window+window]]);
        // 使用 unordered_map 统计 vector 中每个元素出现次数
        std::unordered_map<int, int> elementCounts;
        for (const auto& element : mergedVector) elementCounts[element]++;
        // 将统计结果放入 vector<pair> 中
        std::vector<std::pair<int, int>> countVector(elementCounts.begin(), elementCounts.end());
        // 按元素个数降序排序
        std::sort(countVector.begin(), countVector.end(), [](const auto& a, const auto& b) {
            return a.second > b.second;
        });

        /*
            2. 确定需要进行tcu计算的vector数量, t_percen
        */
        int size = countVector.size();
        int residue = size%wide;
        //统计每个block中的非零元数量，计算稠密度
        int t_percen = 0; //确定前多少个vector会被用作tcu进行稠密计算
        t_percen = std::count_if(countVector.begin(), countVector.end(), [density](const auto& pair) {
            return pair.second >= density;
        });
        int t_percen_ori = t_percen;
        if((t_percen%wide) != 0)
        {
            // if((t_percen + (wide - t_percen%wide)) <= size) 
            t_percen = (t_percen + (wide - t_percen%wide));
            // else
            // t_percen = (t_percen - (t_percen%wide));

        }
        // if(t_percen<=size){
        // // 将前 k 个数按升序排列
        // std::sort(countVector.begin(), countVector.begin() + t_percen);
        // // 将剩下的数按升序排列
        // std::sort(countVector.begin() + t_percen, countVector.end());
        // }else{
        //     std::sort(countVector.begin(), countVector.end());
        // }
        std::sort(countVector.begin(), countVector.begin()+t_percen_ori);
        std::sort(countVector.begin() + t_percen_ori, countVector.end());
        /*
            3. 根据t_percen分别划分TCU和CUDA的负载
        */

        // TCU 划分
        // 3.1 根据partSize_t划分t_part
        //当前window中有几个block
        int blocks = t_percen/wide; 
        Value_tcu_binary v_tcu;
        if(blocks>0) {v_tcu.tcu_flag = true; }
        std::map<int, int> colmap;
        int c=0;
        for(auto col:countVector){
            if (c<t_percen_ori)
            {
                v_tcu.t_column.push_back(col.first);
            }
            colmap[col.first]=c++;
        }
        if((t_percen_ori%wide)>0){
            for(int tt=0;tt<(wide-(t_percen_ori%wide));tt++)
            v_tcu.t_column.push_back(-1);
        }
        //存放拆分的cuda长短行数据
        std::vector<Value_cuda> large;
        std::vector<Value_cuda> small;

        std::vector<std::vector<float>> t_value_temp = createVectorOfVectors_float(blocks);
        std::vector<long> t_binary_temp = createVectorOfVectors_v2(blocks);
        // std::vector<std::vector<int>> t_row_temp = createVectorOfVectors(blocks);
        // std::vector<std::vector<int>> t_col_temp = createVectorOfVectors(blocks);
        // 指示快内的行偏移
        int p=0;
        for(int j=i*window;j<(i+1)*window;j++)
        {
            //用来存储每行的cuda数据
            std::queue<int> c_col;
            std::queue<int> c_value;
            int cuda_nnz=0;
            for(int m=row[j];m<row[j+1];m++)
            {
                int newCol = colmap[column[m]];
                if(newCol<t_percen_ori)
                {
                    int block = newCol/wide;
                    int bInId= newCol%wide;
                    t_value_temp[block].push_back(values[m]);
                    long offset = p*wide + bInId;
                    long a=1;
                    t_binary_temp[block] |= a<<offset;
                }else{
                    //划分cuda部分
                    c_col.push(column[m]);
                    c_value.push(values[m]);
                    cuda_nnz++;
                }
            }
            //处理当前行剩余的cuda元素
            if(cuda_nnz>0)
            {
                if(cuda_nnz <= shortSize)
                {
                    Value_cuda temp;
                    temp.cuda_residue_flag = true;
                    temp.c_row_offset = cuda_nnz;
                    temp.c_row = i*window + p;
                        if(t_percen_ori>0) 
                        {
                            temp.c_atomic = 1;
                            temp.c_atomic_seq = 0;
                        }
                        else{
                             temp.c_atomic = 0;
                             temp.c_atomic_seq = 0;
                        }
                    for(int m=0; m<cuda_nnz; m++)
                    {
                        temp.c_column.push_back(c_col.front());
                        temp.c_value.push_back(c_value.front());
                        c_col.pop();
                        c_value.pop();
                    }
                    small.push_back(temp);
                }else{
                    if(cuda_nnz <= partSize_c)
                    {
                        Value_cuda temp;
                        temp.cuda_flag = true;
                        temp.c_row_offset = (cuda_nnz);
                        temp.c_row = i*window + p;
                            if(t_percen_ori>0) 
                            {
                                temp.c_atomic = 1;
                                temp.c_atomic_seq = 0;
                            }
                            else{
                                temp.c_atomic = 0;
                                temp.c_atomic_seq = 0;
                            }
                        for(int m=0; m<cuda_nnz; m++)
                        {
                            temp.c_column.push_back(c_col.front());
                            temp.c_value.push_back(c_value.front());
                            c_col.pop();
                            c_value.pop();
                        }
                        large.push_back(temp);
                    }else{
                        // 当前行的非零元被分成多个part，并需要原子操作
                        int parts_row = cuda_nnz/partSize_c;
                        // int cur_value_indice = row[i];
                        for(int cur_i = 0; cur_i<parts_row; cur_i++)
                        {
                            Value_cuda temp;
                            temp.cuda_flag = true;      
                            temp.c_row_offset = partSize_c;
                            temp.c_row = i*window + p;
                            temp.c_atomic = 1;
                            temp.c_atomic_seq = 1;
                            for(int m=0; m<partSize_c; m++)
                            {
                                temp.c_column.push_back(c_col.front());
                                temp.c_value.push_back(c_value.front());
                                c_col.pop();
                                c_value.pop();
                                // cur_value_indice++;
                            }
                            large.push_back(temp);
                        }
                        int residue  = (cuda_nnz%partSize_c);
                        if(residue>0){
                            Value_cuda temp;
                            temp.c_row_offset = residue;
                            temp.c_row = i*window + p;
                            temp.c_atomic = 1;
                            temp.c_atomic_seq = 1;
                            for(int m=0; m<residue; m++)
                            {
                                temp.c_column.push_back(c_col.front());
                                temp.c_value.push_back(c_value.front());
                                c_col.pop();
                                c_value.pop();
                                // cur_value_indice++;
                            }
                            if(residue <= shortSize)
                            {
                                temp.cuda_residue_flag = true;
                                small.push_back(temp);
                            }else{
                                temp.cuda_flag = true;
                                large.push_back(temp);
                            }
                        }
                    }
                }
            }
            p++;
        }
        //整合vector
        v_tcu.t_value = mergeVectors_float(t_value_temp);
        v_tcu.t_binary = t_binary_temp;
        // v_tcu.t_row = mergeVectors(t_row_temp);
        // v_tcu.t_col = mergeVectors(t_col_temp);
        
        std::vector<int> count_tcu = countVectors(t_value_temp);
        if(blocks <= partSize_t)
        {
            v_tcu.row.push_back(blocks);
            //循放入每个block中的非零元数量
            for(int b=0; b<blocks; b++)
            {
                // int sum = std::accumulate(count_tcu.begin()+b, count_tcu.begin()+b+1, 0);
                v_tcu.t_block_offset.push_back(count_tcu[b]);
            }
                if(t_percen_ori==size) 
                {
                    v_tcu.atomic.push_back(0);
                    v_tcu.atomic_seq.push_back(0);
                }
                else {
                    v_tcu.atomic.push_back(1);
                    v_tcu.atomic_seq.push_back(0);
                }
            // if(t_percen_ori==size) v_tcu.atomic.push_back(0);
            // else v_tcu.atomic.push_back(1);
            v_tcu.window.push_back(i);
        }else{
            // 对block太多的块进行分割
            int part_number = blocks / partSize_t;
            int block_residue = blocks % partSize_t;
            for(int j=0; j<part_number; j++)
            {
                v_tcu.row.push_back(partSize_t);
                for(int b=0; b<partSize_t; b++)
                {
                    int offset = j*partSize_t + b;
                    // int offset1 = offset + b;
                    // int sum = std::accumulate(count_tcu.begin()+offset, count_tcu.begin()+offset1, 0);
                    v_tcu.t_block_offset.push_back(count_tcu[offset]);
                }
                v_tcu.window.push_back(i);
                v_tcu.atomic.push_back(1);
                v_tcu.atomic_seq.push_back(1);
            }
            if(block_residue > 0)
            {             
                v_tcu.row.push_back(block_residue);
                for(int b=0; b<block_residue; b++)
                {
                    int offset = part_number*partSize_t+b;
                    // int offset1 = offset + b;
                    // int sum = std::accumulate(count_tcu.begin()+offset, count_tcu.begin()+offset1, 0);
                    v_tcu.t_block_offset.push_back(count_tcu[offset]);
                }
                v_tcu.window.push_back(i);
                v_tcu.atomic.push_back(1);
                v_tcu.atomic_seq.push_back(1);
            }
        }
        /*
            4. 汇总三部分的结果
        */
        #pragma omp critical  
        res_t[i]=v_tcu;

        #pragma omp critical  
        res_c[i]=large;

        #pragma omp critical  
        res_c_residue[i]=small;
        
    
    }
    /*
        5. 整合所有window的数据
    */
    //tcu
    std::vector<int> t_rowkNew_offset;
    t_rowkNew_offset.push_back(0);
    std::vector<int> t_blockNew_offset;
    t_blockNew_offset.push_back(0);
    std::vector<float> t_valueNew;
    std::vector<int> t_columnNew;
    std::vector<long> t_binaryNew;
    std::vector<int> t_window_rowNew;
    std::vector<int> t_atomicNew;
    std::vector<int> t_atomicNew_seq;

    //cuda 长行
    std::vector<int> c_rowNew_offset;
    c_rowNew_offset.push_back(0);
    std::vector<int> c_rowNew;
    std::vector<int> c_atomicNew;
    std::vector<int> c_atomicNew_seq;
    std::vector<int> c_columnNew;
    std::vector<float> c_valueNew;

    //cuda 短行
    std::vector<int> c_rowNew_offset_residue;
    c_rowNew_offset_residue.push_back(0);
    std::vector<int> c_rowNew_residue;
    std::vector<int> c_atomicNew_residue;
    std::vector<int> c_atomicNew_residue_seq;
    std::vector<int> c_columnNew_residue;
    std::vector<float> c_valueNew_residue;

    //按顺序整合res
    //tcu
    for (const auto& pair : res_t) {
        if (pair.second.tcu_flag == true )
        {
            for(int sub : pair.second.row)
            {
                t_rowkNew_offset.push_back(t_rowkNew_offset.back()+sub);
            }  
            for(int sub : pair.second.t_block_offset)
            {
                t_blockNew_offset.push_back(t_blockNew_offset.back()+sub);
            }  
            t_valueNew.insert(t_valueNew.end(),pair.second.t_value.begin(),pair.second.t_value.end());
            t_columnNew.insert(t_columnNew.end(),pair.second.t_column.begin(),pair.second.t_column.end());
            t_window_rowNew.insert(t_window_rowNew.end(),pair.second.window.begin(),pair.second.window.end());
            t_atomicNew.insert(t_atomicNew.end(),pair.second.atomic.begin(),pair.second.atomic.end());
            t_atomicNew_seq.insert(t_atomicNew_seq.end(),pair.second.atomic_seq.begin(),pair.second.atomic_seq.end());
            t_binaryNew.insert(t_binaryNew.end(),pair.second.t_binary.begin(),pair.second.t_binary.end());
        }   
    }

    //cuda long
    for (const auto& pair : res_c) {
        for( Value_cuda sub : pair.second){
            if (sub.cuda_flag == true )
            {c_rowNew_offset.push_back(c_rowNew_offset.back()+sub.c_row_offset);
            c_rowNew.push_back(sub.c_row);
            c_atomicNew.push_back(sub.c_atomic);
            c_atomicNew_seq.push_back(sub.c_atomic_seq);
            c_columnNew.insert(c_columnNew.end(),sub.c_column.begin(),sub.c_column.end());
            c_valueNew.insert(c_valueNew.end(),sub.c_value.begin(), sub.c_value.end());}
        }
    }

    //cuda short
    for (const auto& pair : res_c_residue) {
        for( Value_cuda sub : pair.second){
            if (sub.cuda_residue_flag == true )
            {c_rowNew_offset_residue.push_back(c_rowNew_offset_residue.back()+sub.c_row_offset);
            c_rowNew_residue.push_back(sub.c_row);
            c_atomicNew_residue.push_back(sub.c_atomic);
            c_atomicNew_residue_seq.push_back(sub.c_atomic_seq);
            c_columnNew_residue.insert(c_columnNew_residue.end(),sub.c_column.begin(),sub.c_column.end());
            c_valueNew_residue.insert(c_valueNew_residue.end(),sub.c_value.begin(), sub.c_value.end());}
        }
    }

    auto end = std::chrono::high_resolution_clock::now();

    // 计算耗时
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;

    auto t_rowNew_offsetTensor1 = torch::from_blob(t_rowkNew_offset.data(), t_rowkNew_offset.size(), torch::kInt32);
    auto t_blockNew_offsetTensor1 = torch::from_blob(t_blockNew_offset.data(), t_blockNew_offset.size(), torch::kInt32);
    auto t_valueTensor1 = torch::from_blob(t_valueNew.data(), t_valueNew.size(), torch::kFloat32);
    auto t_columnTensor1 = torch::from_blob(t_columnNew.data(), t_columnNew.size(), torch::kInt32);
    auto t_window_rowTensor1 = torch::from_blob(t_window_rowNew.data(), t_window_rowNew.size(), torch::kInt32);
    auto t_atomicTensor1 = torch::from_blob(t_atomicNew.data(), t_atomicNew.size(), torch::kInt32);
    auto t_atomicTensor1_seq = torch::from_blob(t_atomicNew_seq.data(), t_atomicNew_seq.size(), torch::kInt32);
    auto t_binaryTensor1 = torch::from_blob(t_binaryNew.data(), t_binaryNew.size(), torch::kInt64);

    torch::Tensor t_rowNew_offsetTensor = torch::empty_like(t_rowNew_offsetTensor1);
    t_rowNew_offsetTensor.copy_(t_rowNew_offsetTensor1);
    torch::Tensor t_blockNew_offsetTensor = torch::empty_like(t_blockNew_offsetTensor1);
    t_blockNew_offsetTensor.copy_(t_blockNew_offsetTensor1);
    torch::Tensor t_valueTensor = torch::empty_like(t_valueTensor1);
    t_valueTensor.copy_(t_valueTensor1);
    torch::Tensor t_columnTensor = torch::empty_like(t_columnTensor1);
    t_columnTensor.copy_(t_columnTensor1);
    torch::Tensor t_window_rowTensor = torch::empty_like(t_window_rowTensor1);
    t_window_rowTensor.copy_(t_window_rowTensor1);
    torch::Tensor t_atomicTensor = torch::empty_like(t_atomicTensor1);
    t_atomicTensor.copy_(t_atomicTensor1);
    torch::Tensor t_binaryTensor = torch::empty_like(t_binaryTensor1);
    t_binaryTensor.copy_(t_binaryTensor1);
    torch::Tensor t_atomicTensor_seq = torch::empty_like(t_atomicTensor1_seq);
    t_atomicTensor_seq.copy_(t_atomicTensor1_seq);

    auto c_row_offsetTensor1 = torch::from_blob(c_rowNew_offset.data(), c_rowNew_offset.size(), torch::kInt32);
    auto c_rowTensor1 = torch::from_blob(c_rowNew.data(), c_rowNew.size(), torch::kInt32);
    auto c_atomicTensor1 = torch::from_blob(c_atomicNew.data(), c_atomicNew.size(), torch::kInt32);
    auto c_atomicTensor1_seq = torch::from_blob(c_atomicNew_seq.data(), c_atomicNew_seq.size(), torch::kInt32);
    auto c_colTensor1 = torch::from_blob(c_columnNew.data(), c_columnNew.size(), torch::kInt32);
    auto c_valueTensor1 = torch::from_blob(c_valueNew.data(), c_valueNew.size(), torch::kFloat32);
    
    torch::Tensor c_row_offsetTensor = torch::empty_like(c_row_offsetTensor1);
    c_row_offsetTensor.copy_(c_row_offsetTensor1);
    torch::Tensor c_rowTensor = torch::empty_like(c_rowTensor1);
    c_rowTensor.copy_(c_rowTensor1);
    torch::Tensor c_atomicTensor = torch::empty_like(c_atomicTensor1);
    c_atomicTensor.copy_(c_atomicTensor1);
    torch::Tensor c_atomicTensor_seq = torch::empty_like(c_atomicTensor1_seq);
    c_atomicTensor_seq.copy_(c_atomicTensor1_seq);
    torch::Tensor c_colTensor = torch::empty_like(c_colTensor1);
    c_colTensor.copy_(c_colTensor1);
    torch::Tensor c_valueTensor = torch::empty_like(c_valueTensor1);
    c_valueTensor.copy_(c_valueTensor1);
    

    auto c_row_offsetTensor1_residue = torch::from_blob(c_rowNew_offset_residue.data(), c_rowNew_offset_residue.size(), torch::kInt32);
    auto c_rowTensor1_residue = torch::from_blob(c_rowNew_residue.data(), c_rowNew_residue.size(), torch::kInt32);
    auto c_atomicTensor1_residue = torch::from_blob(c_atomicNew_residue.data(), c_atomicNew_residue.size(), torch::kInt32);
    auto c_atomicTensor1_residue_seq = torch::from_blob(c_atomicNew_residue_seq.data(), c_atomicNew_residue_seq.size(), torch::kInt32); 
    auto c_colTensor1_residue = torch::from_blob(c_columnNew_residue.data(), c_columnNew_residue.size(), torch::kInt32);
    auto c_valueTensor1_residue = torch::from_blob(c_valueNew_residue.data(), c_valueNew_residue.size(), torch::kFloat32);
    
    torch::Tensor c_row_offsetTensor_residue = torch::empty_like(c_row_offsetTensor1_residue);
    c_row_offsetTensor_residue.copy_(c_row_offsetTensor1_residue);
    torch::Tensor c_rowTensor_residue = torch::empty_like(c_rowTensor1_residue);
    c_rowTensor_residue.copy_(c_rowTensor1_residue);
    torch::Tensor c_atomicTensor_residue = torch::empty_like(c_atomicTensor1_residue);
    c_atomicTensor_residue.copy_(c_atomicTensor1_residue);
    torch::Tensor c_atomicTensor_residue_seq = torch::empty_like(c_atomicTensor1_residue_seq);
    c_atomicTensor_residue_seq.copy_(c_atomicTensor1_residue_seq);
    torch::Tensor c_colTensor_residue = torch::empty_like(c_colTensor1_residue);
    c_colTensor_residue.copy_(c_colTensor1_residue);
    torch::Tensor c_valueTensor_residue = torch::empty_like(c_valueTensor1_residue);
    c_valueTensor_residue.copy_(c_valueTensor1_residue);

    return {t_rowNew_offsetTensor, t_blockNew_offsetTensor,t_columnTensor, t_valueTensor, t_window_rowTensor, t_atomicTensor, t_atomicTensor_seq, t_binaryTensor,
            c_row_offsetTensor, c_rowTensor, c_atomicTensor, c_atomicTensor_seq, c_colTensor, c_valueTensor,
            c_row_offsetTensor_residue, c_rowTensor_residue, c_atomicTensor_residue, c_atomicTensor_residue, c_colTensor_residue, c_valueTensor_residue, torch::tensor(elapsed.count())};

}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

    m.def("block_sgt_tcu", &block_sgt_tcu_kernel, "纯tcu");
    m.def("block_csr_tcu", &block_csr_tcu_kernel, "ME-TCF");
    m.def("block_bcrs_tcu", &block_bcrs_tcu_kernel, "bcrs");
    m.def("block_csr_tcu_part", &block_csr_tcu_part_kernel, "纯tcu");
    m.def("block_csr_binary_tcu", &block_csr_binary_tcu_kernel, "纯tcu+binary");
    m.def("block_csr_binary_tcu_part", &block_csr_binary_tcu_part_kernel, "纯tcu 加partSize ");

    m.def("block_csr_binary_sddmm_tcu_part", &block_csr_binary_sddmm_tcu_part_kernel, "sddmm 纯tcu 加partSize ");
    m.def("block_csr_binary_sddmm_tcu_part_c", &block_csr_binary_sddmm_tcu_part_c_kernel, "sddmm 纯tcu 加partSize, 列优先");
    m.def("block_csr_binary_sddmm_tcu_part_c1", &block_csr_binary_sddmm_tcu_part_c1_kernel, "sddmm 纯tcu 加partSize, 列优先");

    m.def("block_csr_cuda_v2", &block_csr_cuda_kernel, "cuda长短行");
    m.def("block_csr_sddmm_cuda_v2", &block_csr_sddmm_cuda_kernel, "cuda长短行");
    
    m.def("block_tf32_tcu_cuda_csr", &block_tf32_tcu_cuda_csr_kernel, "Libra + csr");
    m.def("block_tf32_tcu_cuda_binary", &block_tf32_tcu_cuda_binary_kernel, "Libra + csr_binary");
    m.def("block_tf32_tcu_cuda_binary_new", &block_tf32_tcu_cuda_binary_kernel_new, "Libra + csr_binary");
    m.def("block_tf32_tcu_cuda_binary2", &block_tf32_tcu_cuda_binary_kernel2, "Libra + csr_binary");
    m.def("block_tf32_tcu_cuda_binary_sddmm", &block_tf32_tcu_cuda_binary_kernel_sddmm, "Libra + csr_binary");
    m.def("nnz_profile", &nnz_profile_kernel, "profile nnz");
    m.def("block_tf32_tcu_cuda_binary_seq", &block_tf32_tcu_cuda_binary_kernel_seq, "Libra + csr_binary");
    m.def("block_tf32_tcu_cuda_binary_seq_1", &block_tf32_tcu_cuda_binary_kernel_seq_1, "Libra + csr_binary");
    m.def("block_tf32_tcu_cuda_binary_seq_hcspmm", &block_tf32_tcu_cuda_binary_kernel_seq_hcspmm, "Libra + csr_binary");
    // m.def("block_tf32_tcu_cuda_binary_seq_hcspmm1", &block_tf32_tcu_cuda_binary_kernel_seq_hcspmm1, "no balance");
    m.def("block_tf32_tcu_cuda_binary_seq_pcgcn", &block_tf32_tcu_cuda_binary_kernel_seq_pcgcn, "Libra + csr_binary");
    m.def("block_tf32_tcu_cuda_binary_seq_rtgnn", &block_tf32_tcu_cuda_binary_kernel_seq_rtgnn, "Libra + csr_binary");
    m.def("block_csr_tcu_stream", &block_csr_tcu_kernel_stream, "测stream");


}
