#include "data_reader.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cuda_fp16.h>

bool read_data_from_files(
    const std::string& binary_file_path,
    std::vector<int>& tcblocks_offset,
    std::vector<double>& opdA,
    std::vector<int>& metadata,
    std::vector<int>& old_col_all,
    int& num_rows,
    int& num_cols,
    int& num_nonzeros
) {

    std::ifstream binary_file(binary_file_path, std::ios::binary);
    if (!binary_file) {
        std::cerr << "error open file" << std::endl;
        return false;
    }

    binary_file.read(reinterpret_cast<char*>(&num_rows), sizeof(num_rows));
    binary_file.read(reinterpret_cast<char*>(&num_cols), sizeof(num_cols));
    
    binary_file.read(reinterpret_cast<char*>(&num_nonzeros), sizeof(num_nonzeros));

    int num_tcblocks_offset;
    binary_file.read(reinterpret_cast<char*>(&num_tcblocks_offset), sizeof(num_tcblocks_offset));
    tcblocks_offset.resize(num_tcblocks_offset);
    binary_file.read(reinterpret_cast<char*>(tcblocks_offset.data()), num_tcblocks_offset * sizeof(int));

    int num_opdA;
    binary_file.read(reinterpret_cast<char*>(&num_opdA), sizeof(num_opdA));
    opdA.resize(num_opdA);
    binary_file.read(reinterpret_cast<char*>(opdA.data()), num_opdA * sizeof(double));

    int num_metadata;
    binary_file.read(reinterpret_cast<char*>(&num_metadata), sizeof(num_metadata));
    metadata.resize(num_metadata);
    binary_file.read(reinterpret_cast<char*>(metadata.data()), num_metadata * sizeof(int));

    int num_old_col_all;
    binary_file.read(reinterpret_cast<char*>(&num_old_col_all), sizeof(num_old_col_all));
    old_col_all.resize(num_old_col_all);
    binary_file.read(reinterpret_cast<char*>(old_col_all.data()), num_old_col_all * sizeof(int));

    binary_file.close();


    return true;
 }