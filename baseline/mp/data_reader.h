#include <vector>
#include <string>
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
);
