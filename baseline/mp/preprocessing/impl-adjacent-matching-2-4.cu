#include <iostream>
#include <vector>
#include <cassert>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <iomanip>

#include <thrust/scan.h>
#include <thrust/for_each.h>
#include <thrust/execution_policy.h>
#include <cooperative_groups.h>
#include <filesystem> 
#include <fstream>
extern "C" {
    #include "libmmio/mmio.c"
}


const char* get_config_value(const char* key) {
    static char buffer[512];
    FILE* config_file = fopen("../../path.txt", "r");
    if (!config_file) {
        fprintf(stderr, "Error: Cannot open path.txt\n");
        exit(EXIT_FAILURE);
    }

    while (fgets(buffer, sizeof(buffer), config_file)) {
        char* delimiter = strchr(buffer, '=');
        if (!delimiter) continue;

        *delimiter = '\0';
        char* key_start = buffer + strspn(buffer, " \t");
        char* key_end = buffer + strcspn(buffer, " \t");
        *key_end = '\0';

        if (strcmp(key_start, key) == 0) {
            char* value = delimiter + 1;
            while (*value && (*value == ' ' || *value == '\t')) value++;
            value[strcspn(value, "\r\n")] = '\0';
            fclose(config_file);
            return value;
        }
    }

    fclose(config_file);
    fprintf(stderr, "Error: %s not found in path.txt\n", key);
    exit(EXIT_FAILURE);
}


namespace Matching_And_Padding {
/****************************************
 *          Adjacent Matching           *
 ****************************************/

// #define TEST_DATA_PATH

const char* data_root = get_config_value("dataset_path");

const char* resolve_dataset_dir() {
    static char resolved_path[512];
    snprintf(resolved_path, sizeof(resolved_path), "%s/dataset", data_root);
    if (std::filesystem::is_directory(resolved_path)) return resolved_path;

    snprintf(resolved_path, sizeof(resolved_path), "%s/datasets", data_root);
    if (std::filesystem::is_directory(resolved_path)) return resolved_path;

    snprintf(resolved_path, sizeof(resolved_path), "%s", data_root);
    return resolved_path;
}

const char* dataset_dir = resolve_dataset_dir();

char mtx_path[512];

// TC block size
constexpr int TC_row_len  = 16; 
constexpr int TC_col_len  = 16; 
constexpr int Iterative_K = 10; 
constexpr double Iterative_factor = 0.01; 

struct NonzeroElement{ double val; int row, col; }; // 16 bytes
struct SparseMatrix {
    int rows, cols, nonzeros;
    int tc_rownum, tc_colnum;
    std::vector<NonzeroElement>* windows;   
    NonzeroElement* nz_list;               
    NonzeroElement* nz_list_dev;
    int* window_offset;                     
    int* window_offset_dev;
    MM_typecode matcode;                    
    bool is_symmetric;
};

struct UnitsPack{
    int *col1_dev, *col2_dev, *end_dev, *window_unit_num_dev;
    uint32_t *vec_dev;
};
struct MatchResult {
    int *col1_dev, *col2_dev, *end_dev; 
    uint32_t *vec_dev;
    int *match_num_dev; 
    int *padded_block_num; 
};


struct PaddedResult{
    double* opdA_dev;
    int *metadata_dev;
    int *old_col_dev;
};

struct MetadataResult{
    int* window_offset;
    double* opdA;
    int *metadata, *old_col;
};


#define TIME_INIT std::chrono::time_point<std::chrono::high_resolution_clock> time_start, time_end
#define TIME(flag) time_##flag = std::chrono::high_resolution_clock::now()
//#define TIME_COUNT std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count()
#define TIME_COUNT (std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start).count() / 1000.0)

SparseMatrix* read_matrix(const char* mtx_name, int& nonzero_block) { 

    snprintf(mtx_path, sizeof(mtx_path), "%s/%s.mtx", dataset_dir, mtx_name);
    FILE* fp = fopen(mtx_path, "r");
    if (fp == NULL) {
        std::cerr << "Error: File " << mtx_path << " does not exist." << std::endl;
        exit(1);
    }
    std::cerr << "| file path: " << mtx_path << std::endl;

    SparseMatrix* mtx_ptr = (SparseMatrix*)malloc(sizeof(SparseMatrix));
    SparseMatrix& mtx = *mtx_ptr;


    if (mm_read_banner(fp, &mtx.matcode) != 0) {
        std::cerr << "Could not process Matrix Market banner." << std::endl;
        exit(1);
    }
    if (!mm_is_coordinate(mtx.matcode)) {
        std::cerr << "Invalid matrix type." << std::endl;
        exit(1);
    }

    mtx.is_symmetric = mm_is_symmetric(mtx.matcode);
    if (mtx.is_symmetric) {
        std::cerr << "| Matrix is symmetric, will be expanded to full matrix" << std::endl;
    }

    if (mm_read_mtx_crd_size(fp, &mtx.rows, &mtx.cols, &mtx.nonzeros) != 0) {
        std::cerr << "Reading matrix size failed." << std::endl;
        exit(1);
    }
    std::cerr << "| matrix info: rows=" << mtx.rows << ", cols=" << mtx.cols << ", nonzeros=" << mtx.nonzeros << std::endl;
    

    if (mtx.rows % TC_row_len != 0) mtx.rows += TC_row_len - (mtx.rows % TC_row_len);
    assert(mtx.rows % TC_row_len == 0);
    mtx.tc_rownum = mtx.rows / TC_row_len;
    mtx.tc_colnum = (mtx.cols + TC_col_len - 1) / TC_col_len; 
    mtx.windows = new std::vector<NonzeroElement>[mtx.tc_rownum];
    


    int expected_nonzeros = mtx.is_symmetric ? mtx.nonzeros * 2 : mtx.nonzeros;
    mtx.nz_list = (NonzeroElement*)malloc(sizeof(NonzeroElement) * expected_nonzeros);
    mtx.window_offset = (int*)malloc(sizeof(int) * (mtx.tc_rownum + 1)); 

     nonzero_block = 0;
     double val;
     int total_elements = 0;
     
     for (int i = 1, row, col, tc_row, tc_col; i <= mtx.nonzeros; i++) {
         fscanf(fp, "%d%d%lf", &row, &col, &val);
         --row, --col; 
         
         tc_row = row / TC_row_len;
         tc_col = col / TC_col_len;
         int local_row = row % TC_row_len; 
         mtx.windows[tc_row].push_back((NonzeroElement){val,local_row, col});
         total_elements++;
 
         if (mtx.is_symmetric && row != col) {
             tc_row = col / TC_row_len; 
             local_row = col % TC_row_len;
             mtx.windows[tc_row].push_back((NonzeroElement){val,local_row, row});
             total_elements++;
         }
 
         #ifdef COUNT_ORIGINAL_TC_BLOCKS_QUANTITY
         unsigned long long tc_id = (unsigned long long)tc_row * mtx.tc_colnum + tc_col;
         if (Original_TC_block_occupy.find(tc_id) == Original_TC_block_occupy.end()) {
             Original_TC_block_occupy.insert(tc_id);
             nonzero_block++;
         }
         
         if (mtx.is_symmetric && row != col) {
             tc_row = col / TC_row_len; 
             tc_col = row / TC_col_len;
             tc_id = (unsigned long long)tc_row * mtx.tc_colnum + tc_col;
             if (Original_TC_block_occupy.find(tc_id) == Original_TC_block_occupy.end()) {
                 Original_TC_block_occupy.insert(tc_id);
                 nonzero_block++;
             }
         }
         #endif
     }

          if (mtx.is_symmetric) {
            mtx.nonzeros = total_elements;
            std::cerr << "| After symmetrization: nonzeros=" << mtx.nonzeros << std::endl;
        }
   
    int nz_cnt = 0, size_cnt;
    NonzeroElement* list_p = mtx.nz_list;
    for (int i = 0; i < mtx.tc_rownum; i++) {
        mtx.window_offset[i] = nz_cnt;
        size_cnt = mtx.windows[i].size();

        if (size_cnt > 0) {
            std::sort(mtx.windows[i].begin(), mtx.windows[i].end(), 
                [](const NonzeroElement& a, const NonzeroElement& b) -> bool {
                    return a.col < b.col;
                }
            );

            std::memcpy(list_p, mtx.windows[i].data(), sizeof(NonzeroElement) * size_cnt);
        }
        list_p += size_cnt;
        nz_cnt += size_cnt;
    }
    mtx.window_offset[mtx.tc_rownum] = nz_cnt;
    assert(nz_cnt == mtx.nonzeros);
    assert(list_p == mtx.nz_list + mtx.nonzeros);

    fclose(fp);
    return mtx_ptr;
}

template<unsigned int Blocksize>
__global__ void pack_columns_into_units_kernel(const NonzeroElement* nz_list, const int* window_offset, 
    int* col1_dev, int* col2_dev, int* end_dev, int* window_unit_num_dev, uint32_t* vec_dev) {
    const int window_id = blockIdx.x;
    const int window_l = window_offset[window_id], window_r = window_offset[window_id + 1];
    const int window_size = window_r - window_l;
    const int array_size = window_size + ((window_size % (Blocksize * 2) == 0) ? 0 : (Blocksize * 2) - window_size % (Blocksize * 2));


    if (window_size == 0 || window_size > 2048) {
        return; 
    }

    // 28672 bytes + 6146 bytes = 34818 bytes = 34 KB
    __shared__ uint32_t nz_cols[2048];  
    __shared__ uint16_t col_id[2048 + 1]; 
    __shared__ int unit_col0[1024], unit_col1[1024], unit_end[1024];
    __shared__ uint32_t unit_vec[2048];
    __shared__ uint16_t unit_offset[1024];
    int block_colid_sum, unit_num;


    for (int pos = threadIdx.x, bound = Blocksize; ; pos += Blocksize, bound += Blocksize) {
        if (pos < window_size) {
            unit_vec[pos] = (uint32_t)1 << (nz_list[window_l + pos].row * 2);
            nz_cols[pos] = nz_list[window_l + pos].col;
        }
        __syncthreads();
        if (pos < window_size) {
            if (pos == 0 || nz_cols[pos] != nz_cols[pos - 1]) {
                col_id[pos] = 1; 
            } else {
                col_id[pos] = 0;
            }
        }
        if (bound >= window_size) break;
    }
    __syncthreads();

    for (int pos = threadIdx.x * 2 + 1; pos < array_size; pos += Blocksize * 2) {
        col_id[pos] += col_id[pos - 1];
        __syncthreads();
        {
            bool active = true;
            for (int step = 1; step < Blocksize; step <<= 1) {
                if ((threadIdx.x & step) == 0) active = false;
                if (active == true) {
                    col_id[pos] += col_id[pos - step * 2];
                }
                __syncthreads();
            }
        }

        if (threadIdx.x == Blocksize - 1) {
            block_colid_sum = col_id[pos];
            col_id[pos] = 0;
        }
        __syncthreads();

        for (int step = Blocksize - 1; step; step >>= 1) {
            if ((threadIdx.x & step) == step) {
                int temp = col_id[pos - step - 1];
                col_id[pos - step - 1] = col_id[pos];
                col_id[pos] += temp;
            }
            __syncthreads();
        }

        int temp = col_id[pos - 1], temp2 = col_id[pos];
        __syncthreads();
        if (threadIdx.x != 0) col_id[pos - 2] = temp2 - 1;  
        col_id[pos - 1] = temp + temp2 - 1;                
        
        if (threadIdx.x == Blocksize - 1) {
            col_id[pos] = block_colid_sum - 1;
            col_id[pos + 1] += block_colid_sum;
        }
        __syncthreads();
    }

    unit_num = col_id[window_size - 1] / 2 + 1;
    if (threadIdx.x == 0) {
        if ((col_id[window_size - 1] & 1) == 0) { 
            unit_col1[unit_num - 1] = -1;
            unit_end[unit_num - 1] = window_r;
        }
    }
    __syncthreads();

    for (int pos = threadIdx.x; pos < window_size; pos += Blocksize) {
        if (pos == 0 || col_id[pos] != col_id[pos - 1]) {
            int unit_idx = col_id[pos] / 2;
            if ((col_id[pos] & 1) == 0) {
                unit_offset[unit_idx] = pos;
                unit_col0[unit_idx] = window_l + pos;
            } else {
                unit_col1[unit_idx] = window_l + pos;
            }
        }
    }
    __syncthreads();

    for (int pos = threadIdx.x, bound = Blocksize, unit_idx; ; pos += Blocksize, bound += Blocksize) {
        if (pos < window_size) {
            unit_idx = col_id[pos] / 2;
            if (((col_id[pos] & 1) == 1) && (pos == window_size - 1 || col_id[pos] != col_id[pos + 1])) {
                unit_end[unit_idx] = window_l + pos + 1;
            }
        }
        __syncthreads();
        if (pos < window_size) {
            col_id[pos] = pos - unit_offset[unit_idx]; 
        }
        if (bound >= window_size) break;
    }
    __syncthreads();

    for (int pos = threadIdx.x; pos < window_size; pos += Blocksize)
        if (col_id[pos] >= 16) unit_vec[pos - 16] += unit_vec[pos];
    __syncthreads();
    for (int pos = threadIdx.x; pos < window_size; pos += Blocksize)
        if (col_id[pos] < 16 && col_id[pos] >= 8) unit_vec[pos - 8] += unit_vec[pos];
    __syncthreads();
    for (int pos = threadIdx.x; pos < window_size; pos += Blocksize)
        if (col_id[pos] < 8  && col_id[pos] >= 4) unit_vec[pos - 4] += unit_vec[pos];
    __syncthreads();
    for (int pos = threadIdx.x; pos < window_size; pos += Blocksize)
        if (col_id[pos] < 4  && col_id[pos] >= 2) unit_vec[pos - 2] += unit_vec[pos];
    __syncthreads();
    for (int pos = threadIdx.x; pos < window_size; pos += Blocksize)
        if (col_id[pos] == 1) unit_vec[pos - 1] += unit_vec[pos];
    __syncthreads();

    for (int pos = threadIdx.x; pos < unit_num; pos += Blocksize) {
        col1_dev[window_l + pos] = unit_col0[pos];
        col2_dev[window_l + pos] = unit_col1[pos];
        end_dev[window_l + pos] = unit_end[pos];
        vec_dev[window_l + pos] = unit_vec[unit_offset[pos]];
    }
    if (threadIdx.x == 0) {
        window_unit_num_dev[blockIdx.x] = unit_num;
    }
    __syncthreads();
}

UnitsPack* pack_columns_into_units(SparseMatrix* mtx_ptr, int& tc_blocks) {
    SparseMatrix& mtx = *mtx_ptr;
    cudaMalloc(&mtx.nz_list_dev, mtx.nonzeros * sizeof(NonzeroElement));
    cudaMemcpy(mtx.nz_list_dev, mtx.nz_list, mtx.nonzeros * sizeof(NonzeroElement), cudaMemcpyHostToDevice);
    cudaMalloc(&mtx.window_offset_dev, (mtx.tc_rownum + 1) * sizeof(int));
    cudaMemcpy(mtx.window_offset_dev, mtx.window_offset, (mtx.tc_rownum + 1) * sizeof(int), cudaMemcpyHostToDevice);

    UnitsPack* units_ptr = (UnitsPack*)malloc(sizeof(UnitsPack));
    UnitsPack& units = *units_ptr;
    cudaMalloc(&units.col1_dev, mtx.nonzeros * sizeof(int));
    cudaMalloc(&units.col2_dev, mtx.nonzeros * sizeof(int));
    cudaMalloc(&units.end_dev, mtx.nonzeros * sizeof(int));
    cudaMalloc(&units.window_unit_num_dev, (mtx.tc_rownum + 1) * sizeof(int));
    cudaMalloc(&units.vec_dev, mtx.nonzeros * sizeof(uint32_t));

    pack_columns_into_units_kernel<128><<<mtx.tc_rownum, 128>>>(
        mtx.nz_list_dev, mtx.window_offset_dev,
        units.col1_dev, units.col2_dev, units.end_dev, units.window_unit_num_dev, units.vec_dev
    );

    constexpr int STREAM_NUM = 4;
    cudaStream_t streams[STREAM_NUM];
    for (int i = 0; i < STREAM_NUM; i++) {
        cudaStreamCreate(&streams[i]);
    }
    NonzeroElement* list = mtx.nz_list;
    for (int w = 0; w < mtx.tc_rownum; w++) {
        int window_l = mtx.window_offset[w];
        int window_r = mtx.window_offset[w + 1];

        if (window_r - window_l <= 2048) continue;

        int unit_num_h = 0;
        std::vector<int> col1_h, col2_h, end_h;
        std::vector<uint32_t> vec_h;

        uint32_t vec = 0;
        int cols[2] = {-1, -1};
        int colnum = 0;

        for (int l = window_l, r; l < window_r; l = r, colnum++) {
            for (r = l; r < window_r && list[r].col == list[l].col; r++) {
                vec += (uint32_t)1 << (2 * list[r].row);
            }
            cols[colnum & 1] = l;
            if (colnum & 1) {
                col1_h.emplace_back(cols[0]);
                col2_h.emplace_back(cols[1]);
                end_h.emplace_back(r);
                vec_h.emplace_back(vec);
                unit_num_h++;

                vec = 0, cols[0] = cols[1] = -1;
            }
        }
        if (colnum & 1) { 
            col1_h.emplace_back(cols[0]);
            col2_h.emplace_back(-1);
            end_h.emplace_back(window_r);
            vec_h.emplace_back(vec);
            unit_num_h++;

            colnum++;
        }

        cudaStream_t& trans = streams[w % STREAM_NUM];
        cudaMemcpyAsync(units.col1_dev + window_l, col1_h.data(), 
            unit_num_h * sizeof(int), cudaMemcpyHostToDevice, trans);
        cudaMemcpyAsync(units.col2_dev + window_l, col2_h.data(), 
            unit_num_h * sizeof(int), cudaMemcpyHostToDevice, trans);
        cudaMemcpyAsync(units.end_dev + window_l, end_h.data(), 
            unit_num_h * sizeof(int), cudaMemcpyHostToDevice, trans);
        cudaMemcpyAsync(units.vec_dev + window_l, vec_h.data(), 
            unit_num_h * sizeof(uint32_t), cudaMemcpyHostToDevice, trans);
        cudaMemcpyAsync(units.window_unit_num_dev + w, &unit_num_h, 
            sizeof(int), cudaMemcpyHostToDevice, trans);
    }
    for (int i = 0; i < STREAM_NUM; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    cudaDeviceSynchronize();
    return units_ptr;
}

__device__ inline bool units_match_check(const uint32_t& a, const uint32_t& b) { 
    static constexpr uint32_t MASK = 0xaaaaaaaa;
    const uint32_t c = a | b;
    return (((a & b) | ((c << 1) & c)) & MASK) == 0;
}

template<unsigned int Blocksize>
__global__ void match_units_simple_kernel(const int* window_offset, const int* unit_num,
    int* col1_dev, int* col2_dev, int* end_dev, uint32_t* vec_dev, int* match_num_dev, int* padded_block_num,
    int* col1_non, int* col2_non, int* end_non, uint32_t* vec_non 
) {
    const int window_id = blockIdx.x;
    const int window_size = unit_num[window_id];
    const int window_l = window_offset[window_id], window_r = window_l + window_size;

    if (window_size == 0) return;

    constexpr int TMP_SIZE = 512;
    constexpr int SLIDE_WINDOW_LEN = 512;
    __shared__ int col1_tmp[TMP_SIZE], col2_tmp[TMP_SIZE], end_tmp[TMP_SIZE];
    __shared__ uint32_t vec_tmp[TMP_SIZE];
    __shared__ int match_flag[Blocksize];
    __shared__ int match_num, non_match_num, last_match_num;
    __shared__ int sum_result;
    __shared__ bool break_iterate;
    int now_window_size = window_size;

    if (threadIdx.x == 0) {
        match_num = last_match_num = 0;
        break_iterate = false;
    }
    __syncthreads();

    for (int k = 0; k < Iterative_K; k++) {
        int k_ident = k & 1;
        if (threadIdx.x == 0) { non_match_num = 0; } 

        for (int from = 0, end = SLIDE_WINDOW_LEN; ; from += SLIDE_WINDOW_LEN, end += SLIDE_WINDOW_LEN) {
            int slide_size; 
            if (from + SLIDE_WINDOW_LEN <= now_window_size) {
                slide_size = SLIDE_WINDOW_LEN;
            } else {
                slide_size = now_window_size - from;
            }
            int padded_slide_size = slide_size + (slide_size & 1); 
            int pos1 = threadIdx.x * 2 + k_ident;
            int pos2 = (pos1 + 1) % padded_slide_size;
            int valid_num = 0;
            __syncthreads();

            if (pos1 < padded_slide_size) { 
                if (pos1 < slide_size) {
                    valid_num++;
                    col1_tmp[pos1] = col1_non[window_l + from + pos1];
                    col2_tmp[pos1] = col2_non[window_l + from + pos1];
                    end_tmp[pos1] = end_non[window_l + from + pos1];
                    vec_tmp[pos1] = vec_non[window_l + from + pos1];
                }
                if (pos2 < slide_size) {
                    valid_num++;
                    col1_tmp[pos2] = col1_non[window_l + from + pos2];
                    col2_tmp[pos2] = col2_non[window_l + from + pos2];
                    end_tmp[pos2] = end_non[window_l + from + pos2];
                    vec_tmp[pos2] = vec_non[window_l + from + pos2];
                }
            }
            __syncthreads();

            bool have_match = false;
            if (valid_num == 2 && units_match_check(vec_tmp[pos1], vec_tmp[pos2])) {

                match_flag[threadIdx.x] = (1 << 10); // match
                have_match = true;
            } else {
                match_flag[threadIdx.x] = 1; // non_match
            }
            __syncthreads();

            { 
                bool active = (threadIdx.x < 128); 
                int pos = threadIdx.x * 2 + 1;
                if (active) match_flag[pos] += match_flag[pos - 1];
                __syncthreads();

                for (int step = 1; step < Blocksize; step <<= 1) { 
                    if ((threadIdx.x & step) == 0) active = false;
                    if (active) {
                        match_flag[pos] += match_flag[pos - step * 2];
                    }
                    __syncthreads();
                }

                if (threadIdx.x == 0) { 
                    sum_result = match_flag[Blocksize - 1];
                    match_flag[Blocksize - 1] = 0;
                }
                __syncthreads();

                active = (threadIdx.x < 128);
                for (int step = Blocksize - 1; step; step >>= 1) {
                    if (active && ((threadIdx.x & step) == step)) {
                        int temp = match_flag[pos - step - 1];
                        match_flag[pos - step - 1] = match_flag[pos];
                        match_flag[pos] += temp;
                    }
                    __syncthreads();
                }

                int temp, temp2;
                if (active) {
                    temp = match_flag[pos - 1];
                    temp2 = match_flag[pos];
                }
                __syncthreads();
                if (active) {
                    if (threadIdx.x != 0) match_flag[pos - 2] = temp2;
                    match_flag[pos - 1] = temp + temp2;
                }
                
                if (threadIdx.x == 127) { 
                    match_flag[pos] = sum_result;
                }
                __syncthreads();
            }

            if (have_match) { 
                int rank = (match_flag[threadIdx.x] >> 10) - 1;
                // if (rank < 0) {
                //     printf("!! error 706\n");
                // }
                int target = window_l + match_num + rank * 2;
                col1_dev[target] = col1_tmp[pos1];
                col1_dev[target + 1] = col1_tmp[pos2];
                col2_dev[target] = col2_tmp[pos1];
                col2_dev[target + 1] = col2_tmp[pos2];
                end_dev[target] = end_tmp[pos1];
                end_dev[target + 1] = end_tmp[pos2];
                vec_dev[target] = vec_tmp[pos1];
                vec_dev[target + 1] = vec_tmp[pos2];
            } else if (pos1 < padded_slide_size) { 
                int rank = (match_flag[threadIdx.x] & 511) - 1;
                // if (rank < 0) {
                //     printf("!! error 720\n");
                // }
                int cur_match = (sum_result >> 10) * 2;
                int padded_slide_size_new = padded_slide_size - cur_match; 
                int pos1_new = rank * 2 + k_ident;
                int pos2_new = (pos1_new + 1) % padded_slide_size_new;
                if (pos1 < slide_size) {
                    col1_non[window_l + non_match_num + pos1_new] = col1_tmp[pos1];
                    col2_non[window_l + non_match_num + pos1_new] = col2_tmp[pos1];
                    end_non[window_l + non_match_num + pos1_new] = end_tmp[pos1];
                    vec_non[window_l + non_match_num + pos1_new] = vec_tmp[pos1];
                }
                if (pos2 < slide_size) {
                    col1_non[window_l + non_match_num + pos2_new] = col1_tmp[pos2];
                    col2_non[window_l + non_match_num + pos2_new] = col2_tmp[pos2];
                    end_non[window_l + non_match_num + pos2_new] = end_tmp[pos2];
                    vec_non[window_l + non_match_num + pos2_new] = vec_tmp[pos2];
                }
            }
            __syncthreads();

            if (threadIdx.x == 0) { 
                // {
                //     int total = (sum_result & 511) + (sum_result >> 10);
                //     if (total != Blocksize) {
                //         printf("!! error 745\n");
                //     }
                // }
                int match_delta = (sum_result >> 10) * 2;
                match_num += match_delta;
                non_match_num += slide_size - match_delta;
            }
            __syncthreads();

            if (end >= now_window_size) {
                // if (threadIdx.x == 0) {
                //     if (end != 512) {
                //         printf("!! error 757\n");
                //     }
                // }

                break;
            }
        }

        now_window_size = non_match_num; 
        if (threadIdx.x == 0) { 
            if (match_num - last_match_num < last_match_num * Iterative_factor) {
                break_iterate = true;
            } else {
                last_match_num = match_num;
            }
        }
        __syncthreads();

        if (break_iterate) { 
            break;
        }
    }


    for (int pos = threadIdx.x, end = Blocksize; ; pos += Blocksize, end += Blocksize) {
        if (pos < now_window_size) { 
            col1_dev[window_r - 1 - pos] = col1_non[window_l + pos];
            col2_dev[window_r - 1 - pos] = col2_non[window_l + pos];
            end_dev[window_r - 1 - pos] = end_non[window_l + pos];
            vec_dev[window_r - 1 - pos] = vec_non[window_l + pos];
        }
        if (end >= now_window_size) break;
    }
    __syncthreads();

    if (threadIdx.x == 0) { // assert check
        // if (now_window_size + match_num != window_size) {
        //     printf("!! error 774, now_window_size = %d, match_num = %d, window_size = %d\n", now_window_size, match_num, window_size);
        // }

        match_num_dev[window_id] = match_num;
        int column_num = match_num * 2 + now_window_size * 4;
        if (column_num % TC_col_len != 0) {
            column_num += TC_col_len - (column_num % TC_col_len);
        }
        padded_block_num[window_id + 1] = column_num / 4; 
    }
    __syncthreads();
}

MatchResult* match_units(SparseMatrix* mtx_ptr, UnitsPack* units_ptr) {
    SparseMatrix& mtx = *mtx_ptr;
    UnitsPack& units = *units_ptr;
    MatchResult* match_ptr = (MatchResult*)malloc(sizeof(MatchResult));
    MatchResult& match = *match_ptr;
    cudaMalloc(&match.col1_dev, mtx.nonzeros * sizeof(int));
    cudaMalloc(&match.col2_dev, mtx.nonzeros * sizeof(int));
    cudaMalloc(&match.end_dev, mtx.nonzeros * sizeof(int));
    cudaMalloc(&match.vec_dev, mtx.nonzeros * sizeof(uint32_t));
    cudaMalloc(&match.match_num_dev, (mtx.tc_rownum + 1) * sizeof(int));
    cudaMalloc(&match.padded_block_num, (mtx.tc_rownum + 1) * sizeof(int));
    {
        const int zero = 0;
        cudaMemcpy(match.padded_block_num, &zero, sizeof(int), cudaMemcpyHostToDevice); 
    }
    
    match_units_simple_kernel<256><<<mtx.tc_rownum, 256>>>(
        mtx.window_offset_dev, units.window_unit_num_dev,
        match.col1_dev, match.col2_dev, match.end_dev, match.vec_dev, match.match_num_dev, match.padded_block_num,
        units.col1_dev, units.col2_dev, units.end_dev, units.vec_dev
    );

    cudaDeviceSynchronize();
    return match_ptr;
}

template<unsigned int Blocksize>
__global__ void padding_kernel(
    const NonzeroElement* nz_list, const int* window_offset, const int* window_unit_num,
    const int* col1_dev, const int* col2_dev, const int* end_dev, const uint32_t* vec_dev, const int* window_match_num, const int* block_offset,
    double* opd_dev, int* metadata_dev, int* old_col_dev
) {
    const int window_id = blockIdx.x;
    const int unit_num = window_unit_num[window_id];
    const int window_l = window_offset[window_id], window_r = window_l + unit_num;

    const int match_num = window_match_num[window_id];
    const int non_match_num = unit_num - match_num;

    const int block_l = block_offset[window_id], block_r = block_offset[window_id + 1];
  
    const int nz_block_num = match_num / 2 + non_match_num;
    const int padded_block_num = block_r - block_l - nz_block_num; 

    const int array_l = block_l * 32; //, array_r = block_r * 32; 

    auto tile = cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());
    const int tile_id   = threadIdx.x / 32; 
    const int tile_rank = threadIdx.x % 32; 
    const int line_id   = tile_rank / 2;  
    const int line_rank = tile_rank % 2;   

    __shared__ double buf[64 * (Blocksize / 32)];
    #define buf_idx(i, j) ((tile_id * 64) + ((i) * 4) + (j))


    for (int match_pos = tile_id * 2, block_pos = tile_id; match_pos < match_num; match_pos += 16, block_pos += 8) {
        buf[buf_idx(line_id, line_rank)] = 0; 
        buf[buf_idx(line_id, line_rank + 2)] = 0;
        tile.sync();

        uint32_t vec = 0; 
        for (int unit_rank = 0; unit_rank < 2; unit_rank++) {
            int col1 = col1_dev[window_l + match_pos + unit_rank];
            int col2 = col2_dev[window_l + match_pos + unit_rank];
            int end  = end_dev[window_l + match_pos + unit_rank];
            vec += vec_dev[window_l + match_pos + unit_rank];
            
            int pos = col1 + tile_rank;
            if (pos < end) {
                int buf_col = (unit_rank * 2) + ((col2 != -1) && (pos >= col2));
                int buf_row = nz_list[pos].row;
                double buf_val = nz_list[pos].val;
                buf[buf_idx(buf_row, buf_col)] = buf_val;
            }
            if (tile_rank == 0) { 
                old_col_dev[(block_l + block_pos) * 4 + unit_rank * 2] = nz_list[col1].col;
                old_col_dev[(block_l + block_pos) * 4 + unit_rank * 2 + 1] = ((col2 == -1) ? -1 : nz_list[col2].col);
            }

            tile.sync();
        }

        int line_nz_num = (vec >> (line_id * 2)) & 3; 
        int array_pos = array_l + block_pos * 32 + tile_rank;
        if (line_nz_num == 0) { 
            opd_dev[array_pos] = 0; // nan
            metadata_dev[array_pos] = 2 + line_rank;
        }
        else if (line_nz_num == 1) {
            if (line_rank == 0) {
                int tmp_pos;
                double tmp_val = 0;
                for (int i = 0; i < 3; i++) {
                    if (buf[buf_idx(line_id, i)] != 0) {
                        tmp_pos = i;
                        tmp_val = buf[buf_idx(line_id, i)];
                        break;
                    }
                }
                if (tmp_val == 0) {
                    opd_dev[array_pos] = 0; // nan
                    metadata_dev[array_pos] = 2;
                } else {
                    opd_dev[array_pos] = tmp_val;
                    metadata_dev[array_pos] = tmp_pos;
                }
            } else {
                opd_dev[array_pos] = buf[buf_idx(line_id, 3)]; // nz or NAN
                metadata_dev[array_pos] = 3;
            }
        }
        else { 
            int check_col = line_rank * 3; // 0->0, 1->3
            const int delta = 1 - line_rank * 2; // 0->1, 1->-1
            for (int i = 0; i < 2; i++) {
                if (buf[buf_idx(line_id, check_col)] != 0) {
                    opd_dev[array_pos] = buf[buf_idx(line_id, check_col)];
                    metadata_dev[array_pos] = check_col;
                    array_pos += delta;
                }
                check_col += delta;
            }
        }

        tile.sync(); 
    }

    int match_block_num = match_num / 2;
    for (int non_match_pos = tile_id; non_match_pos < non_match_num; non_match_pos += 8) {
        int block_pos = non_match_pos;
        buf[buf_idx(line_id, line_rank)] = 0;
        tile.sync();

        int col1 = col1_dev[window_r - 1 - non_match_pos];
        int col2 = col2_dev[window_r - 1 - non_match_pos];
        int end  = end_dev[window_r - 1 - non_match_pos];
        uint32_t vec = vec_dev[window_r - 1 - non_match_pos];

        {
            int pos = col1 + tile_rank;
            if (pos < end) {
                int buf_col = ((col2 != -1) && (pos >= col2));
                int buf_row = nz_list[pos].row;
                double buf_val = nz_list[pos].val;
                buf[buf_idx(buf_row, buf_col)] = buf_val;
            }
            if (tile_rank == 0) {
                old_col_dev[(block_l + match_block_num + block_pos) * 4] = nz_list[col1].col;
                old_col_dev[(block_l + match_block_num + block_pos) * 4 + 1] = ((col2 == -1) ? -1 : nz_list[col2].col);
                old_col_dev[(block_l + match_block_num + block_pos) * 4 + 2] = -1;
                old_col_dev[(block_l + match_block_num + block_pos) * 4 + 3] = -1;
            }
        }
        tile.sync();
        
        int line_nz_num = (vec >> (line_id * 2)) & 3;
        int array_pos = array_l + (match_block_num + block_pos) * 32 + tile_rank;
        if (line_nz_num == 0) {
            opd_dev[array_pos] = 0; // nan
            metadata_dev[array_pos] = 2 + line_rank;
        }
        else if (line_nz_num == 1) {
            if (line_rank == 0) {
                int tmp_pos = (buf[buf_idx(line_id, 0)] == 0);
                double tmp_val = buf[buf_idx(line_id, tmp_pos)];
                opd_dev[array_pos] = tmp_val;
                metadata_dev[array_pos] = tmp_pos;
            } else {
                opd_dev[array_pos] = 0; // nan
                metadata_dev[array_pos] = 3;
            }
        }
        else {
            opd_dev[array_pos] = buf[buf_idx(line_id, line_rank)];
            metadata_dev[array_pos] = line_rank;
        }

        tile.sync();
    }


    for (int padding_pos = tile_id; padding_pos < padded_block_num; padding_pos += 8) {
        int array_pos = array_l + (match_block_num + non_match_num + padding_pos) * 32 + tile_rank;
        opd_dev[array_pos] = 0; // nan
        metadata_dev[array_pos] = 2 + line_rank;
        if (tile_rank < 4) {
            old_col_dev[(block_l + match_block_num + non_match_num + padding_pos) * 4 + tile_rank] = -1;
        }

        tile.sync();
    }

    __syncthreads();
}

PaddedResult* padding(SparseMatrix* mtx_ptr, UnitsPack* units_ptr, MatchResult* match_ptr, int& padded_col_num) {
    SparseMatrix& mtx = *mtx_ptr;
    UnitsPack& units = *units_ptr;
    MatchResult& match = *match_ptr;
    thrust::inclusive_scan(thrust::device, match.padded_block_num, match.padded_block_num + mtx.tc_rownum + 1, match.padded_block_num);
    int block_num;
    cudaMemcpy(&block_num, &match.padded_block_num[mtx.tc_rownum], sizeof(int), cudaMemcpyDeviceToHost); 
    padded_col_num = block_num * 4;
    const int nz_len = block_num * 32; 

    PaddedResult* padded_ptr = (PaddedResult*)malloc(sizeof(PaddedResult));
    PaddedResult& padded = *padded_ptr;
    cudaMalloc(&padded.opdA_dev, nz_len * sizeof(double));
    cudaMalloc(&padded.metadata_dev, nz_len * sizeof(int));
    cudaMalloc(&padded.old_col_dev, padded_col_num * sizeof(int));

    padding_kernel<256><<<mtx.tc_rownum, 256>>>(
        mtx.nz_list_dev, mtx.window_offset_dev, units.window_unit_num_dev,
        match.col1_dev, match.col2_dev, match.end_dev, match.vec_dev, match.match_num_dev, match.padded_block_num,
        padded.opdA_dev, padded.metadata_dev, padded.old_col_dev
    );

    cudaDeviceSynchronize();
    return padded_ptr;
}

MetadataResult* unwrap_metadata(SparseMatrix* mtx_ptr, MatchResult* match_ptr, 
    PaddedResult* padded_ptr, const int& padded_col_num) {
    SparseMatrix& mtx = *mtx_ptr;
    MatchResult& match = *match_ptr;
    PaddedResult& padded = *padded_ptr;

    MetadataResult* result_ptr = (MetadataResult*)malloc(sizeof(MetadataResult));
    MetadataResult& result = *result_ptr;
    assert(padded_col_num % TC_col_len == 0);
    const int nz_len = padded_col_num * 8;
    
    result.window_offset = (int*)malloc((mtx.tc_rownum + 1) * sizeof(int));
    result.opdA = (double*)malloc(nz_len * sizeof(double));
    result.metadata = (int*)malloc(nz_len * sizeof(int));
    result.old_col = (int*)malloc(padded_col_num * sizeof(int));

    cudaStream_t copy_stream[4];
    for (int i = 0; i < 4; i++) {
        cudaStreamCreate(&copy_stream[i]);
    }
    cudaMemcpyAsync(result.window_offset, match.padded_block_num, (mtx.tc_rownum + 1) * sizeof(int), 
        cudaMemcpyDeviceToHost, copy_stream[0]);
    cudaMemcpyAsync(result.opdA, padded.opdA_dev, nz_len * sizeof(double),
        cudaMemcpyDeviceToHost, copy_stream[1]);
    cudaMemcpyAsync(result.metadata, padded.metadata_dev, nz_len * sizeof(int),
        cudaMemcpyDeviceToHost, copy_stream[2]);
    cudaMemcpyAsync(result.old_col, padded.old_col_dev, padded_col_num * sizeof(int),
        cudaMemcpyDeviceToHost, copy_stream[3]);

    cudaStreamSynchronize(copy_stream[1]); // opdA
    cudaStreamSynchronize(copy_stream[2]); // metadata
    constexpr int TC_width = TC_col_len / 4;
    constexpr int nz_per_TC = TC_width * 32;
    assert(nz_len % nz_per_TC == 0);
    {
        double opd_tmp[nz_per_TC];
        int metadata_tmp[nz_per_TC];

        for (int from = 0; from < nz_len; from += nz_per_TC) {
            for (int block_from = 0; block_from < nz_per_TC; block_from += 32) {
                for (int i = 0; i < TC_row_len; i++) { // 2:4 assume
                    int pos = from + block_from + i * 2;
                    int target = (i * 8) + (block_from / 32) * 2;
                    opd_tmp[target] = result.opdA[pos];
                    opd_tmp[target + 1] = result.opdA[pos + 1];
                    metadata_tmp[target] = result.metadata[pos];
                    metadata_tmp[target + 1] = result.metadata[pos + 1];
                }
            }
            std::memcpy(result.opdA + from, opd_tmp, nz_per_TC * sizeof(double));
            std::memcpy(result.metadata + from, metadata_tmp, nz_per_TC * sizeof(int));
        }
    }

    cudaStreamSynchronize(copy_stream[0]); // window_offset
    for (int w = 0; w <= mtx.tc_rownum; w++) {
        result.window_offset[w] /= TC_width;
    }

    cudaStreamSynchronize(copy_stream[3]); // old_col

    return result_ptr;

}

} // namespace Matching_And_Padding
void create_directory(const std::string& path) {
    std::filesystem::create_directories(path);
}

void write_preprocessed_data(const char* filename,
    int original_M, int original_K, 
    int nonzeros,
    int* window_offset, int window_offset_len,
    double* opdA, int opdA_len,
    int* metadata, int metadata_len,
    int* old_col, int old_col_len) 
{
    FILE* fp = fopen(filename, "wb");
    if (!fp) {
    std::cerr << "Error creating output file: " << filename << std::endl;
    return;
    }

    fwrite(&original_M, sizeof(int), 1, fp);
    fwrite(&original_K, sizeof(int), 1, fp);

    fwrite(&nonzeros, sizeof(int), 1, fp);


    fwrite(&window_offset_len, sizeof(int), 1, fp);
    fwrite(window_offset, sizeof(int), window_offset_len, fp);

    fwrite(&opdA_len, sizeof(int), 1, fp);
    fwrite(opdA, sizeof(double), opdA_len, fp);

    fwrite(&metadata_len, sizeof(int), 1, fp);
    fwrite(metadata, sizeof(int), metadata_len, fp);

    fwrite(&old_col_len, sizeof(int), 1, fp);
    fwrite(old_col, sizeof(int), old_col_len, fp);

    fclose(fp);
}



int main(int argc, const char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <matrix_name>" << std::endl; 
        std::cerr << "Example: " << argv[0] << " TF16" << std::endl;
        return -1;
    }
    cudaFree(0); 

    using namespace Matching_And_Padding;
    TIME_INIT;
    double match_time = 0.0, padding_time = 0.0;

    int original_tc_blocks_quantity; 
    TIME(start);
    auto mtx = read_matrix(argv[1], original_tc_blocks_quantity);
    TIME(end);
    std::cerr << "# read_matrix \t\t\t[" << TIME_COUNT << " ms]" << std::endl;

    int compressed_tc_blocks_quantity; 
    TIME(start);
    auto units_per_window = Matching_And_Padding::pack_columns_into_units(mtx, compressed_tc_blocks_quantity);
    TIME(end);
    std::cerr << "# pack_columns_into_units \t[" << TIME_COUNT << " ms]" << std::endl;

    TIME(start);
    auto match_result = Matching_And_Padding::match_units(mtx, units_per_window);
    TIME(end);
    std::cerr << "# match_units \t\t\t[" << TIME_COUNT << " ms]" << std::endl;
    match_time = TIME_COUNT;

    int padded_col_num; 
    TIME(start);
    auto padded_list = Matching_And_Padding::padding(mtx, units_per_window, match_result, padded_col_num);
    TIME(end);
    std::cerr << "# padding \t\t\t[" << TIME_COUNT << " ms]" << std::endl;
    padding_time = TIME_COUNT;

    TIME(start);
    auto unwrapped_result = Matching_And_Padding::unwrap_metadata(mtx, match_result, padded_list, padded_col_num);
    TIME(end);
    std::cerr << "# unwrap_metadata \t\t[" << TIME_COUNT << " ms]" << std::endl;

// #define PRINT_MAX_UNIT_NUM
#ifdef PRINT_MAX_UNIT_NUM
    {
        int *unit_num_h = (int*)malloc((mtx->tc_rownum + 1) * sizeof(int));
        cudaMemcpy(unit_num_h, units_per_window->window_unit_num_dev, (mtx->tc_rownum + 1) * sizeof(int), cudaMemcpyDeviceToHost);
        int max_unit_num = 0;
        for (int w = 0; w < mtx->tc_rownum; w++) {
            max_unit_num = std::max(max_unit_num, unit_num_h[w]);
        }
        std::cerr << "! max_unit_num: " << max_unit_num << std::endl;
    }
#endif

// #define PRINT_UNITS
#ifdef PRINT_UNITS
    {
        int *col1_h, *col2_h, *end_h, *unit_num_h;
        uint32_t* vec_h;
        col1_h = (int*)malloc(mtx->nonzeros * sizeof(int));
        col2_h = (int*)malloc(mtx->nonzeros * sizeof(int));
        end_h  = (int*)malloc(mtx->nonzeros * sizeof(int));
        unit_num_h = (int*)malloc((mtx->tc_rownum + 1) * sizeof(int));
        vec_h = (uint32_t*)malloc(mtx->nonzeros * sizeof(uint32_t));

        cudaMemcpy(col1_h, units_per_window->col1_dev, mtx->nonzeros * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(col2_h, units_per_window->col2_dev, mtx->nonzeros * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(end_h, units_per_window->end_dev, mtx->nonzeros * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(unit_num_h, units_per_window->window_unit_num_dev, (mtx->tc_rownum + 1) * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(vec_h, units_per_window->vec_dev, mtx->nonzeros * sizeof(uint32_t), cudaMemcpyDeviceToHost);

        for (int w = 0; w < std::min(mtx->tc_rownum, 10); w++) {
            std::cerr << "window " << w << ": ";
            std::cerr << "unit_num = " << unit_num_h[w] << ", units = ";
            for (int i = 0; i < std::min(unit_num_h[w], 10); i++) {
                std::cerr << "[" << col1_h[mtx->window_offset[w] + i] << " " << col2_h[mtx->window_offset[w] + i] << " " << end_h[mtx->window_offset[w] + i] << " | " << vec_h[mtx->window_offset[w] + i] << "] ";
            }
            std::cerr << std::endl;
        }
    }
#endif

// #define PRINT_PADDING
#ifdef PRINT_PADDING
    {
        std::cerr << "padded_col_num = " << padded_col_num << std::endl;

        const int nz_len = padded_col_num * 8;
        double* opd_h = (double*)malloc(nz_len * sizeof(double));
        int* metadata_h = (int*)malloc(nz_len * sizeof(int));
        cudaMemcpy(opd_h, padded_list->opdA_dev, nz_len * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(metadata_h, padded_list->metadata_dev, nz_len * sizeof(int), cudaMemcpyDeviceToHost);

        std::cerr << "opdA: [ ";
        for (int i = 0; i < std::min(100, padded_col_num / 2 * Matching_And_Padding::TC_row_len); i++) {
            double val = opd_h[i];
            if (val == 0) std::cerr << "nan ";
            else std::cerr << std::fixed << std::setprecision(1) << val << " ";
        }
        std::cerr << "]" << std::endl;
    
        std::cerr << "metadata: [ ";
        for (int i = 0; i < std::min(100, padded_col_num / 2 * Matching_And_Padding::TC_row_len); i++) {
            std::cerr << metadata_h[i] << " ";
        }
        std::cerr << "]" << std::endl;
    }
#endif

// #define PRINT_PADDED
#ifdef PRINT_PADDED
    std::cerr << "window_offset: [ ";
    for (int i = 0; i < std::min(100, mtx->tc_rownum + 1); i++) {
        std::cerr << unwrapped_result->window_offset[i] << " ";
    }
    std::cerr << "]" << std::endl;

    std::cerr << "old_col: [ ";
    for (int i = 0; i < std::min(100, padded_col_num); i++) {
        std::cerr << unwrapped_result->old_col[i] << " ";
    }
    std::cerr << "]" << std::endl;
#endif

//#define PRINT_ARRAY
#ifdef PRINT_ARRAY
    std::cerr << "window_offset: [ ";
    for (int i = 0; i < std::min(100, mtx->tc_rownum + 1); i++) {
        std::cerr << unwrapped_result->window_offset[i] << " ";
    }
    std::cerr << "]" << std::endl;

    std::cerr << "opdA: [ ";
    for (int i = 0; i < std::min(100, padded_col_num / 2 * Matching_And_Padding::TC_row_len); i++) {
        if (unwrapped_result->opdA[i] == 0) std::cerr << "nan ";
        else std::cerr << std::fixed << std::setprecision(1) << unwrapped_result->opdA[i] << " ";
    }
    std::cerr << "]" << std::endl;

    std::cerr << "metadata: [ ";
    for (int i = 0; i < std::min(100, padded_col_num / 2 * Matching_And_Padding::TC_row_len); i++) {
        std::cerr << unwrapped_result->metadata[i] << " ";
    }
    std::cerr << "]" << std::endl;

    std::cerr << "old_col: [ ";
    for (int i = 0; i < std::min(100, padded_col_num); i++) {
        std::cerr << unwrapped_result->old_col[i] << " ";
    }
    std::cerr << "]" << std::endl;
#endif

// #define PRINT_MATCH
#ifdef PRINT_MATCH
    {
        int *col1_h = (int*)malloc(mtx->nonzeros * sizeof(int));
        int *col2_h = (int*)malloc(mtx->nonzeros * sizeof(int));
        int *end_h = (int*)malloc(mtx->nonzeros * sizeof(int));
        int *unit_num_h = (int*)malloc((mtx->tc_rownum + 1) * sizeof(int));
        int *match_num_h = (int*)malloc((mtx->tc_rownum + 1) * sizeof(int));
        cudaMemcpy(col1_h, match_result->col1_dev, mtx->nonzeros * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(col2_h, match_result->col2_dev, mtx->nonzeros * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(end_h, match_result->end_dev, mtx->nonzeros * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(unit_num_h, units_per_window->window_unit_num_dev, (mtx->tc_rownum + 1) * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(match_num_h, match_result->match_num_dev, (mtx->tc_rownum + 1) * sizeof(int), cudaMemcpyDeviceToHost);

        std::cerr << "Matched & Non-matched Units:" << std::endl;
        for (int i = 0; i < std::min(5, mtx->tc_rownum); i++) {
            const int unit_num = unit_num_h[i];
            const int match_num = match_num_h[i];
            const int non_match_num = unit_num - match_num;
            const int window_l = mtx->window_offset[i];
            const int window_r = window_l + unit_num;

            std::cerr << "  window " << i << ":" << std::endl;
            std::cerr << "  unit_num: " << unit_num << std::endl;
            std::cerr << "    Match: ";
            for (int t = 0; t < std::min(20, match_num); t += 2) {
                std::cerr << "(" << mtx->nz_list[col1_h[t + window_l]].col << ", " << mtx->nz_list[col1_h[t + 1 + window_l]].col << ") ";
            }
            std::cerr << std::endl;
    
            std::cerr << "    Non-match: ";
            for (int t = 0; t < std::min(10, non_match_num); t++) {
                std::cerr << mtx->nz_list[col1_h[window_r - 1 - t]].col << " ";
            }
            std::cerr << std::endl;
        }
    }
#endif

const char* base_path = get_config_value("dataset_path");
char data_bin_path[512];
const char *input = argv[1];
const char *last_slash = strrchr(input, '/');
const char *base_name = last_slash ? (last_slash + 1) : input;

snprintf(data_bin_path, sizeof(data_bin_path), 
    "%s/dataset_mp_processed/2-4-cu/0.Adjacent-matching/%s_data.bin", 
    base_path, base_name);


char output_dir[512];
snprintf(output_dir, sizeof(output_dir), "%s/dataset_mp_processed/2-4-cu/0.Adjacent-matching", base_path);
create_directory(output_dir);

write_preprocessed_data(data_bin_path,
    mtx->rows,mtx->cols, 
    mtx->nonzeros,
    unwrapped_result->window_offset, mtx->tc_rownum + 1,
    unwrapped_result->opdA, padded_col_num / 2 * Matching_And_Padding::TC_row_len,
    unwrapped_result->metadata, padded_col_num / 2 * Matching_And_Padding::TC_row_len,
    unwrapped_result->old_col, padded_col_num);


const char* csv_filename = "../../result/time_and_tb_num/time_adjacent-2-4-cuda.csv";
std::ofstream csv_file(csv_filename, std::ios::app);
if (csv_file.is_open()) {

    csv_file.seekp(0, std::ios::end);
    bool write_header = (csv_file.tellp() == 0);
    
    if (write_header) {
        csv_file << "dataSet,match_time(ms),padding_time(ms),total_time(ms)\n";
    }

    csv_file << base_name << ","
             << std::fixed << std::setprecision(3) << match_time << ","
             << std::fixed << std::setprecision(3) << padding_time << ","
             << std::fixed << std::setprecision(3) << (match_time + padding_time)
             << "\n";
    
    csv_file.close();
} else {
    std::cerr << "Error: Failed to open time result file: " << csv_filename << std::endl;
}


    return 0;
}
