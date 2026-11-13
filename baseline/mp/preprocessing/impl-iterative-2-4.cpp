#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <cstring>
#include <unordered_set>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <fstream>
#include "libmmio/mmio.h"
#include <filesystem> 



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
/*******************************************
 *            Iterative Matching           *
 *******************************************/

#define COUNT_ORIGINAL_TC_BLOCKS_QUANTITY
// #define TEST_DATA_PATH
// #define TESTMODE
#ifdef TESTMODE
    #define TESTDATA(a) a
#else
    #define TESTDATA(a)
#endif

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

// 兼容 <name>/<name>.mtx 布局: mtx_name 形如 ddi 时先找 datasets/ddi/ddi.mtx
const char* resolve_mtx_path(const char* mtx_name) {
    static char resolved_path[512];
    snprintf(resolved_path, sizeof(resolved_path), "%s/%s.mtx", dataset_dir, mtx_name);
    if (std::filesystem::exists(resolved_path)) return resolved_path;
    snprintf(resolved_path, sizeof(resolved_path), "%s/%s/%s.mtx", dataset_dir, mtx_name, mtx_name);
    return resolved_path;
}

char mtx_path[512];

// TC block size
constexpr int TC_row_len = 16; 
constexpr int TC_col_len = 16;


struct NonzeroElement{ int row, col; double val; }; 

struct SparseMatrix {
    int rows, cols, nonzeros;
    int tc_rownum, tc_colnum;
    std::vector<NonzeroElement>* windows;   
    NonzeroElement* nz_list;                
    int* window_offset;                     
    MM_typecode matcode;                   
    bool is_symmetric; 
};


struct Unit{ NonzeroElement *col1, *col2, *end; uint32_t vec; };
struct UnitsPack{
    std::vector<Unit> unit_list; 
    int* window_offset; 
};

struct MatchResult {
    Unit* unit_list;
    int *match_offset; 
};

struct UnitNode{ Unit* data; UnitNode* next; };
struct Padded{
    double opd[2 * TC_row_len];
    int metadata[2 * TC_row_len];
};

struct PaddedResult{
    Padded* padded_list;
    int* window_offset;
};

struct MetadataResult{
    int* window_offset;
    double* opdA;
    int *metadata, *old_col;
};

int __TESTDATA_max_elements_per_window;

std::unordered_set<unsigned long long> Original_TC_block_occupy;

#define TIME_INIT std::chrono::time_point<std::chrono::high_resolution_clock> time_start, time_end
#define TIME(flag) time_##flag = std::chrono::high_resolution_clock::now()
//#define TIME_COUNT std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count()
#define TIME_COUNT (std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start).count() / 1000.0)

SparseMatrix* read_matrix(const char* mtx_name, int& nonzero_block) {
    snprintf(mtx_path, sizeof(mtx_path), "%s", resolve_mtx_path(mtx_name));
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
         mtx.windows[tc_row].push_back((NonzeroElement){local_row, col, val});
         total_elements++;

         if (mtx.is_symmetric && row != col) {
             tc_row = col / TC_row_len;  
             local_row = col % TC_row_len;
             mtx.windows[tc_row].push_back((NonzeroElement){local_row, row, val});
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
        std::memcpy(list_p, mtx.windows[i].data(), sizeof(NonzeroElement) * size_cnt);
        list_p += size_cnt;
        nz_cnt += size_cnt;
    }
    mtx.window_offset[mtx.tc_rownum] = nz_cnt;
    assert(nz_cnt == mtx.nonzeros);
    assert(list_p == mtx.nz_list + mtx.nonzeros);

    fclose(fp);
    return mtx_ptr;
}

UnitsPack* pack_columns_into_units(SparseMatrix* mtx_ptr, int& tc_blocks) {
    UnitsPack* pack_ptr = new UnitsPack;
    UnitsPack& pack = *pack_ptr;
    SparseMatrix& mtx = *mtx_ptr;
    pack.window_offset = (int*)malloc(sizeof(int) * (mtx.tc_rownum + 1));
    NonzeroElement* list = mtx.nz_list;

    tc_blocks = 0;
    for (int i = 0; i < mtx.tc_rownum; i++) { 
        pack.window_offset[i] = pack.unit_list.size();
        int window_l = mtx.window_offset[i];
        int window_r = mtx.window_offset[i + 1];
        std::sort(list + window_l, list + window_r, 
            [](const NonzeroElement& a, const NonzeroElement& b) -> bool {
                return a.col != b.col ? a.col < b.col : a.row < b.row;
            }
        ); 

        TESTDATA(__TESTDATA_max_elements_per_window = std::max(__TESTDATA_max_elements_per_window, window_r - window_l));

        uint32_t vec = 0;
        NonzeroElement* cols[2] = {NULL, NULL};
        int colnum = 0;

        for (int l = window_l, r; l < window_r; l = r, colnum++) {
            for (r = l; r < window_r && list[r].col == list[l].col; r++) {
                // [fix] 同一个 (col,row) 出现多次时: vec 只能计一个位置, 而 padding
                //   用 block[.][row] = val 顺序覆盖赋值. 因此把数值累加到最后一个
                //   重复条目 (赋值顺序上最后写), 既与 vec 的单次计数一致, 又等价于
                //   scipy/torch 的 sum_duplicates 语义.
                if (r > l && list[r].row == list[r - 1].row) {
                    list[r].val += list[r - 1].val;
                    continue;
                }
                vec += (uint32_t)1 << (2 * list[r].row);
            }
            cols[colnum & 1] = list + l;
            if (colnum & 1) {
                pack.unit_list.push_back((Unit){cols[0], cols[1], list + r, vec});
                vec = 0, cols[0] = cols[1] = NULL;
            }
        }
        if (colnum & 1) { 
            pack.unit_list.push_back((Unit){cols[0], NULL, list + window_r, vec});
            colnum++;
        }

        tc_blocks += (colnum + TC_col_len - 1) / TC_col_len;
    }
    pack.window_offset[mtx.tc_rownum] = pack.unit_list.size();

    return pack_ptr;
}

inline bool units_match_check(const uint32_t& a, const uint32_t& b) { 
    static constexpr uint32_t MASK = 0xaaaaaaaa;
    const uint32_t c = a | b;
    return (((a & b) | ((c << 1) & c)) & MASK) == 0;
}

MatchResult* match_units(UnitsPack* units_ptr, const int& window_num, int& units_pack_num) {
    UnitsPack& units = *units_ptr;
    MatchResult* result_ptr = (MatchResult*)malloc(sizeof(MatchResult));
    MatchResult& result = *result_ptr;

    int unit_num = units_pack_num = units.unit_list.size();
    result.unit_list = (Unit*)malloc(sizeof(Unit) * unit_num);
    result.match_offset = (int*)malloc(sizeof(int) * window_num);

    for (int w = 0; w < window_num; w++) {
        const int l = units.window_offset[w];
        const int r = units.window_offset[w + 1];
        
        // Handle empty window case
        if (l == r) {
            result.match_offset[w] = 0;
            continue;
        }
        
        // Copy all units to a temporary array for processing
        std::vector<Unit> window_units(units.unit_list.begin() + l, units.unit_list.begin() + r);
        int units_in_window = r - l;
        
        // Create a vector to track which units have been matched
        std::vector<bool> matched(units_in_window, false);
        
        // Vector to store matched pairs (indices)
        std::vector<std::pair<int, int>> matched_pairs;
        
        // Find all matching pairs
        for (int i = 0; i < units_in_window; i++) {
            if (matched[i]) continue; // Skip already matched units
            
            for (int j = i + 1; j < units_in_window; j++) {
                if (matched[j]) continue; // Skip already matched units
                
                if (units_match_check(window_units[i].vec, window_units[j].vec)) {
                    // Found a match - record indices
                    matched_pairs.push_back({i, j});
                    matched[i] = matched[j] = true;
                    break;
                }
            }
        }
        
        // Now we have all matched pairs, copy them first
        Unit* dest_ptr = result.unit_list + l;
        for (const auto& pair : matched_pairs) {
            *dest_ptr++ = window_units[pair.first];
            *dest_ptr++ = window_units[pair.second];
        }
        
        // Store how many units were matched
        result.match_offset[w] = matched_pairs.size() * 2;
        
        // Now copy all unmatched units
        for (int i = 0; i < units_in_window; i++) {
            if (!matched[i]) {
                *dest_ptr++ = window_units[i];
            }
        }
        
        // Verify we've processed exactly the right number of elements
        assert(dest_ptr == result.unit_list + r);
        
        // Update total units count for padding
        units_pack_num -= matched_pairs.size();
    }

    return result_ptr;
}

// bool __TEST_PADDING_FLAG;

PaddedResult* padding(UnitsPack* units_ptr, MatchResult* match_ptr, 
    const int& window_num, const int& units_pack_num, int& padded_col_num) {
    UnitsPack& units   = *units_ptr;
    MatchResult& match = *match_ptr;
    PaddedResult* result_ptr = new PaddedResult;
    PaddedResult& result = *result_ptr;
    result.padded_list = (Padded*)malloc(sizeof(Padded) * units_pack_num);
    result.window_offset = (int*)malloc(sizeof(int) * (window_num + 1));
    
    Padded* pad_ptr = result.padded_list;
    double block[4][TC_row_len]; // 4x16 block

    padded_col_num = 0;
    for (int w = 0; w < window_num; w++) {
        const int l = units.window_offset[w], r = units.window_offset[w + 1];
        const int match_num = match.match_offset[w];
        const int non_match_num = r - l - match_num;
        Unit* ptr = match.unit_list + l;
        result.window_offset[w] = pad_ptr - result.padded_list;

        for (int i = 0; i < match_num; i += 2, pad_ptr++) {
            uint32_t vec = ptr->vec + (ptr + 1)->vec;

            std::memset(block, 0, sizeof(double) * 4 * TC_row_len);
 
            if (ptr->col2 != NULL) {
                for (auto p0 = ptr->col1; p0 != ptr->col2; p0++)
                    block[0][p0->row] = p0->val;
                for (auto p1 = ptr->col2; p1 != ptr->end; p1++)
                    block[1][p1->row] = p1->val;
            } else {
                for (auto p0 = ptr->col1; p0 != ptr->end; p0++)
                    block[0][p0->row] = p0->val;
            }
            ptr++;
            if (ptr->col2 != NULL) {
                for (auto p2 = ptr->col1; p2 != ptr->col2; p2++)
                    block[2][p2->row] = p2->val;
                for (auto p3 = ptr->col2; p3 != ptr->end; p3++)
                    block[3][p3->row] = p3->val;
            } else {
                for (auto p2 = ptr->col1; p2 != ptr->end; p2++)
                    block[2][p2->row] = p2->val;
            }
            ptr++;

            double* opd_ptr = pad_ptr->opd;
            int* meta_ptr = pad_ptr->metadata;

            for (int k = 0; k < TC_row_len; k++) {
                int nz_cnt = vec & 3; 
                vec >>= 2;

                if (nz_cnt == 0) { 
                    *opd_ptr = NAN;
                    *(opd_ptr + 1) = NAN;
                    opd_ptr += 2;

                    *meta_ptr = 2;
                    *(meta_ptr + 1) = 3;
                    meta_ptr += 2;
                }
                else if (nz_cnt == 1) {
                    for (int t = 0; t < 4; t++) if (block[t][k] != 0) {
                        if (t != 3) {
                            *opd_ptr = block[t][k];
                            *(opd_ptr + 1) = NAN;
                            opd_ptr += 2;

                            *meta_ptr = t;
                            *(meta_ptr + 1) = 3;
                            meta_ptr += 2;
                        } else {
                            *opd_ptr = NAN;
                            *(opd_ptr + 1) = block[t][k];
                            opd_ptr += 2;

                            *meta_ptr = 2;
                            *(meta_ptr + 1) = 3;
                            meta_ptr += 2;
                        }
                        break;
                    }
                }
                else {
                    for (int t = 0; t < 4; t++) if (block[t][k] != 0) {
                        *opd_ptr = block[t][k];
                        opd_ptr++;
                        *meta_ptr = t;
                        meta_ptr++;
                    }
                }
            }

            assert(opd_ptr == pad_ptr->opd + 2 * TC_row_len);
            assert(meta_ptr == pad_ptr->metadata + 2 * TC_row_len);
        }

        for (int i = 0; i < non_match_num; i++, pad_ptr++) {
            uint32_t vec = ptr->vec;
            memset(block, 0, sizeof(double) * 2 * TC_row_len);
            if (ptr->col2 != NULL) {
                for (auto p0 = ptr->col1; p0 != ptr->col2; p0++)
                    block[0][p0->row] = p0->val;
                for (auto p1 = ptr->col2; p1 != ptr->end; p1++)
                    block[1][p1->row] = p1->val;
            } else {
                for (auto p0 = ptr->col1; p0 != ptr->end; p0++)
                    block[0][p0->row] = p0->val;
            }
            ptr++;

            double* opd_ptr = pad_ptr->opd;
            int* meta_ptr = pad_ptr->metadata;

            for (int k = 0; k < TC_row_len; k++) {
                int nz_cnt = vec & 3; 
                vec >>= 2;

                if (nz_cnt == 0) {
                    *opd_ptr = NAN;
                    *(opd_ptr + 1) = NAN;
                    opd_ptr += 2;

                    *meta_ptr = 2;
                    *(meta_ptr + 1) = 3;
                    meta_ptr += 2;
                }
                else if (nz_cnt == 1) {
                    int t = (block[0][k] == 0);
                    *opd_ptr = block[t][k];
                    *(opd_ptr + 1) = NAN;
                    opd_ptr += 2;

                    *meta_ptr = t;
                    *(meta_ptr + 1) = 3;
                    meta_ptr += 2;
                }
                else {
                    *opd_ptr = block[0][k];
                    *(opd_ptr + 1) = block[1][k];
                    opd_ptr += 2;

                    *meta_ptr = 0;
                    *(meta_ptr + 1) = 1;
                    meta_ptr += 2;
                }
            }

            assert(opd_ptr == pad_ptr->opd + 2 * TC_row_len);
            assert(meta_ptr == pad_ptr->metadata + 2 * TC_row_len);
        }

        assert(ptr == match.unit_list + r);
        int col_num = match_num * 2 + non_match_num * 4;
        if (col_num % TC_col_len != 0) {
            col_num += TC_col_len - (col_num % TC_col_len);
        }
        assert(col_num % TC_col_len == 0);
        padded_col_num += col_num;
    }

    assert(pad_ptr == result.padded_list + units_pack_num);
    result.window_offset[window_num] = pad_ptr - result.padded_list;
    return result_ptr;
}

MetadataResult* unwrap_metadata(UnitsPack* units_ptr, MatchResult* match_ptr, 
    PaddedResult* padded_ptr, const int& window_num, const int& padded_col_num) {
    UnitsPack& units = *units_ptr;
    MatchResult& match = *match_ptr;
    PaddedResult& padded = *padded_ptr;

    MetadataResult* result_ptr = new MetadataResult;
    MetadataResult& result = *result_ptr;
    result.window_offset = (int*)malloc(sizeof(int) * (window_num + 1));
    result.opdA = (double*)malloc(sizeof(double) * (padded_col_num / 2 * TC_row_len));
    result.metadata = (int*)malloc(sizeof(int) * (padded_col_num / 2 * TC_row_len));
    result.old_col = (int*)malloc(sizeof(int) * padded_col_num);

    double null_opd[2 * TC_row_len];
    int null_metadata[2 * TC_row_len];
    for (int i = 0; i < 2 * TC_row_len; i += 2) {
        null_opd[i] = NAN;
        null_opd[i + 1] = NAN;
        null_metadata[i] = 2;
        null_metadata[i + 1] = 3;
    }

    double* opdA_p = result.opdA;
    int* metadata_p = result.metadata;

    constexpr int TC_block_width = TC_col_len / 4; 
    int TC_blocks_count = 0;
    for (int w = 0; w < window_num; w++) {
        result.window_offset[w] = TC_blocks_count;
        int l = padded.window_offset[w];
        int r = padded.window_offset[w + 1];
        int block_num = r - l;
        if (block_num % TC_block_width != 0) {
            block_num += TC_block_width - (block_num % TC_block_width);
        } 
        assert(block_num % TC_block_width == 0);

        const int tc_num = block_num / TC_block_width;
        TC_blocks_count += tc_num;
        double* opd_pad[TC_block_width];
        int* metadata_pad[TC_block_width];
        for (int i = 0; i < tc_num; i++) {
            for (int k = 0; k < TC_block_width; k++) {
                if (l < r) {
                    opd_pad[k] = padded.padded_list[l].opd;
                    metadata_pad[k] = padded.padded_list[l].metadata;
                    l++;
                } else {
                    opd_pad[k] = null_opd;
                    metadata_pad[k] = null_metadata;
                }
            }

            for (int t = 0; t < TC_row_len; t++) {
                for (int k = 0; k < TC_block_width; k++) {
                    *opdA_p = *opd_pad[k];
                    ++opdA_p, ++opd_pad[k];
                    *opdA_p = *opd_pad[k];
                    ++opdA_p, ++opd_pad[k];

                    *metadata_p = *metadata_pad[k];
                    ++metadata_p, ++metadata_pad[k];
                    *metadata_p = *metadata_pad[k];
                    ++metadata_p, ++metadata_pad[k];
                }
            }
        }
    }
    result.window_offset[window_num] = TC_blocks_count;
    assert(opdA_p == result.opdA + (padded_col_num / 2 * TC_row_len));
    assert(metadata_p == result.metadata + (padded_col_num / 2 * TC_row_len));

    int* old_col_p = result.old_col;
    for (int w = 0; w < window_num; w++) {
        const int l = units.window_offset[w];
        const int r = units.window_offset[w + 1];
        const int match_num = match.match_offset[w];
        const int non_match_num = r - l - match_num;
        for (int i = l; i < l + match_num; i++) {
            if (match.unit_list[i].col2 != NULL) {
                *old_col_p = match.unit_list[i].col1->col;
                ++old_col_p;
                *old_col_p = match.unit_list[i].col2->col;
                ++old_col_p;
            } else {
                *old_col_p = match.unit_list[i].col1->col;
                ++old_col_p;
                *old_col_p = -1;
                ++old_col_p;
            }
        }
        for (int i = l + match_num; i < r; i++) {
            if (match.unit_list[i].col2 != NULL) {
                *old_col_p = match.unit_list[i].col1->col;
                ++old_col_p;
                *old_col_p = match.unit_list[i].col2->col;
                ++old_col_p;
            } else {
                *old_col_p = match.unit_list[i].col1->col;
                ++old_col_p;
                *old_col_p = -1;
                ++old_col_p;
            }
            *old_col_p = -1;
            ++old_col_p;
            *old_col_p = -1;
            ++old_col_p;
        }
        const int col_num = match_num * 2 + non_match_num * 4;
        if (col_num % TC_col_len != 0) { 
            const int add_col = TC_col_len - (col_num % TC_col_len);
            std::memset(old_col_p, -1, sizeof(int) * add_col);
            old_col_p += add_col;
        }
    }
    assert(old_col_p == result.old_col + padded_col_num);

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

    TIME_INIT;

    double match_time = 0.0, padding_time = 0.0;

    int original_tc_blocks_quantity;
    TIME(start);
    auto mtx = Matching_And_Padding::read_matrix(argv[1], original_tc_blocks_quantity);
    TIME(end);
    std::cerr << "# read_matrix \t\t\t[" << std::fixed << std::setprecision(3) << TIME_COUNT << " ms]" << std::endl;

    int compressed_tc_blocks_quantity;
    TIME(start);
    auto units_per_window = Matching_And_Padding::pack_columns_into_units(mtx, compressed_tc_blocks_quantity);
    TIME(end);
    std::cerr << "# pack_columns_into_units \t[" << std::fixed << std::setprecision(3) << TIME_COUNT << " ms]" << std::endl;

    int units_pack_num; 
    TIME(start);
    auto match_result = Matching_And_Padding::match_units(units_per_window, mtx->tc_rownum, units_pack_num);
    TIME(end);
    match_time = TIME_COUNT;
    std::cerr << "# match_units \t\t\t[" << std::fixed << std::setprecision(3) << match_time << " ms]" << std::endl;

    int padded_col_num; 
    TIME(start);
    auto padded_list = Matching_And_Padding::padding(units_per_window, match_result, mtx->tc_rownum, units_pack_num, padded_col_num);
    TIME(end);
    padding_time = TIME_COUNT;
    std::cerr << "# padding \t\t\t[" << std::fixed << std::setprecision(3) << padding_time << " ms]" << std::endl;

    TIME(start);
    auto unwrapped_result = Matching_And_Padding::unwrap_metadata(units_per_window, match_result, padded_list, mtx->tc_rownum, padded_col_num);
    TIME(end);
    std::cerr << "# unwrap_metadata \t\t[" << std::fixed << std::setprecision(3) << TIME_COUNT << " ms]" << std::endl;

// #define PRINT_UNITS
#ifdef PRINT_UNITS
    {
        for (int w = 0; w < std::min(mtx->tc_rownum, 10); w++) {
            std::cerr << "window " << w << ": ";
            int unit_num = units_per_window->window_offset[w + 1] - units_per_window->window_offset[w];
            std::cerr << "unit_num = " << unit_num << ", units = ";
            for (int i = 0; i < std::min(unit_num, 10); i++) {
                auto now_unit = units_per_window->unit_list[units_per_window->window_offset[w] + i];
                std::cerr << "[" << now_unit.col1 - mtx->nz_list << " ";
                if (now_unit.col2 == NULL) std::cerr << "-1 ";
                else std::cerr << now_unit.col2 - mtx->nz_list << " ";
                std::cerr << now_unit.end - mtx->nz_list << " | " << now_unit.vec << "] ";
            }
            std::cerr << std::endl;
        }
    }
#endif

// #define PRINT_PADDING
#ifdef PRINT_PADDING
    std::cerr << "opdA: [ ";
    int limit = std::min(100, padded_col_num / 2 * Matching_And_Padding::TC_row_len);
    int out_count = 0;
    bool end_flag = false;
    for (Matching_And_Padding::Padded* ptr = padded_list->padded_list; ; ptr++) {
        for (int i = 0; i < 32; i++) {
            std::cerr << std::fixed << std::setprecision(1) << ptr->opd[i] << " ";
            out_count++;
            if (out_count >= limit) {
                end_flag = true;
                break;
            }
        }
        if (end_flag == true) {
            break;
        }
    }
    std::cerr << "]" << std::endl;

    std::cerr << "metadata: [ ";
    out_count = 0;
    end_flag = false;
    for (Matching_And_Padding::Padded* ptr = padded_list->padded_list; ; ptr++) {
        for (int i = 0; i < 32; i++) {
            std::cerr << std::fixed << std::setprecision(1) << ptr->metadata[i] << " ";
            out_count++;
            if (out_count >= limit) {
                end_flag = true;
                break;
            }
        }
        if (end_flag == true) {
            break;
        }
    }
    std::cerr << "]" << std::endl;
#endif

// #define PRINT_ARRAY
#ifdef PRINT_ARRAY
    std::cerr << "window_offset: [ ";
    for (int i = 0; i < std::min(1000000, mtx->tc_rownum + 1); i++) {
        std::cerr << unwrapped_result->window_offset[i] << " ";
    }
    std::cerr << "]" << std::endl;
    
    std::cerr << "opdA: [ ";
    for (int i = 0; i < std::min(1000000, padded_col_num / 2 * Matching_And_Padding::TC_row_len); i++) {
        std::cerr << std::fixed << std::setprecision(1) << unwrapped_result->opdA[i] << " ";
    }
    std::cerr << "]" << std::endl;

    std::cerr << "metadata: [ ";
    for (int i = 0; i < std::min(100000, padded_col_num / 2 * Matching_And_Padding::TC_row_len); i++) {
        std::cerr << unwrapped_result->metadata[i] << " ";
    }
    std::cerr << "]" << std::endl;

    std::cerr << "old_col: [ ";
    for (int i = 0; i < std::min(100000, padded_col_num); i++) {
        std::cerr << unwrapped_result->old_col[i] << " ";
    }
    std::cerr << "]" << std::endl;
#endif

// #define PRINT_MATCH
#ifdef PRINT_MATCH
    std::cerr << "Matched & Non-matched Units:" << std::endl;
    for (int i = 0; i < std::min(5, mtx->tc_rownum); i++) {
        std::cerr << "  window " << i << ":" << std::endl;
        int l = units_per_window->window_offset[i];
        int r = units_per_window->window_offset[i + 1];

        std::cerr << "    Match: ";
        Matching_And_Padding::Unit* p = match_result->unit_list + l;
        int match_len = match_result->match_offset[i];
        int test_pair_num = 0;
        for (int j = 0; j < match_len; j += 2, p += 2) {
            if (test_pair_num < 10)
                std::cerr << "(" << p->col1->col << ", " << (p + 1)->col1->col << ") ";
            test_pair_num++;
        }
        std::cerr << std::endl;

        std::cerr << "    Non-match: ";
        test_pair_num = 0;
        int non_match_len = r - l - match_len;
        for (int j = 0; j < non_match_len; j++, p++) {
            if (test_pair_num < 10)
                std::cerr << p->col1->col << " ";
            test_pair_num++;
        }
        std::cerr << std::endl;
    }
#endif

#ifdef TESTMODE
    std::cerr << "! Max elements per window: " << Matching_And_Padding::__TESTDATA_max_elements_per_window << std::endl;
#endif

    if (original_tc_blocks_quantity) {
        std::cerr << "| original_tc_blocks_quantity: " << original_tc_blocks_quantity << std::endl;
    }
    std::cerr << "| compressed_tc_blocks_quantity: " << compressed_tc_blocks_quantity << std::endl;
    std::cerr << "| final tc_blocks offset value: " << unwrapped_result->window_offset[mtx->tc_rownum] << std::endl;
    
    const char* base_path = get_config_value("dataset_path");
    char data_bin_path[512];
    const char *input = argv[1];
    const char *last_slash = strrchr(input, '/');

    const char *base_name = last_slash ? (last_slash + 1) : input;

    snprintf(data_bin_path, sizeof(data_bin_path), 
        "%s/dataset_mp_processed/2-4/1.Iterative-matching/%s_data.bin", 
        base_path, base_name);
    
    char output_dir[512];
    snprintf(output_dir, sizeof(output_dir), "%s/dataset_mp_processed/2-4/1.Iterative-matching", base_path);
    create_directory(output_dir);

    write_preprocessed_data(data_bin_path,
        mtx->rows,mtx->cols, 
        mtx->nonzeros,
        unwrapped_result->window_offset, mtx->tc_rownum + 1,
        unwrapped_result->opdA, padded_col_num / 2 * Matching_And_Padding::TC_row_len,
        unwrapped_result->metadata, padded_col_num / 2 * Matching_And_Padding::TC_row_len,
        unwrapped_result->old_col, padded_col_num);

    
    const char* csv_filename = "../../result/time_and_tb_num/time_iter-2-4.csv";
    std::ofstream csv_file(csv_filename, std::ios::app);
    if (csv_file.is_open()) {

        csv_file.seekp(0, std::ios::end);
        bool write_header = (csv_file.tellp() == 0);
        
        if (write_header) {
            csv_file << "dataSet,match_time(ms),padding_time(ms),total_time(ms),"
                     << "original_tc_blocks,compressed_tc_blocks,final_tc_blocks_offset_valuet\n";
        }
        

        csv_file << base_name << ","
                 << std::fixed << std::setprecision(3) << match_time << ","
                 << std::fixed << std::setprecision(3) << padding_time << ","
                 << std::fixed << std::setprecision(3) << (match_time + padding_time) << ","
                 << original_tc_blocks_quantity << ","
                 << compressed_tc_blocks_quantity << ","
                 <<  unwrapped_result->window_offset[mtx->tc_rownum]  << "\n";
        
        csv_file.close();
    } else {
        std::cerr << "Error: Failed to open time result file: " << csv_filename << std::endl;
    }


    return 0;
}
