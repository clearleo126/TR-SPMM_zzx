/**
 * block_matching.cpp — 双路路由 2:4 结构化稀疏匹配 (Dual-Routing TC+SPTC)
 *
 * 算法原理:
 *   将稀疏矩阵按 16 行划分为窗口, 通过位运算进行 1:2/2:4 结构化匹配.
 *   匹配失败的"孤儿列"不再无条件补 0, 而是进入 Dense Pool 进行双路路由:
 *     - 凑满 dense_threshold 列 → 打包为 TC 16×8 稠密块 (TC 路由)
 *     - 不足 dense_threshold → SPTC Fallback 补 0 (每组最多 2 列)
 *
 * 线程安全: 每窗口独立 dense_pool, 结果存入预分配 per-window 数组,
 *   OpenMP 循环后主线程聚合, 零锁开销.
 *
 * 暴露接口:
 *   matching_utils.match_2to4(row_ptr, col_ind, values,
 *                              window_size=16, dense_threshold=8, t_max=8)
 */
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <omp.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

// ============================================================================
// 数据结构
// ============================================================================
struct NonzeroElement{ int row,col ; double val; }; 


struct WinColumn {
    int      col_id;
    uint16_t mask;
    int      nnz;  
    NonzeroElement *col_start, *col_end;
};

struct SparseMatrix {
    int rows, cols, nonzeros;
    int num_windows;
    NonzeroElement* nz_list;                
    int* window_offset;
};


struct Unit{ WinColumn col1, col2; uint16_t G1; uint16_t G2;};


// 存储每个窗口sptc块的数据
struct Win_sptc {
    int window_offset = 0;
    std::vector<double> value;
    std::vector<int> metadata;
    std::vector<int> col_old;
};

// 存储每个窗口tc块的数据
struct Win_tc {
    // 记录当前窗口block数量
    int window_offset = 0;
    // 记录窗口内每个block的非零元个数
    std::vector<int> tc_offset;
    // 记录窗口内每个block局部位置索引
    std::vector<uint8_t> tc_local_id;
    std::vector<int> col_old;
    std::vector<double> value;
};








// ============================================================================
// 核心算法: 单窗口双路路由匹配
//
// 参数:
//   win_rows        - 窗口实际行数
//   columns         - 窗口内候选列 (已按 col_id 排序)
//   dense_threshold - TC 路由阈值
//   t_max           - 细粒度容忍阈值
//
// 列 ID 为 -1 表示虚拟全0列.
// ============================================================================
static void match_window_2to4(
    int win_rows,
    std::vector<WinColumn>& columns,
    int dense_threshold,
    int t_max,
    // 出参:
    std::vector<Unit>& out_sptc_flat,   // [num_groups * 4]
    std::vector<WinColumn>& out_tc_flat     // [num_blocks * dense_threshold]
)
{
    int n = (int)columns.size();
    if (n == 0) return;

    // ================================================================
    // 步骤 2: 细粒度滑动视窗匹配 (1:2 Pairs)
    //
    // 找不到匹配 → 列 ID 压入 local_dense_pool, fine_fb++
    // ================================================================
    const int SEARCH_WINDOW = 32;

    std::vector<bool> matched(n, false);
    std::vector<Unit> units;
    std::vector<WinColumn> local_dense_pool;  // 窗口局部 Dense Pool (列 ID)

    // 循环非零列
    for (int i = 0; i < n; ++i) {
        if (matched[i]) continue;

        uint16_t mask_i = columns[i].mask;
        int best_j = -1;
        int min_cost = t_max + 1;
        int j_end = std::min(i + 1 + SEARCH_WINDOW, n);

        for (int j = i + 1; j < j_end; ++j) {
            if (matched[j]) continue;
            int cost = __builtin_popcount(mask_i & columns[j].mask);
            if (cost == 0) { best_j = j; break; }
            if (cost <= t_max && cost < min_cost) { min_cost = cost; best_j = j; }
        }

        if (best_j != -1) {
            Unit u;
            u.col1 = columns[i];
            u.col2 = columns[best_j];
            u.G1   = mask_i & columns[best_j].mask;
            u.G2   = mask_i | columns[best_j].mask;
            // p.nnz  = columns[i].nnz + columns[best_j].nnz;
            units.push_back(u);
            matched[i] = true;
            matched[best_j] = true;
        } else {
            local_dense_pool.push_back(columns[i]);
            matched[i] = true;
        }
    }

    // ================================================================
    // 步骤 3: 粗粒度滑动视窗匹配 (2:4 Groups)
    //
    // Z == 0 → SPTC Group + O(1) fake_zeros.
    // !found → 拆散 Pair 的 2 列进 local_dense_pool, coarse_fb += 2.
    // ================================================================
    const int GROUP_SEARCH_WINDOW = 1024;

    int np = (int)units.size();
    std::vector<bool> unit_used(np, false);

    for (int i = 0; i < np; ++i) {
        if (unit_used[i]) continue;

        const Unit& ua = units[i];
        bool found = false;
        int j_end = std::min(i + 1 + GROUP_SEARCH_WINDOW, np);

        for (int j = i + 1; j < j_end; ++j) {
            if (unit_used[j]) continue;

            const Unit& ub = units[j];
            uint16_t Z = (ua.G1 & ub.G2) | (ua.G2 & ub.G1);

            if (Z == 0) {
                out_sptc_flat.push_back(ua);
                out_sptc_flat.push_back(ub);


                unit_used[i] = true;
                unit_used[j] = true;
                found = true;
                break;
            }
        }

        if (!found) {
            if (ua.col1.col_id >= 0) local_dense_pool.push_back(ua.col1);
            if (ua.col2.col_id >= 0) local_dense_pool.push_back(ua.col2);
            unit_used[i] = true;
        }
    }

    // ================================================================
    // 步骤 4: 局部双路路由决策
    //
    // ≥ dense_threshold 列 → TC 16×8 稠密块
    // <  dense_threshold 列 → SPTC Fallback (每组最多 2 列 + 补0)
    // ================================================================
    int nd = (int)local_dense_pool.size();
    int processed = 0;

    // 路径 A: TC 路由
    while (nd - processed >= dense_threshold) {
        for (int k = 0; k < dense_threshold; ++k)
            out_tc_flat.push_back(local_dense_pool[processed + k]);
        processed += dense_threshold;
    }

    // 路径 B: SPTC Fallback — 尾部残余, 每组最多塞 2 个真实列
    static const WinColumn kEmptyCol = {-1, 0, 0, nullptr, nullptr};
    int residue = nd - processed;
    if (residue > 0) {
        for (int i = 0; i < residue; ++i) {
            WinColumn col = local_dense_pool[processed + i];
            uint16_t G1 = 0;
            uint16_t G2 = col.mask;
            out_sptc_flat.push_back({col, kEmptyCol, G1, G2});
        }
    }

}

static void data_format(
    const std::vector<Unit>& sptc_flat,
    const std::vector<WinColumn>& tc_flat,
    int dense_threshold,
    Win_sptc& sptc_data,
    Win_tc&   tc_data)
{
    // ========================================================================
    // 1. SPTC 数据打包（对齐 impl-adjacent-matching-2-4.cpp 存储规范）
    // ========================================================================
    int total_units = sptc_flat.size();
    // 连续 8 个 unit 组成一个 16x16 的 Block
    int num_sptc_blocks = (total_units + 7) / 8;
    sptc_data.window_offset = num_sptc_blocks; // 暂时只记录当前窗口的 Block 数量

    // 构建对齐后的 padded_units 缓存，不足 8 倍数的部分末尾填充全零的空 unit
    std::vector<Unit> padded_units = sptc_flat;
    Unit empty_unit;
    empty_unit.col1 = {-1, 0, 0, nullptr, nullptr};
    empty_unit.col2 = {-1, 0, 0, nullptr, nullptr};
    empty_unit.G1 = 0;
    empty_unit.G2 = 0;
    while (padded_units.size() < num_sptc_blocks * 8) {
        padded_units.push_back(empty_unit);
    }

    // 极其高效的单列指定行权重值查找 lambda
    // 因为在输入前，nz_list 已经按 col 优先、row 升序排过序，所以这个线性查找极快
    auto get_val = [](const WinColumn& col, int row) -> double {
        if (col.col_id == -1 || col.col_start == nullptr) return 0.0;
        for (auto* p = col.col_start; p < col.col_end; ++p) {
            if (p->row == row) return p->val;
        }
        return 0.0;
    };

    // 按 Block 维度开始打包
    for (int b = 0; b < num_sptc_blocks; ++b) {
        // 【1.1 原始列索引 col_old 填充】
        // 一个 block 由 8 个 Unit 组成，对应 16 个原始列。填充的空列自然填入 -1。
        for (int u = 0; u < 8; ++u) {
            sptc_data.col_old.push_back(padded_units[b * 8 + u].col1.col_id);
            sptc_data.col_old.push_back(padded_units[b * 8 + u].col2.col_id);
        }

        // 【1.2 权值与元数据打包：行优先（Row-Major）遍历 16 行】
        for (int r = 0; r < 16; ++r) {
            // 每连续 2 个 unit（即 4 列）组成一个 2:4 的 Group，一个 block 含有 4 个 Group
            for (int g = 0; g < 4; ++g) {
                const auto& col_a1 = padded_units[b * 8 + g * 2].col1;
                const auto& col_a2 = padded_units[b * 8 + g * 2].col2;
                const auto& col_b1 = padded_units[b * 8 + g * 2 + 1].col1;
                const auto& col_b2 = padded_units[b * 8 + g * 2 + 1].col2;

                // 抓取该 Group 内 4 个位置对应的真实浮点权重
                double vals[4] = {
                    get_val(col_a1, r),
                    get_val(col_a2, r),
                    get_val(col_b1, r),
                    get_val(col_b2, r)
                };

                // 统计该 4 元组中的非零元数量并记录相对索引位置
                int nz_cnt = 0;
                int nz_indices[4];
                for (int k = 0; k < 4; ++k) {
                    if (vals[k] != 0.0) {
                        nz_indices[nz_cnt++] = k;
                    }
                }

                // 根据非零元数量，执行严格对齐的假 0 (用 0.0 代替) 填充与元数据打码
                if (nz_cnt == 0) {
                    // 全空：强行塞两个 0.0，默认硬件索引为 2 和 3
                    sptc_data.value.push_back(0.0);
                    sptc_data.value.push_back(0.0);
                    sptc_data.metadata.push_back(2);
                    sptc_data.metadata.push_back(3);
                }
                else if (nz_cnt == 1) {
                    int t = nz_indices[0];
                    if (t < 3) {
                        // 孤儿非零元在左侧：右侧补 0.0，索引映射为 [t, 3]
                        sptc_data.value.push_back(vals[t]);
                        sptc_data.value.push_back(0.0);
                        sptc_data.metadata.push_back(t);
                        sptc_data.metadata.push_back(3);
                    } else {
                        // 孤儿非零元在最右侧(t=3)：左侧补 0.0，索引映射为 [2, 3]
                        sptc_data.value.push_back(0.0);
                        sptc_data.value.push_back(vals[t]);
                        sptc_data.metadata.push_back(2);
                        sptc_data.metadata.push_back(3);
                    }
                }
                else {
                    // 正常情况（>= 2）：直接提取前两个非零元，并原样写入其绝对列号作为索引
                    int t1 = nz_indices[0];
                    int t2 = nz_indices[1];
                    sptc_data.value.push_back(vals[t1]);
                    sptc_data.value.push_back(vals[t2]);
                    sptc_data.metadata.push_back(t1);
                    sptc_data.metadata.push_back(t2);
                }
            }
        }
    }

    // ========================================================================
    // 2. TC 数据打包（高压缩 ME-TCF 格式，支持 16×8 / 16×16 稠密路由）
    // ========================================================================
    // 首先判断当前窗口的密集列池是否为空，照顾全走 SPTC 稀疏通道的极稀疏矩阵
    if (tc_flat.empty()) {
        tc_data.window_offset = 0;
    } else {
        // 由于在构建 out_tc_flat 时天然是 dense_threshold 的倍数，故无需考虑边界补齐
        int num_tc_blocks = tc_flat.size() / dense_threshold;
        tc_data.window_offset = num_tc_blocks;

        for (int b = 0; b < num_tc_blocks; ++b) {
            int col_start_idx = b * dense_threshold;

            // 【2.1 写入当前稠密块的原始列索引】
            for (int c = 0; c < dense_threshold; ++c) {
                tc_data.col_old.push_back(tc_flat[col_start_idx + c].col_id);
            }

            int block_nnz = 0; // 局部非零元计数器

            // 【2.2 行优先（Row-Major）扫描提取非零元数据及局部编码】
            for (int r = 0; r < 16; ++r) {
                for (int c = 0; c < dense_threshold; ++c) {
                    const auto& col = tc_flat[col_start_idx + c];
                    
                    double val = 0.0;
                    bool found = false;
                    if (col.col_start != nullptr) {
                        for (auto* p = col.col_start; p < col.col_end; ++p) {
                            if (p->row == r) {
                                val = p->val;
                                found = true;
                                break;
                            }
                        }
                    }

                    // 发现非零权重，记录到 ME-TCF 中
                    if (found && val != 0.0) {
                        tc_data.value.push_back(val);
                        // 8-bit 完美存储 16x16 局部位置索引: row * 16 + local_col
                        uint8_t local_id = static_cast<uint8_t>(r * 16 + c);
                        tc_data.tc_local_id.push_back(local_id);
                        block_nnz++;
                    }
                }
            }
            // 记录当前块内的非零值数量（先保留为单块计数，最终汇总时再求前缀和）
            tc_data.tc_offset.push_back(block_nnz);
        }
    }
}

// ============================================================================
// 顶层入口: 对整个稀疏矩阵执行双路路由匹配
//
// 线程安全: 预分配 per-window 数组, OpenMP 各线程写入 win 索引 (无竞争).
//   循环结束后主线程串行聚合.
// ============================================================================
static void run_2to4_matching(
    SparseMatrix* mtx_ptr,
    int           window_size,
    int           dense_threshold,
    int           t_max,
// 【新增参数】将接收修改的全局大数组引用传进来
    std::vector<int>&    global_sptc_window_offset,
    std::vector<double>& global_sptc_value,
    std::vector<int>&    global_sptc_metadata,
    std::vector<int>&    global_sptc_col_old,
    std::vector<int>&    global_tc_window_offset,
    std::vector<int>&    global_tc_offset,
    std::vector<uint8_t>& global_tc_local_id,
    std::vector<int>&    global_tc_col_old,
    std::vector<double>&  global_tc_value)
{

    SparseMatrix& mtx = *mtx_ptr;
    int num_windows = mtx.num_windows;


    // 预分配 per-window 容器 (不同 win 索引无竞争)
    std::vector<Win_sptc> res_sptc(num_windows);
    std::vector<Win_tc> res_tc(num_windows);

    NonzeroElement* list = mtx.nz_list; // 获取底层连续非零元素数组的首地址

    auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for schedule(dynamic, 16)
    for (int win = 0; win < mtx.num_windows; ++win) {
        int row_start = mtx.window_offset[win];
        int row_end   = mtx.window_offset[win + 1];
        int win_rows  = row_end - row_start;

        // 关键重排：按照“列索引优先”对当前窗口内的非零元进行排序！
        // 如果列索引相同，则按行索引从小到大排。这样同一列的元素在内存中就会彻底连续。
        std::sort(list + row_start, list + row_end, 
            [](const NonzeroElement& a, const NonzeroElement& b) -> bool {
                return a.col != b.col ? a.col < b.col : a.row < b.row;
            }
        );


        std::vector<WinColumn> columns;
        uint16_t vec = 0;
        columns.reserve(512);
        int r;
        for(int l = row_start; l < row_end; l = r) {
            for(r = l; r < row_end && list[r].col == list[l].col; ++r) {
                int local_row = list[r].row;
                uint16_t bit  = (uint16_t)1 << local_row;
                vec |= bit;
            }
            columns.push_back({list[l].col, vec, __builtin_popcount(vec), list + l, list + r});
            vec = 0;
        }

        

        if (columns.empty()) continue;

        // ---- 单窗口双路路由匹配 ----
        std::vector<Unit> sptc_flat;
        std::vector<WinColumn> tc_flat;

        match_window_2to4(win_rows, columns, dense_threshold, t_max,
                          sptc_flat, tc_flat);


        
        Win_sptc sptc_data;
        Win_tc   tc_data;
        data_format(sptc_flat, tc_flat, dense_threshold, sptc_data, tc_data);
        // 无锁写入 per-window 数组
        
        res_sptc[win] = std::move(sptc_data);
        res_tc[win]   = std::move(tc_data);
    }

    auto end = std::chrono::high_resolution_clock::now();
    // 计算耗时
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;


    // 前缀和（CSR指针格式）初始位注入
    global_sptc_window_offset.push_back(0);
    global_tc_window_offset.push_back(0);
    global_tc_offset.push_back(0);

    int current_sptc_block_sum = 0;
    int current_tc_block_sum   = 0;
    int current_tc_nnz_sum     = 0;

    // 严格按窗口顺序遍历聚合，确保拓扑空间连续性
    for (int win = 0; win < mtx.num_windows; ++win) {
        
        // ---- 1. 聚合 SPTC 稀疏块数据 ----
        auto& src = res_sptc[win];
        if (src.window_offset > 0) {
            // 累加当前窗口的稀疏块数量，构建全局前缀和偏移
            current_sptc_block_sum += src.window_offset;
            global_sptc_window_offset.push_back(current_sptc_block_sum);

            // 铺平数据搬运
            global_sptc_value.insert(global_sptc_value.end(), src.value.begin(), src.value.end());
            global_sptc_metadata.insert(global_sptc_metadata.end(), src.metadata.begin(), src.metadata.end());
            global_sptc_col_old.insert(global_sptc_col_old.end(), src.col_old.begin(), src.col_old.end());
        } else {
            // 若该窗口完全没有 SPTC 块，偏移量保持不变
            global_sptc_window_offset.push_back(current_sptc_block_sum);
        }

        // ---- 2. 聚合 TC 稠密块数据 ----
        auto& src_tc = res_tc[win];
        if (src_tc.window_offset > 0) {
            // 累加当前窗口的 TC 块数量，构建全局前缀和偏移
            current_tc_block_sum += src_tc.window_offset;
            global_tc_window_offset.push_back(current_tc_block_sum);

            // ME-TCF 核心：将单块的非零元数量累加为全局非零元前缀和
            for (int nnz_count : src_tc.tc_offset) {
                current_tc_nnz_sum += nnz_count;
                global_tc_offset.push_back(current_tc_nnz_sum);
            }

            // 铺平数据搬运
            global_tc_local_id.insert(global_tc_local_id.end(), src_tc.tc_local_id.begin(), src_tc.tc_local_id.end());
            global_tc_col_old.insert(global_tc_col_old.end(), src_tc.col_old.begin(), src_tc.col_old.end());
            global_tc_value.insert(global_tc_value.end(), src_tc.value.begin(), src_tc.value.end());
        } else {
            // 若该窗口全走 SPTC 通道（TC 块为空），偏移量保持不变
            global_tc_window_offset.push_back(current_tc_block_sum);
        }
    }
    

    return;
}

// 预处理稀疏矩阵
SparseMatrix* deal_matrix(
    const int*    row_ptr,
    const int*    col_ind,
    const float*  values,
    int           rows,
    int           cols,
    int           window_size)
{
    int num_windows = (rows + window_size - 1) / window_size;
    SparseMatrix* mtx_ptr = (SparseMatrix*)malloc(sizeof(SparseMatrix));
    SparseMatrix& mtx = *mtx_ptr; // 使用引用方便后续书写
    mtx.rows = rows;
    mtx.cols = cols;
    mtx.nonzeros = row_ptr[rows];
    mtx.num_windows = num_windows;
    mtx.window_offset = (int*)malloc((num_windows + 1)*sizeof(int));
    mtx.nz_list = (NonzeroElement*)malloc(row_ptr[rows]*sizeof(NonzeroElement));
    NonzeroElement* list_p = mtx.nz_list;

    int nz_cnt = 0, size_cnt;
    for (int win = 0; win < num_windows; ++win) {
        int row_start = win * window_size;
        int row_end   = std::min(row_start + window_size, rows);
        mtx.window_offset[win] = nz_cnt;

        size_cnt = 0;
        for (int r = row_start; r < row_end; ++r) {
            size_cnt += row_ptr[r + 1] - row_ptr[r];
            for (int e = row_ptr[r]; e < row_ptr[r + 1]; ++e) {
                list_p->row = r - row_start;
                list_p->col = col_ind[e];
                list_p->val = values[e];
                list_p++;
            }
        }
        nz_cnt += size_cnt;
    }

    mtx.window_offset[num_windows] = nz_cnt;
    assert(nz_cnt == mtx.nonzeros);
    return mtx_ptr;
}

// ============================================================================
// pybind11 接口
// ============================================================================
py::dict match_2to4_py(
    py::array_t<int,   py::array::c_style | py::array::forcecast> row_ptr_arr,
    py::array_t<int,   py::array::c_style | py::array::forcecast> col_ind_arr,
    py::array_t<float, py::array::c_style | py::array::forcecast> values_arr,
    int window_size     = 16,
    int dense_threshold = 16,
    int t_max           = 8)
{
    py::buffer_info row_buf = row_ptr_arr.request();
    py::buffer_info col_buf = col_ind_arr.request();
    py::buffer_info val_buf = values_arr.request();

    const int*   row_ptr = static_cast<const int*>(row_buf.ptr);
    const int*   col_ind = static_cast<const int*>(col_buf.ptr);
    const float* values  = static_cast<const float*>(val_buf.ptr);

    int rows = (int)row_buf.shape[0] - 1;
    int nnz  = (int)col_buf.shape[0];

    int cols = 0;
    for (int i = 0; i < nnz; ++i)
        if (col_ind[i] >= cols) cols = col_ind[i] + 1;

    if (window_size < 1 || window_size > 64)
        throw std::runtime_error("window_size 必须在 [1, 64] 范围内");
    if (rows <= 0 || cols <= 0)
        throw std::runtime_error("无效的矩阵维度");

    auto mtx = deal_matrix(row_ptr, col_ind, values, rows, cols, window_size);

    // sptc
    std::vector<int>    global_sptc_window_offset;
    std::vector<double> global_sptc_value;
    std::vector<int>    global_sptc_metadata;
    std::vector<int>    global_sptc_col_old;

    // tc
    std::vector<int>     global_tc_window_offset;
    std::vector<int>     global_tc_offset;
    std::vector<uint8_t> global_tc_local_id;
    std::vector<int>     global_tc_col_old;
    std::vector<double>  global_tc_value;

    run_2to4_matching(
        mtx, window_size, dense_threshold, t_max,
        global_sptc_window_offset, global_sptc_value, global_sptc_metadata, global_sptc_col_old,
        global_tc_window_offset, global_tc_offset, global_tc_local_id, global_tc_col_old, global_tc_value
    );


    // 【核心转换】利用深拷贝模板构建通用的 C++ vector -> Torch Tensor 转换器
    auto to_torch_tensor = [](void* data, size_t size, torch::ScalarType dtype) {
        // 特殊防御：处理空矩阵（例如某些矩阵极稠密，完全没有 SPTC 数据；或者极稀疏完全没有 TC 块）
        if (size == 0) {
            return torch::empty({0}, torch::TensorOptions().dtype(dtype));
        }
        // A. 建立影子 View（零拷贝，但生命周期不安全）
        auto temp_tensor = torch::from_blob(data, {static_cast<int64_t>(size)}, torch::TensorOptions().dtype(dtype));
        // B. 申请独立安全的 PyTorch 显存/内存块
        auto safe_tensor = torch::empty_like(temp_tensor);
        // C. 深拷贝数据（防止 C++ 向量销毁后出现悬挂野指针崩溃）
        safe_tensor.copy_(temp_tensor);
        return safe_tensor;
    };


    py::dict result;

    // ---- 转换并打包 SPTC Tensor 队列 ----
    result["sptc_window_offset"]   = to_torch_tensor(global_sptc_window_offset.data(), global_sptc_window_offset.size(), torch::kInt32);
    result["sptc_value"]           = to_torch_tensor(global_sptc_value.data(), global_sptc_value.size(), torch::kFloat64); // double 对应 Float64
    result["sptc_metadata"]        = to_torch_tensor(global_sptc_metadata.data(), global_sptc_metadata.size(), torch::kInt32);
    result["sptc_col_old"]         = to_torch_tensor(global_sptc_col_old.data(), global_sptc_col_old.size(), torch::kInt32);

    // ---- 转换并打包 TC ME-TCF Tensor 队列 ----
    result["tc_window_offset"]     = to_torch_tensor(global_tc_window_offset.data(), global_tc_window_offset.size(), torch::kInt32);
    result["tc_offset"]            = to_torch_tensor(global_tc_offset.data(), global_tc_offset.size(), torch::kInt32);
    result["tc_local_id"]          = to_torch_tensor(global_tc_local_id.data(), global_tc_local_id.size(), torch::kUInt8); // uint8_t 对应 Byte/UInt8
    result["tc_col_old"]           = to_torch_tensor(global_tc_col_old.data(), global_tc_col_old.size(), torch::kInt32);
    result["tc_value"]             = to_torch_tensor(global_tc_value.data(), global_tc_value.size(), torch::kFloat64);

    // 自动释放 mtx 堆空间
    free(mtx->window_offset);
    free(mtx->nz_list);
    free(mtx);

    return result;
}


// ============================================================================
// 模块定义
// ============================================================================
PYBIND11_MODULE(matching_utils, m) {
    m.doc() = R"pbdoc(
        Dual-Routing TC+SPTC 结构化稀疏匹配模块

        核心算法:
          1. 16行窗口划分 → uint16_t 编码
          2. 滑动视窗细粒度匹配 (1:2 Pair)
          3. Z_conflict==0 粗粒度匹配 (2:4 Group)
          4. 匹配失败列 → Dense Pool → 双路路由:
               ≥dense_threshold → TC 16×8 块
               <dense_threshold  → SPTC Fallback 补0
          5. O(1) fake_zeros 计算

        接口:
          match_2to4(row_ptr, col_ind, values,
                     window_size=16, dense_threshold=16, t_max=8) -> dict
    )pbdoc";

    m.def("match_2to4", &match_2to4_py,
          py::arg("row_ptr"),
          py::arg("col_ind"),
          py::arg("values"),
          py::arg("window_size")     = 16,
          py::arg("dense_threshold") = 16,
          py::arg("t_max")           = 8,
          R"pbdoc(
对 CSR 格式稀疏矩阵执行 Dual-Routing TC+SPTC 匹配.

参数:
    row_ptr (np.ndarray):    CSR 行偏移, dtype=int32
    col_ind (np.ndarray):    CSR 列索引, dtype=int32
    values (np.ndarray):     CSR 数值, dtype=float32
    window_size (int):       窗口行数, 默认 16
    dense_threshold (int):   TC 路由阈值, 默认 16
    t_max (int):             细粒度容忍阈值, 默认 8
          )pbdoc");
}
