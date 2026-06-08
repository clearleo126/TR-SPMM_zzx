#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <chrono>

#define SPTC_CONFLICT_MASK 0xaaaaaaaa

// O(1) 冲突检测：检查两列是否可以合并
__device__ __forceinline__ bool check_match_2(uint32_t a, uint32_t b) {
    uint32_t c = a | b;
    return (((a & b) | ((c << 1) & c)) & SPTC_CONFLICT_MASK) == 0;
}

// Warp 级贪心匹配 Kernel
__global__ void warp_match_kernel(uint32_t* vecs, int* out_matches, int num_cols) {
    int tid = threadIdx.x;
    int lane_id = tid % 32;
    
    // 每个线程负责读取一列的 2-bit 编码 (vec)
    // 如果超出实际列数，赋予全冲突的掩码 0xffffffff
    uint32_t my_vec = (lane_id < num_cols) ? vecs[lane_id] : 0xffffffff;
    
    // 简单的 Warp 内贪心匹配：由 Thread 0 寻找兼容的 3 个伙伴凑齐 2:4
    if (lane_id == 0) {
        int partners[4] = {0, -1, -1, -1};
        int count = 1;
        uint32_t current_combined_vec = my_vec;
        
        // 遍历 Warp 内的其他线程
        for (int i = 1; i < 32 && count < 4; i++) {
            // 使用 Warp Shuffle 极速读取其他线程的 vec，不需要走 Shared Memory
            uint32_t other_vec = __shfl_sync(0xffffffff, my_vec, i);
            
            if (i < num_cols) {
                if (check_match_2(current_combined_vec, other_vec)) {
                    partners[count++] = i;
                    current_combined_vec |= other_vec; // 更新合并后的状态
                }
            }
        }
        
        // 输出匹配结果
        for(int i = 0; i < 4; i++) {
            out_matches[i] = partners[i];
        }
    }
}

int main() {
    // 模拟 8 列数据的 2-bit 编码
    // 假设窗口内有 16 行，每行 2-bit
    std::vector<uint32_t> h_vecs = {
        0x00000001, // Col 0: 行 0 有 1 个非零元
        0x00000002, // Col 1: 行 0 有 2 个非零元 (与 Col 0 冲突!)
        0x00000004, // Col 2: 行 1 有 1 个非零元
        0x00000008, // Col 3: 行 1 有 2 个非零元 (与 Col 2 冲突!)
        0x00000010, // Col 4: 行 2 有 1 个非零元
        0x00000000, // Col 5: 全空
        0x00000000, // Col 6: 全空
        0x00000000  // Col 7: 全空
    };
    int num_cols = h_vecs.size();

    uint32_t* d_vecs;
    int* d_matches;
    cudaMalloc(&d_vecs, num_cols * sizeof(uint32_t));
    cudaMalloc(&d_matches, 4 * sizeof(int));
    cudaMemcpy(d_vecs, h_vecs.data(), num_cols * sizeof(uint32_t), cudaMemcpyHostToDevice);

    // 预热
    warp_match_kernel<<<1, 32>>>(d_vecs, d_matches, num_cols);
    cudaDeviceSynchronize();

    // 计时
    auto start = std::chrono::high_resolution_clock::now();
    warp_match_kernel<<<1, 32>>>(d_vecs, d_matches, num_cols);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    int h_matches[4];
    cudaMemcpy(h_matches, d_matches, 4 * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Warp 匹配结果 (列索引): ";
    for (int i = 0; i < 4; i++) {
        std::cout << h_matches[i] << " ";
    }
    std::cout << std::endl;
    
    std::chrono::duration<double, std::nano> elapsed = end - start;
    std::cout << "Kernel 执行时间: " << elapsed.count() << " 纳秒!" << std::endl;

    cudaFree(d_vecs);
    cudaFree(d_matches);
    return 0;
}