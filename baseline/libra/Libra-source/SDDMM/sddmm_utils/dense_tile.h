
#include <mma.h>
#include <cstdint>
#include <stdio.h>
#include <cuda_fp16.h>
#include <torch/torch.h>
struct mmaDenseTile_tf32_v2{
        const float *  values_;
        const int *  column_idxs_;
        const int rhs_cols_;
        const int lane_id_;
        const int warpin_id;
        const int warp_id;
        const float *matrix_base_;
        float *dense_tile_;
        //存放当前线程拿到的一个double值，并将该double值拆分成4个half分别进行转置放置
        float *sparse_fragment_;

        __device__ __forceinline__ mmaDenseTile_tf32_v2(
        long row_offset_vec,
        const float * values,
        const int *  column_idxs,
	    int rhs_cols,
        int offset, 
        int lane_id, 
        const float*  matrix, 
        //row_offsets= column_indices_tile
        // const int *row_offsets,
        float * dense_tile,
        float *sparse_fragment):
            rhs_cols_(rhs_cols),
            lane_id_(lane_id),
            warpin_id(lane_id & 31),
            warp_id(lane_id>>5),
            //当前block在全局的列偏移
            matrix_base_(matrix + offset),
            //8的意思是vector的长度
            values_((values + row_offset_vec*8) + (lane_id & 31)),
            //对4*16的RHS读取，每行连续读8个线程，共4行，所以需要>>3
            column_idxs_(column_idxs + row_offset_vec + ((lane_id & 31)%4)),
            dense_tile_(dense_tile),
            sparse_fragment_(sparse_fragment)
            {}
    
        __device__ __forceinline__ void Fetch(int colEdge, int dimN_index){

            sparse_fragment_[0]= __ldg(values_);
            const long row_offsets_ = __ldg(column_idxs_);
            values_ += 32;
            column_idxs_ += 4;
            // (warp_id<<4) 每个warp有16列
            //行偏移,(warpin_id%8)*2),每行8个线程，每个线程读两个float数
            const int global_offset = (warp_id<<4) + (warpin_id/4);
            const long offset = (row_offsets_*rhs_cols_) + global_offset;
            float dense_tile_fp32[2]={0.0,0.0};
            for(int i=0;i<2;i++)
            {
                if((dimN_index+global_offset+i)<colEdge)
                dense_tile_fp32[i]=__ldg(matrix_base_ + offset + i*8);
            } 
            for(int i=0;i<2;i++)
                dense_tile_[i]=dense_tile_fp32[i];    
        }

        // Load the residual and compute the matrix product
        __device__ __forceinline__ void ResidueLoad(int colEdge, int dimN_index){
            sparse_fragment_[0]=__ldg(values_);
            const long row_offsets_ = __ldg(column_idxs_);
            const int global_offset = (warp_id<<4) + (warpin_id/4);
            float dense_tile_fp32[2]={0.0,0.0};
            if(row_offsets_ >= 0){
                const long offset = (row_offsets_*rhs_cols_) + global_offset;
                // matrix_base_=matrix_base_ + (row_offsets_*rhs_cols_) + global_offset;
                 for(int i=0;i<2;i++)
                 {
                    if((dimN_index+global_offset+i*8)<colEdge)
                        dense_tile_fp32[i]=__ldg(matrix_base_ +offset+ i*8);
                 }   
            }
            for(int i=0;i<2;i++)
                dense_tile_[i]=dense_tile_fp32[i];
        }
    };


struct mmaDenseTile_tf32{
    const float *  values_;
    const int *  column_idxs_;
    const int rhs_cols_;
    const int lane_id_;
    const int warpin_id;
    const int warp_id;
    const float *matrix_base_;
    float *dense_tile_;
    //存放当前线程拿到的一个double值，并将该double值拆分成4个half分别进行转置放置
    float *sparse_fragment_;

    __device__ __forceinline__ mmaDenseTile_tf32(
    long row_offset_vec,
    const float * values,
    const int *  column_idxs,
    int rhs_cols,
    int lane_id, 
    const float*  matrix, 
    //row_offsets= column_indices_tile
    // const int *row_offsets,
    float * dense_tile,
    float *sparse_fragment):
        rhs_cols_(rhs_cols),
        lane_id_(lane_id),
        warpin_id(lane_id & 31),
        warp_id(lane_id>>5),
        //当前block在全局的列偏移
        matrix_base_(matrix),
        //8的意思是vector的长度
        values_((values + row_offset_vec*8) + (lane_id & 31)),
        //对4*16的RHS读取，每行连续读8个线程，共4行，所以需要lane_id & 31)%4)
        column_idxs_(column_idxs + row_offset_vec + ((lane_id & 31)%4)),
        dense_tile_(dense_tile),
        sparse_fragment_(sparse_fragment)
        {}

    __device__ __forceinline__ void Fetch(int colEdge, int dimN_index){

        sparse_fragment_[0]= __ldg(values_);
        const long row_offsets_ = __ldg(column_idxs_);
        values_ += 32;
        column_idxs_ += 4;
        // (warp_id<<4) 每个warp有16列
        const int block_offset = (warp_id<<4) + (warpin_id/4);
        const long global_offset = (row_offsets_*rhs_cols_) +  dimN_index + block_offset;
        float dense_tile_fp32[2]={0.0,0.0};
        for(int i=0;i<2;i++)
        {
            if((dimN_index+block_offset+i)<colEdge)
            dense_tile_fp32[i]=__ldg(matrix_base_ + global_offset + i*8);
        } 
        for(int i=0;i<2;i++)
            dense_tile_[i]=dense_tile_fp32[i];    
    }

    // Load the residual and compute the matrix product
     __device__ __forceinline__ void ResidueLoad(int colEdge, int dimN_index){
            sparse_fragment_[0]= __ldg(values_);
            const long row_offsets_ = __ldg(column_idxs_);
            float dense_tile_fp32[2]={0.0,0.0};
            if(row_offsets_ >= 0){
                const int block_offset = (warp_id<<4) + (warpin_id/4);
                const long global_offset = (row_offsets_*rhs_cols_) +  dimN_index + block_offset;
                // matrix_base_=matrix_base_ + (row_offsets_*rhs_cols_) + global_offset;
                 for(int i=0;i<2;i++)
                 {
                    if((dimN_index+global_offset+i*8)<colEdge)
                        dense_tile_fp32[i]=__ldg(matrix_base_ +global_offset+ i*8);
                 }   
            }
            for(int i=0;i<2;i++)
                dense_tile_[i]=dense_tile_fp32[i];
        }
};



//16
    struct mmaDenseTile_tf32_16{
        const float *  values_;
        const int *  column_idxs_;
        const int rhs_cols_;
        const int lane_id_;
        const int warpin_id;
        const int warp_id;
        const float *matrix_base_;
        float *dense_tile_;
        //存放当前线程拿到的一个double值，并将该double值拆分成4个half分别进行转置放置
        float *sparse_fragment_;

        __device__ __forceinline__ mmaDenseTile_tf32_16(
        long row_offset_vec,
        const float * values,
        const int *  column_idxs,
	    int rhs_cols,
        int offset, 
        int lane_id, 
        const float*  matrix, 
        //row_offsets= column_indices_tile
        // const int *row_offsets,
        float * dense_tile,
        float *sparse_fragment):
            rhs_cols_(rhs_cols),
            lane_id_(lane_id),
            warpin_id(lane_id & 31),
            warp_id(lane_id>>5),
            //当前block在全局的列偏移
            matrix_base_(matrix + offset),
            //8的意思是vector的长度
            values_((values + row_offset_vec*16) + (lane_id & 31)),
            //对4*16的RHS读取，每行连续读8个线程，共4行，所以需要>>3
            column_idxs_(column_idxs + row_offset_vec + ((lane_id & 31)%4)),
            dense_tile_(dense_tile),
            sparse_fragment_(sparse_fragment)
            {}
    
        __device__ __forceinline__ void Fetch(int colEdge, int dimN_index){
            // float sparse[2];
            // float a = 1.0;
            // for(int i=0; i<2; i++)
            // {
            //     sparse[i]=__ldg(values_);
            //     //sparse_fragment_[i]=1.0;
            //     values_ += 32;
            // }
            // for(int i =0; i<2; i++)
            // sparse_fragment_[i]=a;
            for(int i =0; i<2; i++){
            sparse_fragment_[i]= __ldg(values_);
            // if(blockIdx.x==0 and blockIdx.y==1220 and blockIdx.z==0 and threadIdx.x==0)
            // printf("%f ", sparse_fragment_[i]);
            values_ += 32;}
            const long row_offsets_ = __ldg(column_idxs_);
            // values_ += 32;
            column_idxs_ += 4;
            // (warp_id<<4) 每个warp有16列
            //行偏移,(warpin_id%8)*2),每行8个线程，每个线程读两个float数
            const int global_offset = (warp_id<<3) + ((warpin_id/4));
            const long offset = (row_offsets_*rhs_cols_) + global_offset;

            float dense_tile_fp32[1]={0.0};
            if((dimN_index+global_offset)<colEdge)
            dense_tile_fp32[0] =__ldg(matrix_base_ +offset);
             
             *(dense_tile_)=dense_tile_fp32[0];
        }

        // Load the residual and compute the matrix product
        __device__ __forceinline__ void ResidueLoad(int colEdge, int dimN_index){
            // float sparse[2];
            //             float a = 1.0;
            // for(int i=0; i<2; i++)
            // {
            //     // sparse_fragment_[i]=__ldg(values_);
            //     sparse[i]=__ldg(values_);
            //     values_ += 32;
            // }
            // for(int i =0; i<2; i++)
            // sparse_fragment_[i]=a;
            for(int i =0; i<2; i++){
            sparse_fragment_[i]= __ldg(values_);
            values_ += 32;}
            const long row_offsets_ = __ldg(column_idxs_);
            const int global_offset = (warp_id<<3) + ((warpin_id/4));
            // float dense_tile_fp32[2]={0.0,0.0};
            float dense_tile_fp32[1]={0.0};
            if(row_offsets_ >= 0){
                const long offset = (row_offsets_*rhs_cols_) + global_offset;
                // matrix_base_=matrix_base_ + (row_offsets_*rhs_cols_) + global_offset;
 
                if((dimN_index+global_offset)<colEdge)
               dense_tile_fp32[0] = __ldg(matrix_base_ +offset);
                
            }
            *(dense_tile_)=dense_tile_fp32[0];
            // if(threadIdx.x==36 & blockIdx.y==0)
            // {
            //     printf("%d\n",dimN_index+global_offset);
            //     printf("%d\n",colEdge);
            // }
            // int k=(warpin_id/4)%2; //T0,1,2,3,8,9,...
            // if(k==0){
            //     *(dense_tile_ +  warp_id*64 + ((warpin_id%8)*2)*4 + warpin_id/8) = dense_tile_fp32[0];
            //     *(dense_tile_ +  warp_id*64 + (((warpin_id%8)*2)+1)*4 + warpin_id/8) = dense_tile_fp32[1];
            // }
            // else{
            //     *(dense_tile_ +  warp_id*64 + (((warpin_id%8)*2)+1)*4 + warpin_id/8) = dense_tile_fp32[1];
            //     *(dense_tile_ +  warp_id*64 + ((warpin_id%8)*2)*4 + warpin_id/8) = dense_tile_fp32[0];
            // }
        }
    };
