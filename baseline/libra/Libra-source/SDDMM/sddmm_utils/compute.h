#include <mma.h>
#include <cstdint>
#include <stdio.h>
#include <cuda_fp16.h>
#include <torch/extension.h>


/*
TF32
*/
struct mmaComputeUtils_tf32_gen{
    const float* lhs_tile_;
    const float* rhs_tile_;
    float* output_fragment_;
    const int lane_id_;
    const int warpin_id;
    const int warp_id;
    int dimN_index_;
    int col_;
    int row_;
    int dimW_;

    // Constructor
    __device__ __forceinline__ mmaComputeUtils_tf32_gen(
        const float *lhs_tile,
        const float *rhs_tile,
        float* output_fragment,
        int lane_id):
            lhs_tile_(lhs_tile),
            rhs_tile_(rhs_tile),
            output_fragment_(output_fragment),
            lane_id_(lane_id),
            warpin_id(lane_id & 31),
            warp_id(lane_id>>5){}
    
    // Compute
    __device__ __forceinline__ void TileMAC(int dimW, long col, long col1,int dimMori, long row){
        //load lhs_matrix
        float lhs_fragment_tf32[2]={0.0, 0.0};
        uint32_t *lhs_fragment = reinterpret_cast<uint32_t *>(lhs_fragment_tf32);
        if(row < dimMori){
        lhs_fragment_tf32[0]=*(lhs_tile_+((warpin_id%4)));
        lhs_fragment_tf32[1]=*(lhs_tile_+((warpin_id%4))+4);
        lhs_tile_=lhs_tile_+8;}

        //load rhs_matrix
        float rhs_fragment_tf32[4]={0.0, 0.0, 0.0, 0.0};
        uint32_t *rhs_fragment = reinterpret_cast<uint32_t *>(rhs_fragment_tf32);
        if(col>=0){
        rhs_fragment_tf32[0]=*(rhs_tile_ + (warpin_id%4));
        rhs_fragment_tf32[2]=*(rhs_tile_ + (warpin_id%4) + 4);
        }
        if((col1)>=0){
        rhs_fragment_tf32[1]=*(rhs_tile_ + (warpin_id%4) + ((col1-col)*dimW));
        rhs_fragment_tf32[3]=*(rhs_tile_ + (warpin_id%4) + 4 + ((col1-col)*dimW));
        }
        //更新rhs_tile
        rhs_tile_=rhs_tile_+8;
            // if(warpin_id==4&warp_id==1)
            // {
            //     half *p=reinterpret_cast<half *>(rhs_fragment);
            //     for(int i=0;i<8;i++)
            //     printf("thread_id:%d, value is:%f\n", warpin_id,__half2float(*(p+i)));
            // }
     
       
        asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
            : "=f"(output_fragment_[0]), "=f"(output_fragment_[1]), "=f"(output_fragment_[2]), "=f"(output_fragment_[3])
            : "r"(rhs_fragment[0]), "r"(rhs_fragment[1]), "r"(rhs_fragment[2]), "r"(rhs_fragment[3]),
             "r"(lhs_fragment[0]), "r"(lhs_fragment[1]), 
             "f"(output_fragment_[0]), "f"(output_fragment_[1]), "f"(output_fragment_[2]), "f"(output_fragment_[3]));      
    } 

    
    

    __device__ __forceinline__ void TileMACResidue(int residue,int dimW, long col, long col1, int dimMori,  long row){
        //load lhs_matrix
        float lhs_fragment_tf32[2]={0.0, 0.0};
        uint32_t *lhs_fragment = reinterpret_cast<uint32_t *>(lhs_fragment_tf32);
        if((warpin_id%4)<residue)
        lhs_fragment_tf32[0]=*(lhs_tile_+(warpin_id%4));
        if(((warpin_id%4) + 4)<residue)
        lhs_fragment_tf32[1]=*(lhs_tile_+(warpin_id%4) + 4);

        //load rhs_matrix
        float rhs_fragment_tf32[4]={0.0, 0.0, 0.0, 0.0};
        uint32_t *rhs_fragment = reinterpret_cast<uint32_t *>(rhs_fragment_tf32);
        if(col>=0){
            if((warpin_id%4)<residue)
            rhs_fragment_tf32[0]=*(rhs_tile_ +(warpin_id%4));
            if(((warpin_id%4) +4)<residue)
            rhs_fragment_tf32[2]=*(rhs_tile_ +((warpin_id%4)*2) +8);
        }
        if((col1)>=0){
            if((warpin_id%4)<residue)
            rhs_fragment_tf32[1]=*(rhs_tile_ + (warpin_id%4) + ((col1-col)*dimW));
            if(((warpin_id%4) +4)<residue)
            rhs_fragment_tf32[3]=*(rhs_tile_ + (warpin_id%4) + 4 + ((col1-col)*dimW));
        }

        //MMA 
        asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
            : "=f"(output_fragment_[0]), "=f"(output_fragment_[1]), "=f"(output_fragment_[2]), "=f"(output_fragment_[3])
            : "r"(rhs_fragment[0]), "r"(rhs_fragment[1]), "r"(rhs_fragment[2]), "r"(rhs_fragment[3]),
             "r"(lhs_fragment[0]), "r"(lhs_fragment[1]), 
             "f"(output_fragment_[0]), "f"(output_fragment_[1]), "f"(output_fragment_[2]), "f"(output_fragment_[3]));      
        }

};

struct mmaComputeUtils_tf32_gen_v2{
    const float* lhs_tile_;
    const float4* rhs_tile_;
    float4* block_a_;
    // float2* block_b_;
    float* output_fragment_;
    const int lane_id_;
    const int warpin_id;
    const int warp_id;
    int dimN_index_;
    int col_;
    int row_;
    int dimW_;

    // Constructor
    __device__ __forceinline__ mmaComputeUtils_tf32_gen_v2(
        const float *lhs_tile,
        const float4 *rhs_tile,
        float4* block_a,
        float* output_fragment,
        int lane_id):
            lhs_tile_(lhs_tile),
            rhs_tile_(rhs_tile),
            block_a_(block_a),
            // block_b_(block_b),
            output_fragment_(output_fragment),
            lane_id_(lane_id),
            warpin_id(lane_id & 31),
            warp_id(lane_id>>5){}
    
    // Compute
    __device__ __forceinline__ void TileMAC(int dimW, int col, int dimMori, int row){
        
        // block_b_[warpin_id] = make_float2(0.0f, 0.0f);
        block_a_[warpin_id] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        // block_b_[warpin_id] = make_float2(0.0f, 0.0f);
        //搬运到shared memory
        //搬运lhs 到 block_b
        // if(row < dimMori and warp_id==0)
        // {
        //     block_b_[warpin_id] = __ldg(lhs_tile_);
        // }

        if(col>=0)
        block_a_[warpin_id] = __ldg(rhs_tile_);

        // __syncthreads();
        __syncwarp();
        // float * block_b_f = reinterpret_cast<float *>(block_b_);
        float * block_a_f = reinterpret_cast<float *>(block_a_);


        // if(warpin_id==0&warp_id==0 & blockIdx.y==0)
        // {
        //     for(int i=0;i<64;i++){
        //     printf("%f ", *(block_b_f+i));
        //     if(i%8==7)
        //     printf("\n");}

        //     printf("\n");

        //     for(int i=0;i<128;i++){
        //     printf("%f ", *(block_a_f+i));
        //   if(i%8==7)
        //     printf("\n");}
        //     printf("\n");
        // }


        //load lhs_matrix from block_b_
        float lhs_fragment_tf32[2]={0.0, 0.0};
        uint32_t *lhs_fragment = reinterpret_cast<uint32_t *>(lhs_fragment_tf32);
        if(row < dimMori){
        lhs_fragment_tf32[0]=*(lhs_tile_+((warpin_id%4)));
        lhs_fragment_tf32[1]=*(lhs_tile_+((warpin_id%4))+4);
        lhs_tile_=lhs_tile_+8;}

        //load rhs_matrix from block_a_
        float rhs_fragment_tf32[4]={0.0, 0.0, 0.0, 0.0};
        uint32_t *rhs_fragment = reinterpret_cast<uint32_t *>(rhs_fragment_tf32);
        int offset = (warpin_id%4) + (warpin_id/4)*8;
        // if(col1>=0){
        rhs_fragment_tf32[0]=*(block_a_f + offset);
        rhs_fragment_tf32[2]=*(block_a_f + offset + 4);
        // }
        // if((col2)>=0){
        rhs_fragment_tf32[1]=*(block_a_f + offset + 64);
        rhs_fragment_tf32[3]=*(block_a_f + offset + 68);
        // }
        //更新rhs_tile
        rhs_tile_=rhs_tile_+2;

     
        // if(warpin_id==0&warp_id==0 & blockIdx.y==0)
        // {
        //     for(int i=0;i<2;i++)
        //     printf("%f ", *(lhs_fragment_tf32+i));

        //     printf("\n");

        //     for(int i=0;i<4;i++)
        //     printf("%f ", *(rhs_fragment_tf32+i));

        //     printf("\n");
        // }

        asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
            : "=f"(output_fragment_[0]), "=f"(output_fragment_[1]), "=f"(output_fragment_[2]), "=f"(output_fragment_[3])
            : "r"(rhs_fragment[0]), "r"(rhs_fragment[1]), "r"(rhs_fragment[2]), "r"(rhs_fragment[3]),
             "r"(lhs_fragment[0]), "r"(lhs_fragment[1]), 
             "f"(output_fragment_[0]), "f"(output_fragment_[1]), "f"(output_fragment_[2]), "f"(output_fragment_[3]));      
    } 

    
    

    // __device__ __forceinline__ void TileMACResidue(int residue,int dimW,  int col1,int col2, int dimMori,  long row){
    //     //load to shared mmeory
    //     block_b_[warpin_id] = __ldg(lhs_tile_);
    //     block_a_[warpin_id] = __ldg(rhs_tile_);
    //     float * block_b_f = reinterpret_cast<float *>(block_b_);
    //     float * block_a_f = reinterpret_cast<float *>(block_a_);
    //     int offset = (warpin_id%4) + (warpin_id/4)*8;
    //     //load lhs_matrix
    //     float lhs_fragment_tf32[2]={0.0, 0.0};
    //     uint32_t *lhs_fragment = reinterpret_cast<uint32_t *>(lhs_fragment_tf32);
    //     if((warpin_id%4)<residue)
    //     lhs_fragment_tf32[0]=*(block_b_f+ warpin_id);
    //     if(((warpin_id%4) + 4)<residue)
    //     lhs_fragment_tf32[1]=*(block_b_f+ warpin_id + 32);

    //     //load rhs_matrix
    //     float rhs_fragment_tf32[4]={0.0, 0.0, 0.0, 0.0};
    //     uint32_t *rhs_fragment = reinterpret_cast<uint32_t *>(rhs_fragment_tf32);
    //     if(col1>=0){
    //         if((warpin_id%4)<residue)
    //         rhs_fragment_tf32[0]=*(block_a_f + offset);
    //         if(((warpin_id%4) +4)<residue)
    //         rhs_fragment_tf32[0]=*(block_a_f + offset + 4);
    //     }
    //     if((col2)>=0){
    //         if((warpin_id%4)<residue)
    //         rhs_fragment_tf32[1]=*(block_a_f + offset + 64);
    //         if(((warpin_id%4) +4)<residue)
    //         rhs_fragment_tf32[1]=*(block_a_f + offset + 68);
    //     }

    //     //MMA 
    //     asm volatile(
    //     "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
    //         : "=f"(output_fragment_[0]), "=f"(output_fragment_[1]), "=f"(output_fragment_[2]), "=f"(output_fragment_[3])
    //         : "r"(rhs_fragment[0]), "r"(rhs_fragment[1]), "r"(rhs_fragment[2]), "r"(rhs_fragment[3]),
    //          "r"(lhs_fragment[0]), "r"(lhs_fragment[1]), 
    //          "f"(output_fragment_[0]), "f"(output_fragment_[1]), "f"(output_fragment_[2]), "f"(output_fragment_[3]));      
    //     }

};

/*
FP16
*/

struct mmaComputeUtils_fp16_gen{
    const half2* lhs_tile_;
    const half2* rhs_tile_;
    uint32_t* output_fragment_;
    const int lane_id_;
    const int warpin_id;
    const int warp_id;
    int dimN_index_;
    int col_;
    int row_;
    int dimW_;

    // Constructor
    __device__ __forceinline__ mmaComputeUtils_fp16_gen(
        const half2 *lhs_tile,
        const half2*rhs_tile,
        uint32_t* output_fragment,
        int lane_id):
            lhs_tile_(lhs_tile),
            rhs_tile_(rhs_tile),
            output_fragment_(output_fragment),
            lane_id_(lane_id),
            warpin_id(lane_id & 31),
            warp_id(lane_id>>5){}
    
    // Compute
    __device__ __forceinline__ void TileMAC(int dimW, long col, long col1,int dimMori, long row){
        //load lhs_matrix
        at::Half lhs_fragment_half[4]={0.0,0.0,0.0,0.0};
        half2 *lhs_fragment_half2 = reinterpret_cast<half2*>(lhs_fragment_half);
        uint32_t *lhs_fragment=reinterpret_cast<uint32_t*>(lhs_fragment_half);
        if(row < dimMori){
        lhs_fragment_half2[0] = *(lhs_tile_+(warpin_id%4));
        lhs_fragment_half2[1] = *(lhs_tile_+(warpin_id%4)+4);

        // lhs_fragment_half[0]=*(lhs_tile_+((warpin_id%4)*2));
        // lhs_fragment_half[1]=*(lhs_tile_+((warpin_id%4)*2)+1);
        // lhs_fragment_half[2]=*(lhs_tile_+((warpin_id%4)*2)+8);
        // lhs_fragment_half[3]=*(lhs_tile_+((warpin_id%4)*2)+9);
        lhs_tile_=lhs_tile_+8;}

        //load rhs_matrix
        at::Half rhs_fragment_half[8]={0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        half2 *rhs_fragment_half2 = reinterpret_cast<half2*>(rhs_fragment_half);
        uint32_t *rhs_fragment = reinterpret_cast<uint32_t *>(rhs_fragment_half);
        if(col>=0){
        // rhs_fragment_half[0]=*(rhs_tile_ + ((warpin_id%4)*2));
        // rhs_fragment_half[1]=*(rhs_tile_ + ((warpin_id%4)*2) + 1);
        // rhs_fragment_half[4]=*(rhs_tile_ + ((warpin_id%4)*2) + 8);
        // rhs_fragment_half[5]=*(rhs_tile_ + ((warpin_id%4)*2) + 9);
        rhs_fragment_half2[0] = *(rhs_tile_+(warpin_id%4));
        rhs_fragment_half2[2] = *(rhs_tile_+(warpin_id%4)+4);
        }
        if((col1)>=0){
        int temp = ((col1-col)*dimW)/2;
        rhs_fragment_half2[1] = *(rhs_tile_+(warpin_id%4) + temp);
        rhs_fragment_half2[3] = *(rhs_tile_+(warpin_id%4)+4 + temp);
        // rhs_fragment_half[2]=*(rhs_tile_ + ((warpin_id%4)*2) + ((col1-col)*dimW));
        // rhs_fragment_half[3]=*(rhs_tile_ + ((warpin_id%4)*2) + 1 + ((col1-col)*dimW));
        // rhs_fragment_half[6]=*(rhs_tile_ + ((warpin_id%4)*2) + 8 + ((col1-col)*dimW));
        // rhs_fragment_half[7]=*(rhs_tile_ + ((warpin_id%4)*2) + 9 + ((col1-col)*dimW));
        }
        //更新rhs_tile
        rhs_tile_=rhs_tile_+8;
            // if(warpin_id==4&warp_id==1)
            // {
            //     half *p=reinterpret_cast<half *>(rhs_fragment);
            //     for(int i=0;i<8;i++)
            //     printf("thread_id:%d, value is:%f\n", warpin_id,__half2float(*(p+i)));
            // }
     
        //MMA 
        asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 \t"
            "{%0,%1}, \t"
            "{%2,%3,%4,%5}, \t"
            "{%6,%7}, \t"
            "{%0,%1}; ":
            "+r"(output_fragment_[0]), "+r"(output_fragment_[1]):
            "r"(rhs_fragment[0]),  "r"(rhs_fragment[1]),  "r"(rhs_fragment[2]),  "r"(rhs_fragment[3]),
            "r"(lhs_fragment[0]), "r"(lhs_fragment[1])
        ); 

    }
    

    // __device__ __forceinline__ void TileMACResidue(int residue,int dimW, long col, long col1, int dimMori,  long row){
    //     //load lhs_matrix
    //     at::Half lhs_fragment_half[4]={0.0,0.0,0.0,0.0};
    //     uint32_t *lhs_fragment=reinterpret_cast<uint32_t*>(lhs_fragment_half);
    //     if(row < dimMori){
    //      if((((warpin_id%4)*2))<residue)
    //     lhs_fragment_half[0]=*(lhs_tile_+((warpin_id%4)*2));
    //     if((((warpin_id%4)*2) + 1)<residue)
    //     lhs_fragment_half[1]=*(lhs_tile_+((warpin_id%4)*2) +1);
    //     if((((warpin_id%4)*2) + 8)<residue)
    //     lhs_fragment_half[2]=*(lhs_tile_+((warpin_id%4)*2) +8);
    //     if((((warpin_id%4)*2) + 9)<residue)
    //     lhs_fragment_half[3]=*(lhs_tile_+((warpin_id%4)*2) +9);
    //     }

    //     //load rhs_matrix
    //      at::Half rhs_fragment_half[8]={0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    //     uint32_t *rhs_fragment = reinterpret_cast<uint32_t *>(rhs_fragment_half);
    //     if(col>=0){
    //         if(((warpin_id%4)*2)<residue)
    //         rhs_fragment_half[0]=*(rhs_tile_ +((warpin_id%4)*2));
    //         if(((warpin_id%4)*2 +1)<residue)
    //         rhs_fragment_half[1]=*(rhs_tile_ +((warpin_id%4)*2) +1);
    //         if(((warpin_id%4)*2 +8)<residue)
    //         rhs_fragment_half[4]=*(rhs_tile_ +((warpin_id%4)*2) +8);
    //         if(((warpin_id%4)*2 +9)<residue)
    //         rhs_fragment_half[5]=*(rhs_tile_ +((warpin_id%4)*2) +9);
    //     }
    //     if((col1)>=0){
    //         if(((warpin_id%4)*2)<residue)
    //         rhs_fragment_half[2]=*(rhs_tile_ + ((warpin_id%4)*2) + ((col1-col)*dimW));
    //         if(((warpin_id%4)*2 +1)<residue)
    //         rhs_fragment_half[3]=*(rhs_tile_ + ((warpin_id%4)*2) + 1 + ((col1-col)*dimW));
    //         if(((warpin_id%4)*2 +8)<residue)
    //         rhs_fragment_half[6]=*(rhs_tile_ + ((warpin_id%4)*2) + 8 + ((col1-col)*dimW));
    //         if(((warpin_id%4)*2 +9)<residue)
    //         rhs_fragment_half[7]=*(rhs_tile_ + ((warpin_id%4)*2) + 9 + ((col1-col)*dimW));
    //     }
    //     //MMA 
    //     asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 \t"
    //         "{%0,%1}, \t"
    //         "{%2,%3,%4,%5}, \t"
    //         "{%6,%7}, \t"
    //         "{%0,%1}; ":
    //         "+r"(output_fragment_[0]), "+r"(output_fragment_[1]):
    //         "r"(rhs_fragment[0]),  "r"(rhs_fragment[1]),  "r"(rhs_fragment[2]),  "r"(rhs_fragment[3]),
    //         "r"(lhs_fragment[0]), "r"(lhs_fragment[1])
    //     );  
    // }

    
};



struct mmaComputeUtils_fp16_gen_cc_only{
    const half2* lhs_tile_;
    const half2* rhs_tile_;
    uint32_t* output_fragment_;
    const int lane_id_;
    const int warpin_id;
    const int warp_id;
    int dimN_index_;
    int col_;
    int row_;
    int dimW_;

    // Constructor
    __device__ __forceinline__ mmaComputeUtils_fp16_gen_cc_only(
        const half2 *lhs_tile,
        const half2*rhs_tile,
        uint32_t* output_fragment,
        int lane_id):
            lhs_tile_(lhs_tile),
            rhs_tile_(rhs_tile),
            output_fragment_(output_fragment),
            lane_id_(lane_id),
            warpin_id(lane_id & 31),
            warp_id(lane_id>>5){}
    
    // Compute
    __device__ __forceinline__ void TileMAC(int dimW, long col, long col1,int dimMori, long row, half * smem_A, half * smem_B, half * smem_ACC){
        //load lhs_matrix
        at::Half lhs_fragment_half[8]={0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0};
        half *lhs_fragment_half_temp = reinterpret_cast<half*>(lhs_fragment_half);
        half2 *lhs_fragment_half2 = reinterpret_cast<half2*>(lhs_fragment_half);
        uint32_t *lhs_fragment=reinterpret_cast<uint32_t*>(lhs_fragment_half);
        if(row < dimMori){
        lhs_fragment_half2[0] = *(lhs_tile_+(warpin_id%4));
        lhs_fragment_half2[1] = *(lhs_tile_+(warpin_id%4)+4);
        
        for (int t = 0; t < 4; ++t) 
        smem_A[warpin_id * 4 + t] = lhs_fragment_half_temp[t];

        lhs_tile_=lhs_tile_+8;}

        //load rhs_matrix
        at::Half rhs_fragment_half[8]={0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        half *rhs_fragment_half_temp = reinterpret_cast<half*>(rhs_fragment_half);
        half2 *rhs_fragment_half2 = reinterpret_cast<half2*>(rhs_fragment_half);
        uint32_t *rhs_fragment = reinterpret_cast<uint32_t *>(rhs_fragment_half);
        if(col>=0){

        rhs_fragment_half2[0] = *(rhs_tile_+(warpin_id%4));
        rhs_fragment_half2[2] = *(rhs_tile_+(warpin_id%4)+4);
        }
        if((col1)>=0){
        int temp = ((col1-col)*dimW)/2;
        rhs_fragment_half2[1] = *(rhs_tile_+(warpin_id%4) + temp);
        rhs_fragment_half2[3] = *(rhs_tile_+(warpin_id%4)+4 + temp);

        }

        for (int t = 0; t < 8; ++t) 
        smem_B[warpin_id * 8 + t] = rhs_fragment_half_temp[t];

        //更新rhs_tile
        rhs_tile_=rhs_tile_+8;
     
        __syncwarp();

        half * output_fragment = reinterpret_cast<half *>(output_fragment_);
        //开始计算MMA
        for(int mm=0; mm<4; mm++){

            // sparse_fragment1
            int peer_s = (warpin_id%4)*8 + mm; 
            #pragma unroll
            for (int t = 0; t < 4; ++t) {
                lhs_fragment_half_temp[t] = smem_A[(peer_s) * 4 + t];
                lhs_fragment_half_temp[t+4] = smem_A[(peer_s+4)  * 4 + t];
            }

            // dense_fragment peer_d+0,1,2,3
            int peer_d = (warpin_id/4)*4 + mm; 
            #pragma unroll
            for (int t = 0; t < 8; ++t) {
                rhs_fragment_half_temp[t] = smem_B[peer_d * 8 + t];
                // if(threadIdx.x==0 && blockIdx.x==0 && blockIdx.y==0) printf("%f\n", __half2float(dense_fragment[t]));

            }

            // if(threadIdx.x==0 && blockIdx.x==0 && blockIdx.y==0) printf("")
            output_fragment[0] = __hadd(output_fragment[0], __hmul(lhs_fragment_half_temp[0], rhs_fragment_half_temp[0]));
            output_fragment[0] = __hadd(output_fragment[0], __hmul(lhs_fragment_half_temp[1], rhs_fragment_half_temp[1]));
            output_fragment[0] = __hadd(output_fragment[0], __hmul(lhs_fragment_half_temp[2], rhs_fragment_half_temp[4]));
            output_fragment[0] = __hadd(output_fragment[0], __hmul(lhs_fragment_half_temp[3], rhs_fragment_half_temp[5]));

            output_fragment[1] = __hadd(output_fragment[1], __hmul(lhs_fragment_half_temp[4], rhs_fragment_half_temp[0]));
            output_fragment[1] = __hadd(output_fragment[1], __hmul(lhs_fragment_half_temp[5], rhs_fragment_half_temp[1]));
            output_fragment[1] = __hadd(output_fragment[1], __hmul(lhs_fragment_half_temp[6], rhs_fragment_half_temp[4]));
            output_fragment[1] = __hadd(output_fragment[1], __hmul(lhs_fragment_half_temp[7], rhs_fragment_half_temp[5]));


            output_fragment[2] = __hadd(output_fragment[2], __hmul(lhs_fragment_half_temp[0], rhs_fragment_half_temp[2]));
            output_fragment[2] = __hadd(output_fragment[2], __hmul(lhs_fragment_half_temp[1], rhs_fragment_half_temp[3]));
            output_fragment[2] = __hadd(output_fragment[2], __hmul(lhs_fragment_half_temp[2], rhs_fragment_half_temp[6]));
            output_fragment[2] = __hadd(output_fragment[2], __hmul(lhs_fragment_half_temp[3], rhs_fragment_half_temp[7]));

            output_fragment[3] = __hadd(output_fragment[3], __hmul(lhs_fragment_half_temp[4], rhs_fragment_half_temp[2]));
            output_fragment[3] = __hadd(output_fragment[3], __hmul(lhs_fragment_half_temp[5], rhs_fragment_half_temp[3]));
            output_fragment[3] = __hadd(output_fragment[3], __hmul(lhs_fragment_half_temp[6], rhs_fragment_half_temp[6]));
            output_fragment[3] = __hadd(output_fragment[3], __hmul(lhs_fragment_half_temp[7], rhs_fragment_half_temp[7]));
        }

        __syncwarp();


        // //MMA 
        // asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 \t"
        //     "{%0,%1}, \t"
        //     "{%2,%3,%4,%5}, \t"
        //     "{%6,%7}, \t"
        //     "{%0,%1}; ":
        //     "+r"(output_fragment_[0]), "+r"(output_fragment_[1]):
        //     "r"(rhs_fragment[0]),  "r"(rhs_fragment[1]),  "r"(rhs_fragment[2]),  "r"(rhs_fragment[3]),
        //     "r"(lhs_fragment[0]), "r"(lhs_fragment[1])
        // ); 

    }

    
};