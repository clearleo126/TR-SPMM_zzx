#ifndef CONFIG_H
#define CONFIG_H

#define WARP_SIZE 32
#define tile_M 16
#define tile_K 16
#define tile_N 8

#define tile_K_sp 8

#define metadata_step 8 

#define A_tile_size tile_M*tile_K_sp


// #define DEV_ID 1


#define FLOAT2(pointer) (reinterpret_cast<float2*>(&(pointer))[0])
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
// // #define N 64 // 32 64 128

#define HALF2(pointer) (reinterpret_cast<half2*>(&(pointer))[0])


// #define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

//sparse mma
#define SPMMA16816(RD0, RD1, RD2, RD3, RA0, RA1, RB0, RB1, RC0, RC1, RC2, RC3, RM)   \
    asm volatile( \
      "mma.sp.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 " \
      "{%0,  %1,  %2,  %3}," \
      "{%4,  %5}," \
      "{%6,  %7}," \
      "{%8, %9, %10, %11}, %12, 0x1;" \
      : "=f"(RD0), "=f"(RD1), "=f"(RD2), "=f"(RD3) \
      : "r"(RA0), "r"(RA1), \
        "r"(RB0), "r"(RB1), \
        "f"(RC0), "f"(RC1), "f"(RC2), "f"(RC3), "r"(RM)) \

//     asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n" \
//                  : "=f"(RD0), "=f"(RD1), "=f"(RD2), "=f"(RD3)                                                                                \
//                  : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3), "r"(RB0), "r"(RB1), "f"(RC0), "f"(RC1), "f"(RC2), "f"(RC3))


// #define HMMA1688(RD0, RD1, RD2, RD3, RA0, RA1, RA2, RA3, RB0, RB1, RC0, RC1, RC2, RC3)                                                    \
//     asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n" \
//                  : "=f"(RD0), "=f"(RD1), "=f"(RD2), "=f"(RD3)                                                                                \
//                  : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3), "r"(RB0), "r"(RB1), "f"(RC0), "f"(RC1), "f"(RC2), "f"(RC3))

#endif