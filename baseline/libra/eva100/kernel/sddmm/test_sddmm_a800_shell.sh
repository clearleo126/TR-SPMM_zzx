#!/bin/bash

# python /home/shijinliang/module/Libra/eva100/kernel/sddmm/sddmm_a800_fp16_test_args.py 32 &&

# python /home/shijinliang/module/Libra/eva100/kernel/sddmm/sddmm_a800_fp16_test_args.py 128



python /home/shijinliang/module/Libra/eva100/kernel/sddmm/sddmm_a800_tf32_test_args.py 32 &&

python /home/shijinliang/module/Libra/eva100/kernel/sddmm/sddmm_a800_tf32_test_args.py 128
        