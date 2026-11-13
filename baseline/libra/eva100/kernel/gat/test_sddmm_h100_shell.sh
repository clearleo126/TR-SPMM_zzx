#!/bin/bash

python /home/shijinliang/module/Libra/eva100/kernel/gat/sddmm_h100_tf32_test_args.py 32 &&


python /home/shijinliang/module/Libra/eva100/kernel/gat/sddmm_h100_tf32_test_args.py 64 &&

        
python /home/shijinliang/module/Libra/eva100/kernel/gat/sddmm_h100_tf32_test_args.py 128 &&


python /home/shijinliang/module/Libra/eva100/kernel/gat/sddmm_h100_fp16_test_args.py 32 &&


python /home/shijinliang/module/Libra/eva100/kernel/gat/sddmm_h100_fp16_test_args.py 64 &&

        
python /home/shijinliang/module/Libra/eva100/kernel/gat/sddmm_h100_fp16_test_args.py 128

