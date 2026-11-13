#!/bin/bash

python /home/shijinliang/module/Libra/eva100/kernel/spmm/spmm_h100_fp16_test_args.py 128 &&

python /home/shijinliang/module/Libra/eva100/kernel/spmm/spmm_h100_fp16_test_args.py 256 &&
        
# python /home/shijinliang/module/Libra/eva100/kernel/spmm/spmm_h100_fp16_test_args.py 32 &&


python /home/shijinliang/module/Libra/eva100/kernel/spmm/spmm_h100_tf32_test_args.py 128 &&

python /home/shijinliang/module/Libra/eva100/kernel/spmm/spmm_h100_tf32_test_args.py 256
        
# python /home/shijinliang/module/Libra/eva100/kernel/spmm/spmm_h100_tf32_test_args.py 32