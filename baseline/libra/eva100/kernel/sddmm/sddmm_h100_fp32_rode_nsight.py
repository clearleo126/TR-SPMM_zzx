import torch
import pandas as pd
import csv
import subprocess
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import time
import sys

current_dir = os.path.dirname(__file__)
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))

file_name = '/home/shijinliang/module/Libra-sc25/res/h100/spmm/tf32/test.csv'
data = sys.argv[1]
# start_time = time.time()
# for index, row in df.iterrows():
#     count+=1
    
#     data = [row['dataSet']]
#     with open(file_name, 'a', newline='') as csvfile:
#         csvfile.write(','.join(map(str, data)))
        
shell_command = "/home/shijinliang/module/flashsparse-ae/Baseline/RoDe/build/eval/eval_sddmm_f32_n32 " + "/home/shijinliang/module/flashsparse-ae/Baseline/RoDe/dataset/" + data + '/' + data + ".mtx >> " + file_name

subprocess.run(shell_command, shell=True)

# end_time = time.time()
# execution_time = end_time - start_time

dimN = 128
