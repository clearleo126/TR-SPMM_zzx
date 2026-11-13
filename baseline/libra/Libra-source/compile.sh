#!/bin/bash
# Insatll Libra
python setup.py install &&

cd ./SpMM &&
python setup.py install &&

cd .. &&

cd ./SDDMM &&
python setup.py install &&

# cd ./LibraAMD &&
# python setup.py install &&