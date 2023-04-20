#!/bin/bash
# Job name:
#SBATCH --job-name=test_train
#
# Account:
#SBATCH --account=fc_caddatabase
#
# Partition:
#SBATCH --partition=savio3_gpu
#
# Wall clock limit:
#SBATCH --time=5:00:00
#
#SBATCH --nodes=1
#
# Number of tasks (one for each GPU desired for use case) (example):
#SBATCH --ntasks=1

module load python 3.8
#module load keras
pip install --user -r /global/home/users/ethantqiu/AdditiveParts/requirements.txt
# module load ml/tensorflow
#module list
module load cuda/9.0
#module load cuda/9.0/cudnn/7.1
module load cuda
#module --ignore_cache load
## Command(s) to run:
#python tester.py >& tester.out
python3 additive_parts/utils/tensormaker.py
