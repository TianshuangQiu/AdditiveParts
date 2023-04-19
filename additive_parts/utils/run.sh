#!/bin/bash
# Job name:
#SBATCH --job-name=pointcloudify
#
# Account:
#SBATCH --account=fc_caddatabase
#
# Partition:
#SBATCH --partition=savio2_bigmem
#
# Wall clock limit:
#SBATCH --time=5:00:00
#
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#
# Number of tasks (one for each GPU desired for use case) (example):
#SBATCH --ntasks=1

#module load python #/3.7
#module load keras
#pip install --user keras
# module load ml/tensorflow
#module list
#module load cuda/9.0
#module load cuda/9.0/cudnn/7.1
#module load cuda
#module --ignore_cache load
## Command(s) to run:
#python tester.py >& tester.out
conda activate /global/users/scratch/ethantqiu/print
python3 utils/tensormaker.py
