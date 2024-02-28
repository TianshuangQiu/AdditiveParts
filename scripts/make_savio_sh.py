import random
from autolab_core import gen_experiment_id

rand_id = gen_experiment_id()
PREFACE = """#!/bin/bash
# Job name:
#SBATCH --job-name=%s
#
# Account:
#SBATCH --account=fc_caddatabase
#
# Partition:
#SBATCH --partition=savio3_gpu
#
# Number of nodes:
#SBATCH --nodes=1
#
# Number of tasks (one for each GPU desired for use case) (example):
#SBATCH --ntasks=1
#
# Processors per task:
# Always at least twice the number of GPUs (savio2_gpu and GTX2080TI in savio3_gpu)
# Four times the number for TITAN and V100 in savio3_gpu and A5000 in savio4_gpu
# Eight times the number for A40 in savio3_gpu
#SBATCH --cpus-per-task=2
#
#Number of GPUs, this can be in the format of "gpu:[1-4]", or "gpu:K80:[1-4] with the type included
#SBATCH --gres=gpu:GTX2080TI:1
#
# Wall clock limit:
#SBATCH --time=30:00:00
#
## Command(s) to run (example):
module load python/3.8.8
source activate /global/scratch/users/ethantqiu/envs/3d
"""
for data in [10000, 50000, 100000]:
    for epoch in [10, 20]:
        for lr in [0.01, 0.001]:
            for batch_size in [16]:
                for nneighbor in [16, 128]:
                    for nblocks in [2, 4]:
                        for transformer_dim in [64, 128, 256]:
                            rand_id = gen_experiment_id()
                            with open(f"{data}_gridsearch_{rand_id}.sh", "w") as w:
                                w.write(PREFACE % rand_id)
                                w.write(
                                    f"python trainPCE.py TSFM_{data}_n_{nblocks}_dim_{transformer_dim} {data} {epoch} {lr} {batch_size} {nneighbor} {nblocks} {transformer_dim} -savio\n"
                                )
