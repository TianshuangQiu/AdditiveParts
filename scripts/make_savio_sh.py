import random

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
#SBATCH --cpus-per-task=4
#
#Number of GPUs, this can be in the format of "gpu:[1-4]", or "gpu:K80:[1-4] with the type included
#SBATCH --gres=gpu:GTX2080TI:1
#
# Wall clock limit:
#SBATCH --time=24:00:00
#
## Command(s) to run (example):
module load anaconda3
conda init
source /global/home/users/ethantqiu/.bashrc
conda activate /global/scratch/users/ethantqiu/envs/3d
"""
for data in [10000, 50000, 100000]:
    for epoch in [5, 10, 15]:
        for lr in [1e-3, 1e-4, 1e-5]:
            for kernel_size in [3, 5]:
                for activation_fn in ["ReLU", "Sigmoid"]:
                    for learning_rate in [1e-3, 1e-4, 1e-5]:
                        for batch_size in [4]:
                            for dataset_type in [
                                "depth_image",
                                "rotated_depth_image",
                                "distance_field",
                            ]:
                                rand_id = f"CNN_{dataset_type}_kernel{kernel_size}_activ{activation_fn}_e{epoch}_lr{learning_rate}_b{batch_size}"
                                with open(f"{data}_gridsearch_{rand_id}.sh", "w") as w:
                                    w.write(PREFACE % rand_id)
                                    w.write(
                                        f"python scripts/train_3dcnn.py --kernel_size {kernel_size} \
                                            --epochs {epoch} --activation_fn{activation_fn} \
                                            --learning_rate {lr} --batch_size {batch_size} \
                                            --num{data} --type {dataset_type}\n"
                                    )
