#!/bin/bash
# Job name:
#SBATCH --job-name=test3
#SBATCH -o test3.out
# Partition:
#SBATCH --partition=mtech
#SBATCH --nodes=1
#SBATCH --ntasks=1
## Processors per task:
#SBATCH --cpus-per-task=8
#
#SBATCH --gres=gpu:1 ##Define number of GPUs
#
## Command(s) to run (example):
module load python/3.10.pytorch
# mpirun pip install numpy
# mpirun pip3 install efficientnet_pytorch --user
# mpirun pip3 install optuna
#mpirun pip3 install scikit-image --user
mpirun python3 /csehome/m23mac008/cvproject/draft.py >> test3.out