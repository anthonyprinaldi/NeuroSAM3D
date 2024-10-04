#!/bin/bash

#SBATCH --account=aiconsgrp
#SBATCH --job-name=train-neurosam-dinov2-encoder
#SBATCH --nodelist=prospero
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=20
#SBATCH --mem=100GB
#SBATCH --gres=gpu:a100:2
#SBATCH --time=7-00:00:00
#SBATCH --output=train-neurosam-dinov2-encoder-%j.out

# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export TORCH_DISTRIBUTED_DEBUG=INFO

source ../data/venv/bin/activate
pwd

srun python -u train.py fit --config configs/dino.yaml --data.cache_dir "/scratch/localscratch/slurm_job.$SLURM_JOB_ID.0/"
