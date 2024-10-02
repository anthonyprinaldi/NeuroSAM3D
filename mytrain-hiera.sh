#!/bin/bash

#SBATCH --account=aiconsgrp
#SBATCH --job-name=train-neurosam-hiera-encoder
#SBATCH --nodelist=prospero
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=20
#SBATCH --mem=100GB
#SBATCH --gres=gpu:a100:2
#SBATCH --time=07-00:00:00
#SBATCH --output=train-neurosam-hiera-encoder-%j.out

source ../data/venv/bin/activate
pwd

srun python -u train.py fit --config configs/hiera.yaml
