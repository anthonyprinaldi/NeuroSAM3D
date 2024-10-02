#!/bin/bash

#SBATCH --account=aiconsgrp
#SBATCH --job-name=prepare-data
#SBATCH --nodelist=curie
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=02-00:00:00
#SBATCH --output=prepare-data-%j.out

source ../data/venv/bin/activate
pwd
python -u data.py -dt Tr
python -u data.py -dt Val
python -u data.py -dt Ts