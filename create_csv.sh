#!/bin/bash

#SBATCH --job-name gcd4da-cop-phase0-tsm_hmdb2ucf
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=3G
#SBATCH --time 1-0
#SBATCH --partition batch_ce_ugrad
#SBATCH -x moana-y2
#SBATCH -o slurm/logs/slurm-%A-%x.out

python -u create_csv.py

exit 0