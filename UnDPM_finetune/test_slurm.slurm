#!/bin/bash
#SBATCH -J inference
#SBATCH -p bme_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -N 1
#SBATCH -t 1-0:00:00
#SBATCH --output=slurm_out/inference.out
#SBATCH --error=slurm_out/inference.err

hostname
date
nvidia-smi

python main.py --config inference.yml --exp exp --inference_all --fid --timesteps 50 --eta 0 --seed 46  --ni

date