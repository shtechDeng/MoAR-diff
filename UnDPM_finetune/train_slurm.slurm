#!/bin/bash
#SBATCH -J finetuneDPM_with_T2
#SBATCH -p bme_a10040g
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH -N 1
#SBATCH -t 5-0:00:00
#SBATCH --output=slurm_out/finetuneDPM_with_T2.out
#SBATCH --error=slurm_out/finetuneDPM_with_T2.err

hostname
date
nvidia-smi

python main.py --config bcp.yml --exp exp --doc finetuneDPM_with_T2 --resume_training --ni

date