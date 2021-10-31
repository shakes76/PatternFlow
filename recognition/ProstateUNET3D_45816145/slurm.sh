#!/bin/bash
#SBATCH --job-name=Unet
#SBATCH --nodelist=c4130-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output="slurm_logs/%j.out"
#SBATCH --time=1:00:00 # 1 hour

source ~/miniconda3/bin/activate ~/tf2
python driver.py