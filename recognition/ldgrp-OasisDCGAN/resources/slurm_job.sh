#!/bin/bash
#SBATCH --cpus-per-task=2
#SBATCH --job-name=oasisdcgan
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH -w, --nodelist=c4130-2
#SBATCH --time=15:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=example@example.com
#SBATCH -o run.out

srun --gres=gpu:4 python ../driver.py --config config.json
