#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=16G

#SBATCH --job-name="OAI AKOA Knee Perceiver"
#SBATCH --mail-user=s4589325@uqconnect.edu.au
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH -o DCGANs-celeba.txt
#SBATCH -e DCGANs-celeba.err

nvidia-smi

source ../../../miniconda3/bin/activate s4589325

python3 ./Driver_Script.py
