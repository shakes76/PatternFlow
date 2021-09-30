#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=16G

#SBATCH --job-name="CelebA Image Generation DCGANs"
#SBATCH --mail-user=username@uqconnect.edu.au
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH -o DCGANs-celeba.txt
#SBATCH -e DCGANs-celeba.err

nvidia-smi
export PYTHONPATH=~/.local/lib/python3.6/site-packages
python3 ./dcgan_celeba.py
