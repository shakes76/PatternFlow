#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --job-name="DCGAN"
#SBATCH --mail-user=zhien.zhang@uqconnect.edu.au
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=~/45471861_StyleGAN/Output/%j.out

export PYTHONPATH=~/.local/lib/python3.6/site-packages/
python run.py