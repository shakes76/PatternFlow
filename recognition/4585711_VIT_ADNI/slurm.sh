#!/bin/bash
#SBATCH --job-name=train
#SBATCH --partition=gpu
#SBATCH --mail-type=ALL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=m.r.harris@uqconnect.edu.au    # Where to send mail	
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8GB
#SBATCH --constraint=V100
#SBATCH --time=0-03:00 # time (D-HH:MM)

eval "$(conda shell.bash hook)"

module load cuda

conda activate tf

python ./train.py