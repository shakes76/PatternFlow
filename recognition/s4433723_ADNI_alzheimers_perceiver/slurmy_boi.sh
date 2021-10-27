#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=akoa_perceiver
#SBATCH -o log.out
#SBATCH -e log.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s4433723@student.uq.edu.au

#activate conda environment
source /home/Student/${USER}/.bashrc
conda activate perceiver

#run application
srun python perceiver.py

#deactivate conda environment just incase
conda deactivate