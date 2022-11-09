#!/bin/bash

#SBATCH --time=24:00:00   # walltime limit (HH:MM:SS)
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=1   # 1 processor core(s) per node 
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu    # gpu node(s)
#SBATCH --job-name="transformer"
#SBATCH --mail-user=pritish.roy@uq.edu.au   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output="slurm_perceiver_output.txt" # job standard output file (%j replaced by job id)
#SBATCH --error="slurm_perceiver_error.txt" # job standard error file (%j replaced by job id)

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
conda activate tf
python main.py
