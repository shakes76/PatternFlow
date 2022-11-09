#!/bin/bash
time=0-00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --job-name="Test Goliath Paris"
#SBATCH --mail-user=sen.qian@uq.net.au
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH —output=output_dir/%j.out

module load tensorflow/2.1.0
nvcc -V
conda info --envs
conda create --prefix=/home/Student/envs/s4574278 python=3.8
conda activate /home/Student/envs/s4574278
conda env update --file tools.yml
cd /home/Student/s4574278
python driver.py