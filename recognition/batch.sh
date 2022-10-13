#!/bin/bash
#SBATCH --job-name=diff256
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=vgpu20
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=michael.holmes1@uqconnect.edu.au

conda activate diff
#python3 /home/Student/s4580249/PatternFlow/recognition/train.py test1 /home/Student/s4580249/PatternFlow/recognition/AKOA_Analysis/ -i 128 -b 64 -e 200 -t 1000
python3 /home/Student/s4580249/PatternFlow/recognition/predict.py test1.pth -o /home/Student/s4580249/PatternFlow/recognition/inference/ -i 10