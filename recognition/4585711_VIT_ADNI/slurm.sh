#!/bin/bash
#SBATCH --job-name=train
#SBATCH --partition=gpu
#SBATCH --mail-type=ALL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=m.r.harris@uqconnect.edu.au    # Where to send mail	
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8GB
#SBATCH --time=0-03:00 # time (D-HH:MM)

__conda_setup="$('/home/s4585711/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/s4585711/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/home/s4585711/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/s4585711/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup

module load cuda

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

conda activate tf

python train.py