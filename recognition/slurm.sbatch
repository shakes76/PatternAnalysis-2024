#!/bin/bash
#SBATCH --partition=a100
#SBATCH --gres=gpu:1

# set name of job
#SBATCH --job-name=cifar_rf

# total runtime, enable to use GPUs with max times
#SBATCH --time=1-00:00:00 #1 days

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=<YOUR EMAIL HERE>

# output file
#SBATCH --output=cifar_rf_1.out

# load modules

#conda init
source /home/Staff/${USER}/.bashrc

#Conda envs
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
#conda activate conda-jax
conda activate conda-torch

#go to data directory
cd Dev/deep-compute/projects/receptive_fields/

#print python version
python --version

#run script
python test_cifar_complex.py