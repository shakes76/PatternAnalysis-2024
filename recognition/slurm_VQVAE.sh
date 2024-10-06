#!/bin/bash
#SBATCH --job-name=s4878126_VQVAE
#--partition=comp3710
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a100
#SBATCH -o s4878126_VQVAE.out

conda init
conda activate torch
python utils.py
python modules.py
python dataset.py
# python train.py
python predict.py