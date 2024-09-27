#!/bin/bash
#SBATCH --time=1-00:00:00 # 1 day
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a100
#SBATCH --job-name=antis_transformer_eats_brains
#SBATCH --mail-user=s4753820@uq.edu.au 
#SBATCH -o gan_run.out
conda activate torch
python train.py
