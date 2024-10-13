#!/bin/bash
#SBATCH --partition=comp3710
#SBATCH --account=comp3710 
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s4635974@student.uq.edu.au
#SBATCH --output=early_stopping_lr=0.002.out

conda activate torch
python train.py