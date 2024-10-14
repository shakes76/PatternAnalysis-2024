#!/bin/bash
#SBATCH --partition=comp3710
#SBATCH --account=comp3710 
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s4635974@student.uq.edu.au
#SBATCH --output=_lr=0.0004_bs=32.out

conda activate torch
python train.py