#!/bin/bash
#SBATCH --partition=comp3710
#SBATCH --account=comp3710 
#SBATCH --gres=gpu:1
#SBATCH --time=0-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s4635974@student.uq.edu.au
#SBATCH --output=lr=test.out

conda activate torch
python train.py