#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --account=comp3710 
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s4635974@student.uq.edu.au
#SBATCH --output=predict.out

conda activate torch
python predict.py