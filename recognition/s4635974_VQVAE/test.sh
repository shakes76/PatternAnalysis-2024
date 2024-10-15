#!/bin/bash
#SBATCH --partition=a100-test
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s4635974@student.uq.edu.au
#SBATCH --output=predict_final.out

conda activate torch
python predict.py