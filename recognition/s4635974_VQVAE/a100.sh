#!/bin/bash
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --time=0-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s4635974@student.uq.edu.au
#SBATCH --output=testing_a100.out

conda activate torch
python train.py