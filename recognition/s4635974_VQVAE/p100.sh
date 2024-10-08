#!/bin/bash
#SBATCH --partition=p100
#SBATCH --gres=gpu:1
#SBATCH --time=0-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s4635974@student.uq.edu.au
#SBATCH --output=test.out

conda activate torch
python train.py