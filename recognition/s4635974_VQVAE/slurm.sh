#!/bin/bash
#SBATCH --partition=a100
#SBATCH --gres=gpu:1

conda activate torch
python dataset.py