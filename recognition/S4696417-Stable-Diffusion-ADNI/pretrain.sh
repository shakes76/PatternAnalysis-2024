#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=a100
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --job-name=Pretraining
#SBATCH --mail-user=l.osullivan2@student.uq.edu.au
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=END
#SBATCH -o pre_train_out.out
conda activate torch
python pre_train.py