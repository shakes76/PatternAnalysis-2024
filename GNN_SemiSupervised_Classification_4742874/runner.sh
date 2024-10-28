#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a100
#SBATCH --job-name=s4742874_gnn
#SBATCH -o s4742874_gnn.out

conda activate torch
python main.py -s -l -e 2
