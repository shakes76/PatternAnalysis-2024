#!/bin/bash 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=1 
#SBATCH --gres=gpu:a100 
#SBATCH --job-name=train_test 
#SBATCH -o predict_test.out 

conda activate torch_env
python predict.py     
