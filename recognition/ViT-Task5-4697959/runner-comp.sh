#!/bin/bash
#SBATCH --partition=comp3710 
#SBATCH --account=comp3710 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a100
#SBATCH --job-name=ViT_train
#SBATCH -o Res_ViT.out
#SBATCH --time=0-12:00:00

conda activate demo

# Run the training script
python train.py

# Run the prediction script after training
python predict.py