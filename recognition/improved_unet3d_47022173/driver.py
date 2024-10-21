"""
This file contains the driver code to run the training or prediction with the 3D U-Net model.
It uses the argparse library to parse the command line arguments and run the appropriate function.
"""

import argparse
from utils import *
from train import train
from predict import predict

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, required=True, help="Train, debug or predict.")
    parser.add_argument('-s', '--system', type=str, required=True, help="Local or rangpur.")
    parser.add_argument('-p', '--model-path', type=str, help="Path to the model file for predict.")
    parser.add_argument('-lr', '--learning-rate', type=float, help="Learning rate for optimizer.")
    parser.add_argument('-bs', '--batch-size', type=int, help="Batch size for loader.")
    parser.add_argument('-e', '--epochs', type=int, help="Epochs to run for training.")
    parser.add_argument('-wd', '--weight-decay', type=float, help="Weight decay for optimizer.")
    parser.add_argument('-ss', '--step-size', type=int, help="Step size for scheduler.")
    parser.add_argument('-g', '--gamma', type=float, help="Gamma for scheduler.")
    args = parser.parse_args()


    # Set default parameters for system
    if args.system == "rangpur":
        images_path = "/home/groups/comp3710/HipMRI_Study_open/semantic_MRs/"
        masks_path = "/home/groups/comp3710/HipMRI_Study_open/semantic_labels_only/"
        epochs = 100
        batch_size = 5
    else:
        images_path = "./data/semantic_MRs_anon/"
        masks_path = "./data/semantic_labels_anon/"
        epochs = 10
        batch_size = 2

    if args.mode == "train" or args.mode == "debug":
        # Training parameters    
        batch_size = args.batch_size if args.batch_size else batch_size
        epochs = args.epochs if args.epochs else epochs
        lr = args.learning_rate if args.learning_rate else LR_D
        weight_decay = args.weight_decay if args.weight_decay else WD_D
        step_size = args.step_size if args.step_size else SS_D
        gamma = args.gamma if args.gamma else G_D

        train(
            args.mode,
            images_path,
            masks_path,
            lr,
            weight_decay,
            step_size,
            gamma,
            epochs,
            batch_size)
        
    elif args.mode == "predict" and args.model_path:
        predict(
            args.model_path,
            images_path, 
            masks_path)
    else:
        print("Invalid mode. Please select train or predict with model path.")
