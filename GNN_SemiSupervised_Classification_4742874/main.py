"""
File: main.py
Description: Entry point for GNN classification with CLI arguments
Course: COMP3710 Pattern Recognition
Author: Liam Mulhern (S4742847)
Date: 26/10/2024
"""

import argparse

import train

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='COMP3710 Pattern Recognition: Graph Neural Network')

    parser.add_argument(
        '--load',
        '-l',
        action='store_true',
        help="Load the specified model from ./models/<model_name>.pth if it exists"
    )

    parser.add_argument(
        '--save',
        '-s',
        action='store_true',
        help="Save the specified model to ./models/<model_name>.pth if it exists"
    )

    parser.add_argument(
        '--inference',
        '-i',
        type=int,
        default=-1,
        help="Run inference on the specified model from ./models/<model_name>.pth if it exists with the test data from the index"
    )

    parser.add_argument(
        '--epochs',
        '-e',
        type=int,
        default=100,
        help='Number of epochs to train the model for'
    )

    parser.add_argument(
        '--learning_rate',
        '-lr',
        type=float,
        default=0.1,
        help='The learning rate to train the model with'
    )

    parser.add_argument(
        '--batch_size',
        '-b',
        type=int,
        default=200,
        help='The dataloader batch size'
    )

    parser.add_argument(
        '--train',
        '-t',
        action='store_true',
        help="Train the specified model from ./models/<model_name>.pth if it exists"
    )

    parser.add_argument(
        '--display',
        '-d',
        action='store_true',
        help="Train the specified model from ./models/<model_name>.pth if it exists"
    )

    args = parser.parse_args()

    if args.train:
        train.run_gnn_training(
            epochs=args.epochs ,
            batch_size=args.batch_size ,
            learning_rate=args.learning_rate,
            is_load=args.load,
            is_save=args.save
        )
