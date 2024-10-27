"""
File: main.py
Description: Entry point for GNN classification with CLI arguments
Course: COMP3710 Pattern Recognition
Author: Liam Mulhern (S4742847)
Date: 26/10/2024
"""

import argparse

parser = argparse.ArgumentParser(description='COMP3710 Pattern Recognition: Graph Neural Network')

parser.add_argument(
    '--load',
    '-l',
    action='store_true',
    help="Load the specified model from ./models/<model_name>.pth if it exists"
)

parser.add_argument(
    '--inference',
    '-i',
    type=int,
    default=-1,
    help="Run inference on the specified model from ./models/<model_name>.pth if it exists with the test data from the index"
)

parser.add_argument(
    '--display',
    '-d',
    action='store_true',
    help='Display the csv data for the specified model using matplotlib'
)

parser.add_argument(
    '--epochs',
    '-e',
    type=int,
    default=100,
    help='Number of epochs to train the model for'
)

args = parser.parse_args()

if not args.load:
    if input("WARNING: Overwriting model [Y/n] ") != 'Y':
        exit()


