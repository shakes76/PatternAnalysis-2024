"""
Used to train, validate, test and save a model. 
Option to just test or perform all processes.

Usage example: python train.py -t eval -w yolo/best_tuned.pt | to run testing/evaluation on a model with trained weights.

@author Ewan Trafford
"""

import modules
import dataset
import argparse

# trains the model and validates it
def run_training(weights_path):
    """
    Trains an instance of a YOLOv11 model with weights specified. 
    Doing so will save a new train folder containing weights, plots and validation results.

    Args:
        weights_path (string): The absolute or relative path to the .pt file containing model weights.
    """

    model = modules.YOLOv11(weights_path)
    results = model.train(data = "Data/lesion_detection.yaml", epochs = 50, save = True)
    # Do note that when training finishes it also returns all the metrics found in run_testing

# evaluates model performance 
def run_testing(weights_path):
    """
    Evaluates or tests an instance of a YOLOv11 model with weights specified. 
    Will calculate mAP75 as well as plots and results

    Args:
        weights_path (string): The absolute or relative path to the .pt file containing model weights.
    """

    model = modules.YOLOv11(weights_path)
    metrics = model.val(plots = True, split = 'test', max_det = 1)
    print("The mean average precision at IoU threshold 0.75 is: " + str(metrics.box.map75)) 
    # Mean average precision at IoU threshold 0.75, indicative of model performance near IoU requirement of 0.8


def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Train or Evaluate the model.')
    
    # Adding command-line arguments
    parser.add_argument('-t', choices=['train', 'eval'], required=True,
                        help="Specify whether to train or evaluate the model.")
    parser.add_argument('-w', required=True,
                        help="Path to the model weights.")
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Access the arguments
    task = args.t
    weights_path = args.w
    
    # Print the arguments for verification
    print(f"Task: {task}")
    print(f"Weights Path: {weights_path}")
    
    # No need to run both as run_training does everything in run_testing when training finishes.
    if task == 'train':
        run_training(weights_path)
        pass
    elif task == 'eval':
        run_testing(weights_path)
        pass

if __name__ == '__main__':
    main()
