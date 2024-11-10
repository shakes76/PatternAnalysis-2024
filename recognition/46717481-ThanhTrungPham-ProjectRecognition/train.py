# Import required modules
from modules import YOLOSegmentation  # Custom module for YOLO segmentation tasks
import os                            # For operating system related operations
import shutil                        # For high-level file operations
import yaml                          # For reading/writing YAML configuration files
from PIL import Image               # For image processing operations
import numpy as np                  # For numerical operations

def create_dataset_yaml():
    """
    Creates a YAML configuration file for the dataset structure.
    This file tells YOLOv8 where to find the training and validation data.
    """
    yaml_content = {
        'path': '/home/Student/s4671748/comp3710-project/data',  # Root directory of dataset
        'train': 'images/train',      # Directory containing training images (relative to path)
        'val': 'images/val',          # Directory containing validation images (relative to path)
        'names': ['lesion'],          # List of class names (in this case, only 'lesion')
        'nc': 1                       # Number of classes to detect/segment
    }
    
    # Write the configuration to dataset.yaml file
    with open('dataset.yaml', 'w') as f:
        yaml.dump(yaml_content, f)

def main():
    """
    Main function that orchestrates the training process:
    1. Creates dataset configuration
    2. Initializes model
    3. Sets training parameters
    4. Executes training
    5. Evaluates the model
    """
    
    # Create the dataset configuration file
    create_dataset_yaml()
    
    # Initialize YOLOv8 model for segmentation
    # 'yolov8n-seg.pt' is the nano (smallest) version of YOLOv8 segmentation model
    model = YOLOSegmentation('yolov8n-seg.pt')
    
    # Define training arguments/hyperparameters
    training_args = {
        'data': 'dataset.yaml',         # Path to dataset configuration file
        'epochs': 1,                    # Number of training epochs
        'imgsz': 640,                   # Input image size
        'batch': 4,                     # Batch size for training
        'device': 0,                    # GPU device index (0 = first GPU)
        'name': 'isic2018_run_victor',  # Name of the training run
        'save': False,                  # Whether to save training results
        'cache': False,                 # Whether to cache images in memory
    }
    
    # Start the training process using defined parameters
    results = model.train(training_args)
    
    # Evaluate the trained model on validation set
    model.evaluate()

# Standard Python idiom to ensure that the main() function is only run
# if the script is executed directly (not imported as a module)
if __name__ == "__main__":
    main()