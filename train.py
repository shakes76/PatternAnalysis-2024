from ultralytics import YOLO
import os
import shutil
import yaml
from PIL import Image
import numpy as np

def create_dataset_yaml():
    yaml_content = {
        'path': '/home/Student/s4671748/comp3710-project/data',
        'train': 'images/train',
        'val': 'images/val',
        'names': ['lesion'],
        'nc': 1  # number of classes
    }
    
    with open('dataset.yaml', 'w') as f:
        yaml.dump(yaml_content, f)

def main():
    
    # Create dataset.yaml
    create_dataset_yaml()
    
    # Initialize YOLOv8 model
    model = YOLO('yolov8n-seg.pt')  # Load YOLOv8 nano segmentation model
    
    # Training configuration
    training_args = {
        'data': 'dataset.yaml',
        'epochs': 80,
        'imgsz': 640,
        'batch': 8,
        'device': 0,  # Use GPU if available
        'name': 'isic2018_run_victor',
        'save': True,
        'cache': True,
    }
    
    # Start training
    results = model.train(**training_args)
    
    # Validate the model
    model.val()

if __name__ == "__main__":
    main()