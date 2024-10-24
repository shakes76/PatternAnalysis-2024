from ultralytics import YOLO
import os
import shutil
import yaml
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np

def prepare_dataset():
    # Define paths
    base_dir = '/home/Student/s4671748/comp3710-project'
    dataset_dir = '/home/groups/comp3710/ISIC2018'
    images_dir = os.path.join(dataset_dir, 'ISIC2018_Task1-2_Training_Input_x2')
    masks_dir = os.path.join(dataset_dir, 'ISIC2018_Task1_Training_GroundTruth_x2')
    
    # Create necessary directories
    os.makedirs(os.path.join(base_dir, 'dataset', 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'dataset', 'images', 'val'), exist_ok=True)
    
    # Get all image files
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
    
    # Split dataset
    train_images, val_images = train_test_split(image_files, test_size=0.2, random_state=42)
    
    # Process training set
    for img_file in train_images:
        # Copy image
        shutil.copy(
            os.path.join(images_dir, img_file),
            os.path.join(base_dir, 'dataset', 'images', 'train', img_file)
        )
    
    # Process validation set
    for img_file in val_images:
        # Copy image
        shutil.copy(
            os.path.join(images_dir, img_file),
            os.path.join(base_dir, 'dataset', 'images', 'val', img_file)
        )

def create_dataset_yaml():
    yaml_content = {
        'path': '/home/Student/s4671748/comp3710-project/dataset',
        'train': 'images/train',
        'val': 'images/val',
        'names': ['lesion'],
        'nc': 1  # number of classes
    }
    
    with open('dataset.yaml', 'w') as f:
        yaml.dump(yaml_content, f)

def main():
    # Prepare dataset
    prepare_dataset()
    
    # Create dataset.yaml
    create_dataset_yaml()
    
    # Initialize YOLOv8 model
    '''model = YOLO('yolov8n-seg.pt')  # Load YOLOv8 nano segmentation model
    
    # Training configuration
    training_args = {
        'data': 'dataset.yaml',
        'epochs': 1,
        'imgsz': 640,
        'batch': 8,
        'device': 0,  # Use GPU if available
        'name': 'isic2018_run',
        'patience': 30,  # Early stopping patience
        'save': True,
        'cache': True,
        'project': 'ISIC2018'
    }
    
    # Start training
    results = model.train(**training_args)
    
    # Validate the model
    model.val()'''

if __name__ == "__main__":
    main()