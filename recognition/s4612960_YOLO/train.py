import torch
import torch.nn as nn
import torch.nn.functional as F 
import time
from dataset import *
from modules import *


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print("cuda")
if not torch.cuda.is_available():
    print("cpu")  



# hyperparameters
epochs = 31
image_size = 640
batch_size = 16

#Train data - change directories as needed
MODEL_WEIGHTS_PATH = r'/content/drive/MyDrive/COMP3710_YOLO/yolov8n.pt'
YAML_CONFIG_PATH = r'/content/drive/MyDrive/COMP3710_YOLO/yolov8n.yaml'
OUTPUT_PATH = r'/content/drive/MyDrive/COMP3710_YOLO/results'

def main():
    # Load the YOLOv8 model with pre-trained weights
    model = load_yolov8_model(MODEL_WEIGHTS_PATH)

    # Prepare the overrides dictionary for training parameters
    overrides = {
        'data': YAML_CONFIG_PATH,
        'epochs': epochs,
        'imgsz': image_size,
        'batch': batch_size,
        'optimizer': 'AdamW',
        'lr0': 0.001,
        'momentum': 0.9,
        'weight_decay': 0.0005,
        'project': OUTPUT_PATH,
        'save_period': 10
    }

    # Start training the model
    print("Starting training...")
    model.train(**overrides)

if __name__ == "__main__":
    main()
