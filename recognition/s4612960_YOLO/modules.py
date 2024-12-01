import torch 
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print("cuda")
if not torch.cuda.is_available():
    print("cpu")  
    
class YOLOv8Model:
    def __init__(self, weights_path='yolov8n.pt', device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = YOLO(weights_path)
        self.model.to(self.device)
        print(f"Model load completed")

    def train(self, **kwargs):
        self.model.train(**kwargs)
        print("Finished training")

def load_yolov8_model(weights_path='yolov8n.pt', device=None):
    return YOLOv8Model(weights_path, device)
    
