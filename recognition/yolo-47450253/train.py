from ultralytics import YOLO
import torch

DATA_YML = "./data.yml"
OUTPUT = "./models"

device = 'cuda' if torch.cuda.is_available() else 'cpu' #default to gpu
model = YOLO("yolo11n.pt")
model = model.to(device)

settings = {
        'data': DATA_YML,
        'epochs': 75,
        'imgsz': 640,
        'project': OUTPUT,
        'save_period': 10
    }

model.train(**settings)