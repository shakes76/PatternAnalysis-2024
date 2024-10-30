import torch
from ultralytics import YOLO


DATA_YML = "./data.yml"
OUTPUT = "./models"

device = 'cuda' if torch.cuda.is_available() else 'cpu' #default to gpu
model = YOLO("yolo11n.pt").to(device)

settings = {
        'data': DATA_YML,
        'epochs': 10,
        'imgsz': 640,
        'batch': -1,
        'iterations': 10
    }


if __name__ == '__main__':
    results = model.tune(**settings)