import torch
from ultralytics import YOLO


DATA_YML = "./data.yml"
OUTPUT = "./models"


def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu' #default to gpu
    model = YOLO("yolo11n.pt").to(device)

    settings = {
            'data': DATA_YML,
            'epochs': 75,
            'imgsz': 640,
            'project': OUTPUT,
            'save_period': 10,
            'workers': 5,
            'batch': -1
        }
    results = model.train(**settings)


if __name__ == '__main__':
    train()