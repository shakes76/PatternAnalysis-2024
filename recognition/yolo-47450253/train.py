import torch
from ultralytics import YOLO


DATA_YML = "./data.yml"
OUTPUT = "./models"

SETTINGS = {
            'data': DATA_YML,
            'epochs': 75,
            'imgsz': 640,
            'workers': 5,
            'batch': -1,
            'lr0': 0.01,
            'optimizer': "AdamW",
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'project': OUTPUT,
            'save_period': 10
        }

'''
train()
    Trains a model using the given dataset (DATA_YML) and saves it to OUTPUT
    Hyperperameters can be changed in the settings dictionary
'''
def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu' #default to gpu
    model = YOLO("yolo11n.pt").to(device) 
    
    results = model.train(**SETTINGS)


if __name__ == '__main__':
    train()