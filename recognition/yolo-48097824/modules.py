import torch
from ultralytics import YOLO

def assign_device():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("The device selected is " + str(device))

    return device

def use_yolo(device):

    model = YOLO("yolov8n.yaml")
    model.to(device)

    return model
