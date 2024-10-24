import torch
from ultralytics import yolov8

def assign_device():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("The device selected is " + str(device))

    return device

def use_yolo(device):

    model = yolov8("yolov8n.yaml")
    model.to(device)

    return model
