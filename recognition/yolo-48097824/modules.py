import torch
from ultralytics import yolov8

def assign_device():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("The device selected is " + device)

    return device

def use_yolo(device):

    model = yolov8()
