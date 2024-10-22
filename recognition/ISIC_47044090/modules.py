from ultralytics import YOLO
import torch

def yolo_model():
    """
    gets YOLO model and enables GPU (if available)
    """
    model = YOLO("yolov7.pt") # gets initial weights
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device) # transfers device to GPU if available
    return model