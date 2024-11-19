from ultralytics import YOLO
import torch

def yolo_model(weights_path):
    """
    gets YOLO model and enables GPU (if available)

    Parameters:
        weights_path: path to yolov7.pt file
    """
    model = YOLO(weights_path) # gets initial weights
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device) # transfers device to GPU if available
    return model