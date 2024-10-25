import torch
from ultralytics import YOLO

class YOLOv8Model:
    """Class for initializing, training, and running YOLOv8."""

    def __init__(self, weights_path='yolov8n.pt', device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = YOLO(weights_path)
        self.model.to(self.device)
        print(f"Loaded YOLOv8 model from {weights_path} on {self.device}")

    def train(self, **kwargs):
        """
        Train the YOLOv8 model with the provided parameters.
        :param kwargs: Training parameters (e.g., data, epochs, batch size, etc.)
        """
        print("Starting training with the following parameters:")
        for key, value in kwargs.items():
            print(f"{key}: {value}")

        self.model.train(**kwargs)
        print("Training completed. Model and results saved!")

def load_yolov8_model(weights_path='yolov8n.pt', device=None):
    """Function to initialize the YOLOv8 model."""
    return YOLOv8Model(weights_path, device)