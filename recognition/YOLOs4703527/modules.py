import torch
from ultralytics import YOLO

class YOLOv8Model:
    def __init__(self, weights_path='yolov8n.pt', device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = YOLO(weights_path)
        self.model.to(self.device)
        print(f"Loaded YOLOv8 model from {weights_path} on {self.device}")

def load_yolov8_model(weights_path='yolov8n.pt', device=None):
    """Helper function to initialize the YOLOv8 model."""
    return YOLOv8Model(weights_path, device)

if __name__ == "__main__":
    MODEL_WEIGHTS_PATH = r'C:\Users\raulm\OneDrive\Desktop\Yolo\yolov8n.pt'

    model = load_yolov8_model(MODEL_WEIGHTS_PATH)
   