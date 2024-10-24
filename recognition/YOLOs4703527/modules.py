import torch
from ultralytics import YOLO

class YOLOv8Model:
    def __init__(self, weights_path='yolov8n.pt', device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = YOLO(weights_path)
        self.model.to(self.device)
        print(f"Loaded YOLOv8 model from {weights_path} on {self.device}")
    
    def train(self, data_yaml, epochs=50, img_size=640, output_path='results'):
        """
        Train the YOLOv8 model on the specified dataset.
        :param data_yaml: Path to the YAML dataset configuration.
        :param epochs: Number of training epochs.
        :param img_size: Image size for training.
        :param output_path: Where to save training results.
        """
        print(f"Starting training for {epochs} epochs...")
        self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=img_size,
            project=output_path,
            name='yolov8_isic',
            device=self.device,
            save=True
        )
        print("Training completed. Model and results saved!")

    def save_model(self, output_path):
        """Save the trained model weights."""
        self.model.save(output_path)

def load_yolov8_model(weights_path='yolov8n.pt', device=None):
    """Helper function to initialize the YOLOv8 model."""
    return YOLOv8Model(weights_path, device)

if __name__ == "__main__":
    MODEL_WEIGHTS_PATH = r'C:\Users\raulm\OneDrive\Desktop\Yolo\yolov8n.pt'

    model = load_yolov8_model(MODEL_WEIGHTS_PATH)
   