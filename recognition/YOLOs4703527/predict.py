import torch
from ultralytics import YOLO

# Paths
MODEL_PATH = r'C:\Users\raulm\OneDrive\Desktop\Yolo\results\yolov8_isic6\weights\best.pt'
TEST_IMAGES_PATH = r'C:\Users\raulm\OneDrive\Desktop\Yolo\test\images'

def main():
    # Load the trained YOLOv8 model
    model = YOLO(MODEL_PATH)

  
    results = model.predict(source=TEST_IMAGES_PATH, imgsz=640, conf=0.5, iou=0.5, save=False)
    return results


if __name__ == "__main__":
    main()
