from modules import load_yolov8_model

# Paths
MODEL_WEIGHTS_PATH = r'C:\Users\raulm\OneDrive\Desktop\Yolo\yolov8n.pt'
YAML_CONFIG_PATH = r'C:\Users\raulm\OneDrive\Desktop\Yolo\yolov8n.yaml'
OUTPUT_PATH = r'C:\Users\raulm\OneDrive\Desktop\Yolo\results'
EPOCHS = 75
IMG_SIZE = 640

def main():
    # Load the YOLOv8 model with pre-trained weights
    model = load_yolov8_model(MODEL_WEIGHTS_PATH)

    # Prepare the overrides dictionary for training parameters
    overrides = {
        'data': YAML_CONFIG_PATH,
        'epochs': EPOCHS,
        'imgsz': IMG_SIZE,
        'batch': 16,
        'optimizer': 'AdamW',
        'lr0': 0.001,
        'momentum': 0.9,
        'weight_decay': 0.0005,
        'project': OUTPUT_PATH,
        'save_period': 10
    }

    # Start training the model
    print("Starting training...")
    model.train(**overrides)

if __name__ == "__main__":
    main()
