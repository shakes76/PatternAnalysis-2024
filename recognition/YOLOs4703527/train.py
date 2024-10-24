from modules import load_yolov8_model

# Paths
MODEL_WEIGHTS_PATH = r'C:\Users\raulm\OneDrive\Desktop\Yolo\yolov8n.pt'  # Pre-trained weights
YAML_CONFIG_PATH = r'C:\Users\raulm\OneDrive\Desktop\Yolo\yolov8n.yaml'  # Dataset YAML
OUTPUT_PATH = r'C:\Users\raulm\OneDrive\Desktop\Yolo\results'  # Output directory
EPOCHS = 50
IMG_SIZE = 640

def main():
    # Load the YOLOv8 model with pre-trained weights
    model = load_yolov8_model(MODEL_WEIGHTS_PATH)

    # Start training the model
    model.train(
        data_yaml=YAML_CONFIG_PATH,
        epochs=EPOCHS,
        img_size=IMG_SIZE,
        output_path=OUTPUT_PATH
    )

if __name__ == "__main__":
    main()
