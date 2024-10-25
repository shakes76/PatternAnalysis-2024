import torch
from ultralytics import YOLO

# Paths
MODEL_PATH = r'C:\Users\raulm\OneDrive\Desktop\Yolo\results\Train3\weights\best.pt'  # Trained model path
TEST_IMAGES_PATH = r'C:\Users\raulm\OneDrive\Desktop\Yolo\test\images'  # Directory with test images
DATA_PATH = r'C:\Users\raulm\OneDrive\Desktop\Yolo\yolov8n.yaml'  # Dataset YAML with metadata

def main():
    # Load the trained YOLOv8 model
    model = YOLO(MODEL_PATH)

    # Evaluate the model on the test dataset with IoU threshold at 0.80
    print("Evaluating model on the test dataset...")
    metrics = model.val(data=DATA_PATH, split='test', iou=0.8)  # Perform validation


    # Run predictions on the test images
    print("\nRunning predictions on the test set")
    results = model.predict(source=TEST_IMAGES_PATH, imgsz=640, conf=0.5, iou=0.8, save=False, visualize=True)

    iou = 0
    total_boxes = 0

    # Loop through results to calculate IoU statistics
    for result in results:
        if result.boxes:
            for box in result.boxes.data: # Access box data correctly
                iou += box[4]
                total_boxes += 1

    # Calculate and print the proportion of boxes with IoU >= 0.80


    print(f"IoU avg: {iou/total_boxes}")



if __name__ == "__main__":
    main()
