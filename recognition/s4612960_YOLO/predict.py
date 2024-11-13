from modules import YOLO, filter_boxes
from dataset import ISICDataset
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import torch
import numpy as np
from ultralytics import YOLO

MODEL_PATH = '/content/drive/MyDrive/COMP3710_YOLO/results/train4/weights/best.pt' 
TEST_IMAGES_PATH = '/content/drive/MyDrive/COMP3710_YOLO/test/images' 
OUTPUT_DIR = '/content/drive/MyDrive/COMP3710_YOLO/predictions'  

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_predictions():
    # Load the trained YOLOv8 model
    model = YOLO(MODEL_PATH)
    print("Model loaded successfully.")

    # Run predictions on the test images
    print("Running predictions on the test images...")
    results = model.predict(source=TEST_IMAGES_PATH, imgsz=640, conf=0.5, iou=0.8, save=True, project=OUTPUT_DIR, stream=True)


    # Process and print results
    for result in results:
        # Display the image path and number of detections
        print(f"Processed image: {result.path}")
        print(f"Number of detections: {len(result.boxes)}")

        # Print bounding box details
        for box in result.boxes.data:
            x1, y1, x2, y2, conf, cls = box[:6]
            print(f"Bounding Box - x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}, Confidence: {conf}, Class: {cls}")

    print(f"Predictions complete! Results are saved in: {OUTPUT_DIR}")

# Run the predictions function
if __name__ == "__main__":
    run_predictions()



"""
def plot_boxes(image_tensor, bounding_box):
    image_tensor = image_tensor.cpu().permute(1, 2, 0)  # Reshape for plotting
    fig, ax = plt.subplots()
    ax.imshow(image_tensor)

    if bounding_box is not None:
        box_coords = bounding_box.cpu()
        x, y, w, h = box_coords[0] - box_coords[2] / 2, box_coords[1] - box_coords[3] / 2, box_coords[2], box_coords[3]
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
        
        # Determine label based on probabilities
        label = "melanoma" if box_coords[5] > box_coords[6] else "seborrheic keratosis"
        
        # Add rectangle patch and label text
        ax.add_patch(rect)
        plt.text(x, y, label, bbox=dict(facecolor='red', alpha=0.5), color='white')

    plt.axis("off")
    plt.show()

def predict(image_path, model):
    # Load and preprocess the image
    image = cv2.imread(image_path)
    image = cv2.resize(image, (416, 416))
    image = torch.from_numpy(image.transpose((2, 0, 1))).float().div(255).unsqueeze(0).to(device)

    # Model prediction
    predictions = model(image)
    best_box = filter_boxes(predictions[0])

    # Display the image with the predicted bounding box
    plot_boxes(image.squeeze(0), best_box)

# Load model and weights
model = YOLO(num_classes=2)
checkpoint_path = "/content/drive/MyDrive/Uni/COMP3710/model.pt"
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# Run prediction on an image


image_path = "/path/to/your/image.jpg"  # Specify the image path here
predict(image_path, model)
"""
