import torch
from modules import LesionDetectionModel
from dataset import ISICDataset
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2
import numpy as np


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_CHECKPOINT = 'model_checkpoints/model_epoch_best.pth'  
IOU_THRESHOLD = 0.8
CONFIDENCE_THRESHOLD = 0.25

#Load the trained model
model = LesionDetectionModel(model_weights=MODEL_CHECKPOINT, device=DEVICE).model
model.eval()

def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2
    xi1, yi1 = max(x1, x1g), max(y1, y1g)
    xi2, yi2 = min(x2, x2g), min(y2, y2g)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area, box2_area = (x2 - x1) * (y2 - y1), (x2g - x1g) * (y2g - y1g)
    return inter_area / (box1_area + box2_area - inter_area)

#Define transformations for test images
test_transforms = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_image(image_path):
    """
    Run inference on a single image and return the detections.

    Parameters:
        image_path (str): Path to the test image.

    Returns:
        list of dicts: Detected bounding boxes with confidence and class labels.
    """
    #Loading and preprocessing the image
    image = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  #Converting to RGB
    img_transformed = test_transforms(Image.fromarray(img_rgb)).unsqueeze(0).to(DEVICE)  #Adding batch dimension

    #Running model inference
    with torch.no_grad():
        predictions = model(img_transformed)[0]
    
    #Applying non-maximum suppression to filter out overlapping boxes
    detections = process_detections(predictions, image.shape[:2], conf_thres=CONFIDENCE_THRESHOLD, iou_thres=IOU_THRESHOLD)
    return detections

def visualize_predictions(image_path, detections):
    """
    Visualize the bounding boxes on the image.

    Parameters:
        image_path (str): Path to the image.
        detections (list of dicts): List of detected bounding boxes with confidence and class labels.
    """
    #Loading the image for visualization
    image = cv2.imread(image_path)
    for det in detections:
        x_min, y_min, x_max, y_max = det['coordinates']
        confidence = det['confidence']
        class_id = det['class']
        
        #Drawing bounding box and label on the image
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        label = f"Class: {class_id}, Conf: {confidence:.2f}"
        cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    #Displaying the image with bounding boxes
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

#Running predictions and visualize
test_image_path = 'path/to/test/image.jpg'  #Replacing with the path to your test image
detections = predict_image(test_image_path)
visualize_predictions(test_image_path, detections)
