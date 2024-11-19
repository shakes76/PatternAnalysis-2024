# Import required modules
from modules import YOLOSegmentation  # Custom YOLO segmentation module
import random                         # For generating random colors
import cv2                           # OpenCV for image processing
import numpy as np                   # For numerical operations

# Initialize the YOLO model with trained weights
# 'best.pt' contains the weights from the best performing epoch during training
model = YOLOSegmentation("runs/segment/train2/weights/best.pt")

# Load the input image for prediction
# This is reading a specific image from the training dataset
img = cv2.imread("data/images/train/ISIC_0015071.jpg")

# Set confidence threshold for predictions
# Model will only return predictions with confidence > 0.4
conf = 0.4

# Perform prediction on the image
# results will contain detected objects, their bounding boxes, and segmentation masks
results = model.predict(img, conf)

# Generate random RGB color for visualization
# Creates a list of 3 random integers between 0-255 for RGB values
color = random.choices(range(256), k=3)

# Process each prediction result
for result in results:
    # Iterate through corresponding masks and bounding boxes
    for mask, box in zip(result.masks.segments, result.boxes):
        # Convert mask coordinates to image coordinates:
        # 1. mask contains normalized coordinates (0-1)
        # 2. multiply by bounding box width to get actual pixel coordinates
        # 3. convert to integer coordinates for drawing
        points = np.int32([np.float64(mask) * box.xywh.numpy()[0][2]])
        
        # Draw the segmentation mask on the image
        # fillPoly fills the area defined by points with the random color
        cv2.fillPoly(img, points, color)

# Save the annotated image
# The output shows the original image with colored segmentation masks
cv2.imwrite("prediction_test.jpg", img)