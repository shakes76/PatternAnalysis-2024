from modules import YOLO, filter_boxes
from dataset import ISICDataset
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import torch
import numpy as np

def plot_boxes(image_tensor, bounding_box):
    """
    Plots the bounding box and label on an image.

    Args:
        image_tensor (torch.Tensor): The image tensor of shape (3, 416, 416).
        bounding_box (torch.Tensor): The bounding box tensor with format [center_x, center_y, width, height, score, label1, label2].
    """
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
    """
    Predicts the bounding box and class label for an image using the model.

    Args:
        image_path (str): Path to the input image.
        model (YOLO): Trained YOLO model.
    """
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
