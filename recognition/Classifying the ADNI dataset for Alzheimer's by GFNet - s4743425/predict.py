"""
This file shows the example usage of the trained model for making predictions and visualising results.
"""

import torch
import os
import matplotlib.pyplot as plt
 #import the trained model architecture
from modules import GFNet 
# Import the data loader
from dataset import dataloader
import numpy as np
import torchvision.utils as vutils


## for local testing
model_path = "trained_model.pth"

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Directory to save plots
assets_dir = 'assets'
if not os.path.exists(assets_dir):
    os.makedirs(assets_dir)


# Function to load the trained model
def load_model(model_path):
    model = GFNet(
        img_size=256,
        patch_size= 16,
        embed_dim=768,
        num_classes=2,
        in_channels=3,
        drop_rate=0.5,
        depth=2,
        mlp_ratio=4.,
        drop_path_rate=0.6
        ).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    print("Model Loaded.\n")
    return model

def visualisations(images, labels, predictions, save_path):
    # The label names to be plotted
    label_names = {0: 'AD', 1: 'NC'}
    # change from tensors and plot first few images with labels and predictions
    images_np = images.cpu().numpy()
    fig = plt.figure(figsize=(10, 10))
    # plot 4 images
    for i in range(1, 5):
        # go (channels, height, width) to (height, width, channels)
        img = np.transpose(images_np[i - 1], (1, 2, 0))  
        ax = fig.add_subplot(2, 2, i)
        ax.imshow(img)
        true_label = label_names[labels[i-1].item()]
        pred_label = label_names[predictions[i-1].item()]
        ax.set_title(f"True: {true_label}, Predicted: {pred_label}")
        ax.axis('off')

    # Save plot to file
    plt.savefig(save_path)
    plt.show()

# function for making predictions of the trained model (use test dataset)
def predict(model, test_loader):
    print("Generating Predictions for test data...")
    with torch.no_grad():

        for i, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            
            # Randomly select 4 images from the batch
            # REF:Gervais, N. (2024). Random Choice with Pytorch? Stack Overflow.
            # https://stackoverflow.com/questions/59461811/random-choice-with-pytorch
            rand_indices = torch.randperm(images.size(0))[:4]
            selected_images = images[rand_indices]
            selected_labels = labels[rand_indices]
            selected_predictions = predictions[rand_indices]
             # Produce visualisations of the selected images
            visualisations(selected_images, selected_labels.cpu(), selected_predictions.cpu(), os.path.join(assets_dir, f'predictions_{i}.png'))
            # print predictions for the current batch
            print(f"Batch {i}:")
            print(f"True labels: {labels.cpu().numpy()}")
            print(f"Predicted labels: {predictions.cpu().numpy()}")
            
            if i == 0:  # Limit to one batch for demonstration, you can modify this
                break

# Main function to load the model and perform predictions
def main():
    # Load the test dataset
    (_, _, test_loader) = dataloader(batch_size=64)
    # Load trained model
    model = load_model(model_path)
    # produce predictions
    predict(model, test_loader)

if __name__ == "__main__":
    main()

