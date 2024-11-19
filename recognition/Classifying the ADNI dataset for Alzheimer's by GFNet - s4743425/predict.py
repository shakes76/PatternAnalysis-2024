"""
This file shows the example usage of the trained model for making predictions and visualising results.
"""

import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from modules import GFNet 
from dataset import *
from train import test
import numpy as np
import random
import argparse
from functools import partial


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to load the trained model
def load_model(model_path):
    model = GFNet(
        img_size=256,
        patch_size= 16,
        embed_dim=512,
        num_classes=2,
        in_channels=1,
        drop_rate=0.5,
        depth=19,
        mlp_ratio=4.,
        drop_path_rate=0.15,
        norm_layer=partial(nn.LayerNorm, eps=1e-6)
        ).to(device)
    model.load_state_dict(torch.load(model_path), strict=False)
    model.eval()  # Set the model to evaluation mode
    print("Model Loaded.\n")
    return model

def visualisations(images, labels, predictions, save_path):
    # The label names to be plotted
    label_names = {0: 'AD', 1: 'NC'}
    # change from tensors and plot first few images with labels and predictions
    images_np = images.cpu().numpy()
    fig = plt.figure(figsize=(10, 10))

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
def predict(model, test_dataset, output_dir):
    print("Generating Predictions for test data...")
    with torch.no_grad():

        # Randomly sample 4 indices from the entire test set
        rand_indices = random.sample(range(len(test_dataset)), 4)
        selected_samples = [test_dataset[i] for i in rand_indices]

        selected_images = torch.stack([sample[0] for sample in selected_samples])  # Stack images
        selected_labels = torch.tensor([sample[1] for sample in selected_samples])  # Stack labels

        # Move images to device
        selected_images = selected_images.to(device)
        
        outputs = model(selected_images)
        _, selected_predictions = torch.max(outputs, 1)

        # Ensure the output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        saved_path = os.path.join(output_dir, 'random_test_predictions.png')

        visualisations(selected_images, selected_labels.cpu(), selected_predictions.cpu(), saved_path)
        
        # Print selected indices and predictions
        print(f"Randomly selected indices: {rand_indices}")
        print(f"True labels: {selected_labels.cpu().numpy()}")
        print(f"Predicted labels: {selected_predictions.cpu().numpy()}")

# Main function to load the model and perform predictions
def main():

    parser = argparse.ArgumentParser(description="Prediction for Alzheimer's Disease classification")
    
    parser.add_argument('--model-path', type=str, required=False, default="trained_model.pth",
                        help="Path to the trained model file (default: 'trained_model.pth')")
    parser.add_argument('--output-dir', type=str, required=False, default="prediction_outputs",
                        help="Directory to save the prediction images (default: 'prediction_outputs')")

    # parse the arguments
    args = parser.parse_args()

    test_loader, test_dataset = test_dataloader(batch_size=32)

    model = load_model(args.model_path)
    # Call test function from train.py
    criterion = nn.CrossEntropyLoss()  # Loss function
    test(device, args.output_dir, model, criterion, test_loader)

    # produce predictions
    predict(model, test_dataset, args.output_dir)

if __name__ == "__main__":
    main()

