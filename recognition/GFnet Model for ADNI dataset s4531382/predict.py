"""
Make prediction and evaluate the model performance on test set.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import ADNITestDataset
from train import create_transforms, set_random_seed
from model import GlobalFilterNetwork
from functools import partial
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from PIL import Image  # Ensure PIL is imported for image handling

def main():
    # === Configuration Parameters ===
    data_path = r'C:\Users\macke\OneDrive\Desktop\COMP3710 A3\AD_NC'  # Path to the test data directory
    show_progress = True       # Whether to show progress bar (True/False)
    batch_size = 32            # Batch size for DataLoader
    device = "cuda"            # Device to run the model on ("cuda" or "cpu")
    test_seed = 0              # Random seed for reproducibility
    # ================================

    # Fix training seed and define some global variables
    set_random_seed(test_seed)
    device = device if torch.cuda.is_available() else "cpu"
    disable_tqdm = not show_progress

    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(script_dir, 'logs/GFNet')

    # Load the dataset
    test_dataset = ADNITestDataset(data_dir=data_path, transform=create_transforms(is_training=False))

    # Create DataLoader
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,          # Adjust based on your system
        pin_memory=True if device == "cuda" else False
    )

    # Create model
    model = GlobalFilterNetwork(
        image_size=210, 
        in_channels=1, 
        patch_size=14, 
        embed_dim=384, 
        depth=12, 
        mlp_ratio=4,
        normalization=partial(nn.LayerNorm, eps=1e-6)
    ).to(device)

    # Load the trained model weights
    model_path = os.path.join(log_dir, 'best_gfnet.pt')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    # Suppress the FutureWarning by specifying weights_only=True if appropriate
    # Note: Ensure that your saved model is compatible with weights_only=True
    # If not, you may need to continue using the default behavior for now
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except RuntimeError as e:
        print(f"RuntimeError while loading the model: {e}")
        print("Attempting to load with strict=False")
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)

    model.eval()

    preds_list = []
    true_list = []

    # Disable gradient computation for inference
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(tqdm(test_loader, disable=disable_tqdm, desc="Testing")):
            # data: [batch_size, 20, 1, 210, 210]
            # labels: [batch_size]
            data, labels = data.to(device), labels.float().to(device)

            batch_size_current = data.size(0)
            num_images = data.size(1)  # Should be 20

            # Reshape data to [batch_size * 20, 1, 210, 210]
            data = data.view(-1, 1, 210, 210)

            # Forward pass
            outputs = model(data)  # Expected shape: [batch_size * 20, ...]
            
            # Apply sigmoid activation
            outputs = torch.sigmoid(outputs).view(batch_size_current, num_images).cpu().numpy()

            # Aggregate predictions per patient (e.g., mean over 20 images)
            aggregated_outputs = outputs.mean(axis=1)

            # Threshold to get final predictions
            preds = (aggregated_outputs > 0.5).astype(int)

            preds_list.extend(preds)
            true_list.extend(labels.cpu().numpy().astype(int))

    preds_list = np.array(preds_list)
    true_list = np.array(true_list)

    # Calculate metrics
    accuracy = accuracy_score(true_list, preds_list)
    conf_matrix = confusion_matrix(true_list, preds_list)
    precision = precision_score(true_list, preds_list, zero_division=0)
    recall = recall_score(true_list, preds_list, zero_division=0)
    f1 = f1_score(true_list, preds_list, zero_division=0)

    # Print results
    print("\nConfusion Matrix:")
    print(f"TN\tFP\n{conf_matrix[0, 0]}\t{conf_matrix[0, 1]}")
    print(f"FN\tTP\n{conf_matrix[1, 0]}\t{conf_matrix[1, 1]}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"Test Accuracy: {accuracy * 100:.2f}%\n")

    # Interactive prompt for user to select a test patient by index
    while True:
        try:
            user_input = input(f"Enter the index of the test patient (0 to {len(test_dataset)-1}) or 'exit' to quit: ")
            if user_input.lower() == 'exit':
                print("Exiting the program.")
                break
            idx = int(user_input)
            if 0 <= idx < len(test_dataset):
                print(f"\nPatient Index: {idx}")
                print(f"Predicted Label: {preds_list[idx]}")
                print(f"Actual Label: {true_list[idx]}\n")
            else:
                print(f"Index out of range. Please enter a value between 0 and {len(test_dataset)-1}.\n")
        except ValueError:
            print("Invalid input. Please enter a valid integer index or 'exit' to quit.\n")

if __name__ == "__main__":
    main()
