import os
import torch
import torch.nn as nn
from modules import GFNet
from dataset import get_datasets, get_dataloaders
from train import train_model, evaluate_model

def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Hyperparameters
    num_epochs = 30
    batch_size = 128
    learning_rate = 1e-4
    weight_decay = 1e-5

    # Data directory
    data_dir = '/home/groups/comp3710/ADNI/AD_NC'

    print("\n=== Preparing Datasets ===")
    try:
        # Prepare datasets and dataloaders
        train_dataset, val_dataset, test_dataset = get_datasets(data_dir, val_split=0.15)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred while loading datasets: {e}")
        return

    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Test set size: {len(test_dataset)}")
    print("Datasets prepared.\n")

    print("=== Initializing DataLoaders ===")
    dataloaders = get_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=batch_size)
    print("DataLoaders initialized.\n")

    # Print class to index mapping
    print("=== Class to Index Mapping ===")
    # Access class_to_idx from the underlying dataset of train_dataset
    if hasattr(train_dataset.dataset, 'class_to_idx'):
        class_to_idx = train_dataset.dataset.class_to_idx
    else:
        # Handle cases where train_dataset.dataset might not have class_to_idx directly
        try:
            class_to_idx = train_dataset.dataset.datasets[0].class_to_idx
        except AttributeError:
            class_to_idx = "Unavailable"
    print(class_to_idx)
    print()

    print("=== Initializing Model ===")
    # Initialize the model
    model = GFNet(
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=2,
        embed_dim=768,
        depth=12,
        mlp_ratio=4.,
        drop_rate=0.,
        drop_path_rate=0.,
        norm_layer=None,
        dropcls=0
    )
    model = model.to(device)
    print("Model initialized.\n")

   # Define loss function and optimizer
    print("=== Setting Up Loss Function and Optimizer ===")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    print("Loss function and optimizer set.\n")

    print("=== Starting Training ===")
    # Train the model
    model, history = train_model(model, dataloaders, criterion, optimizer, num_epochs=num_epochs, device=device)
    print("Training completed.\n")

    print("=== Evaluating Model on Test Set ===")
    # Evaluate the model on test set
    test_loader = dataloaders[2]
    test_accuracy = evaluate_model(model, test_loader, device=device)
    print(f'Final Test Accuracy: {test_accuracy:.4f}\n')

    print("=== Training Process Completed Successfully ===")


if __name__ == '__main__':
    main()
