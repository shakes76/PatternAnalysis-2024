# train.py

import torch
from torch.utils.data import DataLoader
from dataset import ISICDataset, malig_aug, benign_aug
from modules import SiameseNN
import pandas as pd
import os
from pytorch_metric_learning.losses import ContrastiveLoss
from pytorch_metric_learning.reducers import AvgNonZeroReducer
from pytorch_metric_learning.distances import LpDistance
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau


def load_data(excel):
    df = pd.read_csv(excel)
    df = df.drop(columns=['Unnamed: 0', 'patient_id'])
    return df



def siamese_train():
    # Paths
    current_dir = os.getcwd()
    print("Working dir", current_dir)
    excel = os.path.join(current_dir, 'dataset', 'train-metadata.csv')
    images = os.path.join(current_dir, 'dataset', 'train-image', 'image')
    df = load_data(excel=excel)

    # Load data
    train_df, val_df = train_test_split (
        df,test_size=0.2, stratify=df['target'], random_state=42
    )
 

    # Initialize training and validation datasets
    train_dataset = ISICDataset(
        df = train_df,
        images_dir=images,
        transform_benign=benign_aug,
        transform_malignant=malig_aug,
        augment_ratio=0.5  # Adjust based on your needs
    )

    val_dataset = ISICDataset(
        df=val_df,
        images_dir=images,
        transform_benign=benign_aug,
        transform_malignant=malig_aug,
        augment_ratio=0.0  # No augmentation for validation
    )

    # Initialize DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32,
        shuffle=True,  # Shuffling training data
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=32,
        shuffle=False,  # No shuffling for validation data
        num_workers=4,
        pin_memory=True
    )

    # Initialize model and move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SiameseNN(embedding_dim=256).to(device)

    # Initialize Contrastive Loss from PyTorch Metric Learning
    contrastive_loss = ContrastiveLoss(
        pos_margin=0,
        neg_margin=1,
        distance=LpDistance(normalize_embeddings=True, p=2, power=1),
        reducer=AvgNonZeroReducer(),
    )
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    #scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, threshold=0.01, verbose=True)

    # Training parameters
    epochs = 50
    best_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_loss = 0.0

        for images, labels  in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} Training"):
            images, labels = images.to(device), labels.to(device)
            # Forward pass
            embeddings = model(images)

            loss = contrastive_loss(embeddings, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} Validating"):
                images, labels = images.to(device), labels.to(device)

                # Forward pass
                embeddings = model(images)

                loss = contrastive_loss(embeddings, labels)

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")
        # Step the scheduler
        scheduler.step(avg_val_loss)
        # Monitor learning rate
        for idx, param_group in enumerate(optimizer.param_groups):
            print(f"Learning rate for param group {idx}: {param_group['lr']}")

        # save current pest model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), 'siamese_resnet18_best.pth')
            print("Validation loss decreased. Saving model.")
        else:
            print("No improvement in validation loss.")

    # Save the final model
    torch.save(model.state_dict(), 'siamese_resnet18.pth')
    print("Training complete. Models saved.")

    # Plotting the Loss Curves
    plt.figure(figsize=(10,5))
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.show()


# Run the training
def main():
    siamese_train()

if __name__ == "__main__":
    main()
