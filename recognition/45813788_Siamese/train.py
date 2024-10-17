# train.py

import torch
from torch.utils.data import DataLoader
from dataset import ISISCDataset, malig_aug, benign_aug
from modules import SiameseNN
import pandas as pd
import os
from pytorch_metric_learning.losses import ContrastiveLoss
from pytorch_metric_learning.reducers import AvgNonZeroReducer
from pytorch_metric_learning.distances import LpDistance
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt


def load_data(excel):
    df = pd.read_csv(excel)
    df = df.drop(columns=['Unnamed: 0', 'patient_id'])
    benign = df[df['target'] == 0].reset_index(drop=True)
    malignant = df[df['target'] == 1].reset_index(drop=True)
    return malignant, benign


def siamese_train():
    # Paths
    current_dir = os.getcwd()
    excel = os.path.join(current_dir, 'recognition', '45813788_Siamese', 'dataset', 'train-metadata.csv')
    images = os.path.join(current_dir, 'recognition', '45813788_Siamese', 'dataset', 'train-image', 'image')

    # Load data
    malignant_df, benign_df = load_data(excel=excel)

    # Train-validation split with stratification
    benign_train, benign_val = train_test_split(
        benign_df, test_size=0.1, stratify=benign_df['target'], random_state=42
    )
    malignant_train, malignant_val = train_test_split(
        malignant_df, test_size=0.1, stratify=malignant_df['target'], random_state=42
    )

    # Initialize training and validation datasets
    train_dataset = ISISCDataset(
        benign_df=benign_train,
        malignant_df=malignant_train,
        images_dir=images,
        transform_benign=benign_aug,
        transform_malignant=malig_aug,
        augment_ratio=0.5  # Adjust based on your needs
    )

    val_dataset = ISISCDataset(
        benign_df=benign_val,
        malignant_df=malignant_val,
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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Training parameters
    epochs = 5
    best_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_loss = 0.0

        for img1, img2, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} Training"):
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

            print(f"Shape of labels before loss: {labels.shape}")  # Debugging line


            # Forward pass
            y1, y2 = model(img1, img2)
            loss = contrastive_loss(y1, y2, labels.squeeze())

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for img1, img2, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} Validation"):
                img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

                y1, y2 = model(img1, img2)
                loss = contrastive_loss(y1, y2, labels.squeeze())

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

        # Checkpointing
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
