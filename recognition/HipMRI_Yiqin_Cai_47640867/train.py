import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dataset import ProstateSegmentationDataset
from modules import UNet

# Dice coefficient
def dice_coefficient(pred, target, smooth=1e-5):
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def train_model():
    # Testing dataset path (needs to be changed)
    train_image_dir = 'C:/Users/一七/Desktop/HipMRI_study_keras_slices_data/keras_slices_train'
    train_label_dir = 'C:/Users/一七/Desktop/HipMRI_study_keras_slices_data/keras_slices_seg_train'
    val_image_dir = 'C:/Users/一七/Desktop/HipMRI_study_keras_slices_data/keras_slices_validate'
    val_label_dir = 'C:/Users/一七/Desktop/HipMRI_study_keras_slices_data/keras_slices_seg_validate'

    # Load training dataset
    train_dataset = ProstateSegmentationDataset(train_image_dir, train_label_dir, norm_image=True, categorical=False)
    train_dataset = torch.utils.data.Subset(train_dataset, range(1000)) 
    val_dataset = ProstateSegmentationDataset(val_image_dir, val_label_dir, norm_image=True, categorical=False)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    # Initialization model, loss function and optimizer
    model = UNet(in_channels=1, out_channels=1)
    criterion = nn.BCELoss()  
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            print(f"Processing batch {i+1}/{len(train_loader)} in epoch {epoch+1}")
            
            images = images.float()
            labels = labels.float()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_losses.append(running_loss / len(train_loader))

        # Verification 
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_losses.append(val_loss / len(val_loader))

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]}, Val Loss: {val_losses[-1]}')

    # Loss curve
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.legend()
    plt.show()

    # Save model
    torch.save(model.state_dict(), 'unet_model.pth')
    print("Model saved successfully.")

if __name__ == '__main__':
    train_model()
