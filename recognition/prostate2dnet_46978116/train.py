# train.py

import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from modules import UNet, MultiClassDiceLoss
from dataset import ProstateDataset

def main():
    # Hyperparameters
    num_epochs = 20
    batch_size = 64
    learning_rate = 0.005
    num_classes = 6  # Number of segmentation classes

    # Directories for saving outputs
    output_dir = 'outputs/'
    model_dir = 'models/'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data directories
    train_image_dir = 'keras_slices_data/keras_slices_train'
    train_mask_dir = 'keras_slices_data/keras_slices_seg_train'
    val_image_dir = 'keras_slices_data/keras_slices_validate'
    val_mask_dir = 'keras_slices_data/keras_slices_seg_validate'

    # Datasets and loaders
    train_dataset = ProstateDataset(train_image_dir, train_mask_dir, norm_image=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dataset = ProstateDataset(val_image_dir, val_mask_dir, norm_image=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Initialize the model
    model = UNet(num_classes=num_classes).to(device)

    # Use MultiClassDiceLoss
    criterion = MultiClassDiceLoss(num_classes=num_classes)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses, val_losses = [], []
    train_dices, val_dices = [], []

    print('\nStarting training:\n')

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss, running_dice = 0.0, 0.0

        for images, masks in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
            images = images.to(device)  # (B, 1, H, W)
            masks = masks.to(device).long()  # (B, H, W)

            optimizer.zero_grad()
            outputs = model(images)  # (B, C, H, W)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_dice += (1.0 - loss.item())  # Since DiceLoss = 1 - Dice Coefficient

        avg_train_loss = running_loss / len(train_loader)
        avg_train_dice = running_dice / len(train_loader)
        train_losses.append(avg_train_loss)
        train_dices.append(avg_train_dice)

        # Validation phase
        model.eval()
        val_running_loss, val_running_dice = 0.0, 0.0

        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{num_epochs}"):
                images = images.to(device)
                masks = masks.to(device).long()

                outputs = model(images)
                loss = criterion(outputs, masks)

                val_running_loss += loss.item()
                val_running_dice += (1.0 - loss.item())

                # Validation phase
                if (epoch + 1) % 5 == 0 and torch.rand(1).item() < 0.1:
                    idx = torch.randint(0, images.size(0), (1,)).item()
                    preds = outputs.argmax(dim=1)  # Shape: (B, H, W)
                    preds = preds.unsqueeze(1)  # Shape: (B, 1, H, W)
                    preds_sample = preds[idx].float() / (num_classes - 1)  # Shape: (1, H, W)

                    images_sample = images[idx]  # Shape: (C, H, W)

                    masks_sample = masks[idx].unsqueeze(0).float() / (num_classes - 1)  # Shape: (1, H, W)

                    # Match the number of channels
                    C = images_sample.shape[0]
                    if preds_sample.shape[0] != C:
                        preds_sample = preds_sample.repeat(C, 1, 1)
                    if masks_sample.shape[0] != C:
                        masks_sample = masks_sample.repeat(C, 1, 1)

                    # Concatenate along the width
                    concatenated = torch.cat([images_sample, preds_sample, masks_sample], dim=2)

                    # Save the concatenated image
                    save_image(concatenated, os.path.join(output_dir, f'epoch{epoch+1}_sample.png'))


        avg_val_loss = val_running_loss / len(val_loader)
        avg_val_dice = val_running_dice / len(val_loader)
        val_losses.append(avg_val_loss)
        val_dices.append(avg_val_dice)

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {avg_train_loss:.4f}, Train Dice: {avg_train_dice:.4f} "
              f"Val Loss: {avg_val_loss:.4f}, Val Dice: {avg_val_dice:.4f}")

        # Save the model and plot losses every 5 epochs
        if (epoch + 1) % 5 == 0:
            # Save the model
            torch.save(model.state_dict(), os.path.join(model_dir, f'unet_model_epoch{epoch+1}.pth'))

            # Save the loss and dice arrays
            np.save(os.path.join(output_dir, 'train_losses.npy'), np.array(train_losses))
            np.save(os.path.join(output_dir, 'val_losses.npy'), np.array(val_losses))
            np.save(os.path.join(output_dir, 'train_dices.npy'), np.array(train_dices))
            np.save(os.path.join(output_dir, 'val_dices.npy'), np.array(val_dices))

            # Plot the loss and dice coefficient
            plt.figure()
            plt.plot(range(1, epoch+2), train_losses, label='Train Loss')
            plt.plot(range(1, epoch+2), val_losses, label='Val Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig(os.path.join(output_dir, f'loss_epoch{epoch+1}.png'))
            plt.close()

            plt.figure()
            plt.plot(range(1, epoch+2), train_dices, label='Train Dice')
            plt.plot(range(1, epoch+2), val_dices, label='Val Dice')
            plt.xlabel('Epoch')
            plt.ylabel('Dice Coefficient')
            plt.legend()
            plt.savefig(os.path.join(output_dir, f'dice_epoch{epoch+1}.png'))
            plt.close()

    # Save the final model
    torch.save(model.state_dict(), os.path.join(model_dir, 'unet_prostate_segmentation_final.pth'))

if __name__ == "__main__":
    main()
