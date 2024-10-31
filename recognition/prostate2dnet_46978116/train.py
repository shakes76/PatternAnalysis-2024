import torch
import torchvision
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from modules import UNet
from dataset import ProstateDataset
from torch.utils.data import DataLoader
from modulesnew import UNet
import os
import matplotlib.pyplot as plt
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler


batch_size = 32
N_epochs = 25
n_workers = 4
pin = True
device = "cuda" if torch.cuda.is_available() else "cpu"
load_model = False
img_height = 256
img_width = 128
learning_rate = 0.005



# Data directories
train_image_dir = 'keras_slices_data/keras_slices_train'
train_mask_dir = 'keras_slices_data/keras_slices_seg_train'
val_image_dir = 'keras_slices_data/keras_slices_validate'
val_mask_dir = 'keras_slices_data/keras_slices_seg_validate'

def train_fn(loader,model,optimizer,loss_fn,scaler):
    loop = tqdm(loader)
    total_loss = 0.0

    for batch_idx, (data,targets) in enumerate(loop):
        data = data.to(device=device)
        targets = targets.float().squeeze(1).to(device=device)
        targets = targets.long()     

        with torch.amp.autocast(device_type=device):
            predictions = model(data)
            loss = loss_fn(predictions, targets)
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    return total_loss / len(loader)



def main():
    train_trainsform = A.Compose(
       [ A.Resize(height=img_height,width=img_width),
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Normalize(mean=[0.0], std=[1.0], max_pixel_value=255.0),
        ToTensorV2()]

    )

    train_dataset = ProstateDataset(train_image_dir, train_mask_dir, norm_image=True, transform=train_trainsform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dataset = ProstateDataset(val_image_dir, val_mask_dir, norm_image=True,transform=train_trainsform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)


    model = UNet().to(device=device)
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(device=device)
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    train_losses = []
    val_dice_scores = []
    val_losses = []

    for epoch in range(N_epochs):
        print(f"\nEpoch [{epoch+1}/{N_epochs}]")
        avg_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler)
        train_losses.append(avg_loss)
        print(f"Average Training Loss: {avg_loss:.4f}")

        print("Evaluating Dice Score on Validation Set:")
        dice_dict = dice_score(val_loader, model, num_classes=6)
        val_dice_scores.append(dice_dict)

        # Validation Phase - Compute Validation Loss
        val_loss = validate_fn(val_loader, model, loss_fn)
        val_losses.append(val_loss)
        print(f"Average Validation Loss: {val_loss:.4f}")

        scheduler.step()




    print("Saving Prediction Images:")
    save_img(val_loader, model, folder=f"images/epoch_{epoch+1}", device=device, num_classes=6)

    # Plot metrics after training
    plot_metrics(train_losses, val_dice_scores, num_classes=6, save_path="metrics_plot.png")

    #save the model after training
    torch.save(model.state_dict(), "unet2d_final.pth")

def dice_score(loader, model, num_classes=6):
    model.eval()
    dice_dict = {cls: 0.0 for cls in range(num_classes)}
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).long()
            
            # Remove any extra singleton dimensions
            while y.dim() > 3:
                y = y.squeeze(1)
            
            preds = model(x)
            preds = preds.argmax(dim=1)  # Shape: [batch_size, H, W]

            for cls in range(num_classes):
                pred_cls = (preds == cls).float()
                true_cls = (y == cls).float()

                intersection = (pred_cls * true_cls).sum()
                union = pred_cls.sum() + true_cls.sum()

                if union == 0:
                    dice = 1.0  # Perfect score if both pred and true have no pixels for this class
                else:
                    dice = (2.0 * intersection + 1e-8) / (union + 1e-8)
                
                dice_dict[cls] += dice.item()

    # Average Dice score per class
    for cls in dice_dict:
        dice_dict[cls] /= len(loader)

    print(f"Dice Scores per class: {dice_dict}")
    return dice_dict

import matplotlib.pyplot as plt

def plot_metrics(train_losses, val_losses, val_dice_scores, num_classes=6, save_path="metrics_plot.png"):
    """
    Plots training and validation losses, and Dice scores per class over epochs.
    
    Parameters:
        train_losses (list): List of training losses per epoch.
        val_losses (list): List of validation losses per epoch.
        val_dice_scores (list): List of Dice score dictionaries per epoch.
        num_classes (int): Number of segmentation classes.
        save_path (str): Path to save the plot.
    """
    epochs = range(1, len(train_losses) + 1)

    # Create subplots
    plt.figure(figsize=(18, 6))
    
    # Plot Training and Validation Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot Dice Scores per Class
    plt.subplot(1, 2, 2)
    for cls in range(num_classes):
        cls_scores = [epoch_dice[cls] for epoch_dice in val_dice_scores]
        plt.plot(epochs, cls_scores, label=f'Class {cls}')
    plt.title('Dice Score per Class over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()



def save_img(loader, model, folder="images", device="cuda", num_classes=6, max_images=10):
    """
    Saves prediction images for each class, showing input image, ground truth mask, and predicted mask.
    Uses two colors: green for 'Yes' (class present) and black for 'No' (class absent).
    
    Parameters:
        loader (DataLoader): DataLoader for the dataset to save images from.
        model (nn.Module): Trained model for making predictions.
        folder (str): Directory to save the images.
        device (str): Device to perform computations on ('cuda' or 'cpu').
        num_classes (int): Number of segmentation classes.
        max_images (int): Maximum number of images to save.
    """
    model.eval()
    os.makedirs(folder, exist_ok=True)

    green = [0, 255, 0]    # Green
    black = [0, 0, 0]       # Black

    saved_images = 0
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
            if saved_images >= max_images:
                break

            x = x.to(device=device)
            y = y.to(device=device).long()

            # Remove any extra singleton dimensions (e.g., [batch_size, 1, H, W] -> [batch_size, H, W])
            while y.dim() > 3:
                y = y.squeeze(1)

            preds = model(x)  # Shape: [batch_size, num_classes, H, W]
            preds = preds.argmax(dim=1).cpu().numpy()  # Shape: [batch_size, H, W]
            x = x.cpu().numpy()  # Shape: [batch_size, 1, H, W]
            y = y.cpu().numpy()  # Shape: [batch_size, H, W]

            for i in range(x.shape[0]):
                if saved_images >= max_images:
                    break

                input_img = x[i].squeeze(0)  # Shape: [H, W]
                input_img = (input_img * 255).astype(np.uint8)  # Scale to [0, 255]

                true_mask = y[i]  # Shape: [H, W]
                pred_mask = preds[i]  # Shape: [H, W]

                for cls in range(num_classes):
                    # Create binary masks for the current class
                    gt_binary = (true_mask == cls).astype(np.uint8)
                    pred_binary = (pred_mask == cls).astype(np.uint8)

                    # Initialize color masks
                    gt_color = np.zeros((gt_binary.shape[0], gt_binary.shape[1], 3), dtype=np.uint8)
                    pred_color = np.zeros_like(gt_color)

                    # Assign colors based on binary masks
                    gt_color[gt_binary == 1] = green
                    gt_color[gt_binary == 0] = black

                    pred_color[pred_binary == 1] = green
                    pred_color[pred_binary == 0] = black

                    # Plot input image, ground truth mask, and predicted mask for the current class
                    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                    
                    axs[0].imshow(input_img, cmap='gray')
                    axs[0].set_title('Input Image')
                    axs[0].axis('off')

                    axs[1].imshow(gt_color)
                    axs[1].set_title(f'Ground Truth - Class {cls}')
                    axs[1].axis('off')

                    axs[2].imshow(pred_color)
                    axs[2].set_title(f'Predicted Mask - Class {cls}')
                    axs[2].axis('off')

                    plt.tight_layout()
                    save_path = os.path.join(folder, f"img_{saved_images}_class_{cls}.png")
                    plt.savefig(save_path)
                    plt.close(fig)

                saved_images +=1
                if saved_images >= max_images:
                    break

    print(f"Saved {saved_images} prediction images to '{folder}' folder.")
    return


def validate_fn(loader, model, loss_fn):
    """
    Evaluates the model on the validation set.
    
    Parameters:
        loader (DataLoader): Validation data loader.
        model (nn.Module): The model to evaluate.
        loss_fn (Loss): Loss function.
        
    Returns:
        float: Average validation loss for the epoch.
    """
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(loader):
            data = data.to(device=device)
            targets = targets.float().squeeze(1).to(device=device)
            targets = targets.long()

            with torch.cuda.amp.autocast():
                predictions = model(data)
                loss = loss_fn(predictions, targets)
            
            total_loss += loss.item()
    
    average_loss = total_loss / len(loader)
    return average_loss



if __name__ == '__main__':
    main()
