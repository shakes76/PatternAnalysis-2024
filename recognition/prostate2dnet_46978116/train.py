import torch
import torchvision
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from dataset import ProstateDataset
from torch.utils.data import DataLoader
from modules import UNet
import os
import matplotlib.pyplot as plt
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler


batch_size = 64
N_epochs = 50
n_workers = 2
pin = True
device = "cuda" if torch.cuda.is_available() else "cpu"
load_model = False
img_height = 256
img_width = 128
learning_rate = 0.0005



# path to image
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

import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5, ignore_index=None, num_classes=6, threshold=0.75, k=10.0):
        """
        Initializes the DiceLoss.
        
        Parameters:
            smooth (float): Smoothing factor to avoid division by zero.
            ignore_index (int, optional): Specifies a target value that is ignored 
                                          and does not contribute to the input gradient.
            num_classes (int): Number of segmentation classes.
            threshold (float): Dice score threshold below which additional penalty is applied.
            k (float): Steepness parameter for the sigmoid function to approximate the step penalty.
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.threshold = threshold
        self.k = k
        self.last_dice_coeff = None

    def forward(self, inputs, targets):
        """
        Forward pass for DiceLoss.
        
        Parameters:
            inputs (torch.Tensor): Predicted logits from the model (before softmax).
                                   Shape: [batch_size, num_classes, H, W]
            targets (torch.Tensor): Ground truth mask.
                                   Shape: [batch_size, H, W]
        
        Returns:
            torch.Tensor: Computed Dice loss.
        """
        # Ensure targets are of type long
        targets = targets.long()

        # Convert targets to one-hot encoding
        targets_one_hot = torch.zeros_like(inputs)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)

        if self.ignore_index is not None:
            mask = targets != self.ignore_index
            inputs = inputs * mask.unsqueeze(1)
            targets_one_hot = targets_one_hot * mask.unsqueeze(1)

        # Apply softmax to get probabilities
        inputs = torch.softmax(inputs, dim=1)

        # Flatten the tensors
        inputs = inputs.view(inputs.size(0), inputs.size(1), -1)  # [batch_size, num_classes, H*W]
        targets_one_hot = targets_one_hot.view(targets_one_hot.size(0), targets_one_hot.size(1), -1)  # [batch_size, num_classes, H*W]

        # Compute intersection and union
        intersection = (inputs * targets_one_hot).sum(-1)  # [batch_size, num_classes]
        total = inputs.sum(-1) + targets_one_hot.sum(-1)  # [batch_size, num_classes]

        # Compute Dice coefficient
        dice_coeff = (2. * intersection + self.smooth) / (total + self.smooth)  # [batch_size, num_classes]

        # Average Dice coefficient over batch
        dice_coeff = dice_coeff.mean(dim=0)  # [num_classes]

                # Store the Dice coefficients
        self.last_dice_coeff = dice_coeff.detach().cpu().numpy()

        # Compute penalty using sigmoid
        penalty = torch.sigmoid(self.k * (self.threshold - dice_coeff))  # [num_classes]

        # Compute cost per class
        cost = 1 - dice_coeff + penalty  # [num_classes]

        # Compute dice_cost as the sum of squared costs
        dice_cost = torch.sum(cost ** 2)

        return dice_cost, self.last_dice_coeff



# Define Combined Loss
class CombinedLoss(nn.Module):
    def __init__(self, ce_weight=None, dice_weight=1.0, ce_weight_factor=1.0, dice_weight_factor=3.0):
        super(CombinedLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight=ce_weight)
        dice_cost = 0
        self.dice_loss = DiceLoss()
        self.ce_weight = ce_weight_factor
        self.dice_weight = dice_weight_factor
        self.last_dice_coeff = None

    def forward(self, inputs, targets):
        dl = self.dice_loss(inputs, targets)
        self.last_dice_coeff = dl[1]
        ce = self.ce_loss(inputs, targets)
        return self.ce_weight * ce + self.dice_weight * dl[0]
    
    def get_last_dice_coeff(self):
        """
        Retrieves the last computed Dice coefficients.

        Returns:
            np.ndarray: Array of Dice coefficients for each class.
        """
        return self.last_dice_coeff

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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers, pin_memory=True)
    val_dataset = ProstateDataset(val_image_dir, val_mask_dir, norm_image=True,transform=train_trainsform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers, pin_memory=True)

    weights = [1, 1, 1, 2, 10, 4]
    weights = torch.tensor(weights, dtype=torch.float).to(device)
    model = UNet().to(device=device)
    loss_fn = CombinedLoss(ce_weight=weights)
    scaler = torch.amp.GradScaler(device=device)
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    train_losses = []
    val_dice_scores = []
    val_losses = []
    best_val_loss = float('inf')

    for epoch in range(N_epochs):
        print(f"\nEpoch [{epoch+1}/{N_epochs}]")
        avg_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler)
        train_losses.append(avg_loss)
        print(f"Average Training Loss: {avg_loss:.4f}")

        # Validation Phase - Compute Validation Loss
        val_loss = validate_fn(val_loader, model, loss_fn)
        val_losses.append(val_loss[0])
        val_dice_scores.append(val_loss[1])
        print(f"Average Validation Loss: {val_loss[0]:.4f}")
        print(f"Dice Coeff: {val_loss[1]}")

        if val_loss[0] < best_val_loss:
            best_val_loss = val_loss[0]
            # Save the best model
            torch.save(model.state_dict(), "bestsofar_unet2d.pth")
            print("Best model saved.")
                  # Save prediction images at the end of training if not saved already
            if (epoch + 1) % 5 != 0:
                print("Saving best prediction images")
                save_img(val_loader, model, folder=f"images/NEWepoch_{epoch}", device=device, num_classes=6, epoch=epoch)

        # Save images every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f"Saving prediction images for epoch {epoch+1}")
            save_img(val_loader, model, folder=f"images/NEWepoch_{epoch+1}", device=device, num_classes=6, epoch=epoch+1)

      # Save prediction images at the end of training if not saved already
    if N_epochs % 5 != 0:
        print("Saving prediction images at the end of training.")
        save_img(val_loader, model, folder=f"images/NEWepoch_{N_epochs}", device=device, num_classes=6, epoch=N_epochs)




    # Plot metrics after training
    plot_metrics(train_losses,val_losses, val_dice_scores, num_classes=6, save_path="BIGONEmetrics_plot.png")

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



def save_img(loader, model, folder="images", device="cuda", num_classes=6, max_images_per_class=6, epoch=1, classes_per_row=3):
    """
    Saves prediction images for each class, stacking multiple classes on the same page.
    
    Parameters:
        loader (DataLoader): DataLoader for the dataset to save images from.
        model (nn.Module): Trained model for making predictions.
        folder (str): Directory to save the images.
        device (str): Device to perform computations on ('cuda' or 'cpu').
        num_classes (int): Number of segmentation classes.
        max_images_per_class (int): Maximum number of images to save per class.
        epoch (int): Current epoch number for filename reference.
        classes_per_row (int): Number of classes to display per row in the stacked image.
    """
    model.eval()
    os.makedirs(folder, exist_ok=True)

    green = [0, 255, 0]    # Green
    black = [0, 0, 0]       # Black

    # Dictionary to hold images for each class
    class_images = {cls: [] for cls in range(num_classes)}

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
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
                input_img = x[i].squeeze(0)  # Shape: [H, W]
                input_img = (input_img * 255).astype(np.uint8)  # Scale to [0, 255]

                true_mask = y[i]  # Shape: [H, W]
                pred_mask = preds[i]  # Shape: [H, W]

                for cls in range(num_classes):
                    if len(class_images[cls]) >= max_images_per_class:
                        continue  # Skip if already have enough images for this class

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

                    # Combine input, ground truth, and predicted masks horizontally
                    combined = np.hstack((input_img[..., np.newaxis].repeat(3, axis=2), gt_color, pred_color))
                    class_images[cls].append(combined)

                    if len(class_images[cls]) >= max_images_per_class:
                        continue  # Stop collecting images for this class

    # Now, create stacked images per class group
    for row_start in range(0, num_classes, classes_per_row):
        classes_in_row = list(range(row_start, min(row_start + classes_per_row, num_classes)))
        fig, axs = plt.subplots(1, classes_per_row, figsize=(15, 5))

        for idx, cls in enumerate(classes_in_row):
            if idx >= len(axs):
                break  # In case num_classes is not a multiple of classes_per_row

            if len(class_images[cls]) == 0:
                axs[idx].axis('off')
                axs[idx].set_title(f'Class {cls} - No Images')
                continue

            # Concatenate images for the class vertically
            imgs = class_images[cls]
            concatenated = np.vstack(imgs)
            axs[idx].imshow(concatenated)
            axs[idx].set_title(f'Class {cls}')
            axs[idx].axis('off')

        # Hide any unused subplots
        for idx in range(len(classes_in_row), classes_per_row):
            axs[idx].axis('off')

        plt.tight_layout()
        save_path = os.path.join(folder, f"epoch_{epoch}_classes_{row_start}-{row_start + classes_per_row -1}.png")
        plt.savefig(save_path)
        plt.close(fig)

    print(f"Saved stacked prediction images for epoch {epoch} to '{folder}' folder.")
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
    dice_coeffs = []
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(loader):
            data = data.to(device=device)
            targets = targets.float().squeeze(1).to(device=device)
            targets = targets.long()

            with torch.cuda.amp.autocast():
                predictions = model(data)
                loss = loss_fn(predictions, targets)
            
            total_loss += loss.item()
                    # Retrieve Dice coefficients from the loss function
            dice_coeff = loss_fn.get_last_dice_coeff()  # [num_classes]
            dice_coeffs.append(dice_coeff)

    
    average_loss = total_loss / len(loader)
    average_loss = total_loss / len(loader)
    
    # Convert list of Dice coefficients to a NumPy array for averaging
    dice_coeffs = np.array(dice_coeffs)  # Shape: [num_batches, num_classes]
    mean_dice_coeff = dice_coeffs.mean(axis=0)  # Shape: [num_classes]
    
    # Create a dictionary for easy interpretation
    dice_dict = {cls: mean_dice_coeff[cls] for cls in range(len(mean_dice_coeff))}
    return average_loss, dice_dict



if __name__ == '__main__':
    main()
