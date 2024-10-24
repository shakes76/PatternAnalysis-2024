import torch
import torchvision
from dataset import ProstateCancerDataset
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

def save_checkpoint(state, filename = "my_checkpoint.pth.tar"):
    print("--Saving Checkpoint--")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("--loading checkpoint--")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
        train_dir,
        train_seg_dir,
        val_dir,
        val_seg_dir,
        batch_size,
        train_transform,
        val_transform,
        num_workers = 4,
        pin_memory = True,
):
    train_ds = ProstateCancerDataset(image_dir=train_dir, seg_dir=train_seg_dir, transform=train_transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True)
    val_ds = ProstateCancerDataset(image_dir=val_dir, seg_dir=val_seg_dir, transform=val_transform)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=False)
    return train_loader, val_loader
    
def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    #dice_score = 0
    dice_score_per_class = torch.zeros(5).to(device)
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            if len(y.shape) == 4 and y.shape[-1] == 5:  # If y has shape [batch_size, num_classes, H, W]
                y = torch.argmax(y, dim=-1)

            preds = model(x)
            preds = torch.argmax(torch.softmax(preds, dim=1), dim=1)

            preds = preds.view(-1)
            y = y.view(-1)

            num_correct += (preds == y).sum().item()
            num_pixels += torch.numel(preds)

            for cls in range(5):
                preds_cls = (preds == cls).float()
                y_cls = (y == cls).float()

                intersection = (preds_cls * y_cls).sum()
                dice_class_score = (2 * intersection) / (preds_cls.sum() + y_cls.sum() + 1e-8)
                dice_score_per_class[cls] += dice_class_score

        dice_score_per_class = dice_score_per_class / len(loader)  # Average over the batches
        avg_dice_score = dice_score_per_class.mean().item()
        print(f"Got {num_correct}/{num_pixels} pixels correct with accuracy {(num_correct/num_pixels)*100:.2f}")
        print(f"Dice score: {avg_dice_score:.4f}")
        model.train()

def visualize_predictions(loader, model, device, num_images=3):
    """
    Visualizes the input image, ground truth mask, and predicted mask side by side.
    
    Parameters:
    - loader: DataLoader object for validation or test set.
    - model: Your segmentation model.
    - device: torch.device object (e.g., 'cuda' or 'cpu').
    - num_images: Number of images to visualize (default is 3).
    """
    model.eval()
    images_shown = 0

    with torch.no_grad():  # Turn off gradients for validation
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            # Forward pass to get predictions
            preds = model(x)
            preds = torch.argmax(torch.softmax(preds, dim=1), dim=1)  # Get predicted class for each pixel

            if y.shape[-1] == 5:  # If y is one-hot encoded
                y = torch.argmax(y, dim=-1)

            # Move tensors back to CPU for visualization
            x = x.cpu()
            y = y.cpu()
            preds = preds.cpu()

            # Loop through the batch and plot images
            for i in range(x.shape[0]):  # x.shape[0] is the batch size
                if images_shown >= num_images:
                    return  # Exit once we've shown the desired number of images

                fig, ax = plt.subplots(1, 3, figsize=(15, 5))

                # Input image
                ax[0].imshow(np.squeeze(x[i].numpy()), cmap='gray')
                ax[0].set_title('Input Image')
                ax[0].axis('off')

                # Ground truth mask (true segmentation)
                ax[1].imshow(y[i].numpy(), cmap='gray')
                ax[1].set_title('Ground Truth Mask')
                ax[1].axis('off')

                # Predicted mask
                ax[2].imshow(preds[i].numpy(), cmap='gray')
                ax[2].set_title('Predicted Mask')
                ax[2].axis('off')

                plt.show()
                images_shown += 1

    model.train()  # Return to train mode after validatio