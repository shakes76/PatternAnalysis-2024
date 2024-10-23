"""
This file contains the code to train the 3D U-Net model on the training set and validate on the 
validation set. The training loops contains validation and saving of the model every few epochs.
A final trained version of the model is saved at the end of training.

Abdullah Badat (47022173), abdullahbadat27@gmail.com
"""

from dataset import *
import torch
from torch.utils.data import DataLoader
import torchio as tio
from torch.nn import functional as F
from utils import *
from modules import *
from monai.losses.dice import DiceLoss
import nibabel as nib
from typing import List


device = torch.device('cuda' if torch.cuda.is_available() else 'mps')


def save(
    predictions: torch.Tensor, 
    affines: torch.Tensor, 
    epoch: int,
    save_path: str
) -> None:
    """
    Save the model's predictions for a single batch as a NIfTI file.

    Parameters:
    - predictions: The output predictions from the model (logits or softmax outputs).
    - affines: The affine transformations associated with the MRI scans.
    - epoch: The current epoch number, used for naming the saved file.
    - save_path: The directory path to save the predictions.
    """
    # Get predicted class by taking the argmax along the class dimension
    predictions = torch.argmax(predictions, dim=1)

    # Reshape predictions to the correct 3D format [batch, 1, depth, height, width]
    batch_size = int(predictions.shape[0] / (1 * WIDTH * HEIGHT * DEPTH)) 
    predictions = predictions.view(batch_size, 1, WIDTH, HEIGHT, DEPTH).cpu().numpy()

    # Extract the first prediction and remove the batch dimension
    prediction = predictions[0].squeeze()

    # Extract the affine transformation for the first prediction
    affine = affines.numpy()[0]

    # Save the prediction as a NIfTI file
    nib.save(
        nib.Nifti1Image(prediction, affine, dtype=np.dtype('int64')), 
        f"{save_path}/prediction_{epoch}.nii.gz"
    )

def validate(
    model, 
    valid_dataloader: DataLoader
) -> List[float]:
    """
    Perform validation on a 3D segmentation model and compute average Dice scores for each class.

    Parameters:
    - model: The trained PyTorch model to be evaluated.
    - valid_dataloader: The DataLoader object for the validation dataset.

    Returns:
    - List[float]: A list of average Dice scores for each class in the validation set,
    where each value corresponds to the Dice score of a class.
    """
    # Set model to evaluation mode
    model.eval()  
    
    running_loss = 0.0
    dice_scores = [0] * N_CLASSES
    dice_score_labels = DiceLoss(softmax=True, reduction='none')
    dice_score_mean = DiceLoss(softmax=True)

    with torch.no_grad():
        for batch_idx, (inputs, masks, affine) in enumerate(valid_dataloader):
            # Move inputs and masks to the correct device
            inputs, masks = inputs.to(device), masks.to(device)
            
            # One-hot encode the ground truth masks, reshape to [B, C, W, H, D]
            one_hot_masks_3d = F.one_hot(masks, num_classes=N_CLASSES).permute(0, 4, 1, 2, 3)

            # Forward pass
            _, _, logits = model(inputs)

            # Compute the loss
            score = dice_score_labels(logits, one_hot_masks_3d)
            running_loss += dice_score_mean(logits, one_hot_masks_3d)
            for i in range(N_CLASSES):
                dice_scores[i] += 1 - score[0][i].item()

    print("Valid loss: ", float(running_loss) / len(valid_dataloader))
    print([float(score / len(valid_dataloader)) for score in dice_scores])


def train(
    mode: str, 
    images_path: str, 
    masks_path: str, 
    lr: float, 
    weight_decay: float, 
    step_size: int, 
    gamma: float, 
    epochs: int, 
    batch_size: int,
    save_path: str
    ) -> None:
    """
    Train a 3D UNet model on a 3D prostate MRI dataset for segmentation tasks. The model is 
    validated and its predictions are saved every 3 epochs.

    Parameters:
    - mode (str): The training mode, typically specifying the dataset split (e.g., "train").
    - images_path (str): Path to the directory containing the input MRI images.
    - masks_path (str): Path to the directory containing the corresponding ground truth masks.
    - lr (float): Learning rate for the optimizer.
    - weight_decay (float): Weight decay for the optimizer.
    - step_size (int): The number of epochs between each learning rate adjustment.
    - gamma (float): Multiplicative factor for learning rate decay in the scheduler.
    - epochs (int): The number of epochs to train the model.
    - batch_size (int): The batch size for training data.
    - save_path (str): Path to save the model predictions.
    """
    train_transforms = tio.Compose([
        tio.RescaleIntensity((0, 1)),
        tio.RandomFlip(),
        tio.Resize((WIDTH,HEIGHT,DEPTH)),
        tio.RandomAffine(degrees=0),
        tio.RandomElasticDeformation(),
        tio.ZNormalization(),
    ])

    valid_transforms = tio.Compose([
        tio.RescaleIntensity((0, 1)),
        tio.Resize((WIDTH,HEIGHT,DEPTH)),
        tio.ZNormalization(),
    ])

    # Load and process data
    valid_dataset = ProstateDataset3D(images_path, masks_path, "valid", valid_transforms)
    valid_dataloader = DataLoader(
        valid_dataset, 
        batch_size=1, 
        shuffle=True, 
        num_workers=NUM_WORKERS
    )

    train_dataset = ProstateDataset3D(images_path, masks_path, mode, train_transforms)
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=NUM_WORKERS
    )

    # Initialize model, move it to the target device, and apply weight initialization
    model = Modified3DUNet(IN_CHANNELS, N_CLASSES, BASE_N_FILTERS)
    model.to(device)
    model.apply(init_weights)

    # Optimizer and Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # Loss function
    criterion = DiceLoss(softmax=True)

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (inputs, masks, affines) in enumerate(train_dataloader):
            # Move inputs and masks to the device (GPU/CPU)
            inputs, masks = inputs.to(device), masks.to(device)

            # One-hot encode masks for multi-class segmentation
            one_hot_masks_3d = F.one_hot(masks, num_classes=N_CLASSES).permute(0, 4, 1, 2, 3)

            # Forward pass
            optimizer.zero_grad()
            _, predictions, logits = model(inputs)
            
            # Compute loss and backpropagate
            loss = criterion(logits, one_hot_masks_3d)
            loss.backward()
            optimizer.step()

            # Accumulate loss
            running_loss += loss.item()

        # Step the learning rate scheduler
        scheduler.step()

        # Periodic validation and model saving
        if epoch % 3 == 0:
            validate(model, valid_dataloader)
            save(predictions, affines, epoch, save_path)

        # Log training progress
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {running_loss / len(train_dataloader):.4f}")

    # Save the trained model
    torch.save(
        model.state_dict(), 
        f'model_lr_{lr}_e_{epochs}_bs{batch_size}.pth'
    )

    # Final validation after training
    validate(model, valid_dataloader)