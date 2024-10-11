from dataset import MRIDataset
from modules import VQVAE

import os

import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import optim

from torchvision import transforms, utils
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure

# TODO restructure the main class
# TODO comments for the module class
# TODO commit current training procedure

def test_ssim(device: torch.device, model: nn.Module, test_loader: DataLoader):
    """Tests the model using Structured Similarity Index Measure
    on a given test dataset.

    Requires:
        - Model in evaluation mode

    Args:
        model (nn.Module): the current model version
        test_loader (DataLoader): dataloader to test model
        device (torch.device): the training device

    Returns:
        float: average SSIM over a test dataset
    """
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=2.0).to(device)
    ssim_scores = []

    with torch.no_grad():
        for i, image in enumerate(test_loader):
            image = image.to(device)

            # Forward pass of current model
            out, _ = model(image)

            # Append SSIM score
            ssim_score = ssim_metric(out, image)
            ssim_scores.append(ssim_score.item())

    # Average over validation set
    avg_ssim = sum(ssim_scores) / len(ssim_scores) if ssim_scores else 0
    
    return avg_ssim

def create_samples(epoch, execution, model, image, sample_size):
    """Saves training images.

    Args:
        epoch (int): current epoch
        model (nn.Module): current model
        image (any): the batch image
        sample_size (int): number of images to be generated
    """
    # Generate traiing samples
    sample = image[:sample_size]

    with torch.no_grad():
        out, _ = model(sample)

    utils.save_image(
        torch.cat([sample, out], 0),
        f"samples/{str(epoch + 1).zfill(5)}_{str(execution).zfill(5)}.png",
        nrow=sample_size,
        normalize=True,
    )

def train_epoch(epoch, epochs, device, model, optimizer, train_loader, validate_loader):
    criterion = nn.MSELoss()

    latent_loss_weight = 0.25
    sample_size = 10

    for i, image in enumerate(train_loader):
        # Reset model gradient
        model.zero_grad()

        # Load image to device
        image = image.to(device)

        # Forward pass
        out, latent_loss = model(image)

        # Calculate loss
        recon_loss = criterion(out, image)
        latent_loss = latent_loss.mean()
        loss = recon_loss + latent_loss_weight * latent_loss

        # Back propogation
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            # Model in evaluation mode so no changes
            model.eval()

            # Test current SSIM using validation dataset
            epoch_ssim = test_ssim(device, model, validate_loader)
            print(f'Epoch {epoch + 1}/{epochs} - Average SSIM on Validation Set: {epoch_ssim:.4f}')

            # Save training samples
            create_samples(epoch, i, model, image, sample_size) 

            model.train()
    
def main(args):
    TRAIN_PATH = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_train" 
    TEST_PATH = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_test"
    VALIDATE_PATH = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_validate"

    if args.setting == 'local':
        TRAIN_PATH = "C:/university/COMP3710/a3-project/data/keras_slices_data/keras_slices_train"
        VALIDATE_PATH = "C:/university/COMP3710/a3-project/data/keras_slices_data/keras_slices_validate"
        TEST_PATH = "C:/university/COMP3710/a3-project/data/keras_slices_data/keras_slices_test"   
    
    print('> Finding device')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print(f'Using device: {torch.cuda.get_device_name(0)}')
    else:
        print(f'WARNING: using CPU')

    print('> Creating datasets')
    train_dataset = MRIDataset(TRAIN_PATH)
    validate_dataset = MRIDataset(VALIDATE_PATH)
    test_dataset = MRIDataset(TEST_PATH)
    print(f'Loaded train dataset with {len(train_dataset)} images with type {type(train_dataset[0])}')
    print(f'Loaded validate dataset with {len(validate_dataset)} images with type {type(validate_dataset[0])}')
    print(f'Loaded test dataset with {len(test_dataset)} images with type {type(test_dataset[0])}')

    print('> Loading datasets')
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    validate_loader = DataLoader(validate_dataset, batch_size=10, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)
    
    print('> Building model')
    model = VQVAE().to(device)

    print('> Building optimizer')
    optimizer = optim.Adam(model.parameters(), lr=0.3e-4)

    epochs = 20

    print('> Training Model')
    for i in range(epochs):
        train_epoch(i, epochs, device, model, optimizer, train_loader, validate_loader)
    
    print('> Testing final model')

    average_test_ssim = test_ssim(device, model, test_loader)
    print(f'Average SSIM on test set: {average_test_ssim:.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--setting', type=str, default='rangpur')
    parser.add_argument('--epochs', type=int, default=10)

    args = parser.parse_args()
    
    main(args)