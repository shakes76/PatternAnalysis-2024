"""
Functions for model metrics.

@author George Reid-Smith
"""
import torch
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure

def batch_ssim(images, outputs, device, data_range=2.0):
    """Calculates SSIM for a batch of images.

    Args:
        images (torch.Tensor): input image tensor
        outputs (torch.Tensor): reconstructed image tensor
        device (torch.device): device to load SSIM module to
        data_range (float, optional): channel range. Defaults to 2.0.

    Returns:
        float: average ssim for the batch
    """
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=2.0).to(device)

    ssim_score = ssim_metric(images, outputs)

    return ssim_score

def avg_ssim(device, model, loader):
    """Calculated average SSIM over a dataloader.

    Args:
        device (torch.device): the device for inference
        model (nn.Module): VQVAE model
        loader (Dataloader): dataloader of input tensors

    Returns:
        float: average ssim over the input dataloader
    """
    ssim_scores = []

    with torch.no_grad():
        for _, image in enumerate(loader):
            image = image.to(device)

            # Forward pass of current model
            outputs, _ = model(image)

            # Append SSIM score
            ssim_score = batch_ssim(outputs, image, device)
            ssim_scores.append(ssim_score.item())

    # Average over validation set
    avg_ssim = sum(ssim_scores) / len(ssim_scores) if ssim_scores else 0
    
    return avg_ssim

def avg_loss(device, model, criterion, llw, loader):
    """Calculate average reconstruction loss on validation set.

    Args:
        device (torch.device): device for validation reconstruction
        model (nn.Module): the VQVAE model
        criterion (any): criterion for model
        llw (float): latent loss weighting
        validation_loader (Dataloader): validation dataloader

    Returns:
        float: average reconstruction loss on the input dataloader
    """
    losses = []

    with torch.no_grad():
        for _, images in enumerate(loader):
            images = images.to(device)

            outputs, latent_loss = model(images)

            recon_loss = criterion(outputs, images)
            latent_loss = latent_loss.mean()
            loss = recon_loss + llw * latent_loss

            losses.append(loss.item())
        
    avg_loss = sum(losses) / len(losses) if losses else 0

    return avg_loss
