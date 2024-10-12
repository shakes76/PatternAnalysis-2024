import torch
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure

def calculate_ssim(device, model, test_loader):
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
            outputs, _ = model(image)

            # Append SSIM score
            ssim_score = ssim_metric(outputs, image)
            ssim_scores.append(ssim_score.item())

    # Average over validation set
    avg_ssim = sum(ssim_scores) / len(ssim_scores) if ssim_scores else 0
    
    return avg_ssim