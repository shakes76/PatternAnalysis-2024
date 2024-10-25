import torch
import os
from pathlib import Path
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

def compute_loss(original, reconstructed, embedding_loss, prior_logits, beta=0.25):
    """
    Computes the loss for VQ-VAE with PixelCNN prior.
    - x: Original input image
    - x_hat: Reconstructed image
    - embedding_loss: Loss from the vector quantization step
    - prior_logits: Output from PixelCNN
    """
    reconstruction_loss = F.mse_loss(reconstructed, original)  
    prior_loss = F.cross_entropy(prior_logits, original.long()) 
    total_loss = reconstruction_loss + beta * embedding_loss + prior_loss 

    return total_loss

def save_model_and_results(model, results, hyperparameters):
    SAVE_MODEL_PATH = os.getcwd() + '\\results'
    directory = Path(SAVE_MODEL_PATH)
    if not directory.exists():
        os.makedirs(directory)
    results_to_save = {
        'model': model.state_dict(),
        'results': results,
        'hyperparameters': hyperparameters
    }
    torch.save(results_to_save,
               SAVE_MODEL_PATH + '\\vqvae_data.pth')

# Generate predictions and reconstructions without updating gradients
def predict_and_reconstruct(model, data_loader):
    model.eval()
    with torch.no_grad():
        for x in data_loader:
            x = x.to(device)
            _, x_hat, _ = model(x)
            yield x.cpu().numpy(), x_hat.cpu().numpy() 
