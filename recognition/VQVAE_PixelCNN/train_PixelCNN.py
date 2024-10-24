#Import Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os
from modules import *
from train_VQVAE import *
from dataset import *

# Hyperparameter
NUM_EPOCH_PixelCNN = 150

# Training PixelCNN
def train_pixel():
    # Ensure model directory exists
    if not os.path.exists("./Model"):
        os.mkdir("./Model")

    # Initializing dataloader
    dataloader = get_dataloader("HipMRI_study_keras_slices_data", batch_size = BATCH_SIZE)

    # Initialize PixelCNN and optimizer
    pixelcnn_model = PixelCNN().to(device)
    optimizer_ = torch.optim.Adam(pixelcnn_model.parameters(), lr=1e-3)

    # Load pre-trained VQ-VAE model (set to evaluation mode)
    model = Model(HIDDEN_DIM, RESIDUAL_HIDDEN_DIM, NUM_RESIDUAL_LAYER, NUM_EMBEDDINGS, EMBEDDING_DIM, COMMITMENT_COST).to(device)
    model.load_state_dict(torch.load("./Model/Vqvae.pth"))
    model.eval()  # Set the VQ-VAE model to evaluation mode

    tqdm_bar = tqdm(range(NUM_EPOCH_PixelCNN))  # Training progress bar
    pre_loss = float('inf')  # Initialize previous loss
    epochs = 0

    for epoch in tqdm_bar:
        # Load training image batch
        train_img_ = next(iter(dataloader))
        train_img = train_img_.to(device)

        # Get quantized latents from VQ-VAE (without tracking gradients)
        with torch.no_grad():
            _, _, _, quantized = model(train_img)

        # Forward pass through PixelCNN
        output = pixelcnn_model(quantized)

        # Compute loss (MSELoss for continuous output)
        loss = F.mse_loss(output, quantized)

        # Backpropagation
        optimizer_.zero_grad()
        loss.backward()
        optimizer_.step()

        epochs += 1
        tqdm_bar.set_description(f'Loss: {loss.item()}')

        # Save model if loss improves
        with torch.no_grad():
            if loss.item() <= pre_loss:
                print(f"preloss {pre_loss} current loss {loss.item()}")
                print("Saving model...")
                torch.save(pixelcnn_model.state_dict(), "./Model/pixelCNN.pth")
                pre_loss = loss.item()


if __name__ == "__main__":
    # Define device (GPU/CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Start training PixelCNN
    train_pixel()