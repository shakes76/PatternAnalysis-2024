import torch
import torch.optim as optim
from tqdm import tqdm
from modules import *
from dataset import *

def train_model():
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the model, optimizer, and loss function
    encoder = Encoder(latent_dim=256)
    decoder = Decoder(latent_dim=256)
    unet = UNet()
    model = LatentDiffusionModel(encoder, decoder, unet, timesteps=1000)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    model = model.to(device)
    dataloader = load_data("/home/groups/comp3710/ADNI")
    # Training loop
    epochs = 10

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for images, timesteps in progress_bar:
            images = images.to(device)
            timesteps = timesteps.to(device)

            optimizer.zero_grad()

            # Forward pass
            pred_noise, actual_noise = model(images, timesteps)

            # Compute loss
            loss = criterion(pred_noise, actual_noise)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix({'Loss': running_loss / len(dataloader)})

        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {running_loss / len(dataloader)}")
# Make sure this runs only when executed directly
if __name__ == '__main__':
    train_model()