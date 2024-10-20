import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import random
import torch.nn.parallel
import torch.optim as optim
from tqdm import tqdm
from modules import Generator, Discriminator
from dataset import process_adni, process_cifar

def get_w(w_dim, batch_size, num_layers, latent_net, device):
    """
    Get intermediate latent vector (W).
    """
    # Random noise z latent vector
    z = torch.randn(batch_size, w_dim).to(device)

    # Forward pass z through the mapping network to generate w latent vector
    w = latent_net(z)

    return w[None, :, :].expand(num_layers, -1, -1)

def get_noise(num_layers, batch_size, device):
    """
    Generates a random noise vector for a batch of images.
    """
    noise = []
    resolution = 4

    for i in range(num_layers):
        if i == 0:
            n1 = None
        else:
            n1 = torch.randn(batch_size, 1, resolution, resolution, device=device)
        n2 = torch.randn(batch_size, 1, resolution, resolution, device=device)

        noise.append((n1, n2))

        resolution *= 2

    return noise

def train(model_G, model_D, n_epochs, dataloader, lr,
    z_dim, w_dim, in_channels, num_layers, image_size, device):
    """
    Trains the Generator and Discriminator.
    """
    # -- Initialise Loss --
    criterion = nn.BCELoss().to(device)

    # -- Initialise Optimizers --
    print("> Optimizer's Setup")
    optimG = optim.Adam(model_G.parameters(), lr=lr)
    optimD = optim.Adam(model_D.parameters(), lr=lr)

    # -- Training Loop --

    print("> Starting Training Loop")
    for epoch in range(n_epochs):
        for real_images, _ in tqdm(dataloader, desc=f"Epoch {epoch+1}/{n_epochs}"):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)

            # Train Discriminator
            optimD.zero_grad()

            # Real images
            real_output = model_D(real_images)
            d_real_loss = criterion(real_output, torch.ones_like(real_output, device=device))

            # Fake images
            z = torch.randn(batch_size, z_dim, device=device)
            fake_images = model_G(z)
            fake_output = model_D(fake_images.detach())
            d_fake_loss = criterion(fake_output, torch.zeros_like(fake_output, device=device))

            # Combined discriminator loss
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            optimD.step()

            # Train Generator
            optimG.zero_grad()

            # Generate new fake images
            z = torch.randn(batch_size, z_dim, device=device)
            fake_images = model_G(z)
            fake_output = model_D(fake_images)

            # Generator loss
            g_loss = criterion(fake_output, torch.ones_like(fake_output))
            g_loss.backward()
            optimG.step()

        print(f"Epoch [{epoch+1}/{n_epochs}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")


    print("> Training Complete")

    return model_G, model_D

def main():
    # Needed for reproducible results
    manual_seed = 46915474
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.use_deterministic_algorithms(True)

    model_name = "dcgan-mnist"
    print("> Model Name:", model_name)

    # -- Check Device --
    if (torch.backends.mps.is_available()):
        device = torch.device("mps")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("> Device:", device)

    # -- Hyper-parameters --
    num_epochs = 25
    lr = 1e-3 # Use 1e-3 for Adam
    batch_size = 64
    channels = 512
    z_dim = 512
    w_dim = 512
    num_layers = 7
    image_size = 32
    data_name = 'cifar'

    if data_name == "adni":
        image_size = 240
        batch_size = 32
        dataset, dataloader = process_adni(batch_size=batch_size)
    else:
        dataset, dataloader = process_cifar(batch_size=batch_size)

    # -- Initialise Models --
    model_G = Generator(z_dim, w_dim, channels, num_layers).to(device)
    model_D = Discriminator(image_size).to(device)

    # Train model
    model_G, model_D = train(model_G, model_D, num_epochs, dataloader, lr,
        z_dim, w_dim, channels, num_layers, image_size, device)

if __name__ == "__main__":
    main()
