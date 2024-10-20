"""
Contains training code for StyleGAN2.

Acknowledgements:
    https://github.com/aburo8/PatternAnalysis-2023/tree/topic-recognition/recognition/46990480_StyleGAN2
    https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
"""
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import random
import torch.nn.parallel
import torch.optim as optim
from tqdm import tqdm
from modules import Generator, Discriminator, FCBlock
from dataset import process_adni, process_cifar
import math

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

def train(model_G, model_D, latent_net, n_epochs, dataloader, lr,
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
    optimM = optim.Adam(latent_net.parameters(), lr=lr)

    def WGAN_GP_LOSS(discriminator, real, fake, device="cpu"):
        '''
        Computes gradient penalty (loss) for WGAN-GP
        '''
        BATCH_SIZE, C, H, W = real.shape
        beta = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
        interpolated_images = real * beta + fake.detach() * (1 - beta)
        interpolated_images.requires_grad_(True)

        # Calculate discriminator scores
        mixed_scores = discriminator(interpolated_images)

        # Take the gradient of the scores with respect to the images
        gradient = torch.autograd.grad(
            inputs=interpolated_images,
            outputs=mixed_scores,
            grad_outputs=torch.ones_like(mixed_scores),
            create_graph=True,
            retain_graph=True,
        )[0]
        gradient = gradient.view(gradient.shape[0], -1)
        gradient_norm = gradient.norm(2, dim=1)
        gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
        return gradient_penalty

    # -- Training Loop --

    print("> Starting Training Loop")
    g_losses = []
    d_losses = []

    for epoch in range(n_epochs):
        epoch_g_loss = []
        epoch_d_loss = []
        for real_images, _ in tqdm(dataloader, desc=f"Epoch {epoch+1}/{n_epochs}"):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)

            # Get intermediate latent vector and noise
            w = get_w(w_dim, batch_size, num_layers, latent_net, device)
            noise = get_noise(num_layers, batch_size, device)
            with torch.cuda.amp.autocast():
                # Generate a fake image batch with the Generator
                fake = model_G(w, noise)

                # Forward pass the fake image through the discriminator
                discriminator_fake = model_D(fake.detach())

                # Forward pass the real image through the discriminator
                discriminator_real = model_D(real_images)
                criterion = WGAN_GP_LOSS(model_D, real_images, fake, device=device)
                loss_discriminator = (
                    -(torch.mean(discriminator_real) - torch.mean(discriminator_fake))
                    + 10 * criterion
                    + (0.001 * torch.mean(discriminator_real ** 2))
                )

                # Update Discriminator Neural Network => maximize log(D(x)) + log(1 - D(G(z)))
                model_D.zero_grad()
                loss_discriminator.backward()
                optimD.step()

                # Forward pass the fake batch of generated images through the discriminator
                gen_fake = model_D(fake)

                # Compute the generator loss
                loss_gen = -torch.mean(gen_fake)

                # Update the networks
                latent_net.zero_grad()
                model_G.zero_grad()
                loss_gen.backward()
                optimG.step()
                optimM.step()
                epoch_g_loss.append(criterion.item())
                epoch_d_loss.append(loss_discriminator.item())

        # Compute Average losses
        g_losses.append(sum(epoch_g_loss)/len(epoch_g_loss))
        d_losses.append(sum(epoch_d_loss)/len(epoch_d_loss))
        print(f"Epoch [{epoch+1}/{n_epochs}], D Loss: {(sum(epoch_d_loss)/len(epoch_d_loss)):.4f}, G Loss: {(sum(epoch_g_loss)/len(epoch_g_loss)):.4f}")


    print("> Training Complete")

    return model_G, model_D

def main():
    # Needed for reproducible results
    manual_seed = 46915474
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.use_deterministic_algorithms(True)

    model_name = "stylegan2-cifar"
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
    channels = 3
    z_dim = 512
    w_dim = 512
    image_size = 32
    num_layers = int(math.log2(image_size))
    data_name = 'cifar'

    if data_name == "adni":
        image_size = 240
        batch_size = 32
        dataset, dataloader = process_adni(batch_size=batch_size)
    else:
        dataset, dataloader = process_cifar(batch_size=batch_size)

    # -- Initialise Models --
    latent_net = FCBlock(z_dim, w_dim)
    model_G = Generator(num_layers, w_dim).to(device)
    model_D = Discriminator(num_layers).to(device)

    latent_net = latent_net.to(device)
    model_G = model_G.to(device)
    model_D = model_D.to(device)

    # Train model
    model_G, model_D = train(model_G, model_D, latent_net, num_epochs, dataloader, lr,
        z_dim, w_dim, channels, num_layers, image_size, device)

if __name__ == "__main__":
    main()
