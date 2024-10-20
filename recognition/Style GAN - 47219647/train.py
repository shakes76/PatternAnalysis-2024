import torch
import torch.nn as nn
import torch.optim as optim
from dataset import data_set_creator
from modules import StyleGan

def train_gan(model, dataloader, epochs=10, latent_dim=512, lr=1e-4):
    
    # Move model to the correct device (GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.move_to_device()

    # Initialize weights of the model
    model.initialise_weight()

    # Optimizers for generator and discriminator
    optimizerG = optim.Adam(model.get_generator().parameters(), lr=lr, betas=(0.5, 0.9))
    optimizerD = optim.Adam(model.get_discriminator().parameters(), lr=lr * 2, betas=(0.5, 0.9))

    criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy loss for real vs fake classification

    num_classes = 2 

    for epoch in range(epochs):
        for real_images, one_hot_labels in dataloader:
            batch_size = real_images.size(0)

            # Move real images and labels to the device
            real_images = real_images.to(device)
            one_hot_labels = one_hot_labels.to(device)

            # Label tensors for real and fake images
            real_labels = torch.ones(batch_size, 1).to(device)  # Real labels = 1
            fake_labels = torch.zeros(batch_size, 1).to(device)  # Fake labels = 0

            
            noise = model.sample_noise(batch_size, latent_dim)

            optimizerD.zero_grad()

            fake_images = model.get_generator()(noise)
            
            real_scores, _ = model.get_discriminator()(real_images)
            fake_scores, _ = model.get_discriminator()(fake_images.detach()) 

            d_real_loss = criterion(real_scores, real_labels)
            d_fake_loss = criterion(fake_scores, fake_labels)
            d_loss = d_real_loss + d_fake_loss

            d_loss.backward()
            optimizerD.step()

            optimizerG.zero_grad()

            # Discriminator forward pass on fake images (using updated generator)
            fake_scores, _ = model.get_discriminator()(fake_images)

            g_loss = criterion(fake_scores, real_labels)

            g_loss.backward()
            optimizerG.step()

        print(f"Epoch [{epoch + 1}/{epochs}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")
        
        if (epoch + 1) % 5 == 0:
            model.save_checkpoint(epoch + 1, path=f"gan_checkpoint_epoch_{epoch + 1}.pth")

