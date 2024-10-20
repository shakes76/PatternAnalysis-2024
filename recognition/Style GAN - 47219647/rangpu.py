from torchvision.transforms import ToPILImage
from stylegan2_pytorch import StyleGAN2
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import  datasets, transforms
import matplotlib.pyplot as plt


def one_hot_encode(labels, num_classes):
    return F.one_hot(labels, num_classes=num_classes)

def data_set_creator():
    augmentation_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((256, 256)),
        # No vertical flipping was applied
        transforms.RandomHorizontalFlip(),
        # To account for skewing
        transforms.RandomRotation(30),
        # Coloring was altered to stop overfitting
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor()
    ])

    # Change the directory to what is needed
    data_dir = 'recognition/Style GAN - 47219647/AD_NC/test'
    
    dataset = datasets.ImageFolder(root=data_dir, transform=augmentation_transforms)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True, num_workers=6)

    num_classes = len(dataset.classes)  

    def get_one_hot_encoded_loader():
        for batch_images, batch_labels in data_loader:
            one_hot_labels = one_hot_encode(batch_labels, num_classes)
            yield batch_images, one_hot_labels

    return get_one_hot_encoded_loader()




class StyleGan():

    def __init__(self, latent_dim = 512, chanels =1, network_capacity = 16) -> None:
        self.model = StyleGAN2(
            image_size=256,
            latent_dim=latent_dim,
            network_capacity= network_capacity
        )
        self.chanels = chanels
    
    def get_generator(self):
        return self.model.G 

    def get_discriminator(self):
        return self.model.D
    
    def get_style_vector(self):
        return self.model.SE
    
    def initialise_weight(self):
        self.model._init_weights()

    def move_to_device(self):
        self.model.G.to(self.device)
        self.model.D.to(self.device)
        self.model.SE.to(self.device)

    def sample_noise(self, batch_size, latent_dim):
        return torch.randn(batch_size, latent_dim).to(self.device)
    
    def sample_labels(self, batch_size, num_classes):
        labels = torch.randint(0, num_classes, (batch_size,)).to(self.device)
        return labels

    def forward_discriminator(self, real_images, fake_images):
        real_scores, _ = self.model.D(real_images)
        fake_scores, _ = self.model.D(fake_images)
        return real_scores, fake_scores

    def discriminator_loss(self, real_scores, fake_scores):
        real_loss = F.relu(1.0 - real_scores).mean()
        fake_loss = F.relu(1.0 + fake_scores).mean()
        return real_loss + fake_loss

    def generator_loss(self, fake_scores):
        return -fake_scores.mean()

    def save_checkpoint(self, epoch, path="gan_checkpoint.pth"):
        torch.save({
            'epoch': epoch,
            'generator_state_dict': self.model.G.state_dict(),
            'discriminator_state_dict': self.model.D.state_dict(),
            'optimizerG_state_dict': self.optimizerG.state_dict(),
            'optimizerD_state_dict': self.optimizerD.state_dict(),
        }, path)



def train_gan(model, dataloader, epochs=10, latent_dim=512, lr=1e-4):
    discriminator_losses = []
    generator_losses = []
    
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

            discriminator_losses.append(d_loss.item())
            generator_losses.appen(g_loss.item())

        print(f"Epoch [{epoch + 1}/{epochs}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")
        
        if (epoch + 1) % 5 == 0:
            model.save_checkpoint(epoch + 1, path=f"gan_checkpoint_epoch_{epoch + 1}.pth")
        
    return discriminator_losses, generator_losses

discriminator_losses, generator_losses = train_gan(model = StyleGan(), dataloader= data_set_creator(), epochs= 1)

