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

class StyleGan():

    def __init__(self, latent_dim = 512, chanels =1, network_capacity = 16) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = StyleGAN2(
            image_size=256,
            latent_dim=latent_dim,
            network_capacity=network_capacity
        ).to(self.device)
        self.chanels = chanels
    
    def get_generator(self, noise):
        return self.model.G(noise)

    def get_discriminator(self, image):
        return self.model.D(image)
    
    def get_style_vector(self):
        return self.model.SE
    
    def get_G_optim(self):
        return self.model.G_opt
    
    def get_D_optim(self):
        return self.model.D_opt
    
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

