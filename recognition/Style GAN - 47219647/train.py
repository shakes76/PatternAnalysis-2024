import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt

from torchvision.utils import save_image
from tqdm import tqdm
from sklearn.manifold import TSNE

from modules import *
from params import *
from dataset import *

class StyleGAN_Trainer:

    def __init__(self):
        self.gen = Generator(Z_DIM, W_DIM, IN_CHANNELS, img_channels=CHANNELS_IMG).to(DEVICE)
        self.disc = Discriminator(IN_CHANNELS, img_channels=CHANNELS_IMG).to(DEVICE)
        self.step = 0
        self.training_img_num = 3
        self.current_batch_size = BATCH_SIZES[self.current_img_size]
        self.current_img_size = IMAGE_SIZES[self.step]
        self.loader, self.dataset = data_set_creator(image_size = self.current_img_size, batch_size= self.current_batch_size)
        self.disc_losses = []
        self.gen_losses = []
        self.device = DEVICE
        self.blend_factor = BLEND_FACTOR
        self.betas = BETAS

        self.opt_gen = optim.Adam([
                {"params": [param for name, param in self.gen.named_parameters() if "map" not in name]},
                {"params": self.gen.map.parameters(), "lr": 1e-5}
            ], lr=self.current_learning_rate, betas= self.betas)

        self.opt_disc = optim.Adam(self.disc.parameters(), lr=self.current_learning_rate, betas= self.betas)

    def generate_examples(self):
        self.gen.eval()
        for i in range(self.training_img_num):
            with torch.no_grad():
                img = self.gen(torch.randn(1, Z_DIM).to(DEVICE), 1 , self.steps)
                if not os.path.exists(f'saved_examples/step{self.steps}'):
                    os.makedirs(f'saved_examples/step{self.steps}')
                save_image(img * 0.5 + 0.5, f"saved_examples/step{self.steps}/img_{i}.png")
        self.gen.train()

    # Function to compute gradient penalty for WGAN-GP
    def gradient_penalty(self, real, fake):
        batch_size , C, H, W = real.shape
        beta = torch.rand((batch_size, 1, 1, 1)).repeat(1, C, H, W).to(DEVICE)
        interpolated_images = real * beta + fake.detach() * (1 - beta)
        interpolated_images.requires_grad_(True)

        mixed_scores = self.disc(interpolated_images, self.blend_factor, self.step)
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


    def train(self):
        loop = tqdm(self.loader, leave=True)

        for batch_idx, (real, _) in enumerate(loop):
            real = real.to(DEVICE)
            cur_batch_size = real.shape[0]

            noise = torch.randn(cur_batch_size, Z_DIM).to(DEVICE)

            # Forward passes
            fake = self.gen(noise, self.blend_factor, self.step)
            disc_real = self.disc(real, self.blend_factor, self.step)
            disc_fake = self.disc(fake.detach(), self.blend_factor, self.step)

            # Gradient penalty and discriminator loss
            gp = self.gradient_penalty( real, fake, self.blend_factor)

            loss_disc = (- (torch.mean(disc_real) - torch.mean(disc_fake))
                        + LAMBDA_GP * gp
                        + (0.001 * torch.mean(disc_real ** 2)))

            # Update discriminator
            self.disc.zero_grad()
            loss_disc.backward()
            self.opt_disc.step()

            # Generator loss
            gen_fake = self.disc(fake, self.blend_factor, self.step)
            loss_gen = -torch.mean(gen_fake)

            # Update generator
            self.gen.zero_grad()
            loss_gen.backward()
            self.opt_gen.step()

            # Update blend factor (progressive growing blending factor)
            updated_blend_factor  = cur_batch_size / ((PROGRESSIVE_EPOCHS[self.current_image_size] * 0.5) * len(self.dataset))+ self.blend_factor
            self.blend_factor = min( updated_blend_factor,1)

            # Append losses to lists
            self.disc_losses.append(loss_disc.item())
            self.gen_losses.append(loss_gen.item())

            # Display loss in tqdm loop
            loop.set_postfix(gp=gp.item(), loss_disc=loss_disc.item())



    def main_training_loop(self):
        # Main training loop over image sizes
        for step, current_image_size in enumerate(IMAGE_SIZES):
            self.current_batch_size = BATCH_SIZES[current_image_size]
            self.current_learning_rate = LEARNING_SIZES[current_image_size]

            self.step = step

            print(f"Training at image size {current_image_size} with batch size {self.current_batch_size} and learning rate {self.current_learning_rate}")

            # Initialize generator and discriminator
            self.gen = Generator(Z_DIM, W_DIM, IN_CHANNELS, img_channels=CHANNELS_IMG).to(DEVICE)
            self.disc = Discriminator(IN_CHANNELS, img_channels=CHANNELS_IMG).to(DEVICE)

            # Initialize optimizers
            self.opt_gen = optim.Adam([
                {"params": [param for name, param in self.gen.named_parameters() if "map" not in name]},
                {"params": self.gen.map.parameters(), "lr": 1e-5}
            ], lr=self.current_learning_rate, betas= self.betas)

            self.opt_disc = optim.Adam(self.disc.parameters(), lr=self.current_learning_rate, betas= self.betas)

            self.gen.train()
            self.disc.train()

            # Get data loader for the current image size
            self.loader, self.dataset = data_set_creator(image_size=current_image_size, batch_size=self.current_batch_size)

            for epoch in range(PROGRESSIVE_EPOCHS[self.current_image_size]):
                print(f"Epoch [{epoch + 1}/{PROGRESSIVE_EPOCHS[self.current_image_size]}] at image size {self.current_image_size}x{self.current_image_size}")

                self.train()
           
            self.generate_examples()
            
            save_model(self.gen, self.disc, self.opt_gen, self.opt_disc, epoch, step, self.disc_losses, self.gen_losses, file_path=f"model_checkpoint_step{self.step}_epoch{epoch}.pth")


