import imageio
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
import einops

import dataset as ds


class Diffusion(nn.Module):
    def __init__(self, nwork, n_steps=200, min_beta=10**-4, max_beta=0.02, device=None, image_chw=(1,64,64)):
        super(Diffusion, self).__init__()
        self.n_steps = n_steps
        self.device = device
        self.image_chw = image_chw
        self.network = nwork
        self.betas = torch.linspace(min_beta, max_beta, n_steps).to(device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.tensor([torch.prod(self.alphas[:i + 1]) for i in range(len(self.alphas))]).to(device)

    def forward(self, x_0, t, zeta=None):
        """Add noise to input image"""
        n, c, h, w = x_0.shape
        a_bar = self.alpha_bars[t]

        if zeta is None:
            zeta = torch.randn(n, c, h, w).to(self.device)

        noise = a_bar.sqrt().reshape(n, 1, 1, 1) * x_0 + (1 - a_bar).sqrt().reshape(n, 1, 1, 1) * zeta
        return noise

    def backward(self, x, t):
        """Learn reverse diffusion process by predicting the added noise"""
        return self.network(x,t)

def show_forward(diff, loader, device):
    # Showing the forward process
    for batch in loader:
        imgs = batch[0]

        ds.show_images(imgs, "Original images")

        # for percent in [0.25, 0.5, 0.75, 1]:
        #     ds.show_images(
        #         diff(imgs.to(device),
        #              [int(percent * diff.n_steps) - 1 for _ in range(len(imgs))]),
        #         f"DDPM Noisy images {int(percent * 100)}%")
        break

def generate_new_images(diff, n_samples=16, device=None, c=1, h=64, w=64):
    """Start with random noise and backtrack to t=0
    zeta_tau is estimate of noise and apply denoising"""
    frame_idxs = np.linspace(0, diff.n_steps).astype(np.uint)
    frames = []

    with torch.no_grad():
        if device is None:
            device = diff.device

        # Starting from random noise
        x = torch.randn(n_samples, c, h, w).to(device)

        for idx, t in enumerate(list(range(diff.n_steps))[::-1]):
            # Estimating noise to be removed
            time_tensor = (torch.ones(n_samples, 1) * t).to(device).long()
            zeta_tau = diff.backward(x, time_tensor)

            alpha_t = diff.alphas[t]
            alpha_t_bar = diff.alpha_bars[t]

            # Partially denoising the image
            x = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * zeta_tau)

            if t > 0:
                z = torch.randn(n_samples, c, h, w).to(device)

                # Option 1: sigma_t squared = beta_t
                beta_t = diff.betas[t]
                sigma_t = beta_t.sqrt()

                # Option 2: sigma_t squared = beta_tilda_t
                # prev_alpha_t_bar = ddpm.alpha_bars[t-1] if t > 0 else ddpm.alphas[0]
                # beta_tilda_t = ((1 - prev_alpha_t_bar)/(1 - alpha_t_bar)) * beta_t
                # sigma_t = beta_tilda_t.sqrt()

                # Adding some more noise like in Langevin Dynamics fashion
                x = x + sigma_t * z

            # Adding frames to the GIF
            if idx in frame_idxs or t == 0:
                # Putting digits in range [0, 255]
                normalized = x.clone()
                for i in range(len(normalized)):
                    normalized[i] -= torch.min(normalized[i])
                    normalized[i] *= 255 / torch.max(normalized[i])

                # Reshaping batch (n, c, h, w) to be a (as much as it gets) square frame
                frame = einops.rearrange(normalized, "(b1 b2) c h w -> (b1 h) (b2 w) c", b1=int(n_samples ** 0.5))
                frame = frame.cpu().numpy().astype(np.uint8)

                # Rendering frame
                frames.append(frame)

        return x


class BasicBlock(nn.Module):
        def __init__(self, shape, in_c, out_c, kernel_size=3, stride=1, padding=1, activation=None, normalize=True):
            super(BasicBlock, self).__init__()
            self.ln = nn.LayerNorm(shape)
            self.conv1 = nn.Conv2d(in_c, out_c, kernel_size, stride, padding)
            self.conv2 = nn.Conv2d(out_c, out_c, kernel_size, stride, padding)
            self.activation = nn.SiLU() if activation is None else activation
            self.normalize = normalize

        def forward(self, x):
            out = self.ln(x) if self.normalize else x
            out = self.conv1(out)
            out = self.activation(out)
            out = self.conv2(out)
            out = self.activation(out)
            return out

def sinusoidal_embedding(n, d):
    # Returns the standard positional embedding
    embedding = torch.zeros(n, d)
    wk = torch.tensor([1 / 10_000 ** (2 * j / d) for j in range(d)])
    wk = wk.reshape((1, d))
    t = torch.arange(n).reshape((n, 1))
    embedding[:,::2] = torch.sin(t * wk[:,::2])
    embedding[:,1::2] = torch.cos(t * wk[:,::2])

    return embedding

class UNet(nn.Module):
    def __init__(self, n_steps=1000, time_emb_dim=100):
        super(UNet, self).__init__()

        # Sinusoidal embedding
        self.time_embed = nn.Embedding(n_steps, time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed.requires_grad_(False)

        # First half
        self.te1 = self.posi_map(time_emb_dim, 1)
        self.b1 = nn.Sequential(
            BasicBlock((1, 64, 64), 1, 10),
            BasicBlock((10, 64, 64), 10, 10),
            BasicBlock((10, 64, 64), 10, 10)
        )
        self.down1 = nn.Conv2d(10, 10, 4, 2, 1)

        self.te2 = self.posi_map(time_emb_dim, 10)
        self.b2 = nn.Sequential(
            BasicBlock((10, 32, 32), 10, 20),
            BasicBlock((20, 32, 32), 20, 20),
            BasicBlock((20, 32, 32), 20, 20)
        )
        self.down2 = nn.Conv2d(20, 20, 4, 2, 1)

        self.te3 = self.posi_map(time_emb_dim, 20)
        self.b3 = nn.Sequential(
            BasicBlock((20, 16, 16), 20, 40),
            BasicBlock((40, 16, 16), 40, 40),
            BasicBlock((40, 16, 16), 40, 40)
        )
        self.down3 = nn.Conv2d(40, 40, 4, 2, 1)

        # Bottleneck
        self.te_mid = self.posi_map(time_emb_dim, 40)
        self.b_mid = nn.Sequential(
            BasicBlock((40, 8, 8), 40, 20),
            BasicBlock((20, 8, 8), 20, 20),
            BasicBlock((20, 8, 8), 20, 40)
        )

        # Second half
        self.up1 = nn.ConvTranspose2d(40, 40, 4, 2, 1)

        self.te4 = self.posi_map(time_emb_dim, 80)
        self.b4 = nn.Sequential(
            BasicBlock((80, 16, 16), 80, 40),
            BasicBlock((40, 16, 16), 40, 20),
            BasicBlock((20, 16, 16), 20, 20)
        )

        self.up2 = nn.ConvTranspose2d(20, 20, 4, 2, 1)
        self.te5 = self.posi_map(time_emb_dim, 40)
        self.b5 = nn.Sequential(
            BasicBlock((40, 32, 32), 40, 20),
            BasicBlock((20, 32, 32), 20, 10),
            BasicBlock((10, 32, 32), 10, 10)
        )

        self.up3 = nn.ConvTranspose2d(10, 10, 4, 2, 1)
        self.te_out = self.posi_map(time_emb_dim, 20)
        self.b_out = nn.Sequential(
            BasicBlock((20, 64, 64), 20, 10),
            BasicBlock((10, 64, 64), 10, 10),
            BasicBlock((10, 64, 64), 10, 10, normalize=False)
        )

        self.conv_out = nn.Conv2d(10, 1, 3, 1, 1)

    def forward(self, x, t):
        # x is (N, 2, 64, 64) (image with positional embedding stacked on channel dimension)
        t = self.time_embed(t)
        n = len(x)
        out1 = self.b1(x + self.te1(t).reshape(n, -1, 1, 1))  # (N, 10, 64, 64)
        out2 = self.b2(self.down1(out1) + self.te2(t).reshape(n, -1, 1, 1))  # (N, 20, 32, 32)
        out3 = self.b3(self.down2(out2) + self.te3(t).reshape(n, -1, 1, 1))  # (N, 40, 16, 16)

        out_mid = self.b_mid(self.down3(out3) + self.te_mid(t).reshape(n, -1, 1, 1))  # (N, 40, 8, 8)

        out4 = torch.cat((out3, self.up1(out_mid)), dim=1)  # (N, 80, 16, 16)
        out4 = self.b4(out4 + self.te4(t).reshape(n, -1, 1, 1))  # (N, 20, 16, 16)

        out5 = torch.cat((out2, self.up2(out4)), dim=1)  # (N, 40, 32, 32)
        out5 = self.b5(out5 + self.te5(t).reshape(n, -1, 1, 1))  # (N, 10, 32, 32)

        out = torch.cat((out1, self.up3(out5)), dim=1)  # (N, 20, 64, 64)
        out = self.b_out(out + self.te_out(t).reshape(n, -1, 1, 1))  # (N, 1, 64, 64)

        out = self.conv_out(out)

        return out

    def posi_map(self, dim_in, dim_out):
        return nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.SiLU(),
            nn.Linear(dim_out, dim_out))

# Defining model
n_steps, min_beta, max_beta = 1000, 10 ** -4, 0.02  # Originally used by the authors
ddpm = Diffusion(UNet(n_steps), n_steps=n_steps, min_beta=min_beta, max_beta=max_beta, device=ds.device)

show_forward(ddpm, ds.dataloader, ds.device)

generated = generate_new_images(ddpm)
ds.show_images(generated, "Images")