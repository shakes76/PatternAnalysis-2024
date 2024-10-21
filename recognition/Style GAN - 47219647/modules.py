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

# class StyleGan():

#     def __init__(self, latent_dim = 512, chanels =1, network_capacity = 16) -> None:
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#         self.model = StyleGAN2(
#             image_size=256,
#             latent_dim=latent_dim,
#             network_capacity=network_capacity
#         ).to(self.device)
#         self.chanels = chanels
    
#     def get_generator(self, noise):
#         return self.model.G(noise)

#     def get_discriminator(self, image):
#         return self.model.D(image)
    
#     def get_style_vector(self):
#         return self.model.SE
    
#     def get_G_optim(self):
#         return self.model.G_opt
    
#     def get_D_optim(self):
#         return self.model.D_opt
    
#     def initialise_weight(self):
#         self.model._init_weights()


#     def move_to_device(self):
#         self.model.G.to(self.device)
#         self.model.D.to(self.device)
#         self.model.SE.to(self.device)

#     def sample_noise(self, batch_size, latent_dim):
#         return torch.randn(batch_size, latent_dim).to(self.device)
    
#     def sample_labels(self, batch_size, num_classes):
#         labels = torch.randint(0, num_classes, (batch_size,)).to(self.device)
#         return labels

#     def forward_discriminator(self, real_images, fake_images):
#         real_scores, _ = self.model.D(real_images)
#         fake_scores, _ = self.model.D(fake_images)
#         return real_scores, fake_scores

#     def discriminator_loss(self, real_scores, fake_scores):
#         real_loss = F.relu(1.0 - real_scores).mean()
#         fake_loss = F.relu(1.0 + fake_scores).mean()
#         return real_loss + fake_loss

#     def generator_loss(self, fake_scores):
#         return -fake_scores.mean()

#     def save_checkpoint(self, epoch, path="gan_checkpoint.pth"):
#         torch.save({
#             'epoch': epoch,
#             'generator_state_dict': self.model.G.state_dict(),
#             'discriminator_state_dict': self.model.D.state_dict(),
#             'optimizerG_state_dict': self.optimizerG.state_dict(),
#             'optimizerD_state_dict': self.optimizerD.state_dict(),
#         }, path)

class Generator(nn.Module):
    def __init__(self, image_size, latent_dim, network_capacity = 16, transparent = False, attn_layers = [], no_const = False, fmap_max = 512):
        super().__init__()
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.num_layers = int(log2(image_size) - 1)

        filters = [network_capacity * (2 ** (i + 1)) for i in range(self.num_layers)][::-1]

        set_fmap_max = partial(min, fmap_max)
        filters = list(map(set_fmap_max, filters))
        init_channels = filters[0]
        filters = [init_channels, *filters]

        in_out_pairs = zip(filters[:-1], filters[1:])
        self.no_const = no_const

        if no_const:
            self.to_initial_block = nn.ConvTranspose2d(latent_dim, init_channels, 4, 1, 0, bias=False)
        else:
            self.initial_block = nn.Parameter(torch.randn((1, init_channels, 4, 4)))

        self.initial_conv = nn.Conv2d(filters[0], filters[0], 3, padding=1)
        self.blocks = nn.ModuleList([])
        self.attns = nn.ModuleList([])

        for ind, (in_chan, out_chan) in enumerate(in_out_pairs):
            not_first = ind != 0
            not_last = ind != (self.num_layers - 1)
            num_layer = self.num_layers - ind

            attn_fn = attn_and_ff(in_chan) if num_layer in attn_layers else None

            self.attns.append(attn_fn)

            block = GeneratorBlock(
                latent_dim,
                in_chan,
                out_chan,
                upsample = not_first,
                upsample_rgb = not_last,
                rgba = transparent
            )
            self.blocks.append(block)

    def forward(self, styles, input_noise):
        batch_size = styles.shape[0]
        image_size = self.image_size

        if self.no_const:
            avg_style = styles.mean(dim=1)[:, :, None, None]
            x = self.to_initial_block(avg_style)
        else:
            x = self.initial_block.expand(batch_size, -1, -1, -1)

        rgb = None
        styles = styles.transpose(0, 1)
        x = self.initial_conv(x)

        for style, block, attn in zip(styles, self.blocks, self.attns):
            if exists(attn):
                x = attn(x)
            x, rgb = block(x, rgb, style, input_noise)

        return rgb