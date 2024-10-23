'''
[desc]
contains all the model for stable diffusion

@author Jamie Westerhout
@project Stable Diffusion
'''
import torch
import torch.nn as nn
import math
import numpy as np


class VAEDecoder(nn.Module):
    '''
        class for decoder half of the vae
        use to reconstruct images from latent space

        Reduces size by a factor of 8
    '''
    def __init__(self, latent_dim=256, channels=1):
        super(VAEDecoder, self).__init__()
        self.decoder_size = 64
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, self.decoder_size, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(self.decoder_size, self.decoder_size//2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(self.decoder_size//2, channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        '''
            forward through decoder to decode the input
        '''
        x_recon = self.decoder(z)
        return x_recon

class VAEEncoder(nn.Module):
    '''
        class for Encoder half of the vae
        use to encode images in the latent space
    '''
    def __init__(self, latent_dim=256, image_size=200, channels=1):
        super(VAEEncoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.encoder_size = 64
        self.output_image_size = image_size//8
        
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, self.encoder_size//2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.encoder_size//2, self.encoder_size, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.encoder_size, latent_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(), # Output: (batch_szie, latent_dim, image_size/8, image_size/8)
            nn.Flatten(), # Output: (batch_szie, latent_dim * image_size/8 * image_size/8)
        )
        
        self.fc_mu = nn.Linear(latent_dim * self.output_image_size * self.output_image_size, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim * self.output_image_size * self.output_image_size, latent_dim)

        self.decoder_fc = nn.Linear(latent_dim, latent_dim * self.output_image_size * self.output_image_size)

    def forward(self, x):
        '''
            forward through encoder
        '''
        # get mean and variance for sampling
        # and to be used to calculate Kullback-Leibler divergence loss.
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        # get sample
        z = self.sample_latent(mu, logvar)
        h = self.decode(z)

        return h, mu, logvar
    
    def decode(self,x):
        '''
            turn samples into something the decoder can take
        '''
        h = self.decoder_fc(x)
        h = h.view(-1, self.latent_dim , self.output_image_size, self.output_image_size)

        return h

    def sample_latent(self, mu, logvar):
        '''
            sampling latent form the distirbution produced by the vae
        '''
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def get_output_image_size(self):
        '''
            returns the images size after encoding
        '''
        return self.output_image_size

class CrossAttention(nn.Module):
    '''
        class for cross attention on the lables
        to add to the unet
    '''
    def __init__(self, embed_dim, heads):
        '''
            embed dim is the size of the of inputs
            num_heads is the unmber heads used the multiheadattention
        '''
        super(CrossAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=heads, batch_first=True)

    def forward(self, x, context):
        #shapping input
        b, c, h, w = x.size()
        x_flat = x.view(b, c, h * w).permute(0, 2, 1)
        context = context.unsqueeze(1).repeat(1, h * w, 1)
        
        #attention
        attn_output, _ = self.attention(x_flat, context, context)
        
        #reshapping output
        attn_output = attn_output.permute(0, 2, 1).view(b, c, h, w)
        
        return attn_output

# UNet for denoising
class UNet(nn.Module):
    '''
        Class Contains Unet for stable diffusion

    '''
    def __init__(self, latent_dim=256, num_classes=2, image_size = 25):
        '''
            letent dim: size of latent dim from vae
            num_classes: number of lables in this case 0: AD, 1: NC
        '''
        super(UNet, self).__init__()
        self.channel_size = 128
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.time_step_embedding_size = 32

        self.label_embedding = nn.Embedding(num_classes, self.channel_size*4)
        self.cross_attention = CrossAttention(embed_dim=self.channel_size*4, heads=4)

        # Down
        self.enc1 = nn.Sequential(
            nn.Conv2d(latent_dim, self.channel_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.channel_size),
            nn.ReLU(),
            nn.Conv2d(self.channel_size, self.channel_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.channel_size),
            nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(self.channel_size, self.channel_size * 2, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(self.channel_size * 2),
            nn.ReLU(),
            nn.Conv2d(self.channel_size * 2, self.channel_size * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.channel_size * 2),
            nn.ReLU()
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(self.channel_size * 2, self.channel_size * 4, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(self.channel_size * 4),
            nn.ReLU(),
            nn.Conv2d(self.channel_size * 4, self.channel_size * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.channel_size * 4),
            nn.ReLU()
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(self.channel_size * 4, self.channel_size * 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.channel_size * 8, self.channel_size * 4, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Up
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(self.channel_size * 4, self.channel_size * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.channel_size * 4),
            nn.ReLU(),
            nn.Conv2d(self.channel_size * 4, self.channel_size * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.channel_size * 2),
            nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(self.channel_size * 2, self.channel_size * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.channel_size * 2),
            nn.ReLU(),
            nn.Conv2d(self.channel_size * 2, self.channel_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.channel_size),
            nn.ReLU()
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(self.channel_size, self.channel_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.channel_size),
            nn.ReLU(),
            nn.Conv2d(self.channel_size, latent_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(latent_dim),
            nn.Sigmoid()
        )

        self.time_embedding_layer = nn.Linear(self.time_step_embedding_size, latent_dim)

    def forward(self, x, label, t):
        '''
            forwards pass through the unet
        '''
        # Time step embedding
        t_embedding = self.sinusodial_timestep_encoding(t, self.time_step_embedding_size).to(x.device)
        t_embedding = self.time_embedding_layer(t_embedding)  # Map time embedding to latent_dim
        t_embedding = t_embedding.unsqueeze(2).unsqueeze(3).repeat(x.shape[0], 1, self.image_size, self.image_size)

        label_embedding = label.view(x.shape[0], 1).expand(-1,self.channel_size*4).float()

        # Add in time embedings to the first layer
        enc1_out = self.enc1(torch.concat((x,t_embedding), dim=2))
        enc2_out = self.enc2(enc1_out)
        enc3_out = self.enc3(enc2_out)
        
        #bottle neck, add in label emeddings throuhg corss attention at the bottleneck
        bottleneck_out = self.bottleneck(enc3_out)
        bottleneck_out = self.cross_attention(bottleneck_out, label_embedding)

        # Pass through the decoder with skip connections
        dec3 = self.skip(self.dec3(bottleneck_out), enc2_out)
        dec2 = self.skip(self.dec2(dec3), enc1_out)
        dec1 = self.skip(self.dec1(dec2), x) 

        return dec1, bottleneck_out

    def skip(self, current, skip_connection):
        '''
            skip connections
            creates a connection betweent he current layer out and the skip connection layer out
            crops just in case sizes dont match
        '''
        upsampled_cropped = self.crop(current, skip_connection)
        return upsampled_cropped + skip_connection
    
    def crop(self, sample_i, sample_j):
        '''
            crop sample_i to match size of sample_j
        '''
        _, _, h, w = sample_j.size()
        sample_i = sample_i[:, :, :h, :w]
        return sample_i
    
    def sinusodial_timestep_encoding(self, time_step, embedding_dim):
        '''
            creates sinusodial postional encoding 
            of the current time step to be used
            in conditioning, so when runnign reverse diffusion the unet will
            output the appropriate predicted noise for the timestep
        '''
        scaled_time_step = torch.tensor(time_step, dtype=torch.float32).view(1, 1)  # Reshape to (1, 1)
        
        # Calculate the sinusoidal embeddings
        half_dim = embedding_dim // 2
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -(math.log(10000.0) / half_dim))
        emb = scaled_time_step * emb.unsqueeze(0)

        pos_enc = torch.zeros(1, embedding_dim, device=scaled_time_step.device)
        pos_enc[0, 0::2] = torch.sin(emb)  # sin for even indices
        pos_enc[0, 1::2] = torch.cos(emb)  # cos for odd indices
        
        return pos_enc