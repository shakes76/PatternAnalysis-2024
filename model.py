import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torchvision import transforms
import numpy as np
from pytorch_msssim import SSIM

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
        if in_channels != out_channels:
            self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.projection = None

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.projection is not None:
            identity = self.projection(x)
            
        out += identity
        out = self.relu(out)
        return out

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)

    def forward(self, inputs):
        # Convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # Convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings

class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_dims=[32, 64, 128, 256]):
        super().__init__()
        modules = []
        
        for h_dim in hidden_dims:
            modules.append(
                ResidualBlock(in_channels, h_dim)
            )
            modules.append(nn.MaxPool2d(kernel_size=2))
            in_channels = h_dim
            
        self.encoder = nn.Sequential(*modules)

    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, out_channels, hidden_dims=[256, 128, 64, 32]):
        super().__init__()
        modules = []
        in_channels = hidden_dims[0]
        
        for h_dim in hidden_dims[1:]:
            modules.append(nn.Upsample(scale_factor=2))
            modules.append(
                ResidualBlock(in_channels, h_dim)
            )
            in_channels = h_dim
            
        modules.append(
            nn.Conv2d(hidden_dims[-1], out_channels, kernel_size=3, padding=1)
        )
        
        self.decoder = nn.Sequential(*modules)

    def forward(self, x):
        return self.decoder(x)

class VQVAE(pl.LightningModule):
    def __init__(
        self,
        in_channels=1,
        hidden_dims=[32, 64, 128, 256],
        num_embeddings=512,
        embedding_dim=256,
        commitment_cost=0.25,
        learning_rate=1e-4
    ):
        super().__init__()
        
        self.encoder = Encoder(in_channels, hidden_dims)
        self.vq = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.decoder = Decoder(in_channels, hidden_dims[::-1])
        self.learning_rate = learning_rate
        
        self.ssim_module = SSIM(data_range=1.0, size_average=True, channel=1)

    def forward(self, x):
        encoded = self.encoder(x)
        vq_loss, quantized, perplexity, _ = self.vq(encoded)
        reconstructed = self.decoder(quantized)
        return reconstructed, vq_loss, perplexity

    def training_step(self, batch, batch_idx):
        x = batch
        reconstructed, vq_loss, perplexity = self(x)
        
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(reconstructed, x)
        
        # SSIM loss
        ssim_loss = 1 - self.ssim_module(reconstructed, x)
        
        # Total loss
        total_loss = recon_loss + vq_loss + ssim_loss
        
        # Calculate SSIM metric
        ssim_value = 1 - ssim_loss.item()
        
        self.log('train_loss', total_loss)
        self.log('train_recon_loss', recon_loss)
        self.log('train_vq_loss', vq_loss)
        self.log('train_perplexity', perplexity)
        self.log('train_ssim', ssim_value)
        
        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

# Training setup
def train_vqvae(data_path, batch_size=32, num_epochs=100):
    # Data preprocessing
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Initialize model and trainer
    model = VQVAE()
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                monitor='train_ssim',
                mode='max',
                filename='vqvae-{epoch:02d}-{train_ssim:.2f}',
                save_top_k=3
            )
        ]
    )
    
    # Train model
    trainer.fit(model, train_dataloader)
    
    return model

model = train_vqvae("...")