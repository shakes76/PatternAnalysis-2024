from dataset import load_data, show_example_images
import torch
import torchvision
import torch.nn as nn
from tqdm import tqdm
from einops import rearrange
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
import utils
import os

class ResidualBlock(nn.Module):
    """
    One residual layer inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    """

    def __init__(self, in_dim):
        super(ResidualBlock, self).__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_dim, in_dim, 3, 1, 1),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(True),
            nn.Conv2d(in_dim, in_dim, 1),
            nn.BatchNorm2d(in_dim),
        )

    def forward(self, x):
        x = x + self.res_block(x)
        return x
    
class VectorQuantizer(nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE.

    Inputs:
    - n_e : number of embeddings
    - dim_e : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    """

    def __init__(self, n_e, dim_e, beta=0.25):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.dim_e = dim_e
        self.beta = beta # The paper uses 0.25

        self.embedding = nn.Embedding(self.n_e, self.dim_e)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete 
        one-hot vector that is the index of the closest embedding vector e_j

        z (continuous) -> z_q (discrete)

        z.shape = (batch, channel, height, width)

        quantization pipeline:

            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)

        """
        # reshape z -> (batch, height, width, channel) and flatten
        embeddings = self.embedding.weight
        z_shaped = z.permute(0, 2, 3, 1).contiguous().view(-1, self.dim_e)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        distances = torch.sum(z_shaped ** 2, dim=1, keepdim=True) + \
            torch.sum(embeddings**2, dim=1) - 2 * \
            torch.matmul(z_shaped, embeddings.t())

        # find closest encodings
        min_encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e).to(device)
        min_encodings.scatter_(1, min_encoding_indices, 1)
        
        loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
            torch.mean((z_q - z.detach()) ** 2)
        
        dists = dists.view(z.shape[0], z.shape[2], z.shape[3], -1)
        z_q = dists.min(-1)[1]
        return z_q, loss


class ResidualStack(nn.Module):
    """
    A stack of residual layers inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack
    """

    def __init__(self, in_dim, n_res_blocks):
        super(ResidualStack, self).__init__()
        self.n_res_blocks = n_res_blocks
        self.stack = nn.ModuleList(
            [ResidualBlock(in_dim)]*n_res_blocks)

    def forward(self, x):
        for layer in self.stack:
            x = layer(x)
        x = F.relu(x)
        return x

class VQVAE(nn.Module):
    def __init__(self, in_dim, dim, n_res_layers, num_embeddings=512):
        super(VQVAE, self).__init__()
        
        # Adjust the input channels in the encoder from 1 to 64
        self.encoder = nn.Sequential(
            nn.Conv2d(in_dim, in_dim // 2, 4, 2, 1),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(),
            nn.Conv2d(in_dim // 2, in_dim, 4, 2, 1),
            nn.ReLU(),
            # here we have stride of 1, kernal of 2
            nn.Conv2d(in_dim, in_dim, 3, 1, 1),
            ResidualStack(in_dim, n_res_layers)
        )
        
        self.pre_quant_conv = nn.Conv2d(4, 2, kernel_size=1)
        self.codebook = VectorQuantizer(num_embeddings, dim)
        self.post_quant_conv = nn.Conv2d(2, 4, kernel_size=1)
        
        # Commitment Loss Beta
        self.beta = 0.25 # From paper
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                in_dim, in_dim, 3, 1, 1),
            ResidualStack(in_dim, n_res_layers),
            nn.ConvTranspose2d(in_dim, in_dim//2, 4, 2, 1),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(),
            nn.ConvTranspose2d(in_dim//2, 3, 4, 2, 1),
            nn.Tanh()
        )

        
    def forward(self, x):
        # Pass through the encoder
        encoded_output = self.encoder(x)  # Output shape: [B, C, H', W']
        
        # Pre-quantization convolution
        quant_input = self.pre_quant_conv(encoded_output)  # Still 4D: [B, C, H', W']
        
        latents, quantize_loss = self.codebook(quant_input)  # Output shape: [B, H', W']
        
        # Post-quantization convolution
        decoder_input = self.post_quant_conv(latents)
        
        # Decoder part: input to decoder must be 4D
        decoded_output = self.decoder(decoder_input)  # Decoder output in 4D
        
        return encoded_output, decoded_output, latents, quantize_loss

# Initialize
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
train_loader, test_loader, val_loader = load_data()
model = VQVAE().to(device)
if not os.path.exists('models'):
    os.makedirs('models')

ssim_scores = []
train_losses = []
best_epoch = 0

def train_vqvae():
    num_epochs = 7
    optimizer = Adam(model.parameters(), lr=1E-3)
    criterion = torch.nn.MSELoss()
    
    for epoch_idx in range(num_epochs):
        epoch_loss = 0
        epoch_start = time.time()
        for batch, im in enumerate(tqdm(train_loader)):
            start_time = time.time()
            im = im.float().unsqueeze(1).to(device)
            optimizer.zero_grad()
            
            encoded_output, decoded_output, latents, quantize_loss = model(im)
            
            model_loss = criterion(encoded_output, decoded_output)            
            loss = model_loss + quantize_loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
            if (batch + 1) % 50 == 0:
                print('\tIter [{}/{} ({:.0f}%)]\tLoss: {} Time: {}'.format(
                    batch * len(im), len(train_loader.dataset),
                    50 * batch / len(train_loader),
                    epoch_loss/batch, axis=0),
                    time.time() - start_time
                )
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_epoch_loss)
        print('Finished epoch {} in time: {} with loss:'.format(
            epoch_idx + 1, epoch_start - time.time(), avg_epoch_loss))
        
        validate(epoch_idx+1)

    print('Done Training...')

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.grid()
    plt.savefig('training_loss.png')
    plt.close()
    
    return model

def validate(epoch):
    model.eval()
    total_ssim = 0
    
    with torch.no_grad():
        for batch, (x, _) in enumerate(val_loader):
            x = x.to(device)
            
            _, decoded_output, _, _ = model(x)
            
            total_ssim += utils.calc_ssim(decoded_output, x)
        
    epoch_ssim_score = total_ssim/(batch+1)
    ssim_scores.append(epoch_ssim_score)
    if epoch_ssim_score > max(ssim_scores):
        torch.save(model.state_dict(), f'models/checkpoint_epoch{epoch}_vqvae.pt')
        best_epoch = epoch
        print(f"Achieved an SSIM score of {epoch_ssim_score}, NEW BEST! saving model")
    else:
        print(f"Achieved an SSIM score of {epoch_ssim_score}")

def test():
    model.load_state_dict(torch.load(f'models/checkpoint_epoch{best_epoch}_vqvae.pt'))
    model.eval()
    total_ssim = 0
    
    with torch.no_grad():
        for batch, (x, _) in enumerate(test_loader):
            x = x.to(device)
            
            _, decoded_output, _, _ = model(x)
            
            total_ssim += utils.calc_ssim(decoded_output, x)
        
    total_test_ssim = total_ssim/(batch + 1)
    return total_test_ssim
    

def reconstruct_images():
    model.eval()
    with torch.no_grad():        
        for im in test_loader:
            ims = im.float().unsqueeze(1).to(device)
            break

        generated_im, _, _, _ = model(ims)

        ims = (ims + 1) / 2
        generated_im = 1 - (generated_im + 1) / 2

        out = torch.hstack([ims, generated_im])
        output = rearrange(out, 'b c h w -> b () h (c w)')

        grid = torchvision.utils.make_grid(output.detach().cpu(), nrow=10)
        img = torchvision.transforms.ToPILImage()(grid)
        img.save('reconstruction2.png')

    print('Done Reconstruction ...')


if __name__ == "__main__":    
    train_vqvae()
    reconstruct_images()
    test_ssim = test()
    print("Test SSIM achieved as {test_ssim}")