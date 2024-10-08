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


class VQVAE(nn.Module):
    def __init__(self):
        super(VQVAE, self).__init__()
        
        # Adjust the input channels in the encoder from 1 to 64
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
        )
        
        self.pre_quant_conv = nn.Conv2d(4, 2, kernel_size=1)
        self.embedding = nn.Embedding(num_embeddings=3, embedding_dim=2)
        self.post_quant_conv = nn.Conv2d(2, 4, kernel_size=1)
        
        # Commitment Loss Beta
        self.beta = 0.2
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4, 16, 4, stride=2, padding=1),  # Adjust input channels from 16 to 4
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),  # Final output with 1 channel (if your output is grayscale)
        )

        
    def forward(self, x):
        # Pass through the encoder
        encoded_output = self.encoder(x)  # Output shape: [B, C, H', W']
        
        # Pre-quantization convolution
        quant_input = self.pre_quant_conv(encoded_output)  # Still 4D: [B, C, H', W']
        
        # Reshape to flatten the spatial dimensions for quantization process
        B, C, H, W = quant_input.shape  # Extract batch size, channels, height, width
        quant_input_flattened = quant_input.view(B, C, -1).permute(0, 2, 1)  # Flatten to [B, L, C], where L = H * W
        
        # Compute pairwise distances for quantization
        dist = torch.cdist(quant_input_flattened, self.embedding.weight[None, :].repeat((B, 1, 1)))
        
        # Find index of nearest embedding
        min_encoding_indices = torch.argmin(dist, dim=-1)
        
        # Select the embedding weights
        quant_out = torch.index_select(self.embedding.weight, 0, min_encoding_indices.view(-1))
        quant_out = quant_out.view(B, -1, C)  # Reshape back to [B, L, C]
        
        # Compute losses
        commitment_loss = torch.mean((quant_out.detach() - quant_input_flattened) ** 2)
        codebook_loss = torch.mean((quant_out - quant_input_flattened.detach()) ** 2)
        quantize_losses = codebook_loss + self.beta * commitment_loss
        
        # Ensure straight-through gradient estimator
        quant_out = quant_input_flattened + (quant_out - quant_input_flattened).detach()
        
        # Reshape quantized output back to 4D for decoding
        quant_out_4d = quant_out.permute(0, 2, 1).view(B, C, H, W)  # Reshape back to [B, C, H', W']
        
        # Post-quantization convolution
        decoder_input = self.post_quant_conv(quant_out_4d)
        
        # Decoder part: input to decoder must be 4D
        output = self.decoder(decoder_input)  # Decoder output in 4D
        
        return output, quantize_losses


def train_vqvae():
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    train_loader, test_loader = load_data()
    model = VQVAE().to(device)
    
    num_epochs = 7
    optimizer = Adam(model.parameters(), lr=1E-3)
    criterion = torch.nn.MSELoss()
    
    train_losses = []

    for epoch_idx in range(num_epochs):
        epoch_loss = 0
        for im in tqdm(train_loader):
            im = im.float().unsqueeze(1).to(device)
            optimizer.zero_grad()
            out, quantize_loss = model(im)
            
            recon_loss = criterion(out, im)
            loss = recon_loss + quantize_loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_epoch_loss)
        print('Finished epoch {}'.format(epoch_idx + 1))

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

    model.eval()
    n = 50
    with torch.no_grad():
        idxs = torch.randint(0, len(test_loader.dataset), (n, ))
        
        for im in test_loader:
            ims = im.float().unsqueeze(1).to(device)
            break

        generated_im, _ = model(ims)

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
    # train_loader, _ = load_data()
    # show_example_images(train_loader)