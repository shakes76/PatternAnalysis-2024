import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim


# Set the device to GPU if available, otherwise fallback to CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define Encoder
class Encoder(nn.Module):
    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim):
        super(Encoder, self).__init__()
        kernel = 4  # Kernel size
        stride = 2  # Stride for downsampling

        # Convolution stack to process input
        self.conv_stack = nn.Sequential(
            nn.Conv2d(in_dim, h_dim // 2, kernel_size=kernel, stride=stride, padding=1),  # 252 -> 126
            nn.ReLU(),
            nn.Conv2d(h_dim // 2, h_dim, kernel_size=kernel, stride=stride, padding=1),  # 126 -> 63
            nn.ReLU(),
            nn.Conv2d(h_dim, h_dim, kernel_size=kernel-1, stride=stride-1, padding=1),  # 63 -> 63
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers)  # Stack of residual layers
        )

    def forward(self, x):
        return self.conv_stack(x)

# Define ResidualLayer and ResidualStack
class ResidualLayer(nn.Module):
    def __init__(self, in_dim, h_dim, res_h_dim):
        super(ResidualLayer, self).__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_dim, res_h_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(res_h_dim, h_dim, kernel_size=1, bias=False)
        )

    def forward(self, x):
        return x + self.res_block(x)

class ResidualStack(nn.Module):
    def __init__(self, in_dim, h_dim, res_h_dim, n_res_layers):
        super(ResidualStack, self).__init__()
        self.stack = nn.ModuleList([ResidualLayer(in_dim, h_dim, res_h_dim) for _ in range(n_res_layers)])

    def forward(self, x):
        for layer in self.stack:
            x = layer(x)
        return F.relu(x)

# Define Decoder
class Decoder(nn.Module):
    def __init__(self, in_dim, h_dim, res_h_dim, n_res_layers):
        super(Decoder, self).__init__()
        self.inverse_conv_stack = nn.Sequential(
            nn.ConvTranspose2d(in_dim, h_dim, kernel_size=4, stride=2, padding=1),  # 63 -> 126
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers),
            nn.ConvTranspose2d(h_dim, h_dim // 2, kernel_size=4, stride=2, padding=1),  # 126 -> 252
            nn.ReLU(),
            nn.ConvTranspose2d(h_dim // 2, 1, kernel_size=3, stride=1, padding=1)  # Ensure final output is 252x252
        )

    def forward(self, x):
        return self.inverse_conv_stack(x)

# Define VQVAE
class VQVAE(nn.Module):
    def __init__(self, encoder, decoder, embedding_dim, num_embeddings):
        super(VQVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)
    
    def forward(self, x):
        z = self.encoder(x)
        z_flattened = z.view(-1, self.embedding_dim)
        distances = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
                    torch.sum(self.embeddings.weight ** 2, dim=1) - \
                    2 * torch.matmul(z_flattened, self.embeddings.weight.t())
        z_quantized = self.embeddings(torch.argmin(distances, dim=1)).view(z.shape)
        x_recon = self.decoder(z_quantized)
        return x_recon, z_quantized

# # SSIM evaluation function
def evaluate_ssim(original_image, generated_image):
    return ssim(original_image, generated_image, data_range=generated_image.max() - generated_image.min())

#  Initialize list to store SSIM scores

#  Function to save images and SSIM score to a result file
def display_images_and_ssim(original_image, generated_image, ssim_score, idx, output_dir):
    """Display original and generated images along with SSIM score and save them."""
    # Create the figure with the original and generated image
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(original_image, cmap='gray')
    axs[0].set_title('Original Image')
    axs[0].axis('off')
    axs[1].imshow(generated_image, cmap='gray')
    axs[1].set_title(f'Generated Image (SSIM: {ssim_score:.4f})')
    axs[1].axis('off')
    
    # Save the figure
    output_filename = f"image_comparison_{idx}.png"
    plt.savefig(os.path.join(output_dir, output_filename))
    plt.close()  # Close the figure to avoid display overflow
 
