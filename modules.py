
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Decoder(nn.Module):
    """
    This is the p_phi (x|z) network. Given a latent sample z p_phi 
    maps back to the original space z -> x.
    """

    def __init__(self, input_dim, hidden_dim, num_res_layers, res_hidden_dim):
        super(Decoder, self).__init__()
        kernel_size = 4
        stride_val = 2

        self.deconv_stack = nn.Sequential(
            nn.ConvTranspose2d(
                input_dim, hidden_dim, kernel_size=kernel_size-1, stride=stride_val-1, padding=1),
            ResidualStack(hidden_dim, hidden_dim, res_hidden_dim, num_res_layers),
            nn.ConvTranspose2d(hidden_dim, hidden_dim // 2,
                               kernel_size=kernel_size, stride=stride_val, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim//2, 3, kernel_size=kernel_size,
                               stride=stride_val, padding=1)
        )

    def forward(self, x):
        return self.deconv_stack(x)

"""
class Encoder(nn.Module):
    
    This is the q_theta (z|x) network. Given a data sample x q_theta 
    maps to the latent space x -> z.

    For a VQ VAE, q_theta outputs parameters of a categorical distribution.

    Inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack

    

    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim):
        super(Encoder, self).__init__()
        kernel = 4
        stride = 2
        self.conv_stack = nn.Sequential(
            nn.Conv2d(in_dim, h_dim // 2, kernel_size=kernel,
                      stride=stride, padding=1),
            nn.ReLU(),
            nn.Conv2d(h_dim // 2, h_dim, kernel_size=kernel,
                      stride=stride, padding=1),
            nn.ReLU(),
            nn.Conv2d(h_dim, h_dim, kernel_size=kernel-1,
                      stride=stride-1, padding=1),
            ResidualStack(
                h_dim, h_dim, res_h_dim, n_res_layers)

        )

    def forward(self, x):
        return self.conv_stack(x)
"""

class Encoder(nn.Module):
    def __init__(self, num_channels, hidden_dim, num_res_layers, res_hidden_dim):
        super(Encoder, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(num_channels, hidden_dim, kernel_size=4, stride=2, padding=1),  
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),       
            nn.ReLU(),
        )

    def forward(self, x):
        return self.conv_layers(x)
    
class Quantizer(nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE.
    """

    def __init__(self, num_embeddings, embedding_dim, beta_val):
        super(Quantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta_val = beta_val

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)

    def forward(self, latent_z):
        """
        Maps encoder output z to a discrete one-hot vector.
        """

        latent_z = latent_z.permute(0, 2, 3, 1).contiguous()
        flat_z = latent_z.view(-1, self.embedding_dim)
        
        dist = torch.sum(flat_z ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(flat_z, self.embedding.weight.t())

        closest_indices = torch.argmin(dist, dim=1).unsqueeze(1)
        closest_encodings = torch.zeros(
            closest_indices.shape[0], self.num_embeddings).to(device)
        closest_encodings.scatter_(1, closest_indices, 1)

        quantized_z = torch.matmul(closest_encodings, self.embedding.weight).view(latent_z.shape)

        quant_loss = torch.mean((quantized_z.detach()-latent_z)**2) + self.beta_val * \
            torch.mean((quantized_z - latent_z.detach()) ** 2)

        quantized_z = latent_z + (quantized_z - latent_z).detach()

        mean_encodings = torch.mean(closest_encodings, dim=0)
        perplexity_val = torch.exp(-torch.sum(mean_encodings * torch.log(mean_encodings + 1e-10)))

        quantized_z = quantized_z.permute(0, 3, 1, 2).contiguous()

        return quant_loss, quantized_z, perplexity_val, closest_encodings, closest_indices

class ResidualLayer(nn.Module):
    """
    One residual layer.
    """

    def __init__(self, input_dim, hidden_dim, res_hidden_dim):
        super(ResidualLayer, self).__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(input_dim, res_hidden_dim, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(res_hidden_dim, hidden_dim, kernel_size=1,
                      stride=1, bias=False)
        )

    def forward(self, x):
        x = x + self.res_block(x)
        return x


class ResidualStack(nn.Module):
    """
    A stack of residual layers.
    """

    def __init__(self, input_dim, hidden_dim, res_hidden_dim, num_res_layers):
        super(ResidualStack, self).__init__()
        self.stack = nn.ModuleList(
            [ResidualLayer(input_dim, hidden_dim, res_hidden_dim)]*num_res_layers)

    def forward(self, x):
        for layer in self.stack:
            x = layer(x)
        x = F.relu(x)
        return x
    
class VQVAE(nn.Module):
    def __init__(self, hidden_dim, res_hidden_dim, num_res_layers,
                 num_embeddings, embedding_dim, beta_val, save_embeddings_map=False, input_channels=1): 
        super(VQVAE, self).__init__()
        
        self.encoder = Encoder(input_channels, hidden_dim, num_res_layers, res_hidden_dim)  
        self.pre_quant_conv = nn.Conv2d(hidden_dim, embedding_dim, kernel_size=1, stride=1)
        
        self.quantizer = Quantizer(num_embeddings, embedding_dim, beta_val)
        
        self.decoder = Decoder(embedding_dim, hidden_dim, num_res_layers, res_hidden_dim)

        if save_embeddings_map:
            self.img_to_embedding_map = {i: [] for i in range(num_embeddings)}
        else:
            self.img_to_embedding_map = None

    def forward(self, x, verbose=False):

        latent_z = self.encoder(x)
        latent_z = self.pre_quant_conv(latent_z)
        quant_loss, quantized_z, perplexity_val, _, _ = self.quantizer(latent_z)
        reconstructed_x = self.decoder(quantized_z)

        if verbose:
            print('Original data shape:', x.shape)
            print('Encoded data shape:', latent_z.shape)
            print('Reconstructed data shape:', reconstructed_x.shape)
            assert False

        return quant_loss, reconstructed_x, perplexity_val
    