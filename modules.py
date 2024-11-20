import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Decoder(nn.Module):
    """
    This is the p_phi (x|z) network. Given a latent sample z p_phi 
    maps back to the original space z -> x.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_res_layers: int, res_hidden_dim: int):
        super(Decoder, self).__init__()
        kernel_size = 4
        stride = 2

        self.inverse_conv_stack = nn.Sequential(
            nn.ConvTranspose2d(input_dim, hidden_dim, kernel_size=kernel_size-1, stride=stride-1, padding=1),
            ResidualStack(hidden_dim, hidden_dim, res_hidden_dim, num_res_layers),
            nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, kernel_size=kernel_size, stride=stride, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim // 2, 3, kernel_size=kernel_size, stride=stride, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inverse_conv_stack(x)

class Encoder(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, num_res_layers: int, res_hidden_dim: int):
        super(Encoder, self).__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=4, stride=2, padding=1),  
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),       
            nn.ReLU(),
        )
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_stack(x)
    
class VectorQuantizer(nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, beta: float):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)

    def forward(self, z):
        """
        Maps encoder output z to a discrete one-hot vector.
        """
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.embedding_dim)

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.num_embeddings).to(device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # calculate loss
        loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
            torch.mean((z_q - z.detach()) ** 2)

        z_q = z + (z_q - z).detach()
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))
        
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return loss, z_q, perplexity, min_encodings, min_encoding_indices

class ResidualLayer(nn.Module):
    """
    One residual layer.
    """

    def __init__(self, in_dim, h_dim, res_h_dim):
        super(ResidualLayer, self).__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_dim, res_h_dim, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(res_h_dim, h_dim, kernel_size=1,
                      stride=1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.res_block(x)


class ResidualStack(nn.Module):
    """ A stack of residual layers. """

    def __init__(self, in_dim: int, hidden_dim: int, res_hidden_dim: int, num_res_layers: int):
        super(ResidualStack, self).__init__()
        self.stack = nn.ModuleList([ResidualLayer(in_dim, hidden_dim, res_hidden_dim) for _ in range(num_res_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.stack:
            x = layer(x)
        return F.relu(x)
    
class VQVAE(nn.Module):
    def __init__(self, hidden_dim: int, res_hidden_dim: int, num_res_layers: int,
                 num_embeddings: int, embedding_dim: int, beta: float, save_img_embedding_map: bool = False, input_channels: int = 1): 
        super(VQVAE, self).__init__()
        
        self.encoder = Encoder(input_channels, hidden_dim, num_res_layers, res_hidden_dim)  
        self.pre_quantization_conv = nn.Conv2d(hidden_dim, embedding_dim, kernel_size=1, stride=1)
        
        self.vector_quantization = VectorQuantizer(num_embeddings, embedding_dim, beta)
        
        self.decoder = Decoder(embedding_dim, hidden_dim, num_res_layers, res_hidden_dim)

        if save_img_embedding_map:
            self.img_to_embedding_map = {i: [] for i in range(num_embeddings)}
        else:
            self.img_to_embedding_map = None

    def forward(self, x: torch.Tensor, verbose: bool = False):
        z_e = self.encoder(x)
        z_e = self.pre_quantization_conv(z_e)
        embedding_loss, z_q, perplexity, _, _ = self.vector_quantization(z_e)
        x_hat = self.decoder(z_q)

        if verbose:
            print('Original data shape:', x.shape)
            print('Encoded data shape:', z_e.shape)
            print('Reconstructed data shape:', x_hat.shape)

        return embedding_loss, x_hat, perplexity
