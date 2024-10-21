"""
This module have all the deep learning model architecture for the project.
""" 

# Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

# Checking the PyTorch Version
print("PyTorch Version: ", torch.__version__)

# Getting the device (in my case GPU with cuda 12.4)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class VQ(nn.Module):
    '''
    Get quantized value, perplexity and loss
    '''
    def __init__(self, num_embedding, embedding_dim, commitment_cost):
        super(VQ, self).__init__()
        self.K = num_embedding
        self.D = embedding_dim
        self.beta = commitment_cost
        
        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)

    def forward(self, x):
        # Permutate x, [B X D X H X W] -> [B X H X W X D]
        x = x.permute(0, 2, 3, 1).contiguous()
        
        # Get the shape of x
        x_shape = x.shape
        
        # Lower the dimension, [B X H X W X D] -> [BHW X D]
        flat_x = x.view(-1, self.D)
        
        # Calculate the Euclidian distance from x to embedding
        dist = torch.sum(flat_x ** 2, dim=1, keepdim=True) + torch.sum(self.embedding.weight**2, dim=1) - 2 * torch.matmul(flat_x, self.embedding.weight.t())
        
        # Get the encoding has min distance, find the closest one
        min_encode_index = torch.argmin(dist, dim=1).unsqueeze(1)  # [BHW X 1]
        min_encode = torch.zeros(min_encode_index.size(0), self.K).to(device)
        min_encode.scatter_(1, min_encode_index, 1)  # [BHW X K]
        
        # Quantize x and reshape(unflatten)
        quantized_x = torch.matmul(min_encode, self.embedding.weight).view(x.shape)  # [BHW X D]
        quantized_x = quantized_x.view(x_shape)  # [B X H X W X D]
        
        # Calculate loss
        commitment_loss = F.mse_loss(quantized_x.detach(), x)
        embedding_loss = F.mse_loss(quantized_x, x.detach())
        
        # VQ loss function
        loss = commitment_loss * self.beta + embedding_loss
        
        # Convert quantized_x - x into constant, save gradient
        quantized_x = x + (quantized_x - x).detach()
        
        # Calculate perplexity
        mean = torch.mean(min_encode, dim=0)
        perplexity = torch.exp(-torch.sum(mean * torch.log(mean + 1e-10)))
        
        # Reshape quantized_x [B X H X W X D] -> [B X D X H X D]
        quantized_x = quantized_x.permute(0, 3, 1, 2).contiguous()
        
        return loss, quantized_x, perplexity
    

class Encoder(nn.Module):
    '''
    Encode the images
    '''
    def __init__(self, in_channel, hidden_dim, num_res_layer, res_hidden_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, hidden_dim // 2, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.residual = Residual_Block(hidden_dim, hidden_dim, num_res_layer, res_hidden_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.residual(x)
        return x
    
class Residual_Layer(nn.Module):
    '''
    Build a residual layer
    '''
    def __init__(self, in_channel, hidden_dim, res_hidden_layer):
        super(Residual_Layer, self).__init__()
        self.relu = nn.ReLU(True)
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=res_hidden_layer, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=res_hidden_layer, out_channels=hidden_dim, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        x_ = self.relu(x)
        x_ = self.conv1(x_)
        x_ = self.relu(x_)
        x_ = self.conv2(x_)
        result = x + x_
        return result
    
class Residual_Block(nn.Module):
    '''
    Use the residual block to connect the input and output
    '''
    def __init__(self, in_channel, hidden_dim, num_res_layer, res_hidden_dim):
        super(Residual_Block, self).__init__()
        self.residual = nn.ModuleList([Residual_Layer(in_channel, hidden_dim, res_hidden_dim)]*num_res_layer)

    def forward(self, x):
        for layer in self.residual:
            x = layer(x)
        x = F.relu(x)
        return x
    

class Decoder(nn.Module):
    '''
    Define the Decoder which will decode the codebook
    '''
    def __init__(self, in_channel, hidden_dim, num_res_layer, res_hidden_dim):
        super(Decoder, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_channel, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.residual = Residual_Block(hidden_dim, hidden_dim, num_res_layer, res_hidden_dim)
        self.conv2 = nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, kernel_size=4, stride=2, padding=1)
        self.relu1 = nn.ReLU()
        self.conv3 = nn.ConvTranspose2d(hidden_dim // 2, 1, kernel_size=4, stride=2, padding=1)
        self.relu2 = nn.Tanh()

    def forward(self, x):
        x = self.conv1(x)
        x = self.residual(x)
        x = self.conv2(x)
        x = self.relu1(x)
        x = self.conv3(x)
        x = self.relu2(x)

        return x


class Model(nn.Module):
    '''
    Build the model using encoder to encode first, then get quantized value to form the code book.
    Then, take the quantized value to the decoder to reconstruct images
    '''
    def __init__(self, hidden_dim, res_hidden_dim, num_res_layer, num_embeddings, embedding_dim, commitment_cost):
        super(Model, self).__init__()

        self._encoder = Encoder(1, hidden_dim, num_res_layer, res_hidden_dim)
        self._vq_conv = nn.Conv2d(in_channels=hidden_dim, out_channels=num_embeddings, kernel_size=1, stride=1)
        self._vq = VQ(num_embeddings, embedding_dim, commitment_cost)
        self._decoder = Decoder(embedding_dim, hidden_dim, num_res_layer, res_hidden_dim)

    def forward(self, x):
        x = self._encoder(x)
        x = self._vq_conv(x)
        loss, quantized, perplexity = self._vq(x)
        x_re = self._decoder(quantized)

        return loss, x_re, perplexity, quantized

    # Function to directly decode quantized latent space
    def decode(self, quantized):
        return self._decoder(quantized)
    