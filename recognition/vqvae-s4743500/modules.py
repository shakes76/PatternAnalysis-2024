import torch
from torch import nn
from torch.nn import functional as F

class ResidualStack(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens):
        """
        Initialises the ResidualStack module which will be used for the Encoder and Decoder models

        Args:
            num_hiddens (int): The number of feature maps in the input
            num_residual_layers (int): The number of residual layers to stack
            num_residual_hiddens (int): The number of hidden channels in each layer
        """
        super().__init__()
        layers = []
        # Loops through the number of residual layers
        for i in range(num_residual_layers):
            layers.append(
                nn.Sequential(
                    nn.ReLU(), # ReLU activation function
                    nn.Conv2d(
                        in_channels=num_hiddens,
                        out_channels=num_residual_hiddens,
                        kernel_size=3, # 3x3 convolution
                        padding=1,
                    ),
                    nn.ReLU(), # ReLU after the first convolution
                    nn.Conv2d(
                        in_channels=num_residual_hiddens,
                        out_channels=num_hiddens, # Outputs back to the original feature map size
                        kernel_size=1, # 1x1 convolution to reduce the dimensions
                    ),
                )
            )
        # Stores the residual blocks as a ModuleList
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        """
        Forward pass through the residual stack

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, num_hiddens, height, width]
        
        Returns:
            torch.Tensor: The resulting tensor 
        """
        h = x
        for layer in self.layers:
            # Adds identity mapping (skip connection)
            h = h + layer(h)
        # Returns the tensor after applying the ReLU activation to the output
        return torch.relu(h)
    
class Encoder(nn.Module):
    """
    Initialises the Encoder module which downsamples an input using convolutional layers 
    and outputs a latent vector that will be fed into the VectorQuantizer

    Args:
         in_channels (int): The number of input channels (1 for grayscale images)
         num_hiddens (int): The number of hidden channels in the convolutional layers
         num_downsamplying_layers (int): The number of downsamplying layers to reduce the input size
         num_residual_layers (int): The number of residual layers to apply after downsampling
         num_residual_hiddens (int): The number of hidden channels inside each residual layer
    """
    def __init__(
            self,
            in_channels,
            num_hiddens, 
            num_downsampling_layers, 
            num_residual_layers, 
            num_residual_hiddens
    ):
        super().__init__()
        # Initialises a sequential container to hold the downsampling layers
        conv = nn.Sequential()
        for downsampling_layer in range(num_downsampling_layers):
            # Halves the number of hidden channels if it's the first layer
            if downsampling_layer == 0:
                out_channels = num_hiddens // 2
            # Restores the number of hidden channels if it's the second layer
            elif downsampling_layer == 1:
                (in_channels, out_channels) = (num_hiddens // 2, num_hiddens)
            else:
                # All other layers will keep the same number of hidden channels
                (in_channels, out_channels) = (num_hiddens, num_hiddens)

            # Adds the downsampling convolutional layer
            conv.add_module(
                f"down{downsampling_layer}",
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=4, # 4x4 convolution kernel 
                    stride=2, # Stride of 2 downsamples the spatial size by half 
                    padding=1 # Maintains shape consistency 
                ),
            )
            # Adds a ReLU activation function 
            conv.add_module(f"relu{downsampling_layer}", nn.ReLU())
        
        # Adds a final convolutional layer to maintain the same number of hidden channels
        conv.add_module(
            "final_conv",
            nn.Conv2d(
                in_channels=num_hiddens,
                out_channels=num_hiddens,
                kernel_size=3,
                padding=1
            ),
        )
        self.conv = conv

        # Initialises the residual stack after downsampling to further process the data
        self.residual_stack = ResidualStack(
            num_hiddens, num_residual_layers, num_residual_hiddens
        )

    def forward(self, x):
        """
        Forward pass of the encoder

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, in_channels, height, width]

        Returns:
            torch.Tensor: The output after applying downsampling and residual layers
        """
        # Applies downsampling and passes through the residual stacks 
        h = self.conv(x)
        return self.residual_stack(h)
    
class Decoder(nn.Module):
    """
    Initialises the Decoder module which upsamples the latent representation from the encoder
    and outputs a reconstruction of the input image

    Args:
         embedding_dim (int): Dimensionality of the latent embeddings from the encoder
         num_hiddens (int): The number of hidden channels for the convolutional layers
         num_upsampling_layers (int): The number of upsampling layers to upsample the latent code
         num_residual_layers (int): The number of residual layers to apply after the initial convolution
         num_residual_hiddens (int): The number of hidden channels inside each residual layer
    """
    def __init__(
            self,
            embedding_dim, 
            num_hiddens,
            num_upsampling_layers,
            num_residual_layers,
            num_residual_hiddens
    ):
        super().__init__()

        # First convolution to convert the embedding dimension to the number of hidden channels
        self.conv = nn.Conv2d(
            in_channels=embedding_dim,
            out_channels=num_hiddens,
            kernel_size=3,
            padding=1, # Ensures the spatial dimension is maintained
        )

        # Applies the Residual stack to refine the feature maps
        self.residual_stack = ResidualStack(
            num_hiddens, num_residual_layers, num_residual_hiddens
        )
        upconv = nn.Sequential()

        # Loops through the number of upsampling layers 
        for upsampling_layer in range(num_upsampling_layers):
            # Keeps the number of hidden channels for earlier upsampling layers
            if upsampling_layer < num_upsampling_layers - 2:
                (in_channels, out_channels) = (num_hiddens, num_hiddens)
            # If its the second-to-last layer, reduce the number of hidden channels by half
            elif upsampling_layer == num_upsampling_layers - 2:
                (in_channels, out_channels) = (num_hiddens, num_hiddens // 2)
            # If its the last layer, output a single channel for grayscale images
            else:
                (in_channels, out_channels) = (num_hiddens // 2, 1) # 1 for grayscale images 

            # Adds the transposed convolutional layer to perform the upsampling
            upconv.add_module(
                f"up{upsampling_layer}",
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=4, # $x4 kernel fpr upsampling
                    stride=2, # Stride 2 upsamples by double
                    padding=1
                ),
            )

            # Makes sure that the ReLU activation is added for all layers but the last one
            if upsampling_layer < num_upsampling_layers - 1:
                upconv.add_module(f"relu{upsampling_layer}", nn.ReLU())

        self.upconv = upconv

    def forward(self, x):
        """
        Forward pass of the decoder 

        Args:
            x (torch.Tensor): The input tensor of shape [batch_size, embedding_dim, height, width]

        Returns:
            torch.Tensor: The reconstructed image by the model
        """
        # Maps the embeddings to the hidden channels
        h = self.conv(x)
        # Refines the feature maps by applying the residual stack 
        h = self.residual_stack(h)
        # Reconstructs and returns the image 
        x_recon = self.upconv(h)
        return x_recon
    
class VectorQuantizer(nn.Module):
    """
    Initialises the Vector Quantizer module for the VQ-VAE model where it maps the continuous latent space
    into discrete embedding space using the nearest neighbour technique

    Args:
        embedding_dim (int): The dimensionality for each embedding vector 
        num_embeddings (int): The number of embedding vectors in the codebook
        beta (float): The weight for the commitment loss which is used to balance the embedding losses
        decay (float): The decay for updating the embeddings during training
        epsilon (float): The epsilon value which aims to prevent numerical instability when updating the embeddings
    """
    def __init__(self, embedding_dim, num_embeddings, beta = 0.25, decay=0.99, epsilon=1e-5):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.beta = beta  
        self.decay = decay
        self.epsilon = epsilon

        # Initialises embedding vectors using nn.Embedding
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        # Uniformly initialises the embedding weights between -1/num_embeddings and 1/num_embeddings
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, z):
        """
        Forward pass of the Vector Quantizer

        Args: 
            z (torch.Tensor): Input tensor of shape [batch_size, channels, height, width]

        Returns: 
            total_loss (torch.Tensor): The sum between the quantization and embedding commitment loss
            quantized (torch.Tensor): Quantized tensor of the same shape as the input
            encoding_indices (torch.Tensor): Indices of the nearest embeddings for each input latent vector
        """
        # Reshapes the tensor z from BCHW to BHWC
        z = z.permute(0, 2, 3, 1).contiguous()

        # Flattens the tensor from [B,H,W,C] to [B*H*W,C] to easily calculate the distance
        z_flattened = z.view(-1, self.embedding_dim)

        # Computes distance between z and the embedding vectors (uses squared Euclidean distance as the distance metric)
        distance = torch.sum(z_flattened**2, dim=1, keepdim=True) + \
                   torch.sum(self.embedding.weight**2, dim=1) - \
                   2 * torch.matmul(z_flattened, self.embedding.weight.t())

        # Finds the closest embedding index for each vector using argmin()
        encoding_indices = torch.argmin(distance, dim=1)
        quantized = self.embedding(encoding_indices).view(z.shape)

        # Computes the embedding and quantization loss, as well as the total loss (sum of both)
        embedding_loss = F.mse_loss(quantized.detach(), z)
        quantized_loss = F.mse_loss(quantized, z.detach())
        total_loss = quantized_loss + self.beta * embedding_loss

        # Apploies the straight-through estimator for backpropagation
        quantized = z + (quantized - z).detach()

        # Reshapes quantized back to [B,C,H,W]
        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        return total_loss, quantized, encoding_indices

class VQVAE(nn.Module):
    """
    Initialises the VQ-VAE model. It consists of the encoder that downsamples the input image, vector quantizer that 
    maps the latent space to discrete embeddings, and decoder that reconstructs the input image 

    Args:
        in_channels (int): The number of input channels
        num_hiddens (int): The number of hidden units for the encoder and decoder 
        num_downsampling_layers (int): The number of downsampling layers in the encoder 
        num_residual_layers (int): The number of residual layers in the residual stack
        num_residual_hiddens (int): The number of hidden units in the residual layers
        embedding_dim (int): The dimensionality of the embeddings in the vector quantizer
        num_embeddings (int): The number of embeddings
        beta (float): The weight of the commitment loss
        decay (float): The decay factor for updating embeddings
        epsilon (float): The small value for stability 
    """
    def __init__(
            self, 
            in_channels=1, 
            num_hiddens=128,
            num_downsampling_layers=3, 
            num_residual_layers=2,
            num_residual_hiddens=32,
            embedding_dim=128,
            num_embeddings=512,
            beta=0.25, 
            decay=0.99,
            epsilon=1e-5
    ):
        super().__init__()

        # Passes in the Encoder to compress the input image using downsampling layers and residuals
        self.encoder = Encoder(
            in_channels,
            num_hiddens,
            num_downsampling_layers,
            num_residual_layers,
            num_residual_hiddens
        )

        # Reduces the number of channels before quantization
        self.pre_vq_conv = nn.Conv2d(
            in_channels=num_hiddens, out_channels=embedding_dim, kernel_size=1
        )
        # Passes in the Vector Quantizer 
        self.vq = VectorQuantizer(
            embedding_dim, num_embeddings, beta, decay, epsilon 
        ) 

        # Passes in the Decoder to reconstruct the images from the quantized latent representation
        self.decoder = Decoder(
            embedding_dim,
            num_hiddens,
            num_upsampling_layers=num_downsampling_layers, # The number of upsampling layers should match the number of downsampling layers
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens
        )

    def forward(self, x):
        """
        Forward pass for the VQ-VAE model

        Args:
            x (torch.Tensor): Input tensor representing the image with shape [batch_size, channels, height, width]

        Returns: 
            A dictionary that contains the commitment_loss, x_recon, and codebook_embeddings
        """
        # Uses the Encoder to encode the input image
        z = self.pre_vq_conv(self.encoder(x))

        # Uses the Vector Quantizer to map the latent representation to discrete embeddings
        quantization_loss, z_quantized, encoding_indices = self.vq(z)

        # Uses the Decoder to reconstruct the image 
        x_recon = self.decoder(z_quantized)
        return {"commitment_loss": quantization_loss, "x_recon": x_recon, "codebook_embeddings": z_quantized}

