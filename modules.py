import torch
import torch.nn as nn
import torch.nn.functional as func
import utils

"""
    Please note that the letters (B, C, H, W) are used to represent the dimensions of the input tensor in the comments,
    the letters correspond to:
        B - Batch Size
        C - Number of Channels
        H - Height
        W - Width
"""

# Initialise the device
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")


class ResidualLayer(nn.Module):
    """
        A single residual layer that will be used to allow the VQVAE to better avoid the vanishing gradient problem with
        the use of skip connections. Multiple of these layers will be used in each pass through of the model.
    """

    def __init__(self, in_dim, h_dim, res_h_dim):
        """
            Initialise the Residual Layer.

            Input:
                in_dim : the input dimension
                h_dim : the hidden layer dimension
                res_h_dim : the hidden dimension of the residual block
        """
        super(ResidualLayer, self).__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_dim, res_h_dim, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(res_h_dim, h_dim, kernel_size=1,
                      stride=1, bias=False)
        )

    def forward(self, x):
        """
            Passes the input through the residual layer.

            Input:
                x: a tensor input that will be passed through the residual layer
            Output:
                x: the sum of the original tensor and itself after being passed through the residual layer
        """
        x = x + self.res_block(x)
        return x
    
  
class ResidualStack(nn.Module):
    """
        A stack of residual layers
    """

    def __init__(self, in_dim, h_dim, res_h_dim, n_res_layers):
        """
            Initialise the Residual Stack.

            Input:
                in_dim : the input dimension
                h_dim : the hidden layer dimension
                res_h_dim : the hidden dimension of the residual block
                n_res_layers : the number of residual layers in the stack
        """
        super(ResidualStack, self).__init__()
        self.n_res_layers = n_res_layers
        self.stack = nn.ModuleList(
            [ResidualLayer(in_dim, h_dim, res_h_dim)]*n_res_layers)

    def forward(self, x):
        """
            Passes the input through the residual stack.

            Input:
                x: a tensor input that will be passed through the residual stack
            Output:
                x: the tensor after passing through each residual layer and then through a ReLU activation function
        """
        for layer in self.stack:
            x = layer(x)
        x = func.relu(x)
        return x
    

class VectorQuantizer(nn.Module):
    """
        The Vector Quantizer that will take the encoded tensor from the VQVAE encoder and select discrete embeddings
        from a codebook based on distances and the quantisation of the input.
    """

    def __init__(self, n_e, dim_e, beta=0.25):
        """
            Initialise the Vector Quantizer.

            Input:
                n_e : the number of embeddings
                dim_e : the dimension of the embeddings
                beta : the commitment cost to be used in the loss function, default 0.25 as per paper
        """
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.dim_e = dim_e
        self.beta = beta

        self.codebook = nn.Embedding(self.n_e, self.dim_e)  # Create the codebook
        self.codebook.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)  # Initialise the codebook weightings

    def forward(self, z):
        """
            Passes the input tensor through the Vector Quantizer, converting it to discrete embeddings in the latent
            space that are determined based on minimum distance to the codebook.

            Input:
                z : the tensor to be passed through the Vector Quantizer, starts at shape (B, C, H, W)
            Output:
                commitment_loss: the commitment loss for the embedding, modified original formula for more stability:
                    ‖ z(x)− no_grad[quantised_z] ‖^2 -  β‖ quantised_z(x)− no_grad[z] ‖^2
                z_q : the quantised input tensor
        """
        z = z.permute(0, 2, 3, 1).contiguous()  # Shaped to (B, H, W, C)
        z_flattened = z.view(-1, self.dim_e)  # Flattens z to (BHW, C), note C should be 1 always for this data

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.codebook.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.codebook.weight.t())  # Calculate distances

        # Find the encodings closest to the codebook
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.n_e).to(device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # Using the minimum encoding, quantise the input
        quantised_z = torch.matmul(min_encodings, self.codebook.weight).view(z.shape)

        # Compute the commitment loss based on the formula
        commitment_loss = (torch.mean((quantised_z.detach()-z)**2) +
                           self.beta * torch.mean((quantised_z - z.detach()) ** 2))

        # Preserve the gradient before returning
        quantised_z = z + (quantised_z - z).detach()

        # reshape back to match original input shape
        quantised_z = quantised_z.permute(0, 3, 1, 2).contiguous()  # Reshape back to (B, C, H, W)

        return commitment_loss, quantised_z


class VQVAE(nn.Module):
    """
        The VQVAE model that will be trained to encode an input, feed the encoded tensor through a Vector Quantizer to
        select discrete embeddings, and then decode the embeddings back to the original input. The final model will
        allow the generation of new images based on the learned embeddings and the image(s) input to the model.
    """
    def __init__(self, h_dim, res_h_dim, n_res_layers, n_embeddings, embedding_dim):
        """
            Initialise the VQVAE model.

            Input:
                h_dim : the hidden layer dimension
                res_h_dim : the hidden dimension of the residual block
                n_res_layers : the number of residual layers in the stack
                n_embeddings : the number of embeddings
                embedding_dim : the dimension of the embeddings
        """
        super(VQVAE, self).__init__()
        kernel = 4
        stride = 2

        # Encoder aims to take (B, 1, 256, 128) to (B, 128, 64, 32)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, h_dim // 2, kernel_size=kernel,
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

        # Pre-Quantisation Convolution Layer to match embedding dimensions
        self.pre_quant_conv = nn.Conv2d(h_dim, embedding_dim, kernel_size=1, stride=1)
        # Vector quantisation layer
        self.codebook = VectorQuantizer(n_embeddings, embedding_dim)
        # Post-Quantisation Convolution Layer to match input dimensions and allow for decoding
        self.post_quant_conv = nn.ConvTranspose2d(embedding_dim, h_dim, kernel_size=1, stride=1)

        # Decoder aims to take (B, 128, 64, 32) to (B, 1, 256, 128) which is the original tensor shape
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, h_dim, kernel_size=1, stride=1),  # Adjust to match input[32, 128, 64, 32]
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers),
            nn.ConvTranspose2d(h_dim, h_dim // 2, kernel_size=kernel, stride=stride, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(h_dim // 2, 1, kernel_size=kernel, stride=stride, padding=1),
        )
        
        self.apply(utils.weights_init)  # Initialise the weights

    def forward(self, x):
        """
            Passes the input tensor through the VQVAE model.

            Input:
                x : the input tensor to be passed through the VQVAE model
            Output:
                decoded_output : the output tensor after being passed through the decoder
                commitment_loss : the loss from the Vector Quantizer
                encoded_output : the output tensor after being passed through the encoder
                quantised_output : the output tensor after being passed through the Vector Quantizer
        """
        encoded_output = self.encoder(x)
        encoded_output = self.pre_quant_conv(encoded_output)
        commitment_loss, quantised_output = self.codebook(encoded_output)
        quantised_output = self.post_quant_conv(quantised_output)
        decoded_output = self.decoder(quantised_output)
        
        return decoded_output, commitment_loss, encoded_output, quantised_output
