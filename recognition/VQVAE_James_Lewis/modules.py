import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self, input_dim, output_dim, n_res_block, n_res_channel):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_res_block = n_res_block
        self.n_res_channel = n_res_channel
        stride = 2

        self.conv_stack = nn.Sequential(
            nn.Conv2d(1, output_dim // 16, kernel_size=4, stride=stride, padding=1),  # 1 → output_dim/16
            nn.ReLU(),
            nn.Conv2d(output_dim // 16, output_dim // 8, kernel_size=4, stride=stride, padding=1),
            # output_dim/16 → output_dim/8
            nn.ReLU(),
            nn.Conv2d(output_dim // 8, output_dim // 4, kernel_size=4, stride=stride, padding=1),
            # output_dim/8 → output_dim/4
            nn.ReLU(),
            nn.Conv2d(output_dim // 4, output_dim // 2, kernel_size=3, stride=stride - 1, padding=1),
            # output_dim/4 → output_dim/2
            nn.ReLU(),
            nn.Conv2d(output_dim // 2, output_dim, kernel_size=3, stride=stride - 1, padding=1),
            # output_dim/2 → output_dim
            ResidualStack(output_dim, output_dim, n_res_block, n_res_channel)  # Keep the same for the residual stack
        )

    def forward(self, x):
        x = self.conv_stack(x)
        return x

class Decoder(nn.Module):
    def __init__(self, dim, output_dim, n_res_block, n_res_channel):
        super(Decoder, self).__init__()
        self.dim = dim
        self.output_dim = output_dim
        self.n_res_block = n_res_block
        self.n_res_channel = n_res_channel
        stride = 1

        self.inv_conv_stack = nn.Sequential(
            nn.ConvTranspose2d(dim, dim//2, 3, stride, 1),
            ResidualStack(dim//2, dim//2, n_res_block, n_res_channel),
            nn.ConvTranspose2d(dim//2, dim//4, 3, stride, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(dim//4, dim//8, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(dim//8, dim//16, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(dim // 16, 1, 4, 2, 1)  # Output layer for grayscale images
        )

    def forward(self, x):
        x = self.inv_conv_stack(x)
        return x



class VectorQuantizer(nn.Module):
    def __init__(self, embedding_dim, num_embeddings, commitment_cost=0.25, epsilon=1e-5):
        super(VectorQuantizer, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.epsilon = epsilon

        # Define the embedding layer
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, x):
        # Reshape input to [batch_size * height * width, embedding_dim]
        x = x.permute(0, 2, 3, 1).contiguous()
        flat_x = x.view(-1, self.embedding_dim)

        # Compute distances between input features and embeddings
        distance = torch.sum(flat_x ** 2, dim=1, keepdim=True) + \
                   torch.sum(self.embeddings.weight ** 2, dim=1) - \
                   2 * torch.matmul(flat_x, self.embeddings.weight.t())

        # Find the index of the closest embedding
        encoding_indices = torch.argmin(distance, dim=1).unsqueeze(1)

        # Quantize the input by replacing it with the closest embedding
        quantized_x = self.embeddings(encoding_indices).view(x.shape)

        # Calculate the dictionary loss (Equation 3, term 2 in the VQ-VAE paper)
        dictionary_loss = F.mse_loss(quantized_x.detach(), x)

        # Calculate the commitment loss (Equation 3, term 3 in the VQ-VAE paper)
        commitment_loss = F.mse_loss(quantized_x, x.detach())

        # Combine quantized input with straight-through gradient estimator
        quantized_x = x + (quantized_x - x).detach()

        # Return quantized outputs, dictionary loss, and commitment loss
        total_loss = dictionary_loss + self.commitment_cost * commitment_loss

        quantized_x = quantized_x.permute(0, 3, 1, 2).contiguous()
        return quantized_x, total_loss, encoding_indices.view(x.shape[0], -1), self.embeddings.weight

class VQVAE(nn.Module):
    def __init__(self, input_dim, out_dim, n_res_block, n_res_channel, stride, n_embed, commitment_cost, embedding_dims):
        super(VQVAE, self).__init__()

        self.encoder = Encoder(input_dim, out_dim, n_res_block, n_res_channel)
        self.pre_quantization_conv = nn.Conv2d(
            out_dim, embedding_dims, kernel_size=1, stride=1)
        self.vector_quantizer = VectorQuantizer(embedding_dims, n_embed, commitment_cost)
        self.decoder = Decoder(out_dim, input_dim, n_res_block, n_res_channel)

    def forward(self, x):

        z = self.encoder(x)
        z = self.pre_quantization_conv(z)
        z_q,  loss, min_encoding_indices, embeddings = self.vector_quantizer(z)
        x_recon = self.decoder(z_q)

        return x_recon, loss, embeddings



class ResidualStack(nn.Module):
    def __init__(self, in_dim, out_dim, n_res_block, n_res_channel):
        super(ResidualStack, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_res_block = n_res_block
        self.n_res_channel = n_res_channel

        stack = []
        for i in range(n_res_block):
            stack.append(nn.Sequential(nn.ReLU(),
                                       nn.Conv2d(in_dim, n_res_channel, 3, 1, 1),
                                       nn.ReLU(),
                                       nn.Conv2d(n_res_channel, out_dim, 1, 1, 0)))
        self.stack = nn.ModuleList(stack)

    def forward(self, x):
        for res_block in self.stack:
            identity = x  # Save the input for the skip connection
            x = res_block(x) + identity  # Add the input to the output
        return x
