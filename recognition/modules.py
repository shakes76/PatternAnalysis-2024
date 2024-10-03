import torch
import torch.nn as nn
import torch.nn.functional as F

"""
REFERENCES:
Kang, J. (2024). Pytorch-VAE-tutorial. Retrieved 31st August 2024 from 
    https://github.com/Jackson-Kang/Pytorch-VAE-tutorial
Yadav, S. (2019, September 1). Understanding Vector Quantized Variational Autoencoders (VQ-VAE) [Blog]. Medium.
    https://shashank7-iitd.medium.com/understanding-vector-quantized-variational-autoencoders-vq-vae-323d710a888a 
"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Modules are switching to CPU.")
else:
    print(f"Modules are using {device}.")

"""Encodes the images into latent space,
    with lower granularity, by parameterising the a categorical
    posterior distribution over the latent space.
    In a normal VAE, the posterior and prior assume a 
    continuous normal distribution instead of categorial (Yadav, 2019)."""
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,
                 kernel_sizes=(4, 4, 3, 1), stride=2):
        super().__init__()

        k1, k2, k3, k4 = kernel_sizes

        self.strided_conv1 = nn.Conv2d(input_dim, hidden_dim, kernel_size=k1, stride=stride, padding=1)
        self.strided_conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=k2, stride=stride, padding=1)
        
        self.residual_conv1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=k3, padding=1)
        self.residual_conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=k4, padding=0)

        self.proj = nn.Conv2d(hidden_dim, out_channels=output_dim, kernel_size=1)

        self.training = True

        self.ReLU = nn.ReLU()

    def forward(self, input):
        input = self.strided_conv1(input)
        input = self.strided_conv2(input)

        #F.relu(act)
        input = self.ReLU(input)
        out = self.residual_conv1(input)
        out += input

        #F.relu(res_out)
        input = self.ReLU(out)
        out = self.residual_conv2(input)
        out += input

        out = self.proj(out)

        return out

"""TBD"""
class Quantise(nn.Module):
    def __init__(self, n_embeddings, embed_dim, commitment_cost=0.25, decay=0.999, e=1e-5):
        super().__init__()
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.e = e

        init_bound = 1 / n_embeddings
        embedding = torch.Tensor(n_embeddings, embed_dim)
        # print(f"Embedding shape: {embedding.shape}")
        embedding.uniform_(-init_bound, init_bound)
        # Buffers are parameters that should not be touched by the optimiser.
        self.register_buffer("embedding", embedding)
        self.register_buffer("ema_count", torch.zeros(n_embeddings))
        self.register_buffer("ema_weight", self.embedding.clone())

    def encode(self, input):
        M, D = self.embedding.size()
        input_flat = input.detach().reshape(-1, D)

        distances = (-torch.cdist(input_flat, self.embedding, p=2)) ** 2

        indices = torch.argmin(distances.float(), dim=-1)
        quantised = F.embedding(indices, self.embedding)
        quantised = quantised.view_as(input)
        return quantised, indices.view(input.size(0), input.size(1))

    def retrieve(self, random_i):
        quantised = F.embedding(random_i, self.embedding)
        quantised = quantised.transpose(1, 3)
        return quantised

    def forward(self, input):
        # print(f"embedding:{self.embedding.shape}")
        M, D = self.embedding.size()
        # print("M: {M}, D: {D}")
        input_flat = input.detach().reshape(-1, D)

        distances = (-torch.cdist(input_flat, self.embedding, p=2)) ** 2

        indices = torch.argmin(distances.float(), dim=-1)
        # categorical encoding required? (load_data_2D(categorical=True))
        encodings = F.one_hot(indices, M).float()
        # print(f"Encodings shape:{encodings.shape}")
        quantised = F.embedding(indices, self.embedding)
        quantised = quantised.view_as(input)

        if self.training:
            decayed_count = self.decay * self.ema_count + (1 - self.decay)
            # print(f"ema count: {decayed_count.shape} sum of encodings: {torch.sum(encodings, dim=0).shape}")
            self.ema_count = decayed_count * torch.sum(encodings, dim=0)
            n = torch.sum(self.ema_count)
            self.ema_count = (self.ema_count + self.e) / (n + M * self.e) * n

            dw = torch.matmul(encodings.t(), input_flat)
            self.ema_weight = self.decay * self.ema_weight + (1 - self.decay) * dw
            self.embedding = self.ema_weight / self.ema_count.unsqueeze(-1)

        # Loss equations are different in the blog
        reconstr_loss = F.mse_loss(input.detach(), quantised)
        codebook_loss = F.mse_loss(input, quantised.detach())
        commitment_loss = self.commitment_cost * reconstr_loss

        quantised = input + (quantised - input).detach()

        avg = torch.mean(encodings, dim=0)
        # perplexity is defined as the level of entropy
        perplexity = torch.exp(-torch.sum(avg * torch.log(avg + 1e-10)))
        return quantised, commitment_loss, codebook_loss, perplexity


"""Decodes the latent space into higher dimensionality,
   using indexed values from the dictionary of embeddings. (Yadav, 2019)"""
class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, kernel_sizes=(1, 3, 2, 2), stride=2):
        super().__init__()
        k1, k2, k3, k4 = kernel_sizes
        self.inner_proj = nn.Conv2d(input_dim, hidden_dim, kernel_size=1)

        self.residual_conv1= nn.Conv2d(hidden_dim, hidden_dim, kernel_size=k1, padding=0)
        self.residual_conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=k2, padding=1)
        
        self.strided_conv1 = nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=k3, stride=stride, padding=0)
        self.strided_conv2 = nn.ConvTranspose2d(hidden_dim, output_dim, kernel_size=k4, stride=stride, padding=0)
        self.ReLU = nn.ReLU()

    def forward(self, input):
        # print("Inner projection in Decoder...")
        # print(f"Input before decoder projection: {input.shape}")
        input = self.inner_proj(input)
        # print(f"Input after decoder projection: {input.shape}")

        # print("First residual layer in Decoder...")
        out = self.residual_conv1(input)
        # print(f"out:{out.shape}, in:{input.shape}")
        out += input
        #F.relu(res_out)
        input = self.ReLU(out)

        # print("Second residual layer in Decoder...")
        out = self.residual_conv2(input)
        out += input
        #F.relu(res_out)
        out = self.ReLU(out)

        # print("Strided layer 1 in Decoder...")
        out = self.strided_conv1(out)
        # print("Strided layer 2 in Decoder...")
        out = self.strided_conv2(out)

        return out

class Model(nn.Module):
    def __init__(self, Encoder, Quantise, Decoder):
        super().__init__()
        self.encoder = Encoder
        self.quantise = Quantise
        self.decoder = Decoder

    def forward(self, input):
        logit = self.encoder(input)
        # print(f"input: {input.shape} logit:{logit.shape}")
        #Get loss values and quantised logit
        logitq, commitment_loss, codebook_loss, perplexity = self.quantise(logit)
        # get reconstruction from quantised logit
        # print(f"input: {input.shape} logit:{logit.shape} ZQ: {logitq.shape}")
        xHat = self.decoder(logitq)
        
        # return reconstruction and loss values
        return xHat, commitment_loss, codebook_loss, perplexity