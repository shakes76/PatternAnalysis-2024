import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
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
    print("Switching to CPU.")
else:
    print(device)

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

        self.strided_conv1 = nn.Conv2d(
            input_dim, hidden_dim, kernel_size=k1, stride=stride, padding=1)
        self.strided_conv2 = nn.Conv2d(
            hidden_dim, hidden_dim, kernel_size=k2, stride=stride, padding=1)

        self.residual_conv1 = nn.Conv2d(
            hidden_dim, hidden_dim, kernel_size=k3, padding=1)
        self.residual_conv2 = nn.Conv2d(
            hidden_dim, hidden_dim, kernel_size=k4, padding=0)

        self.ReLU = nn.ReLU()

        self.project = nn.Conv2d(
            hidden_dim, out_channels=output_dim, kernel_size=1)

        self.training = True

    def forward(self, x):
        out = self.strided_conv1(x)
        out = self.strided_conv2(out)

        # F.relu(act)
        out = self.ReLU(out)
        res_out = self.residual_conv1(out)
        res_out += out

        # F.relu(res_out)
        out = self.ReLU(res_out)
        res_out = self.residual_conv2(out)
        res_out += out

        return res_out


"""TBD"""


class VQEmbedLayer(nn.Module):
    def __init__(self, embeddings, embed_dim, q_loss=0.25, decay=0.999, e=1e-5):
        super().__init__()
        self.q_loss = q_loss
        self.decay = decay
        self.e = e

        init_bound = 1 / embeddings
        embedding = torch.Tensor(embeddings, embed_dim)
        embedding.uniform(-init_bound, init_bound)
        self.register_buffer("embedding", embedding)
        self.register_buffer("ema_count", embedding)
        self.register_buffer("ema_weight", self.embedding.clone())

    def encode(self, input):
        return

    def retrieve(self, i):
        return

    def forward(self, input):
        return


"""Decodes the latent space into higher dimensionality,
   using indexed values from the dictionary of embeddings. (Yadav, 2019)"""


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, kernel_sizes=(1, 3, 2, 2), stride=2):
        super().__init__()
        k1, k2, k3, k4 = kernel_sizes

        self.inner_project = nn.Conv2d(latent_dim, hidden_dim, kernel_size=1)

        self.residual_conv1 = nn.Conv2d(
            hidden_dim, hidden_dim, kernel_size=k1, stride=stride, padding=0)
        self.residual_conv12 = nn.Conv2d(
            hidden_dim, hidden_dim, kernel_size=k2, stride=stride, padding=1)

        self.strided_conv1 = nn.Conv2d(
            hidden_dim, hidden_dim, kernel_size=k3, stride=stride, padding=1)
        self.strided_conv2 = nn.Conv2d(
            hidden_dim, output_dim, kernel_size=k4, stride=stride, padding=1)
        self.ReLU = nn.ReLU()

    def forward(self, input):
        out = self.inner_project(input)

        res_out = self.residual_conv1(out)
        res_out += out
        # F.relu(res_out)
        out = self.ReLU(res_out)

        res_out = self.residual_conv1(out)
        res_out += out
        # F.relu(res_out)
        res_out = self.ReLU(out)

        res_out = self.strided_conv1(res_out)
        res_out = self.strided_conv2(res_out)

        return res_out


class Model(nn.Module):
    def __init__(self, Encoder, VQEmbeddings, Decoder):
        super().__init__()
        self.Encoder = Encoder
        # TBD VQEmbedLayer
        self.Decoder = Decoder

    def forward(self, input):
        logit = self.Encoder(input)
        # Get loss values and quantised logit
        # get reconstruction from quantised logit

        # return reconstruction and loss values
        return
