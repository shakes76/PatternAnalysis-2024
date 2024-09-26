import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim

"""
REFERENCES:
Kang, J. (2024). Pytorch-VAE-tutorial. Retrieved 31st August 2024 from https://github.com/Jackson-Kang/Pytorch-VAE-tutorial
"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Switching to CPU.")
else:
    print(device)

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()

        self.input = nn.Linear(input_dim, hidden_dim)
        self.input2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.var = nn.Linear(hidden_dim, latent_dim)

        self.ReLU = nn.ReLU()

        self.training = True

    def forward(self, x):
        act = self.ReLU(self.input(x))
        act = self.ReLU(self.input2(act))
        mean = self.mean(act)
        var = self.var(act)

        return mean, var
    

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()
        self.Hidden = nn.Linear(latent_dim, hidden_dim)
        self.Hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.Output = nn.Linear(hidden_dim, output_dim)

        self.ReLU = nn.ReLU()

    def forward(self, x):
        act = self.ReLU(self.Hidden(x))
        act = self.ReLU(self.Hidden2(act))

        xHat = torch.sigmoid(self.Output(act))
        return xHat

class Model(nn.Module):
    def __init__(self, Encoder, Decoder):
        super().__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder

    def reparameterisation(self, mean, var):
        e = torch.randn_like(var).to(device)
        logit = mean + var * e
        return logit

    def forward(self, x):
        mean, var = self.Encoder(x)
        logit = self.reparameterisation(mean, torch.exp(0.5 * var))
        xHat = self.Decoder(logit)

        return xHat, mean, var