import torch
import torch.nn as nn
import torch.nn.functional as F
import ResidualStack

class Encoder(nn.Module):
    """
    Reduces dimensionality of input MRI image
    """
    def __init__(self, in_channels, hidden_channels, latent_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=4, stride=2, padding=1) # downamplng
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(hidden_channels, latent_dim, kernel_size=3, stride=1, padding=1)

        # Need to add Residual Stack here

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x
