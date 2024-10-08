## 1. “modules.py" containing the source code of the components of your model. Each component must be
# implemented as a class or a function
import torch.nn as nn

class GFNet(nn.Module):
    def __init__(self, dim, in_channels, num_classes=10) -> None:
        super().__init__()
        self.dim = dim
        self.in_channels = in_channels
        self.num_classes = num_classes

    def forward(self, x):
        pass


class FeedFowardNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, act_layer=nn.LeakyReLU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.norm_layer = norm_layer
        self.act_layer = act_layer()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = self.fc(x)
        x = self.act_layer(x)
        x = self.norm_layer(x) # WARNING: This might be wrong - don't really know what norm_layer takes as input
        return x
