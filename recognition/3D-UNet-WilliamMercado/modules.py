from torch import nn
import torch

class AnalysisLayer(nn.Module):
    "Generalized analysis layer class to perform analysis."
    def __init__(self, *args, base_width=32,target_width=64, conv_size=3, stride=1, padding=0, pool=False, **kwargs) -> None:
        """
        Initializer for Analysis layer. Allows for general alterations including changing the
        input channels and output channels, the convolution size, stride and padding, as well as an
        option for an initial max pooling

        Args:
            base_width (int, optional): The number of input channels. Defaults to 32.
            target_width (int, optional): The number of output channels. Defaults to 64.
            conv_size (int, optional): The size of kernels for convolutional layers. Defaults to 3.
            stride (int, optional): The stride of convolutional layers. Defaults to 1.
            padding (int, optional): The padding for convolutional layers. Defaults to 0.
            pool (bool, optional): The option on weather or not to include a pooling layer. Defaults to False.
        """
        super(AnalysisLayer, self).__init__()
        mid_width = max(base_width, target_width//2)

        self.pooling = nn.MaxPool3d(kernel_size=2,stride=2) if pool else None

        self.sequence = nn.Sequential(
            nn.Conv3d(in_channels=base_width, out_channels=mid_width, kernel_size=conv_size, stride=stride, padding=padding),
            nn.BatchNorm3d(num_features=mid_width),
            nn.ReLU(),
            nn.Conv3d(in_channels=mid_width, out_channels=target_width, kernel_size=conv_size, stride=stride, padding=padding),
            nn.BatchNorm3d(target_width),
            nn.ReLU(),
        )

    def forward(self,x):
        if self.pooling is not None:
            x = self.pooling(x)
        x = self.sequence(x)
        return x

class SynthesisLayer(nn.Module):
    "Generalized Synthesis Layer class to perform the synthesis path"
    def __init__(self, *args,base_width=512,target_width=256, conv_size=3, stride=1, padding=0, **kwargs) -> None:
        """
        Initializer for synthesis layer. Allows for general alterations including altering the
        number of features in and out, the size of kernels for convolutions, and stride and padding
        for convolutions.

        Args:
            base_width (int, optional): The number of input features. Defaults to 512.
            target_width (int, optional): The number of output features. Defaults to 256.
            conv_size (int, optional): The size of kernels for convolutions. Defaults to 3.
            stride (int, optional): The stride for convolutional layers. Defaults to 1.
            padding (int, optional): The padding for convolutional layers. Defaults to 0.
        """
        super(SynthesisLayer, self).__init__(*args, **kwargs)
        self.upconv = nn.ConvTranspose3d(in_channels=base_width,out_channels=base_width,kernel_size=2,stride=2,padding=0)
        self.sequence = nn.Sequential(
            nn.Conv3d(in_channels=base_width+target_width,out_channels=target_width,kernel_size=conv_size,stride=stride,padding=padding),
            nn.BatchNorm3d(num_features=target_width),
            nn.ReLU(),
            nn.Conv3d(in_channels=target_width,out_channels=target_width,kernel_size=conv_size,stride=stride,padding=padding),
            nn.BatchNorm3d(num_features=target_width),
            nn.ReLU(),
        )


    def forward(self, shortcut, x):
        x = self.upconv(x)
        torch.cat(shortcut, x)
        x = self.sequence(x)
        return x
