"""
modules.py

Holds the general neural network modules used for a 3D UNet.
Modules for Improved 3D UNet to be added at a later date.
"""
from torch import nn
import torch

class AnalysisLayer(nn.Module):
    "Generalized analysis layer class to perform analysis."
    def __init__(self, base_width=32,target_width=64, conv_size=3, stride=1, padding=0, pool=False) -> None:
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
    def __init__(self,base_width=512,target_width=256, conv_size=3, stride=1, padding=0) -> None:
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
        super(SynthesisLayer, self).__init__()
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

class FullUNet3D(nn.Module):
    "Full UNet 3D model built from analysis layers and synthesis layers. Has final 1x1x1 conv."
    def __init__(self, input_width=3, analysis_pad=0,synth_pad = 0,start_width=64,end_width=512) -> None:
        """
        Initializer for Full 3D UNet. Allows for general alterations including altering the maximum
        width of the analysis, the padding for both analysis and synthesis, and the input width

        Args:
            input_width (int, optional): The width of the input data. Defaults to 3.
            analysis_pad (int, optional): The padding for convolution during analysis.
                Defaults to 0.
            synth_pad (int, optional): The padding for convolution during synthesis.
                Defaults to 0.
            start_width (int, optional): The width to start and end for convolution. Defaults to 64.
            end_width (int, optional): The target maximum width for convolution. If during width
                doubling this number is passed, the resulting larger number will be used.
                Defaults to 512.
        """
        super(FullUNet3D,self).__init__()

        # Create Analysis Path from layers.
        self.analysis_path = [AnalysisLayer(base_width=input_width,target_width=start_width,padding=analysis_pad,pool=False)]
        cur_width = start_width
        while cur_width < end_width:
            new_layer = AnalysisLayer(base_width=start_width,target_width=start_width*2,padding=analysis_pad,pool=True)
            self.analysis_path.append(new_layer)
            cur_width *= 2
        
        self.synthesis_path:list[SynthesisLayer] = []
        while cur_width < start_width:
            new_layer = SynthesisLayer(base_width=cur_width,target_width=cur_width//2,padding=synth_pad)
            self.synthesis_path.append(new_layer)
            cur_width //= 2
        
        self.final_conv = nn.Conv3d(in_channels=cur_width,out_channels=input_width,kernel_size=1,stride=1,padding=0)

    def forward(self, x):
        to_shortcut = []
        for layer in self.analysis_path:
            x = layer(x)
            to_shortcut.append(x)
        to_shortcut.pop()
        for layer in self.synthesis_path:
            x = layer(to_shortcut.pop(), x)
        x = self.final_conv(x)
        return x
    
    def to(self, device:torch.device|None=None,dtype:torch.dtype|None=None,non_blocking=False):
        super(FullUNet3D, self).to(device=device,dtype=dtype,non_blocking=non_blocking)
        for layer in self.analysis_path:
            layer.to(device=device,dtype=dtype,non_blocking=non_blocking)
        for layer in self.synthesis_path:
            layer.to(device=device,dtype=dtype,non_blocking=non_blocking)
        return self
