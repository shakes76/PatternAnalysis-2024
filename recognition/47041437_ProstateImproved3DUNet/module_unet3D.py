import torch
import torch.nn as nn

'''
Decoder layers for the UNet3D model. 
This class is responsible for upsampling and concatenating feature maps from the encoder with the corresponding layers of the decoder.
'''
class Decoder(nn.Module):
  def __init__(self, in_channels, middle_channels, out_channels):
    """
        Initializes the Decoder block.
        
        Args:
        in_channels (int): The number of input channels.
        middle_channels (int): The number of middle channels used in the convolution layer.
        out_channels (int): The number of output channels for the final output.
    """
    super(Decoder, self).__init__()
    self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
    self.conv_relu = nn.Sequential(
        nn.Conv3d(middle_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm3d(out_channels)
        )
  def forward(self, x1, x2):
    """
        Forward pass through the decoder block.
        
        Args:
        x1 (Tensor): The feature map from the previous layer to be upsampled.
        x2 (Tensor): The feature map from the corresponding encoder layer to be concatenated with `x1`.
        
        Returns:
        Tensor: The output after applying upsampling, concatenation, and convolution.
    """
    x1 = self.up(x1)
    x1 = torch.cat((x1, x2), dim=1) #add shortcut layer and input layer together
    x1 = self.conv_relu(x1)
    return x1


'''
The UNet3D class defines a 3D U-Net architecture for volumetric image segmentation.

Architecture:
    Encoder and decoder layers progressively downsample and upsample the input, respectively.
    The input tensor is processed through convolutional layers followed by upsampling and concatenation with skip connections.
    The final output is a 3D tensor suitable for segmentation tasks.
'''
class UNet3D(nn.Module):
    def __init__(self):
        """
        Initializes the 3D U-Net model.
        The architecture consists of several encoder layers, followed by decoder layers to reconstruct the output.

        The model is designed for 3D data, like volumetric medical images, and can be extended for various segmentation tasks.
        """
        super().__init__()

        # Encoder part with downsampling layers
        self.layer1 = nn.Sequential(
            nn.Conv3d(1, 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(4, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.MaxPool3d(kernel_size=2),
            nn.ReLU(),
            nn.BatchNorm3d(16))

        self.layer2 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool3d(kernel_size=2),
            nn.ReLU(),
            nn.BatchNorm3d(64))
        
        self.layer3 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool3d(kernel_size=2),
            nn.ReLU(),
            nn.BatchNorm3d(64))
        
        self.layer4 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool3d(kernel_size=2),
            nn.ReLU(),
            nn.BatchNorm3d(128))

        # Decoder part with upsampling and skip connections
        self.decode3 = Decoder(128, 64+64, 64)
        self.decode2 = Decoder(64, 64+64, 64)
        self.decode1 = Decoder(64, 16+16, 16)

        # Final upsampling and convolution layers
        self.decode0 = nn.Sequential(
            nn.ConvTranspose3d(16, 16, kernel_size=2, stride=2),
            nn.Conv3d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, 16, kernel_size=3, padding=1),
            nn.ReLU()
            )
        self.conv_last = nn.Conv3d(16, 6, 1)


    def forward(self, input):
        """
        Forward pass through the entire U-Net model.
        
        Args:
        input (Tensor): The input tensor of shape (Batch size, Channels, Depth, Height, Width)
        
        Returns:
        Tensor: The output tensor after processing through the network
        """

        # Encoder: Downsampling steps
        e1 = self.layer1(input) # 16,64,128,128
        e2 = self.layer2(e1) # 64,32,64,64
        e3 = self.layer3(e2) # 64,16,32,32
        e4 = self.layer4(e3) # 128,8,16,16
        
        # Decoder: Upsampling and concatenation with skip connections
        d3 = self.decode3(e4, e3) # 64,16,32,32
        d2 = self.decode2(d3, e2) # 64,32,64,64
        d1 = self.decode1(d2, e1) # 16,64,128,128
        d0 = self.decode0(d1) # 16,128,256,256

        # Final output layer
        out = self.conv_last(d0) # 6,128,256,256
        
        return out