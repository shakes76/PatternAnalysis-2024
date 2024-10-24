"""
Implementation of a CNN model based on the UNet architecture.
Code adapted from https://medium.com/@fernandopalominocobo/mastering-u-net-a-step-by-step-guide-to-segmentation-from-scratch-with-pytorch-6a17c5916114 
and https://medium.com/analytics-vidhya/unet-implementation-in-pytorch-idiot-developer-da40d955f201

Each component has been implementated as a class or a function
"""

import torch
import torch.nn as nn

class conv_block(nn.Module):
    """
    This module sets up the basic conv2d block that does feature extraction (conv2d),
    batch normalisation (BatchNorm2d) and a ReLU activation.

    This block sets up two 3x3 convolutional layers, each followed by batch normalisation 
    and then a ReLU activation.
    """
    def __init__(self, in_c, out_c):
        
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.relu = nn.ReLU()

    def forward(self, inputs):
        
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x
    
class encoder(nn.Module):
    """
    This module implements the encoder network block of the UNet architecture.

    First it applies the two 3x3 convolutions followed by ReLU activation,
    as defined in the previous class conv_block(). 
    After that it performs a 2x2 max pooling operation with stride 2 for downsampling.
    """
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2), stride=2)

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)

        return x, p
    
class decoder(nn.Module):
    """
    This module implements the decoder network block of the UNet architecture.
    
    It performs a 2x2 up-convolution operation (ConvTranspose2d) with stride 2 for upsampling,
    then concatenates the output with the results of the skip connection
    and then applies the two 3x3 convolutions followed by ReLU activation,
    as defined in class conv_block(). 
    
    """
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c+out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)

        return x
    
class UNet(nn.Module):
    """
    This is the 2d UNet model which will be used for image segmentation.
    The UNet architecture consists of an encoder path and a decoder path,
    which will be implemented using the encoder and decoder blocks defined in this file.
    """
    def __init__(self):
        """
        Initialise the UNet model by creating all necessary layers
        """
        super().__init__()

        # Set up the 4 encoder blocks in the encoder network
        self.encoder_block1 = encoder(3, 64)
        self.encoder_block2 = encoder(64, 128)
        self.encoder_block3 = encoder(128, 256)
        self.encoder_block4 = encoder(256, 512)

        self.encoder_block5 = conv_block(512, 1024) # 512 channels in, 1024 channels out

        #  Set up the 4 decoder blocks in the decoder network
        self.decoder_block1 = decoder(1024, 512)
        self.decoder_block2 = decoder(512, 256)
        self.decoder_block3 = decoder(256, 128)
        self.decoder_block4 = decoder(128, 64)

        # Final layer with a 1x1 convolution to map each 64-component feature vector
        # to the desired number of classes
        self.outputs = nn.Conv2d(64, 6, kernel_size=1, padding=0)

    def forward(self, inputs):
        """
        Forward pass of the UNet model to generate segmentation mask
        """
        # Creating the encoder path
        # Each encoder block returns:
        # (i) the output of the convolutions, which is used as the skip connection for the decoder
        # and (ii) the output of the pooling layer, which is passed into the next encoder block.
        
        skip_connection1, pooled1 = self.encoder_block1(inputs)
        skip_connection2, pooled2 = self.encoder_block2(pooled1)
        skip_connection3, pooled3 = self.encoder_block3(pooled2)
        skip_connection4, pooled4 = self.encoder_block4(pooled3)
        x = self.encoder_block5(pooled4)

        # Creating the decoder path
        d1 = self.decoder_block1(x, skip_connection4)
        d2 = self.decoder_block2(d1, skip_connection3)
        d3 = self.decoder_block3(d2, skip_connection2)
        d4 = self.decoder_block4(d3, skip_connection1)

        # Output segmentation map
        output = self.outputs(d4)

        return output