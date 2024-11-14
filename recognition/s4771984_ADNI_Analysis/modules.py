# Filename: gfnet_model.py
# Project: ADNI Alzheimer’s Classification with GFNet
# Author: Siddhant Gaikwad
# Date: 25/10/2024
# Description: This file defines the GFNet model architecture, a convolutional
#              neural network with a Global Filter Layer used for the classification
#              of Alzheimer's Disease (AD) vs. Cognitive Normal (CN) MRI images.
# =====================================================================

import torch
import torch.nn as nn
import warnings as w
w.filterwarnings('ignore')

class GFNet(nn.Module):
    
    """
    GFNet model for Alzheimer's Disease classification using MRI images.

    This model includes:
    - Convolutional layers for local feature extraction
    - A Global Filter Layer to capture long-range dependencies
    - Fully connected layers for binary classification into Cognitive Normal (CN) and Alzheimer's Disease (AD)

    Parameters:
    - num_classes (int): The number of output classes for classification (default is 2 for CN and AD).

    Attributes:
    - conv1, conv2, conv3, conv4 (nn.Conv2d): Convolutional layers with increasing filter counts for feature extraction.
    - global_filter (nn.Conv2d): Global filter layer for capturing global image patterns.
    - fc1 (nn.Linear): Fully connected layer, initialized based on input size in the forward pass.
    - fc2 (nn.Linear): Final output layer for classification.
    - pool (nn.MaxPool2d): Max pooling layer for downsampling.
    - dropout (nn.Dropout): Dropout layer to reduce overfitting.
    - elu (nn.ELU): ELU activation function for non-linearity.
    
    """
    
    def __init__(self, num_classes=2):
        
        """
        Initializes the layers of GFNet model, including convolutional layers,
        global filter layer, and fully connected layers.

        Parameters:
        - num_classes (int): Number of output classes, default is 2 (CN and AD).

        Returns
        -------
        None.

        """
        
        super(GFNet, self).__init__()
        # Convolutional layers with 96 and 192 filters
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=48, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(48, 96, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(96, 192, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(192, 192, kernel_size=3, padding=1)

        # Global Filter Layer
        self.global_filter = nn.Conv2d(192, 192, kernel_size=1)

        # Fully connected layers
        self.fc1 = None  # Placeholder for fc1, initialized in forward based on input size
        self.fc2 = nn.Linear(256, num_classes)  # Output layer

        # Pooling, dropout, and activation layers
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.elu = nn.ELU()  # ELU activation function

    def forward(self, x):
        
        """
        Forward pass for the GFNet model, which applies layers sequentially to the input data.

        Parameters:
        - x (tensor): Input tensor representing a batch of MRI images.

        Returns:
        - tensor: The output logits for each class (CN or AD).
        """
        
        x = self.pool(self.elu(self.conv1(x)))
        x = self.pool(self.elu(self.conv2(x)))
        x = self.pool(self.elu(self.conv3(x)))
        x = self.pool(self.elu(self.conv4(x)))
        x = self.global_filter(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor for fully connected layers

        if self.fc1 is None:
            self.fc1 = nn.Linear(x.size(1), 256).to(x.device)
        
        x = self.elu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x