"""
File: modules.py
Description: Defines a CNN model for image classification and a Siamese Network
             that uses the CNN as a backbone to measure similarity between image pairs.
             Includes a Contrastive Loss function for training the Siamese Network.
             
Classes:
    CNN: A Convolutional Neural Network for image classification.
    SiameseNetwork: A Siamese Network for similarity learning, using two CNNs.
    ContrastiveLoss: A loss function to enforce similarity learning in the Siamese Network.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

class CNN(nn.Module):
    """
    A convolutional neural network (CNN) for image classification.
    
    This network is designed as a base feature extractor for a Siamese network, 
    consisting of several convolutional and fully connected layers.
    
    Attributes:
        cnn (nn.Sequential): Sequential model containing convolutional and pooling layers.
        flatten (nn.Flatten): Layer to flatten the output for the fully connected layers.
        fc (nn.Sequential): Fully connected layers for classification.
    """

    def __init__(self, shape, num_classes):
        """
        Initialize the CNN model.

        Args:
            shape (tuple): Input image shape as (height, width).
            num_classes (int): Number of output classes for classification.
        """
        super(CNN, self).__init__()
        # Define convolutional and pooling layers
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, stride=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),

            nn.Conv2d(64, 128, kernel_size=4, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.ReLU(inplace=True)
        )

        self.flatten = nn.Flatten()  # Flattening layer for fully connected layers

        # Define fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(self._get_conv_output_size(shape), 512),
            nn.ReLU(inplace=True),
            
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            
            nn.Linear(128, num_classes)
        )

    def _get_conv_output_size(self, shape):
        """
        Helper function to calculate the output size after convolution layers.
        
        Args:
            shape (tuple): Shape of the input image (height, width).
        
        Returns:
            int: Flattened size after the convolutional layers.
        """
        x = torch.zeros(1, 3, *shape)
        x = self.cnn(x)
        x = self.flatten(x)
        return x.numel()

    def forward(self, x):
        """
        Forward pass through the CNN.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """
        x = self.cnn(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

# Siamese Network for learning similarity between two images, consisting of two identical CNNs
class SiameseNetwork(nn.Module):
    """
    Siamese Network for similarity learning between two input images.
    
    This network uses two identical CNNs to process each image and then computes the 
    similarity based on their feature vectors.
    """
    def __init__(self):
        """
        Initialize the Siamese Network.
        
        This network consists of two identical CNN models.
        """
        super(SiameseNetwork, self).__init__()
        self.cnn = CNN((256,256), 2)  # Base CNN model for feature extraction
    
    def forward(self, input1, input2):
        """
        Forward pass for the Siamese Network.
        
        Args:
            input1 (torch.Tensor): First input tensor of shape (batch_size, channels, height, width).
            input2 (torch.Tensor): Second input tensor of shape (batch_size, channels, height, width).
        
        Returns:
            tuple: Output feature vectors for both input images.
        """
        output1 = self.cnn(input1)
        output2 = self.cnn(input2)
        return output1, output2

# Loss function for the Siamese Network
class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss for training the Siamese Network.
    
    This loss function aims to minimize the distance between similar images and 
    maximize the distance between dissimilar images up to a certain margin.
    
    Attributes:
        margin (float): Margin value for dissimilar images.
    """
    def __init__(self, margin=2.0):
        """
        Initialize the ContrastiveLoss function.
        
        Args:
            margin (float): Margin value. Defaults to 2.0.
        """
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        """
        Forward pass for the Contrastive Loss calculation.
        
        Args:
            output1 (torch.Tensor): Feature vector from the first image.
            output2 (torch.Tensor): Feature vector from the second image.
            label (torch.Tensor): Binary label indicating if the images are similar (1) or dissimilar (0).
        
        Returns:
            torch.Tensor: Computed contrastive loss.
        """
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive
