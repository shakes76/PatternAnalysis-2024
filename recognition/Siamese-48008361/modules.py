"""
modules.py

This module defines the Siamese Network architecture, Triplet Loss function,
and helper functions for creating the model and loss for skin lesion classification.

Author: Zain Al-Saffi
Date: 18th October 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class SiameseNetwork(nn.Module):
    """
    Siamese Network architecture based on ResNet50 for feature extraction.

    This network processes triplets of images (anchor, positive, negative)
    and produces embeddings that can be used for classification or similarity comparison.

    Args:
        embedding_dim (int): Dimension of the output embedding space.
    """

    def __init__(self, embedding_dim):
        super(SiameseNetwork, self).__init__()

        # Load pre-trained ResNet50 model
        resnet50 = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        
        # Remove the last fully connected layer
        self.feature_extractor = nn.Sequential(*list(resnet50.children())[:-1])
        
        # Unfreeze the parameters of the feature extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = True
        
        # ResNet50 output features
        resnet_output_features = 2048
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(resnet_output_features, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(512, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        # Classifier
        self.classifier = nn.Linear(embedding_dim, 2)
    
    def forward_once(self, x):
        """
        Forward pass for a single input image.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Embedding tensor of shape (batch_size, embedding_dim).
        """
        out = self.feature_extractor(x)
        out = out.view(out.size(0), -1)
        out = self.fc_layers(out)
        return out

    def forward(self, x1, x2, x3):
        """
        Forward pass for triplet input (anchor, positive, negative).

        Args:
            x1 (torch.Tensor): Anchor image tensor.
            x2 (torch.Tensor): Positive image tensor.
            x3 (torch.Tensor): Negative image tensor.

        Returns:
            tuple: Tuple containing embeddings for anchor, positive, and negative images.
        """
        out1 = self.forward_once(x1)
        out2 = self.forward_once(x2)
        out3 = self.forward_once(x3)
        return out1, out2, out3
    
    def get_embedding(self, x):
        """
        Get the embedding for a single input image.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Embedding tensor of shape (batch_size, embedding_dim).
        """
        return self.forward_once(x)
    
    def classify(self, x):
        """
        Perform classification on the input image.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Classification output tensor of shape (batch_size, num_classes).
        """
        embedding = self.get_embedding(x)
        return self.classifier(embedding)

class TripletLoss(nn.Module):
    """
    Triplet Loss function for training Siamese Networks.

    This loss encourages the network to produce embeddings where the distance between
    an anchor and a positive sample is smaller than the distance between the anchor
    and a negative sample by at least a margin.

    Args:
        margin (float): Margin for the triplet loss. Default is 1.0
    """

    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        """
        Compute the triplet loss.

        Args:
            anchor (torch.Tensor): Embedding of the anchor image.
            positive (torch.Tensor): Embedding of the positive image.
            negative (torch.Tensor): Embedding of the negative image.

        Returns:
            torch.Tensor: Scalar tensor containing the triplet loss.
        """
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        loss = F.relu(distance_positive - distance_negative + self.margin)
        return loss.mean()

def get_model(embedding_dim):
    """
    Create and return an instance of the SiameseNetwork model.

    Args:
        embedding_dim (int): Dimension of the output embedding space.

    Returns:
        SiameseNetwork: An instance of the SiameseNetwork model.
    """
    return SiameseNetwork(embedding_dim=embedding_dim)

def get_loss(margin=1.0):
    """
    Create and return an instance of the TripletLoss.

    Args:
        margin (float): Margin for the triplet loss. Default: 1.0

    Returns:
        TripletLoss: An instance of the TripletLoss.
    """
    return TripletLoss(margin)